"""Periodic output-loop break (SGLANG_ENABLE_OUTPUT_LOOP_BREAK): a runaway
decode repeating a phrase must finish the request instead of burning the
context window. Covers the detector's period window (short periods are
legitimate punctuation runs; long ones exceed the scan), the repeat-count
extension, the finish-reason wire contract, and the update_finish_state
ordering: a detected loop beats the length cap, and with the env off the old
behavior is byte-identical. Drives the real `Req.update_finish_state`; pure
CPU."""

import unittest
from array import array

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.output_loop_detector import detect_periodic_loop
from sglang.srt.managers.schedule_batch import (
    FINISH_LOOP_DETECTED,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EOS_ID = 2


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None


class _MockTokenizerForNormalize:
    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))


def _make_req(output_ids, *, max_new_tokens, vocab_size=10_000, ignore_eos=False):
    sp = SamplingParams(max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)
    sp.normalize(tokenizer=_MockTokenizerForNormalize())
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids={EOS_ID},
        vocab_size=vocab_size,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


def _loop_tokens(period, repeats, offset=10):
    block = list(range(offset, offset + period))
    return block * repeats


class TestOutputLoopDetector(CustomTestCase):
    def test_loop_tail_finishes_request(self):
        # Period-4 block x3 at the tail: the request must finish with
        # FINISH_LOOP_DETECTED carrying the loop geometry.
        tokens = [7, 8, 9] + _loop_tokens(4, 3)
        req = _make_req(tokens, max_new_tokens=1000)
        with envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(True):
            req.update_finish_state(new_accepted_len=1)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LOOP_DETECTED)
        self.assertEqual(req.finished_reason.period, 4)
        self.assertEqual(req.finished_reason.repeats, 3)
        self.assertEqual(req.finished_reason.length, len(tokens))

    def test_loop_beats_length_cap(self):
        # A loop discovered on the very step the output crosses the cap must
        # report the loop (the more specific cause), not FINISH_LENGTH.
        tokens = [7, 8, 9] + _loop_tokens(4, 3)
        req = _make_req(tokens, max_new_tokens=len(tokens) - 1)
        with envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(True):
            req.update_finish_state(new_accepted_len=1)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LOOP_DETECTED)

    def test_env_off_keeps_legacy_behavior(self):
        # Off: a looped tail under the cap must not finish at all, mirroring
        # pre-feature behavior byte for byte. Overridden so the case stays
        # valid where a production env exports the flag as on.
        tokens = [7, 8, 9] + _loop_tokens(4, 3)
        req = _make_req(tokens, max_new_tokens=1000)
        with envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(False):
            req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())

    def test_ignore_eos_exemption(self):
        # ignore_eos contracts untrimmed output (bench, raw-generation
        # tooling): the detector must not finish such a request even on a
        # looped tail.
        tokens = [7, 8, 9] + _loop_tokens(4, 3)
        req = _make_req(tokens, max_new_tokens=1000, ignore_eos=True)
        with envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(True):
            req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())

    def test_finish_json_contract(self):
        # OpenAI's finish_reason Literal has no loop notion, so the wire type
        # is "length"; native meta_info carries the loop details alongside.
        reason = FINISH_LOOP_DETECTED(period=6, repeats=4, length=42)
        as_json = reason.to_json()
        self.assertEqual(as_json["type"], "length")
        self.assertEqual(as_json["length"], 42)
        self.assertEqual(as_json["loop_detected"], {"period": 6, "repeats": 4})

    def test_detector_period_window(self):
        # Three-token and 33-token loops are outside the scan window: short
        # periods are legitimate output ("...", "————"), long ones are not
        # scanned.
        self.assertIsNone(detect_periodic_loop([1, 2, 3] * 5))
        self.assertIsNone(detect_periodic_loop(list(range(33)) * 3))
        # The window's upper edge, exactly three blocks, fires.
        self.assertEqual(detect_periodic_loop(list(range(32)) * 3), (32, 3))

    def test_large_period_loop_breaks_request(self):
        # Production escape (2026-10-06 zcode incident): a runaway whose
        # repeated blocks are ~600 tokens, far beyond the phrase window, once
        # burned ~5k tokens on a single turn. With the extended window the
        # same tail must finish the request carrying the loop geometry; the
        # loop starts mid-output, not at token 0.
        tokens = [(i * 37 + 11) % 9973 for i in range(200)] + _loop_tokens(600, 3)
        req = _make_req(tokens, max_new_tokens=100_000)
        with (
            envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(True),
            envs.SGLANG_OUTPUT_LOOP_BREAK_MAX_PERIOD.override(2048),
        ):
            req.update_finish_state(new_accepted_len=1)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LOOP_DETECTED)
        self.assertEqual(req.finished_reason.period, 600)
        self.assertEqual(req.finished_reason.repeats, 3)

    def test_default_max_period_keeps_legacy_window(self):
        # The descriptor default is the pre-extension phrase window: a
        # 600-token loop is not scanned by it. Asserted on .default, not
        # .get(), so the case stays valid where a production env widens
        # the window.
        self.assertEqual(envs.SGLANG_OUTPUT_LOOP_BREAK_MAX_PERIOD.default, 32)
        self.assertIsNone(detect_periodic_loop(_loop_tokens(600, 3)))
        self.assertIsNone(detect_periodic_loop(_loop_tokens(600, 4)))

    def test_extended_window_seam_boundaries(self):
        # The extended window starts right where the phrase window ends: 33
        # must flip from legacy-miss to extended-hit with no gap at the seam.
        tokens = _loop_tokens(33, 3)
        self.assertIsNone(detect_periodic_loop(tokens))
        self.assertEqual(detect_periodic_loop(tokens, max_period=2048), (33, 3))

    def test_large_period_needs_three_identical_blocks(self):
        # Two blocks stay coincidence at any period; the false-positive guard
        # must hold when the would-be blocks merely share their last token
        # (the prefilter's blind spot) but differ inside.
        self.assertIsNone(detect_periodic_loop(_loop_tokens(600, 2), max_period=2048))
        self.assertEqual(
            detect_periodic_loop(_loop_tokens(600, 3), max_period=2048), (600, 3)
        )
        decoy = list(range(500, 1100)) + list(range(500, 1099)) + [777]
        decoy += list(range(0, 599)) + [777]
        self.assertIsNone(detect_periodic_loop(decoy, max_period=2048))

    def test_drifted_large_block_does_not_fire(self):
        # Only exact repetition matches: a cycle that mutates one token per
        # round escapes the detector until it settles into an exact cycle
        # (the one-hot attractor end state always does). Pins that contract
        # against a fuzzy-matching rewrite.
        block = list(range(600))
        drifted = block[:-1] + [99_001] + block[:-1] + [99_002] + block[:-1] + [99_003]
        self.assertIsNone(detect_periodic_loop(drifted, max_period=2048))

    def test_detector_needs_three_full_blocks(self):
        # Two identical blocks may still be coincidence; three is the trigger.
        self.assertIsNone(detect_periodic_loop(_loop_tokens(4, 2)))
        self.assertEqual(detect_periodic_loop(_loop_tokens(4, 3)), (4, 3))

    def test_detector_counts_beyond_three(self):
        # repeats extends backwards past the trigger so the log line says how
        # deep the loop already was.
        self.assertEqual(detect_periodic_loop(_loop_tokens(4, 5)), (4, 5))

    def test_detector_ignores_varied_text(self):
        # No period under 32 in a non-repeating tail: nothing fires.
        tokens = [(i * 37 + 11) % 9973 for i in range(200)]
        self.assertIsNone(detect_periodic_loop(tokens))
        self.assertIsNone(detect_periodic_loop(tokens[:11]))


if __name__ == "__main__":
    unittest.main()
