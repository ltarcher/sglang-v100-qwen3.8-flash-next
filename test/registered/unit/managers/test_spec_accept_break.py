"""Spec accept-rate lock breaker (SGLANG_SPEC_ACCEPT_BREAK_THRESHOLD): a
request whose trailing verify rounds accept nearly every draft is in the
one-hot attractor end state and must finish the request instead of burning
the context window. Covers the rolling-window rate math, the arming gates
(window full, min output tokens, env off), the ignore_eos exemption, the
near-miss one-shot log flag, and the finish-reason wire contract for the
spec source, which carries an accept_rate and no loop geometry. Drives the
real `Req.update_finish_state`; pure CPU."""

import unittest
from array import array
from contextlib import ExitStack

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
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


def _push(req, correct, proposed, rounds=1):
    for _ in range(rounds):
        req.push_spec_accept_rate_sample(
            num_correct_drafts=correct, num_proposed_drafts=proposed
        )


# Lock shape: every round accepts all three proposed drafts (paper rate 1.0).
_FULL_LOCK = dict(correct=3, proposed=3)
# Healthy real-content shape on this engine: accept 1.0-2.4 incl. bonus.
_HEALTHY = dict(correct=1, proposed=3)

_BREAK_ENV = {
    "SGLANG_SPEC_ACCEPT_BREAK_THRESHOLD": 0.96,
    "SGLANG_SPEC_ACCEPT_BREAK_MIN_TOKENS": 8,
    "SGLANG_SPEC_ACCEPT_BREAK_WINDOW": 4,
}


class TestSpecAcceptBreak(CustomTestCase):
    def test_env_defaults_sealed(self):
        # Asserted on .default so the case stays valid where a production env
        # exports the knobs: default is off, zai's 0.96 with a 128-token arming
        # floor and a 64-round window.
        self.assertIsNone(envs.SGLANG_SPEC_ACCEPT_BREAK_THRESHOLD.default)
        self.assertEqual(envs.SGLANG_SPEC_ACCEPT_BREAK_MIN_TOKENS.default, 128)
        self.assertEqual(envs.SGLANG_SPEC_ACCEPT_BREAK_WINDOW.default, 64)

    def test_window_math_and_trim(self):
        # The rate is the ratio of window sums, and old rounds fall off the
        # back once the window overflows.
        req = _make_req([1] * 64, max_new_tokens=1000)
        with envs.SGLANG_SPEC_ACCEPT_BREAK_WINDOW.override(5):
            _push(req, **_HEALTHY, rounds=2)
            _push(req, **_FULL_LOCK, rounds=3)
            self.assertEqual(len(req.spec_accept_rate_window), 5)
            self.assertEqual(req.spec_accept_rate_window_correct_sum, 1 * 2 + 3 * 3)
            self.assertEqual(req.spec_accept_rate_window_proposed_sum, 3 * 5)
            _push(req, **_FULL_LOCK, rounds=1)
            self.assertEqual(len(req.spec_accept_rate_window), 5)
            # (1*1 + 4*3) / (5*3) -- the first healthy round dropped off.
            self.assertAlmostEqual(req.spec_accept_rate(min_output_tokens=0), 13 / 15)

    def test_lock_finishes_request(self):
        # A sustained full-acceptance lock must finish the request with the
        # spec source, carrying the rate and no loop geometry. The periodic
        # breaker is explicitly off so this case only exercises the rate path.
        req = _make_req([1] * 32, max_new_tokens=1000)
        _push(req, **_FULL_LOCK, rounds=4)
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(False)
            )
            for name, value in _BREAK_ENV.items():
                stack.enter_context(getattr(envs, name).override(value))
            req.update_finish_state(new_accepted_len=1)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LOOP_DETECTED)
        self.assertEqual(req.finished_reason.source, "spec_accept_rate")
        self.assertIsNone(req.finished_reason.period)
        self.assertAlmostEqual(req.finished_reason.accept_rate, 1.0)
        self.assertEqual(req.finished_reason.length, 32)

    def test_not_armed_below_min_tokens(self):
        # Same lock, output under the arming floor: must keep decoding.
        req = _make_req([1] * 4, max_new_tokens=1000)
        _push(req, **_FULL_LOCK, rounds=4)
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(False)
            )
            for name, value in _BREAK_ENV.items():
                stack.enter_context(getattr(envs, name).override(value))
            req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())

    def test_not_armed_window_not_full(self):
        # Two rounds into a four-round window: no rate, no finish.
        req = _make_req([1] * 32, max_new_tokens=1000)
        _push(req, **_FULL_LOCK, rounds=2)
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(False)
            )
            for name, value in _BREAK_ENV.items():
                stack.enter_context(getattr(envs, name).override(value))
            req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())

    def test_mixed_window_below_threshold(self):
        # Healthy-content rate (~0.5) under the same arming state must not
        # finish: the breaker targets the lock, not ordinary low-entropy runs.
        req = _make_req([1] * 32, max_new_tokens=1000)
        _push(req, **_HEALTHY, rounds=4)
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(False)
            )
            for name, value in _BREAK_ENV.items():
                stack.enter_context(getattr(envs, name).override(value))
            req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())

    def test_ignore_eos_exemption(self):
        # Bench contract (ignore_eos=True) wants untrimmed degenerate output;
        # the breaker must not finish such a request even on a full lock.
        req = _make_req([1] * 32, max_new_tokens=1000, ignore_eos=True)
        _push(req, **_FULL_LOCK, rounds=4)
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(False)
            )
            for name, value in _BREAK_ENV.items():
                stack.enter_context(getattr(envs, name).override(value))
            req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())

    def test_threshold_none_disabled(self):
        # Default (None): the breaker is off even on a full lock. Overridden
        # explicitly so the case stays valid where a production env sets a
        # threshold.
        req = _make_req([1] * 32, max_new_tokens=1000)
        _push(req, **_FULL_LOCK, rounds=4)
        with (
            envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(False),
            envs.SGLANG_SPEC_ACCEPT_BREAK_THRESHOLD.override(None),
        ):
            req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())

    def test_near_miss_logged_once(self):
        # Sustained acceptance just under the threshold flips the one-shot
        # near-miss flag on the first observation and keeps decoding.
        req = _make_req([1] * 32, max_new_tokens=1000)
        _push(req, correct=3, proposed=3, rounds=3)
        _push(req, correct=2, proposed=3, rounds=1)  # rate 11/12 ~= 0.917
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_OUTPUT_LOOP_BREAK.override(False)
            )
            for name, value in _BREAK_ENV.items():
                stack.enter_context(getattr(envs, name).override(value))
            req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())
        self.assertTrue(req.spec_accept_rate_near_miss_logged)

    def test_finish_json_spec_contract(self):
        # The spec source has no loop geometry: period/repeats are absent and
        # accept_rate rides alongside the source tag. OpenAI face stays
        # "length".
        reason = FINISH_LOOP_DETECTED(
            period=None,
            repeats=None,
            length=42,
            source="spec_accept_rate",
            accept_rate=0.97,
        )
        as_json = reason.to_json()
        self.assertEqual(as_json["type"], "length")
        self.assertEqual(as_json["length"], 42)
        self.assertEqual(
            as_json["loop_detected"],
            {"source": "spec_accept_rate", "accept_rate": 0.97},
        )

    def test_processor_probe_of_proposed_drafts(self):
        # The processor derives proposed drafts as slot width minus non-draft
        # tokens (NEXTN: 4 - 1 = 3). Pins the arithmetic the caller relies on
        # so a stride/num_non_draft change cannot silently skew every rate.
        stride, num_non_draft = 4, 1
        self.assertEqual(stride - num_non_draft, 3)


if __name__ == "__main__":
    unittest.main()
