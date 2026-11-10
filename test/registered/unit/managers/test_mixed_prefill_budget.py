"""Unit tests for the adaptive mixed-prefill budget controller (L3-1).

Ports of 1Cat vLLM #860; the controller math is exercised on CPU with plain
numbers, the timing pipeline with duck-typed CUDA events.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

from sglang.srt.managers.scheduler_components.mixed_prefill_budget import (
    MixedPrefillBudget,
    MixedPrefillBudgetControl,
    MixedPrefillStep,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeEvent:
    """Duck-typed torch.cuda.Event: query() flips ready after `ready_after`
    queries; elapsed_time returns a fixed value."""

    def __init__(self, ready_after: int = 0, elapsed_ms: float = 100.0, **kwargs):
        self.ready_after = ready_after
        self.elapsed_ms = elapsed_ms
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.queries > self.ready_after

    def record(self) -> None:
        pass

    def elapsed_time(self, _end) -> float:
        return self.elapsed_ms


class TestMixedPrefillBudget(CustomTestCase):
    def make_budget(self, target_ms=250.0, max_tokens=4096, min_tokens=0, align=1):
        return MixedPrefillBudget(
            max_tokens=max_tokens,
            target_ms=target_ms,
            min_tokens=min_tokens,
            align=align,
        )

    def test_seed_budget(self):
        budget = self.make_budget()
        self.assertEqual(budget.tokens, 512)
        # max_tokens below the seed clamps it.
        budget = self.make_budget(max_tokens=256)
        self.assertEqual(budget.tokens, 256)

    def test_seed_respects_min_tokens(self):
        budget = self.make_budget(min_tokens=1024)
        self.assertEqual(budget.tokens, 1024)

    def test_ignores_invalid_samples(self):
        budget = self.make_budget()
        budget.update(
            prefill_tokens=0, decode_tokens=2, budget_tokens=512, elapsed_ms=100
        )
        budget.update(
            prefill_tokens=512, decode_tokens=0, budget_tokens=512, elapsed_ms=100
        )
        budget.update(
            prefill_tokens=512, decode_tokens=2, budget_tokens=512, elapsed_ms=0
        )
        budget.update(
            prefill_tokens=512,
            decode_tokens=2,
            budget_tokens=512,
            elapsed_ms=float("nan"),
        )
        self.assertIsNone(budget.ms_per_token)
        self.assertEqual(budget._fit, [0.0] * 5)

    def test_tail_chunk_does_not_feed_fit(self):
        budget = self.make_budget()
        # A tail chunk under 75% of the budget must not teach the fit.
        budget.update(
            prefill_tokens=100, decode_tokens=2, budget_tokens=512, elapsed_ms=50
        )
        self.assertEqual(budget._fit, [0.0] * 5)
        # A full-budget chunk does.
        budget.update(
            prefill_tokens=512, decode_tokens=2, budget_tokens=512, elapsed_ms=100
        )
        self.assertEqual(budget._fit[0], 1.0)

    def test_growth_is_clamped(self):
        budget = self.make_budget()
        # One sample: raw per-token estimate drives target/cost tokens, but the
        # growth clamp bounds the first jump to 2x + one quantum.
        budget.update(
            prefill_tokens=512, decode_tokens=2, budget_tokens=512, elapsed_ms=10
        )
        # target/cost = 250/(10/512) = 12800 -> clamped to max(512*2, 528) = 1024.
        self.assertEqual(budget.tokens, 1024)

    def test_budget_shrinks_on_slow_step(self):
        budget = self.make_budget(target_ms=100.0)
        budget.update(
            prefill_tokens=4096, decode_tokens=2, budget_tokens=4096, elapsed_ms=800
        )
        # cost = 800/4096 ~ 0.195 tok/ms -> target 100ms -> ~512 tokens.
        self.assertEqual(budget.tokens, 512)

    def test_affine_fit_and_fixed_floor(self):
        budget = self.make_budget(target_ms=100.0, max_tokens=8192)
        # Steps cost 20ms fixed + 0.1ms/token: 512 tok -> 71.2ms, 2048 -> 224.8ms.
        budget.update(
            prefill_tokens=512, decode_tokens=2, budget_tokens=512, elapsed_ms=71.2
        )
        budget.update(
            prefill_tokens=2048, decode_tokens=2, budget_tokens=640, elapsed_ms=224.8
        )
        # affine budget = max(100-20, 20)/0.1 = 800 tokens (quantized to 16).
        self.assertEqual(budget.tokens, 800)

    def test_decay_weights_recent_samples(self):
        budget = self.make_budget(target_ms=100.0, max_tokens=8192)
        # Expensive regime first, then a cheap regime; decay must move the
        # estimate toward the cheap one.
        budget.update(
            prefill_tokens=1024, decode_tokens=2, budget_tokens=1024, elapsed_ms=800
        )
        expensive = budget.tokens
        for _ in range(10):
            budget.update(
                prefill_tokens=1024,
                decode_tokens=2,
                budget_tokens=budget.tokens,
                elapsed_ms=100,
            )
        cheap = budget.tokens
        self.assertGreater(cheap, expensive)

    def test_min_sample_tokens_do_not_feed_fit(self):
        budget = self.make_budget()
        # Below 2*quantum the step is fixed-cost-dominated: no fit sample,
        # even when it is a full-budget chunk.
        budget.update(
            prefill_tokens=13, decode_tokens=2, budget_tokens=16, elapsed_ms=645
        )
        self.assertEqual(budget._fit, [0.0] * 5)
        budget.update(
            prefill_tokens=64, decode_tokens=2, budget_tokens=64, elapsed_ms=400
        )
        self.assertEqual(budget._fit[0], 1.0)

    def test_unreachable_target_gives_up(self):
        budget = self.make_budget(target_ms=250.0, max_tokens=4096)
        # Full-budget steps at the shrunk budget still blow past the target:
        # after 8 such samples the controller must give up and uncap.
        for _ in range(8):
            budget.update(
                prefill_tokens=64,
                decode_tokens=2,
                budget_tokens=64,
                elapsed_ms=400.0,
            )
        self.assertTrue(budget.gave_up)
        self.assertEqual(budget.tokens, 4096)
        # After giving up, further samples change nothing.
        budget.update(
            prefill_tokens=64, decode_tokens=2, budget_tokens=64, elapsed_ms=400.0
        )
        self.assertEqual(budget.tokens, 4096)

    def test_no_give_up_while_budget_is_large(self):
        # Overshooting while the budget is above the collapse threshold is
        # normal adaptation: the budget shrinks but the guard stays armed.
        budget = self.make_budget(target_ms=100.0, max_tokens=1024)
        for _ in range(20):
            budget.update(
                prefill_tokens=512,
                decode_tokens=2,
                budget_tokens=budget.tokens,
                elapsed_ms=150.0,
            )
        self.assertFalse(budget.gave_up)
        self.assertLess(budget.tokens, 512)
        self.assertGreater(budget.tokens, 1024 // 8)

    def test_growth_is_damped_by_observed_headroom(self):
        budget = self.make_budget(target_ms=100.0, max_tokens=8192)
        # A sample that only just met the target must not let the budget
        # double: growth is bounded by target/elapsed of the last full step.
        budget.update(
            prefill_tokens=512, decode_tokens=2, budget_tokens=512, elapsed_ms=99.0
        )
        # raw estimate ~ 517 tokens, growth clamp would allow 640 but desired
        # already fits: budget tracks the observed cost.
        self.assertEqual(budget.tokens, 512)
        # A comfortable margin allows a proportionally larger jump, but the
        # budget tracks the observed cost, not the raw estimate.
        budget.update(
            prefill_tokens=640, decode_tokens=2, budget_tokens=640, elapsed_ms=10.0
        )
        # The EWMA cost (blended with the 99ms sample) gives desired ~957;
        # the 10x headroom allows the full 2x clamp (1024) but the budget
        # tracks the observed cost: 957 quantizes to 944.
        self.assertEqual(budget.tokens, 944)

    def test_growth_tracks_evidence_two_x_margin_for_two_x_jump(self):
        budget = self.make_budget(target_ms=100.0, max_tokens=8192)
        # A 2x margin earns exactly a 2x jump: 50ms for 512 tokens means
        # ~1024 tokens fit the target; the clamp allows the doubling.
        budget.update(
            prefill_tokens=512, decode_tokens=2, budget_tokens=512, elapsed_ms=50.0
        )
        self.assertEqual(budget.tokens, 1024)

    def test_budget_never_exceeds_max(self):
        budget = self.make_budget(target_ms=10_000.0, max_tokens=1024)
        # Huge headroom: the growth clamp allows 2x, capped by max_tokens.
        budget.update(
            prefill_tokens=1024, decode_tokens=2, budget_tokens=1024, elapsed_ms=5
        )
        self.assertEqual(budget.tokens, 1024)

    def test_page_aligned_budget_stays_aligned(self):
        # Regression (QSA crash): the cap handed to PrefillAdder must stay on
        # the KV pool's page granularity — the QSA compress write plan asserts
        # prefix_lens % compress_ratio == 0 for non-mixed extends, and an
        # arbitrary learned cap leaks misaligned chunk boundaries through the
        # adder's fallback paths.
        # Three identical slow samples converge to 336: %16 (passes the
        # quantum) but a half-page boundary under page=64.
        unaligned = self.make_budget(target_ms=100.0, max_tokens=8192)
        aligned = self.make_budget(target_ms=100.0, max_tokens=8192, align=64)
        for _ in range(3):
            unaligned.update(1024, 2, 1024, 300.0)
            aligned.update(1024, 2, 1024, 300.0)
        self.assertEqual(unaligned.tokens, 336)
        self.assertEqual(aligned.tokens, 320)
        # Every subsequent sample keeps the alignment, whatever the fit says.
        for elapsed in (150.0, 90.0, 40.0, 200.0):
            aligned.update(
                prefill_tokens=aligned.tokens,
                decode_tokens=2,
                budget_tokens=aligned.tokens,
                elapsed_ms=elapsed,
            )
            self.assertEqual(aligned.tokens % 64, 0)
            self.assertGreaterEqual(aligned.tokens, 64)

    def test_give_up_budget_is_page_aligned(self):
        # max_tokens=4000 is not a page multiple: the uncapped fallback must
        # still hand over a page-aligned chunk.
        budget = self.make_budget(target_ms=250.0, max_tokens=4000, align=64)
        for _ in range(8):
            budget.update(
                prefill_tokens=64,
                decode_tokens=2,
                budget_tokens=64,
                elapsed_ms=400.0,
            )
        self.assertTrue(budget.gave_up)
        self.assertEqual(budget.tokens, 4000 // 64 * 64)

    def test_align_clamped_to_max_tokens(self):
        # A page larger than the whole chunk budget degrades gracefully.
        budget = self.make_budget(target_ms=250.0, max_tokens=32, align=64)
        self.assertEqual(budget.align, 32)
        self.assertEqual(budget.tokens, 32)


class TestMixedPrefillBudgetControl(CustomTestCase):
    def make_control(self, target_ms=250.0):
        return MixedPrefillBudgetControl(
            chunked_prefill_size=4096, target_ms=target_ms, min_tokens=0
        )

    @staticmethod
    def make_req(**kwargs):
        defaults = dict(return_logprob=False, beam_group=None, input_embeds=None)
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_cap_adds_decode_rows_back(self):
        control = self.make_control()
        running = SimpleNamespace(
            is_empty=lambda: False,
            return_logprob=False,
            reqs=[self.make_req() for _ in range(4)],
        )
        capped = control.maybe_cap_chunked_prefill_size(
            4096, running, waiting_queue=[], chunked_reqs=[], is_mixed_chunk=True
        )
        # budget.tokens(512) + 4 decode rows = 516.
        self.assertEqual(capped, 516)

    def test_cap_noop_without_resident_decode(self):
        control = self.make_control()
        running = SimpleNamespace(is_empty=lambda: True, return_logprob=False, reqs=[])
        self.assertIsNone(
            control.maybe_cap_chunked_prefill_size(
                4096, running, [], [], is_mixed_chunk=True
            )
        )

    def test_cap_noop_without_mixed_chunk(self):
        control = self.make_control()
        running = SimpleNamespace(
            is_empty=lambda: False,
            return_logprob=False,
            reqs=[self.make_req()],
        )
        self.assertIsNone(
            control.maybe_cap_chunked_prefill_size(
                4096, running, [], [], is_mixed_chunk=False
            )
        )

    def test_cap_skips_unmixable_passes(self):
        control = self.make_control()
        running = SimpleNamespace(
            is_empty=lambda: False,
            return_logprob=False,
            reqs=[self.make_req()],
        )
        # A logprob request on any side: the pass will not mix, so the chunk
        # must not be capped (no sample would ever lift the budget).
        self.assertIsNone(
            control.maybe_cap_chunked_prefill_size(
                4096, running, [self.make_req(return_logprob=True)], [], True
            )
        )
        self.assertIsNone(
            control.maybe_cap_chunked_prefill_size(
                4096,
                SimpleNamespace(
                    is_empty=lambda: False,
                    return_logprob=False,
                    reqs=[self.make_req(beam_group=object())],
                ),
                [],
                [],
                True,
            )
        )
        self.assertIsNone(
            control.maybe_cap_chunked_prefill_size(
                4096,
                running,
                [self.make_req(input_embeds=object())],
                [],
                True,
            )
        )
        self.assertIsNone(
            control.maybe_cap_chunked_prefill_size(
                4096, running, [], [self.make_req(return_logprob=True)], True
            )
        )
        # The running batch's own batch-level logprob flag also gates.
        running_lp = SimpleNamespace(
            is_empty=lambda: False, return_logprob=True, reqs=[self.make_req()]
        )
        self.assertIsNone(
            control.maybe_cap_chunked_prefill_size(4096, running_lp, [], [], True)
        )

    def test_cap_follows_budget_learning(self):
        control = self.make_control()
        running = SimpleNamespace(
            is_empty=lambda: False,
            return_logprob=False,
            reqs=[self.make_req() for _ in range(2)],
        )
        # After slow samples the learned budget shrinks; the cap follows.
        control.budget.update(4096, 2, 4096, 1600)  # 250ms / (1600/4096) ~ 640
        capped = control.maybe_cap_chunked_prefill_size(
            4096, running, [], [], is_mixed_chunk=True
        )
        self.assertEqual(capped, control.budget.tokens + 2)
        self.assertLess(capped, 4096)

    def test_control_plumbs_page_alignment(self):
        # The scheduler must hand its page_size down: with align=64 every
        # capped chunk boundary stays page-granular (QSA prefix invariant).
        control = MixedPrefillBudgetControl(
            chunked_prefill_size=4096, target_ms=250.0, min_tokens=0, align=64
        )
        self.assertEqual(control.budget.align, 64)
        running = SimpleNamespace(
            is_empty=lambda: False,
            return_logprob=False,
            reqs=[self.make_req() for _ in range(2)],
        )
        control.budget.update(4096, 2, 4096, 1600)
        capped = control.maybe_cap_chunked_prefill_size(
            4096, running, [], [], is_mixed_chunk=True
        )
        self.assertEqual(capped, control.budget.tokens + 2)
        self.assertEqual(control.budget.tokens % 64, 0)

    def test_step_lifecycle_with_fake_events(self):
        control = self.make_control()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_mixed=lambda: True),
            decoding_reqs=[object(), object()],
            extend_num_tokens=514,
            forward_iter=7,
        )
        # CPU sandbox has no CUDA: replace the event constructor.
        with patch("torch.cuda.Event", _FakeEvent):
            step = control.begin_step(batch)
            self.assertIsInstance(step, MixedPrefillStep)
            self.assertEqual(step.prefill_tokens, 512)
            self.assertEqual(step.decode_tokens, 2)
            self.assertEqual(step.budget_tokens, 512)
            self.assertIs(control._inflight[7], step)

            # Events not ready -> sample waits in the pending deque.
            step.start_event = _FakeEvent(ready_after=1)
            step.end_event = _FakeEvent(ready_after=0, elapsed_ms=120.0)
            result_batch = SimpleNamespace(forward_iter=7)
            control.process_batch_result(result_batch)
            self.assertNotIn(7, control._inflight)
            self.assertEqual(len(control._pending), 1)
            self.assertEqual(control.budget._fit, [0.0] * 5)

            # Later result: events are ready now, the sample feeds the budget.
            control.process_batch_result(SimpleNamespace(forward_iter=8))
        self.assertEqual(len(control._pending), 0)
        self.assertEqual(control.budget._fit[0], 1.0)
        self.assertGreater(control.budget.tokens, 512)

    def test_begin_step_ignores_non_mixed_batches(self):
        control = self.make_control()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_mixed=lambda: False),
            decoding_reqs=[object()],
            extend_num_tokens=1,
            forward_iter=1,
        )
        self.assertIsNone(control.begin_step(batch))
        mixed_no_decode = SimpleNamespace(
            forward_mode=SimpleNamespace(is_mixed=lambda: True),
            decoding_reqs=[],
            extend_num_tokens=100,
            forward_iter=2,
        )
        self.assertIsNone(control.begin_step(mixed_no_decode))


class _FakeCpuGroup:
    """One-slot mailbox shared by fake ranks, modeling the CPU-group
    broadcast: src writes first, everyone else reads."""

    def __init__(self):
        self.box = {}


def _fake_broadcast_pyobj(obj, rank, group, src=0):
    if rank == src:
        group.box["msg"] = obj
        return obj
    return group.box["msg"]


class TestMixedPrefillBudgetControlRankSync(CustomTestCase):
    """The controller must be rank-deterministic: a leader and a follower fed
    the same batch sequence must converge to identical budgets, with the
    sample measured once on the leader and shipped over the CPU group."""

    def make_pair(self, target_ms=250.0):
        group = _FakeCpuGroup()
        leader = MixedPrefillBudgetControl(
            chunked_prefill_size=4096,
            target_ms=target_ms,
            min_tokens=0,
            world_group=SimpleNamespace(rank=0, cpu_group=group),
        )
        follower = MixedPrefillBudgetControl(
            chunked_prefill_size=4096,
            target_ms=target_ms,
            min_tokens=0,
            world_group=SimpleNamespace(rank=1, cpu_group=group),
        )
        return leader, follower

    @staticmethod
    def make_mixed_batch(forward_iter, extend_tokens):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_mixed=lambda: True),
            decoding_reqs=[object(), object()],
            extend_num_tokens=extend_tokens,
            forward_iter=forward_iter,
        )

    def drive_step(self, leader, follower, forward_iter, elapsed_ms, ready_after=0):
        """Run one mixed step through both controllers, mirrored. When
        elapsed_ms is None the leader's events stay unresolved for the call."""
        event_cls = (
            (lambda **kw: _FakeEvent(ready_after=10**9, **kw))
            if elapsed_ms is None
            else _FakeEvent
        )
        with patch("torch.cuda.Event", event_cls):
            lstep = leader.begin_step(self.make_mixed_batch(forward_iter, 514))
            fstep = follower.begin_step(self.make_mixed_batch(forward_iter, 514))
            self.assertIsNotNone(lstep.start_event)
            self.assertIsNone(fstep.start_event)  # followers never time
            if elapsed_ms is not None:
                lstep.start_event = _FakeEvent(
                    ready_after=ready_after, elapsed_ms=elapsed_ms
                )
                lstep.end_event = _FakeEvent(
                    ready_after=ready_after, elapsed_ms=elapsed_ms
                )
            leader.end_step(lstep)
            follower.end_step(fstep)
        leader.process_batch_result(SimpleNamespace(forward_iter=forward_iter))
        follower.process_batch_result(SimpleNamespace(forward_iter=forward_iter))

    def test_follower_converges_to_leader_budget(self):
        leader, follower = self.make_pair()
        with patch(
            "sglang.srt.managers.scheduler_components.mixed_prefill_budget."
            "broadcast_pyobj",
            _fake_broadcast_pyobj,
        ):
            # Slow step shrinks both budgets, fast step grows them again.
            self.drive_step(leader, follower, 1, elapsed_ms=1600.0)
            self.drive_step(leader, follower, 2, elapsed_ms=80.0)
            # Leader's events are not ready yet: both ranks keep the sample
            # pending and stay in lockstep.
            self.drive_step(leader, follower, 3, elapsed_ms=None)
        self.assertEqual(leader.budget.tokens, follower.budget.tokens)
        self.assertEqual(leader.budget._fit, follower.budget._fit)
        self.assertLess(leader.budget.tokens, 512)
        self.assertEqual(len(leader._pending), 1)
        self.assertEqual(len(follower._pending), 1)

    def test_unresolvable_head_is_discarded_identically(self):
        leader, follower = self.make_pair()
        with (
            patch("torch.cuda.Event", _FakeEvent),
            patch(
                "sglang.srt.managers.scheduler_components.mixed_prefill_budget."
                "broadcast_pyobj",
                _fake_broadcast_pyobj,
            ),
        ):

            def never_ready(ready_after=10**9, elapsed_ms=100.0, **kwargs):
                return _FakeEvent(ready_after=ready_after, elapsed_ms=elapsed_ms)

            with patch("torch.cuda.Event", never_ready):
                lstep = leader.begin_step(self.make_mixed_batch(1, 514))
                follower.begin_step(self.make_mixed_batch(1, 514))
                leader.end_step(lstep)
            for _ in range(leader.MAX_RESOLVE_ATTEMPTS):
                leader.process_batch_result(SimpleNamespace(forward_iter=99))
                follower.process_batch_result(SimpleNamespace(forward_iter=99))
        # The stuck head is dropped on both ranks without feeding the fit.
        self.assertEqual(len(leader._pending), 0)
        self.assertEqual(len(follower._pending), 0)
        self.assertEqual(leader.budget._fit, [0.0] * 5)
        self.assertEqual(follower.budget._fit, [0.0] * 5)


class TestMaybeInitMixedPrefillBudgetGate(CustomTestCase):
    """Boot gate for maybe_init_mixed_prefill_budget (review M4, 2026-11-10).

    The controller synchronizes ranks through broadcast_pyobj on the WORLD
    group (src=0, CPU collective). Under DP attention each scheduler replica
    serves a different request stream: the pending-queue path that drains the
    broadcast runs a different number of times per DP group, so the world
    collective mismatches and the engine hangs. DP attention must refuse to
    arm the controller at boot, mirroring the other hard prerequisites.
    """

    def _make_scheduler(self):
        s = object.__new__(Scheduler)
        s.enable_overlap = True
        s.is_mixed_chunk = True
        s.chunked_prefill_size = 4096
        s.page_size = 64
        s.mixed_prefill_budget = None
        return s

    def _run_gate(self, s, attn_dp_enabled):
        env = {"SGLANG_MIXED_PREFILL_STEP_LATENCY_MS": "250"}
        parallel = SimpleNamespace(
            world_group=SimpleNamespace(rank=0),
            attn_dp_enabled=attn_dp_enabled,
        )
        with (
            patch.dict(os.environ, env),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=parallel,
            ),
        ):
            Scheduler.maybe_init_mixed_prefill_budget(s)
        return s.mixed_prefill_budget

    def test_dp_attention_refuses_to_arm(self):
        s = self._make_scheduler()
        self.assertIsNone(self._run_gate(s, attn_dp_enabled=True))

    def test_without_dp_attention_controller_arms(self):
        s = self._make_scheduler()
        control = self._run_gate(s, attn_dp_enabled=False)
        self.assertIsNotNone(control)
        self.assertEqual(control.budget.align, 64)

    def test_target_zero_still_disables(self):
        s = self._make_scheduler()
        env = {"SGLANG_MIXED_PREFILL_STEP_LATENCY_MS": "0"}
        parallel = SimpleNamespace(
            world_group=SimpleNamespace(rank=0), attn_dp_enabled=False
        )
        with (
            patch.dict(os.environ, env),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=parallel,
            ),
        ):
            Scheduler.maybe_init_mixed_prefill_budget(s)
        self.assertIsNone(s.mixed_prefill_budget)


if __name__ == "__main__":
    unittest.main()
