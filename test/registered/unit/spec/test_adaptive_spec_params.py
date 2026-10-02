import json
import tempfile
import unittest

from sglang.srt.environ import envs
from sglang.srt.speculative.adaptive_spec_params import (
    _PROBE_CT,
    AdaptiveSpeculativeParams,
    AdaptiveStepSlot,
    CostAwareSpeculativeParams,
    CostAwareStepSlot,
    _cost_shape,
    make_adaptive_policy,
    resolve_candidate_steps_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


class TestAdaptiveStepSlot(CustomTestCase):
    def _make_params_from_config(self, initial_steps: int, config: dict):
        return AdaptiveStepSlot(initial_steps=initial_steps, cfg=config)

    def test_initial_steps_snaps_to_middle_when_missing(self):
        params = self._make_params_from_config(2, {"candidate_steps": [1, 3, 7]})

        self.assertEqual(params.candidate_steps, [1, 3, 7])
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

    def test_update_respects_warmup_and_interval(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 2,
            },
        )

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_empty_batches_do_not_consume_warmup_or_shift_steps(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 1,
            },
        )

        self.assertFalse(params.update([]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_update_scales_up_across_candidates(self):
        params = self._make_params_from_config(
            1,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_update_can_scale_down_across_candidates_in_one_recompute(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_exact_rise_threshold_does_not_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertFalse(params.update([2, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.5)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_exact_drop_threshold_does_downshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
                "up_hysteresis": 0.5,
            },
        )

        self.assertTrue(params.update([0, 1]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.5)

    def test_hysteresis_can_prevent_premature_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.75,
            },
        )

        self.assertFalse(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)

    def test_down_hysteresis_can_prevent_premature_downshift(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": -0.75,
            },
        )

        self.assertFalse(params.update([2, 2]))
        self.assertEqual(params.current_steps, 7)

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

    def test_multi_batch_sequence_can_ramp_up_then_back_down(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 0.5,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)
        self.assertEqual(params.ema_accept_len, 3.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 1.5)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 0.75)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.375)

    def test_zero_step_mixed_slot_drops_probes_and_rechecks(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [0, 3],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 0)
        self.assertEqual(params.ema_accept_len, 0.0)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 0.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 0)
        self.assertEqual(params.ema_accept_len, 0.0)

    def test_ceiling_coeff_caps_steps(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "ceiling_coeff": 1.0,
            },
        )
        # Force low ema to trigger ceiling
        params.ema_accept_len = 1.0
        self.assertTrue(params.update([1, 1]))
        # ceiling = ceil(1.0 * 1.0) = 1, target capped to 1
        self.assertEqual(params.current_steps, 1)

    def test_ceiling_disabled_by_default(self):
        params = self._make_params_from_config(3, {"candidate_steps": [1, 3, 7]})
        self.assertEqual(params.ceiling_coeff, 0)


class TestAdaptiveSpeculativeParams(CustomTestCase):
    def test_default_config_loads(self):
        params = AdaptiveSpeculativeParams(initial_steps=3)
        self.assertEqual(params._bs_list, [1, 8, 32, 64])
        self.assertEqual(params._slots[1].candidate_steps, [1, 3, 5, 7])
        self.assertEqual(params._slots[8].candidate_steps, [0, 1, 3])
        self.assertEqual(params._slots[32].candidate_steps, [0, 1])
        self.assertEqual(params._slots[64].candidate_steps, [0])

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5], "up_hysteresis": 0.3},
                    "32": {"candidate_steps": [1, 2]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=5, cfg_path=f.name)
        self.assertEqual(params._bs_list, [1, 32])
        # Slots are built straight from the config; the launch flag never pollutes
        # them. initial_steps just selects the smallest slot's starting step.
        self.assertEqual(params._slots[1].candidate_steps, [1, 5])
        self.assertEqual(params._slots[1].current_steps, 5)
        self.assertEqual(params._slots[1].up_hysteresis, 0.3)
        self.assertEqual(params._slots[32].candidate_steps, [1, 2])

    def test_launch_flag_not_injected_into_slots(self):
        # initial_steps lives only in a larger slot. It must NOT be merged into
        # any other slot's candidates: slots come straight from the config.
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5]},
                    "8": {"candidate_steps": [1, 3, 7]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=7, cfg_path=f.name)
        self.assertEqual(params._slots[1].candidate_steps, [1, 5])
        self.assertEqual(params._slots[8].candidate_steps, [1, 3, 7])
        # The slot that does not own initial_steps starts at its own median.
        self.assertEqual(params._slots[1].current_steps, 5)
        self.assertEqual(params._slots[8].current_steps, 7)

    def test_invalid_config_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"not_a_bs": "bad"}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_invalid_steps_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": "bad"}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_empty_steps_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": []}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_global_hysteresis_inherited(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        self.assertEqual(params._slots[1].up_hysteresis, 0.5)

    def test_entry_hysteresis_overrides_global(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3], "up_hysteresis": 0.1},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        self.assertEqual(params._slots[1].up_hysteresis, 0.1)


class TestBatchSizeRouting(CustomTestCase):
    """BS-aware routing: batch size selects the slot, CUDA-graph BS pads first."""

    def _params(self):
        # Slots: bs=1 -> [1,3,5,7], bs=8 -> [0,1,3], bs=32 -> [0,1].
        return AdaptiveSpeculativeParams(initial_steps=3)

    def test_routes_to_floor_slot_without_cuda_graph(self):
        params = self._params()
        # A batch maps to the largest slot BS <= batch (floor), capped at the top slot.
        self.assertEqual(params._route(1).candidate_steps, [1, 3, 5, 7])
        self.assertEqual(params._route(7).candidate_steps, [1, 3, 5, 7])
        self.assertEqual(params._route(8).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(31).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(32).candidate_steps, [0, 1])
        self.assertEqual(params._route(1000).candidate_steps, [0])

    def test_cuda_graph_bs_pads_batch_up_before_routing(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        # bs=5 pads up to the captured graph BS 8 -> slot bs=8.
        self.assertEqual(params._route(5).candidate_steps, [0, 1, 3])
        # bs=17 pads up to 32 -> slot bs=32.
        self.assertEqual(params._route(17).candidate_steps, [0, 1])
        # A batch larger than every captured BS keeps its own value -> top slot.
        self.assertEqual(params._route(100).candidate_steps, [0])

    def test_cuda_graph_bs_for_step_prunes_unreachable_graphs(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        # step=1 is reachable from every slot.
        self.assertEqual(params.cuda_graph_bs_for_step(1), [4, 8, 16, 32])
        # step=3 lives in the bs=1 and bs=8 slots: graphs 4,8,16 floor into them.
        self.assertEqual(params.cuda_graph_bs_for_step(3), [4, 8, 16])
        # step=5 lives only in the bs=1 slot: only graph BS 4 floors into it.
        self.assertEqual(params.cuda_graph_bs_for_step(5), [4])
        # step=7 lives only in the bs=1 slot: only graph BS 4 floors into it.
        self.assertEqual(params.cuda_graph_bs_for_step(7), [4])

    def test_cuda_graph_bs_for_step_returns_none_when_disabled(self):
        params = self._params()
        self.assertIsNone(params.cuda_graph_bs_for_step(7))
        params.set_cuda_graph_bs(None)
        self.assertIsNone(params.cuda_graph_bs_for_step(7))

    def test_observe_verify_feeds_the_routed_slot(self):
        params = self._params()
        # Drive the bs=1 slot up with perfect acceptance; the bs=32 slot is
        # untouched and stays at its single candidate step.
        for _ in range(40):
            params.on_verify_complete([7, 7, 7], batch_size=1)
        self.assertGreater(params.get_steps_for_batch(1), 1)
        self.assertEqual(params.get_steps_for_batch(32), 1)


class TestResolveCandidateSteps(CustomTestCase):
    def test_default_config(self):
        steps = resolve_candidate_steps_from_config()
        self.assertEqual(steps, [0, 1, 3, 5, 7])

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": [2, 4]}}, f)
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [2, 4])

    def test_unions_and_dedups_across_slots(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5]},
                    "8": {"candidate_steps": [3, 5, 7]},
                },
                f,
            )
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [1, 3, 5, 7])


class TestCostAwareStepSlot(CustomTestCase):
    """Cost-aware width policy: argmax(accept_length / round_ms) with probing.

    EMAs are seeded directly so each case isolates one decision rule.
    """

    def _slot(self, candidates, initial_steps):
        return CostAwareStepSlot(
            initial_steps=initial_steps,
            cfg={
                "candidate_steps": candidates,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )

    def _seed(self, slot, steps, cost_ms, accept_len, ct=1.0):
        slot._cost_ms[steps] = cost_ms
        slot._cost_ct[steps] = ct
        slot._accept_len[steps] = accept_len
        slot._accept_ct[steps] = ct

    def test_switches_to_width_with_best_tokens_per_ms(self):
        slot = self._slot([1, 5], initial_steps=1)
        # Both widths measured past the probe threshold so the margin rule is
        # what decides.
        self._seed(slot, 1, cost_ms=10.0, accept_len=2.0, ct=10.0)  # score 0.20
        self._seed(slot, 5, cost_ms=12.0, accept_len=3.0, ct=5.0)  # score 0.25

        # Running width 1 at 10 ms for 2 tokens/round scores 0.20: width 5
        # beats it beyond the margin, so the policy must move despite width 5
        # costing more per round (the acceptance policy would not).
        self.assertTrue(slot.update([1, 1, 1], round_ms=10.0))
        self.assertEqual(slot.current_steps, 5)

    def test_stays_when_better_width_is_within_margin(self):
        slot = self._slot([1, 5], initial_steps=1)
        self._seed(slot, 1, cost_ms=10.0, accept_len=2.0, ct=10.0)  # score 0.20
        self._seed(slot, 5, cost_ms=9.8, accept_len=2.0, ct=5.0)  # score 0.2041

        # Width 5 scores within the 3% margin over width 1 (0.20): the
        # narrower verify stays, guarding the stickiness rule.
        self.assertFalse(slot.update([1, 1, 1], round_ms=10.0))
        self.assertEqual(slot.current_steps, 1)

    def test_probes_unmeasured_width_the_prior_ranks_first(self):
        slot = self._slot([1, 3], initial_steps=1)
        # Measured width 1 is poor: 1.33 tokens at 20 ms scores 0.067. The
        # unmeasured width-3 prior scores 2.4 / 34 = 0.071 and wins on paper.
        # The sweep finishes measuring the incumbent first (_PROBE_CT rounds),
        # then probes width 3: a guessed cost that never gets measured can
        # never be refuted.
        for _ in range(4):
            slot.update([0, 1, 0], round_ms=20.0)
        self.assertEqual(slot.current_steps, 3)

    def test_cold_start_sweep_probes_losing_prior_width(self):
        slot = self._slot([1, 5], initial_steps=1)
        self._seed(slot, 1, cost_ms=10.0, accept_len=2.0, ct=10.0)  # score 0.20

        # Width 5 is unmeasured and its PRIOR ranks it below the incumbent
        # (3.8 over a shape-scaled 24.5 ms guess scores 0.155): the old
        # rank-first-only probe never tried it. On GLM easy text its real
        # accept crushed the prior (5.9 vs 3.8) and cost-aware stayed 6%
        # slower than the acceptance policy for the whole workload -- so the
        # sweep measures every candidate regardless of prior rank.
        self.assertTrue(slot.update([1], round_ms=10.0))
        self.assertEqual(slot.current_steps, 5)

    def test_probe_finishes_before_abandoning(self):
        slot = self._slot([1, 5], initial_steps=1)
        self._seed(slot, 1, cost_ms=10.0, accept_len=2.0, ct=10.0)  # score 0.20

        # Width 5 is mid-probe with one bad round (score 0.033): finish its
        # _PROBE_CT rounds before argmax can send the slot elsewhere --
        # abandoning a probe on the first sample makes decisions noise-driven.
        self._seed(slot, 5, cost_ms=30.0, accept_len=1.0, ct=1.0)
        slot.current_steps = 5
        self.assertFalse(slot.update([0], round_ms=30.0))
        self.assertEqual(slot.current_steps, 5)

    def test_missing_round_time_never_fabricates_a_cost_sample(self):
        slot = self._slot([1, 3], initial_steps=1)
        switched = False
        for _ in range(20):
            switched |= slot.update([0, 1, 0], round_ms=None)

        self.assertEqual(slot._cost_ms, {})
        # Decisions still happen, driven by the shape prior.
        self.assertTrue(switched)

    def test_zero_step_scores_one_token_not_the_linear_prior(self):
        slot = self._slot([0, 1], initial_steps=0)
        # 1 + 0.7*(0-1) = 0.3 would make steps=0 look hopeless and forever
        # unselectable; a no-draft round emits exactly one token.
        self.assertEqual(slot._tokens(0), 1.0)

    def test_cost_guess_scales_from_measured_widths_with_capped_weights(self):
        slot = self._slot([1, 3, 5], initial_steps=1)
        slot._cost_ms = {1: 20.0, 5: 24.5}
        slot._cost_ct = {1: 25.0, 5: 1.0}

        # Scaled to width 3: width-1 contributes 20*1.7 = 34 per round but is
        # capped at weight 20, width-5 contributes 24.5*(1.7/2.45) = 17.
        # (680 + 17) / 21 = 33.19; an uncapped weight would give 33.35.
        self.assertAlmostEqual(slot._cost_ms_est(3), 33.19, places=2)


class TestCostAwarePolicyWiring(CustomTestCase):
    def test_factory_selects_policy_by_env(self):
        with envs.SGLANG_ADAPTIVE_SPEC_COST_AWARE.override(False):
            policy = make_adaptive_policy(initial_steps=3)
        self.assertNotIsInstance(policy, CostAwareSpeculativeParams)

        with envs.SGLANG_ADAPTIVE_SPEC_COST_AWARE.override(True):
            policy = make_adaptive_policy(initial_steps=3)
        self.assertIsInstance(policy, CostAwareSpeculativeParams)

    def test_acceptance_policy_tolerates_round_ms_kwarg(self):
        # The controller passes round_ms unconditionally; the acceptance
        # policy must accept and ignore it.
        params = AdaptiveSpeculativeParams(initial_steps=3)
        self.assertIsNone(
            params.on_verify_complete([2, 2], batch_size=1, round_ms=123.4)
        )

    def test_round_ms_reaches_the_routed_cost_slot(self):
        params = CostAwareSpeculativeParams(initial_steps=3)
        self.assertIsNone(params.on_verify_complete([2], batch_size=1, round_ms=42.0))
        slot = params._route(1)
        self.assertEqual(slot._cost_ms, {3: 42.0})
        self.assertEqual(slot._accept_len, {3: 3.0})


class TestCostAwareMeasurementAging(CustomTestCase):
    """Aging semantics in isolation: the chooser is gated off (huge update
    interval) so rounds only fold measurements and decay counts; a live
    policy would re-probe the stale width mid-phase and break the setup.
    """

    def _slot(self):
        return CostAwareStepSlot(
            initial_steps=3,
            cfg={
                "candidate_steps": [1, 3, 5],
                "warmup_batches": 0,
                "update_interval": 10**9,
            },
        )

    def _run(self, slot, steps, rounds, round_ms, accept):
        slot.current_steps = steps
        for _ in range(rounds):
            slot.update([int(accept) - 1], round_ms)

    def test_stale_width_reverts_to_prior(self):
        slot = self._slot()
        self._run(slot, 3, 60, 900.0, 4)  # ct_3 -> ~35, measured values
        self._run(slot, 1, 200, 600.0, 2)  # width 3 idle: ct 35*0.98^200 < 1
        self.assertNotIn(3, slot._cost_ms)
        self.assertNotIn(3, slot._accept_len)
        # Scoring falls back to the priors, not to the stale measurements.
        self.assertEqual(slot._tokens(3), 1.0 + 0.7 * 2)
        self.assertEqual(slot._cost_ms_est(3), _cost_shape(3) * 600.0 / _cost_shape(1))

    def test_incumbent_never_loses_measurements(self):
        slot = self._slot()
        self._run(slot, 3, 400, 900.0, 4)
        self.assertGreater(slot._cost_ct[3], 1.0)
        self.assertEqual(slot._tokens(3), 4.0)

    def test_decayed_width_reprobes_after_regime_shift(self):
        slot = self._slot()
        self._run(slot, 3, 60, 900.0, 4)
        self._run(slot, 1, 150, 600.0, 2)  # enough idle that ct_3 drops low
        # Decay below the probe count makes width 3 probe-eligible again,
        # instead of being blocked by stale-low cost estimates forever.
        self.assertLess(slot._cost_ct.get(3, 0.0), _PROBE_CT)
        self.assertLess(slot._accept_ct.get(3, 0.0), _PROBE_CT)


if __name__ == "__main__":
    unittest.main()
