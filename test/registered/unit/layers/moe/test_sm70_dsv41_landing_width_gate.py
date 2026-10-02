"""D4-G spill landing width gate: the landing >= T*topk capacity invariant.

Bug regression (P0): spill_assign_cached silently drops every unique expert
past the pool depth, and a CUDA-graph-captured MTP verify batch (bs*4 tokens
x routed top-k 8) exceeded landing=12 -- drops climbed 29 -> 5065 in real
traffic while serving looked healthy, zeroing expert contributions and
poisoning accept stats. A `max(2, ...)` floor on the accepted width re-opened
the same hole for pools of 8-15 slots (T=2 -> 16 uniques > slots). The gate
now derives the captured width from the pool depth; these cases pin the
invariant `_decode_shaped_max_tokens() * topk <= spill_landing_slots()` for
every landing config, the sub-capacity eager fallback, and the prefill gate's
"pool holds every logical expert" precondition.
"""

import os
import unittest
from types import SimpleNamespace

import torch

import sglang.srt.layers.moe.dsv41_expert_spill as spill
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")

LANDING_VAR = "SGLANG_DSV41_SPILL_LANDING"
PREFILL_VAR = "SGLANG_DSV41_SPILL_PREFILL_LANDING"
TOPK = 8  # GLM-5.3 routed top-k; the gate formula's worst-case unit


class TestLandingWidthGate(CustomTestCase):
    def setUp(self):
        self._saved = {v: os.environ.get(v) for v in (LANDING_VAR, PREFILL_VAR)}
        for v in self._saved:
            os.environ.pop(v, None)

    def tearDown(self):
        for v, val in self._saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val

    def _set_slots(self, n: int, var: str = LANDING_VAR) -> None:
        os.environ[var] = str(n)

    def test_width_fits_pool_for_every_config(self):
        # The invariant the P0 incident broke: the widest accepted batch's
        # worst-case unique expert set (T * topk) must fit the pool, and the
        # decode graph width stays capped at 6.
        for slots in [8, 12, 15, 16, 17, 24, 32, 48, 64, 96, 192, 288, 336, 512]:
            self._set_slots(slots)
            width = spill._decode_shaped_max_tokens()
            self.assertGreaterEqual(width, 1, msg=f"slots={slots}")
            self.assertLessEqual(width * TOPK, slots, msg=f"slots={slots}")
            self.assertLessEqual(width, 6, msg=f"slots={slots}")

    def test_sub_capacity_pool_routes_eager(self):
        # Below 8 slots even T=1 cannot fit its worst-case unique set; the
        # gate must route every batch to the eager path (captured shapes
        # then fail loudly at the width guard instead of dropping experts).
        for slots in [0, 1, 4, 7]:
            self._set_slots(slots)
            self.assertEqual(
                spill._decode_shaped_max_tokens(), 0, msg=f"slots={slots}"
            )

    def test_decode_shaped_boundary(self):
        self._set_slots(48)  # width 6
        self.assertTrue(spill._decode_shaped_topk(torch.zeros((6, TOPK), dtype=torch.int32)))
        self.assertFalse(spill._decode_shaped_topk(torch.zeros((7, TOPK), dtype=torch.int32)))
        self.assertFalse(spill._decode_shaped_topk(torch.zeros((0, TOPK), dtype=torch.int32)))

    def test_prefill_landing_needs_full_expert_coverage(self):
        # A wide batch can request every logical expert of a layer in one
        # call; routing it through a narrower pool would drop experts. The
        # gate must hold exactly when the pool covers the layer.
        moe = SimpleNamespace(
            _dsv41_landing_pool=SimpleNamespace(),
            _dsv41_expert_lru=SimpleNamespace(n_experts=288),
        )
        os.environ[PREFILL_VAR] = "1"
        self._set_slots(336)
        self.assertTrue(spill._prefill_landing_ready(moe))
        self._set_slots(192)
        self.assertFalse(spill._prefill_landing_ready(moe))
        os.environ.pop(PREFILL_VAR)
        self.assertFalse(spill._prefill_landing_ready(moe))


if __name__ == "__main__":
    unittest.main()
