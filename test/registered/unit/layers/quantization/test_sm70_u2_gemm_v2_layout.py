"""Unit tests for the sm70_u2_gemm_v2 transposed word layout (u2b2 P5-b).

The v2 kernel consumes a k-major word layout that production derives from the
persistently cached marlin u2 words via ``u2_packed_to_T`` (inverse of the
repack chain). If that inverse drifts from the direct builder, the marlin
kernel never sees it (SGLANG_USE_SM70_U2_GEMM_V2 binds T-layout bytes
exclusively), so the only symptom would be silently wrong expert outputs --
this test is the layout's black-box contract:

    codes -> repack_u2_sm70 -> u2_packed_to_T   ==   repack_u2_sm70T(codes)

bit-exact, across the GLM-5.3 TP4 pool shapes and every packed_macro_n.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.sm70_u2_pool import (
    _u2_v2_layout_ok,
    repack_u2_sm70,
    repack_u2_sm70T,
    requant_u2,
    u2_packed_to_T,
)
from sglang.test.test_utils import CustomTestCase

# GLM-5.3-Flash TP4 per-rank pool shapes (r, c, macro): w13 and w2.
_GLM_SHAPES = [
    (2048, 4096, 256),  # w13: r=n13=2I(rank), c=hidden, macro13
    (4096, 512, 256),  # w2: r=hidden, c=inter(rank), macro2
]


class TestSm70U2GemmV2Layout(CustomTestCase):
    def test_inverse_chain_bit_exact_real_shapes(self):
        torch.manual_seed(0)
        for r, c, macro in _GLM_SHAPES:
            codes = torch.randint(0, 4, (4, r, c), dtype=torch.uint8)
            via_marlin = u2_packed_to_T(repack_u2_sm70(codes, macro), r, c, macro)
            direct = repack_u2_sm70T(codes)
            self.assertTrue(
                torch.equal(via_marlin, direct),
                f"u2_packed_to_T(repack_u2_sm70) != repack_u2_sm70T "
                f"for r={r} c={c} macro={macro}",
            )

    def test_inverse_chain_bit_exact_all_macros(self):
        torch.manual_seed(1)
        for macro in (64, 128, 256):
            # r=256 is the smallest n compatible with every packed_macro_n.
            codes = torch.randint(0, 4, (2, 256, 256), dtype=torch.uint8)
            via_marlin = u2_packed_to_T(
                repack_u2_sm70(codes, macro), 256, 256, macro
            )
            self.assertTrue(torch.equal(via_marlin, repack_u2_sm70T(codes)))

    def test_requant_roundtrip_end_to_end(self):
        """Full chain from fp32 weights through requant, both paths agree."""
        torch.manual_seed(2)
        w = torch.randn(2, 256, 512, dtype=torch.float32) / 20
        codes, _ = requant_u2(w, 128)
        via_marlin = u2_packed_to_T(repack_u2_sm70(codes, 256), 256, 512, 256)
        self.assertTrue(torch.equal(via_marlin, repack_u2_sm70T(codes)))

    def test_shape_gate(self):
        glm = {
            "hidden": 4096,
            "inter": 512,
            "n13": 2048,
            "group_size": 128,
        }
        self.assertTrue(_u2_v2_layout_ok(glm))
        for key, bad in (
            ("hidden", 4064),  # not %64
            ("inter", 508),
            ("n13", 2040),
            ("group_size", 96),
        ):
            layout = dict(glm)
            layout[key] = bad
            self.assertFalse(_u2_v2_layout_ok(layout), f"{key}={bad} must fail")


if __name__ == "__main__":
    unittest.main(verbosity=3)
