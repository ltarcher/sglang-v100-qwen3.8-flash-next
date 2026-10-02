"""SM70 NVFP4 Marlin MoE: gate/up fold-down contract + kernel numerics.

Regression for the fold-overflow NaN. float8_e4m3fn has no inf, so a cast that
overflows yields NaN. Folding the gate/up weight_scale_2 pair UP multiplied
amax-adjacent block scales (near the 448 ceiling) by up/gate > 1 and NaNed
real GLM-5.3 checkpoint scale bytes; every token routed to those experts then
produced NaN logits. The production path folds DOWN to max(gate, up), so both
halves' ratios are <= 1 and overflow is impossible.

Kernel tests quantize synthetic gate/up matrices with INDEPENDENT per-matrix
weight_scale_2 (as real checkpoints ship), run the production fold + repack +
scale processing, and compare fused_marlin_moe against a torch dequant
reference at M=1 (moe_align_single_token path), 4 and 64.

Tiny fixtures only. Do not load a real checkpoint here; the real-ckpt sweep
lives in scripts/m2_real_ckpt_unit.py.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.quantization.marlin_utils import (
    sm70_nvfp4_marlin_process_global_scale,
    sm70_nvfp4_marlin_process_scales,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

# Plain unittest skip (not pytest.mark) so the file also runs via
# `python test_...py` in environments without pytest.
_sm70 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
requires_sm70 = unittest.skipUnless(
    _sm70, "SM70 NVFP4 Marlin MoE requires an NVIDIA V100"
)

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E4M3_MAX = 448.0


def _fold_down(scales: torch.Tensor, own_s2: torch.Tensor, s2_eff: torch.Tensor):
    """Mirror of the SM70 branch in ModelOptNvFp4FusedMoEMethod.

    scales: [R, K/16] e4m3; own_s2/s2_eff: [R] fp32. Returns folded scales and
    the per-row effective s2.
    """
    # scales gets unsqueezed below, so the ratio needs scales.dim() ones.
    ratio = (own_s2 / s2_eff).view(-1, *([1] * scales.dim())).to(scales.device)
    folded = scales.float().unsqueeze(-1).mul(ratio).clamp_(max=E4M3_MAX).squeeze(-1)
    return folded.to(scales.dtype), s2_eff.expand(own_s2.shape[0])


@requires_sm70
class TestFoldDownContract(unittest.TestCase):
    """The fold direction is a hard invariant: up overflows, down cannot."""

    def test_fold_up_overflows_e4m3_to_nan(self):
        # Block scales sitting at the amax-adjacent ceiling, ratio 1.18 like
        # the worst real GLM expert: folding UP must produce NaN bytes.
        scales = torch.full((64, 16), 440.0).to(torch.float8_e4m3fn)
        assert not scales.float().isnan().any()
        folded_up = (scales.float() * 1.18).to(torch.float8_e4m3fn)
        self.assertTrue(folded_up.float().isnan().any())

    def test_fold_down_stays_finite_under_ceiling(self):
        # Realistic amax-adjacent magnitudes and the observed ratio range.
        gen = torch.Generator().manual_seed(0)
        scales = (torch.rand(256, 32, generator=gen) * 480).clamp(max=E4M3_MAX)
        scales = scales.to(torch.float8_e4m3fn).float()
        own_s2 = torch.rand(256, generator=gen) + 0.5
        ratio_mag = 1.0 + torch.rand(256, generator=gen) * 0.18
        s2_eff = own_s2 * ratio_mag  # > own: the fold-down direction
        folded, eff = _fold_down(scales.to(torch.float8_e4m3fn), own_s2, s2_eff)
        self.assertFalse(folded.float().isnan().any())
        self.assertFalse(torch.isinf(folded.float()).any())
        self.assertLessEqual(folded.float().max().item(), E4M3_MAX)

        # Effective dequant must match the unfused original up to e4m3
        # re-rounding of folded = round(scale * ratio): 3 mantissa bits give a
        # worst-case half-ulp relative error of 2**-4.
        code = torch.where(
            torch.rand(256, 32, generator=gen) < 0.5, 1.0, -1.0
        ) * (1.0 + torch.randint(0, 7, (256, 32), generator=gen))
        orig = code * scales * own_s2.view(-1, 1)
        new = code * folded.float() * eff.view(-1, 1)
        rel = ((new - orig).abs() / orig.abs().clamp_min(1e-6)).max().item()
        self.assertLess(rel, 0.08)

    def test_fold_down_equal_scales_is_identity(self):
        # Checkpoints where gate/up ship equal weight_scale_2 take the identity
        # arm; folded == input there.
        scales = (torch.rand(64, 16) * 400 + 40).to(torch.float8_e4m3fn)
        s2 = torch.full((64,), 0.75)
        folded, eff = _fold_down(scales, s2, s2.clone())
        self.assertTrue(torch.equal(folded, scales))
        self.assertTrue(torch.allclose(eff, s2))


@requires_sm70
class TestSm70Nvfp4MarlinMoeKernel(unittest.TestCase):
    E, N, K, TOPK = 4, 128, 512, 2

    @staticmethod
    def _quantize(w: torch.Tensor):
        """modelopt NVFP4: w = code * scale_e4m3 * weight_scale_2, per tensor."""
        r, c = w.shape
        wf = w.float()
        s2 = (wf.abs().amax() / (E4M3_MAX * 6.0)).clamp(min=1e-30)
        blocks = (wf / s2).reshape(r, c // 16, 16)
        block_max = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scales = (block_max / 6.0).to(torch.float8_e4m3fn).squeeze(-1)
        q = blocks / scales.float().unsqueeze(-1)
        sign = (q < 0).to(torch.uint8) << 3
        edges = torch.tensor(
            [(GRID[i + 1] + GRID[i]) / 2 for i in range(7)], device=w.device
        )
        idx = torch.bucketize(q.abs().clamp_max(6.0), edges)
        lut = torch.tensor(GRID, device=w.device, dtype=torch.uint8)
        codes = (lut[idx] | sign).reshape(r, c // 16, 16)
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(r, c // 2)
        return packed, scales, s2

    def _repack(self, weight_u8: torch.Tensor) -> torch.Tensor:
        from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
            gptq_marlin_moe_repack,
        )

        ne, sn, pk = weight_u8.shape
        gptq_layout = (
            weight_u8.contiguous().view(torch.int32).transpose(1, 2).contiguous()
        )
        empty_perm = torch.empty((ne, 0), dtype=torch.int32, device=weight_u8.device)
        return gptq_marlin_moe_repack(gptq_layout, empty_perm, pk * 2, sn, 4)

    def _dequant(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        s2_per_half,
    ) -> torch.Tensor:
        """[R, K/2] u8 + [R, K/16] e4m3 -> [R, K]; s2 scalar or (gate, up)."""
        r = packed.shape[0]
        lut = torch.tensor(GRID + [-g for g in GRID], device=packed.device)
        lo = (packed & 0x0F).long()
        hi = (packed >> 4).long()
        dec = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(r, -1)
        sf = scales.view(torch.float8_e4m3fn).float()
        sf = torch.stack([sf] * 16, dim=-1).reshape(r, -1)
        if s2_per_half.dim() == 1 and s2_per_half.numel() == 2:
            half = r // 2
            s2 = torch.cat(
                [s2_per_half[0].expand(half), s2_per_half[1].expand(r - half)]
            )
        else:
            s2 = s2_per_half.expand(r)
        return dec * sf * s2.view(-1, 1)

    def test_marlin_matches_dequant_reference(self):
        dev = "cuda"
        gen = torch.Generator(device=dev).manual_seed(0)
        E, N, K, TOPK = self.E, self.N, self.K, self.TOPK

        # Gate/up quantized independently: distinct weight_scale_2 per half,
        # exercising the fold on the kernel path.
        w13_u8, w13_sc, w13_s2 = [], [], []
        for _ in range(E):
            pg, sg, g2 = self._quantize(torch.randn(N, K, device=dev, generator=gen) * 0.02)
            pu, su, u2 = self._quantize(torch.randn(N, K, device=dev, generator=gen) * 0.02)
            w13_u8.append(torch.cat([pg, pu]))
            w13_sc.append(torch.cat([sg, su]))
            w13_s2.append(torch.stack([g2, u2]))
        w13_u8 = torch.stack(w13_u8)
        w13_sc = torch.stack(w13_sc)
        w13_s2 = torch.stack(w13_s2).float()
        w2_u8, w2_sc, w2_s2 = [], [], []
        for _ in range(E):
            p2, s2c, s22 = self._quantize(
                torch.randn(K, N, device=dev, generator=gen) * 0.02
            )
            w2_u8.append(p2)
            w2_sc.append(s2c)
            w2_s2.append(s22)
        w2_u8 = torch.stack(w2_u8)
        w2_sc = torch.stack(w2_sc)
        w2_s2 = torch.stack(w2_s2)

        # Production fold-down, then repack + scale processing.
        gate_s2, up_s2 = w13_s2[:, 0], w13_s2[:, 1]
        s2_eff = torch.maximum(gate_s2, up_s2)
        half = w13_sc.shape[1] // 2
        folded_gate, _ = _fold_down(w13_sc[:, :half, :], gate_s2, s2_eff)
        folded_up, _ = _fold_down(w13_sc[:, half:, :], up_s2, s2_eff)
        w13_sc_folded = torch.cat([folded_gate, folded_up], dim=1)
        self.assertFalse(w13_sc_folded.float().isnan().any())

        w13_m = self._repack(w13_u8)
        w2_m = self._repack(w2_u8)
        w13_sc_m, f13 = sm70_nvfp4_marlin_process_scales(
            w13_sc_folded.transpose(1, 2).contiguous(), torch.float16
        )
        w2_sc_m, f2 = sm70_nvfp4_marlin_process_scales(
            w2_sc.transpose(1, 2).contiguous(), torch.float16
        )
        g13 = (
            sm70_nvfp4_marlin_process_global_scale(s2_eff, torch.float16) / f13
        ).float()
        g2 = (
            sm70_nvfp4_marlin_process_global_scale(w2_s2.float(), torch.float16) / f2
        ).float()

        for M in (1, 4, 64):
            x = torch.randn(M, K, device=dev, dtype=torch.float16, generator=gen) * 0.5
            logits = torch.randn(M, E, device=dev, generator=gen)
            topk_vals, topk_ids = torch.topk(logits, TOPK, dim=-1)
            topk_w = torch.sigmoid(topk_vals.float())
            topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(torch.float16)

            out = fused_marlin_moe(
                x,
                w13_m,
                w2_m,
                w13_sc_m,
                w2_sc_m,
                logits,
                topk_w,
                topk_ids.to(torch.int32),
                num_bits=4,
                activation="silu",
                is_gated=True,
                clamp_limit=10.0,
                w1_global_scale=g13,
                w2_global_scale=g2,
            )
            self.assertFalse(
                out.isnan().any(), f"NaN in fused_marlin_moe output at M={M}"
            )

            ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
            for m in range(M):
                for j in range(TOPK):
                    e = int(topk_ids[m, j])
                    d13 = self._dequant(
                        w13_u8[e], w13_sc[e], w13_s2[e]
                    ).view(2 * N, K)
                    d2 = self._dequant(w2_u8[e], w2_sc[e], w2_s2[e].float()).view(K, N)
                    h = (
                        torch.nn.functional.silu(x[m].float() @ d13[:N].T)
                        * (x[m].float() @ d13[N:].T)
                    ).clamp(-10.0, 10.0)
                    ref[m] += float(topk_w[m, j]) * (h @ d2.T)
            diff = (out.float() - ref.to(torch.float16).float()).abs().max().item()
            self.assertLess(
                diff, 2e-2, f"fused_marlin_moe diverges from dequant ref at M={M}"
            )


if __name__ == "__main__":
    unittest.main()
