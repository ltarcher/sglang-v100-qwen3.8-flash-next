"""End-to-end probe of the P4 NVFP4 -> u2b2 conversion path.

Synthesizes checkpoint-shaped NVFP4 expert tensors (block-16 e4m3 scales +
per-half/per-tensor weight_scale_2), runs them through
sm70_u2_pool.convert_moe_layer_to_u2 via a stub FusedMoE layer, then checks
fused_marlin_moe(num_bits=2) against a torch reference built from the same
requantized codes. Also reports the 2-bit error vs the NVFP4 dequant
reference (informational; the quality gate decides go/no-go).

Run (inside the dev container):
  U2_MARLIN_SO=/data/models/marlin_v100-u2-artifacts/_sm70_marlin_v100_moe.abi3.so \
  CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python /opt/sglang/scripts/m4_u2_convert_probe.py
"""

import os

import torch

_SO = os.environ.get(
    "U2_MARLIN_SO",
    "/data/models/marlin_v100-u2-artifacts/_sm70_marlin_v100_moe.abi3.so",
)
torch.ops.load_library(_SO)
import sglang.kernels.ops.moe.moe_wna16_marlin as _mm  # noqa: E402

_mm._marlin_v100_op = torch.ops._moe_C.moe_wna16_marlin_gemm

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (  # noqa: E402
    fused_marlin_moe,
)
from sglang.srt.layers.quantization.dequantization import (  # noqa: E402
    dequantize_nvfp4,
)
from sglang.srt.layers.quantization.sm70_u2_pool import (  # noqa: E402
    convert_moe_layer_to_u2,
    requant_u2,
    resolve_u2_group_size,
)


class _StubRunnerConfig:
    def __init__(self, is_gated):
        self.is_gated = is_gated


class _StubLayer(torch.nn.Module):
    """Just enough FusedMoE surface for convert_moe_layer_to_u2."""

    def __init__(self, tensors, is_gated=True):
        super().__init__()
        self.moe_runner_config = _StubRunnerConfig(is_gated)
        for name, t in tensors.items():
            self.register_parameter(
                name, torch.nn.Parameter(t, requires_grad=False)
            )


def quantize_nvfp4(w, s2):
    """w [E, R, K] fp32 + per-expert s2 [E] -> packed u8 codes + e4m3 scales.

    Mirrors the modelopt checkpoint: block-16 amax/6 scales stored as e4m3
    DIVIDED by s2 (dequant folds s2 back in), low nibble = even K index.
    """
    e, r, k = w.shape
    wg = w.view(e, r, k // 16, 16)
    amax = wg.abs().amax(-1, keepdim=True)
    s_block = (amax / 6.0) / s2.view(e, 1, 1, 1)
    w_s = s_block.to(torch.float8_e4m3fn)
    scale_full = w_s.float() * s2.view(e, 1, 1, 1)
    q = (wg / scale_full).clamp(-6, 6)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=w.device)
    idx = torch.argmin((q.abs().unsqueeze(-1) - lut).abs(), dim=-1)
    codes = torch.where(q < 0, idx + 8, idx).to(torch.uint8).view(e, r, k)
    w_q = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return w_q.contiguous(), w_s.view(e, r, k // 16).contiguous()


def dequant_u2(codes, scales, group_size):
    """Reference dequant of requant_u2 output: w_hat = (code - 2) * scale."""
    e, r, c = codes.shape
    val = (codes.int() - 2).float().view(e, r, c // group_size, group_size)
    return (val * scales.permute(0, 2, 1).unsqueeze(-1).float()).view(e, r, c)


def main():
    assert torch.cuda.get_device_capability()[0] == 7, "SM70 only"
    dev = "cuda"
    torch.manual_seed(0)

    e, hidden, inter = 4, 256, 128
    n13 = 2 * inter  # gated
    g = resolve_u2_group_size(hidden, inter)
    assert g == 128, f"expected group 128, got {g}"

    s2_13 = (0.5 + torch.rand(e, 2, device=dev)) * 0.04  # gate/up differ
    s2_2 = (0.5 + torch.rand(e, device=dev)) * 0.04
    w13 = torch.randn(e, n13, hidden, device=dev) * 0.02
    w2 = torch.randn(e, hidden, inter, device=dev) * 0.02

    w13_q, w13_s = quantize_nvfp4_rows(w13, s2_13)
    w2_q, w2_s = quantize_nvfp4(w2, s2_2)

    layer = _StubLayer(
        {
            "w13_weight": w13_q,
            "w2_weight": w2_q,
            "w13_weight_scale": w13_s,
            "w2_weight_scale": w2_s,
            "w13_weight_scale_2": s2_13.float(),
            "w2_weight_scale_2": s2_2.float(),
        }
    )
    convert_moe_layer_to_u2(layer)

    w13_m = layer.w13_weight.data
    w2_m = layer.w2_weight.data
    s13_m = layer.w13_weight_scale.data
    s2_m = layer.w2_weight_scale.data
    assert w13_m.shape == (e, hidden // 16, n13) and w13_m.dtype == torch.int32
    assert w2_m.shape == (e, inter // 16, hidden) and w2_m.dtype == torch.int32
    assert s13_m.shape == (e, hidden // g, n13) and s13_m.dtype == torch.float16
    assert s2_m.shape == (e, inter // g, hidden) and s2_m.dtype == torch.float16

    # Rebuild the reference from the same module pipeline.
    row_s2 = torch.stack(
        [
            s2_13[:, 0].view(-1, 1).expand(-1, inter),
            s2_13[:, 1].view(-1, 1).expand(-1, inter),
        ],
        dim=1,
    ).reshape(-1, 1)
    w13_hat = dequantize_nvfp4(
        w13_q,
        w13_s.to(dev).float() * row_s2.view(e, n13, 1),
        None,
        torch.float32,
    )
    w2_hat = dequantize_nvfp4(
        w2_q, w2_s.to(dev).float() * s2_2.view(-1, 1, 1), None, torch.float32
    )
    c13, s13r = requant_u2(w13_hat, g)
    c2, s2r = requant_u2(w2_hat, g)
    d13 = dequant_u2(c13, s13r, g)
    d2 = dequant_u2(c2, s2r, g)

    m, topk = 4, 2
    x = torch.randn(m, hidden, device=dev, dtype=torch.float16)
    router = torch.randn(m, e, device=dev)
    topk_vals, topk_ids = torch.topk(router, topk, dim=-1)
    topk_w = torch.softmax(topk_vals.float(), dim=-1).to(torch.float16)
    topk_ids = topk_ids.to(torch.int32)

    out = fused_marlin_moe(
        x,
        w13_m,
        w2_m,
        s13_m,
        s2_m,
        router,
        topk_w,
        topk_ids,
        num_bits=2,
        activation="silu",
        is_gated=True,
    )

    ref_u2 = torch.zeros(m, hidden, device=dev, dtype=torch.float32)
    ref_fp4 = torch.zeros_like(ref_u2)
    for mi in range(m):
        for jx in range(topk):
            ex = int(topk_ids[mi, jx])
            wt = float(topk_w[mi, jx])
            gate_h = x[mi].float() @ d13[ex, :inter].T
            up_h = x[mi].float() @ d13[ex, inter:].T
            ref_u2[mi] += wt * ((torch.nn.functional.silu(gate_h) * up_h) @ d2[ex].T)
            gate4 = x[mi].float() @ w13_hat[ex, :inter].T
            up4 = x[mi].float() @ w13_hat[ex, inter:].T
            ref_fp4[mi] += wt * (
                (torch.nn.functional.silu(gate4) * up4) @ w2_hat[ex].T
            )

    nan = bool(out.isnan().any())
    diff = (out.float() - ref_u2).abs()
    scale = ref_u2.abs().max().item()
    rel = (diff.max() / max(scale, 1e-9)).item()
    cost = ((out.float() - ref_fp4).norm() / max(ref_fp4.norm().item(), 1e-9)).item()
    ok = (not nan) and diff.max().item() <= 0.02 * max(scale, 1e-9)
    print(
        f"[convert e2e] g={g} max_abs={diff.max().item():.3e} "
        f"ref_max={scale:.3f} rel={rel:.2e} nan={nan} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    print(
        f"[u2 vs nvfp4 ref] relative frobenius error = {cost:.3f} "
        f"(informational; quality gate decides)"
    )
    raise SystemExit(0 if ok else 1)


def quantize_nvfp4_rows(w, s2_rows):
    """quantize_nvfp4 with a per-row s2: w [E, R, K], s2_rows [E, 2] where
    rows [0, R/2) use s2[:, 0] and rows [R/2, R) use s2[:, 1]."""
    e, r, k = w.shape
    half = r // 2
    per_row = torch.cat(
        [s2_rows[:, 0:1].expand(-1, half), s2_rows[:, 1:2].expand(-1, r - half)],
        dim=1,
    )  # [E, R]
    # Delegating to quantize_nvfp4 needs a per-expert scalar, so inline the
    # row-broadcast version here (identical math, s2 broadcast per row).
    wg = w.view(e, r, k // 16, 16)
    amax = wg.abs().amax(-1, keepdim=True)
    s_block = (amax / 6.0) / per_row.view(e, r, 1, 1)
    w_s = s_block.to(torch.float8_e4m3fn)
    scale_full = w_s.float() * per_row.view(e, r, 1, 1)
    q = (wg / scale_full).clamp(-6, 6)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=w.device)
    idx = torch.argmin((q.abs().unsqueeze(-1) - lut).abs(), dim=-1)
    codes = torch.where(q < 0, idx + 8, idx).to(torch.uint8).view(e, r, k)
    w_q = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return w_q.contiguous(), w_s.view(e, r, k // 16).contiguous()


if __name__ == "__main__":
    main()
