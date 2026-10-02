"""SM70 marlin_v100 NVFP4 MoE kernel-level unit test (M2 debug).

Bypasses the server entirely: quantize random weights exactly like the mini
generator (E2M1 codes + E4M3 block scales + fp32 weight_scale_2), run them
through the same SM70 processing as ModelOptNvFp4FusedMoEMethod
(gptq_marlin_moe_repack + sm70_nvfp4_marlin_process_scales/_global_scale), call
fused_marlin_moe, and compare against a torch dequant reference.

Run: CUDA_VISIBLE_DEVICES=1 /opt/venv/bin/python /opt/sglang/scripts/m2_marlin_unit.py
"""

import torch

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.quantization.marlin_utils import (
    sm70_nvfp4_marlin_process_global_scale,
    sm70_nvfp4_marlin_process_scales,
)

E, N, K, M, TOPK = 4, 128, 512, 3, 2  # N = moe_intermediate, K = hidden
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
EDGES = ((GRID[1:] + GRID[:-1]) / 2)


def quantize(w):
    """modelopt NVFP4: w = code * scale_e4m3 * weight_scale_2."""
    e, r, c = w.shape
    wf = w.float()
    s2 = (wf.abs().amax(dim=(1, 2)) / (448.0 * 6.0)).clamp(min=1e-30)  # [E]
    blocks = (wf / s2.view(e, 1, 1)).reshape(e, r, c // 16, 16)
    block_max = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scales = (block_max / 6.0).to(torch.float8_e4m3fn)
    q = blocks / scales.float()
    sign = (q < 0).to(torch.uint8) << 3
    idx = torch.bucketize(q.abs().clamp_max(6.0), EDGES)
    codes = (torch.arange(8, device=w.device, dtype=torch.uint8)[idx] | sign).view(
        e, r, c // 16, 16
    )
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(e, r, c // 2)
    return packed, scales.squeeze(-1), s2


def dequant(packed, scales, s2):
    grid = torch.cat([GRID, -GRID])
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    dec = torch.stack([grid[lo], grid[hi]], dim=-1).reshape(packed.shape[0], -1)
    sf = scales.view(torch.float8_e4m3fn).float()
    sf = torch.stack([sf] * 16, dim=-1).reshape(scales.shape[0], -1)
    return dec * sf * s2


def repack(weight_u8):
    """Same as ModelOptNvFp4FusedMoEMethod._repack_nvfp4_weight."""
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )

    num_experts, size_n, packed_k = weight_u8.shape
    size_k = packed_k * 2
    # little-endian view: each int32 word holds 4 packed bytes = 8 adjacent-K
    # nibbles, nibble k at bit 4k -- identical to the production path.
    gptq_layout = (
        weight_u8.contiguous().view(torch.int32).transpose(1, 2).contiguous()
    )  # [E, K/8, N] int32
    empty_perm = torch.empty((num_experts, 0), dtype=torch.int32, device="cuda")
    return gptq_marlin_moe_repack(gptq_layout, empty_perm, size_k, size_n, 4)


def process_scales(scales_e4m3):
    """[E, N, K/16] checkpoint order -> marlin [E, K/16, N] encoded."""
    marlin, factor = sm70_nvfp4_marlin_process_scales(
        scales_e4m3.transpose(1, 2).contiguous(), torch.float16
    )
    return marlin, factor


def main():
    torch.manual_seed(0)
    dev = "cuda"

    # --- checkpoint-style tensors ---
    w13 = torch.randn(E, 2 * N, K, device=dev) * 0.02
    w2 = torch.randn(E, K, N, device=dev) * 0.02
    w13_u8, w13_sc, w13_s2 = quantize(w13)
    w2_u8, w2_sc, w2_s2 = quantize(w2)

    # --- marlin processing (mirror of the SM70 branch) ---
    w13_m = repack(w13_u8)
    w2_m = repack(w2_u8)
    print("repacked w13:", tuple(w13_m.shape), w13_m.dtype)
    w13_sc_m, f13 = process_scales(w13_sc)
    w2_sc_m, f2 = process_scales(w2_sc)
    print("scales:", tuple(w13_sc_m.shape), "factor", f13, f2)
    g13 = sm70_nvfp4_marlin_process_global_scale(w13_s2, torch.float16) / f13
    g2 = sm70_nvfp4_marlin_process_global_scale(w2_s2, torch.float16) / f2
    print("global scale in:", w13_s2.tolist(), "out:", g13.tolist())

    # --- inputs ---
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    gate = torch.randn(M, E, device=dev)
    topk_vals, topk_ids = torch.topk(gate, TOPK, dim=-1)
    topk_w = torch.softmax(topk_vals.float(), dim=-1).to(torch.float16)
    topk_ids = topk_ids.to(torch.int32)

    out = fused_marlin_moe(
        x,
        w13_m,
        w2_m,
        w13_sc_m,
        w2_sc_m,
        gate,
        topk_w,
        topk_ids,
        num_bits=4,
        activation="silu",
        is_gated=True,
        w1_global_scale=g13,
        w2_global_scale=g2,
    )
    print("marlin out:", out.tolist())

    # --- torch reference ---
    ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
    for m in range(M):
        for j in range(TOPK):
            e = int(topk_ids[m, j])
            w = float(topk_w[m, j])
            d13 = dequant(w13_u8[e : e + 1], w13_sc[e : e + 1], w13_s2[e]).view(2 * N, K)
            d2 = dequant(w2_u8[e : e + 1], w2_sc[e : e + 1], w2_s2[e]).view(K, N)
            gate_h = x[m].float() @ d13[:N].T
            up_h = x[m].float() @ d13[N:].T
            h = torch.nn.functional.silu(gate_h) * up_h
            ref[m] += w * (h @ d2.T)
    ref = ref.to(torch.float16)
    print("reference:   ", ref.tolist())
    diff = (out.float() - ref.float()).abs()
    print("max abs diff:", diff.max().item(), "rel:", (diff.max() / ref.abs().max()).item())
    print("nan in marlin out:", bool(out.isnan().any()))


if __name__ == "__main__":
    main()
