"""SM70 marlin_v100 uint2b2 (2-bit) MoE kernel-level unit test (P4 debug).

Bypasses the server entirely: RTN-quantize random weights to the uint2b2
format (codes biased by 2, levels {-2,-1,0,+1} x fp16 group scales), repack
them into the SM70 macro-N word layout with the u4b8 nibble interleave, run
fused_marlin_moe(num_bits=2), and compare against a torch dequant reference.

The repack here is the executable specification of the packing contract
documented in sm70_marlin_u2_gemm.cu: one uint32 word holds 16 codes of one
(k, 16-column) row, and within every 8-column run bit-pair p holds the run's
column (p % 4) * 2 + p / 4 (low halfword = columns 0..7, high = 8..15).

Run (inside the dev container, staged .so from MARLIN_V100_INSTALL_DIR):
  U2_MARLIN_SO=/data/models/marlin_v100-u2-artifacts/_sm70_marlin_v100_moe.abi3.so \
  CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python /opt/sglang/scripts/m4_u2_marlin_unit.py
"""

import os

import torch

# Load the staged build explicitly and inject it into the loader cache, so the
# probe exercises the u2 .so rather than whichever production .so the normal
# candidate search would find first.
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


def quantize_uint2(w, group_size):
    """Symmetric RTN onto the fixed uint(2, 2) decode grid.

    The format has no per-group offset: code c decodes to (c - 2) * scale, so
    the grid is {-2, -1, 0, +1} * scale and the scale must be symmetric-style
    (max |w| / 2 covers the group's negative tail; the +1 level tops out at
    half the positive range).

    w: [E, R, C] float32 (R = gemm N, C = gemm K). Scales are per
    (C-group, R) and returned as [E, C // group_size, R] float16 -- the layout
    the SM70 op derives group_size from (num_groups = scales.size(1)).
    """
    e, r, c = w.shape
    wg = w.view(e, r, c // group_size, group_size)
    s = (wg.abs().amax(dim=-1, keepdim=True) / 2.0).clamp_min(1e-12)
    codes = (torch.round(wg / s).clamp_(-2, 1).to(torch.int32) + 2).to(torch.uint8)
    scales = s.squeeze(-1).permute(0, 2, 1).contiguous().to(torch.float16)
    return codes.view(e, r, c), scales


def dequant_uint2(codes, scales, group_size):
    """Reference dequant: w_hat = (code - 2) * scale. Same shapes as quantize."""
    e, r, c = codes.shape
    val = (codes.int() - 2).float()
    val = val.view(e, r, c // group_size, group_size)
    return (val * scales.permute(0, 2, 1).unsqueeze(-1).float()).view(e, r, c)


def repack_u2_sm70(codes_u8, packed_macro_n):
    """codes_u8 [E, R(n), C(k)] values 0..3 -> b_q_weight [E, C/16, R] int32."""
    e, r, c = codes_u8.shape
    assert c % 16 == 0 and r % 64 == 0 and 64 <= packed_macro_n <= 256
    k_group_tiles = packed_macro_n // 64
    dev = codes_u8.device

    # One word = one (k row, 16 n-columns) pair. Within the 16-column word,
    # column j's code lands at bit position pos[j] (8-column run interleave,
    # see module docstring).
    j = torch.arange(16, device=dev)
    pos = (j % 8 >> 1) + ((j & 1) << 2) + (j // 8) * 8
    codes_kn = codes_u8.permute(0, 2, 1).contiguous()  # [E, K, N]
    pos_ordered = codes_kn.reshape(e, c, r // 16, 16)[..., torch.argsort(pos)].int()
    words = (pos_ordered << (2 * j)).sum(-1).to(torch.int32)  # [E, C, R/16]

    # Scatter words into the macro-N interleaved flat order and expose it as
    # [E, K/16, N]: addr(k, n16) = (k/16)*R + group*kG*64 + (k%16*4 + n16%4)*kG
    #               + ((n16/4) % kG).
    kk = torch.arange(c, device=dev)
    m = torch.arange(r // 16, device=dev)
    n_tile = m // 4
    addr = (
        (kk // 16).view(-1, 1) * r
        + ((kk % 16).view(-1, 1) * 4 + (m % 4).view(1, -1)) * k_group_tiles
        + (n_tile // k_group_tiles).view(1, -1) * k_group_tiles * 64
        + (n_tile % k_group_tiles).view(1, -1)
    )  # [C, R/16]
    out = torch.zeros(e, (c // 16) * r, dtype=torch.int32, device=dev)
    flat_idx = addr.reshape(1, -1).expand(e, -1)
    out.scatter_(1, flat_idx, words.reshape(e, -1))
    return out.view(e, c // 16, r)


def run_case(name, e, n, k, m, topk, group_size, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"

    w13 = torch.randn(e, 2 * n, k, device=dev) * 0.02
    w2 = torch.randn(e, k, n, device=dev) * 0.02
    macro13 = 256 if (2 * n) % 256 == 0 else (128 if (2 * n) % 128 == 0 else 64)
    macro2 = 256 if k % 256 == 0 else (128 if k % 128 == 0 else 64)

    c13, s13 = quantize_uint2(w13, group_size)
    c2, s2 = quantize_uint2(w2, group_size)
    w13_m = repack_u2_sm70(c13, macro13)
    w2_m = repack_u2_sm70(c2, macro2)

    x = torch.randn(m, k, device=dev, dtype=torch.float16)
    gate = torch.randn(m, e, device=dev)
    topk_vals, topk_ids = torch.topk(gate, topk, dim=-1)
    topk_w = torch.softmax(topk_vals.float(), dim=-1).to(torch.float16)
    topk_ids = topk_ids.to(torch.int32)

    out = fused_marlin_moe(
        x,
        w13_m,
        w2_m,
        s13,
        s2,
        gate,
        topk_w,
        topk_ids,
        num_bits=2,
        activation="silu",
        is_gated=True,
    )

    d13 = dequant_uint2(c13, s13, group_size)
    d2 = dequant_uint2(c2, s2, group_size)
    ref = torch.zeros(m, k, device=dev, dtype=torch.float32)
    for mi in range(m):
        for jx in range(topk):
            ex = int(topk_ids[mi, jx])
            wtx = float(topk_w[mi, jx])
            gate_h = x[mi].float() @ d13[ex, :n].T
            up_h = x[mi].float() @ d13[ex, n:].T
            h = torch.nn.functional.silu(gate_h) * up_h
            ref[mi] += wtx * (h @ d2[ex].T)

    nan = bool(out.isnan().any())
    diff = (out.float() - ref).abs()
    scale = ref.abs().max().item()
    rel = (diff.max() / max(scale, 1e-9)).item()
    ok = (not nan) and diff.max().item() <= 0.02 * max(scale, 1e-9)
    print(
        f"[{name}] macro13={macro13} macro2={macro2} g={group_size} "
        f"max_abs={diff.max().item():.3e} ref_max={scale:.3f} rel={rel:.2e} "
        f"nan={nan} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    assert torch.cuda.get_device_capability()[0] == 7, "SM70 only"
    ok = True
    # Both contraction dims (hidden for w13, intermediate for w2) must be
    # multiples of the group size. Under g128 w13's N=2*inter always lands on
    # packed_macro_n 256; macro 128 and 64 need g64 (inter = 64 mod 128).
    ok &= run_case("g64 macro128/256", 4, 192, 256, 4, 2, 64)
    ok &= run_case("g64 macro128/64", 4, 192, 192, 4, 2, 64)
    ok &= run_case("g128 macro256/256", 4, 128, 256, 4, 2, 128)
    # Different M / topk exercises a second moe-block-size and routing shape.
    ok &= run_case("g128 macro256/256 M1 topk8", 16, 256, 256, 1, 8, 128)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
