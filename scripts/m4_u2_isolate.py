"""Per-GEMM isolation for the u2 SM70 MoE kernel (P4 debug).

Runs gemm1 and gemm2 through the raw SM70 op separately, with all-zero
weights (codes = 2 -> value 0), so each stage's output is checkable in
isolation: gemm1 must be exactly 0, silu(gate)*up must be 0, gemm2 must
be exactly 0. Any inf/garbage localizes to the failing stage.

Run (inside the dev container):
  U2_MARLIN_SO=/data/models/marlin_v100-u2-artifacts/_sm70_marlin_v100_moe.abi3.so \
  CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python /opt/sglang/scripts/m4_u2_isolate.py
"""

import os
import sys

import torch

_SO = os.environ.get(
    "U2_MARLIN_SO",
    "/data/models/marlin_v100-u2-artifacts/_sm70_marlin_v100_moe.abi3.so",
)
torch.ops.load_library(_SO)
import sglang.kernels.ops.moe.moe_wna16_marlin as _mm  # noqa: E402

_mm._marlin_v100_op = torch.ops._moe_C.moe_wna16_marlin_gemm

from sglang.kernels.ops.moe.moe_wna16_marlin import (  # noqa: E402
    moe_wna16_marlin_gemm,
)
from sglang.srt.layers.moe.fused_moe_triton import (  # noqa: E402
    moe_align_block_size,
)
from sgl_kernel.scalar_type import ScalarType  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m4_u2_marlin_unit import repack_u2_sm70  # noqa: E402

U2 = ScalarType.uint(2, 2)


def main():
    dev = "cuda"
    e, inter, hidden, m, topk, g = 4, 64, 64, 4, 1, 64
    r13, k13 = 2 * inter, hidden
    r2, k2 = hidden, inter
    macro13 = 256 if r13 % 256 == 0 else (128 if r13 % 128 == 0 else 64)
    macro2 = 256 if r2 % 256 == 0 else (128 if r2 % 128 == 0 else 64)
    block_size_m = 8

    # All-zero weights: code 2 decodes to value 0.
    c13 = torch.full((e, r13, k13), 2, dtype=torch.uint8, device=dev)
    c2 = torch.full((e, r2, k2), 2, dtype=torch.uint8, device=dev)
    s13 = torch.full((e, k13 // g, r13), 1.0, dtype=torch.float16, device=dev)
    s2 = torch.full((e, k2 // g, r2), 1.0, dtype=torch.float16, device=dev)
    w13 = repack_u2_sm70(c13, macro13)
    w2 = repack_u2_sm70(c2, macro2)

    x = torch.eye(m, hidden, device=dev, dtype=torch.float16)
    topk_ids = (torch.arange(m, device=dev)[:, None] % e).to(torch.int32)
    topk_w = torch.ones(m, 1, device=dev, dtype=torch.float16)

    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, block_size_m, e
    )
    ws = torch.zeros(1024, dtype=torch.int, device=dev)

    def report(name, t):
        flat = t.float().flatten()
        print(
            f"[{name}] absmax={flat.abs().max().item():.4f} "
            f"nan={bool(t.isnan().any())} inf={bool(t.isinf().any())}"
        )
        return flat

    cache1 = torch.zeros(m * topk, r13, device=dev, dtype=torch.float16)
    moe_wna16_marlin_gemm(
        x,
        cache1,
        w13,
        None,
        s13,
        None,
        None,
        None,
        None,
        ws,
        sorted_ids,
        expert_ids,
        num_post,
        topk_w,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=U2,
        size_m=m,
        size_n=r13,
        size_k=hidden,
        use_atomic_add=True,
        use_fp32_reduce=True,
    )
    torch.cuda.synchronize()
    c1 = report("gemm1 (expect all 0)", cache1)
    print("  row0[:8] =", [round(v, 3) for v in c1[:8].tolist()])

    gate = cache1[:, :inter]
    up = cache1[:, inter:]
    h = torch.nn.functional.silu(gate) * up
    report("h = silu(gate)*up (expect 0)", h)

    cache3 = torch.zeros(m * topk, k2, device=dev, dtype=torch.float16)
    moe_wna16_marlin_gemm(
        h,
        cache3,
        w2,
        None,
        s2,
        None,
        None,
        None,
        None,
        ws,
        sorted_ids,
        expert_ids,
        num_post,
        topk_w,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=False,
        b_q_type=U2,
        size_m=m * topk,
        size_n=k2,
        size_k=inter,
        use_atomic_add=True,
        use_fp32_reduce=True,
    )
    torch.cuda.synchronize()
    c3 = report("gemm2 (expect all 0)", cache3)
    print("  row0[:8] =", [round(v, 3) for v in c3[:8].tolist()])


if __name__ == "__main__":
    main()
