"""SM70 grouped sparse prefill over the Qwen QSA fp16 token pool (L2-b).

Design record: docs/v100/PREFILL_LEVERS_PLAN.md §1.3 L2-b (L2-a spike verdict:
grouped TileLang 20-23 TFLOP/s vs the 1.4 TFLOP/s sm70 Triton FMA floor).

Per 8 query rows (x 6 heads = 48 WMMA M-rows) the planner builds a pre-sorted
UNION of the group's top-k entries plus a per-row visibility mask; the kernel
walks the union in 64-column tiles and reads K/V straight from the fp16 token
pool through the union index. The L2-b A/B measured the indirection as free
against a materialized dense panel (5.88 vs 6.09 ms, bit-identical output), so
no panel gather exists -- the pool read happens exactly once per element.

Planner contract:
  - input topk_indices are valid-first with -1 padding, entries beyond
    ``min(topk, visible)`` are dropped (same semantics as the production
    Triton chunk kernel's ``row_limit`` clamp, minus its block-size rounding);
  - the union is sorted by logical position, so the panel order is a pure
    function of the selection (never the pool's physical allocation order);
  - every real row keeps all of its own entries in the union, so no live row
    is ever fully masked and sumexp > 0;
  - one scalar readback (max union length -> padded panel width) -- eager
    prefill only; the breakable-graph path keeps the Triton fallback.
"""

from typing import Dict, Optional, Tuple

import torch
import tilelang
import tilelang.language as T

# NOTE: no `from __future__ import annotations` here -- TileLang's eager
# builder evals T.prim_func type hints at module scope and stringified
# annotations cannot see the closure constants.

GROUP_ROWS = 8
NUM_HEADS = 6
HEAD_DIM = 256
BLOCK_I = 64
THREADS = 128
_LOG2E = 1.4426950408889634

# Panel widths are bucketed so each width compiles once; a 512 step keeps the
# masked-tile waste under ~11% at the measured overlap (U~4.6k) while bounding
# the number of JIT variants a long-context workload can hit.
U_PAD_STEP = 512

# Planner key layout: (group_slot << LOGICAL_BITS) | logical, where
# group_slot = group * bs + batch_slot. LOGICAL_BITS = 20 admits the 262k
# context; int32 keys fit while group_slot < 2**11, int64 always fits.
LOGICAL_BITS = 20
_LOGICAL_SENTINEL = (1 << LOGICAL_BITS) - 1

# Hard input cap: planner temporaries are ~30 bytes/entry.
_MAX_PLAN_ENTRIES = 1 << 25

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

_kernels: Dict[Tuple[int, float], object] = {}


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _grouped_qsa_sparse_prefill_sm70(
    union_len: int,
    *,
    sm_scale_log2e: float,
    group_rows: int = GROUP_ROWS,
    num_heads: int = NUM_HEADS,
    dim: int = HEAD_DIM,
    block_I: int = BLOCK_I,
    threads: int = THREADS,
):
    """One program per query-row group; dense pre-sorted union panel read
    through Uidx from the token pool, per-row visibility in Mask.

    Q/Out [R, H, D] fp16; KPool/VPool [pool, D] fp16 (flat token pool rows);
    Uidx [G, union_len] int32 (pool row per union column, 0 in pad columns);
    Mask [G, group_rows, union_len] uint8. Rows past R duplicate the group's
    last live row (their output columns are never stored).
    """
    M = group_rows * num_heads
    D_half = dim // 2
    BI = block_I
    assert union_len % BI == 0
    NI = union_len // BI

    R = T.symbolic("q_rows")
    G = T.symbolic("g")
    P = T.symbolic("pool_rows")

    @T.prim_func
    def main(
        Q: T.Tensor([R, num_heads, dim], "float16"),
        KPool: T.Tensor([P, dim], "float16"),
        VPool: T.Tensor([P, dim], "float16"),
        Uidx: T.Tensor([G, union_len], "int32"),
        Mask: T.Tensor([G, group_rows, union_len], "uint8"),
        Out: T.Tensor([R, num_heads, dim], "float16"),
    ):
        with T.Kernel(G, threads=threads) as bg:
            Q_shared_l = T.alloc_shared([M, D_half], "float16")
            Q_shared_r = T.alloc_shared([M, D_half], "float16")
            K_shared = T.alloc_shared([BI, D_half], "float16")
            V_shared = T.alloc_shared([BI, D_half], "float16")
            S_shared = T.alloc_shared([M, BI], "float16")
            acc_s = T.alloc_fragment([M, BI], "float")
            acc_o_l = T.alloc_fragment([M, D_half], "float")
            acc_o_r = T.alloc_fragment([M, D_half], "float")
            sumexp = T.alloc_fragment([M], "float")
            sumexp_i = T.alloc_fragment([M], "float")
            alpha = T.alloc_fragment([M], "float")
            m_i = T.alloc_fragment([M], "float")
            m_i_prev = T.alloc_fragment([M], "float")

            T.fill(acc_o_l, 0)
            T.fill(acc_o_r, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            real_minus_1 = T.min(group_rows, R - bg * group_rows) - 1
            for m, d in T.Parallel(M, D_half):
                q_row = T.min(bg * group_rows + m // num_heads, R - 1)
                Q_shared_l[m, d] = Q[q_row, m % num_heads, d]
                Q_shared_r[m, d] = Q[q_row, m % num_heads, D_half + d]

            for i_i in T.serial(NI):
                p0 = i_i * BI
                for m, bi in T.Parallel(M, BI):
                    m_src = T.min(m // num_heads, real_minus_1)
                    acc_s[m, bi] = T.if_then_else(
                        Mask[bg, m_src, p0 + bi] > 0, 0, -T.infinity(acc_s.dtype)
                    )
                for bi, d in T.Parallel(BI, D_half):
                    K_shared[bi, d] = KPool[Uidx[bg, p0 + bi], d]
                T.gemm(
                    Q_shared_l,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi, d in T.Parallel(BI, D_half):
                    K_shared[bi, d] = KPool[Uidx[bg, p0 + bi], D_half + d]
                T.gemm(
                    Q_shared_r,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                # re-assert: pad columns read pool row 0 but stay masked
                for m, bi in T.Parallel(M, BI):
                    m_src = T.min(m // num_heads, real_minus_1)
                    acc_s[m, bi] = T.if_then_else(
                        Mask[bg, m_src, p0 + bi] > 0,
                        acc_s[m, bi],
                        -T.infinity(acc_s.dtype),
                    )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for m in T.Parallel(M):
                    alpha[m] = T.exp2((m_i_prev[m] - m_i[m]) * sm_scale_log2e)
                for m, bi in T.Parallel(M, BI):
                    acc_s[m, bi] = T.exp2(
                        acc_s[m, bi] * sm_scale_log2e - m_i[m] * sm_scale_log2e
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for m in T.Parallel(M):
                    sumexp[m] = sumexp[m] * alpha[m] + sumexp_i[m]
                for m, d in T.Parallel(M, D_half):
                    acc_o_l[m, d] = acc_o_l[m, d] * alpha[m]
                    acc_o_r[m, d] = acc_o_r[m, d] * alpha[m]

                T.copy(acc_s, S_shared)
                for bi, d in T.Parallel(BI, D_half):
                    V_shared[bi, d] = VPool[Uidx[bg, p0 + bi], d]
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_l,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi, d in T.Parallel(BI, D_half):
                    V_shared[bi, d] = VPool[Uidx[bg, p0 + bi], D_half + d]
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_r,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            for m, d in T.Parallel(M, D_half):
                q_row = bg * group_rows + m // num_heads
                if q_row < R:
                    Out[q_row, m % num_heads, d] = acc_o_l[m, d] / sumexp[m]
                    Out[q_row, m % num_heads, D_half + d] = acc_o_r[m, d] / sumexp[m]

    return main


def _get_kernel(union_len: int, sm_scale: float):
    key = (union_len, sm_scale)
    kern = _kernels.get(key)
    if kern is None:
        kern = _grouped_qsa_sparse_prefill_sm70(
            union_len, sm_scale_log2e=sm_scale * _LOG2E
        )
        _kernels[key] = kern
    return kern


def build_grouped_union_plan(
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    max_context_len: int,
    *,
    group_rows: int = GROUP_ROWS,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Planner: per-group sorted union indices + per-row visibility mask.

    Returns (Uidx [G, U_pad] int32, Mask [G, group_rows, U_pad] uint8,
    U_pad) or None when the input is out of the planner's envelope. One
    scalar readback picks U_pad; everything else stays on device.
    """
    R, topk = topk_indices.shape
    device = topk_indices.device
    bs = int(cu_seqlens_q.numel()) - 1
    if R <= 0 or topk <= 0 or bs <= 0:
        return None
    if R * topk > _MAX_PLAN_ENTRIES:
        return None

    rows = torch.arange(R, device=device, dtype=torch.int64)
    cu64 = cu_seqlens_q.to(torch.int64)
    batch = torch.searchsorted(cu64[1:], rows, right=True)
    rows_in_req = rows - cu64[batch]
    extend_len = (cu64[1:] - cu64[:-1])[batch]
    seq64 = seq_lens.to(torch.int64)
    visible = seq64[batch] - extend_len + rows_in_req + 1
    row_keep = torch.clamp(visible, max=topk)

    g = rows >> 3 if group_rows == 8 else rows // group_rows
    group_slot = g * bs + batch
    # int32 keys fit while the largest group slot stays under 2**11 (the
    # shifted slot then tops out just below INT32_MAX); otherwise int64.
    int32_keys = (
        int(group_slot[-1]) + 1 < (1 << (31 - LOGICAL_BITS))
        and max_context_len < _LOGICAL_SENTINEL
    )

    cols = torch.arange(topk, device=device, dtype=torch.int32)
    entry_valid = (cols[None, :] < row_keep.to(torch.int32)[:, None]) & (
        topk_indices >= 0
    )
    if int32_keys:
        key = (group_slot.to(torch.int32) << LOGICAL_BITS).unsqueeze(1) | torch.where(
            entry_valid, topk_indices, _LOGICAL_SENTINEL
        )
    else:
        logical64 = torch.where(
            entry_valid, topk_indices.to(torch.int64), _LOGICAL_SENTINEL
        )
        key = (group_slot << LOGICAL_BITS).unsqueeze(1) | logical64
    vals, sidx = torch.sort(key.view(-1))
    del key

    if int32_keys:
        s_logical = vals & _LOGICAL_SENTINEL
        slot_of_sorted = (vals >> LOGICAL_BITS).to(torch.int64)
    else:
        s_logical = vals & _LOGICAL_SENTINEL
        slot_of_sorted = vals >> LOGICAL_BITS
    is_valid = s_logical != _LOGICAL_SENTINEL
    prev = torch.cat([vals[:1] - 1, vals[:-1]])
    vd = (vals != prev) & is_valid
    vd_v = vd.to(vals.dtype)  # int32 in the int32-key path; kept for scatter
    cum = torch.cumsum(vd_v, 0)
    g_sorted = torch.div(slot_of_sorted, bs, rounding_mode="floor")
    first = torch.cat([g_sorted[:1] - 1, g_sorted[:-1]]) != g_sorted
    # every group owns at least one entry (rows are real queries), so the
    # compact id equals the group id and G is known on host
    G = (R + group_rows - 1) // group_rows
    base = torch.zeros(G, device=device, dtype=cum.dtype)
    base.index_put_((g_sorted[first],), cum[first] - vd_v[first])
    # duplicates share their key's distinct rank: cum is flat across them
    u_local = torch.where(is_valid, cum - 1 - base[g_sorted], torch.zeros_like(cum))
    del prev, cum, slot_of_sorted

    max_u = int(u_local.max().item())  # the planner's single readback
    u_pad = max(BLOCK_I, ((max_u + 1 + U_PAD_STEP - 1) // U_PAD_STEP) * U_PAD_STEP)
    # the union never exceeds the group's entry count; clamp the bucket so
    # tiny-topk shapes do not waste (or overflow-guard against) panel width
    u_pad = min(
        u_pad, max(BLOCK_I, ((group_rows * topk + BLOCK_I - 1) // BLOCK_I) * BLOCK_I)
    )
    if max_u + 1 > u_pad:
        # unreachable while the union is a subset of the group's entries;
        # guards against a torn key layout silently corrupting the panel
        return None

    rows_sorted = torch.div(sidx, topk, rounding_mode="floor")
    m_of = rows_sorted % group_rows
    batch_of_entry = batch[rows_sorted]
    # the sorted key carries each entry's logical position already (no second
    # gather into topk_indices); invalid entries clamp to column 0 because
    # their slot never lands in uidx
    logical_entry = torch.where(is_valid, s_logical, torch.zeros_like(s_logical))
    req_rows = req_to_token.index_select(0, req_pool_indices.to(torch.long))
    slot = req_rows[batch_of_entry, logical_entry].to(torch.int32)

    mask_i32 = torch.zeros(G, group_rows, u_pad, device=device, dtype=torch.int32)
    mask_i32.view(-1).index_add_(
        0, (g_sorted * group_rows + m_of) * u_pad + u_local, is_valid.to(torch.int32)
    )
    uidx = torch.zeros(G, u_pad, device=device, dtype=torch.int32)
    # only the distinct entry of each union column may carry the slot value:
    # duplicates share (g, u_local) and index_add_ would sum their slots
    uidx.view(-1).index_add_(
        0, g_sorted * u_pad + u_local, torch.where(vd, slot, torch.zeros_like(slot))
    )
    return uidx, mask_i32.to(torch.uint8), u_pad


def run_grouped_sparse_prefill(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    plan: Tuple[torch.Tensor, torch.Tensor, int],
    sm_scale: float,
) -> torch.Tensor:
    """Run the grouped kernel; q [R, 6, 256] fp16 -> out [R, 6, 256] fp16."""
    uidx, mask, u_pad = plan
    kern = _get_kernel(u_pad, sm_scale)
    pool_rows = k_buffer.shape[0]
    k_pool = k_buffer.view(pool_rows, HEAD_DIM)
    v_pool = v_buffer.view(pool_rows, HEAD_DIM)
    return kern(q, k_pool, v_pool, uidx, mask)
