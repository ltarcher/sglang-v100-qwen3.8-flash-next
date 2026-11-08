"""L2-a spike: SM70 grouped sparse prefill feasibility microbench (Qwen3.8 shape).

Questions (PREFILL_LEVERS_PLAN.md §1.3 L2-a):
  (a) with a pre-sorted DENSE page table (offline gather, zero per-tile
      indirection), can a TileLang kernel beat 17-19 TFLOP/s on the Qwen QSA
      prefill shape?  (emitter floor vs per-tile gather, D=256)
  (b) smem budget account for the grouped form (Qwen 8x6=48 rows vs GLM 8x16).

Measured verdict (2026-11-08, 1x V100-PCIE-32GB, sglang-v100-unified container,
`docker run --gpus '"device=0"' --entrypoint /opt/venv/bin/python <img>
 /work/bench_l2a.py` with this repo mounted at /sglang-workspace; L2A_ONLY_V1=1
skips the already-measured variants):

  [triton baseline]      35.70 ms    1.44 TFLOP/s   (production path, matches
                                                    the L2-0 34.2 ms/launch)
  [V2 per-token gather]  11.20 ms    4.60 TFLOP/s   (DRAM-bound: 8.6 GB/launch
                                                    vs ~900 GB/s floor 9.5 ms)
  [V2b per-token dense]  13.32 ms    3.87 TFLOP/s   (dense page table does NOT
                                                    lift the per-token form)
  [V1 grouped ov=0.82]    5.80 ms  U=4608  20.0 TFLOP/s own-MMA  = 6.2x vs triton
  [V1 grouped ov=0.5 ]    9.52 ms  U=8832  23.4 TFLOP/s own-MMA  (efficiency
                                                    rises with U)
  V1 correctness vs fp32 reference: max|d| = 2e-5 (fp16 rounding).

Verdict: the per-token form is dead at D=256 (bandwidth, not the emitter);
grouping is the load-bearing lever. TileLang production line confirmed
(6.2x kernel speedup > the 3-4x the plan needed); 25 TFLOP/s not broken at
ov=0.82 but the limiter is occupancy (smem 62 KB -> 1 CTA/SM, 407 GB/s), not
the emitter -- an L2-b tuning axis, e.g. BI=32 + 2 CTA/SM co-residency.
Group size must keep M = rows*6 a multiple of 16 -> 8 rows (M=48) is minimal.

Variants (all fp16, sm70 WMMA via tilelang, 128 threads = 4 warps):
  V1  grouped: 8 query rows x 6 heads = 48 M-rows per program, per-group
      pre-sorted UNION panels (dense buffers Kb/Vb [G,U,256] + per-row
      visibility mask). Zero per-tile indirection: panel reads are sequential.
  V2  per-token: GLM-derived kernel (M=16 padded from 6 heads), per-tile row
      gather through an Indices tensor into dense K/V pools (page-size-1).
  V2b per-token contiguous: same kernel but rows pre-compacted per query
      (indices are affine: row*topk + i) -- isolates the index-gather cost.
  Triton baseline: production _sparse_gqa_chunk_prefill via
      sparse_gqa_fwd_interface_triton_ck (FMA floor on sm70).

Shape: Q=4096 rows, Hq=6, D=256, kv_heads=1, topk=2048, pool=65536 rows,
sm_scale = 1/sqrt(256).
"""

import importlib.util
import math
import os

import torch

import tilelang
import tilelang.language as T

DTYPE = "float16"

Q_ROWS = 4096
NUM_HEADS = 6
DIM = 256
TOPK = 2048
POOL_N = 65536
SM_SCALE = 1.0 / math.sqrt(DIM)
LOG2E = 1.4426950408889634
SCALE_FOLDED = SM_SCALE * LOG2E

REPO = "/sglang-workspace"


def load_triton_baseline():
    path = os.path.join(
        REPO,
        "python/sglang/srt/layers/attention/qsa/sparse_attn.py",
    )
    spec = importlib.util.spec_from_file_location("qsa_sparse_attn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sparse_gqa_fwd_interface_triton_ck


# ---------------------------------------------------------------- kernels --


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def grouped_sparse_prefill_sm70(
    group_rows: int,
    num_heads: int,
    dim: int,
    union_len: int,
    *,
    sm_scale: float,
    block_I: int = 64,
    threads: int = 128,
):
    """V1: one program per 8-row query group; dense pre-sorted union panels.

    Q [R, H, D] fp16; Kb/Vb [G, U, D] fp16 dense per-group union of the 8
    rows' topk sets (rows sorted by logical order, zero-padded tail);
    Mask [G, R/rows, U] uint8 per-row visibility; Out [R, H, D] fp16.
    U must be a multiple of block_I. Union always covers each row's own
    topk, so no row is ever fully masked (sumexp > 0).
    """
    M = group_rows * num_heads
    D_half = dim // 2
    BI = block_I
    assert union_len % BI == 0
    NI = union_len // BI

    q_rows = T.symbolic("q_rows")
    G = T.symbolic("g")

    @T.prim_func
    def main(
        Q: T.Tensor([q_rows, num_heads, dim], DTYPE),
        Kb: T.Tensor([G, union_len, dim], DTYPE),
        Vb: T.Tensor([G, union_len, dim], DTYPE),
        Mask: T.Tensor([G, group_rows, union_len], "uint8"),
        Out: T.Tensor([q_rows, num_heads, dim], DTYPE),
    ):
        with T.Kernel(G, threads=threads) as bg:
            Q_shared_l = T.alloc_shared([M, D_half], DTYPE)
            Q_shared_r = T.alloc_shared([M, D_half], DTYPE)
            K_shared = T.alloc_shared([BI, D_half], DTYPE)
            V_shared = T.alloc_shared([BI, D_half], DTYPE)
            S_shared = T.alloc_shared([M, BI], DTYPE)
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

            for m, d in T.Parallel(M, D_half):
                q_row = bg * group_rows + m // num_heads
                Q_shared_l[m, d] = Q[q_row, m % num_heads, d]
                Q_shared_r[m, d] = Q[q_row, m % num_heads, D_half + d]

            for i_i in T.serial(NI):
                p0 = i_i * BI
                # per-row visibility read straight from global (cached);
                # a [group_rows, BI] fragment here trips the sm70 layout
                # inference (non-injective thread mapping vs acc_s's m//H).
                for m, bi in T.Parallel(M, BI):
                    acc_s[m, bi] = T.if_then_else(
                        Mask[bg, m // num_heads, p0 + bi] > 0,
                        0,
                        -T.infinity(acc_s.dtype),
                    )
                # dense sequential reads: no index tensor, no per-tile chase
                for bi, d in T.Parallel(BI, D_half):
                    K_shared[bi, d] = Kb[bg, p0 + bi, d]
                T.gemm(
                    Q_shared_l,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi, d in T.Parallel(BI, D_half):
                    K_shared[bi, d] = Kb[bg, p0 + bi, D_half + d]
                T.gemm(
                    Q_shared_r,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                # re-assert: padded panel rows are zero-filled but stay masked
                for m, bi in T.Parallel(M, BI):
                    acc_s[m, bi] = T.if_then_else(
                        Mask[bg, m // num_heads, p0 + bi] > 0,
                        acc_s[m, bi],
                        -T.infinity(acc_s.dtype),
                    )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for m in T.Parallel(M):
                    alpha[m] = T.exp2((m_i_prev[m] - m_i[m]) * sm_scale)
                for m, bi in T.Parallel(M, BI):
                    acc_s[m, bi] = T.exp2(
                        acc_s[m, bi] * sm_scale - m_i[m] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for m in T.Parallel(M):
                    sumexp[m] = sumexp[m] * alpha[m] + sumexp_i[m]
                for m, d in T.Parallel(M, D_half):
                    acc_o_l[m, d] = acc_o_l[m, d] * alpha[m]
                    acc_o_r[m, d] = acc_o_r[m, d] * alpha[m]

                T.copy(acc_s, S_shared)
                for bi, d in T.Parallel(BI, D_half):
                    V_shared[bi, d] = Vb[bg, p0 + bi, d]
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_l,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi, d in T.Parallel(BI, D_half):
                    V_shared[bi, d] = Vb[bg, p0 + bi, D_half + d]
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_r,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            for m, d in T.Parallel(M, D_half):
                q_row = bg * group_rows + m // num_heads
                Out[q_row, m % num_heads, d] = acc_o_l[m, d] / sumexp[m]
                Out[q_row, m % num_heads, D_half + d] = (
                    acc_o_r[m, d] / sumexp[m]
                )

    return main


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def pertoken_sparse_prefill_sm70(
    num_heads: int,
    dim: int,
    topk: int,
    *,
    sm_scale: float,
    block_I: int = 64,
    threads: int = 128,
    contiguous_rows: bool = False,
):
    """V2/V2b: GLM-derived per-token kernel on the Qwen GQA shape.

    Separate K/V pools (Qwen GQA, unlike absorbed MLA where V == K_nope).
    V2 gathers each panel row through Indices [R, topk] (page-size-1 dense
    pools -- the only indirection left after the dense page table);
    V2b (contiguous_rows=True) replaces the index load with the affine
    row*topk + i, i.e. fully pre-compacted per-row buffers.
    padded_H = 16 (6 heads), M=16, one program per query row.
    """
    padded_H = 16
    D_half = dim // 2
    BI = block_I
    assert topk % BI == 0
    NI = topk // BI

    q_rows = T.symbolic("q_rows")
    pool_n = T.symbolic("pool_n")

    @T.prim_func
    def main(
        Q: T.Tensor([q_rows, num_heads, dim], DTYPE),
        Kp: T.Tensor([pool_n, dim], DTYPE),
        Vp: T.Tensor([pool_n, dim], DTYPE),
        Indices: T.Tensor([q_rows, topk], "int32"),
        Out: T.Tensor([q_rows, num_heads, dim], DTYPE),
    ):
        with T.Kernel(q_rows, threads=threads) as bx:
            Q_shared_l = T.alloc_shared([padded_H, D_half], DTYPE)
            Q_shared_r = T.alloc_shared([padded_H, D_half], DTYPE)
            K_shared = T.alloc_shared([BI, D_half], DTYPE)
            V_shared = T.alloc_shared([BI, D_half], DTYPE)
            mask = T.alloc_fragment([BI], "bool")
            acc_s = T.alloc_fragment([padded_H, BI], "float")
            acc_o_l = T.alloc_fragment([padded_H, D_half], "float")
            acc_o_r = T.alloc_fragment([padded_H, D_half], "float")
            S_shared = T.alloc_shared([padded_H, BI], DTYPE)
            sumexp = T.alloc_fragment([padded_H], "float")
            sumexp_i = T.alloc_fragment([padded_H], "float")
            alpha = T.alloc_fragment([padded_H], "float")
            m_i = T.alloc_fragment([padded_H], "float")
            m_i_prev = T.alloc_fragment([padded_H], "float")

            T.fill(acc_o_l, 0)
            T.fill(acc_o_r, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            s_i = bx
            T.fill(Q_shared_l, 0)
            T.fill(Q_shared_r, 0)
            for h, d in T.Parallel(num_heads, D_half):
                Q_shared_l[h, d] = Q[s_i, h, d]
                Q_shared_r[h, d] = Q[s_i, h, D_half + d]

            for i_i in T.serial(NI):
                for bi in T.Parallel(BI):
                    if contiguous_rows:
                        mask[bi] = True
                    else:
                        mask[bi] = Indices[s_i, i_i * BI + bi] >= 0
                for h, bi in T.Parallel(padded_H, BI):
                    acc_s[h, bi] = T.if_then_else(
                        mask[bi], 0, -T.infinity(acc_s.dtype)
                    )

                for bi, d in T.Parallel(BI, D_half):
                    if contiguous_rows:
                        row = s_i * topk + i_i * BI + bi
                    else:
                        row = T.max(Indices[s_i, i_i * BI + bi], 0)
                    K_shared[bi, d] = Kp[row, d]
                T.gemm(
                    Q_shared_l,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi, d in T.Parallel(BI, D_half):
                    if contiguous_rows:
                        row = s_i * topk + i_i * BI + bi
                    else:
                        row = T.max(Indices[s_i, i_i * BI + bi], 0)
                    K_shared[bi, d] = Kp[row, D_half + d]
                T.gemm(
                    Q_shared_r,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for h, bi in T.Parallel(padded_H, BI):
                    acc_s[h, bi] = T.if_then_else(
                        mask[bi], acc_s[h, bi], -T.infinity(acc_s.dtype)
                    )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h in T.Parallel(padded_H):
                    alpha[h] = T.exp2((m_i_prev[h] - m_i[h]) * sm_scale)
                for h, bi in T.Parallel(padded_H, BI):
                    acc_s[h, bi] = T.exp2(
                        acc_s[h, bi] * sm_scale - m_i[h] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h in T.Parallel(padded_H):
                    sumexp[h] = sumexp[h] * alpha[h] + sumexp_i[h]
                for h, d in T.Parallel(padded_H, D_half):
                    acc_o_l[h, d] = acc_o_l[h, d] * alpha[h]
                    acc_o_r[h, d] = acc_o_r[h, d] * alpha[h]

                T.copy(acc_s, S_shared)
                for bi, d in T.Parallel(BI, D_half):
                    if contiguous_rows:
                        row = s_i * topk + i_i * BI + bi
                    else:
                        row = T.max(Indices[s_i, i_i * BI + bi], 0)
                    V_shared[bi, d] = T.if_then_else(
                        mask[bi], Vp[row, d], 0
                    )
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_l,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for bi, d in T.Parallel(BI, D_half):
                    if contiguous_rows:
                        row = s_i * topk + i_i * BI + bi
                    else:
                        row = T.max(Indices[s_i, i_i * BI + bi], 0)
                    V_shared[bi, d] = T.if_then_else(
                        mask[bi], Vp[row, D_half + d], 0
                    )
                T.gemm(
                    S_shared,
                    V_shared,
                    acc_o_r,
                    clear_accum=False,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            # rows 6..15 of the padded M carry zero Q -> zero scores; softmax
            # over equal scores gives uniform weights and a finite output that
            # is simply never read back (same convention as the GLM kernel).
            for h, d in T.Parallel(num_heads, D_half):
                Out[s_i, h, d] = acc_o_l[h, d] / sumexp[h]
                Out[s_i, h, D_half + d] = acc_o_r[h, d] / sumexp[h]

    return main


# ------------------------------------------------------------------ data --


def gen_row_indices(q_rows, topk, pool_n, overlap, device):
    """Per-row topk positions with controllable adjacent-row overlap.

    Row 0 random; row i keeps `overlap` fraction of row i-1's set and fills
    the rest with fresh random rows (deduped). Sorted ascending per row.
    Returns int32 [q_rows, topk].
    """
    idx = torch.empty(q_rows, topk, dtype=torch.long, device=device)
    cur = torch.randperm(pool_n, device=device)[:topk]
    idx[0] = cur
    n_keep = int(topk * overlap)
    for i in range(1, q_rows):
        keep = cur[:n_keep]
        # over-sample then dedupe against keep
        cand = torch.randperm(pool_n, device=device)[: (topk - n_keep) * 2]
        seen = torch.zeros(pool_n, dtype=torch.bool, device=device)
        seen[keep] = True
        cand = cand[~seen[cand]][: topk - n_keep]
        cur = torch.cat([keep, cand])
        cur = cur.sort().values
        idx[i] = cur
    return idx.to(torch.int32)


def build_grouped_buffers(idx, group_rows, k_pool, v_pool):
    """Per-group sorted union panels + per-row visibility masks.

    Returns Kb, Vb [G, U_pad, D] fp16 (zero-padded tail), Mask [G, 8, U_pad]
    uint8, union raw length.
    """
    G = idx.shape[0] // group_rows
    gidx = idx.long().view(G, group_rows, -1)
    unions = [torch.unique(gidx[g]) for g in range(G)]
    u_raw = max(u.numel() for u in unions)
    u_pad = ((u_raw + 63) // 64) * 64
    dim = k_pool.shape[1]
    dev = k_pool.device
    kb = torch.zeros(G, u_pad, dim, dtype=DTYPE_TO_TORCH, device=dev)
    vb = torch.zeros_like(kb)
    mask = torch.zeros(G, group_rows, u_pad, dtype=torch.uint8, device=dev)
    for g in range(G):
        u = unions[g]
        n = u.numel()
        kb[g, :n] = k_pool[u]
        vb[g, :n] = v_pool[u]
        for r in range(group_rows):
            pos = torch.searchsorted(u, gidx[g, r])
            mask[g, r, pos] = 1
    return kb, vb, mask, u_raw, u_pad


DTYPE_TO_TORCH = torch.float16


def ref_attention(q, k_pool, v_pool, idx, scale):
    """fp32 reference over selected rows. q [R,H,D], idx [R,topk] long."""
    rows = idx.long().clamp(min=0)
    valid = idx >= 0
    k = k_pool[rows].float()
    v = v_pool[rows].float()
    s = torch.einsum("rhd,rkd->rhk", q.float(), k) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    p = s.softmax(-1)
    return torch.einsum("rhk,rkd->rhd", p, v)


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]  # median ms


def main():
    dev = "cuda:0"
    torch.manual_seed(7)
    only_v1 = os.environ.get("L2A_ONLY_V1", "0") == "1"
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"shape: Q={Q_ROWS} H={NUM_HEADS} D={DIM} topk={TOPK} "
        f"pool={POOL_N} scale={SM_SCALE}",
        flush=True,
    )

    k_pool = torch.randn(POOL_N, DIM, dtype=torch.float16, device=dev) * 0.5
    v_pool = torch.randn(POOL_N, DIM, dtype=torch.float16, device=dev) * 0.5
    q = (torch.randn(Q_ROWS, NUM_HEADS, DIM, dtype=torch.float16, device=dev) * 0.5)

    per_row_flops = Q_ROWS * NUM_HEADS * TOPK * DIM * 4  # QK + PV, 2 FLOP/MAC

    results = {}

    # ---------------- Triton production baseline ----------------
    if not only_v1:
        triton_fn = load_triton_baseline()
        idx = gen_row_indices(Q_ROWS, TOPK, POOL_N, overlap=0.82, device=dev)
        cu_q = torch.tensor([0, Q_ROWS], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, POOL_N], dtype=torch.int32, device=dev)
        kv_lens = torch.tensor([POOL_N], dtype=torch.int32, device=dev)
        idx3 = idx.view(Q_ROWS, 1, TOPK).contiguous()
        k4 = k_pool.view(POOL_N, 1, DIM).contiguous()
        v4 = v_pool.view(POOL_N, 1, DIM).contiguous()
        ms = bench(
            lambda: triton_fn(q, k4, v4, idx3, cu_q, cu_k, kv_lens, SM_SCALE)
        )
        results["triton_production"] = ms
        print(
            f"[triton baseline]      {ms:8.2f} ms  "
            f"{per_row_flops / ms * 1e-9:6.2f} TFLOP/s (per-row FLOPs)",
            flush=True,
        )

        # ---------------- V2: per-token, index gather ----------------
        kern2 = pertoken_sparse_prefill_sm70(
            NUM_HEADS, DIM, TOPK, sm_scale=SCALE_FOLDED, contiguous_rows=False
        )
        out2 = kern2(q, k_pool, v_pool, idx)
        ms = bench(lambda: kern2(q, k_pool, v_pool, idx))
        results["v2_pertoken_gather"] = ms
        print(
            f"[V2 per-token gather]  {ms:8.2f} ms  "
            f"{per_row_flops / ms * 1e-9:6.2f} TFLOP/s (per-row FLOPs)",
            flush=True,
        )

        # ---------------- V2b: per-token contiguous ----------------
        # per-row compacted buffers: K/V rows rearranged so row r owns
        # [r*topk, (r+1)*topk) -- the "pre-sorted dense page table" form.
        flat_idx = idx.long().reshape(-1)
        kc = k_pool[flat_idx].contiguous()
        vc = v_pool[flat_idx].contiguous()
        kern2b = pertoken_sparse_prefill_sm70(
            NUM_HEADS, DIM, TOPK, sm_scale=SCALE_FOLDED, contiguous_rows=True
        )
        out2b = kern2b(q, kc, vc, idx)
        ms = bench(lambda: kern2b(q, kc, vc, idx))
        results["v2b_pertoken_dense"] = ms
        print(
            f"[V2b per-token dense]  {ms:8.2f} ms  "
            f"{per_row_flops / ms * 1e-9:6.2f} TFLOP/s (per-row FLOPs)",
            flush=True,
        )
        del kc, vc
        torch.cuda.empty_cache()

    # ---------------- V1: grouped, dense union panels ----------------
    for overlap in (0.82, 0.5):
        idx_g = gen_row_indices(Q_ROWS, TOPK, POOL_N, overlap=overlap, device=dev)
        kb, vb, mask, u_raw, u_pad = build_grouped_buffers(
            idx_g, 8, k_pool, v_pool
        )
        kern1 = grouped_sparse_prefill_sm70(
            8, NUM_HEADS, DIM, u_pad, sm_scale=SCALE_FOLDED
        )
        out1 = kern1(q, kb, vb, mask)
        ms = bench(lambda: kern1(q, kb, vb, mask))
        grouped_flops = (Q_ROWS // 8) * 8 * NUM_HEADS * u_pad * DIM * 4
        results[f"v1_grouped_ov{overlap}"] = ms
        print(
            f"[V1 grouped ov={overlap}] {ms:7.2f} ms  "
            f"U_raw={u_raw} U_pad={u_pad}  "
            f"{grouped_flops / ms * 1e-9:6.2f} TFLOP/s (own MMA FLOPs)  "
            f"-> per-row-equiv {per_row_flops / ms * 1e-9:6.2f} T",
            flush=True,
        )
        if overlap == 0.82:
            out1_full = out1
            idx_ref = idx_g
        del kb, vb, mask
        torch.cuda.empty_cache()

    # ---------------- correctness (rows 0..255, fp32 ref) ----------------
    print("\ncorrectness vs fp32 reference (rows 0..255):", flush=True)
    n_check = 256
    ref = ref_attention(
        q[:n_check], k_pool, v_pool, idx_ref[:n_check].long(), SM_SCALE
    )
    # V1 out for the ov=0.82 config
    d1 = (out1_full[:n_check].float() - ref).abs().max().item()
    print(f"  V1 vs ref : max|d| = {d1:.5f}", flush=True)
    if not only_v1:
        # V2 uses the same distribution but a different idx draw; recompute ref
        ref2 = ref_attention(
            q[:n_check], k_pool, v_pool, idx[:n_check].long(), SM_SCALE
        )
        d2 = (out2[:n_check].float() - ref2).abs().max().item()
        d2b = (out2b[:n_check].float() - ref2).abs().max().item()
        # cross-check: V2b vs V2 (same idx, same math)
        d22 = (out2b[:n_check].float() - out2[:n_check].float()).abs().max().item()
        print(f"  V2 vs ref : max|d| = {d2:.5f}", flush=True)
        print(f"  V2b vs ref: max|d| = {d2b:.5f}", flush=True)
        print(f"  V2b vs V2 : max|d| = {d22:.5f}", flush=True)

    print("\nsummary (median ms / production shape):")
    for k, v in results.items():
        print(f"  {k:24s} {v:8.2f} ms")

    if "triton_production" in results:
        base = results["triton_production"]
        print("\nspeedup vs triton baseline: ")
        for k, v in results.items():
            if k.startswith("v1") or k.startswith("v2"):
                print(f"  {k:24s} x{base / v:6.2f}")


if __name__ == "__main__":
    main()
