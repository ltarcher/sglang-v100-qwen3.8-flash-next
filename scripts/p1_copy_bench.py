#!/usr/bin/env python3
"""P1 microbench: spill_copy_kernel bandwidth vs claim count and grid padding.

P0/P1 puzzle: the in-engine page-in runs at ~2.3 GB/s per rank even with ~180
active blocks (64 splits x 2.8 claims), far below the 10.19 GB/s pin line
rate, and the per-launch time fits "fixed ~2 ms + marginal ~5 GB/s". This
harness calls spill_page_in_cached directly on synthetic tensors so the
kernel's t(claims, n_landing) surface can be measured without engine noise.

Varying n_landing at fixed claims varies the EMPTY-block count (grid =
n_landing * kCopySplits, only claims * kCopySplits blocks work), which
isolates empty-block dispatch cost from per-byte bandwidth. DtoH
cudaMemcpyAsync runs as the reference number for the link.

Single-GPU by default; --gpu pins the device. For the shared-fabric test run
four instances concurrently (one per GPU) and compare per-instance GB/s vs
solo.
"""

from __future__ import annotations

import argparse
import ctypes
import os

import torch

MIB = 1024 * 1024

# Realistic per-expert row mix (Marlin-packed w13/w2 + scales), 3.38 MiB total.
ROW_MIBS = [2.5, 0.4, 0.2, 0.1, 0.1, 0.08]

cudaHostRegisterMapped = 2


def register_host(t: torch.Tensor) -> None:
    rc = ctypes.CDLL("libcudart.so").cudaHostRegister(
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_size_t(t.numel() * t.element_size()),
        ctypes.c_uint(cudaHostRegisterMapped),
    )
    if rc != 0:
        raise RuntimeError(f"cudaHostRegister failed: {rc}")


def prepare(claims: int, n_landing: int, host_rows: int,
            src_ptrs: list, rows: list) -> dict:
    """Allocate every device tensor up front: cudaMalloc while another
    process is cudaHostRegister-ing trips cudaErrorAlreadyMapped on this
    driver, so setup must finish before concurrent measurement begins."""
    dev = torch.device("cuda")
    n_tensors = len(rows)

    dst = [
        torch.empty((n_landing, r), dtype=torch.uint8, device=dev) for r in rows
    ]
    n_logical = 288
    topk = torch.full((1, claims), n_logical - claims, dtype=torch.int32, device=dev)
    topk[0] = torch.arange(n_logical - claims, n_logical, dtype=torch.int32)
    map_table = torch.full((n_logical,), -1, dtype=torch.int32, device=dev)
    host_map = torch.full((n_logical,), -1, dtype=torch.int32, device=dev)
    for i in range(claims):
        host_map[n_logical - claims + i] = i % host_rows
    slot_host_row = torch.full((n_landing,), -1, dtype=torch.int32, device=dev)
    cache_lut = torch.full((n_logical,), -1, dtype=torch.int32, device=dev)
    cache_slot_key = torch.full((n_landing,), -1, dtype=torch.int32, device=dev)
    cache_epoch = torch.zeros((n_landing,), dtype=torch.int32, device=dev)
    cache_clock = torch.zeros((5,), dtype=torch.int32, device=dev)

    from sglang.kernels.ops.moe.sm70_dsv41_spill_pagein import spill_page_in_cached

    src_dev = torch.tensor(src_ptrs, dtype=torch.int64, device=dev)
    dst_dev = torch.tensor(
        [d.data_ptr() for d in dst], dtype=torch.int64, device=dev
    )
    rows_dev = torch.tensor(rows, dtype=torch.int64, device=dev)
    land_ids = torch.empty((topk.shape[0], claims), dtype=torch.int32, device=dev)

    # Clearing the LUT every call forces the claims path every call, which is
    # the regime the decode engine actually measures (rank-global 48 slots
    # thrash across 42 layers; P1 measured ~0.2% lut hits).
    ids = topk.clone()

    def one_call():
        cache_lut.fill_(-1)
        cache_slot_key.fill_(-1)
        cache_epoch.zero_()
        ids.copy_(topk)
        spill_page_in_cached(
            ids,
            land_ids,
            slot_host_row,
            map_table,
            host_map,
            src_dev,
            dst_dev,
            rows_dev,
            cache_lut,
            cache_slot_key,
            cache_epoch,
            cache_clock,
            0,
        )

    return {"claims": claims, "n_landing": n_landing, "one_call": one_call,
            "claimed_bytes": sum(rows) * claims}


def measure(sess: dict, iters: int) -> dict:
    for _ in range(3):
        sess["one_call"]()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        sess["one_call"]()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    return {
        "claims": sess["claims"],
        "n_landing": sess["n_landing"],
        "grid": sess["n_landing"] * 64,
        "ms": ms,
        "gib_s": sess["claimed_bytes"] / MIB / 1024 / (ms / 1e3),
    }


def memcpy_reference(host_rows: int, iters: int) -> dict:
    r = int(ROW_MIBS[0] * MIB)
    src = torch.empty((host_rows, r), dtype=torch.uint8)
    src.random_()
    register_host(src)
    dst = torch.empty((host_rows, r), dtype=torch.uint8, device="cuda")
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return {"ms": ms, "gib_s": src.numel() / MIB / 1024 / (ms / 1e3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--host-rows", type=int, default=64)
    ap.add_argument("--host-slabs", type=int, default=8,
                    help="pre-registered round-robin slabs (allocator reuse of "
                         "a registered page range breaks cudaHostRegister)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--claims", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 48]
    )
    ap.add_argument(
        "--n-landing", type=int, nargs="*", default=None,
        help="subset of landing widths to test (default: all x claims matrix)",
    )
    ap.add_argument(
        "--barrier", type=int, default=0,
        help="wait for N processes to reach the measure phase (concurrent-"
             "registration races corrupt driver state on this box)",
    )
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    pid = os.getpid()

    # 16 B multiples: the kernel's uint4 body requires aligned row starts
    # (the engine's Marlin-packed rows are; synthetic ones must be too).
    rows = [(int(m * MIB) + 15) & ~15 for m in ROW_MIBS]
    slab_bytes = args.host_rows * sum(rows)
    slabs = []
    slab_ptrs = []
    for _ in range(args.host_slabs):
        slab = torch.empty(slab_bytes, dtype=torch.uint8)
        slab.random_()
        register_host(slab)
        slabs.append(slab)
        ptrs = []
        off = 0
        for r in rows:
            ptrs.append(slab.data_ptr() + off)
            off += r * args.host_rows
        slab_ptrs.append(ptrs)
    print(f"[pid {pid} gpu{args.gpu}] {args.host_slabs} slabs x "
          f"{slab_bytes / MIB:.0f} MiB registered")

    ref = memcpy_reference(args.host_rows, args.iters)
    print(f"[pid {pid} gpu{args.gpu}] memcpyAsync DtoH ref: "
          f"{ref['ms']:.3f} ms/64MiB-row = {ref['gib_s']:.2f} GiB/s")

    widths = args.n_landing if args.n_landing else [8, 16, 48]
    sessions = []
    for c in args.claims:
        for nl in widths:
            if nl < c:
                continue
            sessions.append(prepare(c, nl, args.host_rows,
                                    slab_ptrs[(c + nl) % len(slab_ptrs)], rows))
    print(f"[pid {pid} gpu{args.gpu}] {len(sessions)} sessions prepared")

    if args.barrier:
        import glob
        import time
        ready = f"/tmp/p1_bench_ready_{args.gpu}"
        open(ready, "w").close()
        while len(glob.glob("/tmp/p1_bench_ready_*")) < args.barrier:
            time.sleep(0.2)
        print(f"[pid {pid} gpu{args.gpu}] barrier passed, measuring")

    for sess in sessions:
        r = measure(sess, args.iters)
        print(f"[pid {pid} gpu{args.gpu}] claims={r['claims']:3d} "
              f"n_landing={r['n_landing']:3d} "
              f"grid={r['grid']:5d}: {r['ms']:7.3f} ms  "
              f"{r['gib_s']:7.2f} GiB/s of claimed bytes")


if __name__ == "__main__":
    main()
