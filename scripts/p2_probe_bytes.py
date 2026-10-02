#!/usr/bin/env python3
"""Exact page-in bytes per prefill forward from a ROUTEPROBE corpus.

Each probe file is one MoE-layer call's [T, K] topk_ids (r0 = TP rank 0).
Prefill calls (T >= --min-t) come in consecutive runs of 42 (one forward
pass); within a forward, each unique (layer, cold expert) pair costs one
pool-row copy (3.38 MiB/rank). This groups calls that way and reports
bytes per forward, the union coverage, and the implied effective HtoD
bandwidth when paired with the anatomy walls.

    python3 scripts/p2_probe_bytes.py --probe-dir /tmp/route_probe_p16_mtp_c1
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

ROW_MIB = 3.38  # per-rank packed expert row (measured, cache-stats ledger)
N_MOE_LAYERS = 42


def cold_sets(table_path: str | None, n_routed: int, n_kept: int):
    """(layer, expert) -> is_cold under the given placement."""
    if table_path:
        tab = torch.load(table_path, map_location="cpu", weights_only=False)
        cold_ids = tab["cold_ids"]  # [layers, ep, S] coldest first
        # ep_rank 0, spilled = first n_spilled entries.
        out = []
        for l in range(cold_ids.shape[0]):
            cold = set(int(x) for x in cold_ids[l, 0, : n_routed - n_kept])
            out.append(cold)
        return out
    return [set(range(n_kept, n_routed))] * N_MOE_LAYERS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--min-t", type=int, default=64, help="prefill call filter")
    ap.add_argument("--table", default=None, help="cold-set .pt; unset = tail")
    ap.add_argument("--n-routed", type=int, default=288)
    ap.add_argument("--n-kept", type=int, default=144)
    args = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.probe_dir, f"call_r{args.rank}_*.pt"))
    )
    if not files:
        raise SystemExit(f"no probe files under {args.probe_dir}")
    colds = cold_sets(args.table, args.n_routed, args.n_kept)

    # Contiguous prefill runs of 42 calls = one forward pass. Calls within
    # a run map to layers in order.
    forwards = []
    run: list[set[int]] = []
    uniq_all = 0
    cold_all = 0
    for f in files:
        ids = torch.load(f, map_location="cpu", weights_only=False)
        if ids.dim() != 2 or int(ids.shape[0]) < args.min_t:
            continue
        layer = len(run)
        u = {int(x) for x in ids.flatten().unique()}
        run.append(u)
        uniq_all += len(u)
        cold_all += len(u & colds[layer])
        if len(run) == N_MOE_LAYERS:
            forwards.append((uniq_all, cold_all))
            run, uniq_all, cold_all = [], 0, 0
    if run:
        forwards.append((uniq_all, cold_all))

    gib = [c * ROW_MIB / 1024 for _, c in forwards]
    print(f"prefill forwards: {len(forwards)}")
    for i, ((u, c), g) in enumerate(zip(forwards, gib)):
        print(
            f"  forward {i}: unique/layer avg {u / N_MOE_LAYERS:.1f} "
            f"cold/layer avg {c / N_MOE_LAYERS:.1f} "
            f"page-in {g:.2f} GiB (rank {args.rank})"
        )
    tot = sum(gib)
    print(
        f"total page-in: {tot:.2f} GiB over {len(forwards)} forwards "
        f"({tot / max(len(forwards), 1):.2f} GiB/forward avg)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
