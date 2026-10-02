#!/usr/bin/env python3
"""Build the GLM-5.3-Flash per-layer cold-expert table from ROUTEPROBE dumps.

Input: ``SGLANG_SPILL_ROUTE_PROBE`` dumps -- ``call_r<R>_<n>.pt`` files, each a
[T, topk] int32 topk_ids for one routed-MoE layer call, saved in model layer
order (42 routed calls per forward; the NextN draft layer is spill-exempt).
Calls of equal T group into forwards of ``--layers`` calls.

Output: ``{"cold_ids": int64 [layers, ep, local]}`` with local routed expert
ids in coldness order (coldest first). The server takes the first ``n_spilled``
per (layer, ep rank) via ``SGLANG_DSV41_EXPERT_SPILL_COLD_SET=<out.pt>``;
without the file, placement is the static tail ``[kept, local)``.

The printed holdout number is the honest spill-share estimate: the table is
fitted on every forward, but scored on the last ``--holdout-frac`` as if they
were unseen traffic (temporal split; the domain caveat is whatever the probe
run generated).

    python scripts/glm53_cold_set_from_route_probe.py \
        /tmp/route_probe --out /data/develop/MoE4All/artifacts/glm53_cold_set.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import torch


def load_forwards(
    probe_dir: str, layers: int, experts: int
) -> List[List[List[List[int]]]]:
    """Per rank: forwards -> per-layer lists of routed expert ids (T flattened).

    Calls chunk into forwards strictly within one rank: ranks capture the same
    step concurrently, and a trailing partial forward of rank r must not absorb
    the head of rank r+1 (that rotates every later layer's ordinal).
    """
    per_rank: Dict[int, List[Tuple[int, str]]] = {}
    for name in sorted(os.listdir(probe_dir)):
        m = re.match(r"call_r(\d+)_(\d+)\.pt", name)
        if m:
            per_rank.setdefault(int(m.group(1)), []).append(
                (int(m.group(2)), os.path.join(probe_dir, name))
            )
    if not per_rank:
        raise SystemExit(f"no call_r*_*.pt under {probe_dir}")
    ranks: List[List[List[List[int]]]] = []
    widths: Counter = Counter()
    for rank in sorted(per_rank):
        calls: List[torch.Tensor] = []
        for _, path in sorted(per_rank[rank]):
            t = torch.load(path, map_location="cpu", weights_only=True)
            t = t.reshape(-1).to(torch.int64)
            t = t[(t >= 0) & (t < experts)]
            widths[int(t.numel())] += 1
            calls.append(t)
        forwards = []
        for i in range(len(calls) // layers):
            chunk = calls[i * layers : (i + 1) * layers]
            forwards.append([chunk[l].tolist() for l in range(layers)])
        ranks.append(forwards)
    print(
        f"{sum(len(f) for f in ranks)} forwards of {layers} layers across "
        f"{len(ranks)} ranks ({[len(f) for f in ranks]}); "
        f"tokens/call histogram {dict(sorted(widths.items()))}"
    )
    return ranks


def counts_of(forwards: List[List[List[int]]], layers: int, experts: int) -> torch.Tensor:
    cnt = torch.zeros(layers, experts, dtype=torch.float64)
    for fw in forwards:
        for l, ids in enumerate(fw):
            for e in ids:
                cnt[l, e] += 1
    return cnt


def cold_order(cnt: torch.Tensor) -> torch.Tensor:
    """[layers, experts] ids sorted by ascending hit count; ties keep id order."""
    return cnt.argsort(dim=-1, stable=True).to(torch.int64)


def spill_share(cnt: torch.Tensor, cold: torch.Tensor, spill: int) -> float:
    """Share of activations whose expert is in the coldest ``spill`` per layer."""
    hit = cnt.gather(1, cold[:, :spill]).sum()
    return float(hit / cnt.sum().clamp(min=1))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("probe_dir", help="ROUTEPROBE dump directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=42, help="routed MoE layers seen by the probe")
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--ep", type=int, default=1, help="moe_ep_size (GLM TP4 has EP off)")
    ap.add_argument(
        "--first-dense",
        type=int,
        default=0,
        help="model layers before the first routed MoE; the table ships "
        "model-layer_id rows (the row the server reads), so GLM needs 3",
    )
    ap.add_argument("--spill", type=int, nargs="+", default=[144], help="n_spilled to report")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    args = ap.parse_args(argv)

    ranks = load_forwards(args.probe_dir, args.layers, args.experts)
    n_fw = sum(len(f) for f in ranks)
    if n_fw < 10:
        raise SystemExit(f"only {n_fw} forwards; not enough to fit a profile")
    # Temporal split inside each rank: the same forwards are replated on every
    # rank (TP-replicated routing), so a cross-rank split would leak by
    # construction.
    train: List[List[List[int]]] = []
    test: List[List[List[int]]] = []
    for forwards in ranks:
        holdout = max(1, int(len(forwards) * args.holdout_frac))
        train.extend(forwards[:-holdout])
        test.extend(forwards[-holdout:])

    cnt_all = counts_of(forwards, args.layers, args.experts)
    cnt_train = counts_of(train, args.layers, args.experts)
    cnt_test = counts_of(test, args.layers, args.experts)
    tail = torch.arange(args.experts - 1, -1, -1, dtype=torch.int64).expand(
        args.layers, -1
    )

    order = cold_order(cnt_all)
    for spill in args.spill:
        # Tail placement serves kept=[0, local-spill): its spill share on the
        # holdout is what the server sees today.
        tail_share = spill_share(cnt_test, tail, spill)
        fit_share = spill_share(cnt_test, cold_order(cnt_train), spill)
        ship_share = spill_share(cnt_test, order, spill)
        print(
            f"spill {spill}/{args.experts}: holdout spill share "
            f"tail {tail_share:.4f} -> profiled {fit_share:.4f} "
            f"(shipped table {ship_share:.4f}); "
            f"page-in bytes ratio {ship_share / max(tail_share, 1e-9):.3f}x"
        )

    # Ship model-layer_id rows: dense front layers never route to a spill-wired
    # MoE, so their rows get the identity order and are never consulted.
    n_rows = args.first_dense + args.layers
    shipped = torch.arange(args.experts, dtype=torch.int64).expand(
        n_rows, args.experts
    ).contiguous()
    shipped[args.first_dense :] = order
    cold_ids = shipped.unsqueeze(1).expand(n_rows, args.ep, args.experts).contiguous()
    torch.save(
        {
            "cold_ids": cold_ids,
            "source": sorted(glob.glob(os.path.join(args.probe_dir, "call_r*_*.pt"))),
            "routed_hits": int(cnt_all.sum()),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ep": args.ep,
            "local": args.experts,
            "first_dense": args.first_dense,
            "n_forwards": n_fw,
            "holdout_frac": args.holdout_frac,
        },
        args.out,
    )
    print(f"wrote {args.out}: cold_ids {tuple(cold_ids.shape)} (coldest first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
