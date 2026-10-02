#!/usr/bin/env python3
"""P2 prefill anatomy: per-request wall for unique-prefix 4k prompts.

The registered prefill bench reuses prompts, so request 2+ hit the radix
prefix cache and the "warm" number hides the cold-chunk cost. This probe
feeds prompts with distinct openings (radix shares nothing beyond ~10
tokens), one new token out, and reports per-request prefill tok/s so the
cold path and the steady-state path separate cleanly.

Pair the walls with the ROUTEPROBE corpus written by the same boot
(SGLANG_SPILL_ROUTE_PROBE): unique (layer, expert) misses per call x row
bytes is the exact page-in traffic the pool had to pull.

    python3 scripts/p2_prefill_anatomy.py --requests 6 --target-tokens 4096
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

BASE = "http://127.0.0.1:8500"

# Distinct openings force separate radix trees; the bodies mix six varied
# paragraph pools so routing stays production-like rather than uniform.
OPENINGS = [
    "The committee reviewed the proposal in the morning session.",
    "Rain moved across the valley before the market opened.",
    "Our team ships a compiler backend for old workstation GPUs.",
    "The recipe asks for slow reduction, not a hard boil.",
    "Historians disagree about the treaty's second clause.",
    "A small ferry crosses the strait twice each hour.",
]

BODIES = [
    "Modern GPU memory hierarchies place registers closest to the execution "
    "units, then shared memory, then L2, then device memory. Kernel authors "
    "reason about data movement before arithmetic: a register spill costs "
    "far more than a fused multiply, and an uncoalesced load can halve "
    "effective bandwidth. Bank conflicts serialize shared-memory access "
    "that the hardware could otherwise issue in one cycle.",
    "Numerical linear algebra grew from hand computation to machine scale. "
    "Gaussian elimination remains the backbone of dense solves; LU and QR "
    "factorizations trade fill-in against stability; the QR algorithm "
    "tames eigenvalue problems; Krylov methods and preconditioners make "
    "sparse systems tractable without forming matrices explicitly.",
    "A caching layer for a web service must decide what to keep, where to "
    "keep it, and when to admit that a guess was wrong. Invalidation "
    "strategies range from short TTLs to explicit purges; thundering herds "
    "appear the moment a popular key expires; observability decides "
    "whether the on-call engineer learns about the miss rate from a "
    "dashboard or from a customer.",
    "The ferry timetable survived three revisions. Freight takes the early "
    "crossing; passengers cluster at midday; the last run leaves after the "
    "market closes. Winter storms compress the schedule, and the harbour "
    "master trades regularity for safety without announcing it.",
    "Compilers for old hardware live or die on register allocation. The "
    "target has no bf16, limited double throughput, and a memory system "
    "that punishes scattered access. Every kernel rewrite begins with a "
    "counter: bytes moved, not instructions issued, decide the ceiling.",
    "Slow reduction builds flavor that a hard boil destroys. The same "
    "patience shows up in bread, in stock, and in tea. Rushing the stage "
    "does not save time; it moves the work to a later step that costs "
    "more.",
]


def build_prompt(idx: int, rng: random.Random) -> str:
    body = " ".join(BODIES[(idx + j) % len(BODIES)] for j in range(40))
    # A unique numeric seed line keeps even same-pool prompts disjoint from
    # token one onward, so the radix cache cannot carry prefix credit.
    return f"[doc-{idx:03d}-{rng.randrange(10**9)}] {OPENINGS[idx % len(OPENINGS)]} {body}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--out", default="/tmp/p2_prefill_anatomy.json")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    # Wait for health.
    for _ in range(300):
        try:
            urllib.request.urlopen(base + "/health", timeout=2)
            break
        except Exception:
            time.sleep(2)
    else:
        raise SystemExit("server never became healthy")

    rng = random.Random(1234 + args.target_tokens)
    rows = []
    for i in range(args.requests):
        prompt = build_prompt(i, rng)
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        }
        req = urllib.request.Request(
            base + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read())
        wall = time.perf_counter() - t0
        mi = resp["meta_info"]
        ptok = int(mi["prompt_tokens"])
        rows.append(
            {
                "request": i,
                "prompt_tokens": ptok,
                "wall": round(wall, 3),
                "prefill_tok_s": round(ptok / wall, 1),
                "t_end": round(time.time(), 3),
            }
        )
        print(
            f"request {i}: {ptok} prompt tokens in {wall:.2f}s "
            f"= {ptok / wall:.1f} tok/s"
        )

    steady = [r for r in rows[1:]]
    s_tok = sum(r["prompt_tokens"] for r in steady)
    s_wall = sum(r["wall"] for r in steady)
    print(
        f"\nfirst request: {rows[0]['prefill_tok_s']} tok/s; "
        f"steady (requests 1+): {s_tok / s_wall:.1f} tok/s"
    )
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
