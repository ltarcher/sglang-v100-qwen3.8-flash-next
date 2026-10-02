#!/usr/bin/env python3
"""Real-text decode throughput for the P1.6 A/B.

The registered padding-decode bench feeds uniform-random expert traffic,
which is the worst case for a frequency-profiled placement by construction
(every 144/288 split catches ~50% of hits). This bench drives decode with
real continuations so the resident-half profile is actually exercised:
long real prompts, greedy, ignore_eos, one stream per worker, accept and
wall time from /generate meta_info.

    python3 scripts/p16_realtext_decode.py --concurrency 2 --seconds 45
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8500"

# Long, heterogeneous real-text prompts; continuations stay on-manifold so
# routing matches production-like expert frequencies.
PROMPTS = [
    "Write a detailed technical essay about how modern GPU memory hierarchies "
    "work, covering registers, shared memory, L2, and HBM. Include concrete "
    "examples of how kernel authors reason about data movement, typical "
    "bandwidth numbers, and common pitfalls like bank conflicts and wasted "
    "bandwidth on uncoalesced access patterns.",
    "Explain the history of numerical linear algebra from Gauss to modern "
    "iterated methods, covering Gaussian elimination, LU/QR factorizations, "
    "the QR algorithm for eigenvalues, Krylov methods, and preconditioning. "
    "Write it as flowing prose with concrete algorithmic detail.",
    "Draft a careful code review of a hypothetical pull request that adds a "
    "new caching layer to a web service. Discuss invalidation strategies, "
    "thundering herds, TTL vs explicit eviction, observability, and rollback "
    "safety, then propose the review comments you would leave.",
]


def one_stream(prompt: str, max_tokens: int, out: dict, idx: int) -> None:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
            "ignore_eos": True,
        },
    }
    req = urllib.request.Request(
        BASE + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.loads(r.read())
    wall = time.perf_counter() - t0
    mi = resp["meta_info"]
    out[idx] = {
        "completion_tokens": mi["completion_tokens"],
        "wall": wall,
        "accept": mi.get("spec_accept_length"),
        "accept_rate": mi.get("spec_accept_rate"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--seconds", type=int, default=45,
                    help="soft target: max_tokens scaled from a 45 s probe")
    ap.add_argument("--port", type=int, default=8500)
    args = ap.parse_args()
    global BASE  # noqa: PLW0603
    BASE = f"http://127.0.0.1:{args.port}"

    # Scale max_tokens so a run lasts roughly --seconds at ~5 tok/s/stream.
    max_tokens = max(256, int(args.seconds * 5))

    out: dict = {}
    threads = []
    t0 = time.perf_counter()
    for i in range(args.concurrency):
        th = threading.Thread(
            target=one_stream,
            args=(PROMPTS[i % len(PROMPTS)], max_tokens, out, i),
        )
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    wall_all = time.perf_counter() - t0

    total = sum(o["completion_tokens"] for o in out.values())
    accs = [o["accept"] for o in out.values() if o["accept"]]
    print(
        f"realtext decode C={args.concurrency}: aggregate "
        f"{total / wall_all:.1f} tok/s over {wall_all:.1f}s "
        f"({total} tokens); per-stream "
        + " ".join(
            f"{o['completion_tokens'] / o['wall']:.1f}" for o in out.values()
        )
        + " tok/s; accept "
        + (f"{sum(accs) / len(accs):.2f}" if accs else "n/a")
    )
    for i, o in sorted(out.items()):
        print(
            f"  stream {i}: {o['completion_tokens']} tok in {o['wall']:.1f}s "
            f"({o['completion_tokens'] / o['wall']:.1f} tok/s) "
            f"accept={o['accept']} rate={o['accept_rate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
