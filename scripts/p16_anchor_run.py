#!/usr/bin/env python3
"""Run the P1.6 anchor prompts and compare against the P1 baseline outputs.

The three P1 anchor prompts exercise counting (format stability), world
knowledge, and code structure -- enough to catch the P1.6 Arm B corruption
class (broken scale_2 fold -> degraded MoE output -> router tie-collapse).
Greedy sampling; completion length matches the P1 capture (128).

    python3 scripts/p16_anchor_run.py --out /tmp/anchor_c1.json \
        [--baseline /tmp/mtp_anchor_out_p1.json]

With --baseline each prompt is judged against the P1 text by difflib ratio;
garbage (degenerate repetition / all-punctuation) fails outright.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from difflib import SequenceMatcher

BASE = "http://127.0.0.1:8500"

PROMPTS = [
    "Count from 1 to 300 in steps of 2, separated by commas.",
    "Write a short paragraph about why the sky is blue.",
    "Write a python function that merges two sorted lists.",
]

MAX_NEW_TOKENS = 128


def generate(prompt: str) -> dict:
    # Chat endpoint: the P1 baseline outputs carry the templated instruction
    # persona ("The user wants me to ..."), which raw /generate completions
    # do not reproduce.
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_NEW_TOKENS,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    wall = time.perf_counter() - t0
    return {
        "text": out["choices"][0]["message"]["content"],
        "completion_tokens": out["usage"]["completion_tokens"],
        "wall": round(wall, 2),
    }


def degenerate(text: str) -> bool:
    """All-punctuation soup or a single token repeated >= 8 times."""
    if text and not re.search(r"[a-zA-Z0-9一-鿿]", text):
        return True
    toks = text.split()
    if len(toks) >= 8 and len(set(toks)) <= 2:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--port", type=int, default=8500)
    args = ap.parse_args()
    global BASE  # noqa: PLW0603
    BASE = f"http://127.0.0.1:{args.port}"

    out = {}
    for p in PROMPTS:
        out[p] = generate(p)
        print(f"OK {p[:40]!r} wall={out[p]['wall']}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    if not args.baseline:
        return 0
    with open(args.baseline) as f:
        base = json.load(f)
    print("\n=== vs baseline ===")
    bad = 0
    for p in PROMPTS:
        text = out[p]["text"]
        if degenerate(text):
            print(f"GARBAGE: {p[:50]!r}")
            bad += 1
            continue
        ref = base[p]["text"]
        ratio = SequenceMatcher(None, text, ref).ratio()
        flag = "OK" if ratio >= 0.55 else "DRIFT"
        if ratio < 0.55:
            bad += 1
        print(f"{flag}: {p[:50]!r} ratio={ratio:.3f} len={len(text)} (ref {len(ref)})")
    print("ANCHOR", "FAILED" if bad else "PASSED", f"({bad} bad)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
