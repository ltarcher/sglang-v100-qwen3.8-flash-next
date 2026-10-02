"""P2.5 decode bench: single-stream + 4-stream decode against the GLM dev arm.

Usage (inside sglang-v100-dev): /opt/venv/bin/python scripts/p25_decode_bench.py
Reports per-request and aggregate tok/s plus spec accept length. The adaptive
policy's own decisions are read from the server log (grep "adaptive spec").
"""

import concurrent.futures
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8500/generate"

# Real-text openings, distinct per stream (same shape as the P2 prefill bench).
PROMPTS = [
    "The industrial revolution reshaped labor, capital, and cities. " * 12,
    "Plate tectonics explains earthquakes, mountain belts, and volcanoes. " * 12,
    "Photosynthesis converts light, water, and carbon dioxide into sugar. " * 12,
    "Compound interest grows savings, but inflation erodes real returns. " * 12,
]


def gen(prompt, n, timeout=600):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": n,
            "ignore_eos": True,
        },
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return time.perf_counter() - t0, resp["meta_info"]


def accept_of(mi):
    for key in ("spec_accept_length", "accept_length"):
        v = mi.get(key)
        if v:
            return round(float(v), 3)
    return None


def main():
    new_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 256

    _, mi = gen(PROMPTS[0], 1)
    print("warm:", round(_, 2), "s")

    # Convergence load: ~120+ verify rounds so the policy's warmup and the
    # first few decisions pass before the measured window.
    for _ in range(2):
        w, mi = gen(PROMPTS[1], new_tokens * 2)
        print(
            "converge: %d tok in %.1fs = %.2f tok/s; accept=%s"
            % (new_tokens * 2, w, new_tokens * 2 / w, accept_of(mi))
        )

    # Measured: single stream.
    w, mi = gen(PROMPTS[2], new_tokens)
    print(
        "single: %d tok in %.2fs = %.2f tok/s; accept=%s"
        % (new_tokens, w, new_tokens / w, accept_of(mi))
    )

    # Measured: 4 concurrent streams (the dev arm's max_running_requests).
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(gen, PROMPTS[i % len(PROMPTS)], new_tokens)
            for i in range(4)
        ]
        results = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    total = sum(new_tokens for _ in results)
    per_req = ", ".join("%.2f" % (new_tokens / w) for w, _ in results)
    accs = ", ".join(str(accept_of(mi)) for _, mi in results)
    print(
        "x4: total %d tok in %.2fs = %.2f tok/s agg; per-req [%s]; accept [%s]"
        % (total, wall, total / wall, per_req, accs)
    )


if __name__ == "__main__":
    main()
