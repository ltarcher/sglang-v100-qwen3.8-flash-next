"""P2.5 switch-path validation client.

trigger: long random-token request (known to make the cost-aware policy switch
    width mid-request); reports wall time and accept length.
probe:   short clean real-text request; prints the generated text so a human
    can see whether the engine still decodes sanely after the switch.

Usage (inside sglang-v100-dev): /opt/venv/bin/python scripts/p25_switch_trigger.py trigger|probe
"""

import json
import random
import string
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8500/generate"


def post(payload, timeout):
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return time.perf_counter() - t0, resp


def meta_summary(resp, wall):
    mi = resp["meta_info"]
    out = mi["completion_tokens"]
    acc = mi.get("spec_accept_length")
    acc_rate = mi.get("spec_accept_rate")
    return "%d tok in %.1fs = %.2f tok/s; accept=%s rate=%s" % (
        out,
        wall,
        out / wall,
        round(acc, 3) if acc else None,
        round(acc_rate, 3) if acc_rate else None,
    )


def trigger():
    rng = random.Random(7)
    text = " ".join(
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
        for _ in range(900)
    )
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": 1.0,
            "max_new_tokens": 384,
            "ignore_eos": True,
        },
    }
    wall, resp = post(payload, timeout=420)
    print(meta_summary(resp, wall), flush=True)


def probe():
    payload = {
        "text": "The quick brown fox jumps over the lazy dog. " * 4,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 32,
            "ignore_eos": True,
        },
    }
    wall, resp = post(payload, timeout=180)
    print(meta_summary(resp, wall), flush=True)
    print(repr(resp["text"][:200]), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "trigger"
    {"trigger": trigger, "probe": probe}[mode]()
