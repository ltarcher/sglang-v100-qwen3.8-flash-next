#!/usr/bin/env python3
# P0 probe: Kineto decomposition of GLM-5.3-Flash decode/prefill steps.
#
# Drives the server's built-in /start_profile endpoint (num_steps auto-stop),
# waits for the per-rank chrome traces, then aggregates GPU kernels into named
# buckets plus the memcpy ledger, and derives the per-step decomposition:
#
#   wall/step  trace span / steps   (steps = spill-assign launches / layers,
#                                    or --steps)
#   gpu busy   sum(kernel+memcpy) / steps
#   host gap   wall - busy          (the Colibri "host-wait" column)
#
# Buckets: spill page-in (assign+copy), Marlin MoE, KDA/linear attention,
# DSA/indexer attention, allreduce/NCCL, everything else. Memcpys are summed
# per direction with bytes. UVA reads inside spill_copy_kernel are NOT memcpy
# events -- their bytes come from the cache-stats ledger (P0 double ledger),
# not from this trace.
#
# Usage (server already up, e.g. MoE4All scripts/launch_bringup35h.sh):
#   python3 scripts/p0_trace_probe.py decode --steps 12 --max-tokens 48
#   python3 scripts/p0_trace_probe.py prefill --length 7792
#   python3 scripts/p0_trace_probe.py parse <trace.json.gz> [--steps 12]
#
# Env: SGLANG_V100_HOST (default 127.0.0.1), SGLANG_V100_PORT (default 8500,
# the GLM bringup port; Qwen/DSV4.1 use 11435 instead).

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
import urllib.request

HOST = os.environ.get("SGLANG_V100_HOST", "127.0.0.1")
PORT = os.environ.get("SGLANG_V100_PORT", "8500")
BASE = f"http://{HOST}:{PORT}"

# Repo root = this file's dir/.. ; the server sees the same tree at
# /opt/sglang, so a repo-relative output dir is valid from both sides.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "tmp", "p0_trace")

# Spill page-in runs once per MoE layer per step; 42 routed layers in
# GLM-5.3-Flash give the step count without any server-side counter.
PAGEIN_ASSIGN = "spill_assign"
N_LAYERS = 42

# (bucket, name-substring) pairs, first match wins.
BUCKETS = [
    ("spill_pagein", "spill_assign"),
    ("spill_pagein", "spill_copy"),
    ("marlin_moe", "marlin"),
    ("kda_linear_attn", "kda"),
    ("dsa_attention", "indexer"),
    ("dsa_attention", "dsa"),
    ("allreduce", "all_reduce"),
    ("allreduce", "allreduce"),
    ("allreduce", "cross_device_reduce"),
    ("allreduce", "nccl"),
]


def post(path: str, payload: dict, timeout: float = 2400.0):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def make_ids(n: int) -> list[int]:
    return [1000 + (i * 7919) % 50000 for i in range(n)]


def wait_for_traces(out_dir: str, not_before: float, timeout: float) -> list[str]:
    end = time.time() + timeout
    while time.time() < end:
        found = [
            f for f in glob.glob(os.path.join(out_dir, "**", "*.trace.json*"),
                                 recursive=True)
            if os.path.getmtime(f) >= not_before
        ]
        if found:
            time.sleep(3.0)  # let all ranks finish writing
            return sorted(found)
        time.sleep(2.0)
    return []


def run_profile(args, workload) -> list[str]:
    os.makedirs(args.out_dir, exist_ok=True)
    workload(warmup=True)  # JIT/capture/radix warm before the measured window
    body = {
        "output_dir": args.out_dir_server,
        "num_steps": args.steps,
        "activities": [a.strip() for a in args.activities.split(",") if a.strip()],
    }
    t0 = time.time()
    with post("/start_profile", body) as r:
        print("start_profile:", r.read().decode()[:200])
    workload(warmup=False)
    try:
        with post("/stop_profile", {}) as r:
            print("stop_profile:", r.read().decode()[:200])
    except Exception as e:  # num_steps auto-stop may have closed it already
        print("stop_profile:", e)
    traces = wait_for_traces(args.out_dir, t0, args.wait)
    if not traces:
        sys.exit(f"no trace written under {args.out_dir} within {args.wait}s")
    for f in traces:
        print("trace:", f)
    return traces


def decode_workload(args):
    def run(warmup: bool):
        n = 8 if warmup else max(args.max_tokens, args.steps + 4)
        with post("/generate", {
            "input_ids": [1000],
            "sampling_params": {
                "max_new_tokens": n, "temperature": 0.0, "ignore_eos": True},
            "stream": False,
        }) as r:
            out = json.loads(r.read())
            if not warmup:
                meta = out.get("meta_info", {})
                print("decode done:", meta.get("completion_tokens"), "tokens;",
                      "cached_prompt:", meta.get("cached_prompt_tokens"))
    return run


def prefill_workload(args):
    def run(warmup: bool):
        n = 256 if warmup else args.length
        with post("/generate", {
            "input_ids": make_ids(n),
            "sampling_params": {
                "max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
            "stream": False,
        }) as r:
            out = json.loads(r.read())
            if not warmup:
                meta = out.get("meta_info", {})
                print("prefill done:", meta.get("prompt_tokens"), "prompt tokens")
    return run


def bucket_of(name: str) -> str:
    low = name.lower()
    for bucket, needle in BUCKETS:
        if needle in low:
            return bucket
    return "other"


def parse_trace(path: str, steps: int | None) -> None:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        trace = json.load(f)
    events = trace["traceEvents"]
    gpu = [e for e in events
           if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
           and "dur" in e]
    if not gpu:
        sys.exit(f"no GPU events in {path}")

    by_bucket: dict[str, list[float]] = {}
    by_name: dict[str, list] = {}
    memcpy: dict[str, list] = {}
    for e in gpu:
        dur_us = e["dur"]
        name = e["name"]
        by_name.setdefault(name, [0, 0])
        by_name[name][0] += dur_us
        by_name[name][1] += 1
        if e.get("cat") == "gpu_memcpy":
            direction = ("DtoH" if "DtoH" in name else
                         "HtoD" if "HtoD" in name else "other")
            nbytes = int(e.get("args", {}).get("bytes", 0) or 0)
            memcpy.setdefault(direction, [0, 0, 0])
            memcpy[direction][0] += dur_us
            memcpy[direction][1] += 1
            memcpy[direction][2] += nbytes
            bucket = f"memcpy_{direction}"
        else:
            bucket = bucket_of(name)
        by_bucket.setdefault(bucket, [0, 0])
        by_bucket[bucket][0] += dur_us
        by_bucket[bucket][1] += 1

    span_us = max(e["ts"] + e["dur"] for e in gpu) - min(e["ts"] for e in gpu)
    busy_us = sum(e["dur"] for e in gpu)
    n_assign = sum(cnt for name, (_, cnt) in by_name.items()
                   if PAGEIN_ASSIGN in name.lower())
    if steps is None:
        if n_assign:
            steps = max(1, round(n_assign / N_LAYERS))
        else:
            sys.exit("cannot infer steps (no spill_assign launches); pass --steps")
    steps = max(1, steps)

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"GPU events {len(gpu)}  wall {span_us/1e3:.1f} ms  "
          f"busy {busy_us/1e3:.1f} ms  (concurrency {busy_us/max(span_us,1):.2f}x)")
    print(f"inferred steps {steps} (spill_assign launches {n_assign} / {N_LAYERS})")
    print(f"\n{'bucket':<18}{'ms total':>10}{'ms/step':>10}{'calls':>9}"
          f"{'calls/step':>12}")
    for bucket, (dur, cnt) in sorted(by_bucket.items(), key=lambda kv: -kv[1][0]):
        print(f"{bucket:<18}{dur/1e3:>10.1f}{dur/1e3/steps:>10.2f}"
              f"{cnt:>9}{cnt/steps:>12.1f}")
    gap_us = span_us - busy_us
    print(f"{'HOST GAP':<18}{gap_us/1e3:>10.1f}{gap_us/1e3/steps:>10.2f}")
    print(f"{'WALL':<18}{span_us/1e3:>10.1f}{span_us/1e3/steps:>10.2f}")

    if memcpy:
        print("\nmemcpy ledger (UVA in-kernel reads NOT included):")
        for direction, (dur, cnt, nbytes) in sorted(memcpy.items()):
            print(f"  {direction:<6} {cnt:>8} calls  {nbytes/1024**2:>10.1f} MiB "
                  f"{dur/1e3:>8.1f} ms  "
                  f"{nbytes/1e9/max(dur/1e6,1e-9):>6.2f} GB/s")

    print("\ntop kernels:")
    for name, (dur, cnt) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:20]:
        print(f"{dur/1e3:9.1f} ms x{cnt:6d} ({dur/1e3/steps:7.2f} ms/step) "
              f"{name[:100]}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("decode", help="profile T=1 padding decode steps")
    pd.add_argument("--steps", type=int, default=12)
    pd.add_argument("--max-tokens", type=int, default=48)
    pd.add_argument("--activities", default="CPU,GPU")
    pfp = sub.add_parser("prefill", help="profile one cold prefill request")
    # Prefill takes the eager path (hundreds of thousands of tiny launches):
    # the CPU/python tracer taxes it ~9x (measured 90.6 -> 10.5 tok/s) and the
    # multi-chunk trace export has killed a rank. Keep GPU-only and short.
    pfp.add_argument("--length", type=int, default=2048)
    pfp.add_argument("--steps", type=int, default=None,
                     help="profiler num_steps (default: chunked steps + margin)")
    pfp.add_argument("--activities", default="GPU")
    pp = sub.add_parser("parse", help="parse an exported chrome trace")
    pp.add_argument("trace")
    pp.add_argument("--steps", type=int, default=None)

    for sp in (pd, pfp, pp):
        sp.add_argument("--out-dir", default=DEFAULT_OUT)
    for sp in (pd, pfp):
        sp.add_argument("--out-dir-server", default="/opt/sglang/tmp/p0_trace",
                        help="same tree as --out-dir, in the server's mount view")
        sp.add_argument("--wait", type=float, default=300.0)

    args = p.parse_args()
    if args.cmd == "parse":
        parse_trace(args.trace, args.steps)
    elif args.cmd == "decode":
        traces = run_profile(args, decode_workload(args))
        parse_trace(traces[0], args.steps)
    else:
        if args.steps is None:
            args.steps = (args.length + 1023) // 1024 + 4
        traces = run_profile(args, prefill_workload(args))
        parse_trace(traces[0], None)


if __name__ == "__main__":
    main()
