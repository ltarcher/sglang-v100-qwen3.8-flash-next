#!/usr/bin/env python3
# V100 engine benchmark, stdlib-only (no client deps). Reproduces the README
# measurement protocol for both engines:
#
#   prefill: exact-length prompt (input_ids), one scout per length,
#            tok/s = prompt_tokens / TTFT (time to first streamed chunk)
#   decode:  padding decode (ignore_eos) for --seconds, C concurrency,
#            aggregate + per-stream tok/s, accept length from the server's
#            spec_accept_length gauge when MTP is on
#
# Usage:
#   python3 scripts/bench_v100.py prefill --lengths 1024,8192,32768 [--runs 1]
#   python3 scripts/bench_v100.py decode --seconds 15 --concurrency 1,2,3 \
#       [--context 0] [--max-tokens 100000]
#   python3 scripts/bench_v100.py all --lengths 1024,2048 --seconds 15
#
# Env: SGLANG_V100_HOST (default 127.0.0.1), SGLANG_V100_PORT (default 11435).
# Point it at a running server, e.g.
#   bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp
#   python3 scripts/bench_v100.py all --lengths 8192 --seconds 15

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

HOST = os.environ.get("SGLANG_V100_HOST", "127.0.0.1")
PORT = os.environ.get("SGLANG_V100_PORT", "11435")
BASE = f"http://{HOST}:{PORT}"


def post(path: str, payload: dict, timeout: float = 600.0):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def make_ids(n: int) -> list[int]:
    """Deterministic pseudo-random ids in a populated vocab range.

    Random ids defeat the radix cache and the MTP draft (real-text accept
    reads differently -- see the README padding caveat), which is what the
    padding-decode rows want.
    """
    return [1000 + (i * 7919) % 50000 for i in range(n)]


def prefill_one(n: int, timeout: float = 1200.0) -> tuple[float, int]:
    """Stream a --max-new-tokens 1 request; return (ttft_s, server_prompt_tokens)."""
    t0 = time.perf_counter()
    ttft = None
    server_n = None
    with post("/generate", {
        "input_ids": make_ids(n), "sampling_params": {
            "max_new_tokens": 1, "temperature": 0.0, "ignore_eos": False},
        "stream": True,
    }, timeout=timeout) as r:
        for line in r:
            if line.startswith(b"data:") and line.strip() != b"data: [DONE]":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunk = json.loads(line[5:])
                if chunk.get("meta_info", {}).get("prompt_tokens"):
                    server_n = chunk["meta_info"]["prompt_tokens"]
    return ttft, server_n or n


def decode_worker(ctx: int, seconds: float, max_tokens: int,
                  out: list[dict], idx: int):
    """Padding decode for `seconds`; record tokens and accept length."""
    ids = make_ids(ctx) if ctx else [1000]
    sent, accept, toks = 0, [], 0
    t_end = time.perf_counter() + seconds
    try:
        with post("/generate", {
            "input_ids": ids, "sampling_params": {
                "max_new_tokens": max_tokens, "temperature": 0.0,
                "ignore_eos": True},
            "stream": True, "return_logprob": False,
        }) as r:
            for line in r:
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                text = chunk.get("text", "")
                toks += 1  # one streamed chunk == one step's committed tokens
                sent = time.perf_counter()
                mi = chunk.get("meta_info") or {}
                if mi.get("spec_accept_length") is not None:
                    accept = [mi["spec_accept_length"]]
                if time.perf_counter() >= t_end:
                    break
    except Exception as e:  # noqa: BLE001
        out[idx] = {"error": str(e), "tokens": toks}
        return
    out[idx] = {"tokens": toks, "accept": accept[0] if accept else None,
                "span": seconds}


def cmd_prefill(args):
    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"prefill: {BASE}  temp 0, one scout per length\n")
    print(f"{'prompt tokens':>14} {'TTFT (s)':>10} {'tok/s':>9}")
    for n in lengths:
        ttfts = []
        server_n = n
        for _ in range(args.runs):
            ttft, server_n = prefill_one(n)
            ttfts.append(ttft)
        ttft = sum(ttfts) / len(ttfts)
        print(f"{server_n:>14} {ttft:>10.2f} {server_n / ttft:>9.0f}")


def cmd_decode(args):
    concs = [int(x) for x in args.concurrency.split(",")]
    ctxs = [int(x) for x in args.context.split(",")]
    # warmup: JIT kernels + cuda graphs on a throwaway 1-token decode
    decode_worker(0, 5.0, args.max_tokens, {}, 0)
    print(f"decode: {BASE}  padding {args.seconds}s, temp 0, ignore_eos\n")
    for ctx in ctxs:
        print(f"  context {ctx}:")
        for c in concs:
            out: list[dict] = [None] * c
            threads = [threading.Thread(target=decode_worker,
                                        args=(ctx, args.seconds,
                                              args.max_tokens, out, i))
                       for i in range(c)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            span = time.perf_counter() - t0 - args.seconds
            errs = [o for o in out if o and "error" in o]
            toks = sum(o.get("tokens", 0) for o in out if o and "error" not in o)
            accs = [o["accept"] for o in out
                    if o and "error" not in o and o.get("accept")]
            agg = toks / (span if span > 1.0 else args.seconds)
            acc = f"{sum(accs) / len(accs):.2f}" if accs else "n/a"
            note = f"  ERRORS: {errs[0]['error']}" if errs else ""
            print(f"    C={c}: aggregate {agg:6.1f} tok/s"
                  f"  per-stream {agg / c:6.1f}  accept {acc}{note}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prefill", help="TTFT / prefill tok/s at exact lengths")
    pp.add_argument("--lengths", default="8192", help="comma token counts")
    pp.add_argument("--runs", type=int, default=1)
    pp.set_defaults(fn=cmd_prefill)

    pd = sub.add_parser("decode", help="padding decode tok/s + accept")
    pd.add_argument("--seconds", type=float, default=15.0)
    pd.add_argument("--concurrency", default="1,2,3")
    pd.add_argument("--context", default="0", help="prompt context per worker")
    pd.add_argument("--max-tokens", type=int, default=10_000_000)
    pd.set_defaults(fn=cmd_decode)

    pa = sub.add_parser("all", help="prefill then decode")
    pa.add_argument("--lengths", default="8192")
    pa.add_argument("--seconds", type=float, default=15.0)
    pa.add_argument("--concurrency", default="1,2,3")
    pa.add_argument("--context", default="0")
    pa.add_argument("--runs", type=int, default=1)
    pa.add_argument("--max-tokens", type=int, default=10_000_000)

    def run_all(args):
        cmd_prefill(args)
        print()
        cmd_decode(args)
    pa.set_defaults(fn=run_all)

    args = p.parse_args()
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"no server at {BASE}: {e}")
    args.fn(args)


if __name__ == "__main__":
    main()
