"""P3 needle-retrieval probe: multi-needle number retrieval at 1k/4k/16k
tokens against a /generate engine (sglang NVFP4 arm or llama-server Q2_K /
reference arm).

Filler is the P3 corpus cycled to the target length; needles are
"记忆条目 NNNN" lines embedded at even depths; the question asks for the
magic of a named entry. Score = exact retrieval rate. Deterministic (fixed
seed via --seed, temperature 0).

Usage:
  python scripts/p3_needle.py --url http://127.0.0.1:8500 --lengths 1024 4096 16384
"""

import argparse
import json
import random
import re
import time
import urllib.request

CORPUS = open(
    "/data/models/glm53-oracle/fixtures/p3_gate.txt", encoding="utf-8"
).read()

# ~1 token per 3.2 bytes for this mix (Chinese-heavy); padded generously then
# trimmed by token budget reported by the engine when available.
BYTES_PER_TOKEN = 3.4


def build_case(length_tokens, n_needles, rng):
    filler_len = int(length_tokens * BYTES_PER_TOKEN * 0.92)
    filler = (CORPUS * (filler_len // len(CORPUS) + 1))[:filler_len]
    paras = filler.split("\n\n")
    magics = {}
    needle_paras = []
    for i in range(n_needles):
        magic = f"{rng.randrange(10, 99)}-{rng.randrange(1000, 9999)}"
        name = f"条目{i + 1:02d}"
        magics[name] = magic
        needle_paras.append(f"记忆存档:{name} 的魔数是 {magic},请妥善保管。")
    # interleave needles evenly into the filler paragraph list
    slots = max(len(paras), n_needles + 1)
    per = slots // (n_needles + 1)
    merged = []
    ni = 0
    for pi, para in enumerate(paras):
        if ni < n_needles and pi > 0 and pi % per == 0:
            merged.append(needle_paras[ni])
            ni += 1
        merged.append(para)
    while ni < n_needles:
        merged.append(needle_paras[ni])
        ni += 1
    ask = f"以上文档中,条目01 的魔数是多少?只回答魔数本身。"
    return "\n\n".join(merged) + "\n\n" + ask, magics


def gen(url, prompt, timeout=1800, api="sglang"):
    if api == "llama":
        # llama-server native /completion
        payload = {"prompt": prompt, "temperature": 0.0, "n_predict": 32}
        path = "/completion"
        parse = lambda r: r.get("content") or ""
    else:
        payload = {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 32},
        }
        path = "/generate"
        parse = lambda r: r.get("text") or r.get("output") or ""
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    text = parse(resp)
    if isinstance(text, list):
        text = text[0]
    return time.perf_counter() - t0, text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8500")
    ap.add_argument("--api", choices=["sglang", "llama"], default="sglang")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--needles", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    for length in args.lengths:
        hits = 0
        for ci in range(args.cases):
            rng = random.Random(args.seed + 1000 * length + ci)
            prompt, magics = build_case(length, args.needles, rng)
            dt, text = gen(args.url, prompt, api=args.api)
            m = re.search(r"(\d{2}-\d{4})", text)
            ok = m and m.group(1) == magics["条目01"]
            hits += bool(ok)
            print(
                json.dumps(
                    {
                        "len": length,
                        "case": ci,
                        "hit": bool(ok),
                        "got": text[:40],
                        "want": magics["条目01"],
                        "prompt_bytes": len(prompt),
                        "sec": round(dt, 1),
                    },
                    ensure_ascii=False,
                )
            )
        print(f"== len={length}: {hits}/{args.cases} retrieved ==")


if __name__ == "__main__":
    main()
