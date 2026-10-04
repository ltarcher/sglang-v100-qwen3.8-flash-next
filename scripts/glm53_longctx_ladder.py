"""GLM-5.3 long-context ladder: prefill wall + 5-fact needle per rung.

Per rung: (a) pure-prefill wall via max_tokens=1, (b) the 5-fact needle
with a real generation (needle hits + effective decode rate from the same
request). Prompts are deterministic, so re-running a rung against a live
engine exercises radix prefix reuse on the (a) tree; --reuse re-posts the
last rung's pure-prefill prompt and the wall should drop to seconds.

Rung specs are exact prompt-token targets, optionally "N:label". The
filler reserves ~400 tokens for the needle inserts + question, and the
prompt plus the 640-token decode must stay under the forced
--max-total-tokens pool (e.g. rung 261500 against pool 262144).

Usage:
  python scripts/glm53_longctx_ladder.py --url http://127.0.0.1:8111 \
      --rungs 32000:32k 64500:64k
  python scripts/glm53_longctx_ladder.py --url http://127.0.0.1:8111 \
      --rungs 261500:262k --reuse
"""

import argparse
import json
import time
import urllib.request

# Measured ~57.2 tok/unit on the glm53 engine (fixed chat template).
FILLER = (
    "The sun rises in the east and sets in the west. Rivers flow downhill "
    "toward the sea. Birds build nests in trees and sing in the morning. "
    "The library opens at nine and closes at eight. Students read books and "
    "take notes during class. The market sells fresh fruit on weekends. "
)

NEEDLE_FACTS = [
    ("apples", 7391),
    ("trains", 158),
    ("violins", 60234),
    ("penguins", 7),
    ("diamonds", 884516),
]


def post(url, payload, timeout=3600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def build_prompt(approx_tokens, topics):
    filler_units = max(1, (approx_tokens - 400) // 57)
    ctx = [FILLER] * filler_units
    for k, (topic, number) in enumerate(topics):
        pos = int(len(ctx) * (0.15 + 0.18 * k))
        ctx.insert(
            min(pos, len(ctx)),
            f"Remember this fact: the magic number for {topic} is {number}.\n",
        )
    question = (
        "What is the magic number for each of these topics: "
        + ", ".join(t for t, _ in topics)
        + "? List every topic with its number."
    )
    return "".join(ctx), "".join(ctx) + "\n\n" + question


def rung(url, approx_tokens, label, decode_probe_tokens=640):
    prefill_prompt, needle_prompt = build_prompt(approx_tokens, NEEDLE_FACTS)
    res = {"label": label}

    # (a) pure prefill: max_tokens=1
    t0 = time.time()
    out = post(
        url + "/v1/chat/completions",
        {
            "model": "glm",
            "messages": [{"role": "user", "content": prefill_prompt}],
            "max_tokens": 1,
            "temperature": 0,
        },
    )
    res["prefill_prompt_tokens"] = out["usage"]["prompt_tokens"]
    res["prefill_wall_s"] = round(time.time() - t0, 1)
    res["prefill_tok_s"] = round(
        res["prefill_prompt_tokens"] / res["prefill_wall_s"], 1
    )

    # (b) needle with a real generation
    t0 = time.time()
    out = post(
        url + "/v1/chat/completions",
        {
            "model": "glm",
            "messages": [{"role": "user", "content": needle_prompt}],
            "max_tokens": decode_probe_tokens,
            "temperature": 0,
        },
    )
    wall = time.time() - t0
    ans = out["choices"][0]["message"]["content"]
    hits = [str(n) in ans for _, n in NEEDLE_FACTS]
    res["needle_prompt_tokens"] = out["usage"]["prompt_tokens"]
    res["needle_wall_s"] = round(wall, 1)
    res["hits"] = hits
    res["hits_n"] = f"{sum(hits)}/{len(hits)}"
    res["needle_decode_tok_s"] = round(out["usage"]["completion_tokens"] / wall, 1)
    print(
        f"[{label}] prefill {res['prefill_prompt_tokens']} tok in "
        f"{res['prefill_wall_s']}s -> {res['prefill_tok_s']} tok/s | "
        f"needle {res['hits_n']} {hits} | needle decode "
        f"{res['needle_decode_tok_s']} tok/s over {wall:.0f}s",
        flush=True,
    )
    return res, prefill_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8111")
    ap.add_argument("--out", default="/tmp/glm53_longctx_ladder.json")
    ap.add_argument(
        "--rungs",
        nargs="+",
        required=True,
        help="exact target prompt tokens, optionally 'N:label'",
    )
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="re-post the last rung's pure-prefill prompt to check radix "
        "reuse (wall should drop to seconds)",
    )
    args = ap.parse_args()

    res = {"rungs": {}}
    last_prefill_prompt = None
    last_label = None
    for spec in args.rungs:
        tokens, _, label = spec.partition(":")
        label = label or tokens
        res["rungs"][label], last_prefill_prompt = rung(args.url, int(tokens), label)
        last_label = label

    if args.reuse:
        t0 = time.time()
        out = post(
            args.url + "/v1/chat/completions",
            {
                "model": "glm",
                "messages": [{"role": "user", "content": last_prefill_prompt}],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        wall = time.time() - t0
        res["reuse"] = {
            "label": last_label,
            "prompt_tokens": out["usage"]["prompt_tokens"],
            "wall_s": round(wall, 1),
        }
        print(
            f"[reuse:{last_label}] {wall:.1f}s "
            f"({res['reuse']['prompt_tokens']} prompt tokens)",
            flush=True,
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
