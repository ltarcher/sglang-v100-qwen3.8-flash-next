#!/usr/bin/env python
"""Generate oracle fixtures for the mini GLM-5.3-Flash registered test.

Serves as the M0 oracle step: llama-glm5 (llama.cpp fork) runs the identical
fp16 GGUF that the mini checkpoint was built from, and the next-token top-k
logprobs recorded here become the ground truth for
test/registered/e2e/models/test_glm53_mini_sm70.py.

Usage (inside the sglang-v100 container, llama-server already up):
  ORACLE_URL=http://127.0.0.1:8401 /opt/venv/bin/python \
    /opt/sglang/scripts/gen_fixtures_oracle.py

Note on the greedy-text section: a random-weight model emits arbitrary byte
soup that can trip llama-server's content PEG parser (HTTP 500 "does not match
the expected Content-only format"). The fixtures the registered test consumes
are the logprob lists (short outputs, well clear of the parser); the greedy
text is informational and treated best-effort.
"""

import json
import os
import urllib.error
import urllib.request

URL = os.environ.get("ORACLE_URL", "http://127.0.0.1:8321")
OUT = "/data/models/glm53-oracle/fixtures/mini-fp16/fixtures.json"

PROMPTS = [
    "中国的首都是",
    "def fibonacci(n):",
    "The quick brown fox",
    "请用一句话解释什么是注意力机制:",
    "1, 2, 3, 5, 8,",
]


def post(path, payload):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main():
    out = {}
    for p in PROMPTS:
        entry = {}
        try:
            r = post(
                "/completion",
                {
                    "prompt": p,
                    "n_predict": 48,
                    "temperature": 0.0,
                    "top_k": 1,
                    "cache_prompt": False,
                    "seed": 42,
                },
            )
            entry["text"] = r["content"]
            entry["tokens"] = [t["id"] for t in r.get("tokens", [])]
        except urllib.error.HTTPError as e:
            entry["text"] = (
                f"<oracle content-parser rejected greedy output: HTTP {e.code}>"
            )
        # next-token top-8 logprobs: the fixture the registered test consumes
        payload = {
            "prompt": p,
            "n_predict": 4,
            "temperature": 0.0,
            "top_k": 1,
            "n_probs": 8,
            "cache_prompt": False,
            "seed": 42,
        }
        try:
            r2 = post("/completion", payload)
        except urllib.error.HTTPError:
            payload["seed"] = 43
            r2 = post("/completion", payload)
        entry.setdefault("tokens", [])
        entry["probs"] = r2.get("completion_probabilities", [])
        out[p] = entry

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("written", len(out), "prompts to", OUT)
    for p in PROMPTS:
        probs = out[p]["probs"]
        n = len(probs[0]["top_logprobs"]) if probs else 0
        print(repr(p), "probs_entries:", len(probs), "top:", n)


if __name__ == "__main__":
    main()
