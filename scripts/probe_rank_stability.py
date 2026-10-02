"""Probe: where does the oracle top-1 rank in sglang's top-64, vs prefix length?

Short prompts on a random-weight mini model have intrinsically flat
distributions (top-1 ~ -11 nats ~ ln(154880) - 2.6): rank position is then
noise-dominated and useless as a gate. This finds the prefix length where the
oracle top-1 becomes stable enough (top-8 containment) to gate on.

Run inside the sglang-v100 container with both servers up:
  8399 = sglang mini-glm-fp16, 8401 = llama-glm5 oracle on the same weights.
"""

import json
import urllib.request

PARITY_DOC = "/tmp/parity_doc.py"


def post(url, payload):
    return json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
            ),
            timeout=600,
        ).read()
    )


def main():
    ns = {}
    exec(  # noqa: S102 - trusted local file
        open(PARITY_DOC).read().split("def llama_topk")[0], ns
    )
    ids = [int(i) for i in ns["ids"]]

    def sgl_topk(idlist, k=64):
        mi = post(
            "http://127.0.0.1:8399/generate",
            {
                "input_ids": idlist,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": k,
                "logprob_start_position": -1,
            },
        )["meta_info"]
        return {e[1]: e[0] for e in mi["output_top_logprobs"][0]}

    def orc_topk(n, k=8):
        r = post(
            "http://127.0.0.1:8401/completion",
            {
                "prompt": ids[:n],
                "n_predict": 1,
                "n_probs": k,
                "temperature": 0.0,
                "cache_prompt": False,
            },
        )
        return {p["id"]: p["logprob"] for p in r["completion_probabilities"][0]["top_logprobs"]}

    print("prefix_len oracle_top1 sglang_rank lp_delta  sglang_top1_gap")
    for n in (16, 32, 64, 128, 192, 256):
        s = sgl_topk(ids[:n])
        o = orc_topk(n)
        o1 = max(o, key=o.get)
        ordered = sorted(s, key=s.get, reverse=True)
        rank = ordered.index(o1) + 1 if o1 in ordered else None
        d = abs(s[o1] - o[o1]) if o1 in s else None
        gap = s[ordered[0]] - s[ordered[7]] if len(ordered) >= 8 else None
        print(f"{n:10d} {o1:11d} {str(rank):>11s} {str(d):>8s}  {gap}")


if __name__ == "__main__":
    main()
