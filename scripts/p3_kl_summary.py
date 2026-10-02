"""P3 quality-gate summary: parse llama-perplexity KL passes and band the
sglang NVFP4 engine against the fp16 GGUF reference.

The base logits file (llama-perplexity --kl-divergence-base) layout, from
tools/perplexity/perplexity.cpp:

    "_logits_" | u32 n_ctx | i32 n_vocab | i32 n_chunk
    | i32 tokens[n_ctx * n_chunk]
    | per chunk, (n_ctx - 1 - n_ctx/2) records of nv uint16s

A record holds the reference log-softmax of one next-token distribution:
float scale, float min_log_prob, then per vocab entry a uint16 q with
    log_p(v) = scale * q[v] + min_log_prob
(linear 16-bit, log-probs below max-16 flushed to min -- the e^-16 floor).

Subcommands:
    info             print header + verify the file size matches the layout
    sglang-probe     teacher-force the reference tokens through the sglang
                     NVFP4 engine (prompt logprobs, top-k) -> JSON
    compare-sglang   same-top1 / top-k overlap of the engine arm vs the
                     reference arm, per scored position
    summary          grep the two llama-perplexity logs into one table
"""

import json
import struct
import urllib.request

import numpy as np

MAGIC = b"_logits_"


class BaseLogits:
    def __init__(self, path):
        with open(path, "rb") as f:
            # 8-byte magic + u32 n_ctx + i32 n_vocab + i32 n_chunk = 20 bytes
            head = f.read(20)
            if head[:8] != MAGIC:
                raise ValueError(f"{path}: not a logits file")
            self.n_ctx, self.n_vocab, self.n_chunk = struct.unpack("<Iii", head[8:])
            raw = f.read(4 * self.n_ctx * self.n_chunk)
            self.tokens = np.frombuffer(raw, dtype=np.int32).reshape(
                self.n_chunk, self.n_ctx
            )
            self._m = np.memmap(path, dtype=np.uint16, mode="r")
        self.nv = 2 * ((self.n_vocab + 1) // 2) + 4
        self.n_token = self.n_ctx - 1 - self.n_ctx // 2
        self.records = self.n_chunk * self.n_token
        expected = 20 + 4 * self.n_ctx * self.n_chunk + self.records * self.nv * 2
        actual = len(self._m) * 2
        if actual != expected:
            raise ValueError(f"{path}: size {actual} != expected {expected}")

    def record(self, chunk, pos):
        """(tokens, scale-f32, min-f32, q[u16, n_vocab]) for one position."""
        idx = chunk * self.n_token + pos
        off = 10 + 2 * self.n_ctx * self.n_chunk + idx * self.nv
        q = self._m[off : off + self.nv]
        scale = q[:2].view(np.float32)[0]
        min_lp = q[2:4].view(np.float32)[0]
        return self.tokens[chunk], scale, min_lp, q[4 : 4 + self.n_vocab]

    def top1(self, chunk, pos):
        _, scale, min_lp, q = self.record(chunk, pos)
        v = scale * q.astype(np.float32) + min_lp
        return int(np.argmax(v)), v


def cmd_info(args):
    bl = BaseLogits(args.logits)
    print(
        f"n_ctx={bl.n_ctx} n_vocab={bl.n_vocab} n_chunk={bl.n_chunk} "
        f"scored/chunk={bl.n_token} total_scored={bl.records}"
    )


def cmd_sglang_probe(args):
    bl = BaseLogits(args.logits)
    out = []
    for c in range(bl.n_chunk):
        payload = {
            "input_ids": bl.tokens[c].tolist(),
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": args.topk,
        }
        req = urllib.request.Request(
            args.url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        # meta_info.input_top_logprobs[j] = top-k entries [logprob, token_id,
        # None] for the distribution predicting tokens[j]; position 0 = None.
        out.append(
            {
                "chunk": c,
                "input_top_logprobs": resp["meta_info"]["input_top_logprobs"],
            }
        )
        print(f"chunk {c}: {len(resp['meta_info']['input_top_logprobs'])} positions")
    json.dump(out, open(args.out, "w"))


def cmd_compare_sglang(args):
    bl = BaseLogits(args.logits)
    probe = json.load(open(args.probe))
    n = 0
    same = 0
    overlap20 = 0
    for c, resp in enumerate(probe):
        tl = resp.get("input_top_logprobs") or []
        for i in range(bl.n_ctx // 2, bl.n_ctx - 1):
            j = i + 1  # per-request index predicting tokens[j]
            if j >= len(tl) or not tl[j]:
                continue
            ref_top1, ref_lp = bl.top1(c, i - bl.n_ctx // 2)
            entries = tl[j]
            eng_top1 = int(entries[0][1])
            n += 1
            same += eng_top1 == ref_top1
            ref_top20 = set(np.argsort(-ref_lp)[:20].tolist())
            eng_top20 = {int(e[1]) for e in entries[:20]}
            overlap20 += len(ref_top20 & eng_top20) / 20.0
            # probability the reference posterior assigns to the engine top-1
    if n:
        print(
            f"scored={n} same_top1={same} ({100.0 * same / n:.2f}%) "
            f"mean_top20_overlap={overlap20 / n:.4f}"
        )


def cmd_summary(args):
    for tag, path in (("fp16(ref)", args.base_log), ("Q2_K", args.cand_log)):
        text = open(path, encoding="utf-8", errors="replace").read()
        keys = ["Mean PPL", "PPL", "Mean    KLD", "Median  KLD", "99.0%%   KLD", "Same top"]
        print(f"--- {tag} ({path})")
        for ln in text.splitlines():
            if any(k.replace("%%", "%") in ln for k in keys):
                print("   ", ln.strip())


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("info")
    sp.add_argument("--logits", required=True)
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("sglang-probe")
    sp.add_argument("--logits", required=True)
    sp.add_argument("--url", default="http://127.0.0.1:8500")
    sp.add_argument("--topk", type=int, default=20)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_sglang_probe)

    sp = sub.add_parser("compare-sglang")
    sp.add_argument("--logits", required=True)
    sp.add_argument("--probe", required=True)
    sp.set_defaults(fn=cmd_compare_sglang)

    sp = sub.add_parser("summary")
    sp.add_argument("--base-log", required=True)
    sp.add_argument("--cand-log", required=True)
    sp.set_defaults(fn=cmd_summary)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
