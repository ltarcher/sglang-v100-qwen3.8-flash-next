#!/usr/bin/env python3
"""Mini-model A/B for the P2 whole-layer prefill bank.

Boots the mini NVFP4 GLM twice (in-process Engine) with forced spill,
prefill-landing, and a NON-identity cold-set table -- the P1.6 loader bug
was invisible under identity placement -- differing only in
SGLANG_DSV41_SPILL_PREFILL_BANK. Wide-batch generations must match to
greedy-tie tolerance: the bank feeds Marlin the same expert bytes through
different slot indices, so fp16 reduction order may flip borderline tokens.

Run on a free GPU (stop any 4-GPU server first):

    python3 scripts/p2_mini_bank_ab.py
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["SGLANG_DSV41_EXPERT_SPILL_APPLY"] = "1"
# Force most experts to the host mirror: the bank then carries the bulk of
# every forward instead of a token or two.
os.environ["SGLANG_DSV41_EXPERT_SPILL_GB"] = "0.001"
os.environ["SGLANG_DSV41_EXPERT_SPILL_MIN_LOCAL"] = "1"
os.environ["SGLANG_DSV41_EXPERT_SPILL_N_LAYERS"] = "3"
os.environ["SGLANG_ENABLE_DSV41_EXPERT_SPILL_NUMA"] = "0"
os.environ["SGLANG_DSV41_SPILL_LANDING"] = "16"
os.environ["SGLANG_DSV41_SPILL_PREFILL_LANDING"] = "1"

import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, "/opt/sglang/python")

from sglang.srt.entrypoints.engine import Engine  # noqa: E402

MODEL = "/data/models/mini-glm-nvfp4"
TABLE = "/tmp/p2_mini_cold_set.pt"

PROMPTS = [
    "Write a python function that merges two sorted lists.",
    "The ferry crosses the strait twice each hour, carrying freight in the "
    "morning and passengers at midday. Describe how the harbour master "
    "adjusts the schedule when winter storms compress it, and what trades "
    "regularity for safety. " * 6,
    "Count from 1 to 60 in steps of 3, separated by commas.",
    "Modern GPU memory hierarchies place registers closest to the execution "
    "units, then shared memory, then L2. Kernel authors reason about data "
    "movement before arithmetic. " * 8,
]


def build_table() -> None:
    # Non-identity placement: keep a different expert each layer, coldest
    # first so the mirror row order differs from expert id order.
    layers, ep, n = 4, 1, 8
    kept = [2, 5, 0, 7]
    cold = []
    for l in range(layers):
        # Coldest first; the kept id rides at the end (never spilled, since
        # spill_placement takes the first n_spilled entries).
        cold.append([e for e in range(n) if e != kept[l]][::-1] + [kept[l]])
    tab = torch.tensor(cold, dtype=torch.int64).unsqueeze(1)
    assert tab.shape == (layers, ep, n)
    torch.save({"cold_ids": tab, "source": ["p2_mini_bank_ab"]}, TABLE)


def boot(bank: bool) -> Engine:
    os.environ.pop("SGLANG_DSV41_SPILL_PREFILL_BANK", None)
    if bank:
        os.environ["SGLANG_DSV41_SPILL_PREFILL_BANK"] = "1"
    return Engine(
        model_path=MODEL,
        trust_remote_code=True,
        dtype="float16",
        attention_backend="tilelang_fa_v100",
        linear_attn_prefill_backend="triton",
        linear_attn_decode_backend="triton",
        tp_size=1,
        mem_fraction_static=0.15,
        context_length=8192,
        cuda_graph_max_bs_decode=2,
        chunked_prefill_size=512,
        port=8599,
        log_level="info",
    )


def run(eng: Engine) -> dict:
    """Teacher-forced logprob profile + greedy continuation per prompt.

    The mini weights are random, so text comparison is meaningless (every
    token is a near-tie). Same engine bytes fed through different pool slot
    orders must instead agree per prompt position: |dlp| ~ 0 everywhere,
    immune to greedy trajectory divergence.
    """
    out = {}
    for p in PROMPTS:
        sp = {"temperature": 0.0, "max_new_tokens": 1}
        res = eng.generate(p, sp, return_logprob=True, logprob_start_len=0)
        mi = res[0]["meta_info"] if isinstance(res, list) else res["meta_info"]
        # This fork exposes teacher-forced logprobs as input_token_logprobs
        # (paired (token_id, logprob) lists); the first pair is None.
        plp = mi.get("input_token_logprobs")
        vals = []
        for item in plp or []:
            if item is None:
                continue
            lp = item[1] if isinstance(item, (list, tuple)) else item.get("logprob")
            if lp is not None:
                vals.append(float(lp))
        oids = mi.get("output_ids") or [None]
        out[p[:40]] = {
            "prompt_logprobs": vals,
            "first_token": oids[0],
        }
    return out


def main() -> int:
    build_table()
    runs = {}
    for bank in (False, True):
        eng = boot(bank)
        t0 = time.perf_counter()
        runs[bank] = run(eng)
        print(f"bank={bank}: generated in {time.perf_counter() - t0:.1f}s")
        eng.shutdown()
        time.sleep(2)
    worst = 0.0
    for k in runs[False]:
        a = runs[False][k]["prompt_logprobs"]
        b = runs[True][k]["prompt_logprobs"]
        n = min(len(a), len(b))
        if n == 0 or len(a) != len(b):
            print(f"FAIL: {k!r} logprob profile lengths {len(a)} vs {len(b)}")
            return 1
        d = max(abs(x - y) for x, y in zip(a, b))
        mean = sum(abs(x - y) for x, y in zip(a, b)) / n
        worst = max(worst, d)
        print(
            f"{k!r}: n={n} max|dlp|={d:.4f} mean|dlp|={mean:.5f} "
            f"first_tok {runs[False][k]['first_token']} vs "
            f"{runs[True][k]['first_token']}"
        )
        if d > 0.10:
            print("BANK A/B FAILED: logprob displacement above tie noise")
            return 1
    print(f"bank A/B PASSED: worst max|dlp|={worst:.4f} over {len(runs[False])} prompts")
    with open("/tmp/p2_mini_bank_ab.json", "w") as f:
        json.dump(runs, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
