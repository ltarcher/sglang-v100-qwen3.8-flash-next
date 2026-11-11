# ruff: noqa: E402  (imports follow the flush-heavy diagnostic flow deliberately)
"""A/B: 32k row on EMPTY cache vs on WARM cache, full-32q discriminator."""

import json
import importlib.util
import urllib.request
import sys

spec = importlib.util.spec_from_file_location(
    "bq", "scripts/testbench/bench_quality_v100.py"
)
bq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bq)
bq.resolve_server(None)


def flush():
    req = urllib.request.Request(bq.BASE + "/flush_cache", method="POST")
    urllib.request.urlopen(req, timeout=120).read()


from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    bq.args_tokenizer_path(), trust_remote_code=True, use_fast=True
)
rows = bq._read_jsonl(
    bq.Path("/data/sm70_release_datasets/longbench/data/narrativeqa.jsonl")
)
fmt = json.loads((bq.SCRIPT_DIR / "longbench_prompts.json").read_text())["narrativeqa"]
ids = tok(fmt.format(**rows[0]), add_special_tokens=False).input_ids


def probe(n=32):
    prompts, labels, _ = bq.build_gsm8k_prompts(
        bq.Path("/data/sm70_release_datasets/gsm8k"), n, 5
    )
    resps = bq.generate_batch(
        [bq.chat_wrap(x) for x in prompts],
        max_new_tokens=1024,
        temperature=0.0,
        stop=["Question", "Assistant:", "<|separator|>", "<|im_end|>"],
    )
    correct = sum(
        int(bq.gsm8k_answer_value(bq.response_record(r)[0]) == labels[i])
        for i, r in enumerate(resps)
    )
    floods = sum(int(bq.response_record(r)[0].startswith("!!!")) for r in resps)
    return correct, floods


def row():
    resp = bq.generate_one(input_ids=ids, max_new_tokens=128, temperature=0.0)
    text, _, _ = bq.response_record(resp)
    return text.startswith("!!!")


arm = sys.argv[1]
flush()
if arm == "warm":
    probe(12)  # warm the radix cache with short traffic
row()
c, f = probe(32)
print(f"arm={arm}: post-row gsm8k {c}/32 ({c / 32:.1%}) floods={f}", flush=True)
