# ruff: noqa: E402  (imports follow the flush-heavy diagnostic flow deliberately)
import json
import importlib.util
import urllib.request

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


def probe(n):
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


flush()
probe(12)
payload = {
    "input_ids": ids,
    "sampling_params": {"max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0},
    "cache_salt": "isolate32k",
}
import urllib.request as _u
import json as _j

req = _u.Request(
    bq.BASE + "/generate",
    data=_j.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with _u.urlopen(req, timeout=3600) as r:
    _j.loads(r.read())
c, f = probe(32)
print(f"row with cache_salt: gsm8k {c}/32 ({c / 32:.1%}) floods={f}", flush=True)
