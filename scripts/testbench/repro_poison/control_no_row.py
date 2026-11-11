# ruff: noqa: E402  (imports follow the flush-heavy diagnostic flow deliberately)
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


def probe(n=12):
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


for t in range(2):
    flush()
    c0, f0 = probe(12)
    c1, f1 = probe(12)
    print(f"trial{t}: first={c0}/12 f={f0} | second={c1}/12 f={f1}", flush=True)
