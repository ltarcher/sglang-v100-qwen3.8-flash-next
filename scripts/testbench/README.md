# testbench — quality benchmark suite

Engine-agnostic quality benchmarks driven over the running sglang server's
HTTP API (stdlib-only transport, mirrors `scripts/bench_v100.py`).

## bench_quality_v100.py

Ported from the 1Cat-vLLM SM70 release matrix
(`benchmarks/run_sm70_release_matrix.py`), replacing the in-process
`vllm.LLM` calls with `/generate` POSTs. All engine-independent logic is
verbatim: GSM8K 5-shot prompt building + two-tier answer extraction
(`\boxed{}` then last number), the degenerate-output quality gate
(same-token-run / digit-run / repeat-window / bad-marker thresholds), row
selection, and the public-quality prompt templates.

Phases:

| phase | what it measures | pass criterion |
| --- | --- | --- |
| `gsm8k` | 5-shot accuracy (default 32 questions, greedy) | accuracy >= `--gsm8k-threshold` (0.7) |
| `public_quality` | humaneval / mbpp / ifeval degenerate-output gate only (same as 1Cat: no code execution) | gate passed on every row |
| `longbench` | official LongBench prompts + metrics (F1 en/zh, rouge-l, count, retrieval, classification, code-sim) | garble gate (report-only scores) |

### Thinking-model handling

Qwen3.8 is a thinking model. Raw completion prompting echoes the few-shot
pattern on this engine, so every prompt goes through the served
checkpoint's chat template with `enable_thinking=False` (the 1Cat
benchmark contract), and `<think>` blocks are stripped from output before
scoring (`strip_think`). Prompts are sent as `input_ids` for LongBench so
`prompt_tokens` matches what was fit under `--longbench-max-input-tokens`.

### One deviation from the 1Cat gate

`too_few_output_tokens` (min 8) is waived when the model stopped
naturally at <= 8 tokens: ifeval rows explicitly demand ultra-short
answers ("Just choose one phrase from above"), and obeying that is not
degeneration. Logged as `short_stop_waived`.

### Server contract

- `SGLANG_V100_HOST` / `SGLANG_V100_PORT` (default 127.0.0.1:11435);
  the production 8110 compose needs `SGLANG_V100_PORT=8110`.
- The server must be up first (`bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp`).
- `/get_model_info` supplies the tokenizer path, so the client chat
  template always matches the running engine.
- Production caps `--max-running-requests 3`; keep `--concurrency` <= 3.
  LongBench 32k-token prompts against `--max-total-tokens 786432`: run
  LongBench phases serially (`--concurrency 1`, the default).
- **Flush the radix cache before every run**:
  `curl -X POST $BASE/flush_cache`. A single 32k-token request poisons
  the Unified Radix Cache (mamba extra_buffer): afterwards every greedy
  short-prompt answer degrades silently (gsm8k 0.969 -> 0.34, '!' floods),
  with textually coherent output. `flush_cache` restores accuracy
  immediately; evidence in `results/poison_evidence/`. This is an engine
  bug under investigation — the bench reports it, it does not cause it.

### Datasets (outside the repo, downloaded once)

`/data/sm70_release_datasets/`:

- `gsm8k/{train,test}.jsonl` — opencompass copy, test verified byte-equal
  to the official GitHub file
- `public_quality/HumanEval.jsonl.gz`, `sanitized-mbpp.json` (JSON array
  format, not the opencompass JSONL), `ifeval_input_data.jsonl`
- `longbench/data/*.jsonl` — official THUDM data.zip, all 34 files

Prompt templates for the 12 default LongBench datasets are inlined in
`longbench_prompts.json` (extracted from opencompass's verbatim copies of
the official `dataset2prompt.json`). `LONGBENCH_MAXLEN` mirrors the
official `dataset2maxlen.json`. Scoring deps (LongBench zh/rouge/fuzzy
only): `pip install jieba rouge fuzzywuzzy` on the host python.

### Usage

```bash
SGLANG_V100_PORT=8110 python3 scripts/testbench/bench_quality_v100.py gsm8k
SGLANG_V100_PORT=8110 python3 scripts/testbench/bench_quality_v100.py public_quality
SGLANG_V100_PORT=8110 python3 scripts/testbench/bench_quality_v100.py longbench \
    --datasets multifieldqa_zh,dureader --longbench-limit 2
SGLANG_V100_PORT=8110 python3 scripts/testbench/bench_quality_v100.py all
```

Results land under `scripts/testbench/results/<timestamp>/<phase>.json`
with per-request previews for triage.

### Baseline (Qwen3.8-Flash-Next NVFP4, 4x V100, 8110 production compose, 2026-11-10)

| phase | result |
| --- | --- |
| gsm8k 32q | accuracy 0.96875, invalid 0, gate PASS, spec accept 3.22 |
| public_quality 5x3 | gate PASS (1 ifeval short-stop waiver) |
| longbench zh 2x2 | multifieldqa_zh 80.2, dureader 27.9, garble PASS |
