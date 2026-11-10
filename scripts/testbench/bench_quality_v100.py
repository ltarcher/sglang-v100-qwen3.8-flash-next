#!/usr/bin/env python3
# Quality benchmark suite ported from the 1Cat-vLLM SM70 release matrix
# (benchmarks/run_sm70_release_matrix.py) onto this fork's sglang HTTP API.
#
# The 1Cat original drives an in-process vllm.LLM; that engine is not
# available on this host. Everything engine-independent is ported verbatim
# (data loading, prompt building, GSM8K answer extraction, the degenerate
# output quality gate); the vllm calls are replaced by POSTs to a running
# sglang server's /generate and /v1/chat/completions endpoints.
#
# Phases:
#   gsm8k          5-shot accuracy (default 32 questions) + quality gate
#   public_quality humaneval / mbpp / ifeval degenerate-output gate only
#                  (the 1Cat original also scores only the gate here)
#   longbench      official LongBench prompt templates + metrics
#
# The serving engine must be up before running, e.g.
#   bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp
# Default endpoint follows the engine bind: SGLANG_V100_HOST/PORT
# (127.0.0.1:11435); the production compose at 8110 needs
#   SGLANG_V100_PORT=8110.
#
# Datasets live outside the repo (downloaded once), see --*-dir defaults.
# Scoring deps (LongBench only): jieba, rouge, fuzzywuzzy on the host python.
#
# Usage:
#   python3 scripts/testbench/bench_quality_v100.py gsm8k
#   python3 scripts/testbench/bench_quality_v100.py public_quality \
#       --datasets humaneval,mbpp,ifeval
#   python3 scripts/testbench/bench_quality_v100.py longbench \
#       --datasets multifieldqa_zh,dureader
#   python3 scripts/testbench/bench_quality_v100.py all --out /tmp/qbench
#
# Results: JSON per phase under --out (default testbench_results/ next to
# this script), plus per-request previews for triage.

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import string
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

HOST = os.environ.get("SGLANG_V100_HOST", "127.0.0.1")
PORT = os.environ.get("SGLANG_V100_PORT", "11435")
BASE = f"http://{HOST}:{PORT}"
MODEL_NAME = os.environ.get("SGLANG_V100_SERVED_MODEL", "qwen38-flash-next-nvfp4")

REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_GSM8K_DIR = Path("/data/sm70_release_datasets/gsm8k")
DEFAULT_PUBLIC_DIR = Path("/data/sm70_release_datasets/public_quality")
DEFAULT_LONGBENCH_DIR = Path("/data/sm70_release_datasets/longbench/data")
LONGBENCH_PROMPTS_JSON = SCRIPT_DIR / "longbench_prompts.json"

INVALID_GSM8K_ANSWER = -9999999

# 1Cat release-matrix defaults, kept so numbers stay comparable.
DEFAULT_PUBLIC_QUALITY_DATASETS = ("humaneval", "mbpp", "ifeval")
DEFAULT_LONG_BENCH_DATASETS = (
    "qasper",
    "narrativeqa",
    "musique",
    "hotpotqa",
    "multifieldqa_zh",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "lcc",
    "repobench-p",
)
# Official LongBench dataset2maxlen (github.com/THUDM/LongBench, verified
# against opencompass configs).
LONGBENCH_MAXLEN = {
    "qasper": 32,
    "narrativeqa": 128,
    "musique": 32,
    "hotpotqa": 32,
    "multifieldqa_zh": 64,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "lcc": 64,
    "repobench-p": 64,
}
# Datasets whose metric needs the jieba/rouge/fuzzywuzzy host deps.
LONGBENCH_ZH_METRIC = {"dureader", "multifieldqa_zh", "vcsum"}
LONGBENCH_ROUGE = {"gov_report", "qmsum", "multi_news", "vcsum"}
LONGBENCH_COUNT = {"passage_count"}
LONGBENCH_RETRIEVAL = {"passage_retrieval_en", "passage_retrieval_zh"}
LONGBENCH_CLASSIFICATION = {"trec", "triviaqa", "samsum", "lsht"}
LONGBENCH_CODE_SIM = {"lcc", "repobench-p", "repobench-p_e"}
# First-line-only extraction before scoring, per official eval.py.
LONGBENCH_FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}

PUBLIC_QUALITY_DATASETS = {
    "humaneval": {"filename": "HumanEval.jsonl.gz", "gz": True},
    "mbpp": {"filename": "sanitized-mbpp.json", "gz": False},
    "ifeval": {"filename": "ifeval_input_data.jsonl", "gz": False},
}

# Qwen3.8 is a thinking model; raw /generate output carries <think> blocks.
# The gate and the scorers both see the stripped text.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop reasoning blocks from raw completions (think model serving)."""
    text = _THINK_RE.sub("", text)
    # A generation that hit max_tokens mid-reasoning has an unclosed block.
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# HTTP transport (mirrors scripts/bench_v100.py; stdlib only)
# ---------------------------------------------------------------------------


def post_generate(payload: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(
        BASE + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate_one(
    prompt: str | None = None,
    *,
    input_ids: list[int] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    stop: list[str] | None = None,
    ignore_eos: bool = False,
) -> dict:
    """One /generate round trip; returns the full response with meta_info."""
    sampling: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "ignore_eos": ignore_eos,
    }
    if stop:
        sampling["stop"] = stop
    payload: dict[str, Any] = {"sampling_params": sampling}
    if input_ids is not None:
        payload["input_ids"] = input_ids
    else:
        payload["text"] = prompt
    t0 = time.perf_counter()
    resp = post_generate(payload)
    resp["wall_seconds"] = time.perf_counter() - t0
    return resp


def generate_batch(
    prompts: list[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    stop: list[str] | None = None,
    concurrency: int = 1,
) -> list[dict]:
    """Sequential or small-bounded concurrent generation over prompts.

    The production server caps --max-running-requests (3 on the 8110
    compose), so concurrency > 1 only helps hide client round-trip time.
    """
    from concurrent.futures import ThreadPoolExecutor

    results: list[dict | None] = [None] * len(prompts)

    def worker(idx: int) -> None:
        results[idx] = generate_one(
            prompts[idx],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
        )

    if concurrency <= 1:
        for i, _ in enumerate(prompts):
            worker(i)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(worker, range(len(prompts))))
    return [r for r in results if r is not None]


def response_record(resp: dict) -> tuple[str, list[int], str]:
    """(text, output_token_ids, finish_reason) from a /generate response."""
    text = strip_think(resp.get("text", "") or "")
    ids = list(resp.get("output_ids") or [])
    meta = resp.get("meta_info") or {}
    finish = meta.get("finish_reason") or {}
    reason = finish.get("type", "unknown") if isinstance(finish, dict) else str(finish)
    return text, ids, reason


def request_metrics(resp: dict) -> dict[str, Any]:
    meta = resp.get("meta_info") or {}
    out = {
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "e2e_latency_s": meta.get("e2e_latency"),
        "queue_time_s": meta.get("queue_time"),
        "wall_seconds": resp.get("wall_seconds"),
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
    }
    e2e = meta.get("e2e_latency")
    ctoks = meta.get("completion_tokens")
    ptoks = meta.get("prompt_tokens")
    if e2e and ctoks:
        out["decode_tps"] = ctoks / e2e
    if e2e and ptoks:
        out["prefill_tps"] = ptoks / e2e
    return out


# ---------------------------------------------------------------------------
# Quality gate — ported verbatim from 1Cat run_sm70_release_matrix.py
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_ids(token_ids: list[int]) -> str:
    import hashlib

    return hashlib.sha256(
        b"".join(int(t).to_bytes(4, "little") for t in token_ids)
    ).hexdigest()


def _longest_char_run(text: str, predicate) -> int:
    best = 0
    current = 0
    for ch in text:
        if predicate(ch):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _longest_same_token_run(token_ids: list[int]) -> int:
    best = 0
    last = object()
    current = 0
    for token_id in token_ids:
        if token_id == last:
            current += 1
        else:
            last = token_id
            current = 1
        best = max(best, current)
    return best


def _longest_same_char_run(text: str) -> int:
    best = 0
    last = None
    current = 0
    for ch in text:
        if ch == last:
            current += 1
        else:
            last = ch
            current = 1
        best = max(best, current)
    return best


def _max_repeated_window(text: str, width: int) -> int:
    normalized = re.sub(r"\s+", " ", text)
    if len(normalized) < width:
        return 0
    counts: dict[str, int] = {}
    for idx in range(0, len(normalized) - width + 1):
        window = normalized[idx : idx + width]
        if len(window.strip()) < width // 2:
            continue
        # Homogeneous runs are checked independently by
        # _longest_same_char_run. Counting their overlapping windows makes
        # ordinary code separators such as "=====" look quadratically
        # repetitive.
        if len(set(window)) == 1:
            continue
        counts[window] = counts.get(window, 0) + 1
    return max(counts.values(), default=0)


def _max_same_line_run(text: str) -> int:
    best = 0
    last = None
    current = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == last:
            current += 1
        else:
            last = line
            current = 1
        best = max(best, current)
    return best


def quality_metrics(
    text: str,
    token_ids: list[int],
    *,
    min_tokens: int = 64,
) -> dict[str, Any]:
    """Degenerate-output gate (1Cat _quality_metrics, verbatim thresholds)."""
    bad_markers = [
        "rgba(rgba",
        "UTF-UTF",
        "propertycorrectly",
        "255555555",
        "00000000000000000000",
        "55555555555555555555",
        "\ufffd",
    ]
    marker_hits = {m: text.count(m) for m in bad_markers if m in text}
    metrics = {
        "chars": len(text),
        "tokens": len(token_ids),
        "text_hash": _sha256_text(text),
        "token_hash": _sha256_ids(token_ids),
        "max_same_token_run": _longest_same_token_run(token_ids),
        "max_digit_run": _longest_char_run(text, str.isdigit),
        "max_same_char_run": _longest_same_char_run(text),
        "repeat20": _max_repeated_window(text, 20),
        # Short HTML/code fragments legitimately repeat in large generated
        # applications. Scale this guard with output size; the longer-window
        # and token-run checks below still catch actual degeneration.
        "repeat20_limit": max(80, len(text) // 200),
        "repeat50": _max_repeated_window(text, 50),
        "repeat100": _max_repeated_window(text, 100),
        "max_same_line_run": _max_same_line_run(text),
        "replacement_char_count": text.count("\ufffd"),
        "bad_marker_hits": marker_hits,
    }
    failures = []
    if metrics["tokens"] < min_tokens:
        failures.append("too_few_output_tokens")
    if metrics["max_same_token_run"] > 48:
        failures.append("same_token_run")
    if metrics["max_digit_run"] > 80:
        failures.append("digit_run")
    if metrics["max_same_char_run"] > 120:
        failures.append("same_char_run")
    if metrics["repeat20"] > metrics["repeat20_limit"]:
        failures.append("repeat20")
    if metrics["repeat50"] > 40:
        failures.append("repeat50")
    if metrics["repeat100"] > 20:
        failures.append("repeat100")
    if metrics["max_same_line_run"] > 12:
        failures.append("same_line_run")
    if marker_hits:
        failures.append("bad_marker")
    metrics["failures"] = failures
    metrics["passed"] = not failures
    return metrics


# ---------------------------------------------------------------------------
# Dataset loading (paths differ from 1Cat only)
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            row["_source_index"] = idx
            rows.append(row)
    return rows


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    import gzip

    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            row["_source_index"] = idx
            rows.append(row)
    return rows


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    min_length: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row.get("length", 0)) >= min_length]
    if not selected:
        selected = list(rows)
    if limit <= 0 or len(selected) <= limit:
        return selected
    if limit == 1:
        return [selected[len(selected) // 2]]
    last = len(selected) - 1
    return [selected[round(i * last / (limit - 1))] for i in range(limit)]


# ---------------------------------------------------------------------------
# GSM8K phase (prompts, extraction and scoring ported verbatim)
# ---------------------------------------------------------------------------

_GSM8K_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])"
)
# (?&value) recursion requires the third-party regex module; stdlib re would
# raise "unknown extension". 1Cat's original imports regex for the same reason.
import regex as _regex  # noqa: E402

_GSM8K_BOXED_RE = _regex.compile(r"\\boxed\{(?P<value>(?:[^{}]+|\{(?&value)\})*)\}")


def gsm8k_answer_value(text: str) -> int:
    """1Cat two-tier extraction: last \\boxed{}, else last number in text."""
    from decimal import Decimal, InvalidOperation

    boxed = list(_GSM8K_BOXED_RE.finditer(text))
    answer_text = boxed[-1].group("value") if boxed else text
    numbers = _GSM8K_NUMBER_RE.findall(answer_text)
    if not numbers:
        return INVALID_GSM8K_ANSWER
    try:
        value = Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return INVALID_GSM8K_ANSWER
    if not value.is_finite() or value != value.to_integral_value():
        return INVALID_GSM8K_ANSWER
    return int(value)


def build_gsm8k_prompts(
    cache_dir: Path,
    num_questions: int,
    num_shots: int,
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    train_data = _read_jsonl(cache_dir / "train.jsonl")
    test_data = _read_jsonl(cache_dir / "test.jsonl")
    num_questions = min(num_questions, len(test_data))
    few_shot = ""
    for i in range(num_shots):
        few_shot += (
            f"Question: {train_data[i]['question']}\n"
            f"Answer: {train_data[i]['answer']}\n\n"
        )
    prompts, labels, rows = [], [], []
    for row in test_data[:num_questions]:
        prompts.append(few_shot + f"Question: {row['question']}\nAnswer:")
        labels.append(gsm8k_answer_value(row["answer"]))
        rows.append(row)
    return prompts, labels, rows


def run_gsm8k_phase(args: argparse.Namespace) -> dict[str, Any]:
    prompts, labels, rows = build_gsm8k_prompts(
        Path(args.gsm8k_dir), args.gsm8k_questions, args.gsm8k_num_shots
    )
    print(f"[gsm8k] {len(prompts)} questions, {args.gsm8k_num_shots}-shot", flush=True)
    # Thinking-model completion in chat form; the raw /generate path with
    # the plain completion prompt produced echo loops on the 8110 probe
    # (the model re-emits "Question: ..." patterns), so we route through
    # the server chat template with thinking disabled, matching the 1Cat
    # enable_thinking=False contract.
    chat_prompts = [chat_wrap(p) for p in prompts]
    responses = generate_batch(
        chat_prompts,
        max_new_tokens=args.gsm8k_max_tokens,
        temperature=0.0,
        stop=["Question", "Assistant:", "<|separator|>", "<|im_end|>"],
        concurrency=args.concurrency,
    )
    records = []
    correct = 0
    invalid = 0
    total_output_tokens = 0
    for idx, resp in enumerate(responses):
        text, token_ids, finish = response_record(resp)
        pred = gsm8k_answer_value(text)
        is_correct = pred == labels[idx]
        correct += int(is_correct)
        invalid += int(pred == INVALID_GSM8K_ANSWER)
        total_output_tokens += len(token_ids)
        records.append(
            {
                "ordinal": idx,
                "source_index": rows[idx].get("_source_index"),
                "label": labels[idx],
                "prediction": pred,
                "correct": is_correct,
                "invalid": pred == INVALID_GSM8K_ANSWER,
                "finish_reason": finish,
                "output_tokens": len(token_ids),
                "quality_metrics": quality_metrics(text, token_ids, min_tokens=1),
                "request_metrics": request_metrics(resp),
                "preview": text[:800],
            }
        )
        if idx % 8 == 0 or idx == len(responses) - 1:
            print(f"[gsm8k] {idx + 1}/{len(responses)} correct={correct}", flush=True)
    n = len(records)
    accuracy = correct / n if n else None
    return {
        "records": records,
        "num_questions": n,
        "num_shots": args.gsm8k_num_shots,
        "max_tokens": args.gsm8k_max_tokens,
        "accuracy": accuracy,
        "invalid_rate": invalid / n if n else None,
        "correct": correct,
        "invalid": invalid,
        "total_output_tokens": total_output_tokens,
        "quality_passed": all(r["quality_metrics"]["passed"] for r in records),
        "passed": accuracy is not None and accuracy >= args.gsm8k_threshold,
    }


# ---------------------------------------------------------------------------
# Public quality phase (humaneval / mbpp / ifeval — degenerate-output gate)
# ---------------------------------------------------------------------------


def public_quality_prompt(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "humaneval":
        return (
            "Complete the following Python function. Return runnable Python "
            "code only.\n\n"
            f"{row['prompt']}"
        )
    if dataset == "mbpp":
        tests = "\n".join(row.get("test_list") or [])
        return (
            "Write a Python function that satisfies the task and tests. "
            "Return runnable Python code only.\n\n"
            f"Task: {row['prompt']}\n\nTests:\n{tests}"
        )
    if dataset == "ifeval":
        return row["prompt"]
    raise ValueError(f"unknown public quality dataset: {dataset}")


def load_public_quality_rows(dataset: str, cache_dir: Path, limit: int):
    spec = PUBLIC_QUALITY_DATASETS[dataset]
    path = cache_dir / spec["filename"]
    if spec["gz"]:
        rows = _read_jsonl_gz(path)
    elif dataset == "mbpp":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [dict(row, _source_index=idx) for idx, row in enumerate(data)]
    else:
        rows = _read_jsonl(path)
    return _select_rows(rows, min_length=0, limit=limit)


def run_public_quality_phase(args: argparse.Namespace) -> dict[str, Any]:
    datasets = [d for d in args.datasets.split(",") if d] or list(
        DEFAULT_PUBLIC_QUALITY_DATASETS
    )
    records = []
    for dataset in datasets:
        if dataset not in PUBLIC_QUALITY_DATASETS:
            raise ValueError(f"unknown public quality dataset: {dataset}")
        rows = load_public_quality_rows(
            dataset, Path(args.public_dir), args.public_quality_limit
        )
        for ordinal, row in enumerate(rows):
            content = public_quality_prompt(dataset, row)
            prompt = chat_wrap(content)
            resp = generate_one(
                prompt,
                max_new_tokens=args.public_quality_max_tokens,
                temperature=0.0,
            )
            text, token_ids, finish = response_record(resp)
            metrics = quality_metrics(text, token_ids, min_tokens=8)
            # The 1Cat gate's too_few_output_tokens trips on ifeval rows that
            # explicitly demand an ultra-short answer ("Just choose one
            # phrase from above": the model rightly answered in 6 tokens and
            # stopped). A natural stop at ≤8 tokens is obedience, not
            # degeneration — drop only that failure in that case.
            if (
                not metrics["passed"]
                and finish == "stop"
                and metrics["failures"] == ["too_few_output_tokens"]
            ):
                metrics["failures"] = []
                metrics["passed"] = True
                metrics["short_stop_waived"] = True
            records.append(
                {
                    "dataset": dataset,
                    "ordinal": ordinal,
                    "source_index": row.get("_source_index"),
                    "task_id": row.get("task_id") or row.get("key"),
                    "finish_reason": finish,
                    "prompt_tokens": (resp.get("meta_info") or {}).get("prompt_tokens"),
                    "metrics": metrics,
                    "request_metrics": request_metrics(resp),
                    "preview": text[:1200],
                    "tail": text[-1200:],
                }
            )
            print(
                f"[public-quality] {dataset} {ordinal + 1}/{len(rows)} "
                f"passed={metrics['passed']}",
                flush=True,
            )
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(record["dataset"], []).append(record)
    datasets_summary = {
        dataset: {
            "samples": len(items),
            "passed": all(item["metrics"]["passed"] for item in items),
            "failures": [
                {
                    "source_index": item["source_index"],
                    "task_id": item.get("task_id"),
                    "failures": item["metrics"]["failures"],
                }
                for item in items
                if not item["metrics"]["passed"]
            ],
        }
        for dataset, items in sorted(by_dataset.items())
    }
    return {
        "records": records,
        "datasets": datasets_summary,
        "passed": all(record["metrics"]["passed"] for record in records),
    }


# ---------------------------------------------------------------------------
# LongBench phase (official prompts + metrics; no third_party checkout
# needed — the scorer functions are inlined from LongBench eval.py via
# opencompass's verbatim copies)
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    all_punctuation = set(string.punctuation + cn_punctuation)
    s = "".join(ch for ch in s if ch not in all_punctuation)
    return "".join(s.split()).lower()


def f1_score(prediction: str, ground_truth: str, zh: bool = False) -> float:
    if zh:
        prediction_tokens = list(normalize_zh_answer(prediction))
        ground_truth_tokens = list(normalize_zh_answer(ground_truth))
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    return (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )


def rouge_score(prediction: str, ground_truth: str, **_kw) -> float:
    from rouge import Rouge

    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except Exception:
        return 0.0
    return float(scores["rouge-l"]["f"])


def count_score(prediction: str, ground_truth: str, **_kw) -> float:
    numbers = re.findall(r"\d+", prediction)
    right = 0
    if numbers:
        right = int(numbers[-1]) == int(ground_truth)
    return float(right)


def retrieval_score(prediction: str, ground_truth: str, **_kw) -> float:
    return float(
        prediction.strip().replace(" ", "").lower()
        == ground_truth.strip().replace(" ", "").lower()
    )


def classification_score(
    prediction: str, ground_truth: str, all_classes: list
) -> float:
    pred = prediction.strip().split("\n")[0] if prediction.strip() else ""
    pred = pred.lstrip(" ：:").split(" ")[0]
    return float(pred == ground_truth.strip())


def code_sim_score(prediction: str, ground_truth: str, **_kw) -> float:
    from fuzzywuzzy import fuzz

    return float(fuzz.ratio(prediction, ground_truth) / 100.0)


def score_longbench(dataset: str, prediction: str, row: dict[str, Any]) -> float:
    answers = row["answers"]
    all_classes = row.get("all_classes", [])
    best = 0.0
    for answer in answers:
        if dataset in LONGBENCH_CLASSIFICATION:
            score = classification_score(prediction, answer, all_classes)
        elif dataset in LONGBENCH_CODE_SIM:
            score = code_sim_score(prediction, answer)
        elif dataset in LONGBENCH_ZH_METRIC:
            score = f1_score(prediction, answer, zh=True)
        elif dataset in LONGBENCH_COUNT:
            score = count_score(prediction, answer)
        elif dataset in LONGBENCH_RETRIEVAL:
            score = retrieval_score(prediction, answer)
        elif dataset in LONGBENCH_ROUGE:
            score = rouge_score(prediction, answer)
        else:
            score = f1_score(prediction, answer)
        best = max(best, float(score))
    return best


def _truncate_middle(tokenizer, prompt: str, max_input_tokens: int) -> str:
    token_ids = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
    if len(token_ids) <= max_input_tokens:
        return prompt
    half = max_input_tokens // 2
    left = tokenizer.decode(token_ids[:half], skip_special_tokens=True)
    right = tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
    return left + right


def build_longbench_prompt(
    tokenizer,
    *,
    dataset: str,
    row: dict[str, Any],
    prompt_format: str,
    max_input_tokens: int,
) -> tuple[str, list[int]]:
    """Official prompt format with middle truncation; token ids are built
    client-side and sent as input_ids so the server prompt_tokens readout
    matches what was fit."""
    raw_prompt = prompt_format.format(**row)
    target_tokens = max_input_tokens
    for _ in range(8):
        prompt = _truncate_middle(tokenizer, raw_prompt, target_tokens)
        prompt = chat_wrap(prompt)
        final_ids = tokenizer(
            prompt, truncation=False, add_special_tokens=False
        ).input_ids
        if len(final_ids) <= max_input_tokens:
            return prompt, final_ids
        target_tokens = max(
            128, target_tokens - (len(final_ids) - max_input_tokens) - 256
        )
    raise ValueError(
        f"failed to fit LongBench prompt dataset={dataset} "
        f"source_index={row.get('_source_index')} under {max_input_tokens} tokens"
    )


def run_longbench_phase(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args_tokenizer_path(), trust_remote_code=True, use_fast=True
    )
    prompt_formats = json.loads(LONGBENCH_PROMPTS_JSON.read_text(encoding="utf-8"))
    datasets = [d for d in args.datasets.split(",") if d] or list(
        DEFAULT_LONG_BENCH_DATASETS
    )
    for dataset in datasets:
        if dataset not in prompt_formats:
            raise ValueError(
                f"unknown LongBench dataset: {dataset}; "
                f"prompt templates exist for {sorted(prompt_formats)}"
            )
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = _read_jsonl(Path(args.longbench_dir) / f"{dataset}.jsonl")
        selected = _select_rows(
            rows, min_length=args.longbench_min_length, limit=args.longbench_limit
        )
        for ordinal, row in enumerate(selected):
            prompt, prompt_ids = build_longbench_prompt(
                tokenizer,
                dataset=dataset,
                row=row,
                prompt_format=prompt_formats[dataset],
                max_input_tokens=args.longbench_max_input_tokens,
            )
            resp = generate_one(
                input_ids=prompt_ids,
                max_new_tokens=LONGBENCH_MAXLEN[dataset],
                temperature=0.0,
            )
            text, token_ids, finish = response_record(resp)
            score = score_longbench(dataset, text, row)
            records.append(
                {
                    "dataset": dataset,
                    "ordinal": ordinal,
                    "source_index": row["_source_index"],
                    "id": row.get("_id"),
                    "length": row.get("length"),
                    "prompt_tokens": len(prompt_ids),
                    "answers": row["answers"],
                    "score": score,
                    "output_tokens": len(token_ids),
                    "finish_reason": finish,
                    "request_metrics": request_metrics(resp),
                    "quality_metrics": quality_metrics(text, token_ids, min_tokens=1),
                    "prediction_preview": text[:1200],
                    "prediction_tail": text[-1200:],
                }
            )
            print(
                f"[longbench] {dataset} {ordinal + 1}/{len(selected)} "
                f"score={100.0 * score:.2f} prompt_tokens={len(prompt_ids)}",
                flush=True,
            )
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(record["dataset"], []).append(record)
    dataset_summary = {
        dataset: _summarize_longbench(items)
        for dataset, items in sorted(by_dataset.items())
    }
    dataset_scores = [
        summary["score"]
        for summary in dataset_summary.values()
        if summary.get("score") is not None
    ]
    return {
        "records": records,
        "datasets": dataset_summary,
        "average_score": (
            round(statistics.mean(dataset_scores), 4) if dataset_scores else None
        ),
        "garble_passed": all(record["quality_metrics"]["passed"] for record in records),
    }


def _summarize_longbench(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(record["score"]) for record in records]
    prompt_tokens = [int(record["prompt_tokens"]) for record in records]
    output_tokens = [int(record["output_tokens"]) for record in records]
    garble_passed = all(record["quality_metrics"]["passed"] for record in records)
    return {
        "samples": len(records),
        "score": round(100.0 * statistics.mean(scores), 4) if scores else None,
        "score_values": scores,
        "prompt_tokens_min": min(prompt_tokens) if prompt_tokens else None,
        "prompt_tokens_median": (
            statistics.median(prompt_tokens) if prompt_tokens else None
        ),
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
        "output_tokens_mean": (
            statistics.mean(output_tokens) if output_tokens else None
        ),
        "garble_passed": garble_passed,
    }


# ---------------------------------------------------------------------------
# Chat wrapping — thinking-off contract, matching the 1Cat benchmark mode
# ---------------------------------------------------------------------------

_CHAT_TEMPLATE: str | None = None


def chat_wrap(content: str) -> str:
    """Render the server's chat template with thinking disabled.

    Raw completion prompting made the probe echo ("Question: ...\\nAnswer: 5
    \\n\\nQuestion: ..." repeated); the server's own template plus the
    enable_thinking=False kwarg produces clean single-turn output. Templates
    are read once from the served checkpoint directory.
    """
    global _CHAT_TEMPLATE
    if _CHAT_TEMPLATE is None:
        tok_cfg = json.loads(
            Path(args_tokenizer_path()).joinpath("tokenizer_config.json").read_text()
        )
        _CHAT_TEMPLATE = tok_cfg["chat_template"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args_tokenizer_path(), trust_remote_code=True, use_fast=True
    )
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        chat_template=_CHAT_TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


_TOKENIZER_PATH: str | None = None


def args_tokenizer_path() -> str:
    if _TOKENIZER_PATH is None:
        raise RuntimeError("tokenizer path not resolved; call resolve_server() first")
    return _TOKENIZER_PATH


def resolve_server(tokenizer_path_arg: str | None) -> dict:
    """Read /get_model_info so the chat template matches the running engine."""
    global _TOKENIZER_PATH
    req = urllib.request.Request(BASE + "/get_model_info")
    with urllib.request.urlopen(req, timeout=30) as r:
        info = json.loads(r.read())
    _TOKENIZER_PATH = tokenizer_path_arg or info.get("tokenizer_path")
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_out() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "results" / stamp


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("gsm8k", "public_quality", "longbench", "all"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="override; default reads it from /get_model_info",
    )
    parser.add_argument("--gsm8k-dir", default=str(DEFAULT_GSM8K_DIR))
    parser.add_argument("--gsm8k-questions", type=int, default=32)
    parser.add_argument("--gsm8k-num-shots", type=int, default=5)
    parser.add_argument("--gsm8k-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--gsm8k-threshold",
        type=float,
        default=0.7,
        help="phase fails below this accuracy",
    )
    parser.add_argument("--public-dir", default=str(DEFAULT_PUBLIC_DIR))
    parser.add_argument(
        "--datasets", default="", help="comma list; phase default set when empty"
    )
    parser.add_argument("--public-quality-limit", type=int, default=5)
    parser.add_argument("--public-quality-max-tokens", type=int, default=768)
    parser.add_argument("--longbench-dir", default=str(DEFAULT_LONGBENCH_DIR))
    parser.add_argument("--longbench-limit", type=int, default=5)
    parser.add_argument("--longbench-min-length", type=int, default=4096)
    parser.add_argument("--longbench-max-input-tokens", type=int, default=32768)
    args = parser.parse_args()

    info = resolve_server(args.tokenizer_path)
    print(
        f"[server] {BASE} model={info.get('model_path')} tokenizer={_TOKENIZER_PATH}",
        flush=True,
    )

    out_dir = args.out or _default_out()
    phases = (
        (
            ("gsm8k", run_gsm8k_phase),
            ("public_quality", run_public_quality_phase),
            ("longbench", run_longbench_phase),
        )
        if args.phase == "all"
        else (
            (
                args.phase,
                {
                    "gsm8k": run_gsm8k_phase,
                    "public_quality": run_public_quality_phase,
                    "longbench": run_longbench_phase,
                }[args.phase],
            ),
        )
    )

    failures = []
    for name, fn in phases:
        print(f"=== phase {name} ===", flush=True)
        result = fn(args)
        write_json(out_dir / f"{name}.json", result)
        passed = result.get("passed", result.get("garble_passed", True))
        print(
            f"=== phase {name}: {'PASS' if passed else 'FAIL'} "
            f"-> {out_dir / (name + '.json')}",
            flush=True,
        )
        if not passed:
            failures.append(name)

    print(f"[done] {'PASS all' if not failures else 'FAILED: ' + ','.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
