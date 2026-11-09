#!/usr/bin/env python
"""Controlled A/B harness for SGLANG_SM70_QSA_GROUPED_PREFILL output quality.

Plan and gates: docs/v100/PREFILL_LEVERS_PLAN.md §8. Three boots (A1 flag=0,
A2 flag=0 again, B flag=1), one prompt matrix per boot sent serially (bs=1,
temperature=0, fixed max_new_tokens, ignore_eos) with top-2 output logprobs,
so any flag-induced divergence is measured against the boot-to-boot noise
floor (A1 vs A2) instead of a single run.

Usage (engine must be idle; each arm = one boot + one command):
  python scripts/ab_qsa_grouped_quality_v100.py run --arm A1 --base http://127.0.0.1:8110 --out /tmp/ab_grouped
  python scripts/ab_qsa_grouped_quality_v100.py run --arm A2 --base http://127.0.0.1:8110 --out /tmp/ab_grouped
  python scripts/ab_qsa_grouped_quality_v100.py run --arm B  --base http://127.0.0.1:8110 --out /tmp/ab_grouped
  python scripts/ab_qsa_grouped_quality_v100.py analyze --out /tmp/ab_grouped --arms A1 A2 B

`run` also re-sends a subset of tags in the same boot (G1 determinism gate)
after /flush_cache. Long prompts are sent as raw input_ids (exact token
counts, no chat template); short tasks are sent as text.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

MODEL_PATH = "/data/models/nvidiaQwen3.8-Flash-Next-NVFP4"
LENGTHS = [1024, 4096, 16384, 32768, 65536, 131072]
NEAR_TIE_MARGIN = 1e-3
REQUEST_TIMEOUT = 900

# Deterministic filler material (needle haystack). Short sentences with varied
# digit/content so padding is not a degenerate repeated single token.
_FILLER = [
    "仓库东侧的货架编号从 {} 开始递增,每排之间留有半米通道。",
    "值班记录显示 {} 号夜间湿度略高于往月平均水平。",
    "运输单 {} 已由调度台盖章,等待次日清晨装车。",
    "设备巡检表中 {} 项被标注为需要更换密封圈。",
    "会议纪要第 {} 条提到走廊照明将在月底前完成改造。",
    "月度报表第 {} 行的差额已由财务复核确认无误。",
    "司机 {} 在停车场入口登记了车牌和进场时间。",
    "样本 {} 的检测报告将在三个工作日内发送到档案室。",
    "风机的第 {} 号传感器读数每隔十分钟上传一次。",
    "客服工单 {} 的处理时限因节假日顺延了一天。",
]


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _filler_ids(tok, target_tokens: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    ids: list[int] = []
    while len(ids) < target_tokens:
        sentence = _FILLER[rng.randrange(len(_FILLER))].format(rng.randrange(10000))
        ids.extend(tok.encode(sentence, add_special_tokens=False))
    return ids[:target_tokens]


def _filler_text(target_tokens: int, seed: int) -> str:
    """~target_tokens of filler text (same generator as _filler_ids)."""
    rng = random.Random(seed)
    out: list[str] = []
    n = 0
    while n < target_tokens:
        s = _FILLER[rng.randrange(len(_FILLER))].format(rng.randrange(10000))
        out.append(s)
        n += len(s) // 2  # ~2 chars/token for CJK-heavy filler
    return "".join(out)


def _needle_prompt_ids(tok, target: int, seed: int):
    """Haystack of `target` tokens with a key ID needle at ~40%, QA suffix."""
    rng = random.Random(seed)
    key = f"FLK-{rng.randrange(10000, 99999)}"
    ids = _filler_ids(tok, target, seed)
    cut = int(len(ids) * 0.4)
    needle = tok.encode(f"(备注:本区备份密钥 ID 为 {key}。)", add_special_tokens=False)
    ids = ids[:cut] + needle + ids[cut : target - len(needle)]
    q = tok.encode(
        "\n\n问:上文提到的备份密钥 ID 是什么?只回答 ID 本身,不要多余文字。\n答:",
        add_special_tokens=False,
    )
    return key, ids + q


_MATH_CTX = [
    (1287, 463),
    (5093, 218),
    (7645, 382),
    (3148, 927),
    (8521, 674),
    (9164, 235),
]
_MATH5 = [
    (
        "一个水箱每分钟进水 14 升,每分钟漏水 6 升,水箱容量 240 升,从空开始多少分钟后满?",
        "30",
    ),
    ("计算 37 的平方减去 19 的平方。", "1008"),
    ("把 7/8 化成小数,保留四位。", "0.8750"),
    ("一件商品先涨价 25% 再打八折,现价相对原价变化百分之几?", "0%"),
    ("等差数列首项 3 公差 7,前 20 项和是多少?", "1390"),
]
_CODE5 = [
    "用 Python 写一个函数,判断字符串是否是合法的 IPv4 地址,并给出两个测试样例。",
    "用 Python 实现一个 LRU 缓存类,要求 get/put 都是 O(1),并解释设计要点。",
    "用 Python 写一个函数,把驼峰命名转成下划线命名,处理连续大写的情况。",
    "用 Python 实现归并排序,并统计比较次数,给出随机数组的运行示例。",
    "用 Python 写一个函数,解析简单 CSV 行并处理引号内逗号,给出示例。",
]
_ZH5 = [
    "请把下面这句话改写成更正式的通知语气:这事儿得赶紧办,别拖了。",
    "请给'电梯困人自救'写一段 200 字以内的安全提示,分三条列出。",
    "请指出下面句子中的语病并改正:通过这次活动,使我们认识到团队合作的重要性。",
    "请用'尽管、然而、因此'三个词写一段逻辑通顺的短文,主题是城市夜间经济。",
    "请把这句话翻译成书面语:这个方案我们回头再看看,估计得改不少地方。",
]
_AGENT5 = [
    "你是一个运维助手。日志显示某服务每分钟重启一次,请列出你会按顺序执行的 5 个排查步骤,每步给出具体命令或检查点。",
    "你是一个数据分析助手。给你一个月销数据目录(30 个 CSV),请设计一个流程找出销量骤降的品类,说明每步的输入输出。",
    "你是一个代码评审助手。请列出评审一个新加的 REST 端点时必查的 6 项,每项说明要防的具体问题。",
    "你是一个发布助手。请为一次灰度发布设计回滚预案,包含触发条件、执行命令、验证手段三部分。",
    "你是一个排障助手。用户反馈页面偶发白屏,请列出从客户端到服务端的 5 个定位手段,并说明各自的判读依据。",
]

# Canary task (2026-11-08): the WorkBuddy "pelican rides a bicycle" prompt is
# the user's de-facto quality gate -- structural breakage here (missing
# animateTransform, malformed SVG) is how the anomaly is judged in practice.
# analyze() checks structural markers automatically and dumps the HTML per
# arm for a manual browser render.
_PELICAN_PROMPT = (
    "创建一个完整的 HTML 文件(bicyle10088.html),其中包含一个 SVG 动画:"
    "主题是鹈鹕骑自行车。\n"
    "SVG 需包含:鹈鹕的头部、身体、翅膀、双腿,自行车的两个车轮、车架、踏板。\n"
    "车轮需使用 animateTransform 实现连续旋转;鹈鹕腿部使用同步的上下踩踏动画;\n"
    "身体略微前后摆动以模拟骑行动作。\n"
    "要求结构清晰、可直接在浏览器运行,不依赖外部脚本。不要参考当前目录的任何东西。"
)
_SVG_MARKERS = ["<!DOCTYPE html", "<svg", "animateTransform", "</svg>", "</html>"]


def build_tasks(tok) -> list[dict]:
    """Full matrix: 21 base tasks + 2 per length bucket = 33.

    v2: every task goes through the chat template. Raw continuations were the
    v1 flaw -- without the template the model falls into degenerate completion
    attractors ("百度知道" loops) whose boundary tokens (`\\n\\n` vs
    `assistant`) are genuine near-ties, so run-to-run kernel noise flips them
    constantly and poisons both the flip-rate and the answer checks. Short
    answer tasks use enable_thinking=False; the pelican canary keeps thinking
    on (the real WorkBuddy flow).
    """

    def templated(tag, typ, content, max_new, expect=None, thinking=False):
        messages = [{"role": "user", "content": content}]
        try:
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:  # template without the kwarg
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return dict(tag=tag, type=typ, prompt=text, max_new=max_new, expect=expect)

    tasks: list[dict] = []
    for i, (q, ans) in enumerate(_MATH5):
        tasks.append(templated(f"math{i}", "math", q, 64, ans))
    for i, q in enumerate(_CODE5):
        tasks.append(templated(f"code{i}", "code", q, 512))
    for i, q in enumerate(_ZH5):
        tasks.append(templated(f"zh{i}", "zh", q, 256))
    for i, q in enumerate(_AGENT5):
        tasks.append(templated(f"agent{i}", "agent", q, 768))
    tasks.append(templated("pelican_svg", "svg", _PELICAN_PROMPT, 6144, thinking=True))
    for L in LENGTHS:
        seed = L
        key, ids = _needle_prompt_ids(tok, L, seed)
        doc = tok.decode(ids)
        tasks.append(templated(f"needle_{L}", "needle", doc, 24, key))
        a, b = _MATH_CTX[LENGTHS.index(L)]
        q = f"忽略上文与题目无关的内容。请直接计算 {a} × {b},只给出数字结果。"
        # Arms must send byte-identical prompts; the haystack sits at ~L
        # tokens of doc plus the question (v1's exact-length trimming does not
        # survive the template and does not need to).
        tasks.append(
            templated(
                f"math_{L}",
                "math",
                _filler_text(L, seed + 1) + "\n\n" + q,
                64,
                str(a * b),
            )
        )
    return tasks


def _generate(base: str, payload: dict) -> dict:
    r = requests.post(f"{base}/generate", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _extract(resp: dict) -> dict:
    meta = resp.get("meta_info", {})
    ids, margins = [], []
    top = meta.get("output_top_logprobs") or []
    for entry in top:
        lps = [e[0] for e in entry]
        ids.append(entry[0][1])
        margins.append([lps[0], lps[1] if len(lps) > 1 else None])
    if not ids:
        # No logprobs requested/metadata missing: ids recovered offline.
        pass
    return dict(
        text=resp.get("text", ""),
        ids=ids,
        margins=margins,
        prompt_tokens=meta.get("prompt_tokens"),
        completion_tokens=meta.get("completion_tokens"),
    )


def _thinking_len(text: str) -> int:
    if "<think>" not in text:
        return 0
    return text.split("<think>", 1)[1].split("</think>", 1)[0].count(" ")


def run(args) -> int:
    tok = _tokenizer()
    tasks = build_tasks(tok)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / f"arm_{args.arm}.jsonl"
    if args.fresh and rec_path.exists():
        rec_path.unlink()
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["tag"])

    # JIT warmup: the grouped kernel compiles per shape bucket on first hit;
    # keep it out of the statistics.
    payload = dict(
        input_ids=_filler_ids(tok, 8192, 7),
        sampling_params=dict(temperature=0, max_new_tokens=8, ignore_eos=True),
        return_logprob=False,
    )
    print("[warmup] 8k prefill ...", flush=True)
    _generate(args.base, payload)

    # Cold-prefill baseline: flush so every task (and every arm) runs the full
    # chunked prefill instead of a cached-prefix extend, which uses a different
    # kernel path and would break bit-level comparison across arms.
    requests.post(f"{args.base}/flush_cache", timeout=60)

    def send(task: dict, tag_suffix: str = "") -> None:
        tag = task["tag"] + tag_suffix
        payload = dict(
            sampling_params=dict(
                temperature=0, max_new_tokens=task["max_new"], ignore_eos=True
            ),
            return_logprob=True,
            top_logprobs_num=2,
            logprob_start_len=-1,
        )
        if "input_ids" in task:
            payload["input_ids"] = task["input_ids"]
        else:
            payload["text"] = task["prompt"]
        t0 = time.time()
        try:
            resp = _generate(args.base, payload)
            rec = _extract(resp)
        except Exception as e:  # noqa: BLE001 - record and continue the sweep
            rec = dict(
                text="",
                ids=[],
                margins=[],
                error=repr(e),
                prompt_tokens=None,
                completion_tokens=None,
            )
        rec.update(
            tag=tag,
            type=task["type"],
            max_new=task["max_new"],
            expect=task.get("expect"),
            wall_s=round(time.time() - t0, 2),
        )
        with rec_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[{args.arm}] {tag}: {rec['completion_tokens']} tok, {rec['wall_s']}s",
            flush=True,
        )

    for task in tasks:
        if task["tag"] not in done:
            send(task)

    # G1 gate: same-boot rerun after cache flush must be bit-identical.
    requests.post(f"{args.base}/flush_cache", timeout=60)
    g1_tags = [t["tag"] for t in tasks if t["type"] in ("math", "needle")][:8]
    g1_tags += ["code0", "agent0", "pelican_svg"]
    for tag in g1_tags:
        task = next(t for t in tasks if t["tag"] == tag)
        send(task, tag_suffix="_rep")
    print(f"[{args.arm}] done -> {rec_path}", flush=True)
    return 0


def _load_arms(out: Path, arms: list[str]) -> dict[str, dict[str, dict]]:
    loaded: dict[str, dict[str, dict]] = {}
    for arm in arms:
        recs: dict[str, dict] = {}
        for line in (out / f"arm_{arm}.jsonl").read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                recs[rec["tag"]] = rec
        loaded[arm] = recs
    return loaded


def _first_div(a: dict, b: dict):
    """Return (divergence index or None, margin at divergence from b)."""
    if a.get("ids") and b.get("ids"):
        for i, (x, y) in enumerate(zip(a["ids"], b["ids"])):
            if x != y:
                return i, (b["margins"][i] if i < len(b.get("margins", [])) else None)
        if len(a["ids"]) != len(b["ids"]):
            return min(len(a["ids"]), len(b["ids"])), None
        return None, None
    # Fallback: character-level text diff.
    ta, tb = a.get("text", ""), b.get("text", "")
    if ta == tb:
        return None, None
    i = next(
        (i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb))
    )
    return -i, None  # negative = char offset (ids missing)


def _cmp(a: dict, b: dict) -> dict:
    idx, margin = _first_div(a, b)
    flip = idx is not None
    near = None
    if flip and margin and margin[1] is not None:
        near = (margin[0] - margin[1]) < NEAR_TIE_MARGIN
    return dict(flip=flip, idx=idx, margin=margin, near_tie=near)


def analyze(args) -> int:
    out = Path(args.out)
    arms = args.arms
    loaded = _load_arms(out, arms)
    tags = sorted(k for k in loaded[arms[0]] if not k.endswith("_rep"))
    rows, red_flags = [], []
    pair_a = [arms[0], arms[1]]  # boot-to-boot noise floor pair

    # G1: intra-boot determinism.
    g1_fail = []
    for arm in arms:
        for tag, rec in loaded[arm].items():
            if tag.endswith("_rep"):
                base_rec = loaded[arm].get(tag[:-4])
                if base_rec is None:
                    continue
                c = _cmp(base_rec, rec)
                if c["flip"]:
                    g1_fail.append(f"{arm}/{tag} idx={c['idx']}")
    print(
        f"\n== G1 same-boot determinism: "
        f"{'FAIL ' + str(g1_fail) if g1_fail else 'PASS (all reps bit-identical)'}"
    )

    def fmt_margin(m):
        if not m or m[1] is None:
            return "-"
        return f"{m[0] - m[1]:.2e}"

    print(
        f"\n{'tag':<16} {'len':>7} {'A-floor':>8} {'B flip':>7} "
        f"{'div idx':>8} {'margin':>9} {'near-tie':>8}"
    )
    for tag in tags:
        rec0 = loaded[arms[0]][tag]
        floor = _cmp(loaded[pair_a[0]][tag], loaded[pair_a[1]][tag])["flip"]
        bflips = []
        for arm in arms[2:]:
            c = _cmp(loaded[arms[0]][tag], loaded[arm][tag])
            if c["flip"]:
                bflips.append((arm, c))
        flag = ""
        for arm, c in bflips:
            if c["near_tie"] is False:
                flag = "  <-- RED: far-margin flip (systematic?)"
                red_flags.append(tag)
            if floor:
                flag += "  (note: floor also flipped)"
        mark = "Y" if bflips else "."
        rows.append((tag, rec0.get("prompt_tokens"), floor, bflips))
        print(
            f"{tag:<16} {str(rec0.get('prompt_tokens')):>7} "
            f"{'Y' if floor else '.':>8} {mark:>7} "
            f"{str(bflips[0][1]['idx'] if bflips else '-'):>8} "
            f"{fmt_margin(bflips[0][1]['margin'] if bflips else None):>9} "
            f"{str(bflips[0][1]['near_tie']) if bflips else '-':>8}{flag}"
        )

    # G3: task-level correctness on the tagged-answer subsets.
    print("\n== G3 expected-answer hits (per arm) ==")
    for arm in arms:
        hits, total = 0, 0
        for tag, rec in loaded[arm].items():
            if tag.endswith("_rep") or rec.get("expect") is None:
                continue
            total += 1
            if rec.get("expect") and rec["expect"] in rec.get("text", ""):
                hits += 1
        print(f"  {arm}: {hits}/{total}")

    # Canary structural gate: the pelican SVG must carry the required markers;
    # dump per-arm HTML for a manual browser render (the user's real verdict).
    print("\n== G3-canary pelican_svg structural markers ==")
    for arm in arms:
        rec = loaded[arm].get("pelican_svg")
        if rec is None:
            print(f"  {arm}: MISSING")
            continue
        text = rec.get("text", "")
        missing = [m for m in _SVG_MARKERS if m not in text]
        (out / f"pelican_{arm}.html").write_text(text, encoding="utf-8")
        verdict = "OK" if not missing else f"MISSING {missing}"
        print(f"  {arm}: {verdict} -> {out}/pelican_{arm}.html (chars={len(text)})")

    # Thinking-length distribution shift (over-thinking signal).
    print("\n== thinking-ish length (chars, tasks with <think>) ==")
    for arm in arms:
        lens = [
            _thinking_len(r.get("text", ""))
            for r in loaded[arm].values()
            if not r["tag"].endswith("_rep") and "<think>" in r.get("text", "")
        ]
        print(f"  {arm}: n={len(lens)} {'sum=' + str(sum(lens)) if lens else ''}")

    n_b_flip = sum(1 for _, _, _, bf in rows if bf)
    print(
        f"\n== verdict ==\n"
        f"B-arm flipped tags: {n_b_flip}/{len(tags)}; "
        f"floor flipped tags: {sum(1 for _, _, fl, _ in rows if fl)}; "
        f"red flags: {red_flags or 'none'}\n"
        "Gates: PASS only if G1 clean, B flips do not exceed the A-floor "
        "rate, every B flip is near-tie, G3 hits equal, thinking lengths "
        "stable. Any red flag -> do NOT re-enable the kernel; open a bug.\n"
        "Final call needs the manual pelican render: pelican_<arm>.html in "
        "the out dir must ride correctly in a browser (user's real gate)."
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--arm", required=True, help="arm name, e.g. A1/A2/B")
    pr.add_argument("--base", default="http://127.0.0.1:8110")
    pr.add_argument("--out", default="/tmp/ab_grouped")
    pr.add_argument(
        "--fresh",
        action="store_true",
        help="truncate this arm's JSONL instead of resuming",
    )
    pa = sub.add_parser("analyze")
    pa.add_argument("--out", default="/tmp/ab_grouped")
    pa.add_argument(
        "--arms",
        nargs="+",
        default=["A1", "A2", "B"],
        help="first two arms form the boot-to-boot floor pair",
    )
    args = p.parse_args()
    return run(args) if args.cmd == "run" else analyze(args)


if __name__ == "__main__":
    sys.exit(main())
