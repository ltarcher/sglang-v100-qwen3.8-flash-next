#!/usr/bin/env bash
# Bit-exact oracle: pinned vs staged PLE backend through the Phase 2
# compute-ahead decode gate (docs/v100/PLE_NVME_TIER_DESIGN.md §11.9 steps 3-4).
#
# Runs the SAME Qwen3.8-Flash-Next-NVFP4 engine twice (4x V100, TP4, no
# hicache) inside the sglang-v100-unified image with the working tree mounted
# at /w (PYTHONPATH=/w/python), sending an identical request sequence to both:
#   warmup 1 req -> 3 sequential reqs (decode bs=1 bucket)
#                  -> 3 staggered reqs (0.6 s apart: each prefill runs solo so
#                     prefill batching is admission-race-free, while decode
#                     windows overlap and walk the padded buckets 1->2->3,
#                     then staggered finishes walk 3->2->1)
# and compares every response's output token ids AND per-token logprobs
# bit-exactly between the pinned and the staged run. The two backends share
# every kernel; the only difference is where the PLE gather reads its rows
# (pinned table vs gate-filled pinned staging), so any token/logprob delta
# means the staged decode-gate path changed numerics.
#
# ORACLE_MTP=1 runs the production MTP config instead (EAGLE 3 steps / topk 1
# / 4 draft tokens): the decode graphs capture TARGET_VERIFY and every decode
# round flows through the gate's verify prep (§11.9 step 4). The staged boot
# additionally runs with SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG=1, and both boots
# are scanned: any "gate ids != in-graph ids" line fails the oracle, and the
# per-boot mean MTP accept length is reported (staged must equal pinned —
# bit-exact outputs already imply it).
#
# Usage (repo root, target machine, GPUs idle):
#   bash scripts/oracle_ple_staged_v100.sh [MODEL_DIR] [OUT_DIR]
# Env overrides:
#   ORACLE_IMAGE   (default sglang-v100-unified:latest)
#   ORACLE_PORT    (default 11436; NOT the production 11435/8110)
#   ORACLE_PLE_DIR (default /data/sglang-ple-staged; staged table dump,
#                  ~51 GB, first staged boot materializes it)
#   ORACLE_MTP     (default 0; 1 = production MTP TARGET_VERIFY oracle)
#   ORACLE_STAGED_AHEAD (default 0; >0 = chunk-ahead prefetch depth
#                  §11.9.5 on the staged arm: bit-exactness then also
#                  covers the speculative staging path)
# Stop the production engine first: one engine per machine, and the oracle
# binds all 4 GPUs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-/data/models/nvidiaQwen3.8-Flash-Next-NVFP4}"
OUT="${2:-/tmp/ple_staged_oracle_$(date +%m%d%H%M%S)}"
IMAGE="${ORACLE_IMAGE:-sglang-v100-unified:latest}"
PORT="${ORACLE_PORT:-11436}"
PLE_DIR="${ORACLE_PLE_DIR:-/data/sglang-ple-staged}"
MTP="${ORACLE_MTP:-0}"
AHEAD="${ORACLE_STAGED_AHEAD:-0}"
LONGCASE="${ORACLE_LONG_PROMPT:-0}"
[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }
mkdir -p "$OUT" "$PLE_DIR"

log() { echo "[oracle $(date +%H:%M:%S)] $*"; }

run_server() { # $1 = backend (pinned|staged); starts + waits for health
  local backend="$1"
  local name="oracle-ple-$backend"
  local -a extra=()
  [[ "$backend" == staged ]] && extra=(--ple-offload-backend staged)
  if [[ "$MTP" == 1 ]]; then
    # Production MTP config (docker-compose-sglang-v100-mtp-8110.yaml):
    # verify graphs capture TARGET_VERIFY at req width 4.
    extra+=(
      --speculative-algorithm EAGLE
      --speculative-draft-model-path /data/models/nvfp4
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
    )
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true
  local -a extra_env=()
  if [[ "$backend" == staged ]]; then
    # Mismatch self-check on the staged boot only: the gate compares its own
    # ids against the in-graph ids each replay (needs the debug buffer, so
    # only wired when the gate exists). Any mismatch line fails the oracle.
    # REUSE=1 skips the 11.9 GiB/rank table rewrite when the on-disk header
    # matches (files persist across boots on /data) — without it the 4-rank
    # write takes ~5.5 min and trips the 480 s unbalanced-loading barrier
    # (UNBALANCED_MODEL_LOADING_TIMEOUT_S, load_model_utils.py). First boot
    # (no on-disk files) must write, so also raise that barrier via the
    # env override added in §11.9.2.
    extra_env=(
      -e SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG=1
      -e SGLANG_QWEN4_PLE_STAGED_REUSE=1
      -e SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT=1800
    )
    if [[ "$AHEAD" != 0 ]]; then
      extra_env+=(-e "SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD=$AHEAD")
    fi
  fi
  # Deterministic CUDA for the long_prompt non-determinism investigation
  extra_env+=(-e SGLANG_FORCE_DETERMINISTIC="${SGLANG_FORCE_DETERMINISTIC:-0}")
  log "starting $backend server on :$PORT (logs: $OUT/server_$backend.log)"
  local attempt
  for attempt in 1 2 3; do
    docker run -d --name "$name" --gpus all --ipc host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p "$PORT:$PORT" \
    -v "$MODEL:/data/models/nvfp4:ro" \
    -v "$REPO:/w:ro" \
    -v sglang-v100-jit:/root/sglang-v100-jit \
    -v "$PLE_DIR:/root/.cache/sglang/ple_staged" \
    -e NCCL_P2P_LEVEL=PXB \
    -e SGLANG_CUSTOM_AR_ALLOW_PCIE=1 \
    -e SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
    -e SGLANG_SM70_FORCE_FP16=1 \
    -e SGLANG_SM70_DENSE_GEMV=1 \
    -e SGLANG_SM70_QWEN_FUSIONS=1 \
    -e SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192 \
    -e SGLANG_SM70_QSA_GROUPED_PREFILL=0 \
    -e SGLANG_MIXED_PREFILL_STEP_LATENCY_MS=0 \
    -e SGLANG_MAMBA_CONV_DTYPE=float16 \
    -e SGLANG_MAMBA_SSM_DTYPE=float16 \
    -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
    -e SGLANG_NUMA_BIND_V2=0 \
    -e SGLANG_FORCE_CPU_IMAGE_PREPROCESSING=1 \
    -e SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP=1 \
    -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e HF_HUB_OFFLINE=1 \
    -e SGLANG_V100_KV_HOST_TIER=1 \
    -e SGLANG_V100_KV_HEAD_LAYOUT=1 \
    -e SGLANG_V100_KV_TIER_PHYS_TOKENS=786432 \
    -e SGLANG_V100_DECODE_CUDA_BUILD_DIR=/root/sglang-v100-jit/ext/sm70_longctx_decode \
    -e SGLANG_V100_NVFP4_MOE_BUILD_DIR=/root/sglang-v100-jit/ext/sm70_nvfp4_moe \
    -e SGLANG_JIT_CACHE_DIR=/root/sglang-v100-jit/jit \
    -e PYTHONPATH=/w/python \
    "${extra_env[@]}" \
    "$IMAGE" /opt/venv/bin/python -m sglang.launch_server \
      --trust-remote-code \
      --model-path /data/models/nvfp4 \
      --served-model-name qwen38next-nvfp4 \
      --dtype float16 \
      --quantization modelopt_fp4 \
      --attention-backend tilelang_fa_v100 \
      --linear-attn-prefill-backend tilelang \
      --linear-attn-decode-backend triton \
      --kv-cache-dtype auto \
      --tensor-parallel-size 4 \
      --host 0.0.0.0 --port "$PORT" \
      --mem-fraction-static 0.90 \
      --context-length 262144 \
      --max-total-tokens 786432 \
      --max-running-requests 3 \
      --max-mamba-cache-size 20 \
      --mamba-radix-cache-strategy extra_buffer \
      --mamba-full-memory-ratio 0.2 \
      --chunked-prefill-size 4096 \
      --max-prefill-tokens 4096 \
      --cuda-graph-backend-prefill disabled \
      --cuda-graph-max-bs-decode 3 \
      --cuda-graph-bs-decode 1 2 3 \
      --ple-offload-embedding \
      --warmups sampling \
      --sleep-on-idle \
      "${extra[@]}" >>"$OUT/docker_run_$backend.err" 2>&1 && break
    # Transient failure (typically: the previous arm's 11436 bind not yet
    # released after docker rm -f). Drop the half-created container, wait,
    # retry; the real server log streams via docker logs below either way.
    log "docker run failed for $backend (attempt $attempt/3)" >&2
    tail -5 "$OUT/docker_run_$backend.err" >&2
    docker rm -f "$name" >/dev/null 2>&1 || true
    if [[ "$attempt" == 3 ]]; then
      return 1
    fi
    sleep 5
  done
  # `docker run -d` detaches, so the container's stdout never reaches the
  # redirect above; stream the real server log via `docker logs -f` instead.
  docker logs -f "$name" >"$OUT/server_$backend.log" 2>&1 &
  LOGS_PID=$!
  log "waiting for $backend /health (weight load + graph capture + \
staged materialization can take a while)..."
  for _ in $(seq 1 720); do
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      log "$backend server healthy"
      return 0
    fi
    if ! docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true; then
      log "$backend container died; last log lines:" >&2
      sleep 2; kill "$LOGS_PID" 2>/dev/null || true
      tail -40 "$OUT/server_$backend.log" >&2
      return 1
    fi
    sleep 10
  done
  kill "$LOGS_PID" 2>/dev/null || true
  log "timed out waiting for $backend /health" >&2
  return 1
}

collect() { # $1 = backend, $2 = outfile; identical request sequence per run
  python3 - "$PORT" "$2" "$AHEAD" "$LONGCASE" <<'PYEOF'
import json, sys, time, urllib.request

port, out_path, ahead, long_case = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
url = f"http://127.0.0.1:{port}/generate"

PROMPTS = [
    "The industrial revolution began in Britain in the late eighteenth "
    "century. Textile manufacturing was transformed by inventions such as "
    "the spinning jenny and the water frame. Steam power soon spread to "
    "mining, transportation and agriculture, reshaping cities across "
    "Europe. Explain how",
    "A hash table maps keys to buckets using a hash function. Collisions "
    "occur when two distinct keys land in the same bucket, and they are "
    "resolved either by chaining entries in a linked list or by probing "
    "for the next open slot. Load factor and resize policy determine "
    "performance. Continue describing",
    "The mitochondrion is the powerhouse of the cell, generating ATP "
    "through oxidative phosphorylation. Electrons harvested from glucose "
    "travel down the respiratory chain, pumping protons across the inner "
    "membrane. ATP synthase then converts the proton gradient into "
    "chemical energy. Describe what happens next in",
]

def gen(prompt, max_new):
    body = json.dumps({
        "text": prompt,
        "sampling_params": {
            "temperature": 0, "max_new_tokens": max_new,
            "ignore_eos": True, "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)

results = {}
# 1) warmup (identical in both runs: warms JIT + radix state deterministically)
results["warmup"] = gen("Hello, tell me a one-sentence fact.", 8)
# 2) sequential: bs=1 bucket, full 262k-idle engine, race-free
for i, (p, m) in enumerate(zip(PROMPTS, [24, 24, 24])):
    results[f"seq_{i}"] = gen(p, m)
# 3) staggered overlap: solo prefills (deterministic batching), decode
#    windows overlap so the padded decode buckets are exercised, and the
#    staggered lengths walk the decode bs 3 -> 2 -> 1
#    (a simultaneous burst is NOT bit-comparable: concurrent prefill
#    admission races change the extend batch composition across boots)
results["ov_0"] = gen(PROMPTS[0], 96)
time.sleep(0.6)
results["ov_1"] = gen(PROMPTS[1], 72)
time.sleep(0.6)
results["ov_2"] = gen(PROMPTS[2], 48)
# 4) long multi-chunk prefill (chunk-ahead arms only): the short prompts above
#    fit in one chunked_prefill_size=4096 chunk, so the §11.9.5 speculative
#    staging never engages; this ~30k-token prompt crosses ~8 chunk
#    boundaries and walks the ahead FIFO (submit i+1..i+d while i computes,
#    bit-exact consume check on every chunk). Deterministic repeated text so
#    both boots send byte-identical input.
if ahead != "0" or long_case == "1":
    para = ("The canal barge carried coal downriver at a steady pace, "
            "passing locks and mills whose wheels turned in the current. "
            "Engineers surveyed the towpath while laborers unloaded crates "
            "at the wharf, and the toll clerk recorded every manifest. ")
    results["long_prefill"] = gen(para * 700, 32)

def canon(r):
    mi = r["meta_info"]
    out = {"text": r["text"], "token_ids": r["output_ids"]}
    lp = mi.get("output_token_logprobs")
    if lp is not None:
        out["logprobs"] = [[x[0], x[1]] for x in lp]
    for k in ("spec_accept_length", "spec_verify_ct", "spec_accept_rate"):
        if mi.get(k) is not None:
            out[k] = mi[k]
    return out

with open(out_path, "w") as f:
    json.dump({k: canon(v) for k, v in results.items()}, f, indent=1)
print(f"wrote {out_path}")
PYEOF
}

check_server_log() { # $1 = backend; gate self-check
  local backend="$1"
  local logfile="$OUT/server_$backend.log"
  if [[ "$backend" == staged ]]; then
    if ! grep -q "gate wired" "$logfile"; then
      log "FAIL: staged boot has no 'gate wired' line — prep gate never built"
      return 1
    fi
    if grep -q "gate ids != in-graph ids" "$logfile"; then
      log "FAIL: staged boot logged PLE prep mismatch (gate ids vs in-graph ids)"
      grep "gate ids != in-graph ids" "$logfile" | head -5 >&2
      return 1
    fi
    if grep -q "gather ids != pre-staged ids" "$logfile"; then
      log "FAIL: staged boot logged PLE prefill pre-staged ids mismatch"
      grep "gather ids != pre-staged ids" "$logfile" | head -5 >&2
      return 1
    fi
    local hits
    hits=$(grep -c "gate ids == in-graph ids" "$logfile" || true)
    log "staged gate self-check: $hits matching steps, 0 mismatches"
    if [[ "$AHEAD" != 0 ]]; then
      if ! grep -q "PLE chunk-ahead prefetch wired" "$logfile"; then
        log "FAIL: staged boot has no 'PLE chunk-ahead prefetch wired' line"
        return 1
      fi
      grep "PLE chunk-ahead prefetch wired" "$logfile" | head -1 >&2
      log "staged chunk-ahead arm: depth=$AHEAD"
    fi
  fi
}

stop_server() {
  docker rm -f "oracle-ple-$1" >/dev/null 2>&1 || true
}

trap 'stop_server pinned; stop_server staged' EXIT

# Resume support: a completed pinned collection is reused as-is (both boots
# send the identical request sequence, so pinning the first half is safe).
if [[ -s "$OUT/outputs_pinned.json" ]]; then
  log "reusing existing pinned outputs at $OUT/outputs_pinned.json"
else
  run_server pinned
  collect pinned "$OUT/outputs_pinned.json"
  check_server_log pinned
  stop_server pinned
fi

run_server staged
collect staged "$OUT/outputs_staged.json"
check_server_log staged
stop_server staged

python3 - "$OUT/outputs_pinned.json" "$OUT/outputs_staged.json" "$MTP" <<'PYEOF'
import json, sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
MTP = sys.argv[3] == "1"
assert a.keys() == b.keys(), (list(a), list(b))
bad = []
for k in a:
    if a[k] != b[k]:
        for field in ("token_ids", "logprobs", "text"):
            if a[k].get(field) != b[k].get(field):
                x, y = a[k].get(field), b[k].get(field)
                first = next((i for i, (p, q) in enumerate(
                    zip(x or [], y or [])) if p != q), None)
                bad.append(f"{k}.{field}: first diff at {first}")
if bad:
    print("ORACLE FAIL: pinned vs staged differ")
    for line in bad:
        print(" ", line)
    sys.exit(1)
n = sum(len(a[k]["token_ids"]) for k in a)
lp = sum(len(a[k].get("logprobs", [])) for k in a)
print(f"ORACLE PASS: {len(a)} responses, {n} output tokens and {lp} "
      "logprob values bit-identical between pinned and staged backends")
if MTP:
    vals = [a[k]["spec_accept_length"] for k in a if "spec_accept_length" in a[k]]
    if vals:
        print(f"MTP accept length: mean {sum(vals)/len(vals):.3f} over "
              f"{len(vals)} responses (production baseline ~2.1; identical "
              "bit-exact outputs imply pinned and staged accept agree)")
PYEOF
