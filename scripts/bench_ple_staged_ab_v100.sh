#!/usr/bin/env bash
# A/B throughput gate: PLE staged backend vs pinned through the Phase 2
# compute-ahead decode gate (docs/v100/PLE_NVME_TIER_DESIGN.md §11.9 step 5).
#
# Boots the SAME production MTP engine twice (pinned, then staged; 4x V100,
# TP4, EAGLE 3/1/4, decode CUDA graphs on) and runs scripts/bench_v100.py
# against each: a prefill scout (64k, ~262k TTFT) and a padding-decode ladder
# at warm state, two passes per point (pass 1 warms radix + the staged page
# cache, pass 2 is the measured warm run). Server flags mirror the 8110
# production compose (docker-compose-sglang-v100-mtp-8110.yaml: 0.90 mem
# fraction, KV host tier 786432 tokens PAIR, max-running-requests 3,
# decode graphs bs 1 2 3). Ladder request validation requires input + 4 MTP
# draft slots + completion <= 262144:
#   C=1 @ ctx 250k (max_new 12000), C=2 @ ctx 128k (12000), C=2 @ 64k (20000)
# The gate:
#   - staged warm aggregate decode throughput >= pinned - 3%  (per point)
#   - MTP accept length not lower than pinned
#   - zero "retract" lines in the staged server log
# The script prints the A/B table and PASS/FAIL per gate but always exits 0 —
# read the summary, numbers are the deliverable (README registration).
#
# Usage (repo root, target machine, GPUs idle, no other engine):
#   bash scripts/bench_ple_staged_ab_v100.sh [MODEL_DIR] [OUT_DIR]
# Env overrides: ORACLE_IMAGE / ORACLE_PORT (see oracle_ple_staged_v100.sh),
#   BENCH_SECONDS (default 30), BENCH_SKIP_PREFILL (default 0),
#   AHEAD_DEPTH (default 0; >0 = chunk-ahead prefetch depth §11.9.5 on the
#   staged arm), BENCH_SKIP_PINNED=1 (run the staged arm only and diff it
#   against a previous run's staged results in BENCH_BASELINE_DIR — used for
#   the chunk-ahead depth ladder where the pinned arm is irrelevant).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-/data/models/nvidiaQwen3.8-Flash-Next-NVFP4}"
OUT="${2:-/data/ple_staged_bench_$(date +%m%d%H%M%S)}"
IMAGE="${ORACLE_IMAGE:-sglang-v100-unified:latest}"
PORT="${ORACLE_PORT:-11436}"
PLE_DIR="${ORACLE_PLE_DIR:-/data/sglang-ple-staged}"
SECONDS_BENCH="${BENCH_SECONDS:-30}"
SKIP_PREFILL="${BENCH_SKIP_PREFILL:-0}"
AHEAD="${AHEAD_DEPTH:-0}"
SKIP_PINNED="${BENCH_SKIP_PINNED:-0}"
BASELINE_DIR="${BENCH_BASELINE_DIR:-}"
[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }
mkdir -p "$OUT"

log() { echo "[bench $(date +%H:%M:%S)] $*"; }

run_server() { # $1 = backend (pinned|staged)
  local backend="$1"
  local name="bench-ple-$backend"
  local -a extra=()
  [[ "$backend" == staged ]] && extra=(--ple-offload-backend staged)
  local -a extra_env=()
  if [[ "$backend" == staged ]]; then
    # REUSE=1 skips the 11.2 GiB/rank materialization rewrite (files persist
    # on /data from the oracle run); the raised barrier is harmless insurance.
    extra_env=(
      -e SGLANG_QWEN4_PLE_STAGED_REUSE=1
      -e SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT=1800
    )
    if [[ "$AHEAD" != 0 ]]; then
      extra_env+=(-e "SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD=$AHEAD")
    fi
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true
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
    -e SGLANG_V100_KV_PAIR_SPLIT_TESTBENCH=1 \
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
      --speculative-algorithm EAGLE \
      --speculative-draft-model-path /data/models/nvfp4 \
      --speculative-num-steps 3 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 4 \
      --warmups sampling \
      --sleep-on-idle \
      "${extra[@]}" >>"$OUT/docker_run_$backend.err" 2>&1 && break
    # Transient failure (typically: the previous arm's port bind not yet
    # released after docker rm -f); drop the half-created container, wait, retry.
    log "docker run failed for $backend (attempt $attempt/3)" >&2
    tail -5 "$OUT/docker_run_$backend.err" >&2
    docker rm -f "$name" >/dev/null 2>&1 || true
    if [[ "$attempt" == 3 ]]; then
      return 1
    fi
    sleep 5
  done
  docker logs -f "$name" >"$OUT/server_$backend.log" 2>&1 &
  LOGS_PID=$!
  for _ in $(seq 1 720); do
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      log "$backend server healthy"
      { free -h; grep -E 'MemAvailable|Shmem:' /proc/meminfo; } \
        > "$OUT/mem_$backend.txt" 2>&1
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

stop_server() {
  docker rm -f "bench-ple-$1" >/dev/null 2>&1 || true
}

bench_backend() { # $1 = backend
  local backend="$1"
  export SGLANG_V100_PORT="$PORT"
  if [[ "$SKIP_PREFILL" != 1 ]]; then
    log "[$backend] prefill scout 64k/262100"
    python3 "$REPO/scripts/bench_v100.py" prefill --lengths 65536,262100 \
      | tee "$OUT/bench_${backend}_prefill.txt"
  fi
  # Ladder "ctx:C:max_new", two passes each: pass 1 warms, pass 2 measured.
  local spec ctx c mt
  for spec in 250000:1:12000 131072:2:12000 65536:2:20000; do
    ctx="${spec%%:*}"; c="$(( $(printf '%s' "$spec" | cut -d: -f2) ))"
    mt="$(printf '%s' "$spec" | cut -d: -f3)"
    for pass in 1 2; do
      log "[$backend] decode ctx=$ctx C=$c pass=$pass (${SECONDS_BENCH}s)"
      python3 "$REPO/scripts/bench_v100.py" decode \
        --seconds "$SECONDS_BENCH" --concurrency "$c" --context "$ctx" \
        --max-tokens "$mt" \
        | tee "$OUT/bench_${backend}_decode_${ctx}_pass${pass}.txt"
    done
  done
}

trap 'stop_server pinned; stop_server staged' EXIT

if [[ "$SKIP_PINNED" == 1 ]]; then
  [[ -n "$BASELINE_DIR" && -d "$BASELINE_DIR" ]] || {
    log "BENCH_SKIP_PINNED=1 needs BENCH_BASELINE_DIR with prior staged results" >&2
    exit 1
  }
  run_server staged
  bench_backend staged
  stop_server staged
else
  run_server pinned
  bench_backend pinned
  stop_server pinned

  run_server staged
  bench_backend staged
  stop_server staged
fi

# A/B summary: pull pass-2 warm decode rows + accept from each output.
# BENCH_SKIP_PINNED runs diff the staged arm against BENCH_BASELINE_DIR's
# staged results (baseline depth-0 staged run) instead of the pinned arm.
python3 - "$OUT" "$SECONDS_BENCH" "${SKIP_PINNED}" "${BASELINE_DIR:-}" \
  "${AHEAD}" <<'PYEOF'
import re, sys

out, seconds, skip_pinned, baseline_dir, ahead = (
    sys.argv[1], float(sys.argv[2]), sys.argv[3] == "1", sys.argv[4], sys.argv[5])

def parse(path):
    row = {}
    for line in open(path):
        m = re.search(r"aggregate\s+([\d.]+) tok/s\s+per-stream\s+([\d.]+)"
                      r"\s+accept\s+([\dn/a]+)", line)
        if m:
            row = {"agg": float(m.group(1)), "stream": float(m.group(2)),
                   "accept": m.group(3)}
    return row

def parse_prefill_ttft(path):
    # cmd_prefill rows: "<prompt tokens> <TTFT s> <tok/s>"
    ttfts = {}
    for line in open(path):
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+\d+\s*$", line)
        if m:
            ttfts[int(m.group(1))] = float(m.group(2))
    return ttfts

print(f"\n=== A/B summary (warm pass 2, {seconds:.0f}s padding, "
      f"staged arm ahead_depth={ahead}) ===")
print(f"{'point':>16} {'base agg':>12} {'staged agg':>12} {'delta':>8} "
      f"{'base acc':>11} {'staged acc':>11}")
fail = []
for point in ("250000:C1", "131072:C2", "65536:C2"):
    ctx, c = point.split(":")
    if skip_pinned:
        pa = parse(f"{baseline_dir}/bench_staged_decode_{ctx}_pass2.txt")
    else:
        pa = parse(f"{out}/bench_pinned_decode_{ctx}_pass2.txt")
    s = parse(f"{out}/bench_staged_decode_{ctx}_pass2.txt")
    if not pa or not s:
        print(f"{point:>16}  MISSING RESULTS")
        fail.append(point)
        continue
    delta = (s["agg"] - pa["agg"]) / pa["agg"] * 100
    print(f"{point:>16} {pa['agg']:>10.1f}  {s['agg']:>10.1f}  {delta:>+6.1f}%  "
          f"{pa['accept']:>11} {s['accept']:>11}")
    if delta < -3.0:
        fail.append(point)
    if pa["accept"] != "n/a" and s["accept"] != "n/a" \
            and float(s["accept"]) < float(pa["accept"]):
        fail.append(f"{point}-accept")

# Prefill TTFT scout (chunk-ahead's primary target is the prefill tail wait;
# reported only when both sides ran the prefill scout).
if not skip_pinned:
    pf_a, pf_s = f"{out}/bench_pinned_prefill.txt", f"{out}/bench_staged_prefill.txt"
else:
    pf_a, pf_s = (f"{baseline_dir}/bench_staged_prefill.txt",
                  f"{out}/bench_staged_prefill.txt")
ttft_a, ttft_s = parse_prefill_ttft(pf_a), parse_prefill_ttft(pf_s)
if ttft_a and ttft_s:
    print("\nprefill scout TTFT (s):")
    for n in sorted(set(ttft_a) & set(ttft_s)):
        d = (ttft_s[n] - ttft_a[n]) / ttft_a[n] * 100
        print(f"  {n:>8}: base {ttft_a[n]:>7.2f}  staged {ttft_s[n]:>7.2f}  "
              f"{d:>+6.1f}%")

staged_log = f"{out}/server_staged.log"
# "retract" only counts scheduler retraction lines, not the server_args dump
# (whose field names contain 'retraction'); real retractions log as
# "Retracted N requests" / "retracted_reqs" style scheduler lines.
retracts = sum(1 for line in open(staged_log, errors="ignore")
               if "retract" in line.lower() and "server_args" not in line)
print(f"staged retract lines: {retracts}")
if retracts:
    fail.append("retract")
gate = "gate wired" in open(staged_log, errors="ignore").read()
print(f"staged gate wired: {gate}")
if ahead != "0":
    wired = "PLE chunk-ahead prefetch wired" in open(
        staged_log, errors="ignore").read()
    print(f"staged chunk-ahead wired (depth={ahead}): {wired}")
    if not wired:
        fail.append("chunk-ahead-not-wired")
    for line in open(staged_log, errors="ignore"):
        if "ahead_hits=" in line:
            print("  " + line.strip()[-160:])

print("\nBENCH PASS" if not fail else f"\nBENCH FAIL: {fail}")
PYEOF
