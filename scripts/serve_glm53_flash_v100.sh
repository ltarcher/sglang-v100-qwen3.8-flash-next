#!/usr/bin/env bash
# Serve GLM-5.3-Flash (320B/18B MoE, 34x KDA linear attention + 11x DSA,
# modelopt NVFP4 checkpoint) on 4x V100-32GB, TP=4.
#
# Default mode is the P4 resident-pool deployment: the routed experts are
# requantized NVFP4 -> 2-bit (u2b2, group 128) once at boot and the FULL pool
# stays in VRAM (~19 GiB/rank). No expert ever touches host RAM: no spill
# pool, no page-in, no landing pool. Quality was gated against the u4+spill
# deployment on the same engine (same-top1 91.7%, needle 5/5) -- see
# docs/v100/GLM53_FLASH_PLAN.md appendix G.
#
# Before the FIRST serve, run the (GPU-touching) validation once:
#   bash scripts/smoke_v100.sh
#
# Usage (from the repo root):
#   bash scripts/serve_glm53_flash_v100.sh            # u2 pool, target-only
#   bash scripts/serve_glm53_flash_v100.sh mtp        # u2 pool + NEXTN MTP-3/4
#   bash scripts/serve_glm53_flash_v100.sh spill      # u4 + host spill (P1.6 form)
#   bash scripts/serve_glm53_flash_v100.sh spill-mtp  # u4 + spill + MTP
#   bash scripts/serve_glm53_flash_v100.sh dsa        # u2 pool + DSA sparse path
#   bash scripts/serve_glm53_flash_v100.sh dsa-mtp    # u2 pool + DSA + MTP
#
# The spill modes are the pre-P4 deployment shape, kept for A/B: experts
# split resident-half / host-spill with page-in per forward (decode ~4.6x
# slower than the u2 pool, prefill ~2.5x slower at 8k).
#
# Env overrides:
#   GLM53_MODEL=/path/to/GLM-5.3-Flash-NVFP4             (required)
#   SGLANG_V100_VENV=/path/to/venv                       (default: $HOME/sglang-v100-venv)
#   GLM53_GPUS=0,1,2,3
#   SGLANG_V100_HOST=0.0.0.0    (shared with Qwen/DSV41)
#   SGLANG_V100_PORT=11435      (shared with Qwen/DSV41; not 30000)
#   GLM53_MEM_FRACTION=0.92     (see the flag comment before raising)
#   GLM53_CONTEXT=8192          (validated ceiling so far; see flag comment)
#   GLM53_U2_GROUP=128          (u2 scale group; 64/32 are the kernel fallbacks)
#   GLM53_U2_STAGE_DIR=/path    (boot-time requant staging; MUST be real disk)
#   GLM53_U2_CACHE=1            (persist converted u2 pools under the stage dir;
#                                boots of the same model/TP skip the ~22 min
#                                requant AND skip reading the ~150 GB of
#                                checkpoint expert bytes the pools replace --
#                                expert weight/weight_scale tensors are not
#                                read from the checkpoint at all on a hit; a
#                                missing cache file at process time fails the
#                                boot loudly instead of loading zero experts;
#                                ~76 GB on disk, auto-rebuilt on model or TP
#                                change)
set -euo pipefail

MODE="${1:-target}"
case "$MODE" in target|mtp|spill|spill-mtp|dsa|dsa-mtp) ;; *) echo "mode must be target|mtp|spill|spill-mtp|dsa|dsa-mtp" >&2; exit 1;; esac

VENV="${SGLANG_V100_VENV:-$HOME/sglang-v100-venv}"
[[ -x "$VENV/bin/python" ]] || VENV="${VIRTUAL_ENV:-$VENV}"
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV; set SGLANG_V100_VENV" >&2; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${GLM53_MODEL:-}"
[[ -n "$MODEL" ]] || { echo "set GLM53_MODEL to the GLM-5.3-Flash-NVFP4 checkout" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${GLM53_GPUS:-0,1,2,3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# The venv's bin must be on PATH, not just its python: the SM70 JIT kernels
# shell out to `ninja`, which is a venv-local binary with no system copy.
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
# Runtime JIT kernels shell out to a bare `nvcc`, which defaults to the host
# `gcc`. Pick the newest complete GCC <= 14 (CUDA 12.x caps host support at
# GCC 14 and many distros default to a GCC 15 with no cc1plus).
# NVCC_PREPEND_FLAGS is needed separately from CUDAHOSTCXX: the runtime JIT
# invokes a bare nvcc, which reads the former and not the latter.
for _v in 14 13 12; do
  if [[ -x "/usr/bin/g++-$_v" ]] && ls /usr/libexec/gcc/*/$_v/cc1plus >/dev/null 2>&1; then
    export CC="/usr/bin/gcc-$_v" CXX="/usr/bin/g++-$_v" CUDAHOSTCXX="/usr/bin/g++-$_v"
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-$_v"
    break
  fi
done
export TORCH_CUDA_ARCH_LIST=7.0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
# Same machine-dependent comm setup as serve_qwen38_flash_next_nvfp4_v100.sh:
# without SGLANG_CUSTOM_AR_ALLOW_PCIE=1 the custom all-reduce refuses a
# PCIe-only 4x V100 box and every small AR falls back to NCCL. Detect the
# NVLink mesh; an explicit NCCL_P2P_LEVEL in the environment still wins.
if [[ -z "${NCCL_P2P_LEVEL:-}" ]]; then
  if nvidia-smi topo -m 2>/dev/null | grep -E '^GPU[0-9]' | grep -qE 'NV[0-9]'; then
    export NCCL_P2P_LEVEL=NVL
  else
    export NCCL_P2P_LEVEL=PXB
    export SGLANG_CUSTOM_AR_ALLOW_PCIE="${SGLANG_CUSTOM_AR_ALLOW_PCIE:-1}"
  fi
fi
# Volta has no bf16; the mamba/linear-attention path must run fp16 on sm70.
export SGLANG_MAMBA_CONV_DTYPE=float16
# KDA recurrent (temporal) state: fp16 is the production choice (2026-10-07
# A/B: fp32 bought zero loop improvement at +143MB/rank and a pool trim).
# The sm70 force-fp16 branch in mamba_utils.py only pins fp16 when this var
# is unset, so an explicit float32 here is honored if ever needed.
export SGLANG_MAMBA_SSM_DTYPE=float16
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_SM70_QWEN_FUSIONS=1
# Runaway guards (2026-10-06, four runaway incidents that burned up to 42k+
# tokens / 7 min each before abort). LIMIT clamps max_new_tokens for ALL
# requests (client-passed values included, with a capping warning); the loop
# breaker actually terminates a formed cycle: output-tail periodic runs
# (period 4-32 tokens, >=3 identical rounds) finish the request with
# type=length + loop_detected{period,repeats}. ignore_eos=True requests
# (bench contract) are exempt, so gates are unaffected; needle/decode gates
# re-verified clean after enabling (5/5; 75.4 tok/s vs gate 74). First day in
# production it caught a real think-section runaway in 17 s (period 15,
# cut@636). See docs/v100/GLM53_FLASH_PLAN.md K.9/K.10 and the unit test
# test/registered/unit/managers/test_output_loop_detector.py.
export SGLANG_MAX_NEW_TOKENS_LIMIT="${SGLANG_MAX_NEW_TOKENS_LIMIT:-32768}"
export SGLANG_ENABLE_OUTPUT_LOOP_BREAK="${SGLANG_ENABLE_OUTPUT_LOOP_BREAK:-1}"
# Period-scan upper bound of the loop breaker. Default 32 keeps the phrase
# window; 2048 covers the observed large-period runaway variant (~600-1000
# token repeated blocks; one such loop escaped the breaker on day one and
# burned ~5k tokens until the client timeout). A cut still needs >=3
# identical blocks, so worst-case detection latency is 3x the period; the
# per-step scan stays ~0.25 ms via a first-element prefilter (measured on a
# clean 20k-token tail). See docs/v100/GLM53_FLASH_PLAN.md K.14.
export SGLANG_OUTPUT_LOOP_BREAK_MAX_PERIOD="${SGLANG_OUTPUT_LOOP_BREAK_MAX_PERIOD:-2048}"
# Spec accept-rate lock breaker (2026-10-08): finishes a request whose
# trailing verify rounds accept ~every draft (paper accept_rate = correct/
# proposed drafts, no bonus) -- the one-hot attractor end state. Independent
# of the period scanner above: fires before three exact repeats accumulate
# and covers periods past its 2048 window. Threshold None disables; 0.96 is
# zai's production guard, and on this engine the deep lock fingerprints at
# 1.00 while healthy content stays <=0.47 -- the workbuddy block-repeat loop
# form (~0.5) is the period scanner's job, not this one. WINDOW sets the
# detection latency (a lock needs ~that many rounds to saturate the average);
# MIN_TOKENS keeps short legitimate high-acceptance runs (tables, formulas)
# from arming it. ignore_eos requests are exempt (bench contract). See
# test/registered/unit/managers/test_spec_accept_break.py.
export SGLANG_SPEC_ACCEPT_BREAK_THRESHOLD="${SGLANG_SPEC_ACCEPT_BREAK_THRESHOLD:-0.96}"
export SGLANG_SPEC_ACCEPT_BREAK_MIN_TOKENS="${SGLANG_SPEC_ACCEPT_BREAK_MIN_TOKENS:-128}"
export SGLANG_SPEC_ACCEPT_BREAK_WINDOW="${SGLANG_SPEC_ACCEPT_BREAK_WINDOW:-64}"
# Thinking-discipline text appended server-side to every chat system message
# (jinja chat-template path only; unset keeps prompts untouched). Same-quest
# A/B on this checkpoint (agent shape, 5000-token budget): without it 14/16
# runs entered a thinking-segment repetition loop, with it 0/5 finished as
# normal tool calls. A strengthened addendum showed no further effect in a
# 62k-token-context testbed (11/12 vs 10/12 loops), and loop rate is flat
# from 7.5k to 62k prompt tokens, so the text stays as-is -- long-context
# loops are a checkpoint property the breaker contains. Changing the text
# shifts every prompt hash once (cold radix prefix cache rebuild). See
# docs/v100/GLM53_FLASH_PLAN.md K.16 and
# test/registered/unit/entrypoints/test_chat_system_suffix.py.
export SGLANG_CHAT_SYSTEM_SUFFIX="${SGLANG_CHAT_SYSTEM_SUFFIX:-Reasoning effort: low. Keep your thinking short: sketch the plan in a few dozen lines at most, then write the final answer. Do not enumerate alternatives exhaustively.}"
# Thinking budget (2026-10-07, K.17 follow-up): wired and live. K.17's
# revert blamed the NEXTN verify path, but v1/v2 share the same
# traverse_tree -> fill_vocab_mask; the real root cause was the GLM tool
# constraint keying as full_assistant_ebnf, which ReasonerGrammarBackend
# returned unwrapped, so the budget had no attach point. The fix wraps it
# in owning mode under --enable-strict-thinking (accepts/fills forwarded
# while thinking; budget exhaustion forces the </think> row) and a filter
# that preserves the inner EBNF row instead of resetting it. Verified live:
# a bare chat request stops at reasoning_tokens=4097 (= 4096 budget + the
# forced </think>) and still emits content; prefill/decode/needle gates
# unchanged. See docs/v100/GLM53_FLASH_PLAN.md K.18 and
# test/registered/unit/constrained/test_reasoner_grammar_backend.py.
export SGLANG_MAX_THINK_TOKENS="${SGLANG_MAX_THINK_TOKENS:-4096}"
# P5-b A step: the mHC pre/post elementwise kernels run the fp16 TileLang
# port. Measured on the u2 dsa-mtp c1024 recipe: ~7.6k prefill 1275-1300 ->
# 1436-1454 tok/s (+12%), needle 5/5, near-full-pool eviction regression
# clean, decode/accept unchanged. The projection GEMM stays chunked cuBLAS
# fp32 -- the sm70 TileLang MMA emitter has no fp32 operand form. Unset
# either var to fall back to the torch elementwise path.
export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-1}"
export SGLANG_OPT_USE_TILELANG_MHC_POST="${SGLANG_OPT_USE_TILELANG_MHC_POST:-1}"
# Measured win: the allocator grows in place instead of defragmenting, which
# the u2 boot needs (a ~19 GiB pool is carved out while capture tries to
# reserve its workspace). torch 2.9.1 only honors PYTORCH_CUDA_ALLOC_CONF --
# the PYTORCH_ALLOC_CONF name (which c10/core/AllocatorConfig.h documents as
# the primary variable) is silently ignored in practice (memory_snapshot
# reports is_expandable False), and without it the 42-layer u2 pool restore
# fragments +2.15 GB/rank, clamping a 230400-token KV pool to 146176.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER=1
export FLASHINFER_WORKSPACE_BASE="${GLM53_U2_STAGE_DIR:-$HOME/.cache/sglang-glm53-jit}"
export TRITON_CACHE_DIR="$FLASHINFER_WORKSPACE_BASE/triton"
# V100 runtime pin is sglang-kernel 0.4.6.post1, not upstream 0.4.7.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export PYTHONPATH="$REPO/python"

if [[ "$MODE" == spill || "$MODE" == spill-mtp ]]; then
  # P1.6-form u4 deployment: resident half-pool + 20 GiB/rank host spill,
  # per-layer landing pool for prefill. The mirror is cudaHostRegister'd.
  export SGLANG_DSV41_EXPERT_SPILL_APPLY=1
  export SGLANG_DSV41_EXPERT_SPILL_GB=20
  export SGLANG_DSV41_EXPERT_SPILL_N_LAYERS=42
  export SGLANG_DSV41_SPILL_LANDING=336
  export SGLANG_DSV41_EXPERT_SPILL_PREFILL_LANDING=1
  SPEC=()
else
  # P4 resident pool. The stage dir receives the transient per-layer NVFP4
  # staging memmaps during the boot-time requantization (~43 GB written and
  # unlinked over the boot). It MUST be real disk: tmpfs (e.g. many /tmp) is
  # host RAM and defeats the whole point.
  export SGLANG_SM70_U2_EXPERT_POOL=1
  export SGLANG_DSV41_SPILL_LANDING=0
  export SGLANG_SM70_U2_STAGE_DIR="${GLM53_U2_STAGE_DIR:-$HOME/.cache/sglang-glm53-u2-stage}"
  export SGLANG_SM70_U2_CACHE="${GLM53_U2_CACHE:-0}"
  # U2B2 P5-b: bind the u2 pool to the v2 transposed-word GEMM (M>1: prefill
  # chunks and the M=4 spec verify step; M=1 decode stays on marlin u2).
  # Boot-time contract: flipping it changes which bytes the pool binds, so it
  # needs a restart; boot fails hard if the installed marlin .so lacks the op
  # (binding T bytes with the marlin kernel would read them as garbage).
  # Measured A/B on the u2 dsa-mtp c1024 recipe, same boot, only this env
  # flipped: 7936-token prefill x9 1343 -> 1490 tok/s (+10.9%), spec decode
  # 68.2 -> 74.7 tok/s (+9.5%), accept 3.39 and needle 5/5 unchanged; the
  # boot binding self-test (bit equality through fused_marlin_moe) gates all
  # four ranks. Default 0 pending the production-compose re-validation.
  export SGLANG_USE_SM70_U2_GEMM_V2="${SGLANG_USE_SM70_U2_GEMM_V2:-0}"
  mkdir -p "$SGLANG_SM70_U2_STAGE_DIR"
  SPEC=()
fi

if [[ "$MODE" == mtp || "$MODE" == spill-mtp || "$MODE" == dsa-mtp ]]; then
  SPEC+=(--speculative-algorithm NEXTN --speculative-num-steps 3
         --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
         --enable-linear-replayssm-spec --max-running-requests 4)
  # Draft proposals sample from q=softmax(logits/T) instead of argmax, so they
  # match the temperature-sampled verify: measured A/B 2026-10-08 put
  # real-content accept at 2.58-2.75 (baseline 1.0-2.4, decode 20-50 -> 42-64
  # tok/s client-side). Unbiased: the output distribution is unchanged, and
  # the loop-attractor families are identical. Greedy rows (top_k=1,
  # including the temp-0 rewrite) still propose argmax, so the greedy decode
  # gate keeps its baseline fingerprint (75.5-81.8 tok/s @ accept 4.00 /
  # rate 1.00). Zero per-step overhead: steps/s flat ~20-25/s across phases.
  SPEC+=(--speculative-use-rejection-sampling)
fi

# P5-a: light up the 11 DSA layers' indexer + topk + sparse attention on the
# sm70 TileLang kernels (the triton default runs them as full attention).
# The kpool fused topk emits topk + pool_size-1 = 2051 columns for GLM
# (index_kpool=4 with always_select_tail), so the kpool decode path MUST run
# the fused JIT topk with the sgl-kernel identity page-table consumer: the
# torch transform path asserts the non-kpool 2048 width and is a composition
# upstream never executes. SGLANG_OPT_USE_TOPK_V2=0 keeps the folded topk-v2
# plan off (zero sm70 coverage, shrinks the capture surface). This exact
# composition passed the A4 gate (tops 24/24, needles 5/5, prefill 8k
# 1390 tok/s vs 717 full-attn baseline).
ATTN_ARGS=(--attention-backend triton)
if [[ "$MODE" == dsa || "$MODE" == dsa-mtp ]]; then
  export SGLANG_DSA_FUSE_TOPK="${SGLANG_DSA_FUSE_TOPK:-1}"
  export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
  ATTN_ARGS=(--attention-backend dsa
             --dsa-prefill-backend tilelang
             --dsa-decode-backend tilelang
             --dsa-topk-backend sgl-kernel)
fi

args=(
  --trust-remote-code
  --model-path "$MODEL"
  --served-model-name glm53-flash-nvfp4
  # GLM-5.3-Flash's chat template teaches the NEW no-newline tool-call format
  # (<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>).
  # Without this flag the raw block leaks into content and agent clients get
  # no tool_calls field (2026-10-06 zcode incident); the glm45 key is the
  # GLM-4.5-era detector whose regex requires a newline after the name and
  # does NOT match this template -- glm47 is the exact-format detector.
  # Thinking model: the template opens <|assistant|><think>, so generation
  # starts INSIDE the thinking segment; without this parser the thinking text
  # all lands in content (the 2026-10-06 runaway incident). glm45 splits
  # <think>..</think> into reasoning_content vs content for display only.
  --reasoning-parser glm45
  --tool-call-parser glm47
  # Enables the ReasonerGrammarBackend token-filter layer that
  # SGLANG_MAX_THINK_TOKENS attaches to (above). Without it the default
  # path is byte-identical to upstream (the full-assistant EBNF stays
  # unwrapped), so this flag is the budget's on/off switch.
  --enable-strict-thinking
  --dtype float16
  --quantization modelopt_fp4
  "${ATTN_ARGS[@]}"
  --tensor-parallel-size 4
  --host "${SGLANG_V100_HOST:-0.0.0.0}"
  --port "${SGLANG_V100_PORT:-11435}"
  # 0.92 is what every P4 boot ran (18.9 GiB expert pool + ~8 GiB
  # dense/attention on a 32 GB card leaves little slack). The measured
  # OOM chain on this engine is at boot/capture, not under load: raising
  # this further starves capture, lowering it shrinks the KV pool for
  # nothing. Change together with GLM53_CONTEXT.
  --mem-fraction-static "${GLM53_MEM_FRACTION:-0.92}"
  # 8192 is the P4-era default. The fp16-KV long-context recipe validated on
  # the pool path (2026-10-05/06) is --context-length 230400 together with
  # --max-total-tokens 230400, --max-mamba-cache-size 8 and
  # --mem-fraction-static 0.94 (212992 if 230400 OOMs at boot/capture); the
  # production compose pins that set. Do not bump context and mem fraction
  # together on faith -- the OOM chain shows up at boot/capture, not load.
  --context-length "${GLM53_CONTEXT:-8192}"
  # P4 benches used 1024. GLM's routed-expert gather is per-chunk on the
  # spill path and the prefill wall grows with position either way; larger
  # chunks are untested on the pool path.
  --chunked-prefill-size 1024
  --cuda-graph-max-bs-decode 4
  # The u2 requantization + capture can exceed the default watchdog on the
  # first boot (cold Triton/Marlin JIT).
  --watchdog-timeout 900
  # The 182 GB checkpoint streams through the page cache once; dropping it
  # after load keeps ~100+ GB of host RAM from staying dirty-resident.
  --weight-loader-drop-cache-after-load
  "${SPEC[@]}"
)

cd "$REPO"
exec "$VENV/bin/python" -m sglang.launch_server "${args[@]}"
