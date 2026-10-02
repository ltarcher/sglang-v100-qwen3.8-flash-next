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
set -euo pipefail

MODE="${1:-target}"
case "$MODE" in target|mtp|spill|spill-mtp) ;; *) echo "mode must be target|mtp|spill|spill-mtp" >&2; exit 1;; esac

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
export SGLANG_MAMBA_SSM_DTYPE=float16
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_SM70_QWEN_FUSIONS=1
# Measured win: the allocator grows in place instead of defragmenting, which
# the u2 boot needs (a ~19 GiB pool is carved out while capture tries to
# reserve its workspace).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
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
  mkdir -p "$SGLANG_SM70_U2_STAGE_DIR"
  SPEC=()
fi

if [[ "$MODE" == mtp || "$MODE" == spill-mtp ]]; then
  SPEC+=(--speculative-algorithm NEXTN --speculative-num-steps 3
         --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
         --enable-linear-replayssm-spec --max-running-requests 4)
fi

args=(
  --trust-remote-code
  --model-path "$MODEL"
  --served-model-name glm53-flash-nvfp4
  --dtype float16
  --quantization modelopt_fp4
  --attention-backend triton
  --tensor-parallel-size 4
  --host "${SGLANG_V100_HOST:-0.0.0.0}"
  --port "${SGLANG_V100_PORT:-11435}"
  # 0.92 is what every P4 boot ran (18.9 GiB expert pool + ~8 GiB
  # dense/attention on a 32 GB card leaves little slack). The measured
  # OOM chain on this engine is at boot/capture, not under load: raising
  # this further starves capture, lowering it shrinks the KV pool for
  # nothing. Change together with GLM53_CONTEXT.
  --mem-fraction-static "${GLM53_MEM_FRACTION:-0.92}"
  # 8192 is the largest context the P4 boots validated (probe boot matrix).
  # Raising it needs the KV-pool/activation trade re-measured on the pool
  # path; do not bump this and the mem fraction together on faith.
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
