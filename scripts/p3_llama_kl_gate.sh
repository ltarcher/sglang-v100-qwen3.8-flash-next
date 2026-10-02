#!/usr/bin/env bash
# P3 2-bit quality gate, llama.cpp arm (host CPU, outside the container).
#
# Measures the P4 double-quantization chain NVFP4 -> Q2_K directly: the fp16
# GGUF produced by convert_hf_to_gguf.py stores MoE experts as NVFP4 (the
# converter repacks them verbatim; only dense tensors are f16), so
# llama-quantize --allow-requantize from that file quantizes exactly the
# deployed checkpoint state, and --kl-divergence scores Q2_K against it
# (same-top1%, mean/median KLD, PPL diff -- the D-档 three-metric template).
#
# Conversion traps (both cost a full 45-min re-run if missed):
#   - --no-nextn: in-file NextN tensors are only consumed with a draft MTP
#     context (common.cpp sets ctx_type MTP for the draft side only), so a
#     perplexity context fails the tensor-count check on the main file. The
#     NextN tensors live in mtp-glm53-f16.gguf instead.
#   - --use-temp-file + TMPDIR on a writable disk dir: without it the writer
#     accumulates all 193G of tensor data in process RAM. The spool dir must
#     be writable by the invoking user (root-owned dirs silently fall back
#     to /tmp on the root fs).
#
# Stages are idempotent and run on the HOST (not in sglang-v100-dev):
#   0. convert    NVFP4 checkpoint -> f16 GGUF, experts stay NVFP4 (~194G)
#   1. quantize   glm53-f16.gguf -> glm53-q2k.gguf (~110G)
#   2. base       fp16 reference pass, dumps teacher logits (mmap)
#   3. candidate  Q2_K pass scored against the dumped logits
#
# Usage: bash scripts/p3_llama_kl_gate.sh <stage>
set -euo pipefail

ROOT=/data/develop/llama-glm5
BIN=$ROOT/build/bin
ORACLE=/data/models/glm53-oracle
CKPT=/data/models/GLM-5.3-Flash-NVFP4
F16=$ORACLE/glm53-f16.gguf
Q2K=$ORACLE/glm53-q2k.gguf
TEXT=$ORACLE/fixtures/p3_gate.txt
LOGITS=$ORACLE/p3_base_logits.bin
THREADS=$(nproc)
# CAND_GGUF / CAND_LOG override which candidate the `candidate` stage scores,
# e.g. the P4-faithful mixed arm (experts Q2_K, dense f16):
#   CAND_GGUF=$ORACLE/glm53-q2k-experts.gguf CAND_LOG=$ORACLE/p3_q2k_experts_pass.log \
#     bash scripts/p3_llama_kl_gate.sh candidate
CAND_GGUF=${CAND_GGUF:-$Q2K}
CAND_LOG=${CAND_LOG:-$ORACLE/p3_q2k_pass.log}

stage=${1:-all}

run_llama() {  # leave LD_LIBRARY_PATH pointing at the build's ggml libs
    LD_LIBRARY_PATH=$ROOT/build/bin "$@"
}

case $stage in
convert)
    mkdir -p "$ORACLE/spool"
    (cd "$ROOT" && TMPDIR="$ORACLE/spool" python3 convert_hf_to_gguf.py \
        "$CKPT" --outfile "$F16" --outtype f16 --no-nextn --use-temp-file) \
        2>&1 | tee "$ORACLE/convert_f16_full.log"
    ;;
quantize)
    test -f "$F16"
    run_llama "$BIN/llama-quantize" --allow-requantize "$F16" "$Q2K" Q2_K "$THREADS"
    ;;
base)
    test -f "$F16"
    run_llama "$BIN/llama-perplexity" -m "$F16" -f "$TEXT" --kl-divergence-base "$LOGITS" \
        -c 512 -b 512 -t "$THREADS" -ngl 0 2>&1 | tee "$ORACLE/p3_base_pass.log"
    ;;
candidate)
    test -f "$CAND_GGUF" && test -f "$LOGITS"
    run_llama "$BIN/llama-perplexity" -m "$CAND_GGUF" -f "$TEXT" --kl-divergence \
        --kl-divergence-base "$LOGITS" \
        -c 512 -b 512 -t "$THREADS" -ngl 0 2>&1 | tee "$CAND_LOG"
    ;;
all)
    bash "$0" convert && bash "$0" quantize && bash "$0" base && bash "$0" candidate
    ;;
*)
    echo "unknown stage: $stage (convert|quantize|base|candidate|all)" >&2
    exit 1
    ;;
esac
