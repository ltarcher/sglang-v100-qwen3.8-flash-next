#!/usr/bin/env python
"""Generate a miniature GLM-5.3-Flash (glm5_next) checkpoint for V100 bring-up.

Same architecture, tensor naming, and (optionally) NVFP4 packing as
/data/models/GLM-5.3-Flash-NVFP4, scaled down so it fits any GPU and serves as
a numeric oracle target for llama-glm5 and a unit-test fixture for sglang.

The real model is 320B/18B-active; fp16 materialization (~640G) fits nowhere,
so skeleton/fp16 validation must run on this miniature.

Outputs:
  /data/models/mini-glm-fp16/    fp16 weights, no quantization_config
  /data/models/mini-glm-nvfp4/   routed experts in modelopt NVFP4, rest fp16

Layer plan (4 main + 1 NextN):
  L0 KDA + dense MLP, L1/L2 KDA + MoE, L3 DSA + MoE, L4 NextN (DSA + MoE).

NVFP4 packing follows modelopt conventions, cross-checked against
llama-glm5 conversion/base.py::_nvfp4_pack and measured on the real
checkpoint (M2):
  weight          uint8 [out, in/2]  E2M1 codes, element 2i low nibble, 2i+1 high
  weight_scale    F8_E4M3 [out, in/16]
  weight_scale_2  f32 scalar         w = fp4 * scale * weight_scale_2 (small:
                                     amax / (448*6), NOT the inverse)
  input_scale     f32 scalar         1.0 placeholder (matches real checkpoint)

Run inside the sglang-v100 container:
  docker exec sglang-v100-dev /opt/venv/bin/python \
    /opt/sglang/scripts/make_mini_glm_v100.py
"""

from __future__ import annotations

import json
import os
import shutil
import zlib

import torch
from safetensors.torch import save_file

REAL_MODEL = "/data/models/GLM-5.3-Flash-NVFP4"
OUT_FP16 = "/data/models/mini-glm-fp16"
OUT_NVFP4 = "/data/models/mini-glm-nvfp4"
SEED = 20260928

# --- mini dimensions (the real ones divided by the same factors) ---
H = 512            # hidden_size            (real 4096)
QH = 8             # num_attention_heads    (real 64)
KDA_D = 64         # linear_attn head_dim   (real 128)
KV_LORA = 256      # kv_lora_rank           (real 512)
Q_LORA = 256       # q_lora_rank            (real 1536)
NOPE = 64          # qk_nope_head_dim       (real 256)
V_HEAD = 64        # v_head_dim             (real 256)
IDX_H = 8          # index_n_heads          (real 32)
IDX_D = 128        # index_head_dim         (real 128; kpool KV pool asserts 128)
KPOOL = 4          # index_kpool            (real 4)
N_LAYERS = 4       # num_hidden_layers      (real 45)
N_KDA = 3          # layers 0..2 KDA, layer 3 DSA
FIRST_DENSE = 1    # first_k_dense_replace  (real 3)
N_EXP = 8          # n_routed_experts       (real 288)
TOPK = 2           # num_experts_per_tok    (real 8)
MOE_I = 128        # moe_intermediate_size  (real 2048)
DENSE_I = 256      # intermediate_size      (real 12288)
VOCAB = 154880     # real tokenizer, real vocab
NEXTN = N_LAYERS   # nextn layer id (== num_hidden_layers)
HC_MIX = (2 + 4) * 4   # (2 + hc_mult) * hc_mult, hc_mult=4 (real value; deepseek4 hc kernels assert hc==4)

_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_CODE = torch.arange(8, dtype=torch.uint8)
_EDGES = (_GRID[1:] + _GRID[:-1]) / 2  # 7 midpoints


def rand(key: str, *shape, scale=0.02, dtype=torch.float16):
    """Deterministic per-TENSOR random, seeded by the tensor name.

    Seeding by shape was a bug: distinct tensors share a shape (q/k/v proj,
    gate/up/down experts), so they came out bit-identical -- which also
    collapsed every weight_scale_2 to a single value and silently disabled the
    gate/up global-scale mismatch the SM70 fold (M2) exists for. Same name ->
    same weights keeps the fp16 and nvfp4 dirs comparable.
    """
    g = torch.Generator().manual_seed(SEED + zlib.crc32(key.encode("utf-8")))
    return (torch.randn(*shape, generator=g) * scale).to(dtype)


def build_config(quantized: bool) -> dict:
    layer_types = ["linear_attention"] * N_KDA + ["deepseek_sparse_attention"]
    cfg = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "language_only": True,
        "tie_word_embeddings": False,
        "image_token_id": 154854,
        "video_token_id": 154855,
        "image_start_token_id": 154830,
        "image_end_token_id": 154831,
        "video_start_token_id": 154832,
        "video_end_token_id": 154833,
        "text_config": {
            "hidden_size": H,
            "num_hidden_layers": N_LAYERS,
            "num_attention_heads": QH,
            "num_key_value_heads": QH,
            "head_dim": 0,
            "qk_head_dim": NOPE,
            "qk_nope_head_dim": NOPE,
            "qk_rope_head_dim": 0,
            "v_head_dim": V_HEAD,
            "kv_lora_rank": KV_LORA,
            "q_lora_rank": Q_LORA,
            "mla_use_nope": True,
            "mhc": True,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "linear_attn_config": {
                "num_heads": QH,
                "head_dim": KDA_D,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": list(range(N_KDA)),
                "full_attn_layers": [N_KDA],
            },
            "index_n_heads": IDX_H,
            "index_head_dim": IDX_D,
            "index_topk": 16,
            "index_kpool": KPOOL,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "indexer_rope_interleave": True,
            "index_share_for_mtp_iteration": True,
            "indexer_types": ["full"] * N_LAYERS,
            "layer_types": layer_types,
            "first_k_dense_replace": FIRST_DENSE,
            "mlp_layer_types": ["dense"] * FIRST_DENSE
            + ["sparse"] * (N_LAYERS - FIRST_DENSE),
            "n_routed_experts": N_EXP,
            "num_experts_per_tok": TOPK,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "moe_intermediate_size": MOE_I,
            "moe_layer_freq": 1,
            "moe_router_dtype": "float32",
            "output_router_logits": False,
            "router_aux_loss_coef": 0.001,
            "num_nextn_predict_layers": 1,
            "intermediate_size": DENSE_I,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-5,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "initializer_range": 0.02,
            "use_cache": True,
            "max_position_embeddings": 8192,
            "vocab_size": VOCAB,
            "pad_token_id": 154820,
            "eos_token_id": [154820, 154827, 154829],
            "dtype": "float16",
        },
    }
    if quantized:
        real = json.load(open(os.path.join(REAL_MODEL, "config.json")))
        cfg["quantization_config"] = real["quantization_config"]
    return cfg


def e2m1_quantize(w: torch.Tensor):
    """modelopt NVFP4: w = code * scale_e4m3 * weight_scale_2.

    Convention measured against the real checkpoint (dequant std 0.0199 on a
    0.02-scale tensor): ``weight_scale_2 = amax / (448*6)`` (small), block
    values normalized as ``w / weight_scale_2`` reach at most 448*6, block
    scale = block_max / 6 (E4M3, up to 448). NOTE: a cosine self-check cannot
    catch a global-scale convention error (scale-invariant) — magnitude checks
    or an engine-level comparison are required; see M2.
    """
    assert w.dim() == 2 and w.shape[1] % 16 == 0
    out_f, in_f = w.shape
    wf = w.float()

    s2 = (wf.abs().amax() / (448.0 * 6.0)).clamp(min=1e-30)
    blocks = (wf / s2).reshape(out_f, in_f // 16, 16)
    block_max = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scales = (block_max / 6.0).to(torch.float8_e4m3fn)

    q = blocks / scales.float()
    sign = (q < 0).to(torch.uint8) << 3
    idx = torch.bucketize(q.abs().clamp_max(6.0), _EDGES)
    codes = _CODE[idx] | sign

    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(out_f, in_f // 2)
    return packed, scales.reshape(out_f, in_f // 16), s2


def nvfp4_dequantize(packed, scales_u8, s2):
    """Reference dequant used for the generator's self-check only."""
    grid = torch.cat([_GRID, -_GRID])
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    decoded = torch.stack([grid[lo], grid[hi]], dim=-1).reshape(
        packed.shape[0], -1)
    sf = scales_u8.view(torch.float8_e4m3fn).float()
    sf = sf.repeat_interleave(16, dim=1)
    return decoded * sf * s2


def kda_layer(pfx: str, dense_mlp: bool) -> dict[str, torch.Tensor]:
    t = {
        f"{pfx}.self_attn.q_proj.weight": rand(f"{pfx}.self_attn.q_proj.weight", QH * KDA_D, H),
        f"{pfx}.self_attn.k_proj.weight": rand(f"{pfx}.self_attn.k_proj.weight", QH * KDA_D, H),
        f"{pfx}.self_attn.v_proj.weight": rand(f"{pfx}.self_attn.v_proj.weight", QH * KDA_D, H),
        f"{pfx}.self_attn.b_proj.weight": rand(f"{pfx}.self_attn.b_proj.weight", QH, H),
        f"{pfx}.self_attn.f_a_proj.weight": rand(f"{pfx}.self_attn.f_a_proj.weight", KDA_D, H),
        f"{pfx}.self_attn.f_b_proj.weight": rand(f"{pfx}.self_attn.f_b_proj.weight", QH * KDA_D, KDA_D),
        f"{pfx}.self_attn.g_a_proj.weight": rand(f"{pfx}.self_attn.g_a_proj.weight", KDA_D, H),
        f"{pfx}.self_attn.g_b_proj.weight": rand(f"{pfx}.self_attn.g_b_proj.weight", QH * KDA_D, KDA_D),
        f"{pfx}.self_attn.o_proj.weight": rand(f"{pfx}.self_attn.o_proj.weight", H, QH * KDA_D),
        f"{pfx}.self_attn.q_conv1d.weight": rand(f"{pfx}.self_attn.q_conv1d.weight", QH * KDA_D, 1, 4),
        f"{pfx}.self_attn.k_conv1d.weight": rand(f"{pfx}.self_attn.k_conv1d.weight", QH * KDA_D, 1, 4),
        f"{pfx}.self_attn.v_conv1d.weight": rand(f"{pfx}.self_attn.v_conv1d.weight", QH * KDA_D, 1, 4),
        f"{pfx}.self_attn.o_norm.weight": torch.ones(KDA_D, dtype=torch.float16),
        f"{pfx}.self_attn.A_log": rand(f"{pfx}.self_attn.A_log", QH, scale=0.5, dtype=torch.float32),
        f"{pfx}.self_attn.dt_bias": rand(f"{pfx}.self_attn.dt_bias", QH * KDA_D, scale=0.1, dtype=torch.float32),
        f"{pfx}.input_layernorm.weight": torch.ones(H, dtype=torch.float16),
        f"{pfx}.post_attention_layernorm.weight": torch.ones(H, dtype=torch.float16),
        f"{pfx}.hc_attn_base": rand(f"{pfx}.hc_attn_base", HC_MIX, scale=0.1, dtype=torch.float32),
        f"{pfx}.hc_attn_scale": rand(f"{pfx}.hc_attn_scale", 3, scale=0.1, dtype=torch.float32),
        f"{pfx}.hc_attn_fn": rand(f"{pfx}.hc_attn_fn", HC_MIX, 4 * H),
        f"{pfx}.hc_ffn_base": rand(f"{pfx}.hc_ffn_base", HC_MIX, scale=0.1, dtype=torch.float32),
        f"{pfx}.hc_ffn_scale": rand(f"{pfx}.hc_ffn_scale", 3, scale=0.1, dtype=torch.float32),
        f"{pfx}.hc_ffn_fn": rand(f"{pfx}.hc_ffn_fn", HC_MIX, 4 * H),
    }
    if dense_mlp:
        t[f"{pfx}.mlp.gate_proj.weight"] = rand(f"{pfx}.mlp.gate_proj.weight", DENSE_I, H)
        t[f"{pfx}.mlp.up_proj.weight"] = rand(f"{pfx}.mlp.up_proj.weight", DENSE_I, H)
        t[f"{pfx}.mlp.down_proj.weight"] = rand(f"{pfx}.mlp.down_proj.weight", H, DENSE_I)
    return t


def mlp_moe(pfx: str, quantize: bool):
    """Router + shared expert (always fp16) + routed experts (fp16 or NVFP4)."""
    t: dict[str, torch.Tensor] = {
        f"{pfx}.mlp.gate.weight": rand(f"{pfx}.mlp.gate.weight", N_EXP, H),
        f"{pfx}.mlp.gate.e_score_correction_bias": rand(f"{pfx}.mlp.gate.e_score_correction_bias", N_EXP, scale=0.01, dtype=torch.float32),
        f"{pfx}.mlp.shared_experts.gate_proj.weight": rand(f"{pfx}.mlp.shared_experts.gate_proj.weight", MOE_I, H),
        f"{pfx}.mlp.shared_experts.up_proj.weight": rand(f"{pfx}.mlp.shared_experts.up_proj.weight", MOE_I, H),
        f"{pfx}.mlp.shared_experts.down_proj.weight": rand(f"{pfx}.mlp.shared_experts.down_proj.weight", H, MOE_I),
    }
    for e in range(N_EXP):
        for proj, shape in (
            ("gate_proj", (MOE_I, H)),
            ("up_proj", (MOE_I, H)),
            ("down_proj", (H, MOE_I)),
        ):
            epfx = f"{pfx}.mlp.experts.{e}.{proj}"
            w = rand(f"{epfx}.weight", *shape)
            if not quantize:
                t[f"{epfx}.weight"] = w
                continue
            packed, scales, s2 = e2m1_quantize(w)
            deq = nvfp4_dequantize(packed, scales.view(torch.uint8), s2)
            cos = torch.nn.functional.cosine_similarity(
                deq.flatten().float(), w.flatten().float(), dim=0)
            # quality gate for 4-bit (gaussian weights typically land ~0.995);
            # packing fidelity is checked separately in M2 against the engine.
            # The std gate exists because cosine is scale-invariant and cannot
            # catch a weight_scale_2 convention error.
            assert cos > 0.99, f"{epfx}: dequant cos {cos:.6f}"
            std_rel = (deq.std() - w.std()).abs() / w.std()
            assert std_rel < 0.05, f"{epfx}: dequant std off by {std_rel:.3f}"
            t[f"{epfx}.weight"] = packed
            # The real checkpoint stores block scales as F8_E4M3 tensors. Keep
            # the fixture byte-identical: the engine's scale param is fp8_e4m3
            # and copy_ from uint8 casts numerically (66 -> 64), scrambling
            # every scale into garbage.
            t[f"{epfx}.weight_scale"] = scales.view(torch.float8_e4m3fn)
            t[f"{epfx}.weight_scale_2"] = s2.reshape(()).float()
            t[f"{epfx}.input_scale"] = torch.tensor(1.0)
    return t


def dsa_layer(pfx: str, quantize: bool, hc: bool = True) -> dict[str, torch.Tensor]:
    t = {
        f"{pfx}.self_attn.q_a_proj.weight": rand(f"{pfx}.self_attn.q_a_proj.weight", Q_LORA, H),
        f"{pfx}.self_attn.q_a_layernorm.weight": torch.ones(Q_LORA, dtype=torch.float16),
        f"{pfx}.self_attn.kv_a_proj_with_mqa.weight": rand(f"{pfx}.self_attn.kv_a_proj_with_mqa.weight", KV_LORA, H),
        f"{pfx}.self_attn.kv_a_layernorm.weight": torch.ones(KV_LORA, dtype=torch.float16),
        f"{pfx}.self_attn.q_b_proj.weight": rand(f"{pfx}.self_attn.q_b_proj.weight", QH * NOPE, Q_LORA),
        f"{pfx}.self_attn.kv_b_proj.weight": rand(f"{pfx}.self_attn.kv_b_proj.weight", QH * (NOPE + V_HEAD), KV_LORA),
        f"{pfx}.self_attn.o_proj.weight": rand(f"{pfx}.self_attn.o_proj.weight", H, QH * V_HEAD),
        f"{pfx}.self_attn.indexer.wq_b.weight": rand(f"{pfx}.self_attn.indexer.wq_b.weight", IDX_H * IDX_D, Q_LORA),
        f"{pfx}.self_attn.indexer.wk.weight": rand(f"{pfx}.self_attn.indexer.wk.weight", IDX_D, H),
        f"{pfx}.self_attn.indexer.weights_proj.weight": rand(f"{pfx}.self_attn.indexer.weights_proj.weight", IDX_H, H),
        f"{pfx}.self_attn.indexer.k_norm.weight": torch.ones(IDX_D, dtype=torch.float16),
        f"{pfx}.self_attn.indexer.k_norm.bias": torch.zeros(IDX_D, dtype=torch.float16),
        f"{pfx}.self_attn.indexer.index_kpool_compress_gate": rand(f"{pfx}.self_attn.indexer.index_kpool_compress_gate", IDX_D, H),
        f"{pfx}.self_attn.indexer.index_kpool_compress_ape": rand(f"{pfx}.self_attn.indexer.index_kpool_compress_ape", KPOOL, IDX_D),
        f"{pfx}.input_layernorm.weight": torch.ones(H, dtype=torch.float16),
        f"{pfx}.post_attention_layernorm.weight": torch.ones(H, dtype=torch.float16),
    }
    if hc:
        # trunk DSA layers carry mHC; the real L45 (NextN) has none
        t[f"{pfx}.hc_attn_base"] = rand(f"{pfx}.hc_attn_base", HC_MIX, scale=0.1, dtype=torch.float32)
        t[f"{pfx}.hc_attn_scale"] = rand(f"{pfx}.hc_attn_scale", 3, scale=0.1, dtype=torch.float32)
        t[f"{pfx}.hc_attn_fn"] = rand(f"{pfx}.hc_attn_fn", HC_MIX, 4 * H)
        t[f"{pfx}.hc_ffn_base"] = rand(f"{pfx}.hc_ffn_base", HC_MIX, scale=0.1, dtype=torch.float32)
        t[f"{pfx}.hc_ffn_scale"] = rand(f"{pfx}.hc_ffn_scale", 3, scale=0.1, dtype=torch.float32)
        t[f"{pfx}.hc_ffn_fn"] = rand(f"{pfx}.hc_ffn_fn", HC_MIX, 4 * H)
    t.update(mlp_moe(pfx, quantize))
    return t


def main():
    targets = [(False, OUT_FP16), (True, OUT_NVFP4)]
    if os.environ.get("MINI_ONLY_NVFP4"):
        # Skip the fp16 dir: a live server may hold it mmapped, and rewriting
        # the file in place under mmap is a SIGBUS.
        targets = [(True, OUT_NVFP4)]
    for quantize, out in targets:
        tensors: dict[str, torch.Tensor] = {
            "model.language_model.embed_tokens.weight": rand("model.language_model.embed_tokens.weight", VOCAB, H, scale=0.01),
            "model.language_model.norm.weight": torch.ones(H, dtype=torch.float16),
            "lm_head.weight": rand("lm_head.weight", VOCAB, H, scale=0.01),
        }
        for i in range(N_LAYERS):
            pfx = f"model.language_model.layers.{i}"
            if i < N_KDA:
                tensors.update(kda_layer(pfx, dense_mlp=(i < FIRST_DENSE)))
            else:
                tensors.update(dsa_layer(pfx, quantize))
            if i >= FIRST_DENSE and i < N_KDA:
                tensors.update(mlp_moe(pfx, quantize))

        # NextN: a DSA decoder layer (no mHC) plus its own spec head plumbing
        npfx = f"model.language_model.layers.{NEXTN}"
        tensors.update(dsa_layer(npfx, quantize, hc=False))
        tensors[f"{npfx}.eh_proj.weight"] = rand(
            f"{npfx}.eh_proj.weight", H, 2 * H
        )
        tensors[f"{npfx}.enorm.weight"] = torch.ones(H, dtype=torch.float16)
        tensors[f"{npfx}.hnorm.weight"] = torch.ones(H, dtype=torch.float16)
        tensors[f"{npfx}.shared_head.norm.weight"] = torch.ones(H, dtype=torch.float16)

        os.makedirs(out, exist_ok=True)
        json.dump(build_config(quantize), open(os.path.join(out, "config.json"), "w"), indent=2)
        save_file(tensors, os.path.join(out, "model.safetensors"))
        for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json"):
            src = os.path.join(REAL_MODEL, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out, f))
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        print(f"[mini-glm] {out}: {len(tensors)} tensors, {size / 1e6:.0f} MB, quantized={quantize}")


if __name__ == "__main__":
    main()
