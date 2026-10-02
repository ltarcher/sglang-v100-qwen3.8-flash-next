# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper

logger = logging.getLogger(__name__)


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        return WeightsMapper(
            orig_to_new_substr={
                f"model.layers.{text_config.num_hidden_layers}": "model.decoder",
            },
        )

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Mixed checkpoints list the BF16 NextN block in ``quantization_config.ignore``;
        inheriting global FP8 quantization would corrupt its QKV weights."""
        raw_quant_config = getattr(config, "quantization_config", None) or {}
        if hasattr(raw_quant_config, "to_dict"):
            raw_quant_config = raw_quant_config.to_dict()
        ignored = (
            raw_quant_config.get("ignore", [])
            if isinstance(raw_quant_config, dict)
            else []
        )
        nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"
        if nextn_layer_pattern in ignored:
            logger.warning(
                "GLM5 NextN layer %s is checkpoint-declared unquantized; "
                "using BF16 draft modules",
                nextn_layer_pattern,
            )
            return None
        return super()._resolve_nextn_quant_config(config, quant_config)

    @staticmethod
    def _checkpoint_quant_layout(config):
        """Return (nextn_quant_in_checkpoint, lm_head_ignored) for a mixed
        ModelOpt FP4 checkpoint. GLM-5.3 NVFP4 quantizes the NextN block's
        routed experts and leaves everything else (attn projs, gate, shared
        experts, eh_proj, lm_head, embeddings) in BF16 via
        quantization_config.ignore -- the ignore list carries NO entry for
        the NextN layer itself, unlike checkpoints whose NextN block is BF16.
        """
        raw = getattr(config, "quantization_config", None) or {}
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        ignored = raw.get("ignore", []) if isinstance(raw, dict) else []
        nextn_ignored = (
            f"model.layers.{config.num_hidden_layers}.*" in ignored
            or f"model.language_model.layers.{config.num_hidden_layers}.*" in ignored
        )
        head_ignored = any(
            isinstance(p, str) and p.endswith("lm_head") for p in ignored
        )
        return (not nextn_ignored, head_ignored)

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        text_config = getattr(config, "text_config", config)
        # Keep the draft's routed experts fully GPU-resident: the process-wide
        # spill env would shrink this single-layer MoE to the target plan's
        # kept-slot count while the draft forward has no spill remap, sending
        # Marlin past the shrunk weight tensor (illegal memory access at the
        # first draft decode capture).
        from sglang.srt.layers.moe.dsv41_expert_spill import (
            exempt_fused_moe_from_spill,
        )

        _spill_exempt = exempt_fused_moe_from_spill()
        if (
            quant_config is not None
            and getattr(quant_config, "get_name", lambda: None)() == "modelopt_fp4"
        ):
            nextn_quant_in_ckpt, head_ignored = self._checkpoint_quant_layout(
                text_config
            )
            if nextn_quant_in_ckpt:
                # DeepseekModelNextN would otherwise strip the quant config
                # and build BF16 modules that cannot take NVFP4 weights.
                logger.info(
                    "GLM5 NextN: checkpoint quantizes the NextN block; "
                    "keeping NVFP4 draft quantization."
                )
                text_config.nextn_quant_in_checkpoint = True
            if head_ignored:
                # set_embed_and_head installs the target's BF16 head tensor.
                text_config.nextn_head_shares_target_bf16 = True
        with _spill_exempt:
            super().__init__(
                text_config,
                quant_config=quant_config,
                prefix=prefix,
            )

    def load_weights(self, weights):
        if not hasattr(self, "fuse_qkv_a_proj"):
            self.fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        layer_id = self.config.num_hidden_layers
        layer_prefixes = (
            f"model.layers.{layer_id}.",
            f"model.language_model.layers.{layer_id}.",
        )
        nextn_weights = (
            (name, weight)
            for name, weight in weights
            if name.startswith(layer_prefixes)
        )
        return Glm5NextForConditionalGeneration.load_weights(
            self, nextn_weights, is_nextn=True
        )


EntryClass = [Glm5NextForConditionalGenerationNextN]
