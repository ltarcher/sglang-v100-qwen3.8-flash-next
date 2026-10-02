"""M3 probe: live spill placement vs independent CPU repack of the checkpoint.

Launches the mini NVFP4 model in-process with the forced-spill env, then for
every MoE layer compares

- spilled host mirror rows (post Marlin pack + pin) and
- kept GPU rows (post process_weights Marlin pack)

against an independent CPU repack of the checkpoint safetensors built here.
Byte equality on codes/scales, exact equality on global scales; also checks
input_scale rows and the LRU bookkeeping. Any mismatch means the loader
placed an expert in the wrong row/slot or the host pack diverged.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["SGLANG_DSV41_EXPERT_SPILL_APPLY"] = "1"
os.environ["SGLANG_DSV41_EXPERT_SPILL_GB"] = "0.001"
os.environ["SGLANG_DSV41_EXPERT_SPILL_MIN_LOCAL"] = "1"
os.environ["SGLANG_DSV41_EXPERT_SPILL_N_LAYERS"] = "3"
os.environ["SGLANG_ENABLE_DSV41_EXPERT_SPILL_NUMA"] = "0"

import torch  # noqa: E402

sys.path.insert(0, "/opt/sglang/python")

from safetensors import safe_open  # noqa: E402

from sglang.srt.entrypoints.engine import Engine  # noqa: E402
from sglang.srt.layers.quantization.modelopt_quant import (  # noqa: E402
    prepare_moe_nvfp4_layer_for_sm70_marlin,
)

MODEL = "/data/models/mini-glm-nvfp4"


class _Cfg:
    is_gated = True


def _checkpoint_rows(layer_idx: int):
    """Checkpoint-layout expert tensors for one layer, expert-major."""
    path = os.path.join(MODEL, "model.safetensors")
    out = {}
    with safe_open(path, framework="pt") as f:
        for e in range(8):
            g_w = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
            )
            u_w = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"
            )
            g_s = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight_scale"
            )
            u_s = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight_scale"
            )
            g_s2 = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight_scale_2"
            )
            u_s2 = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight_scale_2"
            )
            d_w = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
            )
            d_s = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight_scale"
            )
            d_s2 = f.get_tensor(
                f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight_scale_2"
            )
            out[e] = {
                "w13_weight": torch.cat([g_w, u_w], dim=0).contiguous(),
                "w13_weight_scale": torch.cat([g_s, u_s], dim=0).contiguous(),
                "w13_weight_scale_2": torch.stack([g_s2, u_s2]).reshape(2).float(),
                "w2_weight": d_w.contiguous(),
                "w2_weight_scale": d_s.contiguous(),
                "w2_weight_scale_2": d_s2.reshape(()).float(),
            }
    return out


def _reference_pack(rows):
    """Independent repack of the full expert stack (checkpoint -> Marlin)."""
    dummy = torch.nn.Module()
    dummy.orig_dtype = torch.float16
    dummy.moe_runner_config = _Cfg()
    for attr in (
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_scale_2",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_scale_2",
    ):
        sample = rows[0][attr]
        stack = torch.stack([rows[e][attr] for e in range(len(rows))])
        dummy.register_parameter(
            attr, torch.nn.Parameter(stack.clone(), requires_grad=False)
        )
    prepare_moe_nvfp4_layer_for_sm70_marlin(dummy)
    return {
        attr: getattr(dummy, attr).data
        for attr in (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
        )
    } | {
        "w13_scale2": dummy.w13_scale2.data,
        "w2_scale2": dummy.w2_scale2.data,
    }


def _byte_eq(a: torch.Tensor, b: torch.Tensor, what: str, errs: list) -> None:
    a = a.detach().cpu()
    b = b.detach().cpu()
    if a.shape != b.shape:
        errs.append(f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        return
    if a.dtype in (torch.int32, torch.uint8, torch.float8_e4m3fn):
        if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            n = (a.view(torch.uint8) != b.view(torch.uint8)).sum().item()
            errs.append(f"{what}: {n}/{a.numel()} bytes differ")
    else:
        if not torch.equal(a, b):
            errs.append(f"{what}: values differ (max {(a - b).abs().max().item()})")


def main() -> None:
    eng = Engine(
        model_path=MODEL,
        trust_remote_code=True,
        dtype="float16",
        attention_backend="tilelang_fa_v100",
        linear_attn_prefill_backend="triton",
        linear_attn_decode_backend="triton",
        tp_size=1,
        mem_fraction_static=0.12,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        log_level="warning",
    )
    try:
        mr = getattr(eng, "model_runner", None)
        if mr is None:
            mr = eng.scheduler.model_runner
        moes = [
            m
            for _, m in mr.model.named_modules()
            if m.__class__.__name__ == "FusedMoE"
        ]
        assert moes, "no FusedMoE found"
        print(f"found {len(moes)} FusedMoE layers")

        all_errs = []
        for moe in moes:
            layer_idx = int(getattr(moe, "layer_id", -1))
            plan = getattr(moe, "_dsv41_expert_spill_plan", None)
            assert plan is not None and plan.n_spilled > 0, (
                f"L{layer_idx}: no spill plan applied"
            )
            hosts = getattr(moe, "_dsv41_spill_host", None)
            lru = getattr(moe, "_dsv41_expert_lru", None)
            assert hosts is not None and lru is not None
            kept_ids, cold_ids = moe._dsv41_spill_placement
            print(
                f"L{layer_idx}: kept={kept_ids} cold={cold_ids} "
                f"host_pinned={getattr(moe, '_dsv41_spill_host_pinned', None)}"
            )

            ref = _reference_pack(_checkpoint_rows(layer_idx))
            for attr in (
                "w13_weight",
                "w13_weight_scale",
                "w2_weight",
                "w2_weight_scale",
            ):
                gpu = getattr(moe, attr).data
                for slot, e in enumerate(kept_ids):
                    _byte_eq(
                        gpu[slot],
                        ref[attr][e],
                        f"L{layer_idx} GPU {attr} slot{slot}(expert{e})",
                        all_errs,
                    )
                for row, e in enumerate(cold_ids):
                    _byte_eq(
                        hosts[attr][row],
                        ref[attr][e],
                        f"L{layer_idx} HOST {attr} row{row}(expert{e})",
                        all_errs,
                    )
            for attr in ("w13_scale2", "w2_scale2"):
                gpu = getattr(moe, attr).data
                for slot, e in enumerate(kept_ids):
                    _byte_eq(
                        gpu[slot],
                        ref[attr][e],
                        f"L{layer_idx} GPU {attr} slot{slot}(expert{e})",
                        all_errs,
                    )
                for row, e in enumerate(cold_ids):
                    _byte_eq(
                        hosts[attr][row],
                        ref[attr][e],
                        f"L{layer_idx} HOST {attr} row{row}(expert{e})",
                        all_errs,
                    )
            for attr in ("w13_input_scale", "w2_input_scale"):
                p = getattr(moe, attr, None)
                if p is None:
                    continue
                for row, e in enumerate(cold_ids):
                    ref_row = ref.get(attr)
                    if ref_row is None:
                        break
                    _byte_eq(
                        hosts[attr][row],
                        ref_row[e],
                        f"L{layer_idx} HOST {attr} row{row}(expert{e})",
                        all_errs,
                    )

        if all_errs:
            print(f"PROBE FAILED with {len(all_errs)} mismatches:")
            for e in all_errs[:20]:
                print("  ", e)
            sys.exit(1)
        print("PROBE OK: spill host rows and GPU kept rows match independent pack")
    finally:
        eng.shutdown()


if __name__ == "__main__":
    main()
