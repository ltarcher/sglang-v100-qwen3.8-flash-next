"""M3 dump: verify live spill placement + LRU swap bytes against the checkpoint.

Invoked from the model's ``finalize_after_quant_processing`` hook when
``SGLANG_M3_SPILL_DUMP`` is set (scheduler process, after Marlin pack). For
every FusedMoE it checks

1. kept GPU rows and spilled host mirror rows byte-match an independent CPU
   repack of the checkpoint safetensors,
2. ``RoutedExpertLru.ensure()`` of a spilled expert lands its row in a GPU
   slot byte-identically, then restores the kept residents.

Writes a text report next to the env-given path; never raises.
"""

import json
import os

import torch


class _Cfg:
    is_gated = True


def _checkpoint_rows(model_dir: str, layer_idx: int, n_experts: int):
    from safetensors import safe_open

    path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(path):
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = {}
        for k, v in weight_map.items():
            if f".layers.{layer_idx}.mlp.experts." in k:
                files.setdefault(v, []).append(k)
        out = {}
        for fname, keys in files.items():
            with safe_open(os.path.join(model_dir, fname), framework="pt") as f:
                for k in keys:
                    _store_key(out, k, f.get_tensor(k), layer_idx)
        return out
    out = {}
    with safe_open(path, framework="pt") as f:
        for e in range(n_experts):
            for proj, shard in (("gate_proj", "w1"), ("up_proj", "w3")):
                base = (
                    f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.{proj}"
                )
                w = f.get_tensor(f"{base}.weight")
                s = f.get_tensor(f"{base}.weight_scale")
                s2 = f.get_tensor(f"{base}.weight_scale_2")
                out.setdefault(f"{proj}.weight", []).append(w)
                out.setdefault(f"{proj}.weight_scale", []).append(s)
                out.setdefault(f"{proj}.weight_scale_2", []).append(
                    s2.reshape(1)
                    if s2.dim() == 0
                    else s2.reshape(-1)[:1] if s2.numel() > 1 else s2.reshape(1)
                )
                out[f"{proj}.input_scale"] = out.get(f"{proj}.input_scale", []) + [
                    f.get_tensor(f"{base}.input_scale").reshape(1)
                ]
            base = f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.down_proj"
            out.setdefault("down_proj.weight", []).append(
                f.get_tensor(f"{base}.weight")
            )
            out.setdefault("down_proj.weight_scale", []).append(
                f.get_tensor(f"{base}.weight_scale")
            )
            out.setdefault("down_proj.weight_scale_2", []).append(
                f.get_tensor(f"{base}.weight_scale_2").reshape(())
            )
            out.setdefault("down_proj.input_scale", []).append(
                f.get_tensor(f"{base}.input_scale").reshape(())
            )
    return out


def _store_key(out, k, t, layer_idx):  # pragma: no cover - sharded path
    out.setdefault(k, []).append(t)


def _reference_pack(model_dir: str, layer_idx: int, n_experts: int):
    """CPU repack of the full checkpoint expert stack (independent of the server)."""
    from sglang.srt.layers.quantization.modelopt_quant import (
        prepare_moe_nvfp4_layer_for_sm70_marlin,
    )

    raw = _checkpoint_rows(model_dir, layer_idx, n_experts)
    dummy = torch.nn.Module()
    dummy.orig_dtype = torch.float16
    dummy.moe_runner_config = _Cfg()
    stack_of = lambda name: torch.stack(  # noqa: E731
        [raw[f"{p}.{name}"][e] for e in range(n_experts) for p in ("gate_proj",)]
    )
    w13_w = torch.cat(
        [
            torch.stack(
                [
                    torch.cat(
                        [raw["gate_proj.weight"][e], raw["up_proj.weight"][e]], dim=0
                    )
                    for e in range(n_experts)
                ]
            )
        ]
    )
    w13_s = torch.stack(
        [
            torch.cat(
                [
                    raw["gate_proj.weight_scale"][e],
                    raw["up_proj.weight_scale"][e],
                ],
                dim=0,
            )
            for e in range(n_experts)
        ]
    )
    w13_s2 = torch.stack(
        [
            torch.stack(
                [raw["gate_proj.weight_scale_2"][e], raw["up_proj.weight_scale_2"][e]]
            ).reshape(2)
            for e in range(n_experts)
        ]
    )
    w2_w = torch.stack([raw["down_proj.weight"][e] for e in range(n_experts)])
    w2_s = torch.stack([raw["down_proj.weight_scale"][e] for e in range(n_experts)])
    w2_s2 = torch.stack([raw["down_proj.weight_scale_2"][e] for e in range(n_experts)])
    for attr, val in (
        ("w13_weight", w13_w),
        ("w13_weight_scale", w13_s),
        ("w13_weight_scale_2", w13_s2),
        ("w2_weight", w2_w),
        ("w2_weight_scale", w2_s),
        ("w2_weight_scale_2", w2_s2),
    ):
        dummy.register_parameter(
            attr, torch.nn.Parameter(val.contiguous().clone(), requires_grad=False)
        )
    prepare_moe_nvfp4_layer_for_sm70_marlin(dummy)
    ref = {
        attr: getattr(dummy, attr).data
        for attr in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    }
    ref["w13_scale2"] = dummy.w13_scale2.data
    ref["w2_scale2"] = dummy.w2_scale2.data
    del stack_of
    return ref


def _byte_eq(a, b, what, errs):
    a = a.detach().cpu()
    b = b.detach().cpu()
    if a.shape != b.shape:
        errs.append(f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        return
    if a.dtype in (torch.int32, torch.uint8, torch.float8_e4m3fn):
        if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            n = (a.view(torch.uint8) != b.view(torch.uint8)).sum().item()
            errs.append(f"{what}: {n}/{a.numel()} bytes differ")
    elif not torch.equal(a, b):
        errs.append(
            f"{what}: values differ max={(a.float() - b.float()).abs().max().item()}"
        )


def probe_forward(model, out_path: str) -> None:
    """Numeric MoE parity: full-8 GPU stack vs 5-slot stack + LRU page-in."""
    lines = []
    try:
        from sglang.srt.layers.moe.dsv41_expert_spill import page_in_spill_experts
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        torch.manual_seed(0)
        model_dir = os.environ.get("SGLANG_M3_SPILL_DUMP_MODEL_DIR")
        moes = [
            m for _, m in model.named_modules() if m.__class__.__name__ == "FusedMoE"
        ]
        dev = torch.device("cuda")
        for moe in moes:
            li = int(getattr(moe, "layer_id", -1))
            lru = getattr(moe, "_dsv41_expert_lru", None)
            if lru is None:
                continue
            pool = getattr(moe, "_dsv41_landing_pool", None)
            n_e = int(moe.num_local_experts)
            ref = _reference_pack(model_dir, li, n_e)
            hidden = int(moe.w13_weight.shape[1] * 16)  # marlin [E, K/16, 2N]
            cases = {
                # decode-shaped: every id resident after one ensure (<= 5 slots)
                "decode_t2": ([5, 7, 1, 6], [[5, 7], [1, 6]]),
                "decode_t1_cold": ([5, 2], [[5, 2]]),
                # prefill-shaped, exactly 5 unique (== slot count, no thrash)
                "prefill_t6": ([0, 1, 2, 3, 5], [[0, 5], [1, 5], [2, 1], [3, 2], [0, 3], [2, 5]]),
            }
            for case, (uniq, pairs) in cases.items():
                T, topk = len(pairs), 2
                hs = torch.randn(T, hidden, dtype=torch.float16, device=dev) * 0.5
                ids = torch.tensor(pairs, dtype=torch.int32, device=dev)
                logits = torch.randn(T, n_e, dtype=torch.float32, device=dev)
                tw = torch.rand(T, topk, dtype=torch.float32, device=dev) + 0.25
                tw = tw / tw.sum(-1, keepdim=True)

                common = dict(
                    gating_output=logits,
                    topk_weights=tw,
                    num_bits=4,
                    is_gated=True,
                    activation="silu",
                    inplace=False,
                )
                ref_out = fused_marlin_moe(
                    hidden_states=hs.clone(),
                    w1=ref["w13_weight"].to(dev),
                    w2=ref["w2_weight"].to(dev),
                    w1_scale=ref["w13_weight_scale"].to(dev),
                    w2_scale=ref["w2_weight_scale"].to(dev),
                    w1_global_scale=ref["w13_scale2"].to(dev),
                    w2_global_scale=ref["w2_scale2"].to(dev),
                    topk_ids=ids,
                    **common,
                )

                kept_ids, cold_ids = moe._dsv41_spill_placement
                lru.ensure(sorted(set(uniq)))
                phys = lru.map_ids(ids.clone())
                spill_out = fused_marlin_moe(
                    hidden_states=hs.clone(),
                    w1=moe.w13_weight.data,
                    w2=moe.w2_weight.data,
                    w1_scale=moe.w13_weight_scale.data,
                    w2_scale=moe.w2_weight_scale.data,
                    w1_global_scale=moe.w13_scale2.data,
                    w2_global_scale=moe.w2_scale2.data,
                    topk_ids=phys,
                    **common,
                )
                diff = (ref_out.float() - spill_out.float()).abs()
                rel = (diff / ref_out.float().abs().clamp_min(1e-3)).max().item()
                lines.append(
                    f"L{li} {case}: max_abs={diff.max().item():.3e} "
                    f"max_rel={rel:.3e} any_neg={bool((phys < 0).any())} "
                    f"phys={phys.flatten().tolist()}"
                )
                lru.ensure(list(kept_ids))

            if pool is not None and pool.quant_info is not None:
                refkw = dict(
                    w1=ref["w13_weight"].to(dev),
                    w2=ref["w2_weight"].to(dev),
                    w1_scale=ref["w13_weight_scale"].to(dev),
                    w2_scale=ref["w2_weight_scale"].to(dev),
                    w1_global_scale=ref["w13_scale2"].to(dev),
                    w2_global_scale=ref["w2_scale2"].to(dev),
                )
                for case, (uniq, pairs) in {
                    "landing_t1": ([5, 2], [[5, 2]]),
                    "landing_t2": ([5, 6, 1], [[5, 6], [1, 2]]),
                }.items():
                    T, topk = len(pairs), 2
                    hs = torch.randn(T, hidden, dtype=torch.float16, device=dev) * 0.5
                    ids = torch.tensor(pairs, dtype=torch.int32, device=dev)
                    logits = torch.randn(T, n_e, dtype=torch.float32, device=dev)
                    tw = torch.rand(T, topk, dtype=torch.float32, device=dev) + 0.25
                    tw = tw / tw.sum(-1, keepdim=True)
                    common = dict(
                        gating_output=logits,
                        topk_weights=tw,
                        num_bits=4,
                        is_gated=True,
                        activation="silu",
                        inplace=False,
                        is_expert_parallel=True,
                    )
                    ref_out = fused_marlin_moe(
                        hidden_states=hs.clone(),
                        topk_ids=ids,
                        **refkw,
                        **common,
                    )
                    # e2e contract: whole-batch page-in on the working topk
                    # tensor, then main run (kept slots) + landing run (pool).
                    work = ids.clone()
                    page_in_spill_experts(moe, work)
                    land_ids = moe._dsv41_land_ids
                    main_out = fused_marlin_moe(
                        hidden_states=hs.clone(),
                        w1=moe.w13_weight.data,
                        w2=moe.w2_weight.data,
                        w1_scale=moe.w13_weight_scale.data,
                        w2_scale=moe.w2_weight_scale.data,
                        w1_global_scale=moe.w13_scale2.data,
                        w2_global_scale=moe.w2_scale2.data,
                        topk_ids=work,
                        **common,
                    )
                    land_out = fused_marlin_moe(
                        hidden_states=hs.clone(),
                        w1=pool.quant_info.w13_qweight,
                        w2=pool.quant_info.w2_qweight,
                        w1_scale=pool.quant_info.w13_scales,
                        w2_scale=pool.quant_info.w2_scales,
                        w1_global_scale=pool.quant_info.w13_global_scale,
                        w2_global_scale=pool.quant_info.w2_global_scale,
                        topk_ids=land_ids,
                        **common,
                    )
                    # Controls: the same split run on the full reference pack
                    # (land/main logical ids carry the expert id at exactly
                    # the positions the landing/main runs compute).
                    land_logical = ids.masked_fill(work >= 0, -1)
                    main_logical = ids.masked_fill(work < 0, -1)
                    ref_main = fused_marlin_moe(
                        hidden_states=hs.clone(), topk_ids=main_logical, **refkw, **common
                    )
                    ref_land = fused_marlin_moe(
                        hidden_states=hs.clone(), topk_ids=land_logical, **refkw, **common
                    )
                    total = main_out + land_out
                    diff = (ref_out.float() - total.float()).abs()
                    dmain = (ref_out.float() - main_out.float()).abs().max().item()
                    dland = (land_out.float()).abs().max().item()
                    lines.append(
                        f"L{li} {case}: total_max_abs={diff.max().item():.3e} "
                        f"main_vs_ref_max={dmain:.3e} land_max={dland:.3e} "
                        f"ctl_main_d={(ref_main.float() - main_out.float()).abs().max().item():.3e} "
                        f"ctl_land_d={(ref_land.float() - land_out.float()).abs().max().item():.3e} "
                        f"ref_land_max={ref_land.float().abs().max().item():.3e} "
                        f"work={work.flatten().tolist()} "
                        f"land={land_ids.flatten().tolist()}"
                    )
                    # Byte-check every pool row the page-in just filled.
                    slot = int(land_ids[0, 0].item())
                    if slot >= 0:
                        hr = int(pool.slot_host_row[slot].item())
                        kept0, cold0 = moe._dsv41_spill_placement
                        cold_expert = cold0[hr] if 0 <= hr < len(cold0) else -1
                        for attr, qk in (
                            ("w13_weight", "w13_qweight"),
                            ("w2_weight", "w2_qweight"),
                            ("w13_weight_scale", "w13_scales"),
                            ("w2_weight_scale", "w2_scales"),
                        ):
                            errs = []
                            _byte_eq(
                                getattr(pool.quant_info, qk)[slot],
                                ref[attr][cold_expert] if cold_expert >= 0 else torch.zeros(1),
                                f"L{li} {case} POOL {qk}[{slot}] (host_row {hr}, expert {cold_expert})",
                                errs,
                            )
                            lines.append(
                                f"    pool {qk}[{slot}]: " + ("OK" if not errs else errs[-1])
                            )
                        for gk, rk in (
                            ("w13_global_scale", "w13_scale2"),
                            ("w2_global_scale", "w2_scale2"),
                        ):
                            lines.append(
                                f"    pool {gk}[{slot}]={getattr(pool.quant_info, gk)[slot].item():.6e} "
                                f"ref={ref[rk][cold_expert].item() if cold_expert >= 0 else float('nan'):.6e}"
                            )
                    lru.ensure(list(kept_ids))

                # V2: manual pool — same shapes/dtypes as pool.quant_info but
                # filled with plain torch copies. Row 0 = ref expert 5, rows
                # 1..S = the layer GPU rows (so layer slot s lives at row 1+s
                # and its marlin id shifts by +1). Isolates the page-in kernel
                # and pool memory from the run configuration itself.
                qi = pool.quant_info
                kept0, _ = moe._dsv41_spill_placement
                e2_slot = kept0.index(2)
                e2_row = 1 + e2_slot
                T, topk = 1, 2
                hs = torch.randn(T, hidden, dtype=torch.float16, device=dev) * 0.5
                logits = torch.randn(T, n_e, dtype=torch.float32, device=dev)
                tw = torch.rand(T, topk, dtype=torch.float32, device=dev) + 0.25
                tw = tw / tw.sum(-1, keepdim=True)
                common = dict(
                    gating_output=logits,
                    topk_weights=tw,
                    num_bits=4,
                    is_gated=True,
                    activation="silu",
                    inplace=False,
                )
                man_w13 = torch.empty_like(qi.w13_qweight)
                man_w13[0].copy_(ref["w13_weight"][5].to(dev))
                man_w13[1 : 1 + moe.w13_weight.shape[0]].copy_(moe.w13_weight.data)
                man_w2 = torch.empty_like(qi.w2_qweight)
                man_w2[0].copy_(ref["w2_weight"][5].to(dev))
                man_w2[1 : 1 + moe.w2_weight.shape[0]].copy_(moe.w2_weight.data)
                man_s13 = torch.empty_like(qi.w13_scales)
                man_s13[0].copy_(ref["w13_weight_scale"][5].to(dev))
                man_s13[1 : 1 + moe.w13_weight_scale.shape[0]].copy_(
                    moe.w13_weight_scale.data
                )
                man_s2 = torch.empty_like(qi.w2_scales)
                man_s2[0].copy_(ref["w2_weight_scale"][5].to(dev))
                man_s2[1 : 1 + moe.w2_weight_scale.shape[0]].copy_(
                    moe.w2_weight_scale.data
                )
                man_g13 = torch.zeros_like(qi.w13_global_scale)
                man_g13[0].copy_(ref["w13_scale2"][5].to(dev))
                man_g13[1 : 1 + moe.w13_scale2.shape[0]].copy_(moe.w13_scale2.data)
                man_g2 = torch.zeros_like(qi.w2_global_scale)
                man_g2[0].copy_(ref["w2_scale2"][5].to(dev))
                man_g2[1 : 1 + moe.w2_scale2.shape[0]].copy_(moe.w2_scale2.data)
                mankw = dict(
                    w1=man_w13,
                    w2=man_w2,
                    w1_scale=man_s13,
                    w2_scale=man_s2,
                    w1_global_scale=man_g13,
                    w2_global_scale=man_g2,
                )
                it32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731

                def _run(kw, ids_):
                    return fused_marlin_moe(
                        hidden_states=hs.clone(), topk_ids=it32(ids_), **kw, **common
                    )

                d_full = (
                    _run(refkw, [[5, 2]]).float() - _run(mankw, [[5, e2_row]]).float()
                ).abs().max().item()
                ref_l1 = _run(refkw, [[5, -1]])
                man_l1 = _run(mankw, [[0, -1]])
                d_l1 = (ref_l1.float() - man_l1.float()).abs().max().item()
                ref_m1 = _run(refkw, [[-1, 2]])
                man_m1 = _run(mankw, [[-1, e2_row]])
                d_m1 = (ref_m1.float() - man_m1.float()).abs().max().item()
                lines.append(
                    f"L{li} manual_pool(e2_row={e2_row}): full_d={d_full:.3e} "
                    f"land_d={d_l1:.3e} main_d={d_m1:.3e} "
                    f"ref_l1_max={ref_l1.float().abs().max().item():.3e} "
                    f"man_l1_max={man_l1.float().abs().max().item():.3e}"
                )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"probe_forward error: {exc!r}")
    lines.append("PROBE_FORWARD " + ("?" if not lines else "done"))
    with open(out_path + ".fwd", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def dump_model(model, out_path: str) -> None:
    lines = []
    errs = []
    try:
        hf = getattr(getattr(model, "config", None), "_name_or_path", None) or getattr(
            model, "model_path", ""
        )
        model_dir = os.environ.get("SGLANG_M3_SPILL_DUMP_MODEL_DIR") or hf
        moes = [
            m for _, m in model.named_modules() if m.__class__.__name__ == "FusedMoE"
        ]
        lines.append(f"model_dir={model_dir} fused_moe_layers={len(moes)}")
        for moe in moes:
            li = int(getattr(moe, "layer_id", -1))
            plan = getattr(moe, "_dsv41_expert_spill_plan", None)
            hosts = getattr(moe, "_dsv41_spill_host", None)
            lru = getattr(moe, "_dsv41_expert_lru", None)
            if plan is None or hosts is None or lru is None:
                lines.append(f"L{li}: spill not applied (plan={plan is not None})")
                continue
            kept_ids, cold_ids = moe._dsv41_spill_placement
            lines.append(
                f"L{li}: kept={kept_ids} cold={cold_ids} "
                f"pinned={getattr(moe, '_dsv41_spill_host_pinned', None)}"
            )
            ref = _reference_pack(model_dir, li, int(moe.num_local_experts))
            for attr in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
                gpu = getattr(moe, attr).data
                for slot, e in enumerate(kept_ids):
                    _byte_eq(
                        gpu[slot], ref[attr][e], f"L{li} GPU {attr}[{slot}]e{e}", errs
                    )
                for row, e in enumerate(cold_ids):
                    _byte_eq(
                        hosts[attr][row], ref[attr][e], f"L{li} HOST {attr}[{row}]e{e}", errs
                    )
            for attr in ("w13_scale2", "w2_scale2"):
                gpu = getattr(moe, attr).data
                for slot, e in enumerate(kept_ids):
                    _byte_eq(gpu[slot], ref[attr][e], f"L{li} GPU {attr}[{slot}]e{e}", errs)
                for row, e in enumerate(cold_ids):
                    _byte_eq(hosts[attr][row], ref[attr][e], f"L{li} HOST {attr}[{row}]e{e}", errs)

            # Swap mechanics: page a cold expert in, verify the GPU row it
            # lands on, then restore the kept residents.
            n_routed = int(getattr(moe, "_num_local_routed", 0))
            gpu_ids = torch.arange(n_routed, dtype=torch.int32, device="cuda")
            for e in cold_ids:
                lru.ensure([e])
                phys = lru.map_ids(
                    torch.tensor([[e]], dtype=torch.int32, device="cuda")
                ).item()
                where = f"L{li} SWAP expert{e}->slot{phys}"
                if not (0 <= phys < gpu.shape[0]):
                    errs.append(f"{where}: physical slot {phys} out of range")
                    continue
                for attr in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
                    _byte_eq(
                        getattr(moe, attr).data[phys],
                        ref[attr][e],
                        f"{where} {attr}",
                        errs,
                    )
                for attr in ("w13_scale2", "w2_scale2"):
                    _byte_eq(
                        getattr(moe, attr).data[phys], ref[attr][e], f"{where} {attr}", errs
                    )
                lru.ensure(list(kept_ids))
            del gpu_ids
    except Exception as exc:  # noqa: BLE001 - diagnostics must not kill the server
        errs.append(f"dump_model error: {exc!r}")
    lines.extend(f"ERR {e}" for e in errs)
    lines.append("RESULT " + ("OK" if not errs else f"FAILED ({len(errs)})"))
    try:
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        print("\n".join(lines))
    print("\n".join(lines[-8:]))
