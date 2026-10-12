"""Assemble a GLM-5.3-Flash serving checkpoint that pairs LIL QAD decoder
weights with the proven nvidia NVFP4 MTP graft.

The LIL checkpoint (local-inference-lab/GLM-5.3-Flash-NVFP4 @175ae8ce) is
quantization-aware-distilled: NVFP4 routed experts (layers 3-44), BF16
dense/attention, BF16 vision, and an MXFP8 MTP layer our engine cannot load.
The serving dir therefore takes LIL layers 0-44 verbatim (hardlinks; the
engine never writes model files) and replaces layer 45 with the nvidia
checkpoint's NVFP4 layer-45 tensors, which the u2 draft path already serves
in production (weight_scale_2 is required by convert_moe_layer_to_u2 and
absent from LIL's MXFP8 export).

The quantization contract is copied from the nvidia serving checkpoint
(hf_quant_config.json + config.json quantization_config): the fork's
ModelOptFp4Config parses only quant_algo + exclude_modules, and the LIL
whitelist style (quantized_layers / MXFP8 config_groups) would hard-fail.
Non-expert module families are identical between the two checkpoints, so the
nvidia exclude list covers LIL exactly -- except LIL's layers 0-2 dense MLP,
which LIL exports as plain BF16 (no weight_scale leaves) while the nvidia
contract leaves them NVFP4; those 9 modules are derived as scale-key gaps and
appended to both exclude lists, or the loader dies on the missing scales.
"""

import argparse
import json
import os
import re
import shutil
import sys

from safetensors.torch import safe_open, save_file

AUX_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "LICENSE",
    "README.md",
]
GRAFT_FILE = "mtp-graft.safetensors"
L45_PREFIX = "model.language_model.layers.45."


def load_json(path):
    with open(path) as f:
        return json.load(f)


def check(cond, msg):
    if not cond:
        sys.exit(f"FAIL: {msg}")


def main(
    lil_dir: str,
    nvidia_dir: str,
    out_dir: str,
):
    lil_index = load_json(os.path.join(lil_dir, "model.safetensors.index.json"))
    nvidia_index = load_json(os.path.join(nvidia_dir, "model.safetensors.index.json"))
    lil_map = lil_index["weight_map"]
    nvidia_map = nvidia_index["weight_map"]

    lil_45 = sorted(k for k in lil_map if k.startswith(L45_PREFIX))
    nvidia_45 = sorted(k for k in nvidia_map if k.startswith(L45_PREFIX))
    lil_rest = {k: f for k, f in lil_map.items() if not k.startswith(L45_PREFIX)}

    # Layer 45 must live in dedicated mtp shards so we can drop it whole.
    mtp_files = sorted(set(lil_map[k] for k in lil_45))
    check(
        mtp_files and all(f.startswith("model-mtp-") for f in mtp_files),
        f"LIL layer45 files unexpected: {mtp_files}",
    )
    check(
        mtp_files
        == sorted(set(f for f in lil_map.values() if f.startswith("model-mtp-"))),
        "some model-mtp-* shards hold non-layer45 keys",
    )

    # The nvidia graft must be a strict superset at family level and match the
    # dense keys exactly: the u2 draft conversion reads weight_scale_2 /
    # input_scale, which the LIL MXFP8 export does not carry.
    check(
        set(lil_45) <= set(nvidia_45),
        f"nvidia layer45 missing LIL keys: {sorted(set(lil_45) - set(nvidia_45))[:5]}",
    )
    lil45_expert_fams = {
        k.split(".mlp.experts.", 1)[1].split(".", 1)[1]
        for k in lil_45
        if ".mlp.experts." in k
    }
    nvidia45_expert_fams = {
        k.split(".mlp.experts.", 1)[1].split(".", 1)[1]
        for k in nvidia_45
        if ".mlp.experts." in k
    }
    check(
        lil45_expert_fams <= nvidia45_expert_fams,
        f"nvidia expert families missing: {lil45_expert_fams - nvidia45_expert_fams}",
    )

    # Every non-expert module family in LIL must exist in nvidia, whose
    # exclude_modules list is production-proven on exactly those families.
    def fam(n: str) -> str:
        return re.sub(r"\.\d+\.", ".N.", n)

    lil_nonexpert = {fam(k) for k in lil_rest}
    nvidia_nonexpert = {fam(k) for k in nvidia_map if not k.startswith(L45_PREFIX)}
    check(
        not (lil_nonexpert - nvidia_nonexpert),
        f"LIL-only non-expert families: {sorted(lil_nonexpert - nvidia_nonexpert)[:5]}",
    )

    # Dense-MLP scale gaps: modules where the nvidia contract carries quant
    # scales but the LIL body has only the plain weight. Those modules must be
    # excluded from quantization or the weight loader demands the missing
    # scales. Today that is exactly LIL's BF16 layers 0-2 dense MLP; derive,
    # don't hardcode, so a future LIL revision that quantizes them fails the
    # shape check instead of silently staying unquantized.
    def module_leaves(index_map, prefix_filter):
        mods = {}
        for k in index_map:
            if prefix_filter and not prefix_filter(k):
                continue
            m, leaf = k.rsplit(".", 1)
            mods.setdefault(m, set()).add(leaf)
        return mods

    lil_mods = module_leaves(
        lil_map, lambda k: ".layers." in k and not k.startswith(L45_PREFIX)
    )
    nvd_mods = module_leaves(
        nvidia_map, lambda k: ".layers." in k and not k.startswith(L45_PREFIX)
    )
    scale_gaps = []
    for m, leaves in nvd_mods.items():
        if not (leaves - {"weight"}):
            continue
        missing = leaves - lil_mods.get(m, set())
        if missing:
            check(
                re.fullmatch(
                    r"model\.language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj", m
                ),
                f"unexpected non-dense-MLP scale gap at {m}: {sorted(missing)}",
            )
            scale_gaps.append(m)
    check(scale_gaps, "no scale gaps found; contract amendment premise drifted")

    lil_cfg = load_json(os.path.join(lil_dir, "config.json"))
    nvidia_cfg = load_json(os.path.join(nvidia_dir, "config.json"))
    check(
        lil_cfg.get("architectures") == nvidia_cfg.get("architectures"),
        f"arch mismatch: {lil_cfg.get('architectures')} vs {nvidia_cfg.get('architectures')}",
    )

    # ---- Everything validated; build the serving dir. ----
    os.makedirs(out_dir, exist_ok=True)

    # LIL shards as hardlinks (same fs required; the engine never writes
    # model files). Fall back to a loud error rather than a silent copy.
    linked = 0
    for shard in sorted(set(lil_rest.values())):
        src = os.path.join(lil_dir, shard)
        dst = os.path.join(out_dir, shard)
        check(os.path.isfile(src), f"missing LIL shard {src}")
        if os.path.exists(dst):
            os.unlink(dst)
        try:
            os.link(src, dst)
        except OSError as e:
            sys.exit(f"FAIL: hardlink {src} -> {dst}: {e}")
        linked += 1

    # Extract the nvidia layer45 tensors into one graft shard.
    graft = {}
    for shard in sorted(set(nvidia_map.values())):
        with safe_open(os.path.join(nvidia_dir, shard), framework="pt") as f:
            for name in f.keys():
                if name.startswith(L45_PREFIX):
                    graft[name] = f.get_tensor(name)
    check(
        set(graft) == set(nvidia_45),
        "graft extraction drifted from the nvidia index",
    )
    save_file(graft, os.path.join(out_dir, GRAFT_FILE), metadata={"format": "pt"})

    index_out = {
        "metadata": lil_index.get("metadata", {}),
        "weight_map": {**lil_rest, **{k: GRAFT_FILE for k in nvidia_45}},
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2)

    # Quantization contract: nvidia + the dense-MLP exclusion amendment, both
    # files (each is a live exclude_modules source: config.json's "ignore" and
    # hf_quant_config's "quantization.exclude_modules").
    cfg = dict(lil_cfg)
    quant_cfg = json.loads(json.dumps(nvidia_cfg["quantization_config"]))
    ignore = list(quant_cfg.get("ignore") or [])
    quant_cfg["ignore"] = list(dict.fromkeys(ignore + scale_gaps))
    cfg["quantization_config"] = quant_cfg
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    hf_quant = load_json(os.path.join(nvidia_dir, "hf_quant_config.json"))
    excl = list(hf_quant["quantization"].get("exclude_modules") or [])
    hf_quant["quantization"]["exclude_modules"] = list(dict.fromkeys(excl + scale_gaps))
    with open(os.path.join(out_dir, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant, f, indent=2)
    print(f"  quant contract: nvidia + {len(scale_gaps)} dense-MLP exclusions")

    for name in AUX_FILES:
        src = os.path.join(lil_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_dir, name))

    # Chat-default sampling contract comes from the nvidia serving dir, not
    # LIL (LIL ships zai defaults, temperature 1.0): the swap experiment must
    # keep chat sampling identical to production so the only variable is the
    # weights. u2 fingerprint excludes this file, so edits here stay free.
    shutil.copyfile(
        os.path.join(nvidia_dir, "generation_config.json"),
        os.path.join(out_dir, "generation_config.json"),
    )

    total = sum(
        os.stat(os.path.join(out_dir, name)).st_size
        for name in os.listdir(out_dir)
        if os.path.isfile(os.path.join(out_dir, name))
    )
    print(f"serving dir: {out_dir}")
    print(f"  linked LIL shards: {linked} (mtp dropped: {mtp_files})")
    print(f"  graft: {len(graft)} tensors -> {GRAFT_FILE}")
    print(f"  index entries: {len(index_out['weight_map'])}")
    print(f"  new bytes (graft + configs): {total / 2**30:.1f} GiB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lil-dir", required=True)
    p.add_argument("--nvidia-dir", required=True)
    p.add_argument("--out-dir", required=True)
    main(**{k: v for k, v in vars(p.parse_args()).items()})
