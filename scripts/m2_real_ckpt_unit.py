"""Run the ACTUAL mini-glm-nvfp4 checkpoint tensors through the SM70 marlin
NVFP4 MoE processing and compare against a dequant reference of the same
stored codes. Isolates checkpoint->marlin conversion from the server.

Run: CUDA_VISIBLE_DEVICES=1 /opt/venv/bin/python /opt/sglang/scripts/m2_real_ckpt_unit.py
"""

import torch
from safetensors import safe_open

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.quantization.marlin_utils import (
    sm70_nvfp4_marlin_process_global_scale,
    sm70_nvfp4_marlin_process_scales,
)

CKPT = "/data/models/mini-glm-nvfp4/model.safetensors"
LAYER = 1
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")


def dequant(packed_u8, scales_e4m3, s2_per_half):
    """Decode [R, K/2] u8 + [R, K/16] e4m3 with per-row s2 (or (gate,up) pair).

    s2_per_half: [R] (same s2 for all rows) or [R, 2] (gate/up halves of a
    fused w13 -- rows 0..R/2 use [:,0], rows R/2.. use [:,1]).
    """
    r, packed_k = packed_u8.shape
    grid = torch.cat([GRID, -GRID])
    lo = (packed_u8 & 0x0F).long()
    hi = (packed_u8 >> 4).long()
    dec = torch.stack([grid[lo], grid[hi]], dim=-1).reshape(r, packed_k * 2)
    sf = scales_e4m3.view(torch.float8_e4m3fn).float()
    sf = torch.stack([sf] * 16, dim=-1).reshape(r, packed_k * 2)
    if s2_per_half.numel() == 2 and s2_per_half.dim() <= 1:
        s2_per_half = s2_per_half.view(1, 2)
    if s2_per_half.dim() == 2:
        half = r // 2
        s2 = torch.cat([s2_per_half[:half, 0], s2_per_half[half:, 1]])
    else:
        s2 = s2_per_half
    return dec * sf * s2.view(-1, 1)


def main():
    dev = "cuda"
    torch.manual_seed(0)
    tensors = {}
    with safe_open(CKPT, "pt", device="cpu") as f:
        for e in range(8):
            for name in ("gate_proj", "up_proj", "down_proj"):
                pfx = f"model.language_model.layers.{LAYER}.mlp.experts.{e}.{name}"
                tensors[(e, name, "w")] = f.get_tensor(pfx + ".weight")
                tensors[(e, name, "s")] = f.get_tensor(pfx + ".weight_scale")
                tensors[(e, name, "s2")] = f.get_tensor(pfx + ".weight_scale_2")

    def w(e, name):
        return tensors[(e, name, "w")].to(dev)

    def s(e, name):
        return tensors[(e, name, "s")].to(dev)

    def s2(e, name):
        return tensors[(e, name, "s2")].float().to(dev)

    w0, s0 = w(0, "gate_proj"), s(0, "gate_proj")
    print("gate weight:", tuple(w0.shape), w0.dtype, "scale:", tuple(s0.shape), s0.dtype)
    print("gate s2:", s2(0, "gate_proj").item(), "up s2:", s2(0, "up_proj").item())

    # ---- stack exactly like FusedMoE loader: w13 = [gate; up] ----
    w13_u8 = torch.stack([torch.cat([w(e, "gate_proj"), w(e, "up_proj")]) for e in range(8)])
    w13_sc = torch.stack([torch.cat([s(e, "gate_proj"), s(e, "up_proj")]) for e in range(8)])
    w13_s2 = torch.stack(
        [
            torch.stack([s2(e, "gate_proj"), s2(e, "up_proj")])
            for e in range(8)
        ]
    )  # [E, 2]
    w2_u8 = torch.stack([w(e, "down_proj") for e in range(8)])
    w2_sc = torch.stack([s(e, "down_proj") for e in range(8)])
    w2_s2 = torch.stack([s2(e, "down_proj") for e in range(8)])  # [E]
    E, R2N, PK = w13_u8.shape
    N = R2N // 2
    K = PK * 2
    print("w13:", tuple(w13_u8.shape), "w2:", tuple(w2_u8.shape))

    # ---- production SM70 processing (incl. gate/up fold) ----
    gate_scale2 = w13_s2[:, 0]
    up_scale2 = w13_s2[:, 1]
    print("s2 rel diff max:", ((up_scale2 / gate_scale2 - 1).abs().max().item()))
    s2_eff = torch.maximum(gate_scale2, up_scale2)
    half = w13_sc.shape[1] // 2
    w13_sc_folded = torch.cat(
        [
            (w13_sc[:, :half, :].float() * (gate_scale2 / s2_eff).view(-1, 1, 1))
            .clamp_(max=448.0)
            .to(w13_sc.dtype),
            (w13_sc[:, half:, :].float() * (up_scale2 / s2_eff).view(-1, 1, 1))
            .clamp_(max=448.0)
            .to(w13_sc.dtype),
        ],
        dim=1,
    )
    w13_s2_eff = s2_eff

    def repack(weight_u8):
        ne, sn, pk = weight_u8.shape
        layout = weight_u8.contiguous().view(torch.int32).transpose(1, 2).contiguous()
        empty_perm = torch.empty((ne, 0), dtype=torch.int32, device=weight_u8.device)
        from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
            gptq_marlin_moe_repack,
        )

        return gptq_marlin_moe_repack(layout, empty_perm, pk * 2, sn, 4)

    w13_m = repack(w13_u8)
    w2_m = repack(w2_u8)
    w13_sc_m, f13 = sm70_nvfp4_marlin_process_scales(
        w13_sc_folded.transpose(1, 2).contiguous(), torch.float16
    )
    w2_sc_m, f2 = sm70_nvfp4_marlin_process_scales(
        w2_sc.transpose(1, 2).contiguous(), torch.float16
    )
    g13 = (sm70_nvfp4_marlin_process_global_scale(w13_s2_eff, torch.float16) / f13).float()
    g2 = (sm70_nvfp4_marlin_process_global_scale(w2_s2, torch.float16) / f2).float()
    print("processed scales:", tuple(w13_sc_m.shape), tuple(w2_sc_m.shape))
    print("global scales:", g13.tolist(), g2.tolist())

    # ---- inputs + routing (noaux_tc-style sigmoid, norm_topk_prob) ----
    TOPK = 2

    def run(M):
        torch.manual_seed(0)
        x = torch.randn(M, K, device=dev, dtype=torch.float16) * 0.5
        logits = torch.randn(M, E, device=dev)
        topk_vals, topk_ids = torch.topk(logits, TOPK, dim=-1)
        topk_w = torch.sigmoid(topk_vals.float())
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)
        topk_w = topk_w.to(torch.float16)
        topk_ids32 = topk_ids.to(torch.int32)

        out = fused_marlin_moe(
            x,
            w13_m,
            w2_m,
            w13_sc_m,
            w2_sc_m,
            logits,
            topk_w,
            topk_ids32,
            num_bits=4,
            activation="silu",
            is_gated=True,
            clamp_limit=10.0,
            w1_global_scale=g13,
            w2_global_scale=g2,
        )
        ok = not bool(out.isnan().any())
        diff = (out.float() - _reference(x, topk_ids, topk_w).float()).abs()
        print(
            f"M={M:4d} nan={not ok} max_abs_diff={diff.max().item():.3e} "
            f"out[0,:3]={out[0, :3].tolist()}"
        )
        return ok

    def _reference(x, topk_ids, topk_w):
        M = x.shape[0]
        ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
        for m in range(M):
            for j in range(TOPK):
                e = int(topk_ids[m, j])
                wt = float(topk_w[m, j])
                d13 = dequant(w13_u8[e], w13_sc[e], w13_s2[e]).view(2 * N, K)
                d2 = dequant(w2_u8[e], w2_sc[e], w2_s2[e]).view(K, N)
                gate_h = x[m].float() @ d13[:N].T
                up_h = x[m].float() @ d13[N:].T
                h = torch.nn.functional.silu(gate_h) * up_h
                h = h.clamp(-10.0, 10.0)
                ref[m] += wt * (h @ d2.T)
        return ref.to(torch.float16)

    ok_all = all([run(M) for M in (1, 4, 64)])
    print("ALL OK" if ok_all else "SOME M FAILED")


if __name__ == "__main__":
    main()
