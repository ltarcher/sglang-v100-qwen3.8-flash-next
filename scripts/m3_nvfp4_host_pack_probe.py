"""M3 probe: NVFP4 spill-host CPU pack == production GPU SM70 Marlin pack.

Builds a checkpoint-layout NVFP4 MoE (u8 codes, e4m3 block scales, per-expert
fp32 gate/up weight_scale_2), then:

1. GPU reference: the exact process_weights_after_loading SM70 branch chain
   (fold-down -> single-call gptq_marlin_moe_repack -> sm70 scale encode),
   the path the registered kernel test validated against a dequant reference.
2. Host path: prepare_moe_nvfp4_layer_for_sm70_marlin on CPU rows (per-expert
   stream through the CUDA repacker, CPU scale encode) -- what the spill
   mirror uses.

Asserts byte equality on codes/scales and exact equality on global scales,
plus LRU swap-shape compatibility (host row == GPU row in shape/dtype).
"""

import sys

import torch

sys.path.insert(0, "/opt/sglang/python")

from sglang.srt.layers.quantization.modelopt_quant import (
    prepare_moe_nvfp4_layer_for_sm70_marlin,
)
from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
    gptq_marlin_moe_repack,
)
from sglang.srt.layers.quantization.marlin_utils import (
    sm70_nvfp4_marlin_process_global_scale,
    sm70_nvfp4_marlin_process_scales,
)

E4M3_MAX = 448.0
GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

E, N, K = 4, 512, 1024  # N=2*inter_half, K=hidden


def quantize(w):
    """modelopt NVFP4: w = code * scale_e4m3 * weight_scale_2 (per tensor)."""
    wf = w.float()
    s2 = (wf.abs().amax() / (E4M3_MAX * 6.0)).clamp(min=1e-30)
    blocks = (wf / s2).reshape(w.shape[0], -1, 16)
    bmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scales = (bmax / 6.0).to(torch.float8_e4m3fn).squeeze(-1)
    q = blocks / scales.float().unsqueeze(-1)
    sign = (q < 0).to(torch.uint8) << 3
    edges = torch.tensor(
        [(GRID[i + 1] + GRID[i]) / 2 for i in range(7)], device=w.device
    )
    idx = torch.bucketize(q.abs().clamp_max(6.0), edges)
    lut = torch.tensor(GRID, device=w.device, dtype=torch.uint8)
    codes = (lut[idx] | sign).reshape(*w.shape[:-1], -1, 16)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(*w.shape[:-1], -1)
    return packed, scales, s2


def fold_down(scales, own_s2, s2_eff):
    """Production shape: [E, N, K/16] * [E, 1, 1], direct multiply."""
    ratio = (own_s2 / s2_eff).view(-1, 1, 1)
    return (scales.float() * ratio).clamp_(max=E4M3_MAX).to(scales.dtype)


def gpu_reference(w13_u8, w13_sc, w13_s2, w2_u8, w2_sc, w2_s2):
    """The SM70 branch chain, one repack call for the whole E-stack."""
    gate_s2, up_s2 = w13_s2[:, 0], w13_s2[:, 1]
    s2_eff = torch.maximum(gate_s2, up_s2)
    half = w13_sc.shape[1] // 2
    folded = torch.cat(
        [fold_down(w13_sc[:, :half], gate_s2, s2_eff), fold_down(w13_sc[:, half:], up_s2, s2_eff)],
        dim=1,
    )

    def repack(weight):
        ne, size_n, packed_k = weight.shape
        gptq_layout = (
            weight.cuda().contiguous().view(torch.int32).transpose(1, 2).contiguous()
        )
        perm = torch.empty((ne, 0), dtype=torch.int32, device="cuda")
        return gptq_marlin_moe_repack(gptq_layout, perm, packed_k * 2, size_n, 4).cpu()

    w13_m = repack(w13_u8)
    w2_m = repack(w2_u8)
    s13, f13 = sm70_nvfp4_marlin_process_scales(
        folded.transpose(1, 2).contiguous().cuda(), torch.float16
    )
    s2m, f2 = sm70_nvfp4_marlin_process_scales(
        w2_sc.transpose(1, 2).contiguous().cuda(), torch.float16
    )
    g13 = (
        sm70_nvfp4_marlin_process_global_scale(s2_eff.cuda(), torch.float16) / f13
    ).cpu()
    g2 = (
        sm70_nvfp4_marlin_process_global_scale(w2_s2.cuda(), torch.float16) / f2
    ).cpu()
    return w13_m, w2_m, s13.cpu(), s2m.cpu(), g13, g2


def main():
    assert torch.cuda.is_available(), "probe needs the CUDA repacker"
    torch.manual_seed(0)
    dev = torch.device("cuda")

    w13 = torch.randn(E, 2 * N, K, device=dev) * 0.02
    w2 = torch.randn(E, K, N, device=dev) * 0.02
    w13_u8, s13_list, s13_2 = [], [], []
    for e in range(E):
        pg, sg, g2 = quantize(w13[e, :N])
        pu, su, u2 = quantize(w13[e, N:])
        w13_u8.append(torch.cat([pg, pu]))
        s13_list.append(torch.cat([sg, su]))
        s13_2.append(torch.stack([g2, u2]))
    w13_u8 = torch.stack(w13_u8).cpu()
    w13_sc = torch.stack(s13_list).cpu()
    w13_s2 = torch.stack(s13_2).float().cpu()
    w2_u8, w2_sc, w2_s2 = [], [], []
    for e in range(E):
        p, s, s2 = quantize(w2[e])
        w2_u8.append(p)
        w2_sc.append(s)
        w2_s2.append(s2)
    w2_u8 = torch.stack(w2_u8).cpu()
    w2_sc = torch.stack(w2_sc).cpu()
    w2_s2 = torch.stack(w2_s2).float().cpu()
    assert (w13_s2[:, 0] != w13_s2[:, 1]).all(), "probe needs distinct gate/up s2"

    # ---- GPU production reference.
    ref = gpu_reference(w13_u8, w13_sc, w13_s2, w2_u8, w2_sc, w2_s2)

    # ---- Host pack path (what repack_spill_host_for_sm70_marlin drives).
    dummy = torch.nn.Module()
    dummy.orig_dtype = torch.float16
    dummy.w13_weight = torch.nn.Parameter(w13_u8.clone(), requires_grad=False)
    dummy.w2_weight = torch.nn.Parameter(w2_u8.clone(), requires_grad=False)
    dummy.w13_weight_scale = torch.nn.Parameter(w13_sc.clone(), requires_grad=False)
    dummy.w2_weight_scale = torch.nn.Parameter(w2_sc.clone(), requires_grad=False)
    dummy.w13_weight_scale_2 = torch.nn.Parameter(w13_s2.clone(), requires_grad=False)
    dummy.w2_weight_scale_2 = torch.nn.Parameter(w2_s2.clone(), requires_grad=False)

    class Cfg:
        is_gated = True

    dummy.moe_runner_config = Cfg()
    prepare_moe_nvfp4_layer_for_sm70_marlin(dummy)

    names = ["w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"]
    for name, r in zip(names, ref[:4]):
        got = getattr(dummy, name).data
        assert got.device.type == "cpu", f"{name} must stay on CPU, got {got.device}"
        assert got.shape == r.shape, f"{name}: host {tuple(got.shape)} vs gpu {tuple(r.shape)}"
        assert got.dtype == r.dtype, f"{name}: host {got.dtype} vs gpu {r.dtype}"
        if got.dtype in (torch.int32, torch.float8_e4m3fn):
            same = torch.equal(got, r)
            assert same, f"{name}: host pack != GPU pack (max byte diff {(got.view(torch.uint8).int() - r.view(torch.uint8).int()).abs().max()})"
        print(f"  {name}: {tuple(got.shape)} {got.dtype} == GPU reference")
    for name, r in zip(["w13_scale2", "w2_scale2"], ref[4:]):
        got = getattr(dummy, name).data
        assert torch.equal(got, r), f"{name}: {got} vs {r}"
        print(f"  {name}: {tuple(got.shape)} exact")

    # ---- LRU swap contract: host row shape/dtype == GPU row.
    gpu_w13 = ref[0].cuda()
    host_w13 = dummy.w13_weight.data
    assert host_w13.shape[1:] == gpu_w13.shape[1:] and host_w13.dtype == gpu_w13.dtype
    assert host_w13.shape[0] == E, "spilled rows must cover every host slot"
    print("LRU swap contract OK (host rows match GPU Marlin layout)")

    # ---- Re-entry guard: second pack call must be a no-op marker decision
    # handled by the caller; here just ensure the packed scales are no longer
    # raw e4m3 checkpoint values (encoded S0E5M3 rep is still e4m3 dtype but
    # the scale2 attrs exist only post-pack).
    assert hasattr(dummy, "w13_scale2") and hasattr(dummy, "w2_scale2")
    print("PROBE OK")


if __name__ == "__main__":
    main()
