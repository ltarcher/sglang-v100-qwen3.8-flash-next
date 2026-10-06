"""FP16 projections for measured two/four-token Qwen verification shapes."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

# (rows, output features, input features) -> (threads, lanes per output).
# Shapes where cuBLAS is faster deliberately stay on the existing path.
_CONFIGS = {
    (2, 24, 2560): (64, 32),
    (2, 640, 2560): (256, 32),
    (2, 1, 2560): (64, 32),
    (2, 2560, 2560): (128, 32),
    (4, 24, 2560): (64, 32),
    (4, 640, 2560): (128, 32),
    (4, 2560, 2560): (64, 32),
    (2, 4096, 2560): (256, 32),
    (4, 4096, 2560): (64, 32),
    (2, 3584, 2560): (64, 32),
    (4, 3584, 2560): (64, 32),
    (2, 2560, 1536): (64, 32),
    (4, 2560, 1536): (128, 16),
    (2, 320, 2560): (256, 32),
    (4, 320, 2560): (256, 32),
    (2, 2560, 160): (128, 8),
    (4, 2560, 160): (128, 8),
    (2, 512, 2560): (256, 32),
    (4, 512, 2560): (256, 32),
    (2, 62080, 2560): (128, 32),
    (4, 62080, 2560): (128, 32),
    # GLM-5.3-Flash TP4 verify (M=4) projections, best of 5 configs each on
    # an idle 4x V100 bench (scratch_r2/bench42_skinnny_dense.py); rel err
    # <= 4.5e-4 vs fp32 throughout. The KDA fused_qkvbfg (4, 6416, 4096)
    # (1.11x) and lm_head (1.16x) shapes measured at the bandwidth floor on
    # cuBLAS and deliberately stay there.
    (4, 1024, 4096): (512, 32),  # shared experts gate_up
    (4, 4096, 512): (128, 16),  # shared experts down
    (4, 6144, 4096): (128, 32),  # first-3 dense MLP gate_up, 68.4 vs 98.4us cuBLAS
    (4, 4096, 3072): (128, 32),  # first-3 dense MLP down
    (4, 1536, 4096): (128, 32),  # DSA q_a
    (4, 4096, 1536): (128, 32),  # DSA q_b
    (4, 512, 4096): (256, 32),  # DSA kv_a
    (4, 8192, 512): (128, 16),  # DSA kv_b
    (4, 4096, 2048): (128, 32),  # KDA/DSA o_proj
}


def shape_supported(x, weight):
    return (
        envs.SGLANG_SM70_MTP_SMALL_GEMM.get()
        and (x.shape[0], *weight.shape) in _CONFIGS
    )


@cache_once
def _module(rows, threads, lanes):
    return load_jit(
        "sm70_small_gemm",
        rows,
        threads,
        lanes,
        cuda_files=["elementwise/sm70_small_gemm.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::sm70_small_gemm::run<{rows},{threads},{lanes}>")
        ],
    )


def linear(x, weight):
    threads, lanes = _CONFIGS[(x.shape[0], *weight.shape)]
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _module(x.shape[0], threads, lanes).run(x, weight, out)
    return out
