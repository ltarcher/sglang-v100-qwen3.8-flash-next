from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")


@pytest.fixture
def stated_tp_group():
    """A TP group for a test that runs in a process without one.

    The production call passes the group *into* `use_symmetric_memory`, so
    stubbing that context manager does not stop the read -- the argument is
    evaluated first. Stating it on the context answers every spelling.
    """
    from sglang.srt.runtime_context import get_parallel

    with get_parallel().override(tp_group=None):
        yield


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [0, 1, 8, 17, 32, 64])
@pytest.mark.parametrize("use_norm", [False, True])
def test_mhc_fused_post_pre_matches_unfused(
    monkeypatch, hidden_size, num_tokens, use_norm, stated_tp_group
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang mHC kernels")

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    # This is a single-process kernel unit test with no TP group initialized.
    # mhc_pre / mhc_fused_post_pre allocate the MoE input in the symmetric-memory
    # pool, which asks for the TP group; bypassing the allocation is enough, and
    # then nothing asks. Mirrors the workaround in test_mxfp4_sm90_cutlass.py.
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hc_mult = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    residual = (
        torch.randn(
            num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    post_prev = torch.rand(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_prev = (
        torch.rand(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
        * 0.25
    )
    fn = (
        torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device=device, dtype=torch.float32)
    norm_weight = (
        torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
        if use_norm
        else None
    )
    norm_eps = 1e-6 if use_norm else None

    rms_eps = 1e-6
    hc_eps = 1e-6
    sinkhorn_repeat = 2

    residual_ref = post_ref = comb_ref = layer_ref = None
    if num_tokens > 0:
        residual_ref = mhc_post(x, residual, post_prev, comb_prev)
        post_ref, comb_ref, layer_ref = mhc_pre(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    residual_out, post_out, comb_out, layer_out = mhc_fused_post_pre(
        x,
        residual,
        post_prev,
        comb_prev,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        2.0,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    torch.cuda.synchronize()
    if num_tokens == 0:
        assert residual_out.shape == residual.shape
        assert post_out.shape == (0, hc_mult, 1)
        assert comb_out.shape == (0, hc_mult, hc_mult)
        assert layer_out.shape == (0, hidden_size)
        assert residual_out.dtype == torch.bfloat16
        assert post_out.dtype == torch.float32
        assert comb_out.dtype == torch.float32
        assert layer_out.dtype == torch.bfloat16
        return

    assert residual_ref is not None
    assert post_ref is not None
    assert comb_ref is not None
    assert layer_ref is not None
    assert residual_out.shape == residual_ref.shape
    assert post_out.shape == post_ref.shape
    assert comb_out.shape == comb_ref.shape
    assert layer_out.shape == layer_ref.shape

    torch.testing.assert_close(residual_out, residual_ref, atol=0, rtol=0)
    torch.testing.assert_close(post_out, post_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb_out, comb_ref, atol=1e-3, rtol=1e-3)
    layer_atol = 2e-2 if use_norm else 2e-3
    layer_rtol = 2e-2 if use_norm else 2e-3
    torch.testing.assert_close(layer_out, layer_ref, atol=layer_atol, rtol=layer_rtol)


_GLM_HIDDEN = 4096
_GLM_HC_MULT = 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_norm", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 17, 1024, 4096])
def test_mhc_pre_post_tilelang_matches_torch(
    monkeypatch, dtype, num_tokens, use_norm, stated_tp_group
):
    """TileLang mhc_pre / mhc_post against the torch path (oracle) on
    GLM-5.3-Flash shapes: hc_mult=3, hidden=4096, hc_hidden=12288.

    fp16 is the production dtype on V100 (SM70 has no bf16). num_tokens <=
    2048 exercises the splitk gemm branch (hc_hidden=12288 was newly opened
    there) and 4096 the simple gemm branch; use_norm exercises the fused
    out-norm kernel every GLM layer runs.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang mHC kernels")

    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)

    torch.manual_seed(0)
    device = torch.device("cuda")
    hc_mult = _GLM_HC_MULT
    hidden_size = _GLM_HIDDEN
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    residual = (
        torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=dtype) * 0.1
    )
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1
    fn = (
        torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device=device, dtype=torch.float32)
    norm_weight = (
        torch.randn(hidden_size, device=device, dtype=dtype) * 0.1 if use_norm else None
    )
    norm_eps = 1e-6

    post_ref, comb_ref, layer_ref = mhc._mhc_pre_torch(
        residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 2
    )
    if use_norm:
        # The tilelang driver fuses the out-norm the production layers apply
        # to layer_input; mirror it on the oracle output. Both sides round to
        # the IO dtype at the same point (norm reads the half-precision
        # weighted sum), so this stays a like-for-like comparison.
        layer_ref = F.rms_norm(
            layer_ref.float(), (hidden_size,), norm_weight.float(), eps=norm_eps
        ).to(dtype)

    # SM70 production forces DEEPGEMM_HC_PRENORM off (no DeepGEMM on Volta);
    # the hook does not run in this test process, so state it to mirror prod
    # and reach the TileLang gemm paths.
    with (
        envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False),
    ):
        post_out, comb_out, layer_out = mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            2,
            norm_weight=norm_weight,
            norm_eps=norm_eps if use_norm else None,
        )

    res_ref = mhc._mhc_post_torch(x, residual, post_out, comb_out)
    with envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True):
        res_out = mhc_post(x, residual, post_out, comb_out)

    assert layer_out.dtype == dtype
    assert res_out.dtype == dtype
    assert post_out.dtype == torch.float32
    assert comb_out.dtype == torch.float32

    # bf16 rounds intermediates to 8 mantissa bits, so a different summation
    # order legitimately differs by 1-2 ulp; fp16 (the V100 production
    # dtype) stays at the fp16 ulp floor.
    ulp_tol = 6e-3 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(post_out, post_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb_out, comb_ref, atol=1e-3, rtol=1e-3)
    layer_tol = 2e-2 if use_norm else ulp_tol
    torch.testing.assert_close(layer_out, layer_ref, atol=layer_tol, rtol=layer_tol)
    torch.testing.assert_close(res_out, res_ref, atol=ulp_tol, rtol=ulp_tol)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
