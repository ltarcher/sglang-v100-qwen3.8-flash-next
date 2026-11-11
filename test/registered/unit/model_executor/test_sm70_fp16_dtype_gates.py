"""SM70 (SGLANG_SM70_FORCE_FP16) dtype gates on Volta.

Volta has no native BF16: with the env on (the default), a sub-sm80 device
downgrades the model dtype to fp16, casts leftover bf16 checkpoint tensors,
forces the DSA/linear-attention caches to fp16, and pins the mamba conv cache
to fp16 while an explicit SGLANG_MAMBA_SSM_DTYPE outranks the blanket
downgrade. With the env off, bf16-tuned GDN models opt out entirely. The
device here is mocked, so these gates run on CPU runners.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.configs.mamba_utils import mamba2_state_dtype
from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    cast_bf16_parameters_for_sm70,
    maybe_downgrade_dtype_for_legacy_gpu,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

LOAD_UTILS = "sglang.srt.model_executor.model_runner_components.load_model_utils"


def _model_config(dtype=torch.bfloat16):
    return SimpleNamespace(dtype=dtype)


class TestMaybeDowngradeDtypeForLegacyGpu(CustomTestCase):
    def test_sm80_and_above_never_downgrades(self):
        cfg = _model_config()
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            maybe_downgrade_dtype_for_legacy_gpu(model_config=cfg)
        self.assertIs(cfg.dtype, torch.bfloat16)

    def test_below_sm70_is_rejected(self):
        with patch("torch.cuda.get_device_capability", return_value=(6, 1)):
            with self.assertRaisesRegex(RuntimeError, "sm70"):
                maybe_downgrade_dtype_for_legacy_gpu(model_config=_model_config())

    def test_sm70_with_env_downgrades_to_fp16(self):
        cfg = _model_config(torch.bfloat16)
        with (
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
            patch(f"{LOAD_UTILS}.get_context") as context,
        ):
            maybe_downgrade_dtype_for_legacy_gpu(model_config=cfg)
        self.assertIs(cfg.dtype, torch.float16)
        context.return_value.override.assert_called_once_with(
            "ModelRunner._sm80_dtype_fallback", dtype="float16"
        )

    def test_sm70_env_off_keeps_the_requested_dtype(self):
        cfg = _model_config(torch.bfloat16)
        with (
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
            envs.SGLANG_SM70_FORCE_FP16.override(False),
            patch(f"{LOAD_UTILS}.get_context") as context,
        ):
            maybe_downgrade_dtype_for_legacy_gpu(model_config=cfg)
        self.assertIs(cfg.dtype, torch.bfloat16)
        context.return_value.override.assert_not_called()


class TestCastBf16ParametersForSm70(CustomTestCase):
    @staticmethod
    def _bf16_module() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(4, 4).to(torch.bfloat16),
            torch.nn.Linear(4, 4).to(torch.float32),
        )

    def test_env_on_casts_bf16_parameters_to_fp16(self):
        module = self._bf16_module()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
        ):
            cast_bf16_parameters_for_sm70(module)
        dtypes = {p.dtype for p in module.parameters()}
        self.assertNotIn(torch.bfloat16, dtypes)
        self.assertIn(torch.float16, dtypes)

    def test_env_off_keeps_bf16_parameters(self):
        module = self._bf16_module()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
            envs.SGLANG_SM70_FORCE_FP16.override(False),
        ):
            cast_bf16_parameters_for_sm70(module)
        self.assertIn(torch.bfloat16, {p.dtype for p in module.parameters()})

    def test_sm80_and_above_keeps_bf16_parameters(self):
        module = self._bf16_module()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            cast_bf16_parameters_for_sm70(module)
        self.assertIn(torch.bfloat16, {p.dtype for p in module.parameters()})

    def test_no_cuda_keeps_bf16_parameters(self):
        module = self._bf16_module()
        with patch("torch.cuda.is_available", return_value=False):
            cast_bf16_parameters_for_sm70(module)
        self.assertIn(torch.bfloat16, {p.dtype for p in module.parameters()})


class TestMamba2StateDTypeEnvResolution(CustomTestCase):
    """Priority: SGLANG_MAMBA_SSM_DTYPE > config > default; conv follows env.

    On sm70 with the fp16 pipeline the conv cache is forced to fp16 and the
    SSM cache follows only when no explicit env dtype was set. The serving
    image bakes SGLANG_MAMBA_{CONV,SSM}_DTYPE=float16 into the environment,
    so every case first pins the dtype envs to "explicitly unset" to test the
    real defaults rather than the container's.
    """

    @staticmethod
    @contextlib.contextmanager
    def _sm70_device():
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
        ):
            yield

    @staticmethod
    @contextlib.contextmanager
    def _unset_dtype_envs():
        with (
            envs.SGLANG_MAMBA_CONV_DTYPE.override(None),
            envs.SGLANG_MAMBA_SSM_DTYPE.override(None),
        ):
            yield

    def test_defaults_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self._unset_dtype_envs():
                result = mamba2_state_dtype(None)
        self.assertIs(result.conv, torch.bfloat16)
        self.assertIs(result.temporal, torch.float32)

    def test_conv_dtype_env_maps_values(self):
        with patch("torch.cuda.is_available", return_value=False):
            with envs.SGLANG_MAMBA_CONV_DTYPE.override("float16"):
                self.assertIs(mamba2_state_dtype(None).conv, torch.float16)
            with envs.SGLANG_MAMBA_CONV_DTYPE.override("bfloat16"):
                self.assertIs(mamba2_state_dtype(None).conv, torch.bfloat16)

    def test_conv_dtype_env_invalid_value_falls_back_to_bf16(self):
        with patch("torch.cuda.is_available", return_value=False):
            with envs.SGLANG_MAMBA_CONV_DTYPE.override("int8"):
                self.assertIs(mamba2_state_dtype(None).conv, torch.bfloat16)

    def test_ssm_dtype_env_invalid_value_falls_back_to_float32(self):
        with patch("torch.cuda.is_available", return_value=False):
            with envs.SGLANG_MAMBA_SSM_DTYPE.override("int8"):
                self.assertIs(mamba2_state_dtype(None).temporal, torch.float32)

    def test_sm70_fp16_pipeline_forces_conv_and_unset_ssm(self):
        with self._sm70_device(), self._unset_dtype_envs():
            result = mamba2_state_dtype(None)
        self.assertIs(result.conv, torch.float16)
        self.assertIs(result.temporal, torch.float16)

    def test_sm70_explicit_ssm_env_outranks_the_fp16_pipeline(self):
        with self._sm70_device(), envs.SGLANG_MAMBA_CONV_DTYPE.override(None):
            with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
                result = mamba2_state_dtype(None)
        # Conv follows fp16, but the explicit SSM env is not silently dropped.
        self.assertIs(result.conv, torch.float16)
        self.assertIs(result.temporal, torch.bfloat16)

    def test_sm70_env_off_keeps_env_resolved_dtypes(self):
        with self._sm70_device(), envs.SGLANG_SM70_FORCE_FP16.override(False):
            with self._unset_dtype_envs():
                result = mamba2_state_dtype(None)
        self.assertIs(result.conv, torch.bfloat16)
        self.assertIs(result.temporal, torch.float32)

    def test_sm80_device_ignores_the_fp16_pipeline(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            with self._unset_dtype_envs():
                result = mamba2_state_dtype(None)
        self.assertIs(result.conv, torch.bfloat16)
        self.assertIs(result.temporal, torch.float32)

    def test_config_dtype_resolves_before_the_env(self):
        config = SimpleNamespace(mamba_ssm_dtype="float16")
        with patch("torch.cuda.is_available", return_value=False):
            with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
                self.assertIs(mamba2_state_dtype(config).temporal, torch.float16)
        bad = SimpleNamespace(mamba_ssm_dtype="int8")
        with patch("torch.cuda.is_available", return_value=False):
            with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
                self.assertIs(mamba2_state_dtype(bad).temporal, torch.float32)

    def test_vl_config_reads_text_config(self):
        config = SimpleNamespace(
            text_config=SimpleNamespace(mamba_ssm_dtype="bfloat16")
        )
        with patch("torch.cuda.is_available", return_value=False):
            with envs.SGLANG_MAMBA_SSM_DTYPE.override(None):
                self.assertIs(mamba2_state_dtype(config).temporal, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
