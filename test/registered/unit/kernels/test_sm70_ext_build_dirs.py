"""The SM70 CUDA extension loaders honor their SGLANG_V100_*_BUILD_DIR env vars.

Both JIT extensions (long-context decode partial, NVFP4 MoE decode) default to
`~/.cache/sglang/...` instead of /tmp so a recreated container does not re-pay
the ~2-3 min nvcc compile on every boot (the production compose points both at
the persistent sglang-v100-jit volume). The build directory is resolved before
any compilation runs, so these tests exercise the resolution/fallback/error
paths with `load_inline` mocked — no CUDA toolkit or device required.
"""

import contextlib
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

DECODE_DEFAULT = os.path.expanduser("~/.cache/sglang/sm70_longctx_decode")
NVFP4_DEFAULT = os.path.expanduser("~/.cache/sglang/sm70_nvfp4_moe")


@contextlib.contextmanager
def _writable_dir_for_env(var_name: str):
    """A real writable directory under a controlled env for the override case."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "ext")
        with patch.dict(os.environ, {var_name: target}):
            yield target


@contextlib.contextmanager
def _home_only_environ():
    """A minimal env: the build-dir vars unset, HOME kept for expanduser."""
    home = os.environ.get("HOME", "/root")
    with patch.dict(os.environ, {"HOME": home}, clear=True):
        yield


class TestSm70ExtBuildDirs(CustomTestCase):
    def setUp(self):
        # Lazy import on purpose: importing tilelang_fa_v100 loads tilelang's
        # libcudart stub, whose SONAME shadows the real libcudart if it lands
        # before flashinfer's ctypes resolution (the dtype-gate tests' import
        # chain). pytest collects every test module first, so by the time this
        # setUp runs, the real libcudart is already mapped process-wide.
        from sglang.kernels.ops.moe import sm70_nvfp4_moe_decode
        from sglang.srt.layers.attention.tilelang_fa_v100 import _decode_cuda

        self._decode_cuda = _decode_cuda
        self._nvfp4_moe = sm70_nvfp4_moe_decode
        # Reset both loaders' module state so every case drives a fresh call.
        self._saved = {
            (mod, name): getattr(mod, name)
            for mod, names in (
                (self._decode_cuda, ("_EXT", "_OPS_LOAD_ATTEMPTED")),
                (self._nvfp4_moe, ("_EXT", "_LOAD_ATTEMPTED")),
            )
            for name in names
        }
        for mod, attrs in (
            (self._decode_cuda, ("_EXT", "_OPS_LOAD_ATTEMPTED")),
            (self._nvfp4_moe, ("_EXT", "_LOAD_ATTEMPTED")),
        ):
            for name in attrs:
                setattr(mod, name, None if name == "_EXT" else False)
        self.addCleanup(self._restore)

    def _restore(self):
        for (mod, name), value in self._saved.items():
            setattr(mod, name, value)

    # -- SGLANG_V100_DECODE_CUDA_BUILD_DIR -----------------------------------

    def test_decode_cuda_cached_ext_short_circuits_the_loader(self):
        self._decode_cuda._EXT = "cached-ext"
        with patch("torch.utils.cpp_extension.load_inline") as load_inline:
            self.assertEqual(
                self._decode_cuda._load_sm70_cuda_decode_ops(), "cached-ext"
            )
        load_inline.assert_not_called()

    def test_decode_cuda_env_switch_gates_availability(self):
        with patch.dict(os.environ, {"SGLANG_V100_DECODE_CUDA": "0"}):
            self.assertFalse(self._decode_cuda.sm70_cuda_decode_enabled())
            self.assertFalse(self._decode_cuda.sm70_cuda_decode_available())
        self.assertTrue(self._decode_cuda.sm70_cuda_decode_enabled())

    def test_decode_cuda_availability_needs_an_sm70_device(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.get_device_capability") as cap,
        ):
            self.assertFalse(self._decode_cuda.sm70_cuda_decode_available())
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.get_device_capability", return_value=(7, 0)),
            ):
                self.assertTrue(self._decode_cuda.sm70_cuda_decode_available())
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            ):
                self.assertFalse(self._decode_cuda.sm70_cuda_decode_available())
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch(
                    "torch.cuda.get_device_capability",
                    side_effect=RuntimeError("driver gone"),
                ),
            ):
                self.assertFalse(self._decode_cuda.sm70_cuda_decode_available())
        cap.assert_not_called()

    def test_decode_cuda_env_override_reaches_load_inline(self):
        with _writable_dir_for_env("SGLANG_V100_DECODE_CUDA_BUILD_DIR") as target:
            with (
                patch("os.makedirs"),
                patch(
                    "torch.utils.cpp_extension.load_inline",
                    return_value="fake-ext",
                ) as load_inline,
            ):
                self.assertEqual(
                    self._decode_cuda._load_sm70_cuda_decode_ops(), "fake-ext"
                )
            self.assertEqual(load_inline.call_args.kwargs["build_directory"], target)

    def test_decode_cuda_default_build_dir(self):
        with (
            _home_only_environ(),
            patch("os.makedirs"),
            patch(
                "torch.utils.cpp_extension.load_inline", return_value="fake-ext"
            ) as load_inline,
        ):
            self.assertEqual(self._decode_cuda._load_sm70_cuda_decode_ops(), "fake-ext")
        self.assertEqual(
            load_inline.call_args.kwargs["build_directory"], DECODE_DEFAULT
        )

    def test_decode_cuda_load_failure_returns_none_and_stays_attempted(self):
        with _writable_dir_for_env("SGLANG_V100_DECODE_CUDA_BUILD_DIR"):
            with (
                patch("os.makedirs"),
                patch(
                    "torch.utils.cpp_extension.load_inline",
                    side_effect=RuntimeError("nvcc boom"),
                ),
            ):
                self.assertIsNone(self._decode_cuda._load_sm70_cuda_decode_ops())
                self.assertTrue(self._decode_cuda._OPS_LOAD_ATTEMPTED)
                self.assertIsNone(self._decode_cuda._load_sm70_cuda_decode_ops())

    def test_decode_cuda_missing_source_is_a_quiet_none(self):
        with patch.object(
            self._decode_cuda, "_SRC_PATH", pathlib.Path("/no/such/file.cu")
        ):
            with patch("torch.utils.cpp_extension.load_inline") as load_inline:
                self.assertIsNone(self._decode_cuda._load_sm70_cuda_decode_ops())
        load_inline.assert_not_called()

    # -- SGLANG_V100_NVFP4_MOE_BUILD_DIR -------------------------------------

    def test_nvfp4_moe_cached_ext_short_circuits_the_loader(self):
        self._nvfp4_moe._EXT = "cached-ext"
        with patch("torch.utils.cpp_extension.load_inline") as load_inline:
            self.assertEqual(
                self._nvfp4_moe._load_sm70_nvfp4_moe_decode_ops(), "cached-ext"
            )
        load_inline.assert_not_called()

    def test_nvfp4_moe_env_override_reaches_load_inline(self):
        with _writable_dir_for_env("SGLANG_V100_NVFP4_MOE_BUILD_DIR") as target:
            with (
                patch("os.makedirs"),
                patch(
                    "torch.utils.cpp_extension.load_inline",
                    return_value="fake-ext",
                ) as load_inline,
            ):
                self.assertEqual(
                    self._nvfp4_moe._load_sm70_nvfp4_moe_decode_ops(), "fake-ext"
                )
            self.assertEqual(load_inline.call_args.kwargs["build_directory"], target)

    def test_nvfp4_moe_default_build_dir(self):
        with (
            _home_only_environ(),
            patch("os.makedirs"),
            patch(
                "torch.utils.cpp_extension.load_inline", return_value="fake-ext"
            ) as load_inline,
        ):
            self.assertEqual(
                self._nvfp4_moe._load_sm70_nvfp4_moe_decode_ops(), "fake-ext"
            )
        self.assertEqual(load_inline.call_args.kwargs["build_directory"], NVFP4_DEFAULT)

    def test_nvfp4_moe_load_failure_returns_none_and_stays_attempted(self):
        with _writable_dir_for_env("SGLANG_V100_NVFP4_MOE_BUILD_DIR"):
            with (
                patch("os.makedirs"),
                patch(
                    "torch.utils.cpp_extension.load_inline",
                    side_effect=RuntimeError("nvcc boom"),
                ),
            ):
                self.assertIsNone(self._nvfp4_moe._load_sm70_nvfp4_moe_decode_ops())
                self.assertTrue(self._nvfp4_moe._LOAD_ATTEMPTED)
                self.assertIsNone(self._nvfp4_moe._load_sm70_nvfp4_moe_decode_ops())

    def test_nvfp4_moe_missing_source_is_a_quiet_none(self):
        with patch.object(
            self._nvfp4_moe, "_SRC_PATH", pathlib.Path("/no/such/file.cu")
        ):
            with patch("torch.utils.cpp_extension.load_inline") as load_inline:
                self.assertIsNone(self._nvfp4_moe._load_sm70_nvfp4_moe_decode_ops())
        load_inline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
