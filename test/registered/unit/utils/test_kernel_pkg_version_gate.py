"""The kernel-package version gate honors SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.

V100 is pinned to sglang-kernel 0.4.6.post1 while upstream bumps the floor;
the production compose sets the skip env so the pinned package passes. The
gate is a whitelist: only the four kernel package names (dash/underscore
variants of flashinfer and sglang-kernel) may be skipped, everything else
must still be enforced even with the env set.
"""

import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.utils.common import (
    _should_skip_kernel_pkg_version_check,
    assert_pkg_version,
    check_pkg_version_at_least,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

COMMON = "sglang.srt.utils.common"


class TestKernelPackageVersionGate(CustomTestCase):
    def test_skip_is_scoped_to_the_kernel_package_whitelist(self):
        with envs.SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.override(True):
            self.assertTrue(_should_skip_kernel_pkg_version_check("sglang-kernel"))
            self.assertTrue(_should_skip_kernel_pkg_version_check("sglang_kernel"))
            self.assertTrue(_should_skip_kernel_pkg_version_check("flashinfer-python"))
            self.assertTrue(_should_skip_kernel_pkg_version_check("flashinfer_python"))
            # Not a kernel package: the env must not blanket-disable checks.
            self.assertFalse(_should_skip_kernel_pkg_version_check("torch"))
            self.assertFalse(_should_skip_kernel_pkg_version_check("transformers"))

    def test_skip_defaults_off(self):
        self.assertFalse(_should_skip_kernel_pkg_version_check("sglang-kernel"))

    def test_assert_pkg_version_skips_when_env_set(self):
        with envs.SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.override(True):
            with (
                patch(f"{COMMON}.version", return_value="0.0.1"),
                self.subTest(case="too old"),
            ):
                assert_pkg_version("sglang-kernel", "0.4.7.post1", "upgrade please")
            with (
                patch(
                    f"{COMMON}.version",
                    side_effect=PackageNotFoundError("sglang-kernel"),
                ),
                self.subTest(case="not installed"),
            ):
                assert_pkg_version("sglang-kernel", "0.4.7.post1", "upgrade please")

    def test_assert_pkg_version_enforces_when_env_unset(self):
        with envs.SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.override(False):
            with patch(f"{COMMON}.version", return_value="0.4.5"):
                with self.assertRaisesRegex(Exception, "0.4.5"):
                    assert_pkg_version("sglang-kernel", "0.4.6.post1", "upgrade please")
            with patch(f"{COMMON}.version", return_value="0.4.6.post1"):
                assert_pkg_version("sglang-kernel", "0.4.6.post1", "upgrade please")
            with (
                patch(
                    f"{COMMON}.version",
                    side_effect=PackageNotFoundError("sglang-kernel"),
                ),
                self.assertRaisesRegex(Exception, "not installed"),
            ):
                assert_pkg_version("sglang-kernel", "0.4.6.post1", "upgrade please")

    def test_assert_pkg_version_enforces_non_kernel_packages_even_when_env_set(self):
        with envs.SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.override(True):
            with patch(f"{COMMON}.version", return_value="0.0.1"):
                with self.assertRaises(Exception):
                    assert_pkg_version("some-other-pkg", "0.1.0", "upgrade please")

    def test_check_pkg_version_at_least_short_circuits_when_env_set(self):
        with envs.SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.override(True):
            with patch(f"{COMMON}.version", return_value="0.0.1"):
                self.assertTrue(check_pkg_version_at_least("sglang-kernel", "0.4.7"))

    def test_check_pkg_version_at_least_reports_reality_when_env_unset(self):
        with envs.SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.override(False):
            with patch(f"{COMMON}.version", return_value="0.4.6.post1"):
                self.assertTrue(check_pkg_version_at_least("sglang-kernel", "0.4.6"))
                self.assertFalse(check_pkg_version_at_least("sglang-kernel", "0.4.7"))
            with patch(
                f"{COMMON}.version",
                side_effect=PackageNotFoundError("sglang-kernel"),
            ):
                self.assertFalse(check_pkg_version_at_least("sglang-kernel", "0.4.6"))


if __name__ == "__main__":
    unittest.main()
