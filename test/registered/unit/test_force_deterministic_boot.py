"""SGLANG_FORCE_DETERMINISTIC flips the CUDA determinism switches at boot.

The staged PLE gather has a small intrinsic numeric delta vs the pinned path;
non-deterministic CUDA (cudnn autotune, split-KV reduction order) amplifies it
until the first-token argmax flips and MTP verify rejects every draft. The env
must set CUBLAS_WORKSPACE_CONFIG, enable deterministic algorithms (warn_only,
so an unlisted op degrades instead of raising) and pin cudnn/tf32 before torch
is otherwise touched — all module-level in sglang.launch_server, exercised here
through a fresh interpreter import.
"""

import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-small")

_PROBE = (
    "import os, torch\n"
    "import sglang.launch_server\n"
    "print('DETPROBE',\n"
    "      os.environ.get('CUBLAS_WORKSPACE_CONFIG'),\n"
    "      torch.are_deterministic_algorithms_enabled(),\n"
    "      torch.backends.cudnn.deterministic,\n"
    "      torch.backends.cudnn.benchmark,\n"
    "      torch.backends.cuda.matmul.allow_tf32,\n"
    "      torch.backends.cudnn.allow_tf32,\n"
    "      flush=True)\n"
)


def _boot_probe(force_value: str | None) -> str:
    """Import sglang.launch_server in a clean interpreter with the env pinned.

    Only the vars the import needs to run are passed; SGLANG_FORCE_DETERMINISTIC
    is set to *force_value* ("1", "0", or dropped entirely).
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "PYTHONPATH", "TMPDIR")
        and not key.startswith("SGLANG_")
    }
    if force_value is not None:
        env["SGLANG_FORCE_DETERMINISTIC"] = force_value
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ),
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"probe failed (rc={proc.returncode})\nstdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    return proc.stdout


def _parse_probe(stdout: str):
    line = next(line for line in stdout.splitlines() if line.startswith("DETPROBE"))
    return line.split()[1:]


class TestForceDeterministicBoot(CustomTestCase):
    def test_env_on_pins_cublas_cudnn_and_tf32(self):
        stdout = _boot_probe("1")
        self.assertIn("[SGLANG_FORCE_DETERMINISTIC]", stdout)
        (
            cublas,
            deterministic,
            cudnn_deterministic,
            cudnn_benchmark,
            matmul_tf32,
            cudnn_tf32,
        ) = _parse_probe(stdout)
        self.assertEqual(cublas, ":4096:8")
        self.assertEqual(deterministic, "True")
        self.assertEqual(cudnn_deterministic, "True")
        self.assertEqual(cudnn_benchmark, "False")
        self.assertEqual(matmul_tf32, "False")
        self.assertEqual(cudnn_tf32, "False")

    def test_env_off_keeps_nondeterministic_defaults(self):
        for force_value in ("0", None):
            with self.subTest(force_value=force_value):
                stdout = _boot_probe(force_value)
                (_, deterministic, *_rest) = _parse_probe(stdout)
                self.assertEqual(deterministic, "False")
                self.assertNotIn("[SGLANG_FORCE_DETERMINISTIC]", stdout)


if __name__ == "__main__":
    unittest.main()
