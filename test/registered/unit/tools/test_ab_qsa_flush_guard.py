"""Unit test: ab_qsa_grouped_quality_v100.py refuses the production port.

The A/B harness POSTs /flush_cache twice per arm (cold-prefill baseline +
G1 determinism gate). Pointed at the shared V100 production endpoint
(``SGLANG_V100_HOST/PORT``, default port 11435) it would silently evict the
live engine's radix cache -- review m24. The ``run`` subcommand must refuse
the production port unless the operator passes ``--assume-idle``.
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "ab_qsa_grouped_quality_v100.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=0, suite="base-a-test-cpu")


class TestAbQsaFlushCacheGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_module("ab_qsa_grouped_quality_v100", SCRIPT_PATH)

    def test_production_port_refused_without_assume_idle(self):
        with self.assertRaises(SystemExit) as cm:
            self.script.ensure_idle_test_target("http://127.0.0.1:11435", False)
        self.assertIn("11435", str(cm.exception))
        self.assertIn("--assume-idle", str(cm.exception))

    def test_production_port_allowed_with_assume_idle(self):
        # Explicit operator confirmation overrides the refusal.
        self.script.ensure_idle_test_target("http://127.0.0.1:11435", True)

    def test_idle_port_allowed(self):
        # The documented test-boot port (docstring examples) needs no flag.
        self.script.ensure_idle_test_target("http://127.0.0.1:8110", False)

    def test_run_refuses_before_any_request(self):
        # Wiring guard: run() must exit on the production port before sending
        # anything (the first flush POST would already wipe the cache).
        module = self.script
        module._tokenizer = lambda: None
        module.build_tasks = lambda tok: []
        module._filler_ids = lambda tok, target, seed: []
        module._generate = lambda base, payload: {}
        posted = []
        module.requests = SimpleNamespace(
            post=lambda url, **kw: posted.append(url) or SimpleNamespace(ok=True)
        )
        args = SimpleNamespace(
            arm="A1",
            base="http://127.0.0.1:11435",
            out=Path(self._tmp_out()),
            fresh=False,
            assume_idle=False,
        )
        with self.assertRaises(SystemExit):
            module.run(args)
        self.assertEqual(posted, [], "flush_cache must not be reached")

    @staticmethod
    def _tmp_out():
        import tempfile

        return tempfile.mkdtemp(prefix="ab_guard_test_")

    def test_run_idle_port_sends_two_flushes(self):
        module = self.script
        module._tokenizer = lambda: None
        # Cover the G1 gate's hardcoded tags so its next() lookup succeeds;
        # send() runs against stubbed network + extract only.
        module.build_tasks = lambda tok: [
            {"tag": "math0", "type": "math", "max_new": 8, "prompt": "p"},
            {"tag": "code0", "type": "code", "max_new": 8, "prompt": "p"},
            {"tag": "agent0", "type": "agent", "max_new": 8, "prompt": "p"},
            {"tag": "pelican_svg", "type": "misc", "max_new": 8, "prompt": "p"},
        ]
        module._filler_ids = lambda tok, target, seed: []
        module._generate = lambda base, payload: {}
        module._extract = lambda resp: dict(
            text="", ids=[], margins=[], prompt_tokens=None, completion_tokens=None
        )
        posted = []
        module.requests = SimpleNamespace(
            post=lambda url, **kw: posted.append(url) or SimpleNamespace(ok=True)
        )
        args = SimpleNamespace(
            arm="A1",
            base="http://127.0.0.1:8110",
            out=Path(self._tmp_out()),
            fresh=False,
            assume_idle=False,
        )
        self.assertEqual(module.run(args), 0)
        self.assertEqual([u.endswith("/flush_cache") for u in posted], [True, True])


if __name__ == "__main__":
    unittest.main()
