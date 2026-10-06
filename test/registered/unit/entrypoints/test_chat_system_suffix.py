import unittest

from sglang.srt.entrypoints.openai.serving_chat import append_chat_system_suffix
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SUFFIX = "Keep your thinking short."


class TestAppendChatSystemSuffix(unittest.TestCase):
    """SGLANG_CHAT_SYSTEM_SUFFIX folding: appended to the first system or
    developer message, never to later ones, and hosted by a new system
    message when the request has none or a non-string content list."""

    def test_appends_to_first_system_message(self):
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "hi"},
        ]
        append_chat_system_suffix(messages, SUFFIX)
        self.assertEqual(messages[0]["content"], f"You are a coding agent.\n\n{SUFFIX}")
        self.assertEqual(messages[1]["content"], "hi")

    def test_prefers_system_over_developer_and_skips_later_ones(self):
        messages = [
            {"role": "developer", "content": "dev rules"},
            {"role": "system", "content": "sys rules"},
        ]
        append_chat_system_suffix(messages, SUFFIX)
        self.assertEqual(messages[0]["content"], f"dev rules\n\n{SUFFIX}")
        self.assertEqual(messages[1]["content"], "sys rules")

    def test_empty_system_message_hosts_suffix_verbatim(self):
        messages = [{"role": "system", "content": ""}]
        append_chat_system_suffix(messages, SUFFIX)
        self.assertEqual(messages[0]["content"], SUFFIX)

    def test_inserts_system_message_when_none(self):
        messages = [{"role": "user", "content": "hi"}]
        append_chat_system_suffix(messages, SUFFIX)
        self.assertEqual(
            messages,
            [{"role": "system", "content": SUFFIX}, {"role": "user", "content": "hi"}],
        )

    def test_parts_list_content_gets_dedicated_system_message(self):
        parts = [{"type": "text", "text": "look"}]
        messages = [
            {"role": "system", "content": parts},
            {"role": "user", "content": "hi"},
        ]
        append_chat_system_suffix(messages, SUFFIX)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SUFFIX)
        self.assertEqual(messages[1]["content"], parts)

    def test_env_default_is_off(self):
        self.assertIsNone(envs.SGLANG_CHAT_SYSTEM_SUFFIX.get())


if __name__ == "__main__":
    unittest.main()
