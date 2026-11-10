"""Unit tests for unmatched multimodal placeholder tokens in the legacy path.

When the prompt contains more image placeholder tokens than the request
attaches image items, ``legacy_load_mm_data`` must keep raising by default
(upstream behavior). With ``SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP=1`` the
unmatched tokens are dropped and the request proceeds instead of failing
with a 500 (agent clients echo ``<image>`` in text without the media).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import concurrent.futures
import unittest

from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    Modality,
    MultimodalSpecialTokens,
)
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    # CPU PIL decode so the test needs no GPU; __init__ is bypassed via
    # __new__ and only the attributes used by legacy_load_mm_data are set.
    gpu_image_decode = False

    async def process_mm_data_async(self, *args, **kwargs):  # abstract slot
        raise NotImplementedError


def _make_processor() -> _StubProcessor:
    processor = object.__new__(_StubProcessor)
    processor.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    return processor


def _red_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(255, 0, 0))


def _tokens() -> MultimodalSpecialTokens:
    return MultimodalSpecialTokens(image_token="<image>").build(processor=None)


def _run(processor, prompt, image_data):
    return asyncio.run(
        processor.legacy_load_mm_data(
            prompt=prompt,
            multimodal_tokens=_tokens(),
            image_data=image_data,
        )
    )


class TestUnmatchedTokenDefaultRaises(CustomTestCase):
    """Default: more tokens than data must keep failing (upstream behavior)."""

    def test_two_tokens_one_image_raises(self):
        processor = _make_processor()
        with self.assertRaisesRegex(RuntimeError, "loading multimodal data"):
            _run(processor, "a <image> b <image> c", [_red_image()])

    def test_token_without_any_image_raises(self):
        # Upstream: no data iterator at all -> ValueError, also a 500.
        processor = _make_processor()
        with self.assertRaisesRegex(ValueError, "No data iterator found"):
            _run(processor, "a <image> b", None)


class TestUnmatchedTokenDropEnabled(CustomTestCase):
    """Flag on: unmatched tokens are dropped, matched data still loads."""

    def test_two_tokens_one_image_drops_extra_token(self):
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "a <image> b <image> c", [_red_image()])
        self.assertEqual(len(output.images), 1)
        # The unmatched second placeholder must be removed from the text.
        self.assertEqual(output.input_text, "a <image> b  c")

    def test_token_without_any_image_drops_token(self):
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "a <image> b", None)
        self.assertEqual(len(output.images), 0)
        self.assertEqual(output.input_text, "a  b")

    def test_leading_and_trailing_unmatched_tokens(self):
        # Boundary: unmatched tokens at both ends of the prompt. The matched
        # placeholder is re-emitted as the canonical token, the trailing
        # unmatched one is dropped.
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "<image> a <image>", [_red_image()])
        self.assertEqual(len(output.images), 1)
        self.assertEqual(output.input_text, "<image> a ")

    def test_matched_counts_unchanged(self):
        # Boundary: token count == data count must behave exactly as before.
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "a <image> b", [_red_image()])
        self.assertEqual(len(output.images), 1)
        self.assertEqual(output.input_text, "a <image> b")

    def test_more_data_than_tokens_still_ok(self):
        # Boundary: extra data beyond tokens keeps the existing
        # warn-and-ignore behavior (no raise) with the flag on.
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = _run(processor, "a <image> b", [_red_image(), _red_image()])
        self.assertEqual(len(output.images), 1)
        self.assertEqual(output.input_text, "a <image> b")


def _precomputed_audio() -> dict:
    # A precomputed processor_output passes _load_single_item untouched, so
    # the test needs no audio decode stack.
    return {"format": "processor_output", "audio": [0.0, 0.1]}


class TestStopIterationScopedToModality(CustomTestCase):
    """m21 (docs/v100/CODE_REVIEW_TODO_20261110.md): a StopIteration inside
    submit_data_loading_tasks (a modality's data iterator running dry) used
    to return early and abandon the submission — every remaining text part of
    ANY modality lost its data. The skip must be scoped to the mismatched
    token: later parts (especially other modalities in a mixed prompt) keep
    loading and get consumed by their own tokens."""

    def _tokens_mixed(self) -> MultimodalSpecialTokens:
        return MultimodalSpecialTokens(
            image_token="<image>", audio_token="<audio>"
        ).build(processor=None)

    def _run_mixed(self, audio_data, image_data, **kw):
        processor = _make_processor()
        return asyncio.run(
            processor.legacy_load_mm_data(
                prompt="<audio> a <audio> b <image>",
                multimodal_tokens=self._tokens_mixed(),
                audio_data=audio_data,
                image_data=image_data,
                **kw,
            )
        )

    def test_mixed_prompt_unmatched_audio_keeps_later_image(self):
        # <audio>(match) <audio>(unmatched: iterator dry) <image>(must load).
        # Old behavior: the image token fell into the StopIteration handler
        # and its data was silently dropped (or raised by default).
        processor = _make_processor()
        with envs.SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP.override(True):
            output = self._run_mixed([_precomputed_audio()], [_red_image()])
        self.assertEqual(len(output.audios), 1)
        self.assertEqual(len(output.images), 1)
        # The unmatched audio placeholder is dropped; both matched tokens keep
        # their canonical placeholders.
        self.assertEqual(output.input_text, "<audio> a  b <image>")

    def test_mixed_prompt_default_policy_keeps_later_image(self):
        # Without the drop flag, the unmatched audio token is kept (the
        # request already has precomputed input, mirroring the precomputed
        # echo policy) and the later image still loads instead of being
        # abandoned by the early return.
        processor = _make_processor()
        output = self._run_mixed([_precomputed_audio()], [_red_image()])
        self.assertEqual(len(output.audios), 1)
        self.assertEqual(len(output.images), 1)
        self.assertEqual(output.input_text, "<audio> a <audio> b <image>")

    def test_submission_covers_every_special_token(self):
        # Loader-level contract: one future/task_info entry per special token,
        # with a None sentinel for the token whose data is missing.
        processor = _make_processor()
        text_parts = ["<audio>", " a ", "<audio>", " b ", "<image>"]
        data_iterators = {
            Modality.AUDIO: iter([_precomputed_audio()]),
            Modality.IMAGE: iter([_red_image()]),
        }
        futures, task_info = processor.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=self._tokens_mixed(),
            data_iterators=data_iterators,
        )
        self.assertEqual(len(futures), 3)
        self.assertEqual(len(task_info), 3)
        self.assertIsNone(task_info[1][0])  # sentinel for the dry token
        self.assertIsNotNone(task_info[0][0])
        self.assertIsNotNone(task_info[2][0])


if __name__ == "__main__":
    unittest.main()
