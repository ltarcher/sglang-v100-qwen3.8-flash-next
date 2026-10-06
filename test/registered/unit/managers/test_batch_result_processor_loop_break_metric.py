import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_LOOP_DETECTED,
    Req,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(finish_reason) -> Req:
    sampling_params = SamplingParams(max_new_tokens=32)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.output_ids.append(3)
    req.finished_reason = finish_reason
    return req


def _make_processor(
    metrics_collector, is_stats_logging_rank: bool = True
) -> SchedulerBatchResultProcessor:
    metrics_reporter = MagicMock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    metrics_reporter.is_stats_logging_rank = is_stats_logging_rank
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=SimpleNamespace(page_size=1),
        hisparse_coordinator=None,
        beam_coordinator=MagicMock(),
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=metrics_collector,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=MagicMock(),
        logprob_result_processor=None,
        output_streamer=MagicMock(),
        abort_request=lambda *args, **kwargs: None,
    )


class TestLoopBreakMetric(unittest.TestCase):
    """A loop-finished request bumps sglang:num_output_loop_breaks_total
    exactly once; other finish reasons and enable_metrics=off never touch it."""

    def _run_handle(self, req, enable_metrics, collector, is_stats_logging_rank=True):
        processor = _make_processor(collector, is_stats_logging_rank)
        batch = SimpleNamespace(
            mamba_track_mask_cpu=None,
            mamba_decode_batch_idx_cpu=None,
        )
        logits_output = SimpleNamespace(
            hidden_states=None, customized_info=None, sampling_mask_output=None
        )
        ctx = dict(
            get_exec=SimpleNamespace(
                mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
            ),
            get_disagg=SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            ),
            get_memory=SimpleNamespace(enable_hisparse=False),
            get_observability=SimpleNamespace(enable_metrics=enable_metrics),
        )
        module = "sglang.srt.managers.scheduler_components.batch_result_processor"
        with (
            patch(f"{module}.get_exec", return_value=ctx["get_exec"]),
            patch(f"{module}.get_disagg", return_value=ctx["get_disagg"]),
            patch(f"{module}.get_memory", return_value=ctx["get_memory"]),
            patch(f"{module}.get_observability", return_value=ctx["get_observability"]),
            patch(f"{module}.release_kv_cache"),
        ):
            processor._handle_finish_state_updated_req(
                req, batch, None, 0, logits_output
            )

    def test_loop_finish_increments_counter(self):
        collector = MagicMock()
        req = _make_req(FINISH_LOOP_DETECTED(period=8, repeats=3, length=256))
        self._run_handle(req, enable_metrics=True, collector=collector)
        collector.increment_output_loop_breaks.assert_called_once_with()

    def test_other_finish_reason_does_not_increment(self):
        collector = MagicMock()
        req = _make_req(FINISH_LENGTH(length=32))
        self._run_handle(req, enable_metrics=True, collector=collector)
        collector.increment_output_loop_breaks.assert_not_called()

    def test_metrics_disabled_does_not_increment(self):
        collector = MagicMock()
        req = _make_req(FINISH_LOOP_DETECTED(period=8, repeats=3, length=256))
        self._run_handle(req, enable_metrics=False, collector=collector)
        collector.increment_output_loop_breaks.assert_not_called()

    def test_non_stats_logging_rank_does_not_increment(self):
        # Every TP rank runs the loop check; only the stats-logging rank
        # counts, else sum() over the tp_rank label series multiplies the
        # true loop count by tp_size.
        collector = MagicMock()
        req = _make_req(FINISH_LOOP_DETECTED(period=8, repeats=3, length=256))
        self._run_handle(
            req, enable_metrics=True, collector=collector, is_stats_logging_rank=False
        )
        collector.increment_output_loop_breaks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
