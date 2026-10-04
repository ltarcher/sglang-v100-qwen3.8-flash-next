from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.dsa.kpool_plan import (
    init_kpool_extend_metadata,
    init_kpool_write_plan,
    init_kpool_write_plan_capture,
    init_pooled_paged_mqa_metadata,
    update_kpool_write_plan,
    update_pooled_paged_mqa_metadata,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
        PrecomputedMetadata,
    )
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.layers.attention.dsa_backend import _DSA_IMPL_T, DSAMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass
class _KPoolForwardInputs:
    full_real_page_table: Optional[torch.Tensor] = None
    full_seqlens_expanded: Optional[torch.Tensor] = None


class DeepseekSparseAttnBackendKPoolMixin:
    """KPool-specific metadata and tail handling for the DSA backend."""

    def _check_kpool_tail_backend(
        self,
        topk_indices: Optional[torch.Tensor],
        dsa_impl: _DSA_IMPL_T,
        phase: str,
    ) -> None:
        if (
            topk_indices is None
            or self.dsa_index_kpool <= 1
            or dsa_impl in ("fa3", "tilelang", "trtllm")
        ):
            return
        raise NotImplementedError(
            "index_kpool > 1 appends tail tokens to topk_indices and is "
            f"currently only supported by the FA3/TileLang/TRTLLM DSA {phase} "
            "backend."
        )

    def _resolve_kpool_tail_backend(
        self,
        topk_indices: Optional[torch.Tensor],
        dsa_impl: _DSA_IMPL_T,
    ) -> _DSA_IMPL_T:
        if (
            topk_indices is None
            or self.dsa_index_kpool <= 1
            or dsa_impl != "flashmla_sparse"
        ):
            return dsa_impl
        if self.device_sm_major >= 10:
            return "trtllm"
        if self.device_sm_major == 9:
            return "fa3"
        return dsa_impl

    def _kpool_slots_per_page(self) -> int:
        return getattr(self.token_to_kv_pool, "slots_per_page", self.real_page_size)

    def _build_kpool_paged_mqa_schedule_metadata(self) -> bool:
        # Only the deep_gemm paged-MQA kernel consumes the schedule
        # metadata; it exists on sm90+ only. sm70 runs the fp16 logits
        # kernel, which needs no scheduler.
        if self.device_sm_major == 9:
            return self.num_q_heads in (32, 64)
        if self.device_sm_major < 9:
            return False
        return True

    def _init_kpool_metadata(
        self,
        metadata: DSAMetadata,
        forward_batch: ForwardBatch,
        topk_transform_method: Optional[TopkTransformMethod] = None,
        kpool_inputs: Optional[_KPoolForwardInputs] = None,
    ) -> DSAMetadata:
        if self.dsa_index_kpool <= 1:
            return metadata

        forward_mode = forward_batch.forward_mode
        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_extend_without_speculative():
            assert topk_transform_method is not None
            assert kpool_inputs is not None
            return init_kpool_extend_metadata(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                topk_transform_method=topk_transform_method,
                full_real_page_table=kpool_inputs.full_real_page_table,
                full_seqlens_expanded=kpool_inputs.full_seqlens_expanded,
            )

        if forward_mode.is_decode_or_idle():
            metadata = init_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=1,
                write_start=(forward_batch.seq_lens - 1).to(torch.int32),
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if forward_mode.is_target_verify():
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=self.speculative_num_draft_tokens,
                write_start=forward_batch.seq_lens.to(torch.int32),
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if forward_mode.is_draft_extend_v2():
            spec_info = forward_batch.spec_info
            effective_n_per_batch = (
                spec_info.num_accept_tokens
                if spec_info is not None
                and getattr(spec_info, "num_accept_tokens", None) is not None
                else None
            )
            return init_kpool_write_plan(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=self.speculative_num_draft_tokens,
                write_start=(
                    forward_batch.seq_lens - self.speculative_num_draft_tokens
                ).to(torch.int32),
                slots_per_page=slots_per_page,
                effective_n_per_batch=effective_n_per_batch,
                build_schedule_metadata=build_schedule_metadata,
            )

        return metadata

    def _init_kpool_metadata_capture(
        self, metadata: DSAMetadata, bs: int, forward_mode: ForwardMode
    ) -> DSAMetadata:
        if self.dsa_index_kpool <= 1:
            return metadata

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            metadata = init_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            is_decode = forward_mode.is_decode_or_idle()
            is_v2 = forward_mode.is_draft_extend_v2()
            metadata = init_kpool_write_plan_capture(
                metadata,
                max_bs=bs,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                num_draft_tokens=(
                    1 if is_decode else self.speculative_num_draft_tokens
                ),
                device=self.device,
                is_verify=not is_decode,
                slots_per_page=slots_per_page,
                is_v2=is_v2,
                build_schedule_metadata=build_schedule_metadata,
            )

        return metadata

    def _update_kpool_metadata_replay(
        self,
        metadata: DSAMetadata,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        effective_n_per_batch: Optional[torch.Tensor] = None,
    ) -> None:
        if self.dsa_index_kpool <= 1:
            return

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            update_pooled_paged_mqa_metadata(
                metadata,
                metadata.cache_seqlens_int32,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if not (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            return

        is_decode = forward_mode.is_decode_or_idle()
        is_v2 = forward_mode.is_draft_extend_v2()
        if is_decode:
            write_start = seq_lens.to(torch.int32) - 1
        elif is_v2:
            write_start = seq_lens.to(torch.int32) - self.speculative_num_draft_tokens
        else:
            # Target verify: write_start == seq_lens exactly; the plan kernel
            # casts on load, so skip the per-replay int32 alloc + conversion.
            write_start = seq_lens
        update_kpool_write_plan(
            metadata,
            write_start=write_start,
            req_pool_indices=req_pool_indices,
            real_page_table=metadata.real_page_table,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
            num_draft_tokens=(1 if is_decode else self.speculative_num_draft_tokens),
            forward_mode=forward_mode,
            slots_per_page=slots_per_page,
            effective_n_per_batch=effective_n_per_batch,
        )
        if (
            os.environ.get("SGLANG_SM70_SPARSE_PROBE", "0") == "1"
            and forward_mode.is_target_verify()
            and getattr(self, "_replay_probe_count", 0) < 4
            and not torch.cuda.is_current_stream_capturing()
            and int(seq_lens.max().item()) > 64
        ):
            self._replay_probe_count = getattr(self, "_replay_probe_count", 0) + 1
            probe_plan = metadata.kpool_write_plan
            torch.cuda.synchronize()
            pt1 = metadata.page_table_1
            plan_desc = (
                "None"
                if probe_plan is None or probe_plan.pool_seqlens_per_q is None
                else f"per_q={probe_plan.pool_seqlens_per_q[:4].tolist()} "
                f"ws={probe_plan.write_start.tolist()} "
                f"ptr={probe_plan.pool_seqlens_per_q.data_ptr():#x}"
            )
            # Graph-owned verify buffers stashed at capture: after the previous
            # replay they hold that step's actual in-graph kernel outputs.
            from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (
                _GRAPH_PROBE_STATE,
            )

            # Per-layer NaN census over every stashed verify layer of this
            # graph tier: localizes the first layer whose in-graph q went
            # non-finite (KDA-side upstream vs DSA-sparse-side).
            census = "unset"
            try:
                parts = []
                for (mid, lay), st in sorted(
                    _GRAPH_PROBE_STATE.items(), key=lambda kv: kv[0][1]
                ):
                    if mid != id(metadata):
                        continue
                    qf = st["q"].float()
                    parts.append(
                        f"L{lay}[nan={int(torch.isnan(qf).sum().item())}"
                        f"/{qf.numel()} abs={qf.nan_to_num().abs().max().item():.3g}]"
                    )
                census = " ".join(parts) if parts else "none"
            except Exception as e:  # pragma: no cover - probe only
                census = f"err:{e}"
            blk_census = "unset"
            try:
                from sglang.srt.models.glm5_next import _BLK_NAN_PROBE

                parts = []

                def _blk_key(kv):
                    head = kv[0][1].split(".")[0]
                    return (
                        (int(head) if head.isdigit() else 99),
                        kv[0][1],
                    )

                for (shape, tag), t in sorted(_BLK_NAN_PROBE.items(), key=_blk_key):
                    head = tag.split(".")[0]
                    if head == "tail":
                        pass  # tail.norm: final-norm output, keep unconditionally
                    else:
                        blk = int(head)
                        if not (4 <= blk <= 12 or 43 <= blk <= 44):
                            continue
                    tf = t.float()
                    parts.append(
                        f"{tag}[nan={int(torch.isnan(tf).sum().item())}"
                        f"/{tf.numel()} abs={tf.nan_to_num().abs().max().item():.3g}]"
                    )
                blk_census = " ".join(parts) if parts else "none"
            except Exception as e:  # pragma: no cover - probe only
                blk_census = f"err:{e}"

            # L7 was pinned as the first all-NaN sparse attention; dump the
            # full indexer + sparse chain for the bracketing layers.
            LAYERS = (3, 7, 11, 43)
            g_desc = "unset"
            try:
                parts = []
                for lay in LAYERS:
                    st = _GRAPH_PROBE_STATE.get((id(metadata), lay))
                    if st is None:
                        continue
                    lf = st["logits"].float()
                    ps = st["pool_seqlens"]
                    n_pages = int((int(ps[0].item()) + 63) // 64) + 1
                    region = st["kbuf"][st["pool_bt"][0, :n_pages]]
                    parts.append(
                        f"L{lay}:logits[abs={lf.abs().max().item():.3g} "
                        f"nan={int(torch.isnan(lf).sum().item())} "
                        f"zero={(lf == 0).float().mean().item():.2f}] "
                        f"res0={st['result'][0, :4].tolist()} "
                        f"ps0={int(ps[0].item())} "
                        f"bt0={st['pool_bt'][0, :3].tolist()} "
                        f"knz={(region != 0).float().mean().item():.3f}"
                    )
                g_desc = " | ".join(parts) if parts else "none"
            except Exception as e:  # pragma: no cover - probe only
                g_desc = f"err:{e}"
            sp_desc = "unset"
            try:
                from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (
                    _SPARSE_PROBE_STATE,
                )

                parts = []
                for lay in LAYERS:
                    st = _SPARSE_PROBE_STATE.get((id(metadata), lay))
                    if st is None:
                        continue
                    of = st["out"].float()
                    pt1 = st["pt1"]
                    pool = st["pool"]
                    qf = st.get("q")
                    if qf is not None:
                        qfa = qf.float()
                        q_stat = (
                            f"q[abs={qfa.nan_to_num().abs().max().item():.3g} "
                            f"nan={int(torch.isnan(qfa).sum().item())}]"
                        )
                    else:
                        q_stat = "q[none]"
                    # Byte stride of one pool slot row; block-128 scales are
                    # the trailing 16 bytes (4 x fp32) of each 528-byte row.
                    flat = pool.view(torch.uint8).reshape(-1)
                    row_bytes = pool.element_size() * pool.shape[-1]
                    n_slots = pool.numel() // pool.shape[-1]
                    row0 = pt1[0].to(torch.int64)
                    valid = row0 >= 0
                    vs = row0[valid]
                    sel = vs[:2048]
                    payload = flat[
                        (sel * row_bytes)[:, None]
                        + torch.arange(512, device=pool.device)[None, :]
                    ]
                    sc = (
                        flat[
                            (sel * row_bytes + 512)[:, None]
                            + torch.arange(16, device=pool.device)[None, :]
                        ]
                        .view(torch.float32)
                        .float()
                    )
                    parts.append(
                        f"L{lay}:sp[out_nan={int(torch.isnan(of).sum().item())}"
                        f"/{of.numel()} out_abs={of.nan_to_num().abs().max().item():.3g} "
                        f"{q_stat} "
                        f"pt1[min={int(vs.min().item())} max={int(vs.max().item())} "
                        f"neg={int((~valid).sum().item())} "
                        f"oor={int((vs >= n_slots).sum().item())} "
                        f"n={int(vs.numel())}] "
                        f"pool[e4m3nan={int(((payload == 0x7F) | (payload == 0xFF)).sum().item())} "
                        f"sc_nan={int(torch.isnan(sc).sum().item())} "
                        f"sc_inf={int(torch.isinf(sc).sum().item())}]]"
                    )
                sp_desc = " | ".join(parts) if parts else "none"
            except Exception as e:  # pragma: no cover - probe only
                sp_desc = f"err:{e}"
            print(
                f"[replay-glue] verify md={id(metadata)} "
                f"seq_lens={seq_lens.tolist()} "
                f"pt1_row0={None if pt1 is None else pt1[0, :6].tolist()} "
                f"dsa_exp={metadata.dsa_seqlens_expanded[:4].tolist()} "
                f"{plan_desc} "
                f"rt_row0={metadata.real_page_table[0, :6].tolist()} "
                f"qcensus[{census}] "
                f"blkcensus[{blk_census}] "
                f"idx[{g_desc}] "
                f"sp[{sp_desc}]",
                flush=True,
            )
            # Second stage, after the census is safely on the log: persist the
            # real sparse-call inputs and replay the kernel eagerly with them.
            # A NaN here (where the in-graph call NaNs) convicts the kernel on
            # values; finite output convicts graph-resident state instead.
            if self._replay_probe_count == 1:
                try:
                    rank = torch.distributed.get_rank()
                except Exception:  # noqa: BLE001 -- probe only
                    rank = 0
                if rank == 0:
                    self._sparse_eager_repro(metadata)

    @staticmethod
    def _sparse_eager_repro(metadata: DSAMetadata) -> None:
        """Persist real sparse-call inputs and replay the kernel eagerly.

        Runs once per boot, outside any graph, right after the census print.
        A NaN here convicts the kernel on values; a finite output convicts
        graph-resident state instead. An illegal-memory access here would
        take the engine down, but by then the census is already on the log.
        """
        from sglang.kernels.ops.attention.dsa.tilelang_sparse_sm70 import (
            tilelang_sparse_fwd_sm70,
        )
        from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (
            _SPARSE_PROBE_STATE,
        )

        os.makedirs("/tmp/u2probe", exist_ok=True)
        for lay in (3, 7):
            try:
                st = _SPARSE_PROBE_STATE.get((id(metadata), lay))
                if st is None:
                    continue
                torch.save(
                    {
                        "q": st["q"],
                        "pt1": st["pt1"],
                        "pool": st["pool"],
                        "scale": st["scale"],
                        "d_v": st["d_v"],
                        "graph_out": st["out"],
                    },
                    f"/tmp/u2probe/sparse_L{lay}.pt",
                )
                # Mirror _forward_tilelang: the kernel consumes 64-column
                # blocks, so mask-pad the tail-extended table first.
                pt1 = st["pt1"]
                padding = (-pt1.shape[-1]) % 64
                if padding:
                    pt1 = torch.cat(
                        (pt1, pt1.new_full((*pt1.shape[:-1], padding), -1)),
                        dim=-1,
                    )
                eager = tilelang_sparse_fwd_sm70(
                    q=st["q"],
                    kv=st["pool"],
                    indices=pt1.unsqueeze(1),
                    sm_scale=st["scale"],
                    d_v=st["d_v"],
                )
                torch.cuda.synchronize()
                ef = eager.float()
                gf = st["out"].float()
                diff = (ef - gf).nan_to_num().abs().max().item()
                print(
                    f"[sparse-eager] L{lay} md={id(metadata)} "
                    f"out_nan={int(torch.isnan(ef).sum().item())}"
                    f"/{ef.numel()} "
                    f"abs={ef.nan_to_num().abs().max().item():.3g} "
                    f"vs_graph={diff:.3g}",
                    flush=True,
                )
            except Exception as e:  # pragma: no cover - probe only
                print(f"[sparse-eager] L{lay} err: {e}", flush=True)

    def _update_kpool_metadata_from_precomputed(
        self,
        metadata: DSAMetadata,
        precomputed: PrecomputedMetadata,
        forward_mode: ForwardMode,
    ) -> None:
        if self.dsa_index_kpool <= 1:
            return

        slots_per_page = self._kpool_slots_per_page()
        build_schedule_metadata = self._build_kpool_paged_mqa_schedule_metadata()
        if forward_mode.is_decode_or_idle():
            update_pooled_paged_mqa_metadata(
                metadata,
                precomputed.cache_seqlens,
                forward_mode,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                slots_per_page=slots_per_page,
                build_schedule_metadata=build_schedule_metadata,
            )

        if not (forward_mode.is_decode_or_idle() or forward_mode.is_target_verify()):
            return

        is_verify = forward_mode.is_target_verify()
        write_start = precomputed.cache_seqlens.to(torch.int32)
        write_start = (
            write_start - self.speculative_num_draft_tokens
            if is_verify
            else write_start - 1
        )
        update_kpool_write_plan(
            metadata,
            write_start=write_start,
            req_pool_indices=precomputed.req_pool_indices,
            real_page_table=metadata.real_page_table,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
            num_draft_tokens=self.speculative_num_draft_tokens if is_verify else 1,
            forward_mode=forward_mode,
            slots_per_page=slots_per_page,
        )
        if (
            os.environ.get("SGLANG_SM70_SPARSE_PROBE", "0") == "1"
            and is_verify
            and not getattr(self, "_precomputed_probe_fired", False)
            and not torch.cuda.is_current_stream_capturing()
            and int(precomputed.cache_seqlens.max().item()) > 64
        ):
            self._precomputed_probe_fired = True
            probe_plan = metadata.kpool_write_plan
            torch.cuda.synchronize()
            pt1 = metadata.page_table_1
            pc_rt = precomputed.real_page_table
            pc_pgidx = precomputed.page_indices
            plan_desc = (
                "None"
                if probe_plan is None or probe_plan.pool_seqlens_per_q is None
                else f"per_q={probe_plan.pool_seqlens_per_q[:4].tolist()} "
                f"ws={probe_plan.write_start.tolist()} "
                f"ptr={probe_plan.pool_seqlens_per_q.data_ptr():#x}"
            )
            print(
                f"[replay-glue-pc] verify md={id(metadata)} "
                f"pc_cache={precomputed.cache_seqlens[:4].tolist()} "
                f"pc_req={precomputed.req_pool_indices[:4].tolist()} "
                f"pc_maxk={precomputed.max_seqlen_k} "
                f"pc_pgidx_row0={None if pc_pgidx is None else pc_pgidx[0, :6].tolist()} "
                f"pc_rt_row0={None if pc_rt is None else pc_rt[0, :6].tolist()} "
                f"pt1_row0={None if pt1 is None else pt1[0, :6].tolist()} "
                f"dsa_exp={metadata.dsa_seqlens_expanded[:4].tolist()} "
                f"{plan_desc} "
                f"rt_row0={metadata.real_page_table[0, :6].tolist()}",
                flush=True,
            )
