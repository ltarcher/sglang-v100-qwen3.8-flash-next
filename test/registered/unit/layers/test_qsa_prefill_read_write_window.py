"""QSA prefill read-window vs write-plan audit (CPU-only, no GPU kernels).

Incident context (4xV100, TP4, page_size=64, ratio=4): a probe whose extend
lands on pages recycled from an earlier 32k row is poisoned, while the row
itself is not. The suspect channel is the QSA compressed-K scoring read
window being strictly larger than the current forward's compressed write
plan.

Production expressions under audit (file:line as of this audit):
- Write plan: qwen_sparse_attn_backend.py:814-862 `_qsa_build_write_plan`
  (assert prefix % ratio == 0 at :846, start_blocks = prefix//ratio,
  end_blocks = lengths//ratio, compressed address = token_slot_table[row,
  b*ratio] // ratio).
- Scoring read window: qsa/metadata.py:170-183 `get_prefill_mqa_inputs`
  (complete_blocks = seq_len // ratio, compressed_locs =
  token_slot_table[seq, :complete_blocks*ratio:ratio] // ratio) and
  qsa/metadata.py:16-42 `build_qsa_row_ranges` (row_ends capped at
  min((pos+1)//ratio, seq_len//ratio)).
- Attention read expansion: qsa/kernel.py:93-147
  `torch_expand_qsa_block_indices` (tokens masked to < seq_len, tail
  restricted to [tail_start, seq_len)).

Tests assert the "read indices subset-of written indices" invariants at the
pure index-derivation layer. The strict invariant (scoring window subset of
THIS forward's write plan) is intentionally expected-failure: the prefix
difference set [0, prefix//ratio) is read from the compressed-K pool with
no freshness/epoch guard, which is the documented unguarded window.
"""

import unittest

import torch

from sglang.srt.layers.attention.qsa.kernel import torch_expand_qsa_block_indices
from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 64
RATIO = 4
TOKEN_TOPK = 2048
BLOCK_TOPK = TOKEN_TOPK // RATIO


def write_plan_groups(prefix_len: int, seq_len: int) -> list:
    """Faithful transcription of _qsa_build_write_plan (backend.py:814-862).

    Pure-extend rows: assert prefix % ratio == 0 (:846), start_blocks =
    prefix//ratio, end_blocks = lengths//ratio. The group's compressed slot
    address is token_slot_table[row, b*ratio] // ratio.
    """
    assert prefix_len % RATIO == 0, backend_assert_msg(prefix_len)
    return list(range(prefix_len // RATIO, seq_len // RATIO))


def backend_assert_msg(prefix_len: int) -> str:
    return f"prefix {prefix_len} not ratio-aligned (backend.py:846)"


def scoring_read_groups(seq_len: int) -> list:
    """Faithful transcription of get_prefill_mqa_inputs (metadata.py:170-183):
    complete_blocks = seq_len // ratio; reads every complete group
    [0, complete_blocks)."""
    return list(range(seq_len // RATIO))


def compressed_slot(raw_slot: int) -> int:
    """Compressed-K pool address for a raw token slot (slot // ratio)."""
    return raw_slot // RATIO


def probe_slot_map(prefix_len: int, extend_len: int, recycled_page: int) -> list:
    """Raw slot per logical position for the incident shape.

    Prefix [0, prefix_len) sits on radix pages 64.. (warm-probe pages,
    alive); extend sits on recycled pages handed out by the allocator
    (previously owned by the 32k row).
    """
    slots = [64 * PAGE_SIZE + i for i in range(prefix_len)]
    slots += [recycled_page * PAGE_SIZE + i for i in range(extend_len)]
    return slots


class TestQsaPrefillReadWriteWindow(CustomTestCase):
    # ------------------------------------------------------------------
    # Green guards: the index algebra is internally consistent.
    # ------------------------------------------------------------------

    def test_pure_extend_scoring_window_equals_write_plan(self):
        # Row scenario: 32k prefill, no radix prefix, fresh pages.
        # read window == write plan exactly (both derive from seq_len//ratio).
        for seq_len in (704, 640, 768, 32768):
            read = scoring_read_groups(seq_len)
            write = write_plan_groups(0, seq_len)
            self.assertEqual(read, write, f"seq_len={seq_len}")

    def test_radix_prefix_own_region_covered_by_write_plan(self):
        # Probe scenario: prefix 640 (radix, page+ratio aligned) + extend 704
        # on recycled pages. The read groups inside the extend region must be
        # exactly the write plan.
        prefix, extend = 640, 704
        seq_len = prefix + extend
        read = scoring_read_groups(seq_len)
        write = write_plan_groups(prefix, seq_len)
        own_region_read = [g for g in read if g >= prefix // RATIO]
        self.assertEqual(own_region_read, write)
        # Scoring rows never exceed the packed compressed history.
        starts, ends, cu = build_qsa_row_ranges(
            torch.tensor([seq_len], dtype=torch.int32),
            torch.tensor([seq_len - 1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            RATIO,
        )
        self.assertEqual(int(ends[0]), cu[1].item())
        self.assertEqual(int(ends[0]), len(read))

    def test_chunked_prefill_cumulative_write_covers_scoring_window(self):
        # Chunked row prefill: extends 640/704/768 in three forwards (each
        # ratio-aligned, fresh pages). The final scoring window must be
        # covered by the union of cumulative write plans.
        prefix = 0
        cumulative = set()
        for extend in (640, 704, 768):
            seq_len = prefix + extend
            cumulative.update(write_plan_groups(prefix, seq_len))
            prefix = seq_len
        self.assertEqual(sorted(cumulative), scoring_read_groups(prefix))

    def test_expand_token_reads_stay_within_sequence_bounds(self):
        # Adversarial selection: every valid block index maps into
        # [0, seq_len); tail entries stay inside [tail_start, seq_len).
        seq_len = 640 + 704
        rows = 5
        positions = torch.tensor([639, 1024, 1343, 1344, seq_len - 1])
        lens = torch.full((rows,), seq_len, dtype=torch.long)
        block_indices = torch.full((rows, BLOCK_TOPK), -1, dtype=torch.int32)
        block_indices[:, : BLOCK_TOPK // 2] = torch.randint(
            0, seq_len // RATIO, (rows, BLOCK_TOPK // 2)
        )
        block_indices[:, BLOCK_TOPK - 2] = seq_len // RATIO  # out-of-range block
        block_indices[:, BLOCK_TOPK - 1] = 0
        expanded = torch_expand_qsa_block_indices(
            block_indices, positions, lens, RATIO, TOKEN_TOPK
        )
        valid = expanded[expanded >= 0]
        self.assertTrue(bool(((valid >= 0) & (valid < seq_len)).all()))
        # Contiguity contract: valid entries packed at the front, per row.
        for r in range(rows):
            inv = (expanded[r] == -1).nonzero()
            first = int(inv[0]) if inv.numel() else expanded.shape[1]
            self.assertTrue(
                bool((expanded[r, first:] == -1).all()),
                f"row {r}: valid entries not contiguous at column {first}",
            )

    def test_compressed_address_of_recycled_group_is_owner_independent(self):
        # A recycled page's group compresses into the SAME pool address the
        # previous owner used (slot//ratio addressing is epoch-free).
        slots = probe_slot_map(prefix_len=640, extend_len=704, recycled_page=36288)
        probe_group_addr = compressed_slot(slots[640])  # first extend group
        row_slot_at_same_page = 36288 * PAGE_SIZE  # row-era slot at page 36288
        self.assertEqual(probe_group_addr, compressed_slot(row_slot_at_same_page))
        # Radix prefix groups keep their warm-probe-era addresses.
        self.assertEqual(compressed_slot(slots[0]), compressed_slot(64 * PAGE_SIZE))

    def test_mtp_verify_row_write_emits_only_completing_group(self):
        # Speculative paged rows (backend.py:904-909): lengths = positions+1,
        # prefix = length-1 is ratio-misaligned. The write plan emits at most
        # ONE group (only when the row completes a group); decode scoring
        # reads [0, len//ratio), which the cumulative history covers for a
        # request whose history was written in prior forwards.
        history_len = 1344  # 336 complete groups written before verify
        cumulative = set(range(history_len // RATIO))
        for pos in range(history_len, history_len + 5):
            length = pos + 1
            end_blocks = length // RATIO
            start_blocks = end_blocks - (1 if length % RATIO == 0 else 0)
            cumulative.update(range(start_blocks, end_blocks))
            # Decode scoring window after this step:
            read = list(range(length // RATIO))
            self.assertTrue(set(read).issubset(cumulative), f"pos={pos}")

    # ------------------------------------------------------------------
    # Red (expected failure): the strict invariant fails by design.
    # ------------------------------------------------------------------

    @unittest.expectedFailure
    def test_scoring_window_subseteq_this_forward_write_plan(self):
        # Strict "read indices subset-of written indices (this forward)"
        # invariant. It fails because the scoring read window
        # (metadata.py:177-183) covers ALL complete groups [0, seq_len//ratio)
        # while the write plan (backend.py:846-856) covers only
        # [prefix//ratio, seq_len//ratio). The difference set is read from a
        # compressed-K pool that is never cleared on slot free and carries no
        # epoch/freshness binding -- exactly the unguarded window through
        # which recycled-page residue (or a restored prefix whose compressed
        # entries were not recompressed) can poison the request.
        prefix, extend = 640, 704
        seq_len = prefix + extend
        read = set(scoring_read_groups(seq_len))
        write = set(write_plan_groups(prefix, seq_len))
        diff = sorted(read - write)
        diff_addrs = [compressed_slot(64 * PAGE_SIZE + g * RATIO) for g in diff]
        self.assertEqual(
            len(diff),
            0,
            "read window exceeds this forward's write plan by prefix groups "
            f"{diff[:8]}..{diff[-1]} (count={len(diff)}); their compressed-K "
            f"pool addresses {diff_addrs[:4]}.. are read with no freshness "
            "guard on the current slot epoch",
        )


if __name__ == "__main__":
    unittest.main()
