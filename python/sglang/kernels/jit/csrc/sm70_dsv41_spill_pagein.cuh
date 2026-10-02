// SPDX-License-Identifier: Apache-2.0
// WO-13 D4-G: UVA page-in of spilled MXFP4 expert rows.
// Assigns misses to landing slots and copies host-mapped rows into a shared
// GPU landing pool. Decode routes here whenever a landing pool exists; wide
// (prefill/extend) batches opt in via SGLANG_DSV41_SPILL_PREFILL_LANDING
// when the pool holds every logical expert. This kernel is capturable.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_dsv41 {

// Per-call dedup bound: sized so a wide (prefill) batch can claim every
// logical expert of a layer in one call. assigned_id[]/assigned_slot[] are
// one-thread local arrays; at this size they live in local memory, which is
// fine for a single-block kernel. Decode calls never approach the bound
// (the dispatch gate keeps unique claims <= T*topk), so the larger arrays
// cost nothing there.
inline constexpr int32_t kMaxLanding = 512;
// Pool-slot cap: the persistent cache pool (== landing pool) may exceed the
// per-call dedup bound; slot_host_row walks and the LRU argmin scale with it.
inline constexpr int32_t kMaxPoolSlots = 512;
inline constexpr int32_t kMaxTensors = 8;
inline constexpr int32_t kCopyThreads = 256;
// Copy-grid byte splits per landing slot. One block per slot leaves a V100
// (80 SM) starved: measured 3.5 GB/s at landing=12 vs the 10.19 GB/s pin line
// rate. Active blocks = claimed slots x kCopySplits, and with the 48-slot pool
// the steady claim count is SMALL (main-region residency absorbs most routing;
// P1 measured ~2.3 claims/layer-call) -- so the split count must lift the
// low-claim case: 64 splits puts 2-4 claims at 128-256 blocks. High-claim
// steps just run more waves of 52 KiB chunks; blocks of unused slots exit
// without touching PCIe.
inline constexpr int32_t kCopySplits = 64;

/**
 * \brief Map topk logical ids onto kept GPU slots or landing indices.
 *
 * Hits (map_table[id] >= 0) stay on the layer Marlin table. Misses get a
 * landing slot 0..n_landing-1 (reused if the same id appears twice) and
 * topk_ids is set to -1 so the kept Marlin skips them.
 *
 * \param slot_host_row Output [n_landing]; host row for each used slot, else -1.
 */
__global__ void spill_assign_kernel(int32_t* __restrict__ topk_ids,
                                    int32_t* __restrict__ land_ids,
                                    int32_t* __restrict__ slot_host_row,
                                    const int32_t* __restrict__ map_table,
                                    const int32_t* __restrict__ host_map,
                                    int32_t n_tok,
                                    int32_t k,
                                    int32_t n_logical,
                                    int32_t n_landing) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t assigned_id[kMaxLanding];
  int32_t next = 0;
  for (int32_t s = 0; s < n_landing; ++s) {
    assigned_id[s] = -1;
    slot_host_row[s] = -1;
  }
  const int32_t n = n_tok * k;
  for (int32_t i = 0; i < n; ++i) {
    const int32_t id = topk_ids[i];
    if (id < 0 || id >= n_logical) {
      land_ids[i] = -1;
      continue;
    }
    const int32_t phys = map_table[id];
    if (phys >= 0) {
      topk_ids[i] = phys;
      land_ids[i] = -1;
      continue;
    }
    int32_t slot = -1;
    for (int32_t s = 0; s < next; ++s) {
      if (assigned_id[s] == id) {
        slot = s;
        break;
      }
    }
    if (slot < 0) {
      const int32_t hs = host_map[id];
      const int32_t cap = n_landing < kMaxLanding ? n_landing : kMaxLanding;
      if (hs < 0 || next >= cap) {
        topk_ids[i] = -1;
        land_ids[i] = -1;
        continue;
      }
      slot = next;
      assigned_id[slot] = id;
      slot_host_row[slot] = hs;
      ++next;
    }
    topk_ids[i] = -1;
    land_ids[i] = slot;
  }
}

/**
 * \brief Copy one expert row from a UVA host base into a landing slot.
 *
 * Grid.x is flat (n_landing * kCopySplits): block = slot + split * n_landing,
 * and each split copies a contiguous, 16 B-aligned byte range of every tensor
 * listed in src_ptrs/dst_ptrs/row_bytes. Unused slots (host_row < 0) return
 * immediately across all splits. Forcing a UVA dummy copy (host row 0 or the
 * first mapped row) made every launch copy landing-count expert rows over
 * PCIe: relaunch137 ~283 ms/launch, relaunch138 still ~5 s/verify. Marlin
 * never reads unused dest slots. Rank-invariant duration is not worth that.
 */
__global__ __launch_bounds__(kCopyThreads) void spill_copy_kernel(
    const int32_t* __restrict__ slot_host_row,
    const int64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ row_bytes,
    int32_t n_tensors,
    int32_t n_landing) {
  const int32_t slot = static_cast<int32_t>(blockIdx.x) % n_landing;
  const int32_t split = static_cast<int32_t>(blockIdx.x) / n_landing;
  const int32_t hs = slot_host_row[slot];
  if (hs < 0) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  for (int32_t t = 0; t < n_tensors; ++t) {
    const int64_t nbytes = row_bytes[t];
    if (nbytes <= 0) {
      continue;
    }
    // Split ranges stay 16 B-aligned so the uint4 body keeps its alignment
    // at any slot row (row_bytes themselves are 16 B multiples for the
    // Marlin-packed + scale tensors this pool carries).
    int64_t chunk = (nbytes + kCopySplits - 1) / kCopySplits;
    chunk = (chunk + 15) & ~int64_t{15};
    const int64_t lo = static_cast<int64_t>(split) * chunk;
    if (lo >= nbytes) {
      continue;
    }
    const int64_t hi = (lo + chunk < nbytes) ? (lo + chunk) : nbytes;
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(src_ptrs[t]) + static_cast<int64_t>(hs) * nbytes;
    uint8_t* dst =
        reinterpret_cast<uint8_t*>(dst_ptrs[t]) + static_cast<int64_t>(slot) * nbytes;
    const int64_t body_end = lo + ((hi - lo) & ~int64_t{15});
    int64_t off = lo + static_cast<int64_t>(tid) * 16;
    const int64_t stride = static_cast<int64_t>(nt) * 16;
    for (; off < body_end; off += stride) {
      *reinterpret_cast<uint4*>(dst + off) =
          *reinterpret_cast<const uint4*>(src + off);
    }
    for (int64_t b = body_end + static_cast<int64_t>(tid); b < hi; b += nt) {
      dst[b] = src[b];
    }
  }
}

/**
 * \brief Decode spill page-in: remap topk_ids and fill landing rows from UVA.
 *
 * \param topk_ids      int32 [T, K] in/out (kept physical or -1)
 * \param land_ids      int32 [T, K] out (landing slot or -1)
 * \param slot_host_row int32 [n_landing] workspace
 * \param map_table     int32 [n_logical] GPU slot or -1
 * \param host_map      int32 [n_logical] host row or -1
 * \param src_ptrs      int64 [n_tensors] UVA bases
 * \param dst_ptrs      int64 [n_tensors] landing bases
 * \param row_bytes     int64 [n_tensors] bytes per expert row
 */
inline void spill_page_in(tvm::ffi::TensorView topk_ids,
                          tvm::ffi::TensorView land_ids,
                          tvm::ffi::TensorView slot_host_row,
                          tvm::ffi::TensorView map_table,
                          tvm::ffi::TensorView host_map,
                          tvm::ffi::TensorView src_ptrs,
                          tvm::ffi::TensorView dst_ptrs,
                          tvm::ffi::TensorView row_bytes) {
  using namespace host;
  SymbolicSize n_tok = {"n_tok"};
  SymbolicSize k = {"k"};
  SymbolicSize n_logical = {"n_logical"};
  SymbolicSize n_landing = {"n_landing"};
  SymbolicSize n_tensors = {"n_tensors"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tok, k})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_ids)
      .verify(land_ids);
  TensorMatcher({n_landing})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(slot_host_row);
  TensorMatcher({n_logical})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(map_table)
      .verify(host_map);
  TensorMatcher({n_tensors})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src_ptrs)
      .verify(dst_ptrs)
      .verify(row_bytes);

  const int32_t t = static_cast<int32_t>(n_tok.unwrap());
  const int32_t kk = static_cast<int32_t>(k.unwrap());
  const int32_t nlog = static_cast<int32_t>(n_logical.unwrap());
  const int32_t nland = static_cast<int32_t>(n_landing.unwrap());
  const int32_t ntens = static_cast<int32_t>(n_tensors.unwrap());
  CHECK_HOST(nland > 0 && nland <= kMaxPoolSlots)
      << "sm70_dsv41 spill_page_in: n_landing " << nland;
  CHECK_HOST(ntens > 0 && ntens <= kMaxTensors)
      << "sm70_dsv41 spill_page_in: n_tensors " << ntens;
  CHECK_HOST(t * kk > 0) << "sm70_dsv41 spill_page_in: empty topk";

  const DLDevice dev = device_.unwrap();
  LaunchKernel(1, 32, dev)(
      spill_assign_kernel,
      static_cast<int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(land_ids.data_ptr()),
      static_cast<int32_t*>(slot_host_row.data_ptr()),
      static_cast<const int32_t*>(map_table.data_ptr()),
      static_cast<const int32_t*>(host_map.data_ptr()),
      t,
      kk,
      nlog,
      nland);
  LaunchKernel(nland * kCopySplits, kCopyThreads, dev)(
      spill_copy_kernel,
      static_cast<const int32_t*>(slot_host_row.data_ptr()),
      static_cast<const int64_t*>(src_ptrs.data_ptr()),
      static_cast<const int64_t*>(dst_ptrs.data_ptr()),
      static_cast<const int64_t*>(row_bytes.data_ptr()),
      ntens,
      nland);
}

/**
 * \brief Assign pass of the persistent-cache page-in (see spill_page_in_cached).
 *
 * Same contract as spill_assign_kernel for topk_ids/land_ids/slot_host_row,
 * plus cache maintenance:
 * - main-region hit (map_table[id] >= 0): identity remap; any stale cache
 *   entry for id is invalidated so its slot returns to the LRU pool.
 * - cache hit (cache_lut[id] >= 0): remap to the cached slot, bump its epoch,
 *   NO copy is issued for it.
 * - miss: claim the argmin-epoch slot (never one touched this call), evict
 *   the previous key's LUT entry, record host row for the copy pass.
 */
__global__ void spill_assign_cached_kernel(int32_t* __restrict__ topk_ids,
                                           int32_t* __restrict__ land_ids,
                                           int32_t* __restrict__ slot_host_row,
                                           const int32_t* __restrict__ map_table,
                                           const int32_t* __restrict__ host_map,
                                           int32_t* __restrict__ cache_lut,
                                           int32_t* __restrict__ cache_slot_key,
                                           int32_t* __restrict__ cache_epoch,
                                           int32_t* __restrict__ cache_clock,
                                           int32_t n_tok,
                                           int32_t k,
                                           int32_t n_logical,
                                           int32_t lut_offset,
                                           int32_t n_landing) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t assigned_id[kMaxLanding];
  int32_t assigned_slot[kMaxLanding];
  int32_t next = 0;
  const int32_t epoch = ++cache_clock[0];
  ++cache_clock[1];
  for (int32_t s = 0; s < n_landing; ++s) {
    slot_host_row[s] = -1;
  }
  const int32_t n = n_tok * k;
  for (int32_t i = 0; i < n; ++i) {
    const int32_t id = topk_ids[i];
    if (id < 0 || id >= n_logical) {
      land_ids[i] = -1;
      continue;
    }
    const int32_t phys = map_table[id];
    if (phys >= 0) {
      topk_ids[i] = phys;
      land_ids[i] = -1;
      const int32_t stale = cache_lut[lut_offset + id];
      if (stale >= 0) {
        cache_lut[lut_offset + id] = -1;
        cache_slot_key[stale] = -1;
      }
      continue;
    }
    const int32_t cached = cache_lut[lut_offset + id];
    if (cached >= 0 && cached < n_landing) {
      cache_epoch[cached] = epoch;
      topk_ids[i] = -1;
      land_ids[i] = cached;
      ++cache_clock[2];
      continue;
    }
    int32_t slot = -1;
    for (int32_t s = 0; s < next; ++s) {
      if (assigned_id[s] == id) {
        slot = assigned_slot[s];
        break;
      }
    }
    if (slot < 0) {
      const int32_t hs = host_map[id];
      const int32_t cap = n_landing < kMaxLanding ? n_landing : kMaxLanding;
      if (hs < 0 || next >= cap) {
        topk_ids[i] = -1;
        land_ids[i] = -1;
        ++cache_clock[4];
        continue;
      }
      // Plain global LRU (MoE4All TWO_POOL A/B: fancy policies regress).
      // Epoch protection is implicit: slots claimed this call carry the
      // current epoch, so argmin never picks them while they are live.
      int32_t best = 0;
      for (int32_t s = 1; s < n_landing; ++s) {
        if (cache_epoch[s] < cache_epoch[best]) {
          best = s;
        }
      }
      const int32_t prev_key = cache_slot_key[best];
      if (prev_key >= 0) {
        cache_lut[prev_key] = -1;
      }
      slot = best;
      cache_lut[lut_offset + id] = slot;
      cache_slot_key[slot] = lut_offset + id;
      cache_epoch[slot] = epoch;
      assigned_id[next] = id;
      assigned_slot[next] = slot;
      slot_host_row[slot] = hs;
      ++cache_clock[3];
      ++next;
    }
    topk_ids[i] = -1;
    land_ids[i] = slot;
  }
}

/**
 * \brief Decode spill page-in with a persistent cross-call expert cache.
 *
 * Identical tensor contract to spill_page_in plus four cache buffers that
 * live on the rank-global landing pool and persist across layer calls and
 * CUDA-graph replays (all state is device-side; replay is deterministic).
 * cache_lut is the per-layer slice view (length n_logical) of the rank LUT.
 */
inline void spill_page_in_cached(tvm::ffi::TensorView topk_ids,
                                 tvm::ffi::TensorView land_ids,
                                 tvm::ffi::TensorView slot_host_row,
                                 tvm::ffi::TensorView map_table,
                                 tvm::ffi::TensorView host_map,
                                 tvm::ffi::TensorView src_ptrs,
                                 tvm::ffi::TensorView dst_ptrs,
                                 tvm::ffi::TensorView row_bytes,
                                 tvm::ffi::TensorView cache_lut,
                                 tvm::ffi::TensorView cache_slot_key,
                                 tvm::ffi::TensorView cache_epoch,
                                 tvm::ffi::TensorView cache_clock,
                                 int64_t lut_offset) {
  using namespace host;
  SymbolicSize n_tok = {"n_tok"};
  SymbolicSize k = {"k"};
  SymbolicSize n_logical = {"n_logical"};
  SymbolicSize n_landing = {"n_landing"};
  SymbolicSize n_tensors = {"n_tensors"};
  SymbolicSize n_stats = {"n_stats"};
  SymbolicSize n_lut = {"n_lut"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tok, k})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_ids)
      .verify(land_ids);
  TensorMatcher({n_landing})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(slot_host_row)
      .verify(cache_slot_key)
      .verify(cache_epoch);
  TensorMatcher({n_stats})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(cache_clock);
  TensorMatcher({n_logical})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(map_table)
      .verify(host_map);
  TensorMatcher({n_lut})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(cache_lut);
  TensorMatcher({n_tensors})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src_ptrs)
      .verify(dst_ptrs)
      .verify(row_bytes);

  const int32_t t = static_cast<int32_t>(n_tok.unwrap());
  const int32_t kk = static_cast<int32_t>(k.unwrap());
  const int32_t nlog = static_cast<int32_t>(n_logical.unwrap());
  const int32_t nland = static_cast<int32_t>(n_landing.unwrap());
  const int32_t ntens = static_cast<int32_t>(n_tensors.unwrap());
  CHECK_HOST(nland > 0 && nland <= kMaxPoolSlots)
      << "sm70_dsv41 spill_page_in_cached: n_landing " << nland;
  CHECK_HOST(ntens > 0 && ntens <= kMaxTensors)
      << "sm70_dsv41 spill_page_in_cached: n_tensors " << ntens;
  CHECK_HOST(n_stats.unwrap() >= 5)
      << "sm70_dsv41 spill_page_in_cached: cache_clock too small";
  CHECK_HOST(t * kk > 0) << "sm70_dsv41 spill_page_in_cached: empty topk";
  CHECK_HOST(lut_offset >= 0 &&
             lut_offset + nlog <= static_cast<int32_t>(n_lut.unwrap()))
      << "sm70_dsv41 spill_page_in_cached: lut_offset " << lut_offset
      << " out of range for lut " << n_lut.unwrap();

  const DLDevice dev = device_.unwrap();
  LaunchKernel(1, 32, dev)(
      spill_assign_cached_kernel,
      static_cast<int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(land_ids.data_ptr()),
      static_cast<int32_t*>(slot_host_row.data_ptr()),
      static_cast<const int32_t*>(map_table.data_ptr()),
      static_cast<const int32_t*>(host_map.data_ptr()),
      static_cast<int32_t*>(cache_lut.data_ptr()),
      static_cast<int32_t*>(cache_slot_key.data_ptr()),
      static_cast<int32_t*>(cache_epoch.data_ptr()),
      static_cast<int32_t*>(cache_clock.data_ptr()),
      t,
      kk,
      nlog,
      static_cast<int32_t>(lut_offset),
      nland);
  LaunchKernel(nland * kCopySplits, kCopyThreads, dev)(
      spill_copy_kernel,
      static_cast<const int32_t*>(slot_host_row.data_ptr()),
      static_cast<const int64_t*>(src_ptrs.data_ptr()),
      static_cast<const int64_t*>(dst_ptrs.data_ptr()),
      static_cast<const int64_t*>(row_bytes.data_ptr()),
      ntens,
      nland);
}

}  // namespace sglang::sm70_dsv41
