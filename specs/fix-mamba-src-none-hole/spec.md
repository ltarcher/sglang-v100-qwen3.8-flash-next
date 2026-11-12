# Spec: 修复 mamba 匹配 `src=None` 仍服 Full KV 前缀的正确性洞

## 目标

在**非 HiCache（device-only）**的 unified radix 树上，`match_prefix` 申报的可用 Full-KV 前缀长度
必须 ≤ 最深一个可复用 Mamba 状态覆盖的位置。否则请求会"服 KV 但没有对应状态"：
COW 的 `src_index is None` 时直接返回，`alloc_for_extend` 给请求一个新槽并置
`mamba_needs_clear=True`，即状态从 0 开始，而 `prefix_indices` 仍跳过前 N 个 token 的 KV 计算，
输出必然错误。

修复策略（用户已定调）：**把可服前缀裁到"最后一个有 mamba 状态的匹配祖先"的深度；若整条匹配路径上
没有任何可用状态，则裁到 0（从 0 重算）**。绝不"服 KV 但无状态"。

## 根因（证据链）

1. `MambaComponent.create_match_validator(match_device_only=True)` 恒返回 `True`
   （`python/sglang/srt/mem_cache/unified_cache/components/mamba.py:151-171`）。非 HiCache 时
   tree core 用的正是这个 device-only 校验器（`unified_tree_core.py:905-909`），因此 Full 可以命中到
   一个**没有 mamba 状态**的边界：分歧点 split 出来的中间片段天然无状态
   （`MambaComponent.redistribute_on_node_split` 把状态留在 child），或状态被 LRU 驱逐。
2. `finalize_match_result_in_tree_core`（`mamba.py:173-203`）只算 `mamba_branching_seqlen`，
   不裁剪 `device_indices` / `full_kv_hit_length` / 节点。且它把"可服前缀"当成状态边界：
   `mamba_boundary_len = len(result.device_indices) + result.host_hit_length`。
   非 HiCache 下 `full_kv_hit_length == len(device_indices)`，所以 `branching` 恒为 `None`，
   没有任何机制暴露/修补这个缺口。
3. `finalize_match_result_in_cache`（`mamba.py:205-253`）`src_index is None` 时 `return result`，
   不裁前缀。消费端 `schedule_batch.py:1747-1776` 把 `device_indices` 原样写进
   `req.prefix_indices`，`cache_protected_len = len(prefix_indices)`；`memory_pool.py:1562-1570`
   在 `not req.kv.holds_mamba` 时新分配槽并 `mamba_needs_clear=True`。
4. 复现：chunk=704 容器（`/home/bso/vllm-Qwen3.8/docker-compose-sglang-v100-exp-review.yaml`，
   `SGLANG_DEBUG_MAMBA_RADIX=1`）日志 `[radix-dbg] cow req=... serve=704 full_hit=704 src=None`，
   两次请求 31/32 分歧、输出长度膨胀 2–3×。生产 chunk=4096 本轮未触发（src 有效），但同属真实洞。

**不变量**（insert 侧已保证，修复依赖它）：树上某个节点持有 device mamba 值 ⇒ 该状态的覆盖深度
== 该节点深度。依据：`UnifiedRadixCache` 用 `effective_cache_len = min(各组件 prepare_for_caching_req)`
截断插入 key（`unified_radix_cache.py:1086-1135`），`MambaComponent.prepare_for_caching_req` 返回
`mamba_last_track_seqlen`（捐赠深度）；split 时状态只留在 child（`redistribute_on_node_split`）。
因此"最深有状态祖先的深度"就是安全可服长度。

## 变更（唯一实现文件）

`python/sglang/srt/mem_cache/unified_cache/components/mamba.py`

### 1. `finalize_match_result_in_tree_core` 开头加裁剪

```python
        if not self.tree_core.enable_hicache:
            result = self._clip_to_reusable_state(result)
        last_node = result.best_match_node
        mamba_boundary_len = len(result.device_indices) + result.host_hit_length
        ...（其余逻辑不变）
```

### 2. 新增方法（放在 `finalize_match_result_in_tree_core` 之上）

```python
    def _clip_to_reusable_state(self, result: MatchResult) -> MatchResult:
        """Serve no Full KV prefix that the reusable Mamba state does not cover.

        The device-only tree accepts a Full match boundary without a Mamba
        state: a divergence split leaves its fragment state-less (the state
        stays on the deeper child), and a donated checkpoint can be evicted.
        Serving such a prefix starts the request's state from zero at a
        non-zero position. Clip the match to the deepest matched ancestor that
        still holds a device state; with none, serve nothing and let prefill
        rebuild the state from the prompt.
        """
        ct = self.component_type
        root = self.tree_core.root_node
        node = result.best_match_node
        if node.component_data[ct].value is not None:
            return result
        while node is not root and node.component_data[ct].value is None:
            node = node.parent
        if node is result.best_match_node:
            # No state-bearing ancestor other than the boundary itself.
            return result
        depth = 0
        cur = node
        while cur is not root:
            depth += len(cur.key)
            cur = cur.parent
        return result._replace(
            device_indices=result.device_indices[:depth],
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=0,
            mamba_branching_seqlen=None,
            full_kv_hit_length=depth,
        )
```

说明：

- `depth` 与 `device_indices` 同单位（EAGLE bigram 视图下 `len(node.key)` 与 `device_indices`
  逐 token 一致），切片按节点边界落点，天然对齐。
- 同时改写 `last_device_node` / `last_host_node` / `best_match_node`，使下游 `inc_lock_ref`、
  `cache_protected_len`、insert 的 `prev_prefix_len` 都与裁剪后的前缀一致
  （`schedule_batch.py:1773-1776`、`unified_radix_cache.py:1088/1260`），不会出现
  "insert 按全量 overlap 释放、rebind 只写回裁短边界" 的失配。
- `full_kv_hit_length` 同步改小：非 HiCache 下它与 `len(device_indices)` 同源，改小后
  `finalize` 后续的 `branching_seqlen` 自然为 `None`（状态边界即服边界）。
- 裁剪后 COW 一定能在 `best_match_node` 上取到 `src_index`；`finalize_match_result_in_cache`
  保持不动（若未来被并发驱逐，`src=None` 分支仍是既有保守退路）。

### 3. 更新 device-only 校验器注释

`create_match_validator` 中 `match_device_only=True` 分支结尾的
"COW handles a mamba-less boundary via src=None (full recompute)" 改为说明：
匹配后由 `_clip_to_reusable_state` 把边界裁到最深有状态祖先（或无状态时裁到 0）。

### 明确不做

- 不改 HiCache（host pool / load_back / `mamba_branching_seqlen` 契约）路径：裁剪以
  `not self.tree_core.enable_hicache` 为门；本机复现容器与生产 Qwen3.8 compose 均未开 HiCache 主机池，
  该路径另有 branching/load-back 机制，风险与验证范围不可控。
- 不改 insert / handoff / 状态捐赠逻辑；不改 validator（在 validator 里截断正是注释里警告的
  chunked-stash 失配来源，故在 walk 之后统一裁剪）。

## 测试（TDD：先红后绿）

文件：`test/registered/unit/mem_cache/test_mamba_component_caching_boundary.py`
（复用同目录 `build_fixture` / `CacheConfig`，已注册 `register_cuda_ci`，符合 write-sglang-test 惯例）

新增 class `TestMambaMatchClipsToReusableState`：

```python
    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=8,
        kv_size=4096,
        max_context_len=4096,
    )
    # build_fixture(self.cfg, mamba_cache_chunk_size=64)
```

helper `_insert(tokens, donate_len) -> params`：`_set_track(req, last_seqlen=donate_len)`，
`value = allocator.alloc(len(tokens))`，`InsertParams(key=RadixKey(array("q", tokens)), value=value)`，
`prepare_for_caching_req(req, params, len(tokens), is_finished=True)`，`cache.insert(params)`。

1. `test_split_fragment_without_any_state_serves_nothing`
   - insert1 = tokens 1..128，donate 128（node 128 有状态）；
   - insert2 = `[1..64] + [9001..9064]`，donate 128 ⇒ 在 64 处 split：fragment(64) 无状态，
     child(64..128) 保留 insert1 的状态；
   - `match_prefix(key=[1..64], cow_mamba=True, req=...)`：
     断言 `len(device_indices) == 0`、`full_kv_hit_length == 0`、
     `best_match_node == cache.tree_core.root_node.id`、`mamba_branching_seqlen is None`、
     `req.kv.mamba_cow_src_index is None`、`req.kv.holds_mamba` 仍是请求自己的槽。
   - 当前代码会返回 64 ⇒ 红灯。
2. `test_split_fragment_clips_to_deepest_state_bearing_ancestor`
   - insert1 = 1..704，donate 704；insert2 = 1..1408，donate 1408；insert3 =
     `[1..960] + [9001..9064]`（总长 ≥1024），donate 1024 ⇒ 在 960 处 split：
     fragment(960) 无状态，child(960..1408) 保留 1408 状态，leaf(960..1024) 带 insert3 的状态；
   - `match_prefix(key=[1..960])`：断言 `len(device_indices) == 704`、
     `full_kv_hit_length == 704`、`req.kv.mamba_cow_src_index` 的槽 == insert1 的
     `params.mamba_value` 槽、`mamba_branching_seqlen is None`。
   - 当前代码返回 960 ⇒ 红灯。
3. 回归保险（现有用例已覆盖，无需新增）：`test_match_cow_never_hands_tree_slot_to_request`
   仍须 64/64 且 COW src == 树槽；clip 在"边界自带状态"时必须 no-op。

判据：修复前 1/2 红灯；修复后全绿。

## 验证

1. 红灯：`$PYTHON -m pytest test/registered/unit/mem_cache/test_mamba_component_caching_boundary.py -v`
   （先确认新用例失败）。
2. 绿灯 + 同目录回归：`$PYTHON -m pytest test/registered/unit/mem_cache/ -v`。
3. 容器复现（chunk=704）：把改动后的 `mamba.py` 用
   `docker compose up -d --no-start` + `docker cp` + `docker start` 注入
   `sglang-v100-qwen38-8110`，跑 `/tmp/warm_determinism.py`（flush→A→B1→B2）：
   - 判据：`[radix-dbg] cow ... src=None` 消失（改为裁短后的 serve 长度 + 有效 src），
     B1/B2 输出不再出现长度膨胀；同一 B 前缀两次命中一致性回到现有布局级水平（≥ 现有 11–19/32，
     且无 2–3× 膨胀——注意命中路径存在布局级非确定性，文本 A/B 相等率不是唯一判据）。
   - 可选：`/tmp/accuracy_ab.py` gsm8k 质量不回退（fresh 29 / hit1 28 / hit2 29 of 32 量级）。
4. 生产配置回归（chunk 恢复 4096）：重启容器，`bash scripts/smoke_v100.sh` 通过；
   Qwen3.8 NVFP4 验收命令不回退（prefill ~3,000 tok/s、decode ~100 tok/s/流、MTP accept ~2.1）；
   确认 `[radix-dbg]` 中常规命中仍为有效 src、无前缀被误裁（日志 serve == full_hit 的场景不变）。

## 风险与回退

- 裁剪只会减少命中长度，不会引入错误数据；最坏情况退化为全量 prefill（性能损失），
  可通过 `[radix-dbg]` 的 serve 长度观察频率。生产 chunk=4096 下原有的有效命中
  （边界自带状态）完全不受影响（`_clip_to_reusable_state` 首行 no-op）。
- 若出现意外命中率下降，回退即删除 `finalize_match_result_in_tree_core` 开头的两行调用。

## 需用户确认

- 范围限定非 HiCache（本轮复现与生产 Qwen3.8 compose 形态）；HiCache 路径维持现状。
- 容器回归会临时把 chunked-prefill 改回 704 再恢复 4096，需要占用 8110 容器两个 boot 窗口。
