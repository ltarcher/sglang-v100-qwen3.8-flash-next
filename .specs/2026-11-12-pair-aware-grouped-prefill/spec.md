# Spec: PAIR 感知 SM70 Grouped Prefill（路线 B）

> **结果（2026-11-12 EOD）**:本 spec 全部落地并验证（CPU 54 例、GPU 内核
> 2 passed、smoke、needle QA PASS），但生产 A/B 显示 PAIR 拓扑下 grouped
> 内核因对家 IPC 读 latency-bound 而回归（16k +62~136%），已回退为
> GROUPED=0。完整归因与后续选项见 docs/v100/PREFILL_LEVERS_PLAN.md §8.3
> 与本目录 tasks.md。

日期: 2026-11-12。目标：在 KV tier phase3 + PAIR_PARITY 生产形态（8110 compose）
下重新启用 `SGLANG_SM70_QSA_GROUPED_PREFILL`，拿回 32K+ prefill 速度
（历史锚点 2,600-2,900 tok/s；L2-b 实测 grouped +13~18%）。

## 背景（已确认事实）

- 生产 pool dtype = fp16（`--kv-cache-dtype auto`）→ grouped 内核（fp16 池）
  与生产匹配；fused fp8 / batched prefill 路由生产永不命中。
- 门控 b8bf957f76 在 `_kv_host_tier.phase3` 时 blanket 阻断 grouped：
  planner 直读逻辑 `req_to_token`，而 phase3 设备行是 mod-remapped。
- PAIR 读侧（`kv_host_tier.map_slot_table`，v2 owner-bit dual pointer）：
  `entry = owner_bit<<30 | owner_row*page_size+off`；两 rank 池行索引一致；
  所有页恒驻设备（owner-computes + 无超售 boot 门），无 page-in、无 host
  fallback（gather PAIR slice 同契约）。
- 非 PAIR phase3：host-mirror tag fallback；extend 写与 modulo 槽别名 →
  page-in 与 write-restore 竞态 → **保持阻断**，仅 PAIR 放行。
- 对家 K/V：`ImportedDeviceBuffer` → `imported_as_tensor()` 零拷贝 torch 视图
  （decode `sm70_cuda_qsa_decode` peer_kv 先例）。
- GPU 当前被生产引擎占用 → 本 spec 的 GPU 验收延至下次重启窗口执行。

## 变更清单

### 1. `python/sglang/srt/layers/attention/tilelang_fa_v100/_kernels_grouped_sparse_prefill.py`

**内核 `_grouped_qsa_sparse_prefill_sm70`**（单一内核体，无新 JIT 变体）：

- `main` 签名追加：`KPeer [P, dim] fp16`、`VPeer [P, dim] fp16`、
  `Owner [G, union_len] uint8`（插在 VPool 之后 / Uidx 之前：顺序
  Q, KPool, VPool, KPeer, VPeer, Uidx, Mask, Owner, Out）。
- 4 处池加载点（K left/right、V left/right，现 L138-139/148-149/198-199/207-208）
  改为逐元素双指针分支：

  ```python
  for bi, d in T.Parallel(BI, D_half):
      if Owner[bg, p0 + bi] > 0:
          K_shared[bi, d] = KPeer[Uidx[bg, p0 + bi], d]
      else:
          K_shared[bi, d] = KPool[Uidx[bg, p0 + bi], d]
  ```

- `_get_kernel` 缓存键不变（(union_len, sm_scale)）——Owner 数据驱动分支，
  非编译期参数，不新增 JIT 变体。
- 模块 docstring 增补 PAIR dual-pointer 段（设计出处
  docs/v100/KV_HOST_TIER_DESIGN.md §6.1 v2）。

**Planner `build_grouped_union_plan`**：

- 签名：`req_to_token` 改 `Optional`；新增 kw-only
  `slot_table: Optional[Tensor] = None`、`owner_mask: Optional[int] = None`。
- 返回改 4 元组 `(uidx, mask, u_pad, owner)`；`owner` 恒为
  `[G, u_pad] uint8`（slot_table=None 时全零）。
- slot 解析（现 L341-343）：

  ```python
  if slot_table is not None:
      if owner_mask is None or owner_mask <= 0:
          raise ValueError("slot_table requires the PAIR owner mask")
      req_rows = slot_table          # batch-local phase3 物理表
  else:
      req_rows = req_to_token.index_select(0, req_pool_indices.to(torch.long))
  slot = req_rows[batch_of_entry, logical_entry].to(torch.int32)
  if slot_table is not None:
      peer_bits = (slot >> (owner_mask.bit_length() - 1)) & 1
      slot = slot & (owner_mask - 1)   # 清 owner 位，低 30 位 = 池行
  ```

- Owner 构建（仅 slot_table 路径走 index_add，与 uidx/mask 同款 scatter）：

  ```python
  owner_i32 = torch.zeros(G, u_pad, device=device, dtype=torch.int32)
  if slot_table is not None:
      owner_i32.view(-1).index_add_(
          0, g_sorted * u_pad + u_local,
          torch.where(vd, peer_bits, torch.zeros_like(peer_bits)))
  owner = owner_i32.to(torch.uint8)
  ```

- planner docstring 更新（4 元组、slot_table 语义、owner bit 出处）。

**`run_grouped_sparse_prefill`**：

- 新增 `peer_kv=None` 参数（`(peer_k, peer_v)` torch 视图，行数 = 本地池行数）。
- None 时 KPeer/VPeer 传 k_pool/v_pool 自身（Owner 全零永不取该分支）；
  非 None 时 `.view(rows, HEAD_DIM)`。

### 2. `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`

**`_forward_sm70_grouped_sparse_prefill`**（现 L2463-2552）：

- 签名追加 kw-only `layer=None`。
- 原 phase3 blanket return（L2494-2502）替换为：

  ```python
  _tier = self._kv_host_tier
  tier3 = _tier is not None and _tier.phase3
  slot_table = owner_mask = peer_kv = None
  req_to_token = None
  if tier3:
      if not _tier.pair_split:
          return None            # 非 PAIR phase3：host-mirror + 写别名，保持阻断
      if layer is None:
          return None
      tier_layer = _tier.layer_slot.get(layer.layer_id)
      if tier_layer is None:
          return None
      metadata = self._resolve_metadata(forward_batch)
      slot_table = metadata.phys_token_slot_table
      if slot_table is None:
          return None
      from sglang.srt.mem_cache.pair_split_ipc import imported_as_tensor
      pkv = _tier.pair_peer_kv(tier_layer)
      owner_mask = pkv[0]
      peer_kv = (imported_as_tensor(pkv[1], tuple(k_buffer.shape)),
                 imported_as_tensor(pkv[2], tuple(v_buffer.shape)))
  else:
      req_to_token = self.req_to_token
      if req_to_token is None:
          return None
  ```

- 形状守卫不变（fp16 池、R>=256、stride 等）。
- planner 调用追加 `slot_table=slot_table, owner_mask=owner_mask`。
- 日志行加 PAIR 标注；`run_grouped_sparse_prefill(..., peer_kv=peer_kv)`。

**调用点 `forward_extend`**（现 L2149）：

- `if not tier3 and envs...` → `if envs.SGLANG_SM70_QSA_GROUPED_PREFILL.get():`
  （tier3 定界逻辑内聚到方法内，调用点不再前置判断）；追加 `layer=layer`。
- L2082-2088 注释更新：说明 tier3 下 fused/batched decline、grouped 在 PAIR 下
  以 owner-bit 寻址重入、其余落 tier-aware gather。

### 3. 测试

- `test/registered/unit/layers/attention/test_qwen_sparse_sm70_routing.py`：
  - `test_grouped_prefill_guard_blocks_kv_tier_phase3` 改为
    **phase3 + 非 PAIR 仍阻断**（`_RecordingTier` 加 `pair_split = False`）；
  - 新增 **phase3 + PAIR 放行**：stub backend（`__new__` + `forward_metadata`
    带 phys_token_slot_table）+ tier stub（`pair_split=True`、`layer_slot`、
    `pair_peer_kv` 记录调用）+ patch
    `pair_split_ipc.imported_as_tensor` / `build_grouped_union_plan` /
    `run_grouped_sparse_prefill` / `is_in_breakable_cuda_graph`；
    断言 planner 收到 `slot_table` 与 `owner_mask`、run 收到 `peer_kv`。
- `test/registered/unit/layers/attention/test_grouped_union_plan_cpu.py`：
  - 解包改 4 元组；
  - 新增 slot_table 路径用例：构造含 owner bit 的物理表（一部分本地、
    一部分 bit30=1 对家行），断言 uidx 低 30 位、owner 张量 0/1 分布、
    非 tier 路径 owner 全零；slot_table 无 owner_mask 时 ValueError。
- `test/registered/unit/layers/attention/test_grouped_sparse_prefill_sm70.py`（GPU，延后跑）：
  - 解包改 4 元组；
  - 新增 peer 双指针数值测试：本地池 + 对家池独立随机，slot_table 混合
    本地/对家条目（bit30），参考实现按 owner 位选池做 masked softmax，
    输出与参考一致（atol/rtol 2e-2，沿用现有口径）。
- `test/registered/unit/layers/attention/test_qsa_grouped_prefill_gate.py`：
  - plan 三元组 → 四元组；`build_plan.assert_called_once_with` 追加
    `slot_table=None, owner_mask=None`；`run_kernel.assert_called_once_with`
    追加 `peer_kv=None`。

### 4. 文档同步（fork 拥有区）

- `docs/v100/PREFILL_LEVERS_PLAN.md`：
  - §8 末尾（8.2 之后）新增小节：2026-11-12 质量问题解决（用户裁定）+
    路线 B 落地（PAIR 感知 planner），生产将随下次重启回 ON；GPU 验收清单
    指向 tasks.md。
  - L2-b 实测段（~L144-195）加一行注记：tier phase3 门控已按 PAIR 感知
    定界重开。
- `docs/v100/KV_HOST_TIER_DESIGN.md` §6.1：owner-bit dual pointer 消费者
  列表加 grouped prefill TileLang 内核（第三消费者）。
- `README.md`：`SGLANG_SM70_QSA_GROUPED_PREFILL` 条目条件更新
  （非 tier 与 phase3+PAIR 均可启用；非 PAIR phase3 仍走 gather）。

## 明确不做

- 非 PAIR phase3 的 grouped 放行（page-in 竞态，无解，保持 gather）。
- 新增 SGLANG_* 环境变量（复用现有开关）。
- `patches/`、上游行为分叉、CHUNK_AHEAD/PLE 改动。
- 今日在 production 占卡期间跑 GPU 测试/smoke（生产每机单引擎纪律）。

## GPU 验收清单（下次重启窗口执行，AGENTS.md 纪律）

1. `PYTHON=$HOME/miniconda3/envs/sglang-v100/bin/python` 跑
   `test/registered/unit/layers/attention/test_grouped_sparse_prefill_sm70.py`
   （含新 peer 数值用例；同时即 TileLang 双指针分支的编译验证）。
2. `bash scripts/smoke_v100.sh`。
3. 重建镜像 → compose 重启（env 已是 GROUPED=1）→ 前缀缓存门
   （cached_prompt_tokens 提升 + 输出与全量 prefill 一致）。
4. 三臂回归带数字：prefill 8k 2,687-2,798 / decode 95-114 / accept 2.70-3.15
   不回退；32k+ prefill 实测 A/B（目标回到 2,600-2,900 tok/s 带位）。
5. 数字回填 README 表格与 PREFILL_LEVERS_PLAN.md；`pre-commit run --all-files`。
