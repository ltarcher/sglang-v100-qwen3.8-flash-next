# 代码审查待办：65df49e55c..HEAD（2026-11-10）

对 `65df49e55cde07314d16ad113690ede923ff24de..HEAD`（47 commits，约 13k 行）的质量风险审查结论。
方法：4 个子系统并行审查 + 2 个独立验证器交叉验证 + 冲突项人工裁决；27 条初审中 3 条误报剔除、
2 条 1/2 确认（带保留收录）、1 条降级。审查时 HEAD = dd981328e3。

## 总体判断

生产默认路径（TP4 纯 TP + PAIR 布局 + PLE staged，mixed-chunk 与 grouped prefill 关闭）
无直接可触发的高危缺陷。所有 Critical/Major 均为 **latent**：需打开特定开关或组合配置才可达。
但 §1（grouped 内核 NaN）与 §3（grouped 死守卫）在复开 grouped prefill 离线验证时会直接命中，
复开前必须先修复。

## 待办清单

状态：`[ ]` 待办 / `[x]` 完成。严重度：C=critical、M=major、m=minor。
可达性列说明触发条件；"生产不可达"表示当前生产配置不会触发。

### Critical

- [x] **C1. Grouped prefill 内核 online softmax 缺全掩码 tile 防护 → NaN 且不可恢复**（2026-11-10 结案：机制证伪 + 防御性加固）
  - 位置：`tilelang_fa_v100/_kernels_grouped_sparse_prefill.py:164-177`
  - 触发：`SGLANG_SM70_QSA_GROUPED_PREFILL=1` + 跨请求混组（短行整 tile 全掩码）。
    `alpha=exp2(m_prev-(-inf))=inf` → acc_o/sumexp 中毒，后续 tile 无法恢复。
  - 依据：兄弟内核 `_kernels_paged.py:167-172`、`_kernels_dense_d256_sparse.py:105-113`
    均有 clamp+max 防护，新内核两者皆无；已发布的单请求/低重叠 A/B 验证结构上打不到该路径。
  - **结案结论（TDD 实证）**：审查的机制前提有误——tilelang 0.1.8
    `reduce_max(clear=False)` 是**累加语义**（`m_i = max(m_i, tile_max)`）而非覆盖写；
    内核 `m_i` 初值为有限值 `-(2**30)`，全掩码 tile 时 `m_i` 保持不变（alpha=1），
    当前版本 NaN **不可达**。判别证据链：
    1. 容器内微型探针：全 `-inf` tile + `clear=False` → `m_i` 保持 `-(2**30)`（差值 0）；
    2. 真实 planner+内核端到端：首 tile 全掩码临界形态（首请求 128 项 + 次请求块排在其后、
       pad 尾、row_keep=1）输出有限且与 torch 参考一致（`test_grouped_sparse_prefill_sm70.py`）；
    3. 负对照：同 fixture 强制 `clear=True`（覆盖语义）→ **100% NaN（24576/24576）**，
       证明 fixture 有检出能力、机制链真实存在。
  - 处置：NaN 当前不可达，但安全性完全依赖"累加语义 + 有限初值"两个隐式契约（tilelang
    升级即翻车）。已按兄弟内核纪律防御性加固：内核 reduce 后镜像 `_kernels_paged.py` 的
    `m_i==−inf→0` 钳制 + `m_i=max(m_i,m_prev)` 累加，并同步 docstring 不变量；加固后即使
    翻转为覆盖语义 + `-inf` 初值（C1 假设的爆发形态）也保持 nan=0。回归：多 seed 混组
    数值对齐 torch 参考、ruff 通过。

### Major（latent）

- [ ] **M2. FULL 卷绕布局同余页竞态（P3 write-restore/page-in 内核）**
  - 位置：`mem_cache/kv_host_tier.py:344-461`（map_slots 纯模 `lp%phys_pages` 于 :1161）
  - 触发：仅 FULL 布局（未启用；生产 PAIR 被 :1099 boot 双射门控保护）。同一 forward
    两个同余逻辑页并发 program 非原子竞争 tags/owner，消费端（sm70_longctx_decode.cu:122-133、
    qsa/sparse_attn.py:467-476）不查 tag → 静默错 KV；`_p3_mirror` 注释声称的 host 侧
    uniq 校验不存在（`_p3_cap=logical_pages+1`）。
  - 修复：FULL 模式加 host 侧同余/uniq 页校验，或 gate 为"FULL 不支持 oversubscription"。

- [x] **M3. Grouped prefill 的 mixed 批守卫是死代码**（2026-11-10 结案）
  - 位置：`layers/attention/qwen_sparse_attn_backend.py:2459-2463`
  - 触发：`SGLANG_SM70_QSA_GROUPED_PREFILL=1` 且 mixed-chunk 同开。eager_runner 已在
    metadata init 前把 MIXED 改写为 EXTEND，`forward_mode == MIXED` 恒 False，混批穿透
    守卫进入 grouped planner，触发注释自述的 tail 腐蚀。同文件 :933-936 有正确范例。
  - **结案结论**：守卫改为 `forward_mode == MIXED or _original_forward_mode == MIXED`
    （复用 eager_runner 改写约定，`_original_forward_mode` 为 ForwardBatch 正式
    dataclass 字段，forward_batch_info.py:665）。回归测试追加至
    `test_qwen_sparse_sm70_routing.py`：改写混批（EXTEND+original=MIXED）→ None、
    直接 MIXED → None、纯 EXTEND 合形状批穿透守卫抵达 planner 状态访问（哨兵断言）。
    负对照：旧守卫形态下改写混批用例失败，fixture 有检出能力。CPU 容器实跑 21 passed
    （4 个 `test_dense_prefill_retains_mtp_seed_indices` 失败为 HEAD=dd981328e3 存量、
    git stash 验证与本次改动无关）；ruff check/format 对改动行干净。

- [ ] **M4. 混批预算 head-of-line broadcast 在 DP attention 下死锁**
  - 位置：`managers/scheduler_components/mixed_prefill_budget.py:276-309`；
    init 门 `scheduler.py:1452-1477` 无 dp_size 检查
  - 触发：`SGLANG_MIXED_PREFILL_STEP_LATENCY_MS>0` + DP attention。pending 非空即
    `broadcast_pyobj(src=0)`（CPU 集体），DP 下各 rank 调用次数不同 → 集体失配挂死。
  - 修复：init 门加 dp_size 禁用（镜像 DynamicChunkSizer 的 PP-only 模式），或改
    per-TP-group 广播。

- [ ] **M5. chunk-ahead serve 路径无宽度防护**（1/2 验证器确认，带保留）
  - 位置：`models/qwen4_exp.py:1206-1217`
  - 触发：DP attention + `SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD>0` 同时开。DP 下
    `gather_dp_tokens` gather 宽度为全局宽度而 staged 缓冲按本地宽度填充：n≤max_rows
    时本 rank token 读错行（静默错 embedding）；n>max_rows 时 grid 越界、
    `buffer.data_ptr()+slot*row_bytes` 野指针解引用 pinned 内存。
    保留意见：consume 位级校验存在，但其比较两侧同为本地宽度，DP 失配无法被其捕获。
  - 修复：serve 分支加 `n == staged 宽度` 断言。

### Minor

- [ ] m6. IPC 导出假设 data_ptr==分配基址（1/2 确认，带保留）
  - `mem_cache/pair_split_ipc.py:35-132`。当前逐层独立大块分配下 data_ptr==基址成立，
    且偏移指针会使 GetMemHandle 响亮报错（非静默）；属分配器布局脆弱性。
  - 修复：按 torch IPC 语义记录 base+offset，或加基址断言。
- [ ] m7. `_peer_tags` 镜像无消费者死代码；`_CNT_P3_PEER_MISS` 从未自增
  - `mem_cache/kv_host_tier.py:1244-1314`。每 forward n_layers 次 D2D 拷贝纯开销。
  - 修复：删除 `_refresh_peer_tags` 调用链或接上消费者；挂上计数器。
- [ ] m8. phase3 下 graph replay 路径 step_seq 双增（replay 的 flush 缺 eager :1079 的对称守卫）
  - `layers/attention/qwen_sparse_attn_backend.py:1429-1432`（legacy 分支 :1466-1467 同）。
  - 修复：replay 路径补 `if not phase3` 守卫。影响仅日志节奏。
- [ ] m9. IPC 交换用 world rank 配对 attn_tp_rank^1；无偶数 tp 校验
  - `mem_cache/pair_split_ipc.py:117-122`。attn-TP 组≠world（DP/PP）或奇数 tp 时
    挂死/交叉映射。生产 attn_tp==world 且 TP4 偶数，不受影响。
  - 修复：加 attn_tp==world 与偶数 tp 断言。
- [ ] m10. `QSA_P3_FORCE_HOST`/`QSA_P3_CHECK` 裸读 os.environ，绕过 environ.py 契约
  - `mem_cache/kv_host_tier.py:1035-1036`。同文件其他 SGLANG_SM70_QSA_* 已入册。
  - 修复：迁移到 environ.py 注册（遵循 env-var-conventions）。
- [ ] m11. staged 后端下 update_weights_from_disk 必然失败
  - `models/qwen4_exp.py:1297-1345`。1 行 dummy + `_staged_materialized` 永久短路，
    权重更新无法落地且 NVMe 表永不刷新。仅影响热更新端点，serve 不用。
  - 修复：update 路径强制 re-materialize 或启动期拒绝组合。
- [ ] m12. `stage_async` 对 n>max_rows 直接 ValueError 杀 scheduler
  - `models/qwen4_exp_ple_staged.py:338-343`。阈值 262144 行 = 16384 tokens（已实测核实）；
    `chunked_prefill_size=-1` 或 >16384 的合法配置首条长 prefill 崩溃。
  - 修复：boot 期校验 chunked_prefill_size 与 max_rows 相容或自动上调。
- [ ] m13. chunk-ahead miss 路径先等满整个 fill 再比较 ids
  - `models/qwen4_exp_ple_staged.py:520-534`。ids_np 本已是 host 侧 numpy，可先比较再等；
    miss 时白付整段 NVMe 随机行 IO 于关键路径。默认 CHUNK_AHEAD=0 生产不受影响。
  - 修复：先比 ids 再决定等待。
- [ ] m14. pinned 分配失败静默回退 pageable，GPU 上 kernel 解引用 UB
  - `models/qwen4_exp_ple_staged.py:251-259`。
  - 修复：CUDA 可用时 fail-loud，不降级。
- [ ] m15. staged shard 文件名/meta 不含 checkpoint 指纹
  - `models/qwen4_exp_ple_staged.py:56-60`。REUSE=1（默认 0）时同形状异权重 checkpoint
    静默互用 12.2GiB 表。
  - 修复：name/meta 混入模型路径或权重哈希。
- [ ] m16. `_PLE_CHUNK_STASH` 请求完成/中止后不失效
  - `models/qwen4_exp_ple_prep.py:256-292`。slot 复用弹旧条目触发错误投机预读+满额等待
    （正确性由 consume ids 兜底）；多请求批的 stash 亦不被消费。
  - 修复：请求 finish/abort 或 release slot 时 pop 对应条目。
- [ ] m17. `_mixed_tail_mask` 回退防御缺口
  - `layers/attention/qwen_sparse_attn_backend.py:782-796`。把任何 prefix 非对齐行判为
    decode tail，且 mixed 批整体跳过 prefix 对齐 assert（:842-843）。触发前提（非对齐
    prefill 行进 mixed 批）被 schedule_policy.py:1945-1953 页对齐 snap 保护，属回退加固
    而非可达缺陷。
  - 修复：对非 tail 行保留 assert（零 sync）：`(prefix_lens[~tails] % ratio == 0).all()`。
- [ ] m18. 混批预算 inflight 溢出整体 clear 丢弃全部在途采样
  - `managers/scheduler_components/mixed_prefill_budget.py:365-366`。应 FIFO 逐条驱逐。
  - 修复：改 pop 最旧一条。确定性不受影响（纯本地函数）。
- [ ] m19. compose 与 serve 脚本 mixed-chunk 默认值分叉（原 major 降级：属 eb6cedfdf3
    有意启用并已登记 README）
  - `docker/v100-compose-qwen38-mtp.yaml:63`、`docker/v100-compose.yaml:70`（默认 750）
    vs `scripts/serve_qwen38_flash_next_nvfp4_v100.sh:149`（默认 0）。
  - 修复：compose 默认改 0，750 作为显式覆盖；或两侧注释登记差异。
- [ ] m20. compose 与 serve 脚本 PLE 后端分叉：脚本 staged NVMe、compose pinned RAM
  - `scripts/serve_qwen38_flash_next_nvfp4_v100.sh:327` vs compose（无
    `--ple-offload-backend`，走默认 pinned）。
  - 修复：compose 对齐或注释登记。
- [ ] m21. 多模态 StopIteration 合并语义：混排请求静默丢其他模态 token
  - `multimodal/processors/base_processor.py:1536-1551`（早退点 :1142-1148）。
    纯图像单模态生产不受影响。
  - 修复：早退仅作用于同模态，或区分"数据缺失"与"迭代器提前退出"。
- [ ] m22. ImportedDeviceBuffer 分支缺 e5m2→uint8 byte view
  - `tilelang_fa_v100/_decode_cuda.py:181-191`。e5m2+PAIR decode 响亮报错；
    fp16 生产不受影响。
  - 修复：peer 分支镜像本地 byte view。
- [ ] m23. grouped planner 第二个隐藏 D2H 同步
  - `tilelang_fa_v100/_kernels_grouped_sparse_prefill.py:257`。`int(group_slot[-1])`
    隐式 .item()，与 docstring "one scalar readback" 相悖。
  - 修复：host 侧按形状算上界。
- [ ] m24. A/B 脚本对任意 --base 无守卫调 /flush_cache
  - `scripts/ab_qsa_grouped_quality_v100.py:267-315`。误指 11435 生产端口会清 radix 缓存。
  - 修复：拒绝生产端口或要求 `--assume-idle`。

## 误报剔除记录（勿重报）

1. **预算饱和非对齐 chunk**（mixed_prefill_budget.py:354）：V100 默认路径
   `schedule_policy.py:1945-1953` 对 extend_len/end 双重页对齐 floor，prefix 对齐不变量
   不被破坏；`exact_chunk_fill` 为 gfx95-only（SM70 恒 False）。注意：若未来 V100 启用
   exact_chunk_fill 或绕过 snap 的截断路径，此风险链（饱和 cap 退化 + PrefillAdder 扣
   decode 行）需重新评估。
2. **tier3 draft replay 写旧物理页**（qwen_sparse_attn_backend.py:3301-3316）：draft 池在
   tier3 下整体跳过 tier（kv_cache_configurator.py `draft_pool_tier_free`，draft backend
   `_kv_host_tier=None`）；target verify replay 经 `init_forward_metadata_out_graph` 携带
   真实 out_cache_loc；旧式 `init_forward_metadata_replay_cuda_graph` 无 serving 调用者。
3. **materialize 残留更大旧文件**：文件名完整编码 shape/dtype，本写路径产不出更大的同名文件。

## 处置建议

1. 复开 `SGLANG_SM70_QSA_GROUPED_PREFILL=1` 之前：先修 M3（死守卫）。C1 已结案
   （机制证伪 + clamp 防御性加固 + `test_grouped_sparse_prefill_sm70.py` 回归防护）。
   **M3 已结案（2026-11-10）**：mixed 批守卫已读 `_original_forward_mode` 并配套回归
   测试，复开 grouped prefill 的正确性前置已清空。
2. M2/M4/M5 均为开关组合 latent，可在对应功能启用计划前置为门条件。
3. Minor 项按维护窗口批量处理；m6/m13/m17 为带保留或加固性质，优先级最低。
