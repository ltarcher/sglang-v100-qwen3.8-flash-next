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

- [x] **M2. FULL 卷绕布局同余页竞态（P3 write-restore/page-in 内核）**（2026-11-10 结案）
  - 位置：`mem_cache/kv_host_tier.py:344-461`（map_slots 纯模 `lp%phys_pages` 于 :1161）
  - 触发：仅 FULL 布局（未启用；生产 PAIR 被 :1099 boot 双射门控保护）。同一 forward
    两个同余逻辑页并发 program 非原子竞争 tags/owner，消费端（sm70_longctx_decode.cu:122-133、
    qsa/sparse_attn.py:467-476）不查 tag → 静默错 KV；`_p3_mirror` 注释声称的 host 侧
    uniq 校验不存在（`_p3_cap=logical_pages+1`）。
  - **结案结论（选项 b：gate FULL 不支持 oversubscription）**：根因比内核竞态更深——
    卷绕映射本身非单射，同余双活页无论写（store 互踏）还是读（page-in 无法同时驻留、
    decode 消费端不查 tag）都不可满足，文档 §10 记录的"召回单调劣化"即此缺陷的症状。
    新增 `_require_full_pool_capacity`（boot 期 FULL 臂要求 `logical_pages <= phys_pages`，
    错误信息给出 PHYS_TOKENS/--max-total-tokens/PAIR_PARITY 三条出路；PAIR 双射门
    :1099 不变）。回归测试 `test_kv_host_tier_full_capacity_gate.py`：超配拒启、
    容量相容放行、错误信息出路、容量相容时 slot table 单射性 + 超配别名复现。
    既有 KV tier CPU 套件 31 passed；GPU 内核测试（object 桩、刻意卷绕 fixture）不受
    boot gate 影响。卷绕形态如需复开须先做关联重映射 redesign（KV_HOST_TIER_DESIGN.md
    §10 卷绕注记已同步）。

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

- [x] **M4. 混批预算 head-of-line broadcast 在 DP attention 下死锁**（2026-11-10 结案）
  - 位置：`managers/scheduler_components/mixed_prefill_budget.py:276-309`；
    init 门 `scheduler.py:1452-1477` 无 dp_size 检查
  - 触发：`SGLANG_MIXED_PREFILL_STEP_LATENCY_MS>0` + DP attention。pending 非空即
    `broadcast_pyobj(src=0)`（CPU 集体），DP 下各 rank 调用次数不同 → 集体失配挂死。
  - **结案结论（选项 a：init 门禁用）**：`maybe_init_mixed_prefill_budget` 增加
    `get_parallel().attn_dp_enabled` 硬前置（失败-关闭；选项 b 的 per-TP-group 广播需
    重设计 rank-deterministic 协议，DP attention 未启用，无必要）。回归测试追加至
    `test_mixed_prefill_budget.py::TestMaybeInitMixedPrefillBudgetGate`：DP attention
    拒绝布防 / 非 DP 正常布防（align 透传）/ target=0 仍禁用。TDD 红灯先行（DP 用例
    失败）后实现；全套件 30 passed。DP attention 下该开关混批 chunk 回到原始
    chunked_prefill_size 行为，与其他 DP-attention 不支持特性一致。

- [x] **M5. chunk-ahead serve 路径无宽度防护**（1/2 验证器确认，带保留）（2026-11-10 结案）
  - 位置：`models/qwen4_exp.py:1206-1217`
  - 触发：DP attention + `SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD>0` 同时开。DP 下
    `gather_dp_tokens` gather 宽度为全局宽度而 staged 缓冲按本地宽度填充：n≤max_rows
    时本 rank token 读错行（静默错 embedding）；n>max_rows 时 grid 越界、
    `buffer.data_ptr()+slot*row_bytes` 野指针解引用 pinned 内存。
    保留意见：consume 位级校验存在，但其比较两侧同为本地宽度，DP 失配无法被其捕获。
  - **结案结论（采纳修复方向：serve 分支宽度断言）**：`_gather_staged` serve 分支
    （qwen4_exp.py:1163-1184）在 gather 前断言 `n == staging._staged_ids.numel()`，
    失配 fail-loud（错误信息点名 DP attention + CHUNK_AHEAD 组合与两条禁用路径）；
    非 DP 健康路径下 prep 侧位级 id 匹配蕴含宽度相等，断言恒真、零开销。回归测试
    `test_qwen4_exp_ple_prep.py::TestGatherStagedServeWidthGuard`：全局/本地宽度失配
    拒绝 serve（含 n≤max_rows 子容量形态）、宽度匹配正常 serve（Recording 内核桩，
    TDD 红灯先行）。PLE 全套件 83 passed / 3 skipped。

### Minor

- [x] **m6. IPC 导出假设 data_ptr==分配基址（1/2 确认，带保留）**（2026-11-10 结案）
  - `mem_cache/pair_split_ipc.py:35-132`。当前逐层独立大块分配下 data_ptr==基址成立，
    且偏移指针会使 GetMemHandle 响亮报错（非静默）；属分配器布局脆弱性。
  - **结案结论（选项 b：基址断言 fail-loud）**：`_tensor_ipc_bytes` 导出前断言
    `tensor.data_ptr == untyped_storage().data_ptr()`（分配基址），偏移视图在
    boot 期响亮 ValueError 拒绝而非依赖 GetMemHandle 的下游报错；torch IPC 语义的
    base+offset 重构留待分配器布局实际变化时再做。回归
    `test_pair_split_ipc.py::test_export_rejects_offset_tensor`（fake CudaRTLibrary，
    CPU 可跑：偏移视图被拒、基址张量放行；1 passed 2 skipped GPU-only）。
- [x] **m7. `_peer_tags` 镜像无消费者死代码；`_CNT_P3_PEER_MISS` 从未自增**（2026-11-10 结案）
  - `mem_cache/kv_host_tier.py:1244-1314`。每 forward n_layers 次 D2D 拷贝纯开销。
  - **结案结论（选项 a：删除调用链）**：`_refresh_peer_tags` 整条链与 `_peer_tags`
    镜像一并删除（grep 确认无任何消费者），`_CNT_P3_PEER_MISS` 随之失去意义一并
    清理；每 forward 省 n_layers 次 D2D。KV tier CPU 套件回归通过。
- [x] **m8. phase3 下 graph replay 路径 step_seq 双增（replay 的 flush 缺 eager :1079 的对称守卫）**（2026-11-10 结案）
  - `layers/attention/qwen_sparse_attn_backend.py:1429-1432`（legacy 分支 :1466-1467 同）。
  - **结案结论**：replay 与 legacy replay 两处 flush 补齐 `if not phase3` 守卫，
    与 eager 路径对称。回归测试追加至 `test_qwen_sparse_sm70_routing.py`（4 用例，
    全套件 29 passed）：phase3 replay/legacy flush 不再 bump、非 phase3 正常 bump。
- [x] **m9. IPC 交换用 world rank 配对 attn_tp_rank^1；无偶数 tp 校验**（2026-11-10 结案）
  - `mem_cache/pair_split_ipc.py:117-122`。attn-TP 组≠world（DP/PP）或奇数 tp 时
    挂死/交叉映射。生产 attn_tp==world 且 TP4 偶数，不受影响。
  - **结案结论**：IPC 配对门加两条 boot 断言：`attn_tp_size == world_size` 与
    attn_tp_size 偶数，不满足时启动期 fail-loud（错误信息点名 DP/PP/奇数 TP 不支持
    PAIR IPC 交换）。回归测试覆盖奇数 tp、attn_tp≠world 两种拒绝形态。
- [x] **m10. `QSA_P3_FORCE_HOST`/`QSA_P3_CHECK` 裸读 os.environ，绕过 environ.py 契约**（2026-11-10 结案）
  - `mem_cache/kv_host_tier.py:1035-1036`。同文件其他 SGLANG_SM70_QSA_* 已入册。
  - **结案结论**：environ.py 注册 `SGLANG_V100_KV_TIER_P3_FORCE_HOST`/`_P3_CHECK`
    （EnvBoolWithAlias 保留 `QSA_P3_FORCE_HOST`/`QSA_P3_CHECK` 旧名别名 +
    DeprecationWarning，遵循 env-var-conventions），kv_host_tier.py 调用点迁移到
    `envs.*.get()`。回归 `test_environ.py::TestP3DebugSwitches`（别名回退/新名优先/
    未设默认）；KV tier CPU 子集 30 passed 2 skipped；KV_HOST_TIER_DESIGN.md 同步。
- [x] **m11. staged 后端下 update_weights_from_disk 必然失败**（2026-11-10 结案）
  - `models/qwen4_exp.py:1297-1345`。1 行 dummy + `_staged_materialized` 永久短路，
    权重更新无法落地且 NVMe 表永不刷新。仅影响热更新端点，serve 不用。
  - **结案结论（选项 b：启动期拒绝组合，fail-loud）**：`_release_loading_table`
    释放 PLE 表后给 dummy 参数挂 `_reject_weight_update` 标记，
    `update_weights_from_disk` 检测到 staged 后端在位即拒绝并给出复用 staged 表的
    指引（强制 re-materialize 需重新构建 12.2GiB NVMe 表，热更新端点不值得）。
    回归 `test_qwen4_exp_ple_prep.py::TestStagedWeightUpdateGuard`（staged 后端拒绝、
    非 staged 正常放行）；prep 41 passed + staged 25 passed。
- [x] **m12. `stage_async` 对 n>max_rows 直接 ValueError 杀 scheduler**（2026-11-10 结案）
  - `models/qwen4_exp_ple_staged.py:338-343`。阈值 262144 行 = 16384 tokens（已实测核实）；
    `chunked_prefill_size=-1` 或 >16384 的合法配置首条长 prefill 崩溃。
  - **结案结论（boot 期宽度防护）**：新增 `ensure_max_rows_for_prefill(rows_per_token)`
    （`rows_per_token = (ngram_size-1) × heads_per_ngram`，Qwen3.8 为 16），在
    `Qwen4ExpNGramEmbedding.__init__` staged 分支调用：`chunked_prefill_size ≤ 0`
    （-1/禁用，宽度无界）→ boot ValueError 拒绝；所需行数超过默认容量且
    `SGLANG_QWEN4_PLE_STAGED_MAX_ROWS` 未显式设置 → 自动上调并 warning；显式设置
    但不足 → ValueError（不静默覆盖操作者意图）。`get_schedule().chunked_prefill_size`
    在调度进程 publish 前 memory hook 已解析为具体值（arg_groups/pipeline.py:209），
    模型 init 期读取安全。回归 `test_qwen4_exp_ple_staged.py::TestEnsureMaxRowsForPrefill`
    （默认兼容 no-op / 超容自动上调 / 显式过小拒绝 / 显式相容放行 / -1+None+0 拒绝，
    TDD 红灯先行）；staged 30 passed、prep 41 passed；ruff 干净。
- [x] **m13. chunk-ahead miss 路径先等满整个 fill 再比较 ids**（2026-11-10 结案）
  - `models/qwen4_exp_ple_staged.py:520-534`。ids_np 本已是 host 侧 numpy，可先比较再等；
    miss 时白付整段 NVMe 随机行 IO 于关键路径。默认 CHUNK_AHEAD=0 生产不受影响。
  - **结案结论（先比后等）**：`wait_and_match` 改为先取 pending fill 在
    `stage_async` 时即固定的 host ids（无 pending 则取已 staged ids）与 forward
    自算 ids 逐位比对，仅命中才 `_wait_pending` 阻塞；miss 立即返回 False、零
    IO 等待，fill 继续后台运行（refill 路径 `stage`/`stage_async` 复用缓冲前
    自会等待未完 fill，无竞态）。`PleAheadStaging.consume` 调用点不变，miss 丢
    条目语义不变。PLE_NVME_TIER_DESIGN.md §11.9.5 同步。回归
    `test_qwen4_exp_ple_prep.py::TestPleRowWaitAndMatchCompareBeforeWait`（miss
    不调 `_wait_pending` 且 fill 仍在后台 + hit 等待且缓冲内容正确，TDD 红灯
    先行）；prep+staged 两套件 73 passed；ruff 干净。
- [x] **m14. pinned 分配失败静默回退 pageable，GPU 上 kernel 解引用 UB**（2026-11-10 结案）
  - `models/qwen4_exp_ple_staged.py:251-259`。
  - **结案结论（CUDA 可用时 fail-loud）**：`PleRowStaging.__init__` pinned 分配
    失败时以 `torch.cuda.is_available()` 分流——GPU 机器直接 RuntimeError（错误
    信息点名 gather kernel 直接解引用 buffer 指针、pageable 回退即 UB，并给出
    `ulimit -l`/host RAM 两条出路），仅无 CUDA driver 的 CPU 测试环境保留 pageable
    回退（该环境 gather kernel 永不触达此路径）。回归
    `test_qwen4_exp_ple_staged.py::TestRowStagingPinnedFailLoud`（CUDA 可用 +
    pin 失败 → RuntimeError；无 CUDA + pin 失败 → 非 pinned 回退，TDD 红灯先行）；
    prep+staged 两套件 75 passed；ruff 干净。
- [x] **m15. staged shard 文件名/meta 不含 checkpoint 指纹**（2026-11-10 结案）
  - `models/qwen4_exp_ple_staged.py:56-60`。REUSE=1（默认 0）时同形状异权重 checkpoint
    静默互用 12.2GiB 表。
  - **结案结论（模型路径指纹）**：新增 `checkpoint_fingerprint(name_or_path)`
    （sha1 前 10 位 + 文件系统安全的路径尾，`None`/空 → 空串），`PleStagedTable`
    构造函数新增 `fingerprint` 参数并混入 shard 文件名（`{tag}_{fingerprint}`）与
    header meta（显式 `"fingerprint"` 键，旧文件因值不符 → 强制重写）；
    `Qwen4ExpNGramEmbedding.__init__` 经 `_staged_checkpoint_fingerprint(config)`
    （取 `config._name_or_path`）传入。不选全量权重哈希：每次 boot 读 12+ GiB
    不值得，模型路径是足够便宜的代理。回归
    `test_qwen4_exp_ple_staged.py::TestCheckpointFingerprint`（helper 安全且可区分/
    异指纹换文件名/异指纹 meta 不符强制重写/同指纹 REUSE 复用，TDD 红灯先行）；
    prep+staged 两套件 79 passed；ruff 干净。
- [x] **m16. `_PLE_CHUNK_STASH` 请求完成/中止后不失效**（2026-11-10 结案）
  - `models/qwen4_exp_ple_prep.py:256-292`。slot 复用弹旧条目触发错误投机预读+满额等待
    （正确性由 consume ids 兜底）；多请求批的 stash 亦不被消费。
  - **结案结论（slot 回收点 pop）**：新增 `release_ple_chunk_stash(pool_idx)`，在
    `mem_cache/common.py::release_kv_cache`（finish/abort/retract 统一的 req slot
    回收 choke point，`req_to_token_pool.free(req)` 之后）惰性导入调用——条目不再
    可能活过请求生命周期：slot 复用者不再吃到过期投机 fill（省掉满额 fill 等待），
    多请求批从不被消费的条目也随之不再泄漏。mamba 早退路径（不持有 KV 直接 return）
    不经过该点，但该路径模型（DSV4.1）不使用 PLE chunk stash，无影响。回归
    `test_qwen4_exp_ple_prep.py::TestPleChunkStashRelease`（release pop/未知 slot
    no-op/record→release 生命周期/release_kv_cache 接线端到端，TDD 红灯先行）；
    prep+staged 两套件 83 passed；ruff 干净。
- [x] **m17. `_mixed_tail_mask` 回退防御缺口**（2026-11-10 结案）
  - `layers/attention/qwen_sparse_attn_backend.py:782-796`。把任何 prefix 非对齐行判为
    decode tail，且 mixed 批整体跳过 prefix 对齐 assert（:842-843）。触发前提（非对齐
    prefill 行进 mixed 批）被 schedule_policy.py:1945-1953 页对齐 snap 保护，属回退加固
    而非可达缺陷。
  - **结案结论（非 tail 行保留 device assert，零 sync）**：`_qsa_build_write_plan` 在
    mixed 批下不再整体跳过对齐不变量，改为 `torch._assert_async((prefix_lens[~mix_tails]
    % ratio == 0).all())`（async assert，无 D2H sync）；非 tail 行若 prefix 非对齐即
    响亮失败，防止半写共享压缩组；全 tail 批的空集断言恒真零开销；非 mixed 批严格断言
    不变。回归 `test_qwen_sparse_sm70_routing.py::TestMixedTailPrefixAlignmentAssert`
    （CPU 上直测 write plan：aligned+exempt tail 通过/mixed 非 tail 非对齐 RuntimeError/
    非 mixed 非对齐 RuntimeError/全 tail 通过，TDD 红灯先行），全套件 33 passed；ruff
    干净（存量行格式遗留不动）。
- [x] **m18. 混批预算 inflight 溢出整体 clear 丢弃全部在途采样**（2026-11-10 结案）
  - `managers/scheduler_components/mixed_prefill_budget.py:365-366`。应 FIFO 逐条驱逐。
  - **结案结论（FIFO 驱逐最旧一条）**：溢出时 `pop(next(iter(self._inflight)))` 仅驱逐
    插入序最旧的一条（dict 保插入序），其余并发 forward 的样本存活，采样流不再被
    破坏；被驱逐条目由 `process_batch_result` 的 `pop(key, None)` 无害忽略，确定性
    不受影响（纯本地函数）。回归 `test_mixed_prefill_budget.py::
    TestMixedPrefillBudgetControl::test_inflight_overflow_evicts_oldest_only`（饱和
    +1 条后：最旧被逐、其余存活、窗口恒等于 MAX_INFLIGHT_STEPS，TDD 红灯先行），
    全套件 31 passed；ruff 干净。
- [x] **m19. compose 与 serve 脚本 mixed-chunk 默认值分叉**（2026-11-10 结案）
  - `docker/v100-compose-qwen38-mtp.yaml:63`、`docker/v100-compose.yaml:70`（默认 750）
    vs `scripts/serve_qwen38_flash_next_nvfp4_v100.sh:149`（默认 0）。
  - **结案结论（注释登记差异，不改行为）**：分叉本身是 eb6cedfdf3 有意为之且已登记
    README——compose 是"推荐部署"载体故直接启用实测最优 750ms，裸机 serve 脚本保留
    0 以维持 pre-L3-1 行为需显式 opt-in。三个文件互相补加 divergence note 交叉引用
    （compose→脚本默认 0、脚本→compose 默认 750 及理由），行为零变更、无测试面。
- [x] **m20. compose 与 serve 脚本 PLE 后端分叉：脚本 staged NVMe、compose pinned RAM**（2026-11-10 结案）
  - `scripts/serve_qwen38_flash_next_nvfp4_v100.sh:327` vs compose（无
    `--ple-offload-backend`，走默认 pinned）。
  - **结案结论（注释登记差异，不改行为）**：分叉仅存在于
    `docker/v100-compose-qwen38-mtp.yaml`（绕过 serve 脚本的独立 compose）；
    `docker/v100-compose.yaml` 的 qwen38 服务经 entrypoint 调 serve 脚本，自动继承
    staged，无分叉。该 compose 有意保留 pinned：其机器事实（~125G RAM、无 hicache）
    按 pinned 尺寸标定，切 staged 属 §11.9.4 的实测 A/B 范畴而非默认值。在其
    `--ple-offload-embedding` 处补 divergence note（脚本 staged 的收益/代价、本
    compose 保留 pinned 的原因、opt-in 方法），行为零变更、无测试面。
- [x] **m21. 多模态 StopIteration 合并语义：混排请求静默丢其他模态 token**（2026-11-10 结案）
  - `multimodal/processors/base_processor.py:1536-1551`（早退点 :1142-1148）。
    纯图像单模态生产不受影响。
  - **结案结论（None 哨兵区分"数据缺失"）**：`submit_data_loading_tasks` 的
    StopIteration 早退改为 append `(None, None, None)` task_info + `None` future 哨兵
    并 `continue`（提交覆盖每个特殊 token，后续模态照常提交）；外层消费循环遇到
    modality 为 None 的哨兵条目时按 token 级套用同一 unmatched 策略（显式
    `SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP` 优先于 precomputed echo 保留启发式，
    都不满足则 RuntimeError），并同步消费配对的 None future 保持双迭代器对齐。
    原 `except StopIteration` 处理器保留作安全网。回归
    `test_base_processor_unmatched_tokens.py::TestStopIterationScopedToModality`
    （混排 prompt drop/默认两策略 + loader 级哨兵契约，TDD 红灯先行），本套件
    10 passed；bad_input + image_decode 套件 20 passed 无回归；ruff 干净。
- [x] **m22. ImportedDeviceBuffer 分支缺 e5m2→uint8 byte view**（2026-11-10 结案）
  - `tilelang_fa_v100/_decode_cuda.py:181-191`。e5m2+PAIR decode 响亮报错；
    fp16 生产不受影响。
  - **结案结论（peer 分支镜像本地 byte view）**：`_qsa_cache_view` peer 分支把
    `sm70_as_tensor` 收到的 dtype 由裸 `cache.dtype` 改为
    `torch.uint8 if cache.dtype == torch.float8_e5m2 else cache.dtype`（e5m2 每
    元素 1 字节，byte view 保 numel/shape 不变），与本地分支的
    `cache.view(torch.uint8)` 语义一致；fp16 peer 原样直传。新增 CPU 单测
    `test_qsa_cache_view_pair_dtype.py`（peer e5m2→uint8/peer fp16 直传/本地 e5m2
    view/本地 fp16 直传；stub `ImportedDeviceBuffer`+`_load_sm70_cuda_decode_ops`，
    TDD 红灯先行）4 passed；GPU 侧 `test_kv_host_tier_pair_gpu_kernels.py` 需真机
    （本机无 V100），e5m2+PAIR 的端到端验证留待目标机。ruff 干净（`Path` 存量
    未用 import 不动）。
- [x] **m23. grouped planner 第二个隐藏 D2H 同步**（2026-11-10 结案）
  - `tilelang_fa_v100/_kernels_grouped_sparse_prefill.py:271`。`int(group_slot[-1])`
    隐式 .item()，与 docstring "one scalar readback" 相悖。
  - **结案结论（host 侧形状上界，bound 只高估）**：key 宽度判决改为纯 host 推导
    `max_group_slot = ((R-1)//group_rows) * bs + (bs-1)`——每个 slot 都是
    g*bs+batch 且 g ≤ (R-1)//group_rows、batch ≤ bs-1，上界只高估，最坏情形是
    提前一个 regime 用 int64 key（多 4 字节/entry），不会腐蚀 plan。隐式
    `int(group_slot[-1])` 消除，planner 唯一 readback 恢复为 docstring 声明的
    显式 `int(u_local.max().item())`。新增 CPU 单测
    `test_grouped_union_plan_cpu.py`（`torch.Tensor.__int__` 桩防回归 + int32/
    int64 双 regime 各与纯 python 参考逐元素比对 uidx/mask，mask 按重复出现
    计数对齐 planner 的 index_add_ 累加语义；register_cpu_ci）：临时回退到旧
    readback 时防回归桩命中（红灯实证），恢复后 3 passed（绿灯）；ruff 干净。
    GPU 侧 `test_grouped_sparse_prefill_sm70.py` 需真机（本机无 V100），留待
    目标机跑 smoke + grouped A/B 数字确认。
- [x] **m24. A/B 脚本对任意 --base 无守卫调 /flush_cache**（2026-11-10 结案）
  - `scripts/ab_qsa_grouped_quality_v100.py`（原 :267-315，每次 arm 两处
    POST /flush_cache：冷 prefill 基线 + G1 同 boot 复跑）。误指 11435 生产端口
    会清 radix 缓存。
  - **结案结论（生产端口拒绝 + 显式 --assume-idle 旁路）**：新增
    `ensure_idle_test_target(base, assume_idle)`——`urlparse(base).port == 11435`
    （共享 V100 生产端点，`SGLANG_V100_HOST/PORT` 约定）且未带 `--assume-idle`
    时 `SystemExit` 并给出出路（指向空闲测试 boot 如 :8110，或显式确认 idle）；
    `run()` 入口第一行调用（先于 tokenizer 加载与任何网络请求），`run` 子命令
    新增 `--assume-idle` 开关，docstring usage 同步。旁路仅接受显式 CLI flag、
    不设 env 变量，避免惯性复制生产地址时静默穿透。新增 CPU 单测
    `test_ab_qsa_flush_guard.py`（拒绝/旁路/空闲端口三态 + run 接线端到端：
    生产端口时 0 次 flush、空闲端口时恰 2 次 flush；importlib 加载脚本模块，
    TDD 红灯先行）5 passed；CLI 实测 `run --base :11435` 打印拒绝消息退出；
    ruff 干净。

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
