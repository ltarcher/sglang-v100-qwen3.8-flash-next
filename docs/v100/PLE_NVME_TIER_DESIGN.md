# PLE n-gram 表 NVMe 分层(staged 后端)设计文档

状态:设计定稿,未实施(2026-11-04)。触发条件:KV host tier 需要扩
`SGLANG_V100_KV_HOST_TIER_GB`(当前被 PLE pinned 限制在 2),或需要恢复
hicache 文件分层时再启动实施。
日期:2026-11-04(分析含当日实测数字)
范围:Qwen3.8-Flash-Next NVFP4 线(4×V100-32GB SXM2,TP4,生产配置
`vllm-Qwen3.8/docker-compose-sglang-v100-mtp-8110.yaml`)
参考:`python/sglang/srt/models/qwen4_exp_ple_table.py`(现有 pinned/file 两后端)、
`docs/v100/KV_HOST_TIER_DESIGN.md`(被本方案解锁的内存约束)、
KV_HOST_TIER_DESIGN §环境事实("PLE 47.7G pinned 之上 tier 全镜像会宿主 OOM")

## 1. 背景与动机

Qwen3.8-Flash-Next 自带 PLE n-gram 表(`model-fp8-mtp-ple.safetensors`,51G fp8;
配置 `ngram_vocab_size_base=20,000,000` 行 × `ple_embed_dim=2560`B = 47.7GiB)。
TP4 下每 rank 持 5M 行 ≈ 12.2GiB shard。生产以 `--ple-offload-embedding`
(默认 `pinned` 后端)把它放在 host 内存——**51G 硬 pinned,永不可回收**。

当前内存代价(2026-11-04 实测,生产引擎 healthy 运行中):

| 项 | 数值 |
| --- | --- |
| host RAM 总量 | 125G(与 4090 workload、多个业务容器共享) |
| available | **40G**(swap 已用 11G,机器处于真实内存压力下) |
| PLE pinned | ~51G(1G hugepage 映射,计入 shared) |
| KV host tier pinned | ~24.8G(4×6.19GiB,Phase 3 镜像) |
| hicache | **被迫禁用**(compose 注释:hicache-size 8 实测要 ~134G,开了必被 OOM kill) |

把 PLE 表搬到 NVMe,把"硬 pinned 51G"变成"可回收页缓存 + 47.7G 盘上文件",
可用内存 40G → ~90G 弹性。这是本方案的全部动机;性能代价见 §5,在可接受带内。

## 2. 为什么不能直接用现有 `file` 后端

`qwen4_exp_ple_table.py` 已有 `file` 后端(sparse mmap + `MADV_RANDOM` +
`PleFilePrefetcher`(posix_fadvise WILLNEED)+ `PleFileRssTrimmer`),但它的
工作前提是 **GPU 直接解引用 pageable mmap 指针**——这要求设备报告
`cudaDevAttrPageableMemoryAccessUsesHostPageTables=true`,即统一内存架构
(GB10 / DGX Spark)。文件头注释与 `check_file_backend_supported` 都明确:
离散 GPU(含 V100)会直接 raise;`SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1`
跳过检查则 Triton gather 内核把 pageable VA 当设备指针用,读到垃圾或非法地址。

**结论:必须新增后端枚举值(建议 `staged`),不能复用 file 后端原样。**
file 后端的三件套(确定性文件名契约、prefetcher 思路、RSS 预算思路)可部分借鉴,
但 GB10 的 RSS trimmer 在本方案里不需要——页缓存本来就是可回收的。

## 3. 访问模式实测与代码证据

| 项 | 证据/数值 |
| --- | --- |
| 行号来源 | `_hash_contexts` → `_splitmix64`(qwen4_exp.py)对 2-token 上下文哈希 → **均匀随机**,无 Zipf 局部性可利用 |
| 每 token 行数 | decode fast path 每 token 1 行(`tokens.unsqueeze(1)`,上下文宽度 `ngram_size-1=2`);prefill 每 token 1 行 |
| 每 gather 工作集 | prefill chunk(4096 token)= 4096 行 × 2560B = 10MB/rank;decode bs=2 = 2 行/rank(2.5KB/rank) |
| TP 放大 | `VocabParallelEmbedding` 按 shard 取模映射,4 个 rank 对同一 id 各读**各自 shard** 的行,无跨 rank 去重;聚合 4×,量级安全 |
| TP shard | 每 rank 5M 行 × 2560B = 12.2GiB,4 个文件 |

**核心洞察:每次 gather 的工作集极小(prefill 10MB/rank、decode KB 级),
根本不需要用户态 LRU 缓存——Linux 页缓存就是现成的 L2,且天然可回收。**
覆盖 X% 的表就命中 X%(均匀随机下),缺的部分从 NVMe 补进每-gather 暂存区。

## 4. 方案设计

### 4.1 分层结构

```text
L0  pinned 暂存缓冲    每 rank:prefill ~10MB(4096 行 × 2560B)/ decode KB 级
     ↑ gather 前由 host 侧 IO 线程填入(缺行才读)
L2  页缓存(可回收)   内核自动管理;均匀随机下命中率 ≈ 覆盖率
     ↑ 缺页时从盘读(buffered IO,必须走页缓存,禁 O_DIRECT)
L3  NVMe 表文件        每 rank 12.2GiB shard × 4,离线转换器一次性生成
```

与 pinned 后端的本质差异:gather 内核读的仍然是一个 pinned host 指针
(零拷贝路径不变),只是从"全表"换成"暂存区 + 间接表"。

### 4.2 数据通路(两阶段 gather)

现状:row ID 在 GPU 上由 `_hash_contexts` 算出 → gather 内核直接
`table[row_id]`(图内零拷贝、零 sync)。

改为 gather 前两阶段:

1. **row ID 落 CPU**:decode 搭现有 per-step 采样 D2H 同步的车(上下文就是
   最近 2 个 token id,scheduler 侧本就有 CPU 副本,可在 CPU 侧复算哈希,
   不新增 sync;prefill chunk 本就有 D2H 元数据同步)。哈希逻辑在 CPU 侧
   复刻 `_splitmix64`(纯整数运算,单测对齐 GPU 结果)。
2. **host IO 线程填暂存**:`pread`(buffered)缺行进 pinned 暂存区,
   生成 id→slot 间接表(小,KB 级);H2D 间接表后 launch gather
   (内核仅改两个入参:基址=暂存区、ids=slot 表;`_gather_ple_embedding_from_pinned_kernel`
   本体逻辑不动)。

prefill 流水:`chunked-prefill-size=4096` 下 chunk IO ≈ 42MB(4 rank 聚合),
**跨 chunk 双缓冲**(GPU 算 chunk i 时预取 chunk i+1)即可把 IO 藏进计算。

### 4.3 离线转换器与 boot 路径

- 新增离线脚本:`safetensors checkpoint → 每 rank shard 文件`
  (沿用 `ple_table_file_name` 的确定性命名契约,目录
  `SGLANG_QWEN4_PLE_FILE_DIR`,与 file 后端同布局)。一次写成 ~47.7G,
  顺序写按实测 1.45 GB/s 约 33s。
- boot 只 open + 记录偏移,**不再每 boot 重写全表**(现状 pinned/file 后端
  都是 weight loader 每 boot 重写,NVMe 方案必须避免,否则每次 boot 白写 47.7G)。
- CUDA graph 捕获:capture 阶段 gather 内核真实执行,暂存区必须先于 capture
  填好。现有代码"capture 期跳过 prefetch"的假设要改为"capture 期 gather 的
  行已就位"(对 decode 图,捕获前对该 batch 的行做一次同步填入)。

### 4.4 环境变量契约(遵循 env-var-conventions,默认全关)

| 变量 | 语义 |
| --- | --- |
| `--ple-offload-backend staged` | 新枚举值;pinned/file 行为不变 |
| `SGLANG_QWEN4_PLE_STAGED_DIR` | shard 文件根目录(默认指向 NVMe 上的盘,不得是 tmpfs) |
| `SGLANG_QWEN4_PLE_STAGED_QD` | IO 并发深度(默认 64;实测 QD8 仅 0.23 GB/s,QD 提升预期显著) |
| `SGLANG_QWEN4_PLE_STAGED_DECODE_BUDGET_US` | decode 每步 IO 预算,超时则该步降级为读页缓存热路径并记录(防尾延迟尖刺) |

serve 脚本 `serve_qwen38_flash_next_nvfp4_v100.sh` 与 compose 注释同步更新。

## 5. 性能与收益预估(基于 2026-11-04 实测)

实测基线(`/data` LVM,SOLIDIGM P5620 级,生产引擎同机运行,数字偏保守):
4K 随机读 QD8:57K IOPS / 0.23 GB/s / p50 124μs / p99 337μs;顺序读 1.45 GB/s。
(注意:QD8 是单线程叠加的弱压测;io_uring QD≥64 应显著更好,且顺序带宽
低于该盘标称,疑受同机生产负载影响——实施前须在无生产负载窗口复测定档。)

| 维度 | 判据 |
| --- | --- |
| decode | 每步串行加 124~337μs(步长 ~10ms,含 MTP verify)→ **-1.3%~-3.4%**;页缓存热后趋近下界;2 并发暖聚合 68-133 tok/s 带内,贴近 -3% 门限,**必须实测验收**,不得凭估算放行 |
| prefill | 双缓冲后 ~0-5%;无流水则 ~13%(42MB / 0.23 GB/s ≈ 180ms vs chunk 计算 1.37s) |
| 内存 | 51G 硬 pinned → 可回收页缓存;available 40G → ~90G 弹性;缓解 11G swap;解锁 `SGLANG_V100_KV_HOST_TIER_GB` 从 2 上调 |
| boot | 零表写入,较现状更快;首访问行冷读 |
| 磁盘 | 表 +47.7G(/data 余 393G,可容纳);与 hicache 文件层同盘,IO 争用须在 README 表格登记 |

## 6. 正确性 oracle

1. **逐位一致**:staged 后端读到的字节与 pinned 后端完全相同(同一份 fp8
   数据,仅搬运路径不同)→ 同输入输出应与 pinned 后端逐位一致;这是本方案
   相比其他基础设施改动独有的强 oracle。
2. **Qwen3.8 NVFP4 回归**(仓库硬纪律):smoke_v100.sh 通过、验收命令数字
   不回退(prefill ~3,000 tok/s、decode ~100 tok/s/流、accept ~2.1)、
   生成质量不劣化。金丝雀就是本模型,回归目标与改动对象同一,验收最干净。
3. 页边界场景:行 2560B 跨 4K 页边界(row 偏移 = row_id × 2560)、shard
   取模边界、暂存区 slot 复用,单测必须覆盖(流式 KV 的教训:多 program
   grid 与页边界是高危区,禁止硬编码 grid)。
4. 单测放 `test/registered/unit/`,TDD 流程,先测后实现。

## 7. 风险清单

| # | 风险 | 缓解 |
| --- | --- | --- |
| 1 | decode 新增同步点破坏"图内零拷贝零 sync"契约 | row ID CPU 复算搭现有采样 D2H;间接表 H2D 与 gather 同 stream;超预算降级开关 |
| 2 | capture 期行未就位 → 图内读垃圾 | capture 前同步填入;单测覆盖 capture 路径 |
| 3 | 本机 NVMe 实测偏慢(疑同机负载干扰) | 实施前空载复测;`STAGED_QD` 可调;不达门限则放弃实施(方案可整批丢弃,零残留) |
| 4 | TP4 IO 放大 4× | 均匀随机 + 页缓存去重不成立(各 rank 读不同行),4× 是真实放大,但绝对量级 10KB/步,安全 |
| 5 | 与 hicache 文件层同盘争用 | 生产不同时开;README 登记 |
| 6 | 每页缓存回收抖动(内存压力时页缓存被清 → decode 突发全冷读) | p99 337μs 仍在预算内;`DECODE_BUDGET_US` 兜底 |

## 8. 被否决的替代方案

| 方案 | 否决原因 |
| --- | --- |
| 现有 `file` 后端原样上 | §2:离散 GPU 无 host page table 解引用,设备检查直接拒绝 |
| 用户态 LRU/direct-mapped pinned 行缓存(如 8G) | 每 gather 工作集仅 10MB,页缓存已覆盖该职责;LRU 引入 47.7G×命中率换算的复杂度,收益为负 |
| 缩小 PLE 表 / 量化到 fp4 | 表是 checkpoint 数据,重训练/转换改变数值,违背"不改模型"边界 |
| 保持 pinned + 压缩其他消费者 | KV host tier 与 hicache 是既定路线,空间刚需,无可压缩项 |

## 9. 实施顺序(触发后)

1. 空载复测 NVMe(4K 随机 QD64/128、顺序读写),定档并更新 §5 数字;
2. 离线转换器 + shard 文件生成,单测对齐 checkpoint 字节;
3. `staged` 后端:CPU 哈希复刻、IO 线程、间接表、内核入参改造(TDD);
4. prefill 双缓冲与 capture 前填入;
5. 同机同配置 A/B:staged vs pinned,吞吐数字判据(不用输出 hash 跨 boot
   比较,但 staged vs pinned 同进程内可逐位比对);
6. 过 Qwen3.8 NVFP4 全套回归后,再谈生产切换与 `KV_HOST_TIER_GB` 上调。

## 10. Phase 1 实施记录(2026-11-04)

Phase 1 已落地,TDD 流程(先测后实现,`test/registered/unit/models/
test_qwen4_exp_ple_staged.py` 13/13 通过;实现在 `python/sglang/srt/models/
qwen4_exp_ple_staged.py`):

| 交付物 | 内容 |
| --- | --- |
| shard 文件 | 带头(magic/version/形状/meta)的每 shard 密集文件;`materialize()` 权重加载后一次性写出,`REUSE=1` 时 header 匹配可跳写(默认关,checkpoint 同形状变更不串档) |
| 暂存区 | 每 gather 每行一个 slot 的 pinned 暂存;有效行线程池 preadv 直写(缺行才读),无效 slot 全量清零——slot 跨 gather 复用天然安全 |
| gather 拦截 | `Qwen4ExpPinnedHostEmbedding._gather_staged`:D2H ids → 填暂存 → **内核零改动**(staging 指针 + slot arange + 词表区间展宽,界外行为与 pinned 的 `in_range` 清零语义逐位一致) |
| 生命周期 | `materialize_staged_table()` 在 `load_model_utils` 加载完成后调用,随后释放 12.2GiB/rank pinned 加载表(dummy 参数保留 dtype) |
| 环境变量 | `SGLANG_QWEN4_PLE_STAGED_DIR/_IO_THREADS/_MAX_ROWS/_REUSE`(默认值维持生产行为不变;pinned/file 后端路径零改动) |
| 防呆 | `--ple-offload-backend staged` 必须搭配 `--ple-offload-embedding`;CUDA graph capture 期 gather 显式 raise(提示加 `--cuda-graph-config '{"decode":{"backend":"disabled"}}'`) |

**Phase 1 边界(明确限制)**:

1. **decode CUDA 图不支持**:生产 decode 默认 FULL 图(monolithic replay,
   无 host 参与),staged gather 无法在图内取数。capture 期直接 raise,
   Phase 1 验证需 `--cuda-graph-config '{"decode":{"backend":"disabled"}}'`。
   Phase 2 方案:复用解码图尾部把"下一步 gather 行号"经 captured D2H 写入
   pinned 请求缓冲,调度器 CPU 相位离线填暂存,零新增 sync、零 CPU 哈希复刻;
   MTP verify 路径需把 prep 挂到 draft 图尾部(draft 完成本就是 verify 的
   硬依赖,同步代价不额外串行)。
2. **GPU 数值验证未跑**:staged vs pinned 逐位一致 oracle 需真机
   (`test_qwen4_exp_ple_gpu.py` 风格);本阶段在 s9-pytest 容器(CPU)验证
   了 IO 层全部边界(页跨界、slot 复用、线程数无关、shard 边界)。
3. 懒物化兜底:若 post-load 钩子未覆盖(如 draft worker),首次 gather 会
   现场物化(日志可见),首次请求多付一次 12.2GiB 顺序写。
4. boot 每次重写 shard 文件(REUSE 默认关):12.2GiB/rank 顺序写,实测顺序
   1.45 GB/s 下约 8-10s/rank;`SGLANG_QWEN4_PLE_STAGED_REUSE=1` 可跳过。

### 新增文件与改动面

- 新增:[qwen4_exp_ple_staged.py](file:///data/develop/sglang-v100-sxm2-qwen3.8-flash-next/python/sglang/srt/models/qwen4_exp_ple_staged.py)、
  [test_qwen4_exp_ple_staged.py](file:///data/develop/sglang-v100-sxm2-qwen3.8-flash-next/test/registered/unit/models/test_qwen4_exp_ple_staged.py)
- 修改:`qwen4_exp.py`(staged 分支 + `_gather_staged` + capture 守卫 + 物化/释放)、
  `environ.py`(4 变量)、`memory_hook.py`(staged 校验)、
  `load_model_utils.py`(post-load 物化)、`fields/exec_.py`(CLI choices + help)
- 生产影响:零(默认后端仍为 pinned,全部开关默认关)

## 11. Phase 2 实施方案:decode CUDA 图的 pre-replay staging(设计定稿,未实施)

### 11.1 目标与验收门

解锁 staged 后端与 decode CUDA 图(含 MTP draft/target-verify)共存,摘除
Phase 1 的 `--cuda-graph-config '{"decode":{"backend":"disabled"}}'` 限制。
验收门(同机同配置 A/B,pinned 为基线):

1. staged vs pinned 输出**逐位一致**(oracle,同进程内双后端比对);
2. 2 并发 262k/64k 暖聚合 decode 吞吐 ≥ 基线 -3%(生产口径 100 tok/s);
3. MTP accept 长度不回退(基线 ~2.1)、零 retract;
4. Qwen3.8 NVFP4 smoke + 探针套件全过。

### 11.2 架构:prep → fill → gather 三段式

Phase 1 已保证图内 gather 读的两个指针跨 replay 稳定(staging 基址 +
slot arange)。缺的只是:**replay 前 host 必须把"这一步将要 gather 的行"
填进 staging**。Phase 2 用 captured D2H 让图自己把下一步的行号交给 host,
**不需要 CPU 哈希复刻、不新增 sync**:

```text
步骤 t-1 的图(捕获期即包含):
  ... decoder layers ... → 采样(图内)→ PLE commit
  → [prep kernels] 对每个 PLE 层 L,用图内一致的状态
    (history + 本步采样 token)复算步骤 t 的 gather 行号
  → captured cudaMemcpyAsync D2H → pinned prep 请求缓冲

replay 返回后(host 侧,调度器 CPU 相位,不在 GPU 关键路径上):
  调度器本来就等图完成取采样结果 → 顺带读 pinned prep 缓冲
  → 线程池 pread 缺行入 staging(Phase 1 的 stage() 原样复用)
  → 启动步骤 t 的图 replay

步骤 t 的图:每层 PLE 的 staged gather 读 staging(Phase 1 路径零改动)
```

关键性质:prep 与 gather 用**同一套图内核、同一份输入状态**——prep 只是
把 `compute_ngram_ids` 在"下一步"的输入上重跑一遍,一致性由构造保证,
不存在双实现漂移。

> **修订(2026-11-04,§11.6.1 验证后)**:实测采样在 decode 图**外**
> (eager,`model_runner.sample` 在 replay 后由 tp_worker 调用),prep 不需要
> captured D2H——改为 **sample 之后同流 eager 发 prep 内核 + 普通 D2H**,
> 图捕获零改动;上图的"图尾 captured D2H"与"[prep kernels] 在图内"作废。
> 一致性论据不变:prep 与图内 gather 复算用的是同一 pool history(图内
> commit,stream-ordered)+ 同一采样 token(同流 sample 产物)。

### 11.3 数据结构

| 结构 | 位置 | 生命周期 |
| --- | --- | --- |
| prep 请求缓冲 `next_ids[L][bs_max*rows_per_tok]` | pinned,图尾 D2H 目标(地址稳定,捕获要求) | 每 step 覆写 |
| mismatch 位图 `prep_mismatch[L]` | pinned | 每 step 覆写 |
| staging + slot arange | pinned + device | Phase 1 已有 |
| prep 输入(history 视图、采样 token 视图) | device 静态缓冲 | 图捕获期接线 |

`rows_per_tok = (ngram_size-1) × heads_per_ngram = 16`;4 个 PLE 层 ×
bs_max(4)× 16 = 256 ids/step,D2H 量 KB 级。

### 11.4 模式分解

| 模式 | prep 挂载点 | 说明 |
| --- | --- | --- |
| 纯 decode | decode 图尾 | history + 图内采样 token → 下一步 decode 行号 |
| MTP draft | draft 图尾 | draft token 图内可得,复算 **verify 模式**行号(draft stride 布局) |
| target-verify | 消费 draft prep | verify 图 gather 的 ids 必须与 draft 图 prep 的一致(mismatch 位图兜底) |
| prefill/extend/verify-eager | 无 prep | eager 路径沿用 Phase 1 的即时 D2H staging |
| 首 step / 新 req / retract 后 | host 兜底 | prep 缓冲带"无效"标记;host 见标记则该步退化为 eager gather(禁用该 step 图,或接受一次 D2H sync) |

**边界场景(必须覆盖)**:req 进入/退出(free slot 的 prep 行号无效化)、
retract(序列回退,history 与 prep 失配 → mismatch 位图报警 + eager 回退)、
bs bucket 变化(每 bucket 独立 prep 缓冲,与图的静态 shape 对齐)、
pad 行(state_indices 指向 dummy slot 0,prep 同样产 pad 行号,gather 读
pad 行无害——与现状一致)。

### 11.5 一致性自检(mismatch 位图)

每个含 staged gather 的图在捕获时追加一个比对内核:图内真实
`lookup_ids` vs prep 预测行号,逐行比较写 pinned 位图。host 在既有每步
sample sync 上顺带读位图(4B):

- 全零 → 正常;
- 非零 → 立即 raise(**fail loud,不做静默纠错**),日志带 layer/行/两侧行号。
  这是防"prep 与 gather 漂移"的最后防线,位图检查本身 D2H 仅 4B。

### 11.6 图捕获细节与前置验证项

实施第一天必须先验证的三个假设(任一不成立则方案回炉):

1. **采样在 decode 图内**(本 fork 假设成立):否则 prep 改挂"采样后、
   图外的 host 点",代价为一次显式 sync,方案降级但仍成立;
2. **captured D2H 可行**:multi-stream capture 下(Phase 1 确认 gather 已
   在 side-stream 被捕获)`cudaMemcpyAsync` D2H 到 pinned 同样可捕获,
   用 10 行 toy graph 验证;
3. **PLE commit 时序**:`_commit_ple_batch` 在图内的位置允许 prep 读到
   "commit 后"的 history(commit 在采样后即可;若 commit 在下一图首,
   prep 需自带 commit 语义,复杂度上升一档,单独评估)。

捕获期 prep 内核会真实执行一次(capture runs kernels):首捕获用 dummy
输入,pinned 请求缓冲内容无效 → 首个真实 step 前必须有一次 eager warm
gather 或 host 强制标记"staging 未就位"→ 首 step 走 eager 回退(与
新 req 兜底同路径)。

#### 11.6.1 前置验证结果(2026-11-04,代码考古定论,未跑 GPU toy)

| # | 假设 | 判定 | 证据链 |
| --- | --- | --- | --- |
| 1 | 采样在 decode 图内 | **不成立**(方案反而简化) | decode 图只捕获 `patch_model(model)` 的纯模型 forward(`decode_cuda_graph_runner.py` capture_one_shape L1224-1232 的 `run_once`);`model_runner.sample` 是独立方法,在 replay 返回 logits_output 后由 `tp_worker.forward_batch_generation` eager 调用(L687-691)。因此 prep 全部挂图外:sample 之后同流 eager 发 prep 内核 + 普通 cudaMemcpyAsync D2H,无 capture、无 captured memcpy,mismatch 检查同样图外 |
| 2 | captured D2H 可行 | **moot** | 假设 1 不成立的直接收益:captured D2H 不再出现在任何路径上,风险 #2 消解。为已无必要的 toy 冒生产 OOM 风险(每卡仅余 ~1.3GB)不做 |
| 3 | PLE commit 时序 | **成立**(比预期更干净) | `_commit_ple_batch` 在模型 forward 内、全部 decoder 层之后、final norm 之前(L2108-2111),随图执行;decode fast path 把 `[X_{t-1}, X_t]` 写回 pool(L424-433)。图外 prep 同流读 pool 天然看到 commit 后状态。行号语义:步骤 t+1 gather 行号 = `hash([X_t, X_{t+1}])`,X_{t+1} 即步骤 t 的 sample 输出——prep 输入齐备 |
| 4(新) | overlap 调度下 fill 的落点 | **关键约束,非新成本** | `event_loop_overlap`(scheduler.py L2074-2087):`run_batch(t+1)` 在 `pop_and_process()`(resolve t)**之前** launch,故 CPU 在 launch replay(t+1) 时不可能已读到 prep(t) 结果 → fill 必须有 launch 前的 sync gate(sample(t) event → 读 prep 缓冲 → fill → launch replay)。该串行窗口即 §5 已预算的每步 124~337μs(-1.3~-3.4%),不是新增成本;稳态页缓存全热后趋近 memcpy 下界 |

结论:三假设中唯一不成立的(1)带来的是**架构简化**而非降级;方案核心
(prep→fill→gather 三段、零 CPU 哈希复刻、mismatch fail-loud)全部保留。
前置验证阶段完成,无 GPU 侧遗留;下一个 GPU 触点是 §11.9 第 3 步的
逐位 oracle。

### 11.7 TDD 测试清单

CPU 侧(`test/registered/unit/models/`,先测后写):

| 测试组 | 覆盖 |
| --- | --- |
| prep 请求缓冲 | 布局/stride、多 layer 多 bucket 寻址、无效标记语义、覆写不复用旧值 |
| fill 调度 | prep→stage() 桥接、首 step eager 回退、retract 无效化路径、mismatch 非零即 raise |
| 边界 | bucket 切换、pad 行、req 进出、max_rows 超限(与 Phase 1 同标准) |

GPU 侧(真机,复用 Phase 1 的容器外路径):

| 测试组 | 覆盖 |
| --- | --- |
| toy captured D2H | §11.6 假设 2 |
| 逐位 oracle | 同进程 pinned 后端 vs staged 后端,同输入 decode 32 步输出逐位一致 |
| mismatch 注入 | 人为改 prep 输入 → 位图置位 → raise |
| accept/吞吐 | A/B 门(§11.1) |

### 11.8 环境变量与开关

| 变量 | 语义 |
| --- | --- |
| `SGLANG_QWEN4_PLE_STAGED_DECODE_PREP = EnvBool(True)` | Phase 2 主开关;False 退回 Phase 1 行为(capture 拒绝),便于 A/B 与回退 |
| `SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG = EnvBool(False)` | DEBUG_ 级:每步 dump prep vs 实际行号(oracle 调试用) |

staged 后端 + decode 图 + `DECODE_PREP=False` → 启动期显式 raise
(把 Phase 1 的 capture 期错误提前到启动)。

### 11.9 实施顺序

1. ~~§11.6 三个假设的验证脚本~~ **已完成(2026-11-04,§11.6.1)**:全部由
   代码考古定论,假设 1 不成立但带来架构简化(prep 图外化),假设 2 moot,
   新增假设 4(overlap launch gate);GPU toy 不需要;
2. CPU 单测组(prep 缓冲 / fill 调度 / 边界);
3. prep 内核接线:decode 无 MTP 先行(最简路径),端到端跑通"逐位 oracle";
4. MTP draft/verify prep 接入 + mismatch 自检;
5. 真机 A/B:吞吐门 + accept + 262k/64k 阶梯;
6. 摘除 decode-disabled 限制,更新 compose 注释与 README 表格,过
   Qwen3.8 NVFP4 全套回归。

### 11.10 风险清单

| # | 风险 | 缓解 |
| --- | --- | --- |
| 1 | prep 与图内 gather 的行号漂移(EOS 语义、commit 时序) | mismatch 位图 fail-loud;同内核同输入构造保证;DEBUG dump 对照 |
| 2 | captured D2H 在 multi-stream capture 下不合法 | **已消解(§11.6.1)**:prep 图外化后无 captured memcpy |
| 3 | 首 step / 异常路径回退到 eager 造成吞吐抖动 | 回退仅限首 step 与 retract 后首步;日志计数,稳态占比应为 0 |
| 4 | staging 未就位时图已启动(调度竞态) | prep 缓冲"已消费"标志,fill 未完成前禁止 replay(调度器单线程序天然满足,断言兜底) |
| 5 | 每 bucket prep 放大捕获体积(4 层 × 256 ids) | D2H KB 级;图体积增量 <1%,捕获时间可忽略 |
| 6 | MTP verify 布局对齐(draft stride) | §11.4 单列;先无 MTP 交付,MTP 作为独立增量门 |
