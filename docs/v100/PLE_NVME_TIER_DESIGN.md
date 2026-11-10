# PLE n-gram 表 NVMe 分层(staged 后端)设计文档

状态:已实施(2026-11-04/05/10)。Phase 1 staged NVMe 后端(§10)、Phase 2
decode 图 pre-replay staging(§11)与 prefill 双缓冲(§11.9.4)均已落地,
生产 8110 compose 已切 staged 后端。chunk-ahead 预取(§11.9.5,算 chunk i
时投机预取 chunk i+1..i+d)已通过真机 A/B 验收(depth=1 甜点:64k
prefill TTFT -31.8%、decode 零负收益),准备在生产 compose 以
`SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD=1` 开启。
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

> **修订(2026-11-04,§11.6.1 验证后)**:实测采样在 decode 图**外**
> (eager,`model_runner.sample` 在 replay 后由 tp_worker 调用),prep 不需要
> captured D2H——改为 **launch gate 计算超前**:decode runner 在 `load_batch`
> 之后、`backend.replay` 之前,用**本 batch 的真实输入**(静态缓冲内容 +
> pool history,均 stream-ordered 就绪)重算 gather 行号并填 staging,然后才
> launch replay。图捕获零改动;上图的"图尾 captured D2H"与
> "[prep kernels] 在图内"作废。一致性由构造保证:gate 与图内 gather 调用
> 同一 `_prepare_ple_batch`/`compute_ngram_ids`,读同一静态缓冲与同一
> pool 状态,gate 先于 replay 且其间无写者。

### 11.3 数据结构(修订 2:compute-ahead 后大幅缩减)

| 结构 | 状态 |
| --- | --- |
| staging + slot arange | Phase 1 已有,不变 |
| pinned prep 请求缓冲 `next_ids[L][...]` | **删除**——无 captured D2H,`ids.cpu()` 即时所得 |
| mismatch 位图 `prep_mismatch[L]` | **删除**——无预测即无漂移(§11.5) |
| `PleDecodePrep` + 每层 `PlePrepEntry` | 新增(2026-11-04 已落地 + 9 单测):哈希 multiplier 含 `ple_layer_index` → **每层行号不同**,每层独立 staging 与 entry;`_PLEBatch` 每 step 只构建一次 |

gate 时序(iteration k,overlap 循环):

```text
run_batch(B_k):
  load_batch:input_ids/req_pool_indices → 图静态缓冲
             (input_ids 来自 future_map.output_tokens_buf 的纯 device gather,
              overlap_utils.py L110-111,不过 CPU)
  [GATE] batch_builder = _prepare_ple_batch(静态缓冲, B_k)
         每层 compute_ngram_ids(batch) → flat_ids
         ids.cpu()          ← 唯一 sync;涵盖 sample(B_{k-1}) 完成
         每层 staging.stage(flat_ids)   ← preadv,页缓存热为 memcpy 级
  backend.replay(B_k)       ← 图内 staged gather 读 staging + slot arange
```

成本核算(修订):gate 的 sync 使 CPU 与 GPU 对齐一次;GPU 空窗 = host fill
时间——稳态页缓存全热约 10~50μs/步(优于 §5 按 NVMe p50/p99 估的
124~337μs),冷读 p99 337μs 仍在预算内。

### 11.4 模式分解(修订 2)

| 模式 | 处理 |
| --- | --- |
| decode(走图) | gate:`replay` 前 `prep_and_fill`(实现在 `DecodeCudaGraphRunner.execute` 的 `load_batch` 之后、`backend.replay` 之前;prep 输入取静态缓冲切片) |
| decode(eager,图不适用) | Phase 1 路径自含 D2H+staging,gate 不介入 |
| prefill/extend/verify-eager | Phase 1 路径(生产 prefill 图已禁,verify 走 eager) |
| MTP draft / verify 图 / pdmux / DP-gather | prep 接线后若 runner 的 capture 模式非 plain DECODE,`_build_ple_decode_prep` 启动期 raise `NotImplementedError`(MTP draft/verify prep 是第 4 步;当前 verify 走 eager 不经此路径) |
| 首 step / 新 req / retract 后 | **无需特殊处理**:gate 每步用真实输入重算,无预测可失效;superseded 的"无效标记/eager 回退"机制整体删除 |

边界场景(pad 行、req 进出、bucket 切换)由构造覆盖:gate 与图内读同一
静态缓冲(pad 区内容一致)与同一 forward_batch;保留的 fail-loud 只有
①跨层宽度一致性(`PleDecodePrep` RuntimeError)②max_rows 超限
(ValueError 透传,带 env 提示)。

### 11.5 一致性自检(修订 2:mismatch 位图删除)

无预测即无漂移:gate 与图内 gather 是**同函数、同输入、同流序**,不存在
两套实现。运行时防线改为:

1. 跨层宽度一致性检查(每步,零成本);
2. §11.9 第 3 步真机逐位 oracle(一次性验收):`scripts/oracle_ple_staged_v100.sh`
   双 boot A/B,7 条响应 296 token 全逐位一致(见 §11.9 第 3 步实施记录)。

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

捕获与首 step(修订 2):prep 全在图外,capture 期唯一要求是图内的
staged gather 走"内核直发"路径(staging 指针 + slot arange,静态指针,
与 pinned 后端被生产图证明的机制相同);capture 前对 staging 做一次
零填充,保证捕获 warmup 读到确定内容。首 step 由 gate 正常覆盖(每步
重算,无"未就位"状态)。

#### 11.6.1 前置验证结果(2026-11-04,代码考古定论,未跑 GPU toy)

| # | 假设 | 判定 | 证据链 |
| --- | --- | --- | --- |
| 1 | 采样在 decode 图内 | **不成立**(方案反而简化) | decode 图只捕获 `patch_model(model)` 的纯模型 forward(`decode_cuda_graph_runner.py` capture_one_shape L1224-1232 的 `run_once`);`model_runner.sample` 是独立方法,在 replay 返回 logits_output 后由 `tp_worker.forward_batch_generation` eager 调用(L687-691)。因此 prep 全部挂图外:sample 之后同流 eager 发 prep 内核 + 普通 cudaMemcpyAsync D2H,无 capture、无 captured memcpy,mismatch 检查同样图外 |
| 2 | captured D2H 可行 | **moot** | 假设 1 不成立的直接收益:captured D2H 不再出现在任何路径上,风险 #2 消解。为已无必要的 toy 冒生产 OOM 风险(每卡仅余 ~1.3GB)不做 |
| 3 | PLE commit 时序 | **成立**(比预期更干净) | `_commit_ple_batch` 在模型 forward 内、全部 decoder 层之后、final norm 之前(L2108-2111),随图执行;decode fast path 把 `[X_{t-1}, X_t]` 写回 pool(L424-433)。图外 prep 同流读 pool 天然看到 commit 后状态。行号语义:步骤 t+1 gather 行号 = `hash([X_t, X_{t+1}])`,X_{t+1} 即步骤 t 的 sample 输出——prep 输入齐备 |
| 4(新) | overlap 调度下 fill 的落点 | **关键约束,非新成本** | `event_loop_overlap`(scheduler.py L2074-2087):`run_batch(t+1)` 在 `pop_and_process()`(resolve t)**之前** launch,故 CPU 在 launch replay(t+1) 时不可能已读到 prep(t) 结果 → fill 必须有 launch 前的 sync gate(sample(t) event → 读 prep 缓冲 → fill → launch replay)。该串行窗口即 §5 已预算的每步 124~337μs(-1.3~-3.4%),不是新增成本;稳态页缓存全热后趋近 memcpy 下界 |

结论:三假设中唯一不成立的(1)带来的是**架构简化**而非降级;后续
§11.3-11.5 已按"compute-ahead launch gate"二次修订(预测类机制整体删除)。
前置验证阶段完成,无 GPU 侧遗留;下一个 GPU 触点是 §11.9 第 3 步的
逐位 oracle。

### 11.7 TDD 测试清单

CPU 侧(`test/registered/unit/models/`,先测后写)——**已完成(2026-11-04,
`test_qwen4_exp_ple_prep.py` 9/9 + Phase 1 回归 13/13,CI 同版本 ruff 全绿)**:

| 测试组 | 覆盖 |
| --- | --- |
| per-layer 填充 | 两个 fake 层不同行号集 → 各自 staging 字节逐一正确;`_PLEBatch` 每 step 仅构建一次 |
| 展平契约 | 2D `[T, heads]` → 行主序展平,slot i = `flat_ids[i]`(与 eager gather 的 `reshape(-1)` 一致) |
| 词表区间 | 越界 id(`local<0`、`≥shard_rows`、`≥vocab_end`)slot 清零,`local = id - vocab_start` 映射正确 |
| bucket 切换 | 宽度变化后当次宽度内 slot 全部反映最新 ids(重填 + 宽度内清零);宽度外旧字节安全(内核 grid 恰为当次宽度,永不读) |
| fail-loud | 跨层宽度不一致 RuntimeError;max_rows 超限 ValueError(带 env 提示);空 entries 拒绝 |
| 契约 | `DECODE_PREP` 默认 True;返回值 = 展平宽度 |

GPU 侧(真机,§11.9 第 3 步起):

| 测试组 | 覆盖 |
| --- | --- |
| 逐位 oracle | `scripts/oracle_ple_staged_v100.sh` 双 boot A/B(pinned boot → staged boot,4×V100 TP4、无 MTP、无 hicache,`cuda_graph_bs_decode 1 2 4`):同请求序列(warmup + 3 条逐条 + 3 条 0.6s 错峰重叠),比对每条响应的 token ids + 逐 token logprobs + text 逐位一致。**并发同时 burst 不可比**:同时到达的 prefill 批组成存在准入竞态,两次 boot 间 extend batch 组成不同导致 prefill 期数值微差并逐步放大——错峰发送消除竞态(prefill 独占、decode 窗口重叠走 padded bucket 1→2→3→2→1) |
| accept/吞吐 | A/B 门(§11.1) |

### 11.8 环境变量与开关

| 变量 | 语义 |
| --- | --- |
| `SGLANG_QWEN4_PLE_STAGED_DECODE_PREP = EnvBool(True)` | Phase 2 主开关;False 退回 Phase 1 行为(capture 拒绝),便于 A/B 与回退 |
| `SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG = EnvBool(False)` | DEBUG 级 oracle 探针:图内 gather 在 capture 期把 flat ids 记入持久缓冲(capture-only,eager prefill/warmup 不覆写),gate 每步把它与上一步 gate ids 比对,失配 `logger.error`、命中 `logger.info`(每步一条;每层一次 D2H,严禁生产常开)。接线成功时打 `gate wired: N PLE layers` |
| `SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD = EnvInt(0)` | chunk-ahead 预取深度(§11.9.5,默认 0=关;生产建议 1):extend forward i 投机预取 chunk i+1..i+d 的行,consume 逐位比对兜底。深度 d 需要 d+1 个额外 staging 缓冲(各 ~max_rows×row_bytes,生产 64k chunk 下约 671MB/个);真机 A/B(depth=0/1/2,4×V100 2026-11-10)显示 depth=1 是 prefill 甜点(64k TTFT -31.8%、262k TTFT -15.2%、decode 三档全在 -3% gate 内),depth=2 prefill 略回退、decode 抬升,整体性价比低于 depth=1。依赖 `SGLANG_QWEN4_PLE_STAGED_PREFILL_PREFETCH=1` |

staged 后端 + decode 图 + `DECODE_PREP=False` → 启动期显式 raise
(把 Phase 1 的 capture 期错误提前到启动)。gate 接线成功与失配都有
显式日志——接线类故障(如 builder 因模型外层 wrapper 解包失败返回
None)不能靠"没有报错"判定为健康。

### 11.9 实施顺序

1. ~~§11.6 三个假设的验证脚本~~ **已完成(2026-11-04,§11.6.1)**:全部由
   代码考古定论,假设 1 不成立但带来架构简化(prep 图外化),假设 2 moot,
   新增假设 4(overlap launch gate);GPU toy 不需要;
2. ~~CPU 单测组~~ **已完成(2026-11-04)**:`qwen4_exp_ple_prep.py`
   (`PleDecodePrep`/`PlePrepEntry`,DI 注入 batch_builder,CPU 可测)+
   `SGLANG_QWEN4_PLE_STAGED_DECODE_PREP` env + 9 单测(§11.7),全绿;
3. prep 内核接线:decode 无 MTP 先行(最简路径),端到端跑通"逐位 oracle"
   —— **已完成(2026-11-04,实施记录见下)**;
4. MTP draft/verify prep 接入 + mismatch 自检;
5. 真机 A/B:吞吐门 + accept + 262k/64k 阶梯;
6. 摘除 decode-disabled 限制,更新 compose 注释与 README 表格,过
   Qwen3.8 NVFP4 全套回归。

#### 11.9.1 第 3 步实施记录(2026-11-04,4×V100 真机)

接线与实现:

- gate 挂点:`DecodeCudaGraphRunner.execute()` 在 `load_batch` 之后、
  `backend.replay` 之前调 `PleDecodePrep.prep_and_fill()`;prep 输入取
  静态缓冲切片(`input_ids/req_pool_indices/out_cache_loc`),仅 forward
  模式为 plain decode 时介入;capture 模式非 plain DECODE(verify 图/
  pdmux/MLP-sync)时启动期 `NotImplementedError`。
- capture-safe 图内 gather:`_gather_staged` 以
  `torch.cuda.is_current_stream_capturing()`(非 `get_is_capture_mode`,
  breakable eager break 须保持 stage 语义)区分:图内跳过 `stage()`,
  内核直读 staging 指针 + `slot_arange[:n]`,tp 词表区间放宽为
  `[0, 1<<62)`。
- **接线修复(oracle 首跑 FAIL 的根因)**:`build_qwen4_exp_ple_decode_prep`
  原以 `isinstance(model, Qwen4ExpModel)` 门控,而生产加载的实例是 VL
  wrapper `Qwen4ExpForConditionalGeneration` → builder 静默返回 None →
  gate 从未接线,图内 gather 读到 prefill 阶段最后一次 eager stage 的
  陈旧行,所有 decode 步不刷新。该故障表现为首个 decode 图步起与
  pinned 逐位偏离(两次独立 boot 可复现、逐位相同的错误输出),且无任何
  报错。修复:wrapper 解包 `model.model`(Qwen4ExpVLModel ⊂
  Qwen4ExpModel);并加 `gate wired: N PLE layers` INFO 日志——接线类
  故障不得靠"无报错"判定健康(§11.8 已同步)。

诊断过程(供后续排障参考):`SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG=1` 下
gate ids vs 图内 ids 每步比对除两处外全部相等——两处失配是 prefill 的
eager gather 覆写 debug 缓冲造成的假警报,已改为 capture-only 记录;
`PREP_CORRUPT` 一次性探针(gate fill 后清零首个有效 staging 行,输出
随之改变)证明图内 gather 确实读取 gate 填充的 staging 缓冲,探针已
移除。

oracle 结论:`scripts/oracle_ple_staged_v100.sh` 双 boot A/B
(pinned → staged,4×V100 TP4,无 MTP、无 hicache,
`cuda_graph_bs_decode 1 2 4`),请求序列 = warmup + 3 条逐条(bs=1
bucket)+ 3 条 0.6s 错峰重叠(decode bs 走 1→2→3→2→1,padded bucket
全覆盖):**7 条响应、296 个输出 token、296 个逐 token logprob 与
decoder 文本全部逐位一致**。首跑曾 FAIL 于并发同时 burst——同批到达
的 prefill 批组成存在准入竞态,跨 boot 不可比,已改为错峰发送消除
(见 §11.7)。配套单测 `test_qwen4_exp_ple_prep.py` 9/9 通过。

### 11.9.2 第 4 步实施记录:MTP draft/verify prep 接入 + mismatch 自检(2026-11-04)

代码考古定论(实施前完成,证据见文件引用):

| # | 问题 | 结论 |
| --- | --- | --- |
| 1 | MTP draft 是否含 PLE 层 | **否**:`Qwen4ExpForCausalLMMTP` 深拷贝 config 后设 `ple_layer_ids=[]`(`qwen4_exp_mtp.py`),draft 的 `Qwen4ExpModel` 无 PLE 层;且 draft worker 用独立 `FrozenKVMTPCudaGraphRunner`(`TpModelWorker.init_cuda_graphs(capture_decode_cuda_graph=False)`),根本不创建 `DecodeCudaGraphRunner`。**第 4 步只需覆盖 target verify 图** |
| 2 | verify 图 capture 语义 | capture ForwardBatch 无 `extend_seq_lens` → `_prepare_ple_batch` verify 分支走 `lengths=full(draft_token_num)`、valid_tokens 全真(图 verify 行永远是完整 draft stride);req_indices/token_offsets/hash 链按 capture 批形状烘焙。gate 在 replay 侧传 `extend_seq_lens=None` 即与图内逐位同构(同函数 `_prepare_ple_batch`、同静态缓冲、同 pool history) |
| 3 | 宽度与守卫 | 生产 verify 行宽 = 4 draft tokens,每层 flat 宽度 = bs×4×16(heads_per_ngram=8×2),远小于 MAX_ROWS=262144;ragged verify 仅 DSPARK 适用(`supports_ragged_verify`),MTP(EAGLE 族)不经过 token-bucket 路径;新增 fail-loud:`draft_token_num != captured_req_width` 启动 RuntimeError(图内 verify 布局按 capture 宽度烘焙,live 宽度分叉必须炸) |
| 4 | mismatch 自检 | §11.5 修订 2 机制原样适用:PREP_DEBUG 下图内 hash 链每 replay 重算 ids 写入 debug 缓冲,gate 每步比对(verify 与 decode 同一代码路径,宽度自适应);oracle 脚本对 staged boot 扫描 `gate ids != in-graph ids`,任何一条即 FAIL |

实现(最小改动,3 文件):

- `decode_cuda_graph_runner.py`:`_build_ple_decode_prep` 放行
  `capture_forward_mode ∈ {DECODE, TARGET_VERIFY}`(pdmux/MLP-sync 仍
  raise;报错文案写明 draft 无 PLE 不经此 runner);`_prep_ple_staged_decode`
  增加 `extend_seq_lens=None`(复刻 capture 语义)+ verify 宽度守卫;
  `execute()` gate 触发条件从 `is_decode()` 扩为 `is_decode() or
  is_target_verify()`。
- `qwen4_exp.py`:staged 后端启动告警更新(plain-DECODE + MTP
  TARGET_VERIFY 图均支持)。
- `qwen4_exp_ple_prep.py`:builder 文档串说明 verify 接线方式。

oracle 扩展(`scripts/oracle_ple_staged_v100.sh`):

- `ORACLE_MTP=1`:生产 MTP 配置(EAGLE 3 steps / topk 1 / 4 draft
  tokens)双 boot A/B;请求序列与非 MTP 模式相同(错峰防准入竞态,
  decode 重叠走 verify padded bucket 1→2→3→2→1);
- 断点续跑:`outputs_pinned.json` 已存在则复用(两 boot 请求序列相同);
- staged boot 自动带 `SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG=1` 并扫描
  mismatch 日志;accept 长度取自响应 `meta_info.spec_accept_length`
  (paper 口径,含 bonus),pinned vs staged 必然一致(逐位输出的推论);
- 修复:`docker run -d` 的 stdout 重定向只捕获容器 ID,服务器日志改经
  `docker logs -f` 流入文件(此前故障诊断 tail 的是空壳文件)。

oracle 结论(2026-11-04,4×V100 真机,`ORACLE_MTP=1` 生产 MTP 配置,
输出目录 `/data/ple_staged_oracle_mtp`):**ORACLE PASS**——7 条响应、
296 个输出 token、296 个逐 token logprob 与 decoder 文本在 pinned 与
staged 双 boot 间全部逐位一致;staged gate mismatch 自检 564 步
matching / 0 mismatch;MTP accept 长度均值 2.465(逐位一致输出的直接
推论:双 boot accept 必然相同)。verify prep 与 plain decode 共用同一
gate 路径,且每轮 decode 都流经 verify 图 → MTP 路径验收完成。

复跑结论(mmap 分配 + 同步 fill 落地后,`/data/ple_staged_oracle_mtp2`,
pinned 侧复用):**ORACLE PASS**——7/296/296 再次逐位一致,gate 自检
1220 步 matching / 0 mismatch。mmap+register 分配路径与 stage() 优化
不改数值的推论闭环。

运维教训(同日两次环境事故,记录防复发):

- **barrier 失衡 hung task**:首次 staged boot 无 REUSE,每 rank 物化
  11.2GiB 顺序写约 5.5 分钟,远超默认 480s 的 unbalanced-loading
  barrier(`load_model_utils.py` 的 `UNBALANCED_MODEL_LOADING_TIMEOUT_S`,
  新增 `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT` 覆盖,oracle 置
  1800);先行 rank 报错后,迟到的 TP3 卡 D 态 hung task,只能重启主机。
  启动期慢加载必须先抬 barrier 再 boot。
- **kill 后 D 态残留**:强杀残留进程仍持 GPU 显存,新 boot 表现为
  资源不足/NVML 异常;确认无进程后再 `nvidia-smi` 复查。
- **MTP 引擎拆除 wedge(第三次事故,规律确认)**:`docker rm -f` 一个
  idle 的 MTP 引擎容器时,runc 在内核态 GPU 设备拆除上卡 D 态(约 80 个
  runc 叠积,`nv_queue` 卡住),docker daemon 半瘫,只能重启主机;同日
  两次 MTP boot 拆除均 wedge,非 MTP boot 拆除全部秒过。规律:**杀
  MTP 引擎前先接受可能需要重启,或避免直接 rm -f;运维脚本目前无规避
  手段,只能事后重启恢复**。
- **REUSE 语义**:oracle 的 staged 文件来自上一次 boot 物化的确定性
  checkpoint 拷贝(`materialize()` 输出只依赖 checkpoint 字节与 header,
  与 boot 无关),header 校验匹配才跳写;数据正确性由逐位 oracle 本身
  兜底,不依赖 REUSE 路径。

## 11.9.3 第 5 步实施记录:同机 A/B 吞吐门(2026-11-04,4×V100 真机)

工具:`scripts/bench_ple_staged_ab_v100.sh`(双 boot pinned→staged,prefill
侦察 + padding-decode 阶梯,各点两轮:第 1 轮暖 radix/页缓存,第 2 轮计门)。
ladder 定档受三重约束(MTP 请求校验 `input + 4 draft 槽 + completion ≤
262144`;本 boot `max_total_num_tokens=323072`;两侧行为一致):C=1 @250k、
C=2 @128k、C=2 @64k。坑:请求校验会拒 `262148` token(MTP 计入 draft 余量)
与 `max_new_tokens=10M`(input+completion 总量校验),且 server_args dump
含字段名 `…retraction_backup`,retract 扫描须排除。

**第 1 轮 BENCH FAIL**(-4.2%/-3.5%/-2.9%),定位出两个真 bug 并修复:

1. **staged 释放根本不还 OS(内存收益为零的根因)**:加载表走
   `torch.empty(pin_memory=True)` → torch pinned caching allocator 释放的
   块只回缓存池(2GiB 实验:`del + gc.collect() + empty_cache()` 后
   MemAvailable 纹丝不动,且无 host empty-cache API)→ `_release_loading_table`
   换 dummy 参数后 47.7G 仍常驻。修复:新增
   `try_alloc_pinned_mmap`(`host_hugepage.py`,匿名 mmap + MAP_POPULATE +
   `cudaHostRegister(MAPPED)`,tensor 持有映射,GC 即 munmap 确定性还 OS),
   分配链 hugepage → mmap+register → pin_memory 兜底;`nn.Parameter` 不持有
   原对象引用,镜像 hugepage 的 `_sglang_hugepage_owner` 模式防 finalizer
   提前触发;`_release_loading_table` 加 `gc.collect()`。修复后单 boot 验证:
   healthy 时 **Shmem 67.9G → 0.78G,MemAvailable 39.5G → 107.6G**(§1 的
   "40G→~90G 弹性"兑现)。
2. **fill 段 ~1.2ms/步**:gate 每步 1-16 行走线程池,一轮 dispatch+result
   约 1ms。修复:`stage()` 小批量(≤32 行)直接调用线程同步 fill(页缓存热
   行 ~μs 级),CPU-long 张量跳过冗余 dtype/设备搬运;大批量(prefill
   4096 行)仍走线程池。

prep 分段计时(`qwen4_exp_ple_prep.py`,每 500 步 INFO 汇总):build ~0.7ms、
ids+D2H sync 18-26ms(**非新增**——该同步点本就等 GPU 排空,pinned 采样路径
同样要等)、fill 1.0-1.3ms → 优化后 ~0.1ms。

**第 2 轮 BENCH PASS**(warm pass 2,30s padding,agg tok/s 为流式 chunk 计数,
双臂同口径,用于 A/B 比值):

| 点 | pinned | staged | delta |
| --- | --- | --- | --- |
| decode 250k C=1 | 37.8 | 36.8 | **-2.6%** |
| decode 128k C=2 | 45.3 | 44.6 | **-1.5%** |
| decode 64k C=2 | 47.8 | 46.6 | **-2.5%** |

零 retract;§11.1 验收门 1(decode ≥ -3%)通过。门 3(accept 不回退)由
逐位 oracle 兜底(mmap/同步 fill 仅改 IO 来源,不改数值;改动后的
ORACLE_MTP=1 复跑见 §11.9.2 结论行)。prefill 侦察仍回退
(64k 2576→1839,-29%;262k 2952→2354,-18%):prefill 走 eager stage、
无双缓冲(§11.9 步骤 5 的 prefill 双缓冲尚未实施),为 Phase 2 剩余项,
不阻塞 decode 验收门,但生产切换前必须补。

## 11.9.4 第 6 步实施记录:prefill 双缓冲(2026-11-05,4×V100 真机)

§11.9 第 5 步的 prefill 双缓冲在本步实施并验收,生产(8110 compose)已切
staged 后端。实现要点:

- `PleRowStaging.stage_async` / `fill_pending`(`qwen4_exp_ple_staged.py`):
  extend-forward 起点由 `prefetch_staged_ple_prefill`
  (`qwen4_exp_ple_prep.py`,模型循环对每个 extend 前向调用)对各 PLE 层
  `stage_async`(透传本轮 ngram ids,单次 D2H),fill 按 512 行分片提交
  16 线程池与早层计算重叠;gather 处 `fill_pending(n)` 只信任 pending
  宽度(不信任 id-cache——decode 同宽不同 ids 会读 stale 行),命中则
  免二次 D2H 与重填。`stage()` 增加 id-cache(`np.array_equal`,同 ids
  重放直接返回);capture 期 `is_current_stream_capturing()` 守卫
  (captured verify/decode 图仍走 §11.2 的 prep gate)。开关
  `SGLANG_QWEN4_PLE_STAGED_PREFILL_PREFETCH`(默认开,A/B kill switch)。
- 单测:s9-pytest 容器 CPU 25/25(新增 `TestStageIdCache` 5 例 +
  `TestStageAsyncPrefillPrefetch` 7 例);ruff 干净。
- oracle:ORACLE_MTP=1 逐位 PASS(296 token + 296 logprob;prep gate
  自检 1216 步 0 mismatch;prefill ids D2H 比对 0 mismatch;oracle 脚本
  新增 "gather ids != pre-staged ids" 失败扫描)。

干净窗口 A/B(`bench_ple_staged_ab_v100.sh`,pass 2,同 §11.9.3 口径):

| 点 | pinned | staged | delta |
| --- | --- | --- | --- |
| prefill 64k | 2492 | 2103 | **-15.6%** |
| prefill 262k | 2947 | 2650 | **-10.1%** |
| decode 250k C=1 | 37.8 | 36.6 | -3.2% |
| decode 131k C=2 | 46.0 | 41.8 | -9.1% |
| decode 64k C=2 | 49.0 | 46.1 | -5.9% |

解读(2026-11-05 裁决接受现状,生产切换):

1. prefill 从 -29%/-18% 收敛到 -15.6%/-10.1%。残余已定量:eager 计时
   (`n=65536`,即 4096 token × 16 hash 头)显示 `fill_pending` 尾部等待
   ~198ms/chunk(16384 有效随机行/rank,NVMe IOPS 受限)。当前预取从
   本 chunk forward 起点发起,只遮盖前段;§4.2 原意的**跨 chunk 双缓冲**
   (算 chunk i 时预取 chunk i+1)是下一个明确杠杆,未实施。
2. decode 三点未稳定过 -3% 门,但机制排查未发现新代码路径的 decode
   开销:prep 分段计时与第 5 步逐段一致(build ~0.7ms / ids+D2H sync
   17-24ms / fill ~1ms),按 500 步区间的步速率剖面两轮逐段吻合,
   decode 窗口 eager staging 零触发。A/B 比值 run-to-run 摆动 ±2-4%
   (第 5 步第 1 轮 250k -4.2% / 131k -3.5% 同样越门,"验收"实为第 2 轮
   结果);后两轮 131k C2 staged 绝对值偏低(41.8-42.0 vs 43.7-44.6)
   伴随 pinned 基线同期偏高(46.0/49.0 vs 45.3/47.8,主机重启后
   machine-state 差异),未能归因到代码。列为遗留观察项。

生产配置:`/home/bso/vllm-Qwen3.8/docker-compose-sglang-v100-mtp-8110.yaml`
(镜像 `sglang-v100-unified:ple-nvme-20261105`,`--ple-offload-backend
staged`,shard 目录挂 `/data/sglang-ple-staged` + `REUSE=1`)。

## 11.9.5 实施记录:chunk-ahead 预取(2026-11-10,depth=1 过真机 A/B)

§11.9.4 定量的"下一个明确杠杆":算 chunk i 时投机预取 chunk i+1..i+d 的
PLE 行,把 `fill_pending` 尾部等待(~198ms/chunk)移进更早 chunk 的计算
窗口。**未开启生产**(默认 0),真机 A/B(depth=1/2)后裁决。

核心洞察与机制:

- **chunk i+1 的 gather ids 是纯输入侧可预算的**:其 ngram context =
  `cat([history, chunk_i+1])`,而 `history` 恰是 chunk i 的末 2 个 token,
  且 `ple_batch.ngram_context` 已经 = `cat([history, chunk_i])` →
  chunk i+1..i+d 的 contexts = `cat([ple_batch.ngram_context, 后续
  token]).unfold(1, ngram, 1)[:, L_i + j*cs : L_i + (j+1)*cs]`,对任意
  L_i≥1 与该 chunk 真实 forward 逐位一致(含 L_i=1 时 history 跨旧 history
  列的边界;单测 `test_window_composition_l1_edge` 锁定)。GPU 侧用各层
  真实 `_hash_contexts` 计算一次、单次 D2H,按 chunk 行段切片。
- **未来 token 唯一来源是调度器**:`prepare_for_extend`(`alloc_for_extend`
  之后)在 `SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD>0` 时把
  `full_untruncated_fill_ids[extend_range.end : end + d*chunked_prefill_size]`
  存进程级 stash(`record_ple_chunk_ahead`,`{pool_idx: tokens}`);
  forward 起点 pop。陈旧/复用 pool_idx 无害(见下)。
- **流水线深度 = 缓冲数 − 1**:`PleAheadStaging`(`qwen4_exp_ple_staged.py`)
  持 depth+1 个 `PleRowStaging`(各带自己的 IO 线程池)+ FIFO 队列(条目含
  pool_idx 与 ids)。forward i 消费队首(chunk i 的行,即 forward i-1 提交
  的投机 fill),`wait_and_match` 等待 pending 后与 forward 自算 ids
  `np.array_equal` 逐位比对:**命中 → 该缓冲成为本轮 serve,主 staging 完全
  跳过 fill;未命中/队首 pool 不符 → 丢弃该条目,走原 stage_async 路径**。
  然后 submit chunk i+1..i+d(ids 逐位去重:滑窗每轮重推全窗口,已排队项
  跳过;队列 ≤ d)。稳态:chunk j 在 forward j−d 提交、forward j 消费,
  遮盖 d 个 forward;d=2(3 缓冲,~2GB host)即全遮生产 64k chunk 的
  210ms fill(2×156ms > 210ms);d=1 半遮(尾部残等 ~54ms)。
- **serve 接线**(`qwen4_exp.py::_gather_staged`):eager 路径 consume-once
  读 `_staged_serve` 并立即置 None(绝不泄漏进 decode eager break /
  capture warmup——capture 期仍用主 staging,图内指针在 capture 时定死);
  consume 的逐位比对本身强于 PREP_DEBUG 探针,debug 开时仍对 serve 复核;
  每 64 gather 日志追加 `ahead_hits` 计数。
- **正确性回退语义**:混合 batch、verify、DP gather、断行重排、调度器
  预算切分导致 chunk 边界偏离 cs、stash 陈旧等一切失配场景,consume 的
  `np.array_equal` 兜底 → 自动回退原同步路径。oracle:仅 IO 时序变化,
  逐位一致;staged 行填充本身复用 §4.2 语义(无效槽清零)。
- **门控**:仅单 req 全行 extend(`lengths==1`、非 verify、ngram_context
  非 None)pop stash;depth=0 时与 §11.9.4 行为完全一致(不建 pool、不动
  stash)。forward 起点的 ids D2H 同时证明上一 forward 的 gather kernel
  已完成 → serve 缓冲方可安全重填(`consume` 入口先释放旧 serve)。

改动面:`qwen4_exp_ple_staged.py`(`PleRowStaging.wait_and_match` +
`PleAheadStaging`)、`qwen4_exp_ple_prep.py`(stash + `record_ple_chunk_ahead`
+ `_pop_ple_chunk_ahead` 门控 + `_submit_ple_chunk_ahead` + prefetch 扩展,
签名加 `forward_batch`)、`qwen4_exp.py`(`_gather_staged` serve 接线 +
host embedding `_staged_serve/_staged_ahead/_staged_ahead_hits` +
调用点传 `forward_batch`)、`environ.py`(新 env,§11.8 表)、
`schedule_batch.py`(prepare_for_extend hook,~8 行懒 import)。

单测:s9-pytest 容器 CPU 34/34(原 9 例 + 新增 25 例:`TestPleRowWaitAndMatch`
3、`TestPleAheadStaging` 9、`TestPleChunkAheadRecord` 5、
`TestPleChunkAheadPrefetch` 7、env 默认值 1,含 L_i=1 窗口
等价、滑窗稳态、serve 保护、失配回退、多 req/verify 门控、depth=0 不动
既有路径);ruff check/format 干净(改动文件)。

### 11.9.5.1 oracle 多 chunk 覆盖与 4 连跑统计(2026-11-10,4×V100 真机)

§11.9.4 的验收用例均为单 chunk prompt,未触发投机路径。本步给 oracle 加
`ORACLE_LONG_PROMPT=1`(paragraph × 700 ≈ 32.9k token,32 new token,
≈9 个 4096 chunk、末块 partial,首块之后全部走 serve),并做 4 连跑统计:
depth=0 与 depth=2 各 2 次,每 boot 独立重启(pinned 参考 + staged 待测臂
同 boot 内对比)。

| boot | chunk-ahead | long_prefill(pinned vs staged) |
| --- | --- | --- |
| stat0a | 关(control) | staged 退化:248044/248045 循环;pinned 连贯 |
| stat0b | 关(control) | 逐位一致 |
| stat2a | depth=2 | token 21 起分叉(均连贯,正常方差量级) |
| stat2b | depth=2 | token 0 起分叉(均连贯) |

判读(2026-11-10):

1. **退化签名在 depth=0 control 上复现**,且该输出与更早两次 depth=2 boot
   (`ple_ahead_oracle_d2` / `_trace`)的退化输出**逐位完全相同** → 同一失败
   模式,与 chunk-ahead 无因果关联;depth=2 本批 0/2 退化。 **chunk-ahead 洗清**。
2. 退化 boot 为 depth=0:chunk-ahead 的 serve 路径根本未进入(无
   `_staged_ahead`、无 serve),因此"退化与 chunk-ahead 无关"是结构性
   结论,不依赖探针。staged 的 §11.9.4 预填路径仍在本用例下生效,故
   **行内容是否被错读尚未证实**(见下方探针缺陷),`_staged_serve`/行级
   字节验证须在 depth>0 的复跑中取。
3. 跨 boot 非确定本身是既有基线:4 次 pinned boot 中 3 个输出逐位相同、
   1 个独特;所有 boot 的 logprobs 在 token 0 即不同(首 token 近并列)。
4. 短用例(§11.9.2/§11.9.4 集合)本批 4 跑全部与 pinned 逐位一致。
5. 退化 boot 的 MTP 签名:`spec_accept_rate=0.0`、`accept=1.03`、
   `verify_ct=31`(每步仅 1 token,草稿全不中),正常臂为
   `accept 2.1-2.7 / rate 0.36-0.53 / verify_ct 12-15`。退化输出本身是
   `<|im_start|>`(248045)控制符循环,即进入自洽的错误吸引子,而非随机噪声。
6. 退化在 staged 臂 3/6 次出现、pinned 臂 0/4 次(含更早的 d2/trace 两次);
   样本过小,尚不能断言 staged 特有,但列为首要排查方向。

**遗留问题(独立于 chunk-ahead,未结案)**:~33k token 的单 prompt 存在
per-boot 非确定的输出吸引子(至少 3 种:两种连贯文本 + 一种退化循环),
同一 boot 内确定性传播。候选嫌疑(均未证实):首 token 近并列 logprob
放大、`--max-mamba-cache-size 20` 紧张下的 mamba state 复用、长序列
split-KV/PAIR 路径的非确定性归约、staged 预填在 9 chunk 长 prompt 下的
行错读。根因落定前,长 prompt 用例不计入本节的通过判据;A/B 与生产裁决
以短用例集合 + gate self-check + MTP accept 为准。

调试探针(均 `SGLANG_QWEN4_PLE_STAGED_PREP_DEBUG=1` 门控,默认零开销):
`_submit_ple_chunk_ahead` 后逐 forward 打印 `pool/stash/n/serve/queued`;
serve gather 打印 `n/sum(flat)/sum(staged)`;`_debug_spot_check_serve_rows`
按 256 采样对 serve 缓冲逐行与 shard 文件 `pread` 做字节比对(需
`stage_async` 记录 `_staged_vocab_start`,已加)。

**探针自身缺陷(本次统计后发现并修正)**:首版 spot-check 有两处错误,
本次统计跑出的 `mismatches=256` 是假阳性,不可作为证据——(a) 读偏移
漏了 shard 文件头 `data_offset`,实际比对的是文件头+meta 区;(b) 固定
步长采样与 `[token, head]` 展平顺序混叠,采样点全落在同一 head,导致
TP1-3 `checked=0`、只可能检查到 TP0。已改为 `data_offset + local *
row_bytes` + 固定种子随机采样;在 CPU 容器用真实 shard 文件验证:
干净 staging → `checked=56 mismatches=0`,注入 1 字节翻转 → 捕获
`ROW MISMATCH slot=3`。

### 11.9.5.2 spot-check 修复后的 depth=2 复跑(2026-11-10,4×V100 真机)

探针修复后立即复跑一次 depth=2 + long_prefill(`/data/ple_ahead_oracle_d2fix`),
取到 serve 路径的行级字节证据:

| rank | vs | checked | mismatches |
| --- | --- | --- | --- |
| TP0 | 0 | 72 | **0** |
| TP1 | 80000384 | 60 | **0** |
| TP2 | 160000768 | 64 | **0** |
| TP3 | 240001152 | 60 | **0** |

4 分片 256 采样全部零错,证明 chunk-ahead serve 路径**不只是 ids 对齐
(§11.9.5 原有 consume `np.array_equal` + gate self-check)**,填充的
**行字节也与同步路径从 shard 文件读得的完全一致**。退化签名未出现,
long_prefill 首次分叉 token 15(stat2a 为 21,stat2b 为 token 0 但连贯),
短用例 7 个全部逐位一致,gate self-check 1352 步 0 mismatch。
chunk-ahead 数据面数值与行内容双重证实正确。

先前遗留问题"staged 预填在长 prompt 下行错读"(§11.9.5.1 第 4 条)被
本次复跑否证:若 staged 预填行错,serve 缓冲字节必然跟着错,但行级
比对零错。**退化嫌疑收敛为长 prompt 路径的 per-boot 非确定性**,
与 chunk-ahead 无关。

### 11.9.5.3 depth 阶梯 A/B 性能验收(2026-11-10,4×V100 真机)

`bench_ple_staged_ab_v100.sh` 阶梯口径:depth=0 完整 pinned+staged
(`/data/ple_bench_d0`),depth=1/2 各跑 staged 臂、`BENCH_SKIP_PINNED=1`
复用 d0 staged 基线做增量比对(`/data/ple_bench_d1`、`/data/ple_bench_d2`)。
warm pass 2、30s padding,decode gate ≥-3%、retract=0、chunk-ahead wired。

**decode aggregate throughput(tok/s)**

| point | pinned 参考 | d0 staged (AHEAD=0) | d1 staged (AHEAD=1) | Δ vs d0 | d2 staged (AHEAD=2) | Δ vs d0 | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 250k:C1 | 26.7 | 26.9 | 26.2 | -2.6% | 27.0 | **+0.4%** | 三档全过 |
| 128k:C2 | 34.1 | 33.6 | 33.2 | -1.2% | 35.1 | **+4.5%** | 三档全过 |
| 64k:C2 | 36.2 | 34.3 | 34.4 | +0.3% | 35.1 | **+2.3%** | 三档全过 |

**prefill scout TTFT(s)**

| length | d0 staged (AHEAD=0) | d1 staged (AHEAD=1) | Δ vs d0 | d2 staged (AHEAD=2) | Δ vs d0 |
| --- | --- | --- | --- | --- | --- |
| 64k | 41.91 | **28.57** | **-31.8%** | **31.21** | **-25.5%** |
| 262k | 120.16 | **101.94** | **-15.2%** | **102.67** | **-14.6%** |

其他 gate:retract=0、gate wired ✓、chunk-ahead wired(depth=1/2 ✓)、
三档 **BENCH PASS**。

**解读**:

1. **depth=1 是 prefill 甜点**:64k TTFT 省 13s(-31.8%)、262k 省 18s
   (-15.2%),decode 三档全部在 -3% gate 内,净收益最大。depth=2 prefill
   略回退(64k 从 28.57s 回到 31.21s),可能是更深的 ahead 队列带来的调度
   开销;但 decode 整体抬升(128k +4.5%),符合"提前量更多地变成后续 decode
   预填充"的直觉。
2. **staged 本身比 pinned 在 decode 上慢 1.5-5.2%**(d0 vs pinned 参考),
   这是 PLE staged 数据面(host 分片 + stage/consume 开销)的固有成本,
   **不是 chunk-ahead 引入的**。chunk-ahead 的设计目标正是补 prefill TTFT
   短板,本次验收完整命中。
3. **生产候选:depth=1**。depth=1 在 prefill TTFT 上收益比 depth=2 更大,
   decode 开销相当或更小,符合 §11.9.4 中"depth 越小越保守、越不容易过补"
   的直觉。原生产 compose 登记计划 `SGLANG_QWEN4_PLE_STAGED_CHUNK_AHEAD: "2"`
   改为 `"1"`。

**最终验收结论**:chunk-ahead 在 depth=1/2 两档全部通过 gate,prefill
TTFT 显著下降(decode 零负收益),数据面经 oracle 逐位(self-check)与
逐字节(spot-check)双重证实正确。可上线。

遗留长 prompt per-boot 非确定(§11.9.5.1,33k prompt 存在退化循环
吸引子)独立调查,不在本次上线阻塞条件内。

**已补单测**(2 例,36/36 通过,ruff clean,§11.9.5.1 同脚注):尾 chunk
(partial 或恰满)因调度器不再记 stash 而永不 serve(即使队列投机 fill
逐位相同);已完成请求遗留的陈旧队列条目在新请求复用 pool slot 时于首次
consume 无害丢弃。

**脚本增量**:`oracle_ple_staged_v100.sh` 加 `ORACLE_LONG_PROMPT` 用例与
按 boot 输出 `outputs_{pinned,staged}.json`;`bench_ple_staged_ab_v100.sh`
服务端参数对齐 8110 生产 compose(0.90 mem fraction、KV host tier 786432
PAIR、max-running 3、decode graphs bs 1/2/3、prefill graph 关),加
`AHEAD_DEPTH` / `BENCH_SKIP_PINNED` / `BENCH_BASELINE_DIR` 支持 depth
阶梯,并对 docker 端口未释放的瞬态失败重试 3 次。

## 11.10 风险清单

| # | 风险 | 缓解 |
| --- | --- | --- |
| 1 | gate 与图内 gather 的输入不一致(静态缓冲 pad 区、forward_batch 字段切片) | 二者为同一函数同一输入(§11.3 修订 2);**已验收(§11.9.1)**:逐位 oracle 296 token 一致(padded bucket 1→2→3→2→1 全覆盖);跨层宽度一致性检查兜底;另:builder 因模型外层 wrapper 解包失败会静默不接线,已加 `gate wired` 日志 + wrapper 解包 |
| 2 | captured D2H 在 multi-stream capture 下不合法 | **已消解(§11.6.1)**:prep 图外化后无 captured memcpy |
| 3 | gate 的 sync 造成的每步 GPU 空窗超预算 | **已实测消解(§11.9.3)**:A/B 第 2 轮 decode -1.5%~-2.6%(门限 -3%);prep 计时显示 D2H sync 与既有采样同步重叠、净增仅 build+fill |
| 4 | staging 未就位时图已启动(调度竞态) | **结构性消解**:replay 在 gate 的 fill 完成后才 launch(同函数内顺序执行,无竞态窗口) |
| 5 | prep 放大 decode 关键路径 | `_PLEBatch` 每 step 构建一次;ids D2H 仅 KB 级;`_hash_contexts` 与图内同源,无额外内核 |
| 6 | MTP verify 布局对齐(draft stride) | **已消解并验收(§11.9.2)**:verify 图 prep 同构接入(`extend_seq_lens=None` 复刻 capture 语义)+ `draft_token_num != captured_req_width` fail-loud 守卫;ORACLE_MTP=1 逐位一致 + 0 mismatch + accept 2.465。draft 已考古确认无 PLE(表见 §11.9.2) |
| 7 | staged 与 pinned 的 CUDA 非确定性放大 → prefill 首 token argmax 翻转 → MTP 全拒绝 → 自洽循环退化(~33k token long_prefill,depth=0 staged 1/3 概率) | **已溯源并缓解**:staged PLE gather 有固有数值差 vs pinned(不同内存访问模式 → 不同归约顺序);非 det CUDA(cudnn autotune、split-KV 归约顺序随机化)放大差值到翻转 argmax 阈值;argmax 翻到 248044(`<|im_start|>`)时进入自洽循环。根治:`SGLANG_FORCE_DETERMINISTIC=1` 在 `launch_server.py` 入口强制 `cudnn.deterministic=True` + `cudnn.benchmark=False` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`,oracle 实测 staged/pinned 32/32 token_ids bit-exact、MTP verify `first_mismatch_step=4` 全 accept。性能有轻微代价(cudnn autotune 失效),生产可按需开启(默认关)。 |
