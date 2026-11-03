# 活跃请求 KV host 分层(流式 KV)设计文档

状态:Phase 1 已实施并验收(2026-10-31);Phase 2 页入+手动降级已实施,
位级验收通过,文本级 A/B 判据因引擎基线不收敛而作废(见 §8 记录);
Phase 3 最小闭环(512K×1 直映射)已实施,S2 矩阵与 S3 516K 单请求
验收通过(2026-11-01);S4(MTP+CUDA graph)与 512K×2/×4 调度未做
日期:2026-10-30(同日两次修订:§4 以 np=1 实测基准校准;补测 262k 锚点
并重估 §4.2 外推带);2026-10-31 Phase 1 落地,同日 Phase 2 落地;
2026-11-01 Phase 3 落地
范围:Qwen3.8-Flash-Next 线(4×V100-32GB SXM2,TP4,fp16 KV, production 配置)
参考:Strata `/data/develop/Strata` 的 kv_stream 机制(kv_stream.cu / src/core/layer.cpp)

Phase 1 实施记录(2026-10-31):

- 代码:`srt/mem_cache/kv_host_tier.py`(新,组件 1-3)、`srt/mem_cache/qsa_kv_pool.py`
  (两个池类 `__init__` 末尾 attach)、`srt/layers/attention/qwen_sparse_attn_backend.py`
  (metadata 构建 flush / 图 replay flush / decode note_read)、`srt/environ.py`
  (6 个 `SGLANG_V100_KV_*` 变量,默认关)。单测 `/tmp/test_kv_host_tier_unit.py`
  (镜像逐位一致 / 去重语义 / note_read 计数 / 图捕获守卫)。
- 实现:per-token `out_cache_loc` 页号**零 sync 去重**(sort → 边界标记 → cumsum
  → scatter 进 128 页定长 buffer;垃圾槽重复镜像无害),gather 在计算流,
  D2H 在专用 copy stream。两个实测陷阱:按 token gather 搬 ~64× 冗余字节
  (2296→1581 tok/s);`torch.unique`/boolean indexing 每 flush host-sync
  同样打崩 prefill。
- 验收:262k 三组(greedy 128 输出)对同池 tier-OFF 对照:cold prefill
  2238/2259 vs 2287/2301 tok/s(**-2% = 镜像固有 DMA,4 rank 384MB/chunk**),
  输出 89-94 tok/s、warm radix TTFT 0.74s 均无回归;单测 ALL_OK。
- 环境事实(复跑验证时必读):该机 warmup + 261K 首 chunk 激活峰值贴边,
  0.90 + tier 首请求会 OOM(0.88 也不够,0.86 实测通过);PLE 47.7G pinned
  之上 tier 全镜像(GB=32)会宿主 OOM,验证用 `SGLANG_V100_KV_HOST_TIER_GB=2`。
  容器内 sglang 是镜像层非挂载,recreate 后必须重新 docker cp 四个文件
  (compose 注释有清单)。

Phase 2 实施记录(2026-10-31):

- 代码:`kv_host_tier.py` 增稳定镜像 `_host_final`(host 槽 = 页号)、
  `_page_in_kernel`(UVA 原地恢复)、`_degrade_kernel`(closed 页 qNaN 垃圾化)、
  手动降级(`SGLANG_V100_KV_TIER_DEGRADE_FLAG` 一次性 arm)、per-layer
  resident/garbled/protect 位图;`qwen_sparse_attn_backend.py` 在 EXTEND 与
  DRAFT_EXTEND_V2 走 `on_build_extend`、decode/verify 走 `on_build_decode_like`;
  `environ.py` 增 `SGLANG_V100_KV_TIER_PAGEIN` 与 `..._DEGRADE_FLAG`。
- 根因 1(已修,半页 NaN):`_page_in_kernel` 原 grid=(pi_cap, 2) 按
  `program_id(1)` 分 K/V 两个程序,却共享同一个 `resident[li, page]` 位;两个
  程序 check-then-set 竞态,先完成的一半置 resident=1、另一半 early-out,
  留下恰好半页(32 slot = 16384 元素)qNaN。改为单程序同时搬 K 与 V。
- 根因 2(语义修正):extend restore 原为 `use_protect=False`,会用 mirror 覆盖
  本 forward 之前 decode/verify 写入的 live 行(host 副本对这些行已陈旧);
  改 `use_protect=True`。degrade 内的立即全量回填仍用 `use_protect=False`
  (那时 protect 刚清零、无 live 写入)。
- 判据修正(重要):§7 原"同 boot A/B greedy 逐 token 一致"在本引擎**不可达**,
  与 tier 无关。tier-OFF 基线实测:**cold** 首 token logprob 4/4 位一致,但从
  decode 第 2 步起漂 ~1e-4 nat(MTP 关闭同样复现 → 噪声来自 target decode 路径);
  **warm**(前缀缓存)相对 cold 偏 0.002 nat(519 tok)~0.1 nat(4026 tok),且
  4k 下 warm 自身也逐次漂移。64 token greedy 在近并列处必然翻转(实测 cold
  两次在 char11/16 即分歧)。故 Phase 2 用无噪声判据:
  1. 位级:degrade 全池镜像→垃圾化→立即全量回填后,closed 集 62 页 ×
     target 12 层 + draft 1 层 × 4 rank,设备 K/V 与 `_host_final` **逐字节
     相等**、零 NaN(2026-10-31 实测全部为空集,4/4 rank 通过);
  2. 端到端:d0 logprob 由 prefill 产出(基线可复现),可用首 token logprob
     作等价判据;第 2 步之后只能做噪声包络比较,不作硬门。
- 探针清理:临时诊断(`TEMP-DIAG` 共 25 处,含 memory_pool 的 set_kv_buffer
  NaN 检测与 backend 的 `_tier_nan_census`)全部删除;删后重注入重启,短 prompt
  A/B 复跑确认降级路径只输出 `kv_host_tier degrade #N`,无异常。
- 261k 三臂(去探针后实测,261,886 prompt tokens / greedy 128):
  A cold 117.4s TTFT、B(降级)118.7s、C warm(cached 261,824)0.51s;
  B 触发的 degrade 在 261k 下全量镜像正常(4/4 rank,无 NaN/异常)。greedy
  文本在 B/C 第 2 个字符即分歧,但这是**基线噪声**:同 boot 两次 cold prefill
  的首 token logprob 为 -1.8299 / -1.8586(**差 0.029 nat**),而降级臂 A↔B
  为 -1.86622 / -1.86946(差 0.003 nat),小于 cold/cold 控制噪声;261k 的
  prefill 本身在该机上就不收敛,文本判据在长上下文同样不可用。
- 512K 阶梯推迟到 Phase 3:当前 `--context-length 262144`、0.86 时池
  331,392 tokens/rank(0.90 约 383K),单请求 512K 在 Phase 2(无逐出)下
  物理不可行;§7 的 4×512K 门属于 Phase 3 调度/逐出,Phase 2 退出条件改为
  位一致性 + 261k 噪声包络。
- production compose 恢复(§8 收尾):tier 四行恢复注释、`--mem-fraction-static`
  回 0.90。

Phase 3 实施记录(2026-11-01):

- 代码:`kv_host_tier.py` 重写为 §10 直映射语义(per-layer `tags/protect/hvalid`、
  `_p3_write_restore_kernel` restore-then-claim、`_p3_mirror_kernel` build 开头
  镜像上一 forward 写页、extend 读 `_p3_gather_kernel`(tag 命中读 device,未命中
  读 pinned host)、decode 读 `page_in_selected_phase3` + `phys_token_slot_table`);
  `qsa_kv_pool.py` 供物理槽视图,`qwen_sparse_attn_backend.py` 接三条路径。
  调试门控 `QSA_P3_FORCE_HOST`(强制镜像页读 host)/`QSA_P3_CHECK` 与计数器
  mirror_pages/write_restores/pagein_restores/collisions/host_gather_rows/
  mirror_skips/gather_unvalid/host_dev_mismatch。另修复 `qsa/qsa_indexer.py`
  未完成迁移:调用方传 `out_dtype` 而 srt shim 旧签名无该参数,4 TP rank 启动即
  `TypeError`;两个导入改指 `sglang.kernels.ops.attention.qsa_indexer`(kernels/ops
  版兼容 fp8 e4m3 out_dtype)。
- 启动配置(512K 必须显式 YaRN:checkpoint `rope_type=default` +
  `original_max_position_embeddings=262144`,超出即需扩展,见本记录与
  `scripts/serve_qwen38_flash_next_nvfp4_v100.sh` 头注):`--context-length
  524288 --max-total-tokens 540672` + `--json-model-override-args '{"rope_parameters":
  {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144},
  "text_config": {"rope_parameters": {...同...}, "max_position_embeddings": 524288}}'
  (model_override_args 对 PretrainedConfig 内层走 dict.update 合并,见
  `srt/utils/hf_transformers/config.py`)。
- S2 正确性矩阵(同 boot 一致代码树,nonce 各异防 radix 复用,FORCE_HOST=1/0
  双臂):60K@5%、85K@5%、85K@80%(needle 落卷绕区 lp≈1063→pp≈39)、
  100K@80%、120K@80% 全部逐字命中。两个判据陷阱:/generate 流式返回累计
  text(取末 chunk 而非拼接);思考模型 `<think>` 吃 max_new_tokens 预算
  (512 会截断答案,用 1024)。小配置累计计数器证实卷绕全链路活跃:
  mirror_pages=95112、write_restores=16640、pagein_restores=33941、
  collisions=2348、host_gather_rows=4769280、mirror_skips/gather_unvalid/
  host_dev_mismatch=0。
- S3 516K 实测(2026-11-01,PHYS=327680/logical=540672/YaRN×2,冷启动单请求):
  needle 探针 501,700 目标 ×41.14 tok/doc → prompt 516K;needle doc 位于
  ~412.8K token(lp≈6449→pp=1329 卷绕区),greedy `needle_hit` 逐字命中
  `NEBULA-5X7Q`;末批 full token usage 0.95(≈8064/8448 逻辑页 ≈516K);
  chunked prefill 4096 tok/批约 3s/批,TTFT ≈5 分钟;计数器(steps=128):
  host_gather_rows=72.8M(卷绕区未驻留页真实走 host 镜像读)、
  mirror_pages=179952、write_restores/collisions=0——mirror-keeps-up 设计下
  claim 时宿主副本恒已最新,恢复路径仅镜像滞后时才触发,S2 小配置矩阵已单独
  证实其活跃性。
- 内存实测(修正 §10.1 预估):大配置稳态 MemAvailable 仅 **7.9-8.2 GB**——
  Shmem 101.3 GB(PLE offload ~51G + 宿主镜像 4×6.19 GiB pinned + torch/NCCL
  共享池 ~25G)+ AnonPages ~11G;镜像实际 6.19 GiB/rank(预估 6.0)。prefill
  全程 MemFree 一度 2.9 GB 但 SwapFree 117G 无风暴,4 scheduler RSS 各
  26.1 GB,GPU 31/32.8 GB。复跑必读:启动前确认无其他引擎(Shmem 须清零,
  `docker compose down` 后应回到 ~122 GB);516K prefill 期间无新大块 host
  分配,8 GB 余量可行,但本机已无再容纳第二引擎的空间。
- 事故记录:权重分片 `model-fp8-mtp-ple.safetensors` 宿主文件损坏(同目录其余
  10 分片正常;read→EINVAL/lseek→ESPIPE/mmap→ENODEV,filefrag 元数据完好)致
  权重加载 os error 19、容器崩溃。wget 直连 resolve URL 重下(huggingface_hub
  1.29 对 >50GB 文件强制 hf_xet,本机 `HF_HUB_DISABLE_XET=1`),sha256 精确
  匹配。坏文件保留同目录 `.corrupt` 后缀,待有 root 时取证(dmesg/smartctl/fsck)。

## 1. 背景与目标

现网 KV 全量驻留 GPU 池(fp16,~694k tokens/rank 上限,production compose 473,792)。
扩上下文(>262k)或扩并发(>3 路)时 KV 池不足。e5m2 KV 因 MTP accept 损失被否决,
fp16 是决策基准。目标:**活跃请求的冷 KV 页分放到 host RAM(fp16,无量化),
GPU 只留热窗,decode 期按需页入**,使 512K/1M 长上下文在 1-4 并发下可服务。

与 Strata 的关键差异:Strata 是单会话引擎,prefill 可全量 stage;本设计面向多请求
连续批调度,.prefill 路径不改(仍要求 prefix 瞬态全驻或走流式 prefill),
只对 decode/verify 期做按需页入。

成立前提(本 fork 已满足):Qwen3.8-Flash-Next = 36 GDN(无 KV)+ 12 稀疏注意力层,
decode 只读 top-k 选中块;tilelang_fa_v100 后端已有"选中页 → dense workspace gather"
架构与 page_table 间接寻址,**attention kernel 无需修改**。

## 2. 容量矩阵(512K / 1M × 1 / 2 / 4 并发)

单位约定:KV 计量用每 rank 每 token 12.0 KiB(理论)/ 13.4 KiB(实测,
0.64 GiB ↔ 50,048 tokens,含页取整);"物理总量"= 4 rank 之和(含 KV 头 2× 复制:
2 KV 头 TP4 → 每 rank 1 头,双份冗余)。

host RAM 预算:120G − 系统/tokenizer ~10-12G − hicache host 层(缩至 4-16G)
≈ **92-106 GiB** 可用于 KV 活跃层。NVMe 冷层仅适用于强局部性 + 可预取场景
(随机 top-k 直接落盘不可接受,file tier 实测读回 344 tok/s)。

| 场景 | 物理 KV 总量 | GPU 常驻窗(48k/请求) | host 需求 | 判定 |
| --- | --- | --- | --- | --- |
| 512K × 1 | 24-27 GiB | 0.6 GiB/rank | ~22-25 GiB | 容易 |
| 512K × 2 | 48-54 GiB | 1.2 GiB/rank | ~44-50 GiB | 容易 |
| 512K × 4 | 96-108 GiB | 2.5 GiB/rank | ~88-100 GiB | **RAM 贴边**:缩窗至 24k + hicache 4G 后可行,零余量 |
| 1M × 1 | 48-54 GiB | 0.6 GiB/rank | ~44-50 GiB | 容易;prefill 瞬态 12-13.5 GiB/rank 贴顶 |
| 1M × 2 | 96-108 GiB | 1.2 GiB/rank | ~88-100 GiB | RAM 贴边(同 512K×4) |
| 1M × 4 | 192-216 GiB | 2.5 GiB/rank | ~182-206 GiB | **fp16+复制下不可行**;去复制后 96-108 → 贴边(见 §6 Plan B) |

去复制杠杆(Plan B,减半):512K×4 → 48-54 GiB(从容);1M×2 → 48-54(从容);
1M×4 → 96-108(贴边,等价于今天的 512K×4)。

结论:**512K×4 / 1M×2 是无 Plan B 时的容量上限带;1M×4 必须先做去复制,
且仍处贴边带,叠加局部性验证与 NVMe 预取才可承诺。**

前置门槛(非显存):原生上下文 262,144;512K/1M 需 YaRN factor 2/4
(rope_scaling 配置覆盖)+ 质量验证(§7)。BFLA 选择器为 training-free
均值池化点积,无习得范围限制,外推风险低于习得 indexer,但仍未验证。

## 3. 架构设计

### 3.1 现状与差距

现状:GPU 池(paged fp16)+ hicache host 层 + 磁盘文件层(write_back)。
hicache 只服务"已逐出的 radix prefix"整段回填;活跃请求 KV 100% 驻留 GPU 池,
decode 期无按需拉取。差距只有这一条。

### 3.2 组件(fork 拥有的代码区)

| # | 组件 | 内容 | 复用 |
| --- | --- | --- | --- |
| 1 | 活跃层 host 池 | 每卡 pinned fp16,默认 32G(约 240 万 token/rank 容量);`SGLANG_NUMA_BIND_V2=0` 交错不变 | hicache page_first 布局与传输机制 |
| 2 | 驻留管理器 | 页粒度(64-256 token)驻留位图 + Clock 二次机会;选中页命中置位,miss 页入;冷页降级;radix 共享页 refcount | Strata resolve 的 Clock 思路,Python + 小 CUDA kernel |
| 3 | 池化代表缓存(新增,必做) | BFLA 从 KV 本身均值池化打分;KV 上了 host 后打分须读常驻的 64-token 组代表:16k 组 × 256 × 2B × 12 层 ≈ **100 MB/请求/rank** | Strata idx_pooled 同构 |
| 4 | decode 页入钩子 | attention 前:本步选中页 → 查驻留表 → 非驻页从 pinned RAM 批量拷入 scratch 环形缓冲(128-256 MB)→ 索引重映射到 scratch 槽;decode/verify kernel 零改动 | Strata 批量 miss 拷贝 kernel 思路;`cache_move.py` 已有 GPU↔CPU 整层拷贝 |
| 5 | 调度准入改造 | 按"常驻工作集 + 单个 prefill 瞬态"准入;miss 字节/步超阈值自动回退现行为 | 现有 retract 逻辑 |

拦截点(均已在后端存在,不新造):`_paged_adapter.py` 的
`_get_d256_dense_workspace` gather 路径与 `grouped_decode_forward` page_table 路径。

### 3.3 prefill 两方案

- 方案 a(默认,512K 适用):prefill 要求 prefix 瞬态全驻(512K = 6.0-6.8 GiB/rank,
  可放;1M = 12-13.5 GiB/rank 贴顶)。大 prefill 串行准入。PCIe 增量 ≈ 0。
- 方案 b(1M 或瞬态放不下时):流式 prefill,chunk 4096 的 prefix 注意力经页入路径
  供页,带一层预取(下一 chunk 的页提前拉)。总 PCIe = O(n²/2/page):
  1M ≈ 1.5 TB ≈ 93 s 纯传输,可与其他 chunk 计算重叠,净增 ≤10-20%。

### 3.4 MTP verify

verify 一步 4 个 draft 位置 → 选择×4,scratch 与页入路径同构覆盖。
verify replay 路径正被并行会话改动(gdn_backend verify 索引切片),
动工前必须对齐,避免索引语义二义。

## 4. 性能预估

基线:

- **np=1(2026-10-30 实测,OpenAI 兼容接口,production 配置,输出 128 tok)**:
  prefill 峰值 3762 tok/s @4k,131k 处 2535 tok/s(仅 -33%,衰减远缓于
  二次方最坏预期);输出 96-127 tok/s,131k 处 104 tok/s,与 README 验收
  "decode ~100 tok/s/流"一致。ITL 平均列(23-24.5 ms)与输出速度口径
  不一致,以输出速度为准(np=1 步长 ~9.6-10.4 ms,MTP 开)。
- np=2/4(既有实测,mrr=4 暖机后):4way 暖聚合 130/168/162/195/172
  @8k-96k;2way 生产配置暖聚合 68-133。
- >131k 无实测,以下为外推带(§4.2 给出拟合模型)。

decode 增量模型(每步):池化代表打分(512K:+0.5-1.5 ms;1M:+1-3 ms;
由 np=1 实测的 ITL 增长趋势 2k→131k ≈ +1.5 ms/129k tokens 外推,较初版
估计下调)+ miss 页入(稳态 +0.1-0.6 ms;PCIe 16 GB/s,12 层 × 3-12 页
× 64 KiB)。打分项是长上下文固有成本(与流式无关);流式净增只有 miss 项,
稳态 ≤2-5%。

### 4.1 decode 预估(tok/s;括号内为聚合)

| 场景 | np=1 | np=2 | np=4 |
| --- | --- | --- | --- |
| 512K × 1 | 73-82 | — | — |
| 512K × 2 | — | 46-50 (92-100) | — |
| 512K × 4 | — | — | 34-36 (138-146) |
| 1M × 1 | 63-75 | — | — |
| 1M × 2 | — | 42-48 (85-95) | — |
| 1M × 4 | — | — | 32-35 (130-140) |

np=1 列按实测趋势校准:262k 锚点输出 85-93 tok/s(2026-10-30 实测,
见 §4.2),131k 104 tok/s → -14%/倍增,按此外推 512K/1M。
np=2/4 列仍基于既有 mrr=4 聚合外推,未经同口径校准。

修正项:MTP accept 若从 2.1 降至 1.6(YaRN 长上下文风险),以上 ×0.76
(1M×4 → ~100-107)。冷启动/最坏(选择均匀散布、局部性差):miss 率 30%+
时额外 -10% 以上,由 miss 字节阈值触发回退。

置信度:np=1 中;np=2/4 低-中。miss 率需实测校准;表中数字用于容量规划,
不作为承诺。

### 4.2 prefill 预估(262k 锚点实测 + 外推带)

实测锚点(2026-10-30,production 容器 sglang-v100-qwen38-8110,np=1,
prompt 261,886 tokens(服务端 usage 口径),两次冷测 ±0.2%,flush 后冷测):

| 项 | 262K 实测 |
| --- | --- |
| prefill 速率 | **2296 tok/s**(2293.9 / 2298.8) |
| TTFT | **114 s** |
| 输出 | 85-93 tok/s |
| warm radix TTFT | 0.68 s(同 prompt 二次请求,262k prefix 全命中) |

全段曲线(用户既有 0.5k-131k 数据 + 本次 262k):4k 峰值 3762 → 131k 2535
→ 262k 2296;131k→262k 仅 -9.5%,衰减继续亚线性(若按稠密注意力线性项
模型 1/r=a+b·n 从 65k/131k 外推,262k 应为 ~1320 tok/s,实测快 74%)。
外推取带(1/r 对 n 的幂 p∈[0,1] 两端界定):

| 项 | 512K | 1M |
| --- | --- | --- |
| 有效 prefill 速率 | 670-2200(中位 ~1030) | 340-2100(中位 ~590) |
| TTFT | 4-13 min(中位 ~8.5) | 8-51 min(中位 ~30) |
| 流式 KV 增量 | 方案 a ≈ 0(瞬态 6.0-6.8 GiB/rank;262k 单请求入 426k 池实测可行 ✓) | 方案 b +5-15%(可重叠) |

取带原因:131k→262k 段仍缓,支持亚线性外推;但 1/r 对 n 的段斜率末段回升
(2.08→3.14 ×1e-9),二次项正在抬头,幂次在 262k 之外不可判定。
512K 实测点需 Phase 3 容量(Phase 2 池 331,392 tokens/rank,无逐出)。
TTFT 分钟级仍是 1M 场景第一体验约束(初版 39-70 min 估计落在本带保守端)。

## 5. 环境变量(遵循 environ.py 约定,fork 既有 SGLANG_V100_* 前缀)

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| SGLANG_V100_KV_HOST_TIER | 0 | 活跃层总开关(默认关,现网行为不变) |
| SGLANG_V100_KV_HOST_TIER_GB | 32 | 每 rank 活跃层 host 池大小 |
| SGLANG_V100_KV_HOST_PAGE_TOKENS | 64 | host 页粒度(64/128/256,与 BFLA 组对齐) |
| SGLANG_V100_KV_HOT_WINDOW_TOKENS | 49152 | 每请求保底热窗 |
| SGLANG_V100_KV_MISS_BYTE_LIMIT | 8388608 | 每步 miss 字节上限,超限回退全驻语义 |
| SGLANG_V100_KV_STREAM_PREFILL | 0 | 方案 b 开关(1M 场景) |
| SGLANG_V100_KV_TIER_PAGEIN | 0 | Phase 2 页入路径(需 HOST_TIER=1;稳镜像 + UVA 原地恢复,零 host sync、图安全) |
| SGLANG_V100_KV_TIER_DEGRADE_FLAG | "" | 手动降级测试钩子:文件存在即在下次 decode-like build 一次性全量镜像 + closed 页 qNaN 垃圾化 |
| SGLANG_V100_KV_TIER_DISABLE_EXTEND | 0 | 关闭 Phase 3 逻辑容量扩展:保持 profiled 设备容量,释放字节留给激活(推荐开启;见 §10 头部修订) |

上线时同步 serve 脚本注释与 README 表格(fork 纪律)。

## 6. Plan B 杠杆(按启用成本排序)

1. 窗口/预算参数化(已含):窗 48k→24k,hicache 16G→4G。
2. 去复制(NVLink P2P):2 KV 头 TP4 目前每头双份(rank0/2 共享头0)。改为
   每头单份,缺头 rank 经 NVLink P2P 读邻卡。物理 KV 减半,1M×4 从不可行
   变贴边。侵入 attention TP 逻辑,仅在 §2 矩阵判定不足时启用。
3. NVMe 冷层 + 选择预取:利用选择的时间相关性(步 t 的选中集预测 t+1)。
   仅在局部性实测通过后作为 1M×4 的补充,不作为随机访问的兜底。

## 7. 验证矩阵与验收门

质量(全部 fp16,链路无量化,目标零精度损失):

1. 纯管道等价(Phase 1):开关开、降级禁用 → smoke + 验收命令数字不回退。
2. 位一致性(Phase 2):同 boot A/B,强制全降级,设备 K/V 与 host 镜像逐字节
   相等(纯搬运语义;跨 boot hash 不可比,不用)。**不用** greedy 文本逐 token
   一致——引擎 decode 步本身漂 ~1e-4 nat(§8 Phase 2 记录),文本判据不可达。
3. YaRN 质量门(1M 前置):passkey/NIAH @512K、1M 多档,perplexity 对齐;
   MTP accept @512K/1M 不低于 2.1×0.85。
4. 局部性实测:1×1M(显存无压力)测 miss 率-热层大小曲线,验证
   "热集 ≤50% 覆盖 ≥95% 选择";不成立则 512K×4/1M×2/1M×4 转 Plan B。

性能(全部 4×V100 实测,暖机后;JIT 懒编译首轮剔除):

- 阶梯:512K、768K、1M × np=1/2/4,flush 后 COLD+WARM(沿用
  /tmp/bench_ctx_ladder_distinct.py 口径),记录 prefill 速率、decode
  单流/聚合、miss 率、页入字节/步。
- 验收门:4×512K 聚合 ≥120 tok/s 且 ≥同配置 4×64k 基线的 60%;
  1M 档数字仅记录不设门(首版)。
- 回归门:Qwen3.8 NVFP4 现网验收数字(prefill ~3000、decode ~100/流、
  accept ~2.1)在任何 Phase 合并前复测不回退。

## 8. 分阶段实施

| Phase | 内容 | 行为变化 | 退出条件 |
| --- | --- | --- | --- |
| 1 纯管道 | 组件 1/2/3(降级禁用,全驻留)+ env 开关 | 零(仅常驻 ~100 MB 代表 + 管理开销) | smoke + 验收不回退 |
| 2 页入 | 组件 4 + 手动降级策略;target-only 先行,后接 MTP | 手动配置时生效 | 位一致性 + 261k 噪声包络(512K 需 Phase 3 容量,见 §8 记录) |
| 3 调度 | 组件 5 准入按常驻工作集;512K×2/×4 默认策略 | 生产可开关 | 512K×4 实测达 §7 门 |
| 4 1M | YaRN ×4 + 方案 b 流式 prefill + 1M 阶梯 | 新场景 | §7 质量门 + 阶梯记录 |

## 9. 风险与未决问题

- verify replay 索引语义:与并行会话(gdn_backend)改动协调,PR6 池越界
  同类区域重点测。
- 索引重映射与 radix/retract 释放路径的一致性(三种槽位并存)。
- 4×1M 在 fp16+复制下无解,去复制改造的工程量与 attention TP 正确性风险。
- host pinned 内存上限(cgroup/numatune)需在目标机实测;
  `SGLANG_NUMA_BIND_V2=0` 为前提。
- 262k 锚点已实测(2026-10-30,prefill 2296 tok/s / 输出 85-93 tok/s /
  warm radix 0.68 s);decode 按 -14%/倍增外推,512K/1M prefill 仍为带估计,
  Phase 2 阶梯补 512K 点收窄并回填 §4 表格。运行时确认 production
  page_size=64、池 426,240 tokens。

## 10. Phase 3 实施设计(最小闭环,512K×1)

目标:在 4×V100-32GB 上让 516K prompt + 128 输出单请求跑通。挤基线实测证明
(0.945 + 正常配置)池 518,976 tokens 本身足够放下 516,128 tokens,**OOM 来自
prefill 激活没有余量**(首 4K chunk `F.silu(conv_output)` 80 MiB 申请失败,
free 12 MiB)。因此 Phase 3 最小闭环不是"扩逻辑容量",而是**缩设备池、把差额
换成 host 页**,同时让播散的 top-k 读仍能取到数据。

**逻辑容量扩展(2026-11-03 修订为默认禁用)**:`_tier_extend_logical_capacity`
曾把 `--max-total-tokens` 超出 GPU profile 的部分重新充入逻辑预算(516,352 →
540,672),但 s7/s8 门测证明它在紧池上饿死 prefill 激活——0.93 fraction 的
255K 对照在扩展抬容量后死于 80 MiB `F.silu` 申请,同配置保持 profiled 容量则
存活且 needle 全过。现由 `SGLANG_V100_KV_TIER_DISABLE_EXTEND=1` 关闭扩展
(推荐部署形态):逻辑容量保持 profiled 值,释放的 PHYS 字节留在激活;
`PHYS` 直接按实际服务的最长上下文取整到页(255K → 327,680,实测 255K×1
needle 3/3、`host_dev_mismatch=0`)。注意卷绕约束:上下文必须 ≤ PHYS,
否则 `lp % P` 槽位复用会使召回随超出比单调劣化(150K/156K 边界全过,
170K 起 1/2,255K@196K PHYS 0/8;PHYS=393,216 无卷绕 3/3 同 nonce 转好)。

### 10.1 容量切分

| 项 | 值(每 rank) | 说明 |
| --- | --- | --- |
| 逻辑 token 预算 | 540,672(实配,≈524K context + 输出/对齐余量;设计预估 518,976) | 调度/allocator/`req_to_token` 全部不变 |
| 设备 full-KV 池 | `SGLANG_V100_KV_TIER_PHYS_TOKENS`(默认 327,680) | 4.45 GiB 含压缩 K |
| host 逻辑镜像 | 逻辑页全覆盖 6.19 GiB pinned(实测) | 4 rank ≈ 24.8 GiB(本机 125G 可容) |
| 释放的显存 | ~2.3 GiB/rank | 留给 prefill 激活(0.86 实测所需) |

**压缩 indexer K 保持逻辑容量**(bf16 786 MB/rank)。这是本版最重要的取舍:
512K×1 下它只占 0.8 GiB,不值得做"池化代表 + 二段式选择"(组件 5);组件 5
推迟到 512K×2/×4 或 1M 场景。

### 10.2 页映射:直接映射,无空闲链表

```
logical_pages L = ceil((logical_size + page) / page)      # 8193 @512K
physical_pages P = PHYS_TOKENS / page                      # 5120 @320K
phys_page(lp) = lp % P                                     # 纯函数,静态
phys_slot(s)  = (s // page % P) * page + s % page
```

- 映射是**纯 modulo**,不需要 free list、victim 选择或 host-sync;写入
  `lp` 隐式覆盖 `lp%P` 上原有页,而该页此前每次写后都已镜像到 host
  (flush 保证),所以覆盖永不丢数据。
- 设备侧状态:`tags[P]`(int32,phys→当前 lp,初始化 -1)、`protect[P]`
  (u8,本 forward 写过,page-in 必须跳过)、`host_logical[2, layer, L,
  page, H, D]` pinned fp16。
- 直接映射对"前缀 + 最近热窗"访问是友好的:512K 逻辑跨 320K 物理时,
  只有前 ~200K 与后 ~4K 别名;冷页很少被选中,实测 miss 率是本版核心指标。

### 10.3 三条路径

1. **写**:build 时对 `forward_batch.out_cache_loc`(allocator 给的页对齐
   逻辑 token id,见 `allocator/paged.py` L160-181)先做
   `allocate_write_pages`:`tags[lp%P] != lp` 的页先用 host→phys 全页恢复
   (?**关键**:跨 forward 的半页写入不能丢),再置 tag、置 protect;随后
   `set_kv_buffer` 用物理表。flush 用 Triton kernel 按 `lp` 写
   `host_logical`(kernel 直接写 pinned host,UVA;Phase 2 的 page-in 已验证
   该方向可行)。
2. **decode 读**:metadata build 产逻辑 `token_slot_table`
   (`req_to_token[row, pos]` 的 dense 副本,Phase 2 已有),同时产
   `phys_slot_table = modulo(token_slot_table)`(纯 device 运算、图安全);
   每层 attention 前 `page_in_selected(逻辑页)` 按 tag 检查、miss 时
   host→phys 整页恢复(protect 跳过);decode kernel 改收 `phys_slot_table`
   (现收 `req_to_token_pool.req_to_token`,`qwen_sparse_attn_backend.py`
   L2489-2520),kernel 零改动。`token_slot_table` 仍保留逻辑值给
   `_qsa_write_plan`(压缩槽 = 逻辑 full_slot // ratio)。
3. **extend 读**:同表;FA2 回退的 Triton `_qsa_fused_kv_prepare` 收
   phys 表即可。prefill 的读集合是 top-k 并集,streaming 成本
   O(n²/2/page)≈ 25-40 s @512K(8 分钟 TTFT 的 5%),可接受;不进 §3.3 方案 b
   的预取,先测实际 miss 字节。

### 10.4 不变量(实现与验收都要盯)

- `req_to_token`、radix、hicache、allocator 里**永远只有逻辑 id**;
  物理化只发生在 kernel 边界。
- 一个逻辑页在设备上最多出现一次(tag 唯一);`tags` 与缓冲内容的一致性
  只由两个 kernel 维护:`mirror`(写 host)与 `page_in`(host→phys)。
- protect 页的 host 副本必然陈旧,任何 restore 都不得覆盖它;跨 forward
  半页写入靠 10.3-1 的先恢复后写保证。
- 关闭开关(`SGLANG_V100_KV_TIER_PHYS_TOKENS=0`)时行为与现网逐字节一致。

### 10.5 里程碑

| Stage | 内容 | 判据 |
| --- | --- | --- |
| S1 | 池物理化 + 逻辑压缩 K + 写路径 + 全量 host 镜像(页入关闭) | 261k 三臂数字不回退;内存统计符合 10.1 |
| S2 | decode/eager 选中页页入 + phys 表 | 261k/4k 输出与前值同噪声包络;miss 计数合理 |
| S3 | 516K×1 冷 prefill(`/tmp/phase3_512k_once.py`) | 成功出 128 tokens;记录 TTFT/miss/页入字节 |
| S4 | MTP + CUDA graph(eager MTP 路径本 fork 缺失,先修 `_speculative_max_row_length`) | accept ~2.1;图回放页入位一致(已达成,见 10.7) |

### 10.6 已知限制与回退

- 本版仍是**单 kernel 直映射**,不处理 512K×2/×4 的准入工作集(§3.2 组件 5)。
- **512K×2/×4 宿主内存预估(2026-11-01,×1 实测外推)**:宿主镜像按逻辑页
  全量 pinned 分配(~12.2 KB/token/rank),而逻辑池为全请求共享,×2/×4 首先
  要把 `max-total-tokens` 提到 ~1.07M/~2.15M,镜像随之 53.2/106.3 GB。
  ×1 实测固定开销 ≈97.6 GB(PLE 51 + torch/NCCL 共享池 ~26 + scheduler
  anon ~11 + OS ~8),MemTotal 131.9 GB → **×2 总占用 ~150.8 GB(缺 ~19 GB),
  ×4 ~203.9 GB(缺 ~72 GB),均不可行**;镜像 pinned 不可换出,swap(117G)
  救不了。出路:×2 需把 PLE 下沉 NVMe(省 51 GB → 余 ~32 GB);×4 需 PLE
  下沉 + §6 杠杆 2 去复制(镜像减半 → ~126.6 GB,余 ~5 GB 贴边),再叠加
  NVMe KV 冷层(杠杆 3)才稳。无论内存还是语义,×2/×4 都要先做组件 5 准入,
  并恢复 CUDA graph/MTP(eager 下 ×4 聚合 decode 仅 ~20-24 tok/s,离 §7 门
  ≥120 tok/s 差 5 倍)。次级约束:设备侧压缩 indexer K 保持逻辑容量
  (×4 时 3.15 GiB/rank,对 476 MiB 低水位复核);`--max-mamba-cache-size 20`
  在 ×4 下需按并发复核(mamba extra_buffer 要留并发槽)。
- radix 保留(逻辑 id),但 **hicache 在 Phase 3 档位关闭**:它的
  write-back 假设设备槽=逻辑槽;恢复前缀要靠 read 路径页入而非 hicache。
- MTP 曾在本版关闭(`--disable-cuda-graph` + target-only);S4 已接入
  MTP + CUDA graph(见 10.7),该 fork 缺口已闭合。
- 回退:删 `SGLANG_V100_KV_TIER_PHYS_TOKENS` 即回到 Phase 2/生产语义。

### 10.7 S4 实测记录(2026-11-01,4×V100-32GB,/tmp/phase3-repro-small.yaml)

Phase 3 host KV tier 之上接入 MTP + CUDA graph,三图(target verify、draft
decode、draft extend)全部捕获成功:

- **图捕获**:target verify 130.2s / 0.11 GB、draft decode 3.6s / 0.04 GB、
  draft extend 1.3s / 0.0 GB(bs=1,backend=full)。tier3 图缓冲为持久全宽表
  (`_ensure_tier3_graph_buffers`,宽度 = `req_to_token.shape[1]`),回放期经
  `_tier3_refresh_graph_tables` 入队 `on_build_phase3` + `map_slot_table`
  刷新(流序在 `launch_graph_metadata` 回写 `row_req_pool_indices` 之后),
  满足捕获内核烘焙表宽/行数的契约。
- **正确性**:4K sanity accept len 2.88(cuda graph: True);516K needle
  (`/tmp/phase3_512k_once.py 516000 needle s4run2`,实测 518,080 tokens、
  usage 0.97)命中,decode 全程 cuda graph: True,accept len 3.60-3.85
  (≥ §7 门 1.79);phase3 计数器 steps=192、host_gather_rows=10,321,920、
  **host_dev_mismatch=0**、collisions=0——图回放页入位一致成立。
- **性能**:冷 516K TTFT ~308s(平均 ~1.7K tok/s;池占用 >90% 时降至
  ~1.0K tok/s);decode 98-105 tok/s/流 @518K(vs 现网基线 ~100,无回退)。
- **MTP draft 隔离**:draft 池按预期跳过 host tier(启动 warning 三 rank
  可见),draft 图正常捕获,`flush` Phase 3 no-op;eager spec_paged 路径
  的 Phase 3 raise 移除,与图路径共用 `on_build_phase3`+`map_slot_table`。
- **显存**:稳态 ~29.0-29.6 GB/卡;516K prefill+decode 峰值 32,152 MiB
  (GPU0,余 ~0.6 GiB)——单请求贴边,并发扩大必须先做 §3.2 组件 5 准入。
