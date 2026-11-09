# PREFILL 杠杆立项书(L2 / L3)

立项日期:2026-11-07。状态:**L2-a spike + L2-b 产线完成(2026-11-08,实测 e2e +13~18%);受控三臂 A/B 于 2026-11-09 完成(§8.1:分叉率不超噪声地板,无系统性质量退化);但 2026-11-09 WorkBuddy agentic 实战匹配对照(§8.2)裁定 grouped ON 与工具调用参数缺失/重写螺旋强关联(ON 3/3 失败含一次全新会话 vs OFF 1/1 干净),解码速率两臂无差异、代价全在 agent 轨迹层。生产已回 OFF(2026-11-09);grouped 降级为离线/非工具负载的预填充加速开关**;L3 已于 2026-11-08 生产回滚(混批步固定成本实测 ~660ms + 1.3ms/tok,长 prompt 并发下常驻 decode 崩至 ~2.8 tok/s、入场 prefill 仅 ~320-640 tok/s,净负收益;详见 §2 回滚注)。开工前先读本文对应节的门槛与坑。

来源:1Cat-vLLM v1.5.2 逆向分析(HEAD c98a8d664,PR #387/#666/#860/#754 全部在 tag 内;
Apache-2.0/BSD-3/MIT 许可,可借设计与源码)。逆向全账见记忆
`1cat-71t-anatomy-and-levers`。**L1(D256 dense prefix 换 cuBLAS 混合化)封存**,理由见 §4。

## 0. 共同事实基线

| 事实 | 证据 |
| --- | --- |
| Qwen3.8-Flash-Next 48 层,`full_attention_interval=4` → 12 层全注意力(D256),36 层 GDN | nvidiaQwen3.8-Flash-Next-NVFP4/config.json:15 |
| QSA 稀疏 prefill 的 Triton 路径在 sm70 是 **FMA 地板**(tl.dot→标量,0 MMA);4k 形状自测 **29.5 ms/层**,对比 dense TileLang 3.55 ms/层 | qwen_sparse_attn_backend.py:1974-1978;kernels sm70 分工(PTX 0 条 mma.sync) |
| dense 路由只在**无 prefix 冷序列 ≤8192 token** 生效(= 长文首 chunk);其余 chunk 全走稀疏路径 | qwen_sparse_attn_backend.py:86-113, 1960-1998(`SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS`) |
| GLM DSA 稀疏核 = TileLang WMMA,M=16/128 线程/90KB smem,**17 TFLOP/s 发射器地板**(round-2 判死:gemm_only 等时,每-tile gather 依赖链无罪但 tile 形状封顶) | tilelang_sparse_sm70.py;记忆 glm53-dsa-round2-dead |
| GLM post-P5-a prefill 构成:DSA main_kernel **46.4%**、marlin 23.0%、NCCL 9.9%、cutlass 7.8% | GLM53_FLASH_PLAN.md:910-912 |
| Qwen 冷 prefill 3,080-3,500 tok/s(190k/32k),190k 冷首 turn 61.86s;prefill-first 零步进墙:cold wall ≈ N×context/2,500 | docs/v100/Qwen3.8-Flash-Next.md:67, 79-84, 104 |
| decode 与 L2/L3 无关:QSA decode/verify 恒 CUDA 分组内核;#37 账本 GPU-bound 88.8%,MoE 26.8%、DSA 8.6% | 记忆 glm53-d37-decode-anatomy |
| `--enable-mixed-chunk` 现判死:首个混批触发 QSA metadata device assert,毒死全 4 rank CUDA context | Qwen3.8-Flash-Next.md:105;现场 assert qwen_sparse_attn_backend.py:740, 806 |

1Cat 数字卫生警告(引用纪律):其内核倍数(#387 的 55.15→9.63ms、#390、#268 NCU
计数)树内**无复现脚本**;有脚本的是 #860(benchmark_mixed_prefill_latency.py)与
decode 路由报告。本文所有 1Cat 倍数按"设计模式可信、倍数需本机实测"处理。

---

## 1. L2 立项:SM70 grouped 稀疏 prefill(planner + 行分组 + WMMA)

### 1.1 打击面与预期

| 线 | 现状 | 预期(L2-0 实测后修正) |
| --- | --- | --- |
| Qwen 冷长文 prefill | 首 chunk 后全走 Triton 稀疏(29.5ms/层,FMA;实测 34.2ms/发@4096 chunk)。**L2-0 实测 QSA 稀疏核占 GPU busy 28.3%** | **+20~28%**(3,080→~3,700-3,950 tok/s;核加速 3-4× 口径;份额天花板 +39%)。原 +25~45% 上半段不可达——注意力份额仅 28.3%,远低于 GLM DSA 的 46.4%。1Cat 同模型族 64K 纯 prefill +29.9% 与本机口径相容 |
| Qwen 短请求/工具轮 | 首 chunk 走 dense,短请求占比高 | −10~20%(注意力份额低) |
| GLM prefill 8k | DSA 核 46.4% 份额、17T | **+20~30%**(1,390→~1,700-1,850) |
| decode(两线) | 路径不相干 | **0** |
| 质量 | — | 选中 K/V 集合逐位不变(同 topk 输入),变的是归约顺序/tiling → **非比特等价,须过全套门**(§3) |
| 显存 | — | ≈0(planner 数组 MB 级;1Cat 无全局 grouped 工作区,哈希用 96KB smem) |

### 1.2 设计参照(1Cat #387,`flash-attention-v100/kernel/flash_decode_paged.cu`)

三支柱,逐条对应我们 round-2 判死实验的三个发现:

1. **行分组摊薄 panel**:8 相邻 query 行 × 6 头 = 48 M 行共享一个 32-token KV
   panel(kernel 常量 kGroupedVerifyHeads=6 / MAX_QUERY_TOKENS=8 / BlockN=32,
   512 线程,1 CTA/group;kernel:2044,launcher:4834)。相邻行 topk 重叠实测
   82%。不假设并集、**算并集**,per-CTA 只迭代本组并集页。
2. **GPU planner**(kernel:4110,host wrapper:4362,+0.362ms 单独 launch):
   每行 topk → 逻辑页 → 块表 → 物理微块(atomicCAS 开放寻址哈希去重,
   nibble 掩码 atomicOr),然后 **CUB BlockRadixSort 按
   (active_m_tile_category<<32|logical_owner) 排序 + 类别 pad 到 8 的倍数**,
   输出稠密页表 + uint32 每行可见性掩码。主循环零间接零分支,仅 bitmask skip。
3. **gather 只在 panel copy 阶段**:每 4 token 一次页查、每 8 值一次 128-bit
   `uint4 __ldg`,就地 dequant 进 smem;MMA 只读 smem。

**必带修复(1Cat 自己炸过的)**:planner 哈希序会把物理分配序泄进输出,
跨分配非比特不变(`docs/design/sm70_qsa_page4_allocation_invariance.md`,
修复=按逻辑序排序;测试 tests/kernels/test_sm70_qsa_page4_plan.py);
类别 padding 行读 page-0 空块,fp16 GDN 状态按 E4M3 解出 NaN 时
`0×NaN` 毒死 PV,需 pad-fix 把掩码 0 条目重指到本组首个真实微块
(qsa.py:2166-2179,torch.where 无 host sync、graph-capturable)。

### 1.3 阶段

**L2-0(前置测量,独占机)**:Qwen3.8 冷 64k prefill 的 kernel 构成 profile
(GLM53_FLASH_PLAN G.5 同法)。注意力份额是 §1.1 收益估计里唯一没实测的乘数。
产出:份额表 + 修正后的预期收益,回填本文件。

> **L2-0 实测结果(2026-11-08,4×V100-32GB,生产默认配置/预算关)**:
> 引擎 `launch_server`(MTP EAGLE、chunked-prefill 4096、tilelang_fa_v100);
> 冷 65,601 token prefill(`flush_cache` 后、distinct prompt、max_new_tokens=1),
> e2e 29.3s(2,237 tok/s 冷),GPU busy 25.1s(86% 占空比)。TP-0 torch-profiler
> kernel 聚类(trace:`/tmp/l2prof/trace/l2cold64k-TP-0.trace.json.gz`):
>
> | 分量 | GPU busy | 份额 | 证据 |
> | --- | ---: | ---: | --- |
> | QSA 稀疏 prefill(Triton FMA,`_sparse_gqa_chunk_prefill`) | 7.11s | **28.3%** | 208 发,均值 34.2ms/发(= 12 全注意力层 × 17 chunk;与在案 29.5ms/层@4k FMA 地板吻合) |
> | NCCL AllReduce(RING_LL)+ custom AR/hc_combine | 6.14s + 0.60s | **24.5% + 2.4%** | 1,755 发;L2 范围外的第二大税,另行挂账 |
> | Marlin MoE(两种 tile shape) | 4.57s | 18.2% | 1,632 发 |
> | dense GEMM(cuBLAS s884/cutlass/turbomind) | 3.51s | 14.0% | 10,595 发;混 GDN 投影/首 chunk dense 路由/QKV,非纯注意力 |
> | GDN TileLang(main_kernel/kernel_kernel)+ mamba | 1.08s + 0.32s + 0.04s | ~5.8% | main_kernel p50 31µs(小核),max 2.4ms = 首 chunk dense TileLang 注意力(3.55ms/层口径吻合) |
> | elementwise/misc/other/norm | ~0.9s | ~3.6% | fast_topk 0.57%、QSA metadata(_expand_qsa_block_indices 等)0.06% 均已含 |
>
> **修正结论**:QSA 稀疏核换成 grouped WMMA(3-4×,17-25T)→ 回收 GPU busy
> 18.9-21.2% → **端到端 +23~27%**,份额天花板 +39%(核归零也不到 +40%)。
> §1.1 已按此回填。L2-a spike 判据不变:破 25T ≈ 本表口径下 +27% 附近。

**L2-a(spike,天级,TileLang 内)**:两问一次答:
- (a) 预排序稠密页表(离线 gather 好、零每-tile 间接)能否让现 TileLang 核破
  17-19T?能 → 发射器不是唯一地板,planner-lite 在 TileLang 有产线;不能 →
  钉死必须 CUDA。
- (b) **分组在两条线的 smem 可行性不对称**:Qwen(D256,Hq6)8×6=48 行 Q_shared
  24KB + panel 32KB,96KB 内宽松,**分组可行**;GLM DSA(H=16,D=512)8×16 行
  Q_shared = 128KB **超 Volta 96KB**——GLM 侧分组需换形态(2-4 行小组、或只吃
  planner 不吃分组),spike 必须带上这个预算账。
- 判据:任一形态破 25T → 走 TileLang 产线;全灭 → L2-b 按 CUDA 移植立项。

> **L2-a 实测结果(2026-11-08,1× V100-PCIE-32GB,容器单卡离线 microbench;
> 基准代码 `benchmark/kernels/benchmark_sm70_qsa_grouped_sparse_prefill_l2a.py`,
> 形状 = 产线真实形状:Q=4096 行、Hq=6、kv_heads=1、D=256、topk=2048、
> fp16 池 65,536 行、sm_scale=1/16)**:
>
> | 变体 | 中位耗时 | 有效吞吐 | 判读 |
> | --- | ---: | ---: | --- |
> | Triton 产线基线(`_sparse_gqa_chunk_prefill`) | 35.70 ms | 1.44 T(per-row FLOPs) | 与 L2-0 的 34.2 ms/发吻合;比 FMA 峰(~14T)还低一个量级——不止 FMA 地板,还有 6 头各自重复拉同一份 KV 的读放大(每发潜在 ~51GB) |
> | V2 每-token + 索引 gather(M=16,现 TileLang 核形状移植) | 11.20 ms | 4.60 T | **DRAM 带宽 bound**:8.6GB/发 ÷ ~900GB/s ≈ 9.5ms 地板;D256 把 GLM 的字节/浮点比翻倍,发射器地板(17T)根本没轮到 |
> | V2b 每-token + 稠密页表(零每-tile 间接) | 13.32 ms | 3.87 T | 稠密页表救不了每-token 形态——瓶颈是 6 头摊不掉的每行 KV 读,不是 gather 间接 |
> | **V1 分组(8 行×6 头=48 M 行)+ 组内并集稠密 panel,ov=0.82** | **5.80 ms** | **20.0 T(own MMA)** | **6.2× vs 产线基线**;带宽仅 407GB/s,未撞墙 |
> | V1 分组,ov=0.5(U=8832) | 9.52 ms | 23.4 T | U 越大效率越高——25T 方向真实存在 |
>
> 正确性:V1 vs fp32 参考 max|d| = 2e-5(fp16 舍入级);并集构造保证每行可见
> 自身全部 topk,无全掩码行。
>
> **两问的答案**:(a) 预排序稠密页表对每-token 形态无效(4-5T,带宽死),
> 但**分组形态 20-23T、6.2× 核加速——TileLang 产线成立**(判据原意:不到
> "全灭"且核加速远超立项所需的 3-4×;25T 未破,但限制因素是 occupancy——
> smem 62KB → 1 CTA/SM——而非发射器,属 L2-b 调优轴:BI=32 + 双 CTA 共驻、
> 或 M=48 拆分。注意 Hq=6 下组大小必须使 M=行数×6 为 16 的倍数 → 8 行
> (M=48)即最小可行组)。(b) smem 账实测落账:Qwen M=48 形态 Q_shared_l/r
> 24KB + K/V 32KB + S 6KB ≈ **62KB ✓**(寄存器 acc ~120/线程可编译);
> GLM 侧复算:H=16、D=512 下 8 行 = Q 128KB ✗、4 行 = 136KB ✗、2 行 =
> 100KB ✗(差 4KB 且零余量;V==K_nope 的共享 buffer 省一半也撞已记录的
> swizzle 双布局死结;BI=32 形态撞 FullCol 每 warp ≥16 列的 emitter 约束)
> ——**GLM 线分组在 TileLang 内不可行**,维持 round-2 结论:GLM prefill
> 17T 现状,或另立 CUDA 项(不随 L2-b 走)。
>
> 端到端外推(按 L2-0 账本):QSA 稀疏 7.11s → 1.15s(6.2×),回收 GPU busy
> ~23.7% → **冷 64k prefill 预期 +23~28%**(2,237 → ~2,750-2,860 tok/s,
> 落在 §1.1 窗口上沿;未计 planner-lite 开销,1Cat 口径 +0.362ms/发可忽略;
> 外推需 L2-b 原型在 4 卡实测确认)。

**L2-b(产线)**:L2-a 已裁定 **Qwen 线走 TileLang 产线**(分组 + planner-lite,
不再考虑 1Cat CUDA 路径移植)。工程项:(1) GPU planner——每行 topk → 组内
并集(逻辑序排序,确定性要求见 §1.4)→ 稠密 panel 物理化(K/V gather 到
[G,U,256] 缓冲)+ 每行可见性掩码,需无 host sync、graph-capturable、随图
bucket 预分配;U 按 bucket pad 到 64 倍数。(2) 核从 spike 形态转产线:接
QSA metadata 契约(有效项前置、-1 padding)、早 chunk visible<topk 行、
CUDA graph。(3) occupancy 调优轴:BI=32 + 2 CTA/SM(撞 FullCol ≥16 列
约束,需 64 线程形态验证)或 M=48 拆分,目标 25T+。(4) 验收走 §3 全套门。
GLM 线:分组被 smem 判死、planner 被 round-2 判死 → 本立项书范围内**不动**;
若要动,另立 CUDA 立项。

> **L2-b 实测结果(2026-11-08,4×V100-32GB PCIe,容器;开关
> `SGLANG_SM70_QSA_GROUPED_PREFILL`,默认关,eager prefill only)**:
> 实现 = `_kernels_grouped_sparse_prefill.py`(TileLang 分组核 + sort-based
> GPU planner)+ `qwen_sparse_attn_backend.py` 接线(batched_plan 之后、
> packed gather 之前,ineligible 时原路 fall through)。
>
> **与原工程项的三处偏差(均有 A/B 证据)**:
> 1. **并集索引直读池,不做稠密 panel 物理化**:V1g 间接读 5.88-6.32ms vs
>    V1 稠密物化 6.06-6.09ms,输出逐位一致——间接化零开销,省 ~2.4GB/发
>    往返。(2) 项"graph-capturable"未做:Qwen 生产 prefill 图本就 disabled
>    (`--cuda-graph-backend-prefill disabled`),且 breakable-graph 路径显式
>    返回 None 走 Triton;(3) 项 occupancy 调优未开(见后续轴)。
> 2. **planner 每层每 chunk 必跑,不可按 forward 缓存**:实测确认每个全注意
>    力层各自跑 indexer(qwen4_exp.py 层内 `layer_id % interval == 0` 分支),
>    topk_indices 逐层不同;冷 64k = 17 chunk × 12 层 = 208 次。
> 3. **planner 有一次 scalar readback**(max union length → U_pad bucket),
>    非"无 host sync"严格形态;eager-only 下与 chunk 间串行无重叠
>    (side-stream 跨层重叠被 indexer L+1 依赖判死)。
>
> **planner 成本(单卡实测,产线形状 R=4096/topk=2048)**:int32 全链
> (key=(group_slot<<20)|logical,sort→cumsum→u_rank→index_add_ 散射)
> ~10.65ms/发,其中 sort 1.8ms + elementwise 链 ~3.5ms + index_add_ 2.6ms
> + readback sync;index_put_ 形态实测 17.7ms 已弃用。**fused CUDA/Triton
> planner(smem 内 per-group 排序去重,1Cat 口径 +0.362ms/发)是补齐窗口的
> 后续优化轴**。
>
> **4 卡 A/B(同机同配置,仅开关不同,每臂 3-6 次)**:
>
> | 指标 | baseline(Triton) | grouped(本项) |
> | --- | ---: | ---: |
> | 冷 65,601-token prefill e2e | 28.78-29.83s(中位 28.96) | 24.06-27.11s(中位 25.2) |
> | 冷 prefill 吞吐 | 2,199-2,280 tok/s | 2,420-2,727 tok/s |
> | warm 8k prefill | 3.71s(2,214 tok/s) | 3.62s(2,265 tok/s) |
> | decode 256 tok(MTP on) | 68.3 tok/s | 69.0 tok/s |
> | 前缀缓存门(同 prompt 二发) | — | cached 65,600 tok,0.79s,输出与全量 prefill 逐位一致 |
>
> **判读**:e2e **+13~18%**(低于 §1.1 预期 +23~28% 下沿——差额即 planner
> ~10.65ms/发 × 208 发 ≈ 2.2s 的账,fused planner 后续轴可回收;核侧 6.2×
> 与 L2-a 一致)。三臂回归带内:decode/2k 上下文输出与 baseline 逐位一致;
> ≥4k 上下文 temp-0 出现零星首 token argmax 翻转(两臂输出均连贯、语义
> 等价,与离线 oracle max|Δ| 3e-5~5e-4 的 fp16 舍入包络一致,判读为内核
> 归约序差异的非比特等价签名,非正确性问题)。
>
> **运维注意**:TileLang 按核形状(U_pad bucket,512 步长)JIT,冷缓存首条
> 长 prefill 一次性付出 ~10 个 bucket 编译(实测 47-257s);缓存落在
> sglang-v100-jit 卷后跨重启复用(157 目录/115M,跨臂实测仍有 ~1-2 个
> bucket 重编,~18s)。生产启用需评估该首条成本,或预编译常见 bucket。
>
> **生产启用(2026-11-08)**:8110 compose(docker-compose-sglang-v100-mtp-8110.yaml)
> 已设 `SGLANG_SM70_QSA_GROUPED_PREFILL: "1"`;连带影响已在该文件标注——
> L3-1 mixed-prefill 预算的固定成本地板(350-650ms)按 Triton 核
> (×29.5ms)标定,grouped 后每 chunk 稀疏开销 ~16.6ms,750ms 甜点档待
> 按 §2 ladder 重标。

### 1.4 风险与坑清单

- **归约序变 → 输出非比特等价**:过 §3 门即可上,但**不能**按"等价内核"
  免检;spec verify 路径的 batched/单行一致性(top-8 近 tie 翻转属预期签名,
  参照 dsa-mtp 记录)要有判读预案。
- **CUDA graph**:planner 必须无 host sync、形状随图 bucket 预分配(pad-fix 的
  torch.where+copy_ 形态是他们的可捕获性样板)。
- **确定性**:排序键必须纯逻辑(物理哈希只做去重介质,不进输出序)。
- 我们 vs 他们的输入形状差:page_size 16 vs 4;fp16 池 vs E5M2;
  QSA indexer 输出契约(GLM topk 2048+4=2051 fused 契约,记忆 glm53-p5a-dsa-sm70)。
- 工作量红线:full 版若超 U2B2 项目量级,回报按 §1.1 重新过堂。

---

## 2. L3 立项:自适应 mixed-prefill 预算调度

### 2.1 打击面与预期

病灶(记录在案):prefill-first 引擎里,新请求的串行 chunked prefill 把在跑
decode 按在零步进,cold wall ≈ N×context/2,500,cold tok/s 随上下文线性劣化
(Qwen3.8-Flash-Next.md:104;记忆 qwen38-pr6-req-pool-analysis:"解法只在调度层
未立项")。

1Cat #860 实测口径(有 in-tree 脚本 `benchmarks/benchmark_mixed_prefill_latency.py`,
契约=2 常驻 decode + 2 个 32K 入场,250ms 软目标):

| 指标 | 对照 | 自适应 250ms |
| --- | ---: | ---: |
| prefill 期间 resident 输出 | 6.88 tok/s | **30.34(×4.4)** |
| 最长 resident 更新间隙 | 2.873 s | **0.310 s(−89%)** |
| 入场 TTFT(第 1/2 个) | 14.3/20.6 s | 18.0/35.5 s(**+26%/+72%**) |
| 纯 decode C1/C4/C8 | — | +0.29% / +1.57% / −1.33%(**中性**) |

质量:**零算术改动,最安全的一档**。显存:0。上下文/并发上限:不受影响;
对 2-4 流 ladder 纯改善。代价明码标价:新请求 TTFT 变长,按业务定
`target_ms`(他们 250ms;我们 agent 形态倾向偏小、保 resident 连续性)。

### 2.2 设计参照(`vllm/v1/core/sched/mixed_prefill.py`,纯 Python,可整段读)

`MixedPrefillBudget`:对完成的混合步做衰减(DECAY=0.75)运行和拟合
`elapsed = fixed + per_token×tokens`;affine 预算 `budget=(target−fixed)/per_token`
并带**固定成本地板**(预算永不低于"成本等于 fixed 的 chunk",防反馈塌缩);
种子 512 token;样本过滤:尾 chunk(<75% 预算)不入拟合、graph/eager 步分开
拟合;增长钳制 `min(desired, max(2×tokens, tokens+16))`,量化步进 16。
无 resident decode 时保留常规大 prefill 预算;纯 decode 走原 FULL graph 路。
GPU timing 只在受控混合步建 event、非阻塞查完成、回传纯数值——调度器不等待。

移植到 sglang:scheduler 是冻结核心类——**开工前必读 large-class-style**;
开关 `SGLANG_*` 走 environ.py 约定。**注意 sglang 侧没有现成混合步 timing
回传管道**,worker→scheduler 的 MixedPrefillTiming 等价物是本项目主要的
新工程量(不是算法)。

### 2.3 阶段

**L3-0(前置阻塞,必须先修)**:复现并归因混批毒批。现场是
qwen_sparse_attn_backend.py:740(`end_blocks*ratio <= token_slot_table.shape[1]`)
与 :806(`prefix_lens % ratio == 0`)两个 `torch._assert_async`。
判定哪个先炸、为什么(混合批 mix_running_indices gather 路径的 metadata
形状界?),做最小修复 + 红绿测试。修不好的退路:**不做真 mixed-chunk,
做"prefill 预算限幅 + 步间让位"的粗粒度形态**(预算逻辑照搬,让位粒度变粗),
效果折损待评。

**L3-1(产线)**:预算控制器进 sglang scheduler;验收用我们的 N-way ladder
协议(每 tier flush 冷 + warm 两轮,记忆 qwen38-pr6-req-pool-analysis 口径)
扩一臂"常驻 decode + 入场长 prompt",量 resident 间隙分布、TTFT、accept
不回退、warm decode 曲线不动。

### 2.4 风险

- 与 chunked-prefill 4096 帽、GDN chunk OOM 帽(4k 上限的来历)叠加后,
  预算下界可能撞固定成本地板的地板,需要 250ms→实测 fixed 成本的标定。
- decode CUDA graph bs 覆盖(compose 图帽 bs=3,bs=4 eager 既有记录):
  让位步数变多但每步形态不变,预期无新坑,验收时盯 graph-hit。
- TTFT 恶化是设计代价不是 bug;serve 脚本注释与 README 登记时要写明开关、
  target 值与两侧代价。

---

## 3. 共同验收门(实现 PR 的完成定义)

按 AGENTS.md 数值验证纪律,两杠杆共用:

1. **smoke**: `bash scripts/smoke_v100.sh` 必过(零输出检测)。
2. **Qwen3.8 回归硬门**(基础设施改动先跑):三臂 A/B——prefill 8k、decode、
   MTP accept 与基线带内(GLM53_FLASH_PLAN.md:1351-1353 方法:2,946/2,947/2,877
   量级、accept 1.95-2.58 带);生成输出质量不劣化。
3. **GLM 质量门**:tops top-1 24/24、needle 5/5(判别 topk 域 ≥2051)、
   同 prefix `cached_prompt_tokens` 提升 + 输出与全量 prefill 一致(前缀缓存门)。
4. **性能声明**:4×V100 本机实测,记录硬件/命令/可解读汇总;1Cat 倍数只作
   立项依据,不作验收依据。
5. 开关登记 README 表;`pre-commit run --all-files`。

## 4. 封存记录:L1(D256 dense prefix → cuBLAS 混合化)

- 打击面塌缩:dense 路由只覆盖无 prefix ≤8192 token(§0),端到端收益
  +1~3%,不抵 400MB/rank score workspace(1Cat Q4096 路由实占 384MiB 印证)
  对 YaRN 判死记录的 3.23GB 图后余量的挤压。
- 质量有实证风险:FP16 score 存储+最大值采样是近似路径;1Cat 79T recipe 在
  **真实模型输入溢出炸过**(随机输入全过、冷模型失败,sm70_79t/README.md:41-48),
  其 #666 的 96 题输出质量门至今未过(87 vs 88)。
- 若未来 dense 覆盖面改变(如 >8192 dense 路由复活)或 #754 型调度把 warm
  请求推入 dense 路径,重开本案。

## 5. 挂账(backlog,未立项)

- **冷长 prefill NCCL AllReduce 税**(L2-0 副产物):RING_LL AllReduce 占冷
  64k prefill GPU busy 24.5%(6.14s/1,755 发),+ hc_combine 2.4%。与 QSA 稀疏核
  28.3% 同级,却不在任何杠杆打击面内。候选方向:prefill 两段式 AR 重叠、
  one-shot 自定义 AR 阈值上探(chunk 4096 时是否还该走 RING_LL 待量)。

- **#754 稀疏 retention 边界调度**(1Cat):816-token GDN state block 把 prefill
  预算切成多次 mini-forward,scheduler 只在被保留状态边界停 → Flash-Next
  prefix-on 3,046→5,068 tok/s(+66%)。我们工具轮 0.34-0.61s/500-900 token
  (≈1,000-1,500 tok/s,远低于 3,000 稳态)是同病理指纹。纯调度、零内核,
  与 L3 同族;先量我们的 mamba state block 是否切碎 chunked prefill 再立项。
- QPN fragment 预排 decode GEMV(认命发射地板、打 DRAM 饱和):#37 账本
  wmma skinny dense 仅 20% 份额,低于 L2/L3,继续挂账。

---

## L4 立项:tier 卷绕区 prefill 分窗 staging + 双流预取

立项 2026-11-08。打击面:KV host tier(Phase 3 直映射)开启时,"prefix 超出
device 驻留窗"的 prefill 批次——516K 单流(S3 已验收形态)与未来 256K×2
PAIR+tier(生产目标形态,PAIR 把 device 池砍半、卷绕更早发生)。原生 262k
冷 prefill 全驻留(tier 仅 −2%),不在打击面内。

### L4.1 现状病灶(Phase 3 读路径的三个结构弱点)

S3 读 host 走 `_p3_gather_kernel` 逐行 UVA 直读(扩展 metadata 构建期),
对照 1Cat `vllm/models/qwen4_exp/nvidia/qsa.py:1059-1237` + `host_kv.py:762,943`
的窗式 staging:

| | 我们 Phase 3 | 1Cat 窗式 staging |
| --- | --- | --- |
| DMA 粒度 | kernel 内逐行散读(延迟受限,跑不满 pinned 带宽) | 整窗连续 DMA(stage_plan 一次规划) |
| 与计算关系 | H2D 等待串在 attention 关键路径 | side-stream 双缓冲预取下一窗藏进上一窗计算(stage_slots=2) |
| 重复读 | 同一 host 页被多行选中则多次 UVA | 窗内每页至多搬一次,attention 读 device 副本 |
| 规划 | 每层每步各自为政 | 每步每几何一次 host sync,全层共享计划 |

1Cat 配套细节(移植时要一起看的):`write_staged` 让本步新写页直接进 stage
buffer 不回读 host;超窗则分窗 online-softmax merge;计数器每 CTA 私有。

### L4.2 收益与量级(诚实账)

- 已知锚点:tier 关闭 262k 冷 prefill ≈1.8s/4k 批(2,238-2,301 tok/s);
  S3 516K 卷绕区 ~3s/批(tier 开启后 +40%)——TTFT ~5 分钟里约 1 分钟量级
  是 host 读税。窗式 staging+预取**上限**回收该罚金(516K TTFT −10~15%),
  **下限**接近零(若那 1.2s/批主要是 mirror D2H 或调度空转)。
- 收益口径:**只买 TTFT,不买 decode**(decode 本就 context-flat,
  qwen38-pr6 记忆;1Cat 同理,其 "250K 仅慢 18%" 说法树内无出处,其 16GB
  文档只证 262K 能完成不 OOM,TPOT 归因于 36 GDN O(1)+12 QSA O(2051)
  的模型结构,不归因 host KV)。
- 数值风险:零(同样字节先搬 device,attention 数学不变,验收退化为
  位等价检查);新内核:零(纯 Python + copy 流调度)。

### L4-0 前置测量(判据门,半天)

独占机 boot,`torch.cuda.Event` 级探针测 `_p3_gather_kernel` 在卷绕区批次
的时长占比(计数器 host_gather_rows/mirror_skips 已在,只缺计时):
- **gather 占卷绕区批次 ≥25%** → L4-1 立项实现;
- 否则封存本条(税不在 gather),工程量让给 L2/L3。

### L4-1 实现(过 L4-0 后)

改 `kv_host_tier.py`/后端调度层(fork 拥有区):每步每几何一次 host sync
出窗计划 → 整窗 DMA 进定长 stage buffer → attention 读 device 副本 →
side-stream 预取下一窗。stage buffer 预算见 §6.3 显存敏感环境(YaRN 判死
记录:图后余量 3.23GB、warmup 需 2.1-2.4GB,新分配必须入账)。

### L4.3 验收

S2 矩阵全 rung 逐字命中(位级判据沿用 Phase 2/3)+ S3 516K 计时 A/B
(TTFT 与卷绕区批次时长);生产形态 256K×2 PAIR+tier 落地时复测。
开关走 `SGLANG_V100_KV_TIER_*` 现有命名族(environ.py 约定)。

## L5 立项:retract → host page-out → 按页恢复(请求级 tier 转出)

立项 2026-11-08。提高并发的第三条路(前两条:PAIR 去复制已落地;L3 让
"装下了但不互相拖"已立项)。打击面:**max-running-requests 容量上限本身**——
第 4/5 路入场时,现有 retract 把被逐请求的 KV 直接丢弃、恢复=整段重 prefill
(128K ≈ 1 分钟);本杠杆把 retract 改造成"先 D2H 转出到 host、恢复按页 H2D"
(128K ≈ 10-15s,PCIe 13.1 GB/s 实测带宽),并解除并发上限对"全驻留"的依赖。

### L5.1 设计参照(ninfer HEAD 77c2e5e2 = github/master)

借鉴其**活着的三件套**,不抄其已弃公式(历史教训记录在案):

1. **请求级 snapshot/replay 恢复分级**(`engine_schedule.inl` preemption
   victim;`context_cache/resource_manager.h:1500` paused snapshot):压力下
   转出整个请求,恢复走 replay(少量重放)> snapshot(H2D 搬回)> root
   prefill,按恢复成本选路。**纪律:只 page-out 恢复便宜的**。
2. **可审计性**:每请求记录 pause/restore/recovery transitions 与 route
   (snapshot vs replay vs null),决策质量可离线复盘(对照我们 tier Phase 3
   只有 miss 阈值粗规则、无事后审计)。
3. **预占 victim 选择**:scheduler 侧独立 `preemption_victim` 入口,策略可换。

**明确不抄**:ninfer 重型 portfolio 估价(`Saving=Rebuild−Recovery` +
LiveSession/RecentPrivate/Disposable 权重)——其作者在 b9114396(2026-10-04
"replace context cache",删 5,408 行)已亲手弃用,现行只剩 `last_demand`
128 请求窗口 + 增量 reclaim;不要按我方 2026-11-08 早先误引的 §8.3 公式立项。

### L5.2 与现有设施的映射(零新基础设施)

| ninfer 概念 | 我们已有的对应物 | L5 要做的 |
| --- | --- | --- |
| snapshot(转出) | Phase 1 镜像管道(`_host_final`,页级 fp16 host 权威) | retract 前 flush 镜像即完成(镜像本就 keeps-up,S3 已证) |
| replay(重放) | radix 前缀缓存 + hicache host/disk 层 | 已有,恢复路径接回即用 |
| snapshot 恢复(H2D) | Phase 2/3 页入(`_page_in_kernel`/P3 restore) | 从"活跃请求卷绕"扩到"被 retract 请求" |
| StateImage(状态) | mamba 池 extra_buffer 策略 | GDN 态随 KV 同转出(需验证 mamba host 池支持) |
| per-request 审计 | 无 | 加 pause/restore/route 计数器与 JSONL 事件 |

### L5-0 前置判据(不开工条件,硬门)

- **触发条件(满足其一)**:实测 4 并发下 retract 频率 >0(按 qwen38-pr6
  记忆,现负载 retract=0,当前是空转杠杆);或业务目标并发 ≥5(l3budget
  FULL 布局 4 并发为现形态上限,8110 compose PAIR 锁 3);或多会话切换负载
  实测恢复等待占比显著。
- **不触发则封存本条**——tier Phase 3 的"准入按常驻工作集"与 L3 已覆盖
  现有 3-4 路的实际约束面。

### L5-1 实现(过 L5-0 后)

改 scheduler retract 路径 + `kv_host_tier.py`(fork 拥有区):retract 分级
(先 page-out 后真丢)、恢复按页 H2D、GDN 态转出验证。scheduler 是冻结
核心类,**开工前必读 large-class-style**;开关 `SGLANG_V100_KV_TIER_RETRACT_*`
命名族;验收延用 S2 位级判据 + 新增 retract/restore 计数器审计。

### L5.3 风险

- mamba 线性注意力态的 host 转出/恢复位一致性(GDN 态非 KV,走 mamba 池
  语义,extra_buffer 策略下行为需单独验证)。
- MTP draft KV 与 target KV 的恢复原子性(半恢复请求不可进 decode)。
- 恢复与 L3 混合 prefill 的调度交互:恢复期请求占不占混合预算,需定约。

### L5.4 与 PAIR 约束的交互(2026-11-08 补,决定 L5 能买到多少并发)

PAIR 硬约束分两层,常被混为一谈:

- **物理层(不可消除)**:`device 池 ≥ logical/2`——PAIR 对拆后对家页靠
  P2P 读**对家 device**,不是读 host;全局字节下限 = 逻辑 token 的一半 ×
  12,288 B/tok/rank,任何调度层手段绕不开。
- **准入层(可调度)**:4×256K 失败真因(compose 8110 :191-193)是 logical
  524,288 tok 要求 device 池 ≥262,144 tok/rank = 6.0 GiB,激活余量 20 MiB
  warmup OOM——"4 个请求的瞬时峰值都要满足约束"的问题,非约束本身不可满足。

L5 的作用 = 把约束从"对 max-total-tokens 全体成立"弱化为**"对任一时刻的
活跃请求集合成立"**:空闲会话(等工具结果/排队)page-out 后赎回其 device
半池,新请求入场;恢复时按页 H2D 搬回、重新占住半池。等价表述:**并发能力
= device 池能同时装的活跃请求 + host 镜像能装的所有请求**。按现有账
(PAIR+tier 池 4.5 GB/rank @3 并发),4×256K 变为"3 活跃 + 1 转出等待"的
可行形态;**纯 4 活跃仍不可行**。

边界(诚实账):

1. 单请求约束分文未减:任一活跃请求自己的 token 仍要求其 device 半池驻留
   (对家页是 P2P 读 device)。256K×1 的最小 device 需求不变——L5 不帮
   "单个请求更大"(那是 YaRN 判死记录 §6.3 的权衡组)。
2. **恢复峰值双份问题**:恢复是只增不减的字节流,峰值容量 = 活跃请求 +
   正在恢复的请求;若批量同时唤醒,峰值约束瞬间回到 4×。**L5-1 必须定
   恢复限流/逐个唤醒策略**(恢复请求与活跃请求合计 ≤ device 池可容的
   活跃数),并登记恢复完成事件供审计。
3. 它赎回的只是"空闲会话"的占用:agent 形态(主会话等工具、子代理排队)
   收益真实;4 个请求同时连续 decode 的吞吐型负载无处借力,4×256K 仍不可行。

## 6. 封存记录:L3-0(混批毒批修复,2026-11-07/08)

**结论:`--enable-mixed-chunk` 在 Qwen3.8-Flash-Next(12×QSA + 36×GDN)上
已可安全服务;混批 prefill ~2,400 tok/s(2 常驻 decode + 32k 入场,8×4096
chunk 全部混入 decode 尾行,0 scheduler 异常)。**

- **根因(三层)**:① QSA 压缩写计划断言 `prefix_lens % ratio == 0`
  (`_qsa_build_write_plan`);② 混批把常驻 decode 行转成 extend 行
  (`mix_with_running`),尾行 prefix = seq_len−1,一般非组对齐 → device
  assert 全灭 4 rank;③ 修复最初按 `forward_mode.is_mixed()` 门控,但
  **eager_runner.execute 在 metadata init 之前把 MIXED 改写成 EXTEND**
  (eager_runner.py:261),门控永远看不到 MIXED——即 `mix_running_indices`
  在 worker 侧被 `resolve_forward_inputs` 消费置 None 之后,mixed 身份的
  最后一个幸存信号也被改写吞掉。
- **修复(4 文件,最小 diff)**:
  - `eager_runner.py`:改写 mode 前记录 `_original_forward_mode = MIXED`
    (沿用既有约定:IDLE/TARGET_VERIFY 改写同款);
  - `qwen_sparse_attn_backend.py`:metadata 构建对 `mode.is_mixed() or
    original_mode == MIXED` 计算 `_mixed_tail_mask`(~((prefix % ratio ==
    0)),纯 device 推导,不依赖已被消费的 mix_running_indices);断言改为
    "mix_tails is None 才 assert 全对齐";
  - 尾行完成组用 −1 哨兵 member_rows,压缩分两 pass(chunk 来源 + pending
    ring 来源,`torch.where` 重定向非目标条目到惰性槽 0);**陷阱**:完成组
    的尾行 token(len % ratio == 0)恰不是 pending,必须写真 ring 槽
    (`tail_rows` 强制),否则 ring 重建缺最新成员。
  - 消费者安全核过:`_original_forward_mode` 设 MIXED 对 PLE(is_idle/
    is_decode)、DSV4 indexer(rewritten 批走保守 paged 路径,注释本就
    如此要求)、dense-prefill 门控(混批因行长不等本就排除)均无行为变化。
- **红绿测试**:CPU `test_qsa_mixed_chunk_compress_plan` 4 例(未对齐尾行
  接受、完成组 ring 来源、ring 压缩、尾 token 落真 ring 槽)+ 既有 7 例,
  11/11 绿;GPU e2e 红(修复前 4 rank assert 全灭,CUDA_LAUNCH_BLOCKING
  定位发射点)+ 绿(2 decode + 32k,3 请求 OK)。
- **验收门(§3)**:smoke_v100 过;prefill 8k 2,687-2,798 tok/s(同树 A/B:
  mixed on 2,734-2,798 vs 生产配置 MTP 2,687-2,732,重叠 → 开关与改动对
  prefill 零代价;与镜像内基线 2,840-2,947 的 −4~6% 符合 H.5.6 已记录的
  dev 挂载仪器假象);decode(MTP)95-114 tok/s ≥ 基线带 81-83;accept
  2.70-3.15 ≥ 带 1.95-2.58(数数类贪心文本可预测性高,方向为优);质量
  臂(数数 1-100 完整、链表反转代码、前缀缓存解释)不劣化。
- **遗留**:生产启用已完成(2026-11-08):`docker/v100-compose.yaml` 与
  `docker/v100-compose-qwen38-mtp.yaml` 均设
  `SGLANG_MIXED_PREFILL_STEP_LATENCY_MS=750`(env 可覆写;standalone 文件另带
  `--enable-mixed-chunk` flag),README 性能节已登记;`_original_forward_mode`
  的 DSV4 paged 路径回退仅在有混批流量的 DSV4.1 部署里生效,当前 DSV4.1
  生产未开混批,无影响。

## 7. 封存记录:L3-1(自适应 mixed-prefill 预算控制器,2026-11-08)

**结论:1Cat #860 的 `MixedPrefillBudget` 已移植进 sglang scheduler
(`--enable-mixed-chunk` + `SGLANG_MIXED_PREFILL_STEP_LATENCY_MS=750`,
均默认关闭)。混批 ladder(2 常驻 decode + 32k 入场,4×V100-SXM2 实测):
resident 最长间隙 1.988 s → 1.080 s(−46%),代价入场 TTFT 19.5 → 22.5 s
(+15%)、resident 流 18.5 → 16.3 tok/s;三臂回归全部带内。**

- **交付物(4 文件)**:
  - `python/sglang/srt/managers/scheduler_components/mixed_prefill_budget.py`
    (新建,~330 行):`MixedPrefillStep`(CUDA event 对 + 非阻塞
    `try_resolve`)与 1Cat 控制器(DECAY=0.75 衰减和拟合
    `elapsed = fixed + per_token×tokens`,affine 预算
    `max(target−fixed, fixed)/per_token`,固定成本地板、16-token 量化、
    尾 chunk <75% 预算不入拟合);`MixedPrefillBudgetControl` 为
    scheduler-facing collaborator(large-class-style,调度器只 wire +
    delegate,零域逻辑进 Scheduler 本体)。
  - `python/sglang/srt/environ.py`:`SGLANG_MIXED_PREFILL_STEP_LATENCY_MS`
    (0 禁用)与 `SGLANG_MIXED_PREFILL_MIN_TOKENS`,按 env-var-conventions
    登记;专家开关不设 CLI flag。
  - `python/sglang/srt/managers/scheduler.py`:5 处最小接线(import、
    `maybe_init_mixed_prefill_budget()`、`get_new_batch_prefill` 限幅、
    `run_batch` 前后 event 记录、`process_batch_result` 回传)。
  - `test/registered/unit/managers/test_mixed_prefill_budget.py`:21 例
    CPU 单测(register_cpu_ci,base-a-test-cpu),覆盖种子/尾过滤/钳制/
    阻尼/affine 拟合/give-up/生命周期 fake events。
- **对 1Cat 原版的三处实质修正**(都是 V100 实测逼出来的,原版没有):
  1. **小样本过滤** `min_sample_tokens = 2×min(16, max_tokens)`:V100
     mixed 步固定成本 ~350-650 ms(QSA 稀疏 prefill 12 层×29.5 ms 地板
     ≈354 ms),小于 ~32 token 的混合步 elapsed 全是噪声,拟合会把
     per_token 拉向 0,预算塌缩(500ms 目标实测塌到 48-256 token/步,
     prefill 200 tok/s)。
  2. **不可达目标 give-up 安全阀**:目标低于固定成本地板时(连续 8 个
     满预算样本超时且 budget ≤ max/8)放弃限幅并 WARNING。1Cat 的
     250ms 在 V100 上不可达(地板 350ms+);750ms 是本机合理档。
  3. **headroom 比例阻尼** `growth = min(2.0, target/elapsed)`:原版
     无条件 `min(2×tokens, tokens+16)` 增长,在步成本高方差(同 512
     token 实测 788 ms vs 353 ms)下小样本显得便宜 → 预算 192↔3840
     弹射(resident gap_max 1.577 s),比不开限幅还差。阻尼后稳定在
     ~1500-1600 token/步。
- **timing 回传管道(本项目主要新工程量,§2.2 预判正确)**:sglang 无
  worker→scheduler timing 通道,改为 scheduler 侧在 `run_batch` 的
  forward enqueue 前后各 record 一个 CUDA event(forward stream 上,
  elapsed 精确度量 forward 步);`process_batch_result` 收到的是
  `batch.copy()`,样本不能挂 batch,`MixedPrefillBudgetControl._inflight`
  按 `batch.forward_iter` 键存,MAX_INFLIGHT=16 / MAX_PENDING=8,
  event 未 ready 的样本丢弃(非阻塞,调度器零等待)。
- **预算语义**:限幅返回 `min(chunked_prefill_size, budget.tokens +
  running_bs)`——`PrefillAdder.__init__` 会把混批 decode 行从
  rem_chunk_tokens 扣掉,加回 running_bs 才能让 prefill 部分落在
  budget.tokens。无 resident decode / 不可混合(logprob、beam、
  input_embeds)时返回 None,常规大 chunk prefill 不受影响。
- **验收门(§3)实测**(docker `sglang-v100-unified:latest`,TP4,
  mixed-chunk on,预算 750 ms):
  - smoke_v100 过。
  - target 引擎:prefill 8k 2,766-2,780 tok/s(带 2,687-2,798);质量臂
    3/3(数数 1-100 完整、链表反转、前缀缓存解释)不劣化。
  - MTP 引擎(EAGLE 3/4):decode 95.4-113.9 tok/s(带 95-114);
    prefill 8k 2,683-2,700 tok/s(带内);accept len 2.92(纯 decode 窗)/
    2.98(混批 ladder 窗),均 ≥ 带 2.70-3.15,混批+预算对 accept 无回归。
  - 混批 ladder:见结论行;混批步 ~2,000-2,100 tok/s,预算稳定
    ~1500 token/步(MTP 窗 ~1470)。
- **未决 → 已定位并修复(2026-11-08 深夜)**:复验期间两次 Scheduler/
  NCCL watchdog 击杀,py-spy + NCCL flight-recorder 现场定位为**预算控制器
  破坏了 TP 调度确定性**:sglang 每个 rank 独立跑 Scheduler,锁步依赖所有
  调度输入跨 rank 位相同;原移植把本 rank CUDA event 计时喂进各 rank 自己
  的控制器,微秒级计时差经 EWMA/affine 拟合放大成不同 rank 的 chunk cap
  走向不同量化值 → 各 rank 调度出不同的 forward(decode verify vs
  prefill)→ allreduce 形状不匹配 → 4 rank 全部 enqueue 同一 SeqNum 后
  集体挂死。修复照抄本 fork `DynamicChunkSizer` 的既有先例:计时只在
  leader(world rank 0)做,每个混合步完成时把 elapsed 经 CPU group
  `broadcast_pyobj` 广播,所有 rank 用同一样本更新控制器(每个 ~750ms
  混合步一次微广播,成本可忽略);leader 头部样本经 32 轮仍不 ready 则
  全 rank 一致丢弃。单测新增 leader/follower 收敛与丢弃一致性 2 例
  (共 23 绿)。
- **完整 ladder 协议(§3.2,8k/32k/64k × flush 冷/warm 两轮 × 2 常驻,
  MTP 引擎,修复后代码)**:两臂各跑完整 ladder 无死锁无分叉;resident
  最长间隙 OFF 2.143 s → ON 0.822 s(−62%),gap_mean 0.335 → 0.322
  (混合步节奏不变),warm 轮入场 TTFT 两者 1.3-3.7 s 带内,accept
  OFF 2.805 / ON 2.92-2.98(带 2.70-3.15,不回退)。控制台 EMPTY 流为
  客户端对 32-token 短生成 + MTP 的度量伪影(受控探针验证流式正常),
  引擎侧全部请求正常完成。
- **遗留**:生产默认仍关闭(mixed-chunk 与预算都 opt-in);启用即在
  serve/compose 设 `SGLANG_MIXED_PREFILL_STEP_LATENCY_MS`(>0 自动附带
  `--enable-mixed-chunk`,见 serve 脚本注释)。DSV4.1 未开混批,不适用。

## 8. Grouped prefill 质量受控 A/B 方案(2026-11-08 立项,待执行)

背景:2026-11-08 两次质量异常与 `SGLANG_SM70_QSA_GROUPED_PREFILL=1` 时间上
关联(第二次开关对照干净:ON 异常 / OFF 复测正常;第一次确认是 WorkBuddy
侧记忆污染)。引擎侧既有 A/B 证据(≤2k 逐位一致、≥4k 零翻转)与用户报告
矛盾,需要一次受控定案。理论预期仅 fp16 归约序差异(TileLang 在线 softmax
vs Triton,3e-5~5e-4),temp-0 argmax 在 top-2 极近时蝴蝶分叉,非系统性。

**三臂设计**(不是两臂):

| 臂 | flag | 作用 |
| --- | --- | --- |
| A1 | 0 | 基线 |
| A2 | 0(第二次 boot) | null 对照:boot 间噪声地板(JIT/图捕获/内存布局逐位差异) |
| B | 1 | 待测 |

B 相对 A 的分叉率**超过 A1-vs-A2 地板**才归因于 flag。唯一变量 = flag;
两臂同镜像、compose 其余项一致(mixed-chunk 已摘除、预算 env=0)、同
`--random-seed`(compose 已固定)。页对齐修复(a8ede6f)两臂都含,不构成
臂间差异;若 A/B 无差异但质量仍差于"昨天",补 pre-a8ede6f 代码臂。MTP 保持
生产形态;若发现异常分叉,追加去 MTP 对照。

**执行**(runner:`scripts/ab_qsa_grouped_quality_v100.py`,任务构建已在
容器内离线验证:33 任务 = 20 基础(数学×5 有答案/代码×5/中文×5/agentic×5)
+ WorkBuddy 鹈鹕骑自行车 canary(SVG 动画,用户实际质量门)+ 每长度档
(1k/4k/16k/32k/64k/128k)needle QA + 文档内数学各一):

```bash
# 每臂 = 一次 boot(compose 改 flag)+ 一条命令;8110 停服 ~1.5h
docker run --rm --network host --entrypoint /opt/venv/bin/python \
  -v /data/develop/sglang-v100-sxm2-qwen3.8-flash-next:/sglang-workspace \
  -v /data/models/nvidiaQwen3.8-Flash-Next-NVFP4:/data/models/nvidiaQwen3.8-Flash-Next-NVFP4:ro \
  -v /tmp/ab_grouped:/tmp/ab_grouped \
  sglang-v100-unified:latest /sglang-workspace/scripts/ab_qsa_grouped_quality_v100.py \
  run --arm A1 --base http://127.0.0.1:8110 --out /tmp/ab_grouped
# (A2 同命令 flag=0 第二次 boot;B 臂 flag=1,首条长 prefill 付 JIT 1-4 分钟,已剔除统计)
analyze --out /tmp/ab_grouped --arms A1 A2 B
```

臂内协议:8k warmup(剔除)→ 33 任务 bs=1 串行(temp=0、固定 max_new、
ignore_eos、top-2 logprob)→ `/flush_cache` 后重跑 11 条(含 canary,G1 门)。

**判定门**:

- G1 同 boot 确定性:重跑与首跑必须逐位一致;不一致 = 引擎自身不确定
  (批不变性/overlap),先修它,结果不归因 flag。
- G2 分叉:B 臂分叉 tag 数 ≤ A1-vs-A2 地板,且每个分叉点 top-2 margin
  < 1e-3(蝴蝶特征)。**红旗(任一即判有害,不重开,开 bug)**:far-margin
  分叉、分叉率超地板、thinking 长度分布显著偏移(过度思考信号)。
- G3 任务:数学/needle 有答案子集命中率两臂相等。
- **G3-canary(用户实际质量门)**:鹈鹕骑自行车 SVG 每臂落盘
  `pelican_<arm>.html`,analyze 自动检查结构标记(DOCTYPE/svg/
  animateTransform/html 闭合);最终裁定须人工浏览器渲染三份文件——
  骑行正常为 PASS 必要条件,B 臂结构损坏而 A 臂正常 = 红旗。

**注意**:runner 建议 `docker run --network host`(`--base` 直接写
`http://127.0.0.1:8110`);结果 JSONL 落宿主 `/tmp/ab_grouped`
(臂文件 arm_*.jsonl,可断点续跑)。完成后把裁定回填本节与本文件头状态行。

### 8.1 执行结果与裁定(2026-11-09)

v1 执行暴露两处测试设计缺陷,数据作废(原始数据在 `/tmp/ab_grouped/v1/`):
①裸文本不套 chat template → 模型落入"百度知道"式续写退化态,边界 token
(`\n\n` vs `assistant`)是真实近平手,放大噪声面,G3 判定全灭;②发现引擎
存在 **flag 无关的 run-to-run 噪声**:同 prompt、同 seed、同 boot 内 flush 后
重复 5 次 4 次逐位一致、1 次首 token 分叉——内核选择随内存布局漂移,把近平手
argmax 随机翻转。v2 修复:全任务套 chat template(短答手动注入空 think 块,
Qwen3 模板的 `enable_thinking` kwarg 虽存在但对该模型不生效)。

**v2 结果(33 任务 × 3 臂,44 记录/臂)**:

| 判定门 | 结果 | 裁定 |
| --- | --- | --- |
| G1 同 boot 重跑逐位一致 | A1 7/11、A2 8/11、B 7/11(v1 为 0-1/11) | 残余翻转均为近平手 margin(1.5e-2~6e-2)= 引擎噪声,非 flag |
| G2 分叉率 | 地板(A1vA2)20/33 vs B 23/33 | 同量级,B 不超地板 |
| G2 大 margin 翻转 | 地板 2 个 >1.0,B 4 个 >1.0 | 两臂都有(16k+ 长上下文),1 样本/tag 无统计差异 |
| G3 needle+math_ctx(12 题) | A1 12/12、A2 12/12、B 11/12 | B 差 math_131072 一次,噪声级 |
| G3-canary 结构 | A2 13.3k tok、B 10.1k tok,5/5 标记全过 | 待人工浏览器渲染裁定 |

math0-4 全臂未命中 = `enable_thinking` 未生效、64 token 预算全耗在思考里,
各臂同等失败,不影响对比。v1 里"看起来的 B 臂质量差"与 v2 定量的
"地板=B 臂噪声"共同指向:**用户感知的质量波动主要来自引擎级蝴蝶噪声
(与 flag 无关)+ WorkBuddy 记忆污染**,grouped prefill 无系统性质量退化
证据。

**遗留**:①canary 双臂 HTML(`/tmp/ab_grouped/pelican_{A2,B}_canary.html`)
人工渲染后定案,若正常建议 flag 回 "1" 拿回 +13~18%;②引擎 run-to-run
噪声(内核选择随布局漂移)值得单独治理(确定性内核选择/batch-invariant
路径)——它同时解释"同一 prompt 重试输出漂移",与 grouped 无关。

### 8.2 WorkBuddy agentic 实战匹配对照与裁定(2026-11-09,推翻 §8.1 的生产建议)

§8.1 判定"grouped 无系统性质量退化"后生产回 ON(11-09 10:26 boot);当日
WorkBuddy 鹈鹕骑自行车任务连续三次 21-24 分钟(健康基线 2-4 分钟),agent
自述"尾部混入大量无效片段"。经 shim 请求日志(2.8GB/3834 行,含完整 SSE
响应)取证 + 同 prompt 开关匹配对照,裁定如下。

**取证一:引擎侧零损坏证据**(脚本 `/tmp/l2b_work/shim_*.py`,原始日志已
归档 `/data/develop/l2b_work/shim_debug_groupON_20261107-09.log.gz`):

- 全日志 `finish_reason` 无一处 length → 排除 max_tokens 截断;
- 鹈鹕会话每次 Write 的 arguments 完整干净(全部以 `</html>\n"}` 收尾、
  键齐全)→ 客户端记录的历史参数 = 引擎实际交付内容,引擎输出无损;
- 607 个含 HTML 的读回结果尾部全部干净(仅 19 个良性行号槽标);
- "尾部混入大量无效片段"的自述不可信:13:16 的"结构有问题"无任何读取
  即宣布;13:22 的"读取确认"是 `limit:60` 读 236 行文件,从未看到它声称
  损坏的尾部,且读到的 60 行干净。

**取证二:耗时倍增 = 生成量倍增,不是速率回退**。shim 逐轮 usage + 轮间
墙钟:当日(grouped ON)鹈鹕会话有效速率 42-64 tok/s,与 11-07/11-08 基线
52-67 tok/s 同带;三次失败各生成 64-71k tokens(健康任务 ≈8-10k)。速率
恒定时"耗时 ×8"严格等价于"token 量 ×8"——全部来自重写螺旋(单文件 15 次
完整重写)与 3 次 Write 缺 `file_path` 整体作废(每次烧 9.6-15k tokens)。

**取证三:同 prompt 开关匹配对照(决定性)**。WorkBuddy 对同一 prompt 的
重跑是逐字节相同的请求(prompt 32,607 tok,连 629/21 bootstrap 轮都一致):

| 指标 | ON(14:13,grouped=1,近全新会话) | OFF(15:06,grouped=0,全新会话) |
| --- | --- | --- |
| 生成总量 | 67,509 tokens | **7,457 tokens** |
| Write 次数 | 8+ 次重写 | **1 次,一次过** |
| 工具参数缺失 | 3 次(当日累计) | **0** |
| 总耗时 | ~22 分钟 | **~2.4 分钟** |
| 有效速率 | ~51 tok/s | ~55 tok/s(无差异) |

当日汇总:ON 3/3 失败(含一次全新会话)vs OFF 1/1 干净完成。

**裁定**:

1. §8.1 的"无系统性质量退化"在**静态矩阵口径下依然成立**——分叉率、
   margin 分布确实不超噪声地板;但该口径测不到的分布尾部才是要害:fp16
   归约序扰动(~3e-5~5e-4)使近并列 argmax 翻转概率升高,翻转**落在工具
   调用参数 JSON 里**时代价是灾难性的(参数缺失 → 工具报错进上下文 →
   agent 自疑螺旋),一次翻转即可烧 5-10 分钟。静态矩阵每 tag 只有 1-2 个
   采样点,而 agentic 会话一次任务要生成数万个"必须精确"的 JSON 结构
   token,尾部风险被采样密度放大。
2. **生产回 OFF**(2026-11-09 compose 已改,env `=0`):agentic 工具链
   稳定性 > +13~18% prefill。
3. grouped 降级保留为**离线/非工具负载开关**:纯续写、批量探针、
   needle 类无 tool-call 负载可安全吃 prefill 红利(§8.1 矩阵与 262k
   锚点复刻 2,654-2,871 tok/s 均为其背书)。
4. 鹈鹕几何悬案(两臂脚不踩踏板)独立存在——A/B canary 走 /generate 无
   工具也踩空,属任务难度/模型几何能力,与开关无关,另案处理。

可选定案实验(未执行,不阻塞):再切 ON 重跑同 prompt 一次,复现螺旋即
因果完全闭环。如未来要找回该收益,正确路径是 §8.1 遗留②的确定性治理
(内核选择固定/batch-invariant)先行,而非直接开 flag。
