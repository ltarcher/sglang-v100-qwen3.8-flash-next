# PREFILL 杠杆立项书(L2 / L3)

立项日期:2026-11-07。状态:**已立项,未开工**(本文件即立项书;开工前先读本文对应节的门槛与坑)。

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

| 线 | 现状 | 预期(立项假设,待 L2-0 修正) |
| --- | --- | --- |
| Qwen 冷长文 prefill | 首 chunk 后全走 Triton 稀疏(29.5ms/层,FMA) | **+25~45%**(3,080→4,000-4,500 tok/s;190k 冷首 turn 62s→42-50s)。1Cat 同模型族 64K 纯 prefill +29.9% |
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

**L2-a(spike,天级,TileLang 内)**:两问一次答:
- (a) 预排序稠密页表(离线 gather 好、零每-tile 间接)能否让现 TileLang 核破
  17-19T?能 → 发射器不是唯一地板,planner-lite 在 TileLang 有产线;不能 →
  钉死必须 CUDA。
- (b) **分组在两条线的 smem 可行性不对称**:Qwen(D256,Hq6)8×6=48 行 Q_shared
  24KB + panel 32KB,96KB 内宽松,**分组可行**;GLM DSA(H=16,D=512)8×16 行
  Q_shared = 128KB **超 Volta 96KB**——GLM 侧分组需换形态(2-4 行小组、或只吃
  planner 不吃分组),spike 必须带上这个预算账。
- 判据:任一形态破 25T → 走 TileLang 产线;全灭 → L2-b 按 CUDA 移植立项。

**L2-b(产线)**:按 L2-a 结果二选一。Qwen 线倾向移植 1Cat CUDA 路径
(BSD-3/Apache 可借源),需改配我们的 page-16 fp16 池与 sglang metadata
(他们绑 Page4 微块 + E4M2,不是拿来即用);GLM 线按 spike 结论 TileLang 或
CUDA。改动遵守 fork 纪律:上游文件最小可 rebase,行为开关走 `SGLANG_*`
(environ.py 约定,读 env-var-conventions skill),serve 脚本注释与 README
性能表同步。

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

- **#754 稀疏 retention 边界调度**(1Cat):816-token GDN state block 把 prefill
  预算切成多次 mini-forward,scheduler 只在被保留状态边界停 → Flash-Next
  prefix-on 3,046→5,068 tok/s(+66%)。我们工具轮 0.34-0.61s/500-900 token
  (≈1,000-1,500 tok/s,远低于 3,000 稳态)是同病理指纹。纯调度、零内核,
  与 L3 同族;先量我们的 mamba state block 是否切碎 chunked prefill 再立项。
- QPN fragment 预排 decode GEMV(认命发射地板、打 DRAM 饱和):#37 账本
  wmma skinny dense 仅 20% 份额,低于 L2/L3,继续挂账。

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
- **遗留**:生产启用需在 compose/serve 注册 `--enable-mixed-chunk`(L3-1
  预算控制器落地时一并定参);`_original_forward_mode` 的 DSV4 paged 路径
  回退仅在有混批流量的 DSV4.1 部署里生效,当前 DSV4.1 生产未开混批,无影响。
