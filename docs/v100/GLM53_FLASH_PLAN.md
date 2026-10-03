# GLM-5.3-Flash 落地方案(计划,待评审)

状态:**草案 v1,待确认后开工**。目标硬件:4× V100-32GB(128G 显存)+ ~120G host RAM + NVMe。
硬约束:任何基础设施改动不回归 Qwen3.8 NVFP4 验收数字(见 AGENTS.md 数值验证纪律)。

## 0. 结论摘要

- **架构不用移植**:fork 基线已同步到上游 2026-09-07,上游已合入 GLM-5.3-Flash 全套实现
  ——`models/glm5_next.py`(1738 行,含 mHC、KDA、DSA、MoE)+ `models/glm5_next_nextn.py`
  (MTP 头)+ `configs/glm5_next.py` + `attention/dsa/` 下完整 kpool indexer
  (`dsa_backend_kpool.py`、`dsa_indexer_kpool.py`、`kpool_plan.py`、`kpool_fp8_index.py`)。
- **工作 = 五类 V100/NVFP4 适配**:fp16 audit(Volta 无 bf16,上游代码默认 bf16)、
  attention 内核 sm70 化、NVFP4 权重映射(modelopt → Marlin SM70)、专家 spill
  (182G 权重 > 128G 显存)、MTP/缓存端到端验证。
- **oracle 用 llama-glm5 fork**(`/data/develop/llama-glm5`,已含 glm5next + MTP),
  同输入 logits/文本对拍;mini-GLM 工件用于单测与骨架冒烟。
- **资源是第一设计约束**:NVFP4 权重 182G,TP4 每卡 45.5G > 32G 显存——**必须专家
  spill**(DSV4.1 线已有 host placement/budget/spill 基建可复用),解码性能上限由
  PCIe/RAM 带宽与专家缓存命中率决定。

## 1. 现状盘点

### 1.1 已就绪(直接复用)

| 组件 | 位置 | 状态 |
| --- | --- | --- |
| GLM-5.3-Flash 模型定义 | `python/sglang/srt/models/glm5_next.py` + `glm5_next_nextn.py` | 上游代码,无 V100 痕迹 |
| KDA 线性注意力内核 | `sglang.kernels.ops.attention.fla.*` + `hybrid_linear_attn_backend` | qwen3.8-flash-next 同族内核,V100 已验收 |
| mHC 内核 | `sglang.kernels.ops.layernorm.mhc`(hc_pre/post/contract) | sm70 兼容性**待验证** |
| DSA + indexer + kpool | `python/sglang/srt/layers/attention/dsa/`(全套) | 上游代码;内核多为 triton/tilelang/CUDA,**sm70 化是最大风险项** |
| V100 全注意力后端 | `flash_attn_v100_backend.py` | fork 已有,DSA 退化路径可直接用它 |
| NVFP4 → Marlin SM70 | `layers/quantization/modelopt_quant.py` + `marlin_utils_fp4.py` | qwen3.8 同通路已验收 |
| MTP 基础设施 | qwen3.8 MTP(target/`--spec` 两模式)已验收 | glm5next NextN 结构同型 |
| 前缀缓存 + KDA 状态 | `RadixLinearAttention`、`mem_cache/mamba_radix_cache.py`、`mamba_checkpoint_pool.py` | 上游已有线性注意力状态进 radix 的机制 |
| 专家 spill/host 放置 | `mem_cache/dsv41_host_placement.py`、`dsv41_v100_budget.py`、DSV4.1 MXFP4 专家 spill | DSV4.1 线已验证的机制,GLM 需参数化 |
| 数值 oracle | `/data/develop/llama-glm5`(`glm5next/upstream` 分支 + MTP) | GGUF 转换器 `conversion/glm5next.py` 含全部元数据校验 |

### 1.2 缺口(本计划要补的)

1. **fp16 audit**:glm5_next 上游按 bf16 写(config `dtype: bfloat16`);Volta 无 bf16,
   模型代码 + mHC/FLA/DSA 内核全链路 `SGLANG_SM70_FORCE_FP16` 语义下数值正确。
2. **DSA/indexer/kpool 内核 sm70 化**:上游 dsa 内核假设 sm80+;需逐内核评估
   (triton 可编译 sm70 / 需改写 / 退化),长上下文(>2051,indexer 激活)才需要。
3. **NVFP4 映射核对**:modelopt checkpoint(120 个 safetensors 分片)的量化范围需精确
   核对——`config.json` ignore 列表排除了全部 attention 投影、conv1d、indexer、moe gate、
   dense/shared MLP;实际量化张量集合与 `modelopt_quant.py` 的 Marlin 重打包路径要对上。
4. **专家 spill 参数化**:182G 权重的 host/VRAM 放置方案(DSV4.1 机制是单模型假设)。
5. **MTP + KDA 状态回滚**:glm5next NextN 头是 DSA 层 + plain 残差 + 无 mHC;
   draft 被拒时 34 层 KDA conv+ssm 状态回滚需验证(qwen3.8 MTP 无此问题——需确认
   其 linear 状态回滚机制是否已覆盖 KDA)。
6. **serve 脚本与文档**:`scripts/serve_glm53_flash_v100.sh`、README 性能表登记。

## 2. 资源预算(设计基线)

| 项 | 数值 | 说明 |
| --- | --- | --- |
| NVFP4 权重总量 | 182G | 磁盘实测(safetensors×120) |
| 其中 routed experts | ~150G | 288 专家 × 3 矩阵 × 42 sparse 层,NVFP4 |
| 非专家权重(fp16) | ~25–30G | attention 投影、indexer、dense/shared MLP、embed(154880×4096)、lm_head |
| TP4 静态切分 | 45.5G/卡 | **超出 32G 显存 13.5G/卡 → 专家 spill 是必需项,不是优化项** |
| 显存分配草案/卡 | 权重 26G + 激活/KV/图 4G + 热专家缓存 2G | 精确数字在 M3 按 dsv41_v100_budget 方法定 |
| host RAM | 权重余量 ~60–90G + KV offload + spill 缓存 | NVMe 作冷层(全量 182G 放不进 VRAM+RAM) |
| 解码带宽下限 | 每 token 唯一专家集 ~4.2G(NVFP4) | 8 专家 × 42 层 × ~12.6M 参数;PCIe ~25GB/s → 单流 decode 理论上限 ~25 tok/s,实际取决于专家缓存命中 |
| DSA 层 KV | ~11KB/token(11 层 × 512 lora × fp16) | 262k ctx ≈ 2.9G/序列,另加 kpool index cache |

初版性能门槛(待评审,非承诺):decode ≥ 5 tok/s/流(功能可用)→ ≥ 20 tok/s/流(调优后);
8k warm prefill ≥ 200 tok/s;MTP accept ≥ 2.0。达成多少登记多少,不外推。

> M2 实测修订(safetensors 头全量统计,2026-09-29):总量 181.3 GiB;routed experts
> 163.3 GiB(90.1%,比上表估计高);attn/dense/shared 15.65 GiB;embed+lm_head
> 2.36 GiB。TP4 每卡 45.3 GiB(其中专家 40.8 GiB);专家总量 163.3 GiB > host RAM
> 120G,NVMe 冷层同样是必需项(DSV4.1 476G 同款约束)。

## 3. 里程碑

### M0 — 基线固化与 oracle(约 1–2 天)
1. 固化 Qwen3.8 回归基线:跑 `smoke_v100.sh` + 验收命令,数字存档到本文档附录(后续每个
   里程碑完成后重跑对照)。
2. llama-glm5 构建(CUDA sm70 或 CPU)+ `conversion/glm5next.py` 转换
   `/data/models/GLM-5.3-Flash-NVFP4` → GGUF(保留 NextN,跳过视觉塔),短/长上下文
   (≤2051 与 >2051,触发/不触发 indexer)各固化 3–5 组 prompt 的 logits/文本 fixtures。
3. 造 mini-GLM 工件:同架构缩小版(如 4 层 = 3 KDA + 1 DSA、16 专家、hidden 1024)的
   safetensors + config,fp16 与 modelopt NVFP4 两个版本;llama-glm5 同样转换作 mini oracle。
   用于 M1/M2 的单测与骨架冒烟(真模型 fp16 全量 ~640G,任何机器都放不下,骨架必须用 mini)。

### M1 — mini-GLM FP16 骨架端到端(约 3–5 天,风险:内核兼容性集中在此暴露)
- 范围:glm5_next 全链路 fp16 audit;fla KDA(glm5 差异:plain sigmoid 门、
  `gate_lower_bound=-5`、L2 norm eps 1e-6)在 sm70 编译运行;mHC 三内核 sm70 验证;
  DSA 层先退化全注意力(`flash_attn_v100_backend`);MoE noaux_tc sigmoid + shared 直加;
  `RadixLinearAttention` 状态通路。
- 验证:mini-GLM fp16 在单卡(或 CPU 小配置)端到端生成;logits 与 llama-glm5 mini oracle
  逐层/端到端对拍(小上下文 cos > 0.999 量级);新增单测进 `test/registered/`。

### M2 — NVFP4 权重通路(约 2–4 天)
- 范围:核对 modelopt 分片实际量化张量集(解 `model.safetensors.index.json` 枚举 dtype);
  确认 glm5_next 张量名 → `modelopt_quant.py` Marlin SM70 重打包映射(含
  `swap_weight_nibbles` 与 scales 布局语义);显存不足的 expert 层接 spill 元数据路径。
- 验证:①转换器级 NVFP4 块 dequant vs modelopt 公式(`fp4 × scale × weight_scale_2`)
  余弦 > 0.9999;②mini-GLM NVFP4 端到端 logits vs mini oracle fp16(可容忍量化误差,
  判据为困惑度/文本一致而非逐 token 相等);③Qwen3.8 NVFP4 回归不回归。

### M3 — 真模型端到端(功能跑通,允许慢;约 3–5 天)
- 范围:真模型 NVFP4 + 专家 spill 在 4×V100 拉起(退化 DSA);serve 脚本
  `serve_glm53_flash_v100.sh` 初版(资源参数按第 2 节草案);spill 机制从 DSV4.1 参数化。
- 验证:短上下文(≤2051)生成文本与 llama-glm5 oracle 语义一致;smoke 零输出检测必过;
  记录首版 decode/prefill 实测(不设门槛,只记录)。
- 本步允许性能很差(NVMe 冷专家),只求功能正确与资源不 OOM。

### M4 — DSA + indexer/kpool 完整路径(约 1–2 周,最大技术风险项)
- 范围:逐内核评估 `dsa/` 在 sm70 的可行性(indexer 线性投影、kpool 池化、top-k、
  MLA 吸收路径),triton 内核优先编译验证,不可用的改写或退化;indexer weights 全程
  F32 累加(llama-glm5 同款红线);>2051 上下文激活 indexer。
- 验证:同一 prompt 在 ≤2051(全注意力)与 >2051(indexer)两模式输出一致;
  长上下文 logits 与 llama-glm5 oracle 对拍;前后缀注意力退化对比(quality sanity)。
- 若 kpool 在 sm70 不可行:降级方案为 fork 的 CSA2 路径改造适配 glm5next DSA
  (DSV4.1 已验证的 sm70 稀疏注意力),kpool 缓存语义保留在元数据层。

### M5 — MTP 接入(约 3–5 天)
- 范围:`glm5_next_nextn.py` 头(NextN = DSA 层 + plain 残差 + 无 mHC)接现有 spec 通路
  (qwen3.8 已验收的 target/`--spec` 两模式);draft 拒绝时 KDA conv+ssm 状态回滚验证
  (34 层 × ~2MiB/层);NextN 自己的 DSA/kpool cache 与主模型同步。
- 验证:两模式输出一致性;accept 长度实测(参考 sglang 上游 GLM 数据与 qwen3.8 的 ~2.1);
  长草稿/高拒绝率压力下无状态漂移(连续生成输出与 target-only 一致)。

### M6 — 前缀缓存与性能验收(约 1 周)
- 范围:radix 前缀树 + KDA 状态快照(`mamba_radix_cache`/`mamba_checkpoint_pool` 机制,
  GLM 状态 ~2MiB/层 × 34 层 ≈ 70MiB/序列,只在分支点存副本,进 budget 统一管理);
  hicache 文件分层(storage dir 必须指向磁盘);专家缓存命中率调优(host placement、
  热专家驻留策略);性能验收与 README 登记。
- 验证:共享系统提示多请求 `cached_prompt_tokens` 提升 + 输出与全量 prefill 一致;
  KDA 状态快照 eviction 后恢复正确;最终性能表(硬件、命令、数字)写入 README 与本文档附录。

## 4. 风险与对策

| 风险 | 等级 | 对策 |
| --- | --- | --- |
| dsa/kpool 内核 sm70 不可行 | 高 | M4 首日做可行性探针;降级路径 = CSA2 改造(fork 已验证的 sm70 稀疏注意力) |
| fla/mHC 内核 fp16/sm70 数值问题 | 中 | M1 集中暴露;mini 模型逐内核单测,与 llama-glm5 中间量对拍 |
| 182G 权重放置与 spill 性能 | 高 | 第 2 节预算先行;复用 DSV4.1 spill/budget 基建;性能目标分阶梯,达成多少登记多少 |
| modelopt 量化张量范围与映射不符 | 中 | M2 第一步就是枚举实测,不猜;对照 qwen3.8 同通路 |
| 上游 glm5_next 代码持续演进冲突 | 低 | 锁定当前基线;改动尽量以 env 开关/增量文件表达(AGENTS.md fork 纪律) |
| MTP 下 KDA 状态回滚错误 | 中 | 单测覆盖(拒绝 0/1/2/3 token 各分支);连续生成一致性检查 |

## 5. 验证总策略

1. 每个里程碑完成即重跑 Qwen3.8 NVFP4 回归(smoke + 验收数字),不回归才合入。
2. 数值对拍链:内核级(mini 单测)→ 模型级(mini-GLM)→ 真模型(oracle fixtures)。
3. 性能声明一律 4×V100 实测,记录硬件、命令、可解读汇总;单卡结论不外推。
4. 提交纪律按 AGENTS.md:仅按要求 commit,主题 ≤72 字符,正文带验证证据。

## 附录 A:M2 实施记录(NVFP4 权重通路,已完成 2026-09-29)

- 量化范围核对:routed experts 独占 NVFP4(`weight` U8 + `weight_scale` E4M3/16块 +
  `weight_scale_2` F32),attention/dense/shared/router 全 fp16,与配置 ignore 清单一致。
- SM70 Marlin 通路:`modelopt_quant.py` `ModelOptNvFp4FusedMoEMethod` 在
  is_sm70+use_sm70_marlin 时强制 MARLIN runner;`_repack_nvfp4_weight`
  (int32 view→transpose→`gptq_marlin_moe_repack`)+
  `sm70_nvfp4_marlin_process_scales/_global_scale` 全链内核级验证通过。
- **关键缺陷与修复(权重 scale 折叠溢出)**:gate/up 各带 per-tensor
  `weight_scale_2`,旧实现折叠 UP(块 scale × up/gate > 1)使接近 448 上限的 e4m3
  块 scale 溢出——e4m3 无 inf,溢出 cast 直接得 NaN。真 checkpoint 实测 L2/expert
  {3,4,5}、L3/expert3 溢出(块 scale 峰值 528),凡路由命中即 NaN logits,端到端
  表现为 token id 塌缩 0..9;checkpoint 本身 0 NaN。修复:折叠 DOWN,
  `s2_eff = max(gate, up)`,两半各乘自身 ≤1 比例并 `clamp_(max=448)`,
  `w13_scale2 = s2_eff`;数值上仅损失一次 e4m3 重舍入(半 ulp ≤ 6.25%)。
- mHC 兜底:`model_hook.py` 对 Glm5NextForConditionalGeneration + SM70 自动置
  `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=0`(TileLang mHC 仅 bf16;此前 fp16 serve
  需手工环境变量,现 serve 不再需要)。
- 验证(全部实测):
  - 内核级:注册测试 `test/registered/unit/layers/quantization/test_sm70_nvfp4_marlin_moe.py`
    (fold 合同 3 项 + 合成权重 kernel vs dequant 参考,M∈{1,4,64})全过;
    真 checkpoint 张量扫描 `scripts/m2_real_ckpt_unit.py` max diff
    7.6e-06 / 8.85e-04 / 1.37e-03(M=1/4/64)。
  - 端到端:注册测试 `test/registered/e2e/models/test_glm53_mini_sm70.py`(fp16)与
    `test_glm53_mini_nvfp4_sm70.py`(NVFP4)全过——NaN 塌缩签名(生成 id 多样性 +
    logprob 有限)通过;质量门为无秩依赖质量判据:峰值 |Δlp| ≤ 0.30 + 共享 top-64
    sym-KL ≤ 0.02(随机权重分布近平坦,逐 token 命中门不成立,已弃用)。
  - 量化位移:nvfp4↔oracle sym-KL 0.0025..0.0050 ≈ fp16↔oracle 0.0032..0.0038,
    routed experts 量化不引入可测分布位移。
- 遗留:Qwen3.8 NVFP4 回归(受宿主内存限制,串行补跑)。

## 附录 B:P1.6 常驻半区频率重分区 A/B(2026-10-01)

常驻 144 专家的成员从静态 `[0,144)` 改为按路由频率 profile 选
(`SGLANG_DSV41_EXPERT_SPILL_COLD_SET`,缺省关;表由
`scripts/glm53_cold_set_from_route_probe.py` 从 ROUTEPROBE 语料生成,571 前向,
工件 `scripts/glm53_cold_set_v1.pt`)。完整记录见 MoE4All
`docs/glm53-route-profile-a3.md`;要点:

| 指标(4×V100,mtp arm) | tail 落位 | v1 表 | Δ |
| --- | ---: | ---: | ---: |
| warmup page-in 字节(同 252 calls) | 9.12 GiB | 4.36 GiB | **−52%** |
| 真文本 decode C=1 / C=2 聚合 | 2.7 / 2.7 tok/s | 4.1 / 4.3 tok/s | **+52% / +59%** |
| accept(mtp) | 2.43–2.49 | 2.43–2.49 | 持平 |
| padding decode C=2(注册协议) | 28–30 tok/s | 9.8 | **−66%** |
| prefill 1k / 4k | 114–123 / 329–335 | 116–123 / 329–335 | 中性 |

- 真文本 spill 份额 0.48 → 0.35(A3 预测 0.14,域差收窄);均匀流量是
  profile 的构造性最坏情形(两臂均 ~0.50),padding 协议回归如实登记。
- 机理:page-in 为**逐层延迟受限而非带宽受限**(字节 −52% 而 C=1 步进速率
  不变)→ decode 真杠杆是 P2 整层 banking;本项字节减半进入 P2 预算。
- 实现修复:`FusedMoE` loader 曾把 remap 后的 slot id 传给 scale 加载函数
  二次重查落位表,非恒等 cold-set 下错写 mirror 行(恒等落位幂等故此前的
  校验全绿);现落位只解析一次并显式下传。
- 登记:`SGLANG_DSV41_EXPERT_SPILL_COLD_SET` 已在 `environ.py`(WO-13 D2)
  注册;GLM README 引擎段落与生产 serve 脚本待引擎结项时一并落地,当前
  经 dev 启动器 `scripts/launch_p16_ab.sh` 使用。

## 附录 C:P2 整层 banking + 真瓶颈判定(2026-10-01)

### C.1 已落地并验证

| 项 | 状态 | 证据 |
| --- | --- | --- |
| P2 整层 banking(`SGLANG_DSV41_SPILL_PREFILL_BANK`) | 机制正确,mini A/B bit-exact | 动态页入 54→2.0 GiB/30s(−96%);每 forward 42 次 fill 全部由前模块 prefetch 隐藏;零写回;126 fills/请求 = 42×3 chunk |
| MHC Sinkhorn JIT(`SGLANG_SM70_MHC_SINKHORN_JIT`,默认开) | 参与确认,数值正确 | 直接驱动 `hc_pre`(GLM 几何,fp16):JIT=1 时 11 launches/层 vs eager 139(−92%),`split_sinkhorn_kernel` 每层命中;layer_input/post_mix bit-exact,comb_max|Δ|=1.19e-7(fp32 归约序差);mini 端到端 on/off bit-exact |
| decode 抽检(256 tok,单流,temp 0,prefix 复用) | 不劣化,似有提升 | 5.12 tok/s、accept 2.53(P1.6 参照 4.1/2.43;协议不同,只作不回归证据;机理:decode graph 每步少 ~11k replay 节点) |

### C.2 三次证伪(P2 原 premise 链全部不成立)

1. **"页入是 prefill 瓶颈"**:bank 把动态页入砍掉 96% 且全部隐藏后,wall 不动
   (79.1 vs 80.2,PROBE=1)。
2. **"NCCL 协议(TREE_LL)拖慢"**:LL128 臂 76.2,更差;弃。
3. **"MHC 启动汤(139 launches/层)饿死流水线"**:JIT 上线后 launches −92%
   而 anatomy wall 不动(PROBE=1 76.1 vs 基线 80.2;PROBE=0 诚实数 78.6)。
   启动汤是 host 工作量的真实组成,但不是墙。

另:ROUTEPROBE 的逐 MoE 调用阻塞 D2H 只占 ~3%(PROBE=0 78.6 vs PROBE=1 76.1)。

### C.3 真瓶颈(4×V100 实测,in-server GPU-only torch profile)

GPU **从不空闲**(busy 93.5s > wall 61s,>0.3s 的空隙为零)——瓶颈不是重叠失败,
是 copy 引擎本身:

| 项 | 实测 |
| --- | --- |
| H2D pinned 拷贝 | **56.3s/61s 窗口 = copy 引擎 ~100% 忙**,总 128.4 GB / 2 请求 = **~64 GB/请求**(4096 tok 归一 ≈111 GB/卡,与调研 115 GB 估算吻合,2.8× 最小字节冗余) |
| 拷贝节律 | 0.207s 周期的 302 MB(cur bank)+151 MB(nxt bank)+18.9 MB(scales)三元组;504 个大拷贝 = 42 模块 × 2 bank × 3 chunk |
| 有效带宽 | 每拷贝 2.3 GB/s 稳定(302 MB/126 ms),**4 卡并发合计 9.1 GB/s** |
| 拓扑 | GPU Gen3 x16(7.9 GB/s)→ switch **Gen4 x16 上行(~13.5 GB/s 实效)**;4 路并发摊分 ≈3.4 GB/s/卡为硬件上界,实测 2.28 = 67%(余量被 NCCL P2P 分走) |
| 计算 | Marlin MoE 仅 1.6s/61s;linear-attn 4.3s;NCCL TREE_LL 27.4s(与拷贝重叠,含 spin) |

**结论:chunk 时间 ≈ spill 导入时间。1024-token chunk 需 42 模块 × ~453 MB =
19 GB 导入,2.28 GB/s 下 ≈8.3 s,与实测 chunk wall 9-10 s 吻合。路由在
1024 token × topk 下触及几乎全部 144 个 spilled 专家,逐模块整层导入不可免,
唯一杠杆是字节与驻留。**

### C.4 对后续里程碑的重排

- **P3(2-bit)优先级上调**:字节减半 → prefill 上界 ~2×(~160 tok/s @ 现带宽,
  ~340 @ 3.4 GB/s 摊分上界);decode 同比例受益。
- **P4(2-bit 全池常驻)premise 恢复**:全池 2-bit = 288×2.1 MB×42/2 =
  **12.7 GB/卡,可驻 VRAM**(现状 4-bit 驻留半区已占 12.7 GB)→ 零导入,
  prefill 转算力界(调研 1.6-2.6 s/8k),≥800 重新成为合法门槛。
- P2.5(MTP 宽度)不变;MHC JIT 保留(正确性 + host 效率 + decode replay 减负)。

### C.5 登记与文件

- `SGLANG_SM70_MHC_SINKHORN_JIT` 已入 `environ.py`(§sinkhorn-jit 块);
  `model_hook.py` SM70-GLM 块注释同步;`launch_p16_ab.sh` 增 PROBE 旋钮。
- 测量工件:`/tmp/p2_gate_probe.py`(参与探针)、`scripts/p2_prefill_anatomy.py`
  (cold-prefix anatomy)、in-server `/start_profile`(GPU-only,注意 CPU+GPU
  双活动 4 rank 导出会 OOM-kill 调度器,勿再踩)。

## 附录 D:P2.5 MTP 代价感知宽度(2026-10-01)

### D.1 先决修复:中途换宽把 draft 图输出变成悬垂指针(撞墙事故)

- 症状:adaptive 臂(GLM)mid-request 换宽后连续 12 条 `invalid eagle tree!!!
  ... logprob has nan` 告警,decode 活锁;同负载的对照臂(无 adaptive)零告警
  → 负载无辜,换宽路径有罪。日志全史只有 1 次换宽事件,12 条告警全部紧随其后。
- 根因:`draft_forward` 返回的 topk1 chain 缓冲
  (`_topk1_parents_prealloc`/`_topk1_score_indices_prealloc`)由
  `_rebuild_topk1_chain_buffers` **每次调用都新分配**;而
  `full_cuda_graph_backend.capture_one` 把返回张量记录为图输出——每个宽度的
  capture 都把当时的缓冲指针烤进图里,下一次 rebuild 就把它孤儿化。换宽后首次
  replay 读到的是被分配器回收的内存块:内容恰好还是旧 arange 则侥幸正确
  (宽度 3 之前一直活着纯属此运气——其悬垂块与存活宽度 5 的 refill 别名且首列
  同值);被服务流量重用则 parent_list 变垃圾 → `build_tree_kernel_efficient`
  父查找失败(告警里的 nan 是红鲱鱼)→ retrieve_index 错乱 → KV/mamba 提交腐坏
  → 活锁。Qwen 上同为潜伏炸弹,与 cost-aware 无关。
- 修复:`EagleDraftWorkerBase._topk1_chain_storage` 按宽度持久化
  (分配一次、原位 refill;refill 值是确定性 arange,值上无变化),图不再指向
  孤儿内存。
- 验证:`test/registered/unit/spec/test_topk1_chain_buffers.py`(3 例,已验证
  在修复前代码上失败);修复后同负载 4 次真实换宽(3→5→3→1→3→1)零
  invalid-tree 告警,decode 全程存续,换宽后 probe 输出连贯。

### D.2 代价感知宽度策略(Strata DraftPolicy 移植)

每宽度维护 cost_ms EMA(α=0.1,CUDA event 实测 verify round)与 accept_len
EMA(α=0.05,bonus 含);决策 argmax(tokens/ms),3% margin 粘滞;未测宽度打分
走先验(cost 按实测宽度形状缩放,accept 用 1+0.7(t−1));先验排名第一且几乎未测
的宽度强制 probe `_PROBE_CT=3` 轮。入口:`SGLANG_ADAPTIVE_SPEC_COST_AWARE=1`
+ `--speculative-adaptive` + `--speculative-adaptive-config`;单位测试
`test_adaptive_spec_params.py`(41 例)。

### D.3 测量老化(A/B 暴露的缺陷修补)

- 缺陷:随机 token 污染后,width-3 的 cost 估计被钉死在低位,真实文本上永远
  排不进 probe → 卡在 width 1(accept 2.00;单流 3.25 / x4 3.19 tok/s,污染前
  width-3 稳态 ~3.5/3.46)。
- 机制:每轮衰减——本轮实测宽度的计数按 `ct ← ct×0.98 + 1` 折入(+1 与衰减同式
  抵消,新探测宽度的首个样本 ct=1.0 恰好活过擦除线),其余宽度 ×0.98;ct<1 整组
  擦回先验。持续被测的宽度停在 1/(1−decay)=50 平衡态,永不被擦;被弃宽度闲置
  ~120–140 轮降到 `_PROBE_CT` 以下恢复 probe 资格,~195 轮全擦。
- 陷阱:首版实现"先 +1 再全体 ×0.98",首个样本 1.0×0.98<1 当轮自擦,任何宽度
  永远攒不下测量——wiring 单测(`test_round_ms_reaches_the_routed_cost_slot`)
  当场抓获,+1 必须与衰减折进同一表达式。

### D.4 3 臂 A/B(同机 4×V100,`scripts/p25_decode_bench.py 256`)

| 臂 | 单流 tok/s | accept | x4 聚合 tok/s | 备注 |
| --- | --- | --- | --- | --- |
| cost-aware(sweep 前) | 3.61 | 3.879 | 3.63 | 收敛段 accept 3.938;全程零换宽(width 3 实测分 3.94/1.7 压制 width 5 先验 3.8/2.45,probe 永不触发);trigger 4 次换宽零告警,probe 连贯且策略回到 width 3 |
| acceptance-only adaptive | 3.81 | 5.447 | 3.82 | 开局即换 width 5 并坐稳(converge accept 5.6–5.9);本 bench 文本高度可预测,accept 近宽度不敏感(3→3.9,5→5.9),acceptance 先验无此假设、反而测到了真值 |
| off(固定 width 3) | 3.55 | 3.879 | 3.57 | 与 cost-aware 同宽度同 accept,差 ≤2% → adaptive 机制本体(事件测量 + 策略)开销可忽略 |
| **cost-aware(sweep 后,定稿)** | **3.83** | **5.02** | **3.72** | sweep 舞步 3→5→3→1→5 后坐稳 width 5;converge 2 = 4.05/5.626 与 acceptance 臂持平;x4 差 0.10 是 sweep 成本摊在首个 256-token 窗口;era 内 35 次换宽零 invalid-tree 告警 |

**判读**:adaptive 机器不亏;真差距在宽度——本 bench 文本 accept 近宽度不敏感
(width 3→3.94/4.0,width 5→5.9/6.0),width 5 净赚 ~6%。acceptance 策略靠实测
找到了它;cost-aware 的 accept 先验 1+0.7(t−1)(width 5 → 3.8)系统性低估
高可预测文本,先验排名永远进不了 probe → 修法:冷启动 sweep(见 D.5),修后
cost-aware 稳态与 acceptance 持平且保留代价感知(垃圾流量下 acceptance 无回撤
能力——D.1 trigger 实测随机 token accept 1.03–1.06,而换宽日志里 width 5 的
round cost ~1160–1460 ms vs width 1 ~713–832 ms,≈1.6× 纯浪费;cost-aware
测完即回窄宽)。

### D.5 冷启动 sweep(A/B 判读的直接产物,已定稿验证)

probe 规则从"先验排名第一才 probe"改为"凡测量数 < `_PROBE_CT` 的候选一律补测"
(当前宽度优先测完,其余按先验分降序),argmax 只在全部候选测满后裁决。动机:
accept 先验 1+0.7(t−1) 在高可预测文本上系统性低估(width 5 先验 3.8 vs 实测
5.9),先验排名永远轮不到 width 5 → cost-aware 整场 workload 慢 6%。sweep 成本
有界(每宽度 ≤3 个 verify round),且与老化衔接:被老化擦除的宽度自动重获 sweep
资格。单位测试 +2(sweep 补测先验落败宽度;probe 未测满不弃),套件 43 例全过;
硬件复测(D.4 末行)证实 sweep 后 cost-aware 稳态追平 acceptance 臂。

### D.6 登记与文件

- `SGLANG_ADAPTIVE_SPEC_COST_AWARE`(`environ.py`,含 sweep + 老化语义注释);
  `adaptive_spec_params.py`(策略 + 老化 + sweep);`base_spec_worker.py`(chain
  缓冲持久化);`scripts/p25_decode_bench.py`、`scripts/p25_switch_trigger.py`、
  `scripts/glm53_adaptive_p25.json`。
- 单位测试:`test_topk1_chain_buffers.py`(3)+ `test_adaptive_spec_params.py`
  (43,含 wiring 回归与老化/sweep 语义)。
- 教训入册:mid-request 换宽在 GLM 上从未被真实负载踩过(A/B 前全史仅 1 次
  换宽),spec 路径的"换配置后旧指针是否仍被图引用"要作为固定审查项。

## 附录 E:P3 2-bit 质量门(2026-10-01,go/no-go 证据)

P4(全池 2-bit 常驻)的前提是"NVFP4→Q2_K 再压一档"的质量代价可接受。P3 按
Strata D 档模板收集三件套证据:same-top1% / median+mean KL / PPL diff(teacher-forced,
512-token 块)+ needle 检索,并给出跨引擎参照带(部署中的 sglang NVFP4 GPU 引擎 vs
fp16 GGUF 参照),使 Q2_K 的 KL 有"引擎自身 CPU/GPU 位宽差"这条对照线可攀。

### E.1 测量管线

双引擎三臂,共用同一 teacher-forced token 流(`fixtures/p3_gate.txt`,3987 token
= 8×512 块,2040 个计分位;语料 = 计划文档散文 + README + adaptive_spec_params.py
源码,`scripts/gen_fixtures_oracle.py` 拼装):

| 臂 | 引擎 | 角色 |
| --- | --- | --- |
| reference | llama.cpp CPU,fp16 GGUF(NVFP4 expert 原样内嵌) | teacher logits(`--kl-divergence-base` 落盘) |
| candidate | llama.cpp CPU,Q2_K(`--allow-requantize` 从 NVFP4 再量化) | D 档三件套直接测量 |
| deployed | sglang NVFP4(4×V100,`--spec NEXTN`,生产参数) | 跨引擎参照带 + needle |

脚本:`scripts/p3_llama_kl_gate.sh`(convert/quantize/base/candidate 四阶段)、
`scripts/p3_kl_summary.py`(base logits 解析 + sglang prompt-logprob 探针 + 汇总)、
`scripts/p3_needle.py`(多 needle 数字检索,1k/4k/16k)。

**deployed 臂 boot 矩阵**(probe 只测 prefill logits,decode 图与草稿模型都无关,
取最稳组合;逐坑实测):expert spill 与 decode CUDA-graph 硬耦合——spill 的
host-LRU 换入点(`ensure_spill_experts`)是 capture 非法操作,只能由 breakable
backend 在段边界承接(`serve_dsv41_v100.sh` 的
`--cuda-graph-backend-decode breakable`);FULL backend 下该调用直接进 capture
流 → `cudaErrorStreamCaptureUnsupported`。而 MTP verify 宽度(draft
tokens/req ≥ 2)又超不出 landing 池宽度公式(`_decode_shaped_max_tokens`,
`slots//8`),landing=12 时 capture_bs 被全部过滤 → 空列表崩溃。最终可用矩阵:
`SGLANG_DSV41_SPILL_LANDING=0` + `--cuda-graph-backend-decode disabled` +
**去掉全部 `--speculative-*`**(草稿权重与 spec 状态链不参与 prefill 分布)+
`--mem-fraction-static 0.88` + `--max-total-tokens 16384` +
`--max-mamba-cache-size 8`。后三项是 OOM 链的解:logprob 探针要 [512,154880]
fp32 全分布缓冲(~304 MiB),而 mem-fraction 是"池子吃满预算"的旋钮,池子永远
长到预算线(torch 之外还有 ~3.4 GB NCCL/context 开销),必须用 max-total-tokens
直接封顶 KV 池、max-mamba-cache-size 显式给槽(否则 mamba 槽被挤到
`max_num_reqs=0`)。生产 GLM+MTP+spill 的相容域(landing ≥ bs×width×8 +
breakable,或 2-bit 常驻消灭 spill)归 P4 处理。

### E.2 转换陷阱(NVFP4 checkpoint → GGUF,每条都花过一次 45 分钟重跑)

1. `convert_hf_to_gguf.py --mtp` 是"只导出 MTP 头"(`mtp_only=True`,
   `filter_tensors` 丢弃全部非 NextN 张量)——主模型 GGUF 必须 `--no-nextn`;
   NextN 张量单出 `mtp-glm53-f16.gguf`。困惑度上下文消费不了 in-file NextN
   (ctx_type 仅 draft 侧为 MTP),否则 `expected 1670, got 1664`。
2. GGUFWriter 默认 `use_temp_file=False`:~193G 张量全量驻留进程 RAM。
   必须 `--use-temp-file` + `TMPDIR` 指向可写磁盘目录(root-owned 目录会让
   tempfile 静默回落 /tmp)。
3. `n_experts` 查找键需含 `n_routed_experts`(GLM 命名),否则 per-(layer,proj)
   flush 不触发,fallback 会把 ~300G packed experts 全部攒在 RAM。
4. `gguf-py` `LocalTensor.mmap_bytes()` 用 `np.memmap(mode='c')`:每张量一个私有
   COW 映射,fault 过的页变匿名页;引擎驻留 96G host RAM 时仅剩 ~26G available,
   每个 fault 直进 direct reclaim——表现为 utime 冻结、stime 满核、VMA 数不变、
   RSS ~3.6MB/s 爬升的"假死"。修复 `mode='r'`(只读映射共享 page cache)后,
   blocks≤12 有界复现从无限假死变为 114.8s 完成。诊断手段:`/proc/<pid>/stat`
   的 utime/stime + VMA 计数采样(py-spy/perf 在本机不可用,ptrace yama=1)。
5. **首跑三件套作废事件(参照侧执行 bug,非量化问题)**:第一次 base/candidate
   得到 PPL(base)=229,783、Same top p=0.000%、KLD 2.84——signature 与 mini 随机
   权重模型一致(e^ln(vocab) 量级),即 fp16 参照在算噪声。根因:llama.cpp 的
   `block_nvfp4` 布局无 fp32 super-scale 位置,modelopt `weight_scale_2`
   (~4.65e-5)按设计存独立 `.scale` 张量、由 `build_lora_mm_id(..., w_s)` 在输出
   侧按 expert gather 相乘;`glm5next.cpp` 的 `build_moe_ffn` 短重载调用没传
   `ffn_{up,gate,down}_exps_s` → expert 输出大 21504×(=1/scale2),clamped
   SwiGLU(±10)全饱和 → 43 层后纯噪声。诊断:Python 双侧反量化核对
   (checkpoint modelopt 公式 vs GGUF block_nvfp4 布局),转码逐元素忠实、比值
   恒 21504× 一锤定音;mini(dense、无 MoE)的 oracle 对齐验证不到这条通路。
   修复后首 pass PPL 254,735 → 10.64。教训:**oracle 三件套首跑先验参照自身
   PPL 量级,再读 candidate 数字;凡走 NVFP4 checkpoint 的新模型 builder 都要
   确认传了 `*_exps_s`。**
6. **`--tensor-type` 对 F16 输出文件是死代码**:`llama_tensor_get_type` 的
   manual override 匹配整个包在 `if (ggml_is_quantized(default_type))` 里,
   `... --tensor-type ffn_up_exps=Q2_K ... F16` 静默不生效,把 560G expert
   按 f16 反量化写满盘才被发现。修复:manual 匹配提到守卫外(形状 fallback
   只对量化目标跑),`--dry-run` 先验 `applying manual override` 行数
   (=42 层×3 张量)再实跑。凡是"全局 ftype=高精度 + 按张量降级"的混合量化
   都撞这个 bug。

### E.3 质量门数字

参照侧 PPL(base)=7.46(1785 计分位);解码数学经独立重算闭环
(7.4599,`p3_kl_summary.py` 的 20 字节头 + 折减位索引两处解析 bug 修复后)。

| 指标 | sglang NVFP4(部署带) | 混合 Q2_K-experts(P4 保真) | 均匀 Q2_K(上界对照) |
| --- | --- | --- | --- |
| same-top1% | **91.04**(top-20 overlap 0.863) | 79.38 ± 0.96 | 73.11 ± 1.05 |
| median / mean KL | n/a(探针仅 top-20,无全分布) | 0.130 / 0.270 | 0.258 / 0.448 |
| PPL vs 参照 | n/a | **+10.5%**(×1.105,7.46→8.24) | +29.7%(×1.297) |
| Cor(ln PPL) | — | 95.67% | 93.12% |
| needle 1k | 5/5(两次独立 boot) | **5/5** | (未做) |
| needle 4k | 5/5(前次 boot) | n/a(CPU 臂 ~4 tok/s 不可行) | n/a |
| needle 16k | n/a(引擎 8k 上下文) | n/a(CPU 臂不可行) | n/a |

读法注意:部署带 = 跨引擎差(量化差 + 运行时差混在一起),是"可对照的外界线";
混合臂 = llama 同引擎自比,只含 expert 2-bit 的纯量化差,是 P4 的直接测量。
均匀臂把 attention 也压到 2-bit,夸大 P4 代价,仅作归因对照。
llama 参照臂(F16 GGUF)needle 1k 亦 5/5(`p3_needle_f16_1k.log`);混合臂五案
输出魔数与参照臂逐案相同(`p3_needle_q2kexp_1k.log`)。

### E.4 判读

**GO**(2026-10-01,证据齐后定稿)。

- **PPL 是主判据,且过线**:P4 的形态就是混合臂(experts 2-bit、dense 保持原精度),
  其纯量化代价 +10.5%(7.46→8.24),落在 KV-q4 +8–12% 这一部署上已接受的先例带内。
  均匀臂 +29.7% 不是 P4 的形态,仅证明代价主要来自 experts 本身(归因清晰)。
- **needle 全对**:sglang 1k/4k、llama F16 参照、llama 混合臂四处 needle 1k 均 5/5,
  且混合臂五案魔数与参照臂逐案一致——长程精确检索路径在 2-bit experts 下完好,
  量化损伤表现为分布展宽(KL 0.130/0.270),不破坏 sparse-retrieval 行为。
- **Cor 95.67%**(ln PPL):损伤是平缓的分布性退化,不是个别位置崩坏。

附带限定(P4 落地时仍然成立):

1. **top-1 翻转率真实存在**:混合臂 same-top1 79.4% ≙ 每 5 个位置约 1 个 argmax
   翻转。这是同引擎纯量化差;部署带 91.0% 是跨引擎带,二者不可直接比。推论:P4
   不得再叠加其它降精度机制——均匀臂(attention 也 2-bit)PPL +29.7% 已破带,是反例。
2. **长上下文 2-bit 质量未测**:全部三件套数字来自 512-token teacher-forced、4k
   fixture;needle 4k/16k 在 CPU 臂不可行,只能等 P4 做出真 GPU 2-bit 路径后补测,
   由 P4 自己的质量门承担。
3. **全分布数字只在 llama 侧成立**:部署带探针仅 top-20,无 KL/PPL。

## 附录 F:P4 SM70 2-bit 全池常驻设计(2026-10-01 开题)

前提(P3 E.4 GO + C.3/C.4 解剖):prefill 墙 = spill 导入 H2D,唯一杠杆是字节与
驻留;全池 2-bit = **12.7 GB/卡,预算中性**(现状 4-bit 驻留半区恰好同占 12.7 GB,
4-bit 全池 25.4 GB/卡装不下)。2-bit 常驻后 spill/page-in 整条子系统退出 GLM 路径。

### F.1 内核路径决策:扩展 SM70 Marlin MoE,新增 u2 实例化

实测证据(2026-10-01):已部署的 `_sm70_marlin_v100_moe.abi3.so` 导出的格式面为
`sm70_marlin_{u4, u4b8, u8, u8b128, fp8, mxfp4, nvfp4}_gemm`(`nm -D`),**无 2-bit**;
python 侧 `fused_marlin_moe` 亦 `assert num_bits in [4, 8]`。所以 P4 是新内核代码,
不是开关。

选定路径 = 沿用 SM70 Marlin MoE 架构,克隆一个 `sm70_marlin_u2_gemm.cu`。

可行性已核(2026-10-01,源码 = zhinianqin/marlin_v100 @ 6d72a49,pinned 同
setup_v100_marlin.sh;1cat-vllm 本地树为同族旁证):

- 部署链:python `moe_wna16_marlin_gemm` → `ops.cu` 按 `b_type` 分发 → 同文件内
  `sm70_marlin_{u4,u4b8,...}_gemm`。每格式一个自包含 `.cu`,共享层
  (`Sm70MarlinMoeGemmTraits`/`Sm70MarlinMmaPipelined`/epilogue)**只见 fp16
  fragment**——ThreadMap 就是 CUTLASS fp16 访问映射
  (`MmaCore::IteratorThreadMapB`,8 halves/访问),不携带任何打包格式假设。
  u4b8 的 `IteratorB` 是骨架:uint32 → 每字节 `marlin::dequant<half2, id>` 出
  2×half2,乘缓存的 half2 group scale 进 Volta mma 流水。
- u2 特化清单(全部 per-format 或纯新增,零共享 mainloop 改动):
  1. `csrc/quantization/marlin/dequant.h`:加 `kU2` half2 特化(每字节 4 码 →
     同形状 2×half2;LOP3+hsub2,SM70 只需 fp16 面);
  2. `csrc/quantization/marlin/sm70_marlin_iterator_utils.cuh`:加
     `u2_packed_macro_n_qweight_offset_from_logical`(每 64 列 tile 的 qword 数
     减半;16 码/字,每访问 8 码 = 半字,迭代器缓存整字分两次发射);
  3. 新 `csrc/moe/marlin_moe_wna16/sm70_marlin_u2_gemm.cu`(以 u4b8 的 474 行为
     骨架)+ `ops.cu` 分支与 TORCH_CHECK(group 64/128,无 zp);
  4. `csrc/quantization/marlin/gptq_marlin_repack.cu`:主体
     `pack_factor = 32/num_bits` 泛型,4/8 仅是快路分支——加
     `CALL_IF(2, false, false, CTA_N)` 并验证泛型路;兜底:torch 端一次性重排
     (boot 期 one-shot);
  5. `core/scalar_type.hpp` 加 `kU2` id(python `ScalarType` 同步),
     `fused_marlin_moe` 的 `assert num_bits in [4, 8]` 放行 2。
- 交付形态:全部改动打成 `patches/marlin-v100-u2-experts.patch`(fork 拥有
  patches/,由 setup_v100_marlin.sh 在 pinned rev 上追加应用);现有
  u4/u4b8/nvfp4 codegen 不触碰。
- 工作量级:内核 ~600 行新/改代码,无新科学。**Plan B**(Marlin 改造受阻时):
  Triton W2A16 decode 路径,放弃 prefill 常驻(退化回带内)。Plan C(质量失守
  才用):Q2_K 式两级 scale,需改 scale 读取,二阶不选。

### F.2 2-bit 格式与来源

- 格式:uniform 对称 2-bit、group 128、fp16 scales(K/128 × N/卡)。开销
  2 + 16/128 = 2.125 bpw。
- **质量风险要登记**:P3 门测的是 llama Q2_K(≈2.25–2.3 bpw,16 元子块 4-bit
  scale,更细粒度)。uniform g128 的 scale 预算更少,PPL 可能劣于 +10.5%。
  对策旋钮:group 64(= 2.25 bpw,与 Q2_K scale 密度对齐)先备着;引擎侧 P4 门
  (E.4 限定 #2)用**真内核真格式**重跑三件套 + needle,这是 P4 的 go/no-go,
  不沿用 P3 数字。
- 量化算法:RTN(与 P3 参照同族),来源 = NVFP4 dequant 后逐 group 对称量化。
  GPTQ 误差修正离线版留作质量不达标时的 (b) 案。
- 产线:默认 **boot 期在卡上重量化**(NVFP4 → dequant → 2-bit,常驻池一次成型,
  无新磁盘工件,checkpoint 仍以 NVFP4 为源真值);离线 2-bit checkpoint 缓存作
  加速 boot 的可选件。

### F.3 驻留与内存预算

| 项 | 现状(4-bit 半区 + spill) | P4(2-bit 全池) |
| --- | --- | --- |
| 专家显存/卡 | 12.7 GB(半区)+ host spill 20 GB/卡 | **12.7 GB(全池)** |
| dense+attention | ~8 GB/卡 | ~8 GB/卡 |
| NextN draft | 4.14 GiB/卡 | 4.14 GiB/卡(可议 2-bit) |
| host RAM pin | ~80 GB(spill 池) | **0** |
| page-in | prefill 墙(2.28 GB/s/卡) | **无** |

联动红利:capture 非法的 `ensure_spill_experts` 退出 → decode CUDA-graph 不再需要
breakable/disabled 特判,landing 池公式、MTP verify 宽度互斥(E.1 域)整体消失;
生产 serve 参数面大幅简化。

### F.4 门与验证

1. 内核单元:沿用 M2 探针族(`scripts/m2_marlin_unit.py` 骨架)——u2 重量化 vs
   dequant 参考,max-abs/rel + NaN 检测,先在单卡过。
2. 引擎质量门(P4 自己的,不沿用 P3):同管线三件套(sglang u2 全池 vs llama
   F16 参照的 top-1/top-20 带)+ needle 1k/4k;判据:不劣于部署带一个量级、
   needle 不掉案;PPL 预期落在混合臂 +10.5% 邻域(±格式差),破带则先旋 g64。
3. 性能门(C.4 重排后的合法门槛):prefill ≥800 tok/s @8k;decode 对照 P1.6/P2
   基线;记录 4×V100 实测。
4. 基础设施纪律:marlin .so 是 Qwen3.8 共用件——u2 为纯新增实例化,不得触碰
   现有 u4/nvfp4 codegen;结项时按惯例跑 Qwen3.8 NVFP4 回归。

## 附录 G:P4 SM70 2-bit 全池常驻(2026-10-02 收口)

### G.1 落地形态

按 F.1 选定路径全部落地:SM70 Marlin MoE 克隆 u2 实例化,纯新增、零共享 mainloop
改动。改动面:

| 文件 | 内容 |
| --- | --- |
| `patches/marlin-v100-u2-experts.patch` | 4 hunk:`csrc/moe/marlin_moe_wna16/sm70_marlin_u2_gemm.cu`(新,479 行,以 u4b8 为骨架)、`ops.cu` 分发 + `kU2` ScalarType、`sm70_marlin_iterator_utils.cuh`(u2 qweight 偏移,每 64 列 tile 的 qword 减半)、`CMakeLists.txt` |
| `python/sglang/srt/layers/quantization/sm70_u2_pool.py` | boot 期在卡上重量化:host per-layer memmap staging(`SGLANG_SM70_U2_STAGE_DIR`,避免 43 GB 匿名页)→ H2D chunk → `dequantize_nvfp4`(fp32,折入 `weight_scale_2`)→ RTN u2 → `repack_u2_sm70` → GPU 常驻池;16 专家/chunk |
| `environ.py` | `SGLANG_SM70_U2_EXPERT_POOL` / `SGLANG_SM70_U2_GROUP`(128/64/32 自适应回退)/ `SGLANG_SM70_U2_STAGE_DIR` |
| prebuilt | `_sm70_marlin_v100_moe.abi3.so` 重建(336 MB;u4/nvfp4 codegen 未触碰,Qwen 共用件纪律保持) |

实测池:E=288 H=4096 I=512 g=128 macro 256/256,**0.45 GiB/rank/层** × 42 路由层
≈ **18.9 GiB/卡常驻**(F.3 预算 12.7 GB 只算了 2-bit 码本,漏了 fp16 group scales
与 repack 对齐;32 GB 卡放得下,mem-fraction 0.92 实测可启动)。首层转换 32 s
(含 dequant warmup),其余 41 层 <2 min,boot 增量可忽略。staging 为每次转换的
临时 memmap、转换后 unlink,无跨 boot 陈旧缓存隐患。

### G.2 量化网格事故与修复(质量根因,非管线 bug)

首个网格沿用了 kernel 原生 bias-2 不对称格 `{-2,-1,0,+1}×0.375·amax`,全池上线后
输出退化。定位链:pool 位流、apply 路径、rsf 逐项核对无 bug;分层轨迹显示 L18
(mHC 流合并)比值跳 ×8.1,L25+ 指数发散(429.9×@L44)——与 P3 的 knife-edge 理论
一致:**per-layer MoE cosine 0.7777 在 P3 混合臂(+10.5% PPL)的失守侧**,结构性
clamp 误差(所有 >0.56·amax 的正权重被砍到 0.375·amax)被 mHC 放大成系统性发散。

反事实链(真权重 L3 expert 0,端到端 MoE cosine vs NVFP4 真值):

| 方案 | cos | 备注 |
| --- | ---: | --- |
| 原 bias-2 / 0.375 | 0.7777 | 失守侧 |
| g64 scales | 0.7838 | 换 group 无济于事 |
| 8-level(3-bit) | 0.947 | 需 ~47 GB/卡,装不下 |
| Lloyd-Max 非均匀 4-level 界 | 0.830 | g=128/64/32/16 平坦 |
| midrise bias 1.5 | 0.818 | |
| **bias 1.45 / f 0.36** | **0.8295** | ≈ 理论界,已上线 |

修复实现(单常数级):kernel 解码改两步 fp16-exact 减法——fp16 在 [1024,2048) 的
ULP=1.0,先 `__hsub2` 剥 0x6400 底(得精确码值)再减 fp16 常数 0x3dcd
(= 1.4501953125);builder 侧 `_U2_CODE_BIAS`/`_U2_SCALE_AMAX_MULT` 同步。
**约束已写入 `sm70_u2_pool.py`:bias 与 multiplier 必须与 kernel 一起动**,
单侧改任何一个都会回到 0.778。双侧验证:kernel 解码 == builder 网格
(tight arm cos 1.00000,relmax 6e-4);端到端 8 专家抽查 0.815–0.819,与预测吻合。

### G.3 质量门(通过,不沿用 P3 数字)

同管线 A/B:u2 全池 vs u4+spill 对照(P1.6 形态 boot),24 混合提示
(logprobs top-1,greedy)+ 5-needle(魔数埋深 0.15–0.87,~1k filler):

| 判据 | F.4.2 要求 | 实测 |
| --- | --- | --- |
| same-top1 | 不劣于部署带(91.0%)一个量级 | **91.7%**(22/24) |
| needle | 不掉案 | **5/5 双臂,魔数逐一相同** |
| 分层轨迹 | 无发散 | L25-44 平坦 0.78–1.0(对照峰 1.24;旧网格 429.9) |
| PPL | 混合臂 +10.5% 邻域 | 引擎侧未跑 PPL(留待必要时);same-top1 已达部署带,高于 P3 混合臂先例 79.4% |

生成抽查:CN/EN 事实问答正确,长文连贯,思维链完整(对照旧网格的退化输出)。

### G.4 性能(matched A/B,同日同参数,4×V100)

冷 flush(radix `/flush_cache` + 唯一后缀)协议,chunked-prefill 1024,triton
backend,target-only:

| 项 | u4+spill 对照(P1.6 形态) | u2 全池 | Δ |
| --- | ---: | ---: | ---: |
| prefill 8k | 102 tok/s | 254 tok/s | **+2.5×** |
| prefill 4k | 103 tok/s | 401 tok/s | +3.9× |
| prefill 2k | 103 tok/s | 598 tok/s | +5.8× |
| decode 真文本 C=1(256 tok) | 4.58 tok/s | 21.14 tok/s | **+4.6×** |

- decode 门(对照 P1.6/P2 基线 2.7–4.3 tok/s):**过**,+~4–5×。
- prefill 门(≥800 @8k):**未过**(254)。u2 路径超线性衰减
  (2k→598、4k→401、8k→254;逐 chunk 1.6→3.75 s 随位置增长),对照臂平坦 102
  (spill 导入主导,与 C.3 anatomy ~110 tok/s 吻合)——**墙不在 u2/MoE 路径**
  (C.3:Marlin MoE 仅 1.6 s/61 s 窗口),在随位置增长的注意力侧成本。
- **C.4 的"零导入 → prefill 转算力界(1.6–2.6 s/8k)→ ≥800 合法"前提被证伪**:
  页入彻底清零后 prefill 仍 30 s/8k,距算力界估计 11–18×。≥800 门是按该前提设的,
  判 P4 未达此门但达成本项全部实质目标(spill 子系统退出、decode ~4.6×、质量
  部署带)。残余墙的分解见 G.5。

### G.5 残余 prefill 墙解剖(pool 路径,in-server GPU-only trace,→ P5 交接)

对 8k 冷 prefill 做 12-forward GPU-only profile(TP0,auto-stop,server 存活;
与 C.3 同方法)。8k 请求窗 wall 32.4 s,GPU busy 32.2 s(99.5%,零空转):

| 分量 | GPU busy | 份额 |
| --- | ---: | ---: |
| triton `_fwd_kernel`(DSA 层 extend attention,144 calls) | **24.25 s** | **75.2%** |
| NCCL AllReduce TREE_LL | 4.32 s | 13.4% |
| `sm70_marlin_moe_gemm_kernel`(u2 全池) | 1.66 s | **5.1%** |
| dense GEMM(cutlass/volta fp16) | 1.28 s | 4.0% |
| KDA chunk kernels(34 层线性注意力) | ~0.25 s | 0.8% |

结论与交接:

- **墙 = DSA 层的 triton full-attention**,随上下文位置超线性增长(2k→598、
  8k→254 tok/s 的全部来源)。本引擎的 DSA indexer 未启用(M4 遗留),sparse 层
  目前按 full attention 跑;8k 下 11 层 triton extend 吃掉 3/4 的 GPU 时间。
- u2 全池 Marlin 仅 5.1%(1.66 s/8k,与 C.3 在 u4 上测得的 1.6 s 完全一致)——
  **MoE 侧已不再是任何意义上的瓶颈,2-bit 池的目的达成**。
- KDA 线性注意力 0.8%,可忽略。
- 对照臂核算:u4+spill 同请求 74.8 s ≈ 本窗 32.4 s + spill 导入 ~42 s,数字闭环。
- **P5 杠杆排序**:(1) DSA 稀疏路径(indexer + topk)或 sm70 flash-attention
  路径替代 triton extend(仓库已有 `flash_attn_v100_backend.py` 骨架);
  (2) NCCL 13.4% 与 (1) 重叠后重估;之后 prefill @8k 有 ~3–5× 空间,≥800 门
  在注意力修完后才是合法门槛。

### G.6 MTP arm(u2 + NEXTN 3/1/4,生产 spec 配置)

| 项 | 实测 |
| --- | --- |
| spec-vs-target 一致性 | 24/24 提示 top-4 逐一相同(greedy,logprobs) |
| prefill 8k | 238–239 tok/s(对 target-only 254 中性,−6%) |
| prefill 2k | 573 tok/s |
| decode C=1(client 口径,155–180 tok ×3) | **12.2 tok/s**(12.15–12.23) |
| accept len / rate | 2.0–2.8 / 0.31–0.83(窗口内 scheduler 吞吐 24–36 tok/s) |

- **对 spill 基线仍是 2.4–3×**(P1.6 mtp 臂 4.1–4.3、P2 抽检 5.12 client 口径),
  decode 门过;**但对 u2 target-only(21.1)是 −42%**:MTP 在 pool 路径上当前
  负收益。机理:target decode 快了 5.7× 而 spec 机制开销不变——verify+3 draft
  ~195 ms/步,accept 2.3 需要 ≤~108 ms/步才打平;其中 ~114 ms 是调研报告 §1.1
  的已知未定位项(spill 时代被 556 ms 页入掩盖,pool 路径上成为 decode 第一墙)。
- 结论:P5 的 decode 杠杆就是那个既有开放探针(spec step 解剖),不是 u2 回归;
  在其解决前,pool 路径 C=1 交互建议 target-only(21.1 tok/s)。
  **(2026-10-02 复测结项:见 G.8——decode 29.5、MTP 转正,本条建议作废。)**
- u2 池与 spec 全域相容(F.3 预测成立:无 landing/verify-width 互斥,boot 一次过,
  图捕获无特判;draft 层同样被 u2 化)。

### G.7 登记与遗留

- 环境变量已入 `environ.py`;生产 serve 脚本 `scripts/serve_glm53_flash_v100.sh`
  (target/mtp = u2 池;spill/spill-mtp = P1.6 形态对照)与引擎 README 段落随本
  结项落地。
- 探针工件已转存 MoE4All `scripts/glm53_u2/`(gate 探针 + counterfactual/
  Lloyd-Max/bias-scan 证据链,附 README);trace 解析器 `p4_prof_analyze.py`、
  `p4_prof_split.py` 同放。
- Qwen3.8 NVFP4 回归:已补跑(2026-10-02,见文末附录;**无回归**——首轮 −41~−47%
  系 bench boot 沿用 serve 脚本的 NVL 默认值所致的本机通信配置错误,经生产 compose
  配置复测已回到基线;顺带修复 spill-landing 误过滤 boot 回归与两个 serve 脚本的
  NCCL 拓扑自检)。
- g64 旋钮保留未用(质量门在 g128 即过);GPTQ 误差修正离线版未启用(F.2 的 (b) 案)。

### G.8 P5-b 复测(2026-10-02,统一镜像 + 通信修复):decode 杠杆结项

配置:统一镜像 `sglang-v100-unified:latest`(`dev-ltarcher` = GLM 线合并 +
NCCL 拓扑自检 + 两个镜像构建修复),docker compose 起(`~/vllm-Qwen3.8/
docker-compose-sglang-v100-glm53.yaml`:PXB + `SGLANG_CUSTOM_AR_ALLOW_PCIE=1`,
与 Qwen 生产 compose 同款通信 env;模型/u2 暂存挂载见该文件),u2 全池 +
NEXTN MTP 3/1/4,4×V100,boot ~23 min。bench 客户端口径:flushed radix、
unique suffix;decode 170 tok ×3(`ignore_eos`)。

| 项 | G.6(P4 战役,同配置) | 本次复测 | Δ |
| --- | --- | --- | --- |
| decode C=1 ×3 | 12.2 tok/s | **29.49–29.51** | **+142%** |
| decode vs target-only(21.1) | −42%(负收益) | **+40%,MTP 转正** | |
| prefill 2k | 573 tok/s | 717 | +25% |
| prefill 8k ×3 | 238–239(对照 target 254) | 233 / 255 / 255 | 带内持平 |
| accept len | 2.0–2.8 | 1.4–2.0 窗口(数字串任务 3.42) | 量级一致 |

- **归因**:accept ~1.7 折算 spec 步 ~58 ms(G.6 的 ~195 ms 降 ~137 ms)。
  未定位项的大头不是机制开销,而是通信:战役 boot 的 serve 脚本无
  `ALLOW_PCIE`,custom AR 拒启,decode small AR 全程走 NCCL——与本附录 Qwen
  回归误报同一根因。P2.5 cost-aware width 也在同一 boot,两项收益未做 A/B
  拆分;合并效果即上表。
- **质量复核**(同 boot):`p4_gate_probe.py` 对战役 arm-A JSON 逐位对照,
  same-top1 **93.2%**(384 位,门杠 91.7%)、needle 5/5、算术抽查正确。
  方法注记:裸 `/generate`(无 chat template)在开放提示上会复读,质量判定
  一律走 `/v1/chat/completions`。
- **P5-a 不受影响**:prefill 8k 与战役持平,DSA triton full-attention 75%
  仍是 prefill 第一墙、P5 杠杆 #1;decode 侧的 spec step 解剖杠杆就此结项。
- C=1 交互建议更新:u2 池 + MTP(29.5 tok/s)即为推荐形态,不再需要退回
  target-only。

### G.9 长上下文与 DFlash 评估(2026-10-02 纸面评估;2026-10-03 长上下文实测回填)

**长上下文(目标 262,144;`max_position_embeddings` = 1,048,576,架构合法)**

纸面显存账(每 rank 32 GiB,u2 池形态不变):

| 项 | 每 rank |
| --- | ---: |
| u2 专家池 + dense/attention(G.4 形态) | ~26.9 GiB |
| KV 池(= 32 × 0.92 − 26.9) | **~2.5 GiB** |
| KV 单价:11 层 DSA latent(512 维 fp16,不随 TP 切分) | 11 KB/token/rank |

(34 层 KDA 为固定循环状态,~17 MB/rank/seq,fp32 下 ~35 MB,可忽略。)

**实测回填(2026-10-03;P5-a dsa target-only,4 rung 三次 boot)**

| rung | prompt tok | prefill | prefill tok/s | needle | decode |
| --- | ---: | ---: | ---: | --- | ---: |
| 8k(P5-a 门) | 8,192 | 5.9 s | 1,390 | 5/5 | 23.8 |
| 32k | 31,665 | 22.4 s | 1,414 | 5/5 | 22.9 |
| 64k | 64,155 | 41.9 s | 1,531 | 5/5 | 22.9 |
| ~190k | — | — | ~1,470(逐 chunk 日志) | — | — |

- **性能墙证伪:稀疏路径 prefill 吞吐随上下文持平且略升**(1390→1414→1531),
  decode 恒定 ~23 tok/s。旧 G.5 的"262k prefill 6.9 h / decode 个位数"是
  full-attention triton 路径的外推,对 P5-a 稀疏路径不成立;262k 线性外推
  prefill ≈ **3 分钟**。
- **池账实测**:KV 池 cell = **11.85 KB/token/rank**(11×512×2B latent ≈
  11.26 KB + indexer fp8 key ~0.7 KB,kpool 同池)。0.92 下池 =
  **112,192 tokens / 1.33 GiB**,不是纸面的 ~244k——池不由显存上限决定,
  由 sizer 的 runtime slack(mem-fraction 决定,0.92 → 2.50 GiB)决定。
  `--max-total-tokens` 只能降不能升(`config_from_budget` 断言容量不升)。
- **边界实测**:0.98 boot 池 = **195,392 tokens**、capture 后余 0.57 GiB;
  ~190k 请求 prefill 到 ~107k token 时 eager attention(KDA/DSA 层
  self_attn)workspace OOM(80 MiB 申请、46 MiB 余)。64k 在 2.42 GiB 余量下
  无感;**eager prefill workspace 随上下文增长,墙在 64k 与 190k 之间**。
- **262k = 双条件,不是调参**:① 池还差 ~0.8–1.0 GiB/rank(0.98 池 195k,
  slack 归零也只有 ~247k)——要 fp8 latent KV(cell 减半,0.92 即可装下,
  需过质量门)或 u2 池削减 ~5%(回到 spill 形态,decode 倒退);② prefill
  workspace 需要更大余量(降 mem-fraction 与 ① 冲突,或减 chunked-prefill
  + 收敛 workspace 本身)。两条件都要动 u2/量化形态,需按本节质量门逐项
  验证,非单 boot 参数问题。
- **结论修正:262k 的前提 P5-a 已成立,性能不再是墙;墙只剩显存双条件。**
  现实阶梯 32k→64k 已实测可用(质量 5/5、吞吐持平),可直接进生产形态评估。

**DFlash(draft = `/data/models/GLM-5.3-Flash-DFlash`,2.6 GB)**

6 层 qwen3 型 sliding-window(4096)小模型,block-8 块扩散并行提案,EAGLE 式读
target 第 23/27/31/35/39/43 层 hidden states,target 逐步 verify;与 NEXTN 互斥,
一次只能跑一个 spec 算法。

- 支持面全套在树:`DFLASH` 算法与 `dflash_worker_v2`、`models/dflash.py`、
  `glm5_next` capture hooks(:1035 起,含 capture×mHC 专门分支)、arg 层
  `_handle_dflash` 校验(强制 num_steps=topk=1)。启用只需换 spec 段旗标。
- V100 适配缺口:draft 注意力后端须 triton(README 推荐 `trtllm_mha` 为
  sm80+);draft config bf16 → 必须 fp16(README 警告降精度伤 accept);draft
  KV 直接 fp16(窗口仅 4096,不值得上 fp8)。capture×mHC 分支已 boot 验证:
  结构生效不报错,但见下方实测,未走到能检验其质量的程度。
- 显存:draft 权重 ÷TP4 ≈ 0.65 GiB/rank + 图/KV ~0.2 → KV 池 2.5 → ~1.6 GiB
  (8k 运行无感,长上下文更紧)。
- 吞吐期望(纸面,已被实测推翻,见下):verify 8 token → target 步长 ~90–105 ms
  → **打平点 accept ≈ 3.1**(NEXTN 现状 29.5 tok/s @ accept 1.7、步 ~58 ms)。
  论文口径 accept 3.5–4.5,但被两个本机因素夹击:u2 噪声(capture 特征带 2-bit
  专家噪声,draft 训练于干净 bf16 target,正确性无损——verify 从 target 分布
  采样——但 agreement 可能缩水)、MoE target 步长抬高门槛。
- 长上下文形态优势:sliding-window draft 成本不随 ctx 增长(NEXTN 的 draft 是
  全注意力),真做长上下文时 DFlash 是更对的 spec 形态。

**DFlash 实测回填(2026-10-03;G.9 判决:accept 1.00 << 3.1,不替换 NEXTN)**

在 P5-a dsa target 组合上两轮 boot(图捕获采样器 + `SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=1`
对照),`--speculative-dflash-block-size 8 --speculative-draft-attention-backend triton
--speculative-draft-model-quantization unquant`:

- **accept len 1.00 / accept rate 0.00,每一步、两种采样器模式全部如此**:draft
  8 token 提案从未有一次被 verify 接受,decode 仅 **11.9 tok/s**(对照 NEXTN
  43.9 @ accept 2.0–3.85、target-only 23.8)——纯 verify 开销,比无 spec 慢一倍。
- 输出正确性不受影响(verify 全拒,提交输出 = 贪心 target):tops 21/24 top-8
  对 u2 参照(3 处近平局,与 dsa-mtp 臂同签名);needle 1k 5/5、2.5k 5/5。
- 两个 boot 陷阱:① draft 量化继承——`speculative_draft_model_quantization`
  默认继承 target 的 `modelopt_fp4`(serving_hook),纯 bf16 draft 检查点会被
  loader 在线重量化,必须显式 `unquant`;② `logprobs=true` 请求首步 device
  assert——`compute_spec_logprobs` 用垃圾 draft id 做 gather(logprob_processor
  现有 clamp 防御,拒绝位本不进 commit)。
- 根因排查到此为止的结论:**不是 Volta 内核问题**。flashinfer top_k、selector
  walk(eager+graph)、accept-bonus、prepare-block 四个内核在产线形状下隔离
  probe 全部 20/20 干净;mHC handoff 结构上已接通
  (`dflash_use_aux_hidden_state` 无条件 True → `hc_contract` 每 forward 应用)。
  指纹证据:`out_tokens` 第 1–5 列恒为 **7168(= hidden_size)**,跨请求、跨
  采样器模式不变;第 0 列是真 token;第 6 列未初始化。真扩散提案应逐位变化,
  这是引擎侧 draft block 构造路径(candidate/lattice 缓冲)的集成 bug,内核
  probe 覆盖不到。修复重启此线时,从该缓冲泄漏查起,boot 配方与本节记录可直接复用。
- **判决:维持 NEXTN。** DFlash 在本引擎当前状态下无任何吞吐收益;其长上下文
  形态优势(此节上文)保留为未来重启的理由,重启前置条件是修掉 block 构造 bug
  并让 accept 脱离 1.0。
- **决策:排 P5-a 之后实测;accept > 3.1 才替换 NEXTN;测后无论正负回填本节。**

## 附录:Qwen3.8 NVFP4 回归(2026-10-02,GLM 结项补跑)

**最终结论:无回归。** 基线(README 2026-09-24)在当日机器上复现:

| 项 | 基线 09-24 | 终测 10-02(生产 compose 配置) | 判定 |
| --- | --- | --- | --- |
| smoke | 通过 | **通过** | ✅ |
| prefill | 8,196 tok → 2,977–3,065 tok/s | 5,868 tok → **3,008 tok/s**(3 rep,首两 rep 2,229/2,612 为 warm-up) | ✅ |
| decode padding C=1 | 98.9 tok/s | **92.9 tok/s**(1024 tok `ignore_eos`;bench 客户端与基线不同,单流单跑,−6% 在带内) | ✅ |
| MTP accept | 2.11 | 2.25–2.48 | ✅ 不劣化 |

终测环境:生产 compose(`~/vllm-Qwen3.8/docker-compose-sglang-v100.yaml`,即
`sglang-v100:latest` 镜像 + `NCCL_P2P_LEVEL=PXB` + `SGLANG_CUSTOM_AR_ALLOW_PCIE=1`
+ `SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage`),镜像 venv 为 09-27 重建后的环境,host 为
09-29 重启后的同一台机——基线数字在"当前一切"下成立,venv 重建与重启均不构成残差。

**专用镜像复测(同日)**:按"GLM 改动前后分线"要求,从 GLM 改动之前的
12c6dbda98 拉 `dev-qwen3.8-flash-next`(+ NCCL 自检修复 cherry-pick),构建
`sglang-v100-qwen3.8-flash-next:latest`,同一 compose 配置下复测:decode **93.0
tok/s**、prefill **2,690–3,053 tok/s**、accept 1.85–2.55、greedy 输出与生产镜像
逐字一致、镜像内 smoke 通过——与生产镜像同配置无差异,Qwen 专用镜像线成立
(compose 变体:`~/vllm-Qwen3.8/docker-compose-sglang-v100-qwen38next.yaml`)。

**首轮误报的根因(记为教训)**:首轮 A/B 经 serve 脚本变体裸跑,而脚本 60 行
`export NCCL_P2P_LEVEL=NVL` 是给 NVLink mesh 目标机的默认;本机 PCIe-only(无
NVLink,P2P 经单 PLX)下 NVL 使 NCCL 判定 P2P 等级不足、整体退到 SHM 过 host,
且未设 `SGLANG_CUSTOM_AR_ALLOW_PCIE=1` 令 custom all-reduce 拒启——实测 decode
58.9 tok/s(−41%)、prefill −47%,与本附录最初记录的"环境残差"完全吻合。该误报
曾先后被归因到 09-27 venv 重建与 host 重启,均已证伪(终测环境两者都在)。

修复(两个 serve 脚本同步):`NCCL_P2P_LEVEL` 不再硬编码,启动时以
`nvidia-smi topo -m` 自检 NVLink mesh——有则 NVL,无则 PXB + `SGLANG_CUSTOM_AR_ALLOW_PCIE=1`;
环境显式值仍优先。

首轮数据中仍然成立的部分:三方 A/B 的**相对**结论(WIP 58.9 vs HEAD-clean 46.1
vs 09-25 树 45.6,WIP 的 CUDA GDN 路径 +27%;attention 后端 tilelang=triton;
GPU 核时 20.7 ms/步 ≈ 基线整步 21.3 ms——核本身是基线档)与两条工程抓获
(spill-landing boot 修复、prebuilt 空壳陷阱)。注意 11.6 ms/步 GPU 空窗的"host
残差"解释随配置错误一并作废;trace 里的大交接停顿是 SHM 通信路径的伪影。

GLM 侧连带发现:`serve_glm53_flash_v100.sh` 原先不设 `NCCL_P2P_LEVEL`(NCCL 自检
拓扑 → PXB,正确)但同样缺 `SGLANG_CUSTOM_AR_ALLOW_PCIE`,GLM 全程 decode 的
small AR 走的是 NCCL 而非 custom AR——脚本修复后 P5-b 复测时 decode 或有小幅
免费收益,记入 P5-b 清单(**已复测,见 G.8:decode +142%,MTP 转正**)。

顺带抓获的真 fork 回归(已修):Qwen 裸 boot(未设 `SGLANG_DSV41_SPILL_LANDING`,
默认 6)在 spill 未启用时也被 landing 宽度过滤砍光 spec 图形状而拒启;新增
`spill_decode_landing_active()`(`dsv41_expert_spill.py`,= spill APPLY 且 slots>0)
收口 decode 图过滤与 landing 预热的全部触发点。另:空壳陷阱再次应验——git archive
出的树拷贝缺未跟踪 `prebuilt/*.so` 时,marlin 检测静默落 "much slower Triton
W4A16" 零输出回退(23.2 tok/s、专家输出垃圾),smoke 类检测不可省。容器缺口:dev
容器无 libssl-dev,hicache 原生 hash 扩展编译失败,首轮以 no-hicache 变体 boot
(与生产 override 一致,非仓库回归)。

方法论教训(比数字更重要):**A/B 与基线对比必须冻结整条 launch 配置——通信环境
变量(NCCL_P2P_LEVEL / custom AR 开关)与代码、venv 同级**,只对齐代码和 .so 不够;
README 性能口径以生产 compose 为准,bench 变体 boot 只用于相对比较。

## 附录 H:P5-a DSA 稀疏路径设计(2026-10-02 开题)

目标:点亮 11 层 DSA 的 indexer + topk + 稀疏注意力(M4 欠账),替换 triton
absorbed-MLA 全注意力(G.8 后仍是 prefill 8k 的 ~75% GPU)。门:prefill 8k
≥800 tok/s(此前 C.4 的门在 indexer 未启用时"非法",此路径修完才合法)。

### H.1 现状链路(已定位,代码在树)

- GLM `use_dsa=True`(`glm5_next.py:1216` → `is_deepseek_dsa`),**IndexerKPool
  模块与权重已加载**(`deepseek_v2.py:2023-2061`,`index_kpool=4>1` → kpool 类),
  但 `--attention-backend triton` 下 MLA 层走 absorbed 全注意力,triton 后端以
  `supports_dsa_indexer=False`(`triton_backend.py:153`)声明不吃 indexer;
  eagle-worker 的 draft-backend 门(`eagle_worker_v2.py:438-468`)同源。
- 激活开关 = `--attention-backend dsa`(`DeepseekSparseAttnBackend`,
  `attention_registry.py:140`)。稀疏核经 `dsa_impl` 分派
  (`dsa_backend.py:2009/2023/2318/2332`,extend/decode 各有 tilelang 与 triton 分支)。

### H.2 SM70 缺口清单(全部静态定位完毕)

| # | 缺口 | 位置 | 方案 |
| --- | --- | --- | --- |
| 1 | 索引器 logits 内核全是 FP8:extend ragged `deep_gemm.fp8_mqa_logits`;decode paged `deep_gemm.fp8_paged_mqa_logits`(tilelang 变体限 arch==9) | `dsa_indexer_kpool.py:874-905, 974, 1206` | 新增 fp16 triton 内核(ragged + pooled-paged 两形态),**fp32 累加(llama-glm5 红线)**;fp16 直存比上游 fp8+scale 更precise |
| 2 | 索引器 K 缓存 fp8+scale 布局(head_dim_with_sf=132) | `kpool_fp8_index.py` 写入/读取 | SM70 分支:fp16 直存无 scale;池分配 dtype 已有 Volta fp16 注记(`kv_cache_dtype.py:66`) |
| 3 | 稀疏核选择受门约束:`triton` 仅 ROCm、tilelang(CUDA)需 bf16 KV、kpool>1 的 tail 只认 fa3/tilelang/trtllm | `overrides.py:_check_dsa_backend_constraints`、`dsa_backend_kpool.py:35-52` | 走 triton 稀疏核:扩 `triton_sparse_mla*.py`(DSv4 血统,单 pass+split-K,fp8/bf16 已有)支持 fp16;门以 env 开关放开(遵循 env-var-conventions);kpool tail 先验证索引空间再定是否需扩核 |
| 4 | kpool tail 语义 | 已确认:`n_select=(topk/kpool+1)*kpool-1`(512 池+1 边界池展开 → 2051 真实 token 格),llama-glm5 同式(`glm5next.cpp:11-17`) | 最终 index 在真实 token 空间 → gather 型稀疏核可消费;A2 步实测确认 |
| 5 | 顶层调度三处 `deep_gemm.get_paged_mqa_logits_metadata` 硬调用(deep_gemm 在 Volta=None) | `dsa_backend.py:743/1106/1457`、`dsa_indexer_kpool.py:787` | sm70 分支:fp16 内核自带 metadata 或不需要 |
| 6 | 杂项 | indexer `params_dtype=bf16`(weights_proj 等)、`--dsa-topk-backend` 默认 sgl-kernel | bf16→fp16(SM70_FORCE_FP16 机器约定);topk 先用 `torch`(正确性优先,`SGLANG_DSA_FUSE_TOPK=0`),再测 sgl-kernel |

tilelang d512 稀疏核不选:三次编译失败(LayoutInference/smem/`T.alloc_global`,
MoE4All §5),修复设想(dim-half)是独立高风险工程;triton 是本 fork 主场。

### H.3 执行增量(每步独立可验证)

- **A1 fp16 索引器 logits 内核**:ragged(extend)+ pooled-paged(decode/verify)
  两个 triton 内核;torch 参照单元测试 → llama-glm5 logits 对齐(3k ctx,
  >2051 强制 indexer 路径)。
- **A2 稀疏核 fp16 + 门放开**:`triton_sparse_mla*.py` 加 fp16;
  `_check_dsa_backend_constraints` 的 CUDA-triton 禁令以 sm70 env 开关豁免;
  kpool tail 消费实测。
- **A3 接线**:K 缓存 fp16 直存、调度 metadata sm70 分支、indexer 参数 dtype、
  serve 脚本/compose 增 dsa 参数组(`--attention-backend dsa
  --dsa-prefill-backend triton --dsa-decode-backend triton
  --dsa-topk-backend torch`)。
- **A4 验收**:oracle 门(同 prompt ≤2051 vs >2051 一致性、needle、与
  llama-glm5 的 logits/same-top1)→ **Qwen3.8 NVFP4 回归硬门(基础设施改动
  前置纪律)** → prefill 2k/8k 重测(对照 254/717)。

预算观感:decode/verify 的 indexer logits FLOPs(32h×128d×pooled 4:1)远小于
被替换的全注意力;8k prefill 索引 logits ~55 GFLOP/层 ×11 层,V100 fp16 实测
带内。P5-b 的 NEXTN skip_topk(nextn 层复用 target 索引)不受影响。

### H.4 A1/A2 实测定案(2026-10-02,修订 A2/A3 路线)

两个决定性实验事实,改写了 H.3 预设的路线:

| 事实 | 证据 |
| --- | --- |
| **Triton 在 sm70 没有 MMA 路径**:`tl.dot` 全部降级为标量 `fma.rn.f32`(PTX 0 条 mma.sync / 8192 条 fma),GEMM 形核实测 0.2 TFLOPS | p5a-test 容器 PTX dump + 计时 |
| **TileLang 0.1.8 在 sm70 走 WMMA**:内核源含 `nvcuda::wmma::mma_sync`,fp16 GEMM 实测 49.2 TFLOPS;且自带 `gemm_mma_sm70` emitter | 同容器 `T.gemm` 探针 |

- **A1 完成(已过单元测试,未提交)**:`python/sglang/kernels/ops/attention/dsa/fp16_mqa_logits.py`
  ——头折叠线性化(gate 为每头标量,`Σ_h gate·(q_h·k) = (Σ_h gate·q_h)·k`,
  fp32 折叠、一次 fp16 舍入、fp32 累加;比上游 per-head FP8 还多 ~2^8 精度)+
  分组 cuBLAS ragged GEMM(`out_dtype=float32`)+ 免 tl.dot 的 paged triton 内核。
  测试 `test/registered/kernels/test_fp16_mqa_logits.py` 5/5;实测:extend 全管线
  2.60ms @8k pooled,paged ≤0.114ms(16k pooled 含)。
- **A2 完成(已过 oracle,未提交)**:上游稀疏核 v2 为 Hopper 尺寸(warp 专用化
  384 线程、~170KB smem),v1 全宽 KV 双缓冲也超 Volta 96KB/块上限,且
  QK(trans_B)与 PV(非 trans_B)对同一缓冲要求不同 Volta swizzle
  (即 M4 的 "Get different layout" 失败签名)。`tilelang_sparse_sm70.py`
  以 v1 骨架重排:128 线程(4 warp——sm70 emitter 要求每 warp 整 16×16 tile,
  M=16 时 256 线程会切 M 触发断言)、fp16、K/V 半宽 [64,256] 双角色缓冲
  (buf_K 只做 QK、buf_V 只做 PV)、num_stages=1、免 O_shared。
  **数值:vs fp32 参照 max rel 4.3e-3、cos 1.000000;性能:5.21ms/chunk-1024
  (随机 indices 最坏情形,~880GB/s 已顶 HBM 上限),8k×11 层 ≈ 0.46s/rank
  vs 现状 24.25s ≈ 53×**。接线:`dsa_backend._forward_tilelang` 按 sm major
  分发(参数形状与上游逐位一致)。
- **A3 路线修订(最小 diff)**:**索引 K 缓存维持 fp8+scale 布局与全部写入核
  不动**(P4 质量门语义逐位继承;且省 ~20GB 池内存),只改读侧:
  1. extend:现有 gather 内核不动 → gather 出的 k_u8/k_scale 反量化为 fp16 →
     A1 ragged GEMM。
  2. decode/verify:A1 paged 内核加 fp8-cache 变体(核内 u8→fp8→fp32×scale
     反量化,页内 132B 布局)。
  3. q 侧:sm70 跳过 act_quant(query 保持 fp16),head gate 乘 q_scale=1;
     Hadamard 两侧保留(q 的 rotate_activation 与写核内 _hadamard128,点积差
     常数 √128 不改排序)。
  4. `build_schedule_metadata=False` 绕开 deep_gemm metadata 硬调用;
     group_topk=2048/4=512 走 fused JIT topk(tvm_ffi JIT,sm70 可编译),
     `sgl_kernel.fast_topk_v2` 仅 group_topk=2048 才触达,GLM 配置不经过。
- 后续优化储备(不阻塞验收):spill 同款 smem→smem K→V 复制省一半 gather;
  L2 局部性在真实 topk 下优于随机 indices 探针。

### H.5 A3 落地定案(2026-10-03;A4 引擎验证进行中)

A3 全部代码已落地(未提交),关键事实与偏离 H.3 预设的修订:

| 项 | 定案 |
| --- | --- |
| 索引 K 缓存 | **维持 fp8+scale 布局与写入管线逐位不动**(132B/token);读侧反量化 |
| q 侧 | sm70 跳过 act_quant(`_quantize_q_for_logits` seam):`gate_fold(query, gate·softmax_scale)` → q_eff [n,D] fp16;gate 融入 `weights_proj·n_heads^-0.5·softmax_scale`(q_scale≡1) |
| extend logits | gather 出的 k_u8/k_scale → torch 反量化 fp16 → `fp16_ragged_mqa_logits`(分组 cuBLAS,fp32 out) |
| decode/verify logits | `fp16_paged_mqa_logits_fp8kcache`(triton,核内 e4m3 位运算解码 × fp32 scale;NEXT_N constexpr 支持 MTP verify;输出 256 对齐 stride 兼容 fused topk ABI) |
| 调度 metadata | `_build_kpool_paged_mqa_schedule_metadata` sm<9 → False;dsa_backend.py 两处 metadata 构建加 `deep_gemm is not None` 守卫(deep_gemm 在 Volta 导入失败 = None/Exception) |
| topk | group_topk=512 → `fast_kpool_topk_transform_fused`(tvm_ffi JIT .cuh)**sm70 实测可编译且数值正确**;`SGLANG_DSA_FUSE_TOPK=0` 时 torch 路径(正确性优先);`fast_topk_v2` 仅 group_topk=2048,GLM 不经过 |
| topk 语义实测 | 输出 = 绝对真实 token 位置 ∈ [0, seq_len);短序(<2051)全选;`row_starts=ks` 把 ragged 列映射回绝对池行;判别性 oracle:8k 序 hot/warm 组入选、cold(<160 分)入选 0 |
| 非目标路径 | `_get_topk_ragged`(无 plan 兜底)改为 fail-fast NotImplementedError;triton 全注意力路径(triton backend)不触达 indexer,保持不变 |

- 单元测试 `test/registered/kernels/test_fp16_mqa_logits.py` 8/8(gate_fold、
  ragged 共享/交错范围、paged、ABI stride、fp8 缓存逐位一致 max rel 0.000e+00、
  MTP bs=2×next_n=3 表行折叠)。
- **A4 已启动**:`p5a-engine` 容器,`--attention-backend dsa
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --dsa-topk-backend torch`(fp16 KV 自动解析确认);serve 脚本增 dsa/dsa-mtp
  模式。验收清单同 H.3 A4。

#### H.5.1 首次 boot capture 失败与修复(2026-10-03)

第一次 A4 boot 在 decode CUDA graph capture 崩溃,暴露一个比 logits 内核更深的
事实:**triton backend 下 metadata=None 提前返回,连 compress write 都从未在
Volta 上执行过**——整个 kpool fp8 写管线(含其 bf16 断言)对 sm70 是零覆盖,
"写入管线逐位不动"的说法不成立,它只是从未被运行。逐雷修复:

| 雷 | 位置 | 修复 |
| --- | --- | --- |
| `index_kpool_compress_gate` 硬编码 `dtype=torch.bfloat16`(H.2 #6,A3 漏项) | `dsa_indexer_kpool.py` 建参处 | 参数按机器规则选 fp16(`major<8 && SGLANG_SM70_FORCE_FP16`,同 dtype 降级 hook 谓词;checkpoint bf16 值在 fp16 范围内无损) |
| 写核 bf16 断言链(`slot_k/tail_k/key`):`kpool_softmax_rotate_write_cache`、`kpool_decode_update_and_maybe_write_cache`、`kpool_write_tail_and_maybe_compress` | `kpool_fp8_index.py`(不改) | `_get_q_k_bf16`/`_get_k_bf16` 返回前 `key.to(torch.bfloat16)`——函数名即上游契约,sm80+ 是 no-op;fp16→bf16 精度回到上游水平 |
| decode 写核 `slot_score.dtype == tail_score.dtype`(tail 缓冲硬编码 bf16) | 同上 | 新增 `_compress_gate_score(x)` helper 统一 3 处 gate 乘积并 `.to(torch.bfloat16)`(选择分与上游同为 bf16 精度;extend/verify 断言本就接受多型) |

伴随探针(p5a-test,免 boot 验证):fp16 Hadamard JIT 可编译且 roundtrip
err≈1e-3(fp16 舍入级);`torch.compile(dynamic=True)` fp32 head-gate 链在
sm70 inductor 可编译(~3.4s 首编)。`k_norm` LayerNorm(fp32 参)走 native
路径自动 cast,安全。附带确认:weights_proj 保持 fp32 是红线设计(调用侧
`x.float()`),ape fp32 由写核契约要求,均不动。

修复后重启同一 docker run(env/参数全同)。A4 后续验收以本次 boot 为准。

#### H.5.2 第二次 boot 失败:sm70 triton 无 fp8e4nv,写核改 uint8 + 核内编码(2026-10-03)

第二次 boot 在 decode capture 崩于 `_kpool_decode_update_and_maybe_write_cache_kernel`
编译:`ValueError("type fp8e4nv not supported in this architecture…")`——sm70 triton
仅支持 `fp8e4b15`/`fp8e5`,而上游写核把 fp8 缓冲以 `torch.float8_e4m3fn` 视图传入,
launch 时指针元素类型即 fp8e4nv。修法(仅 `kpool_fp8_index.py`):

- 四个写核的池缓冲参数统一收 **uint8 裸字节指针**(入口既有
  `buf.dtype == torch.uint8` 断言,池缓冲本就是 uint8 分配;fp32 视图传 scale 不变)。
- 核内 RNE 编码 `_f32_to_e4m3_u8`:abs→指数字段→`p=max(e-3,-9)`→`rint(u·2^-p)`
  (libdevice.rint=RNE)→尾数溢出规格化→字节拼装+符号位。五个 store 全部只写不读,
  store 前 `quantized` 已 clamp ±448,故编码定义域即 |x|≤448。
- **位级验证**:对 |x|≤448 全域 32804 个采样值(含全部 binade 边界与 RNE ties)
  与 torch `.to(float8_e4m3fn)` 逐位一致(0 mismatch);域外 torch 溢出为 NaN
  (0x7f)不饱和,但核内不可达。
- 编译+写路径探针(p5a-test 免 boot):四个写核入口以生产 dtype 组合(req/chunk_src
  int64、n_from_tail/tail_logical_base int32、write_loc int64)全部编译通过;核 1
  encode store 落位 128/128 非零字节 + fp32 scale 正确。**生产 dtype 组合本身是
  编译正确性的一部分**:assemble 核 slot 循环两分支对 `off` 的类型必须一致
  (int64),混合 i32/i64 探针即触发 triton type-join 报错——生产全 int64 布局
  无此问题。
- 伴随排查:dsa 目录其余 fp8 视图(`dsa_indexer_kpool.py:1091/1303/1332`)均在
  torch 域(view+`.float()` 反量化或拷贝),无 triton 指针;per-batch deep_gemm
  回退有 sm70 fail-fast 守卫(A3);`dsa_indexer.py`(非 kpool)不在 GLM 路径。

修复后第三次 boot(同一 docker run)。A4 验收以本次 boot 为准。

#### H.5.3 第三次 boot 失败定位:kpool 解码必须走 fused topk(H.5.2 修复后)(2026-10-03)

第三次 boot 越过写核(H.5.2 修复有效),崩在 `transform_index_page_table_decode`
的 `assert topk_indices.shape[1] == 2048`:实际 2051。根因是**启动组合,不是内核**:

- kpool fused topk JIT 核(`kpool_topk_transform.cuh`,`-DSGL_GROUP_TOPK=512`,
  radix topk,无 fp8/wmma,TVM-FFI 包装)的输出列数 = topk + (pool_size-1 且传
  seq_lens)= 2048+3:KPOOL 语义里最后不满池的 1-3 个尾 token
  (`index_kpool_always_select_tail=True`)恒被追加为额外列。
- 该 2051 宽输出的合法消费者只有 fused 页表路径(`use_fused_topk=True`,
  `_get_fused_topk_page_table` 恒等返回——fused 核已经按 token 级 page_table_1
  gather 过)。`transform_index` 按契约只吃 2048 宽,是给非 kpool(indexer
  nsa)路径的。`SGLANG_DSA_FUSE_TOPK=0 + --dsa-topk-backend torch` 对 kpool
  decode 是上游从未执行的组合(上游默认 FUSE_TOPK=True;torch 后端只用于非
  kpool 计数路径)。
- **sm70 探针全绿**(`docker exec p5a-test`,`TORCH_CUDA_ARCH_LIST=7.0` JIT 编译):
  PAGED 全选/部分尾池(6/5/1)/判别式(top-512 组精确集合)、RAGGED offsets、
  RAGGED+row_starts(+seq_lens→2051)。注意核把 page_table_1 当 **token 级表**
  直接索引(raw_token→KV 槽),探针误传池级表会 OOB 读零,不是内核 bug。
- 第四次 boot 改上游原生组合:`SGLANG_DSA_FUSE_TOPK=1` +
  `--dsa-topk-backend sgl-kernel` + `SGLANG_OPT_USE_TOPK_V2=0`(v2 折叠计划
  零 sm70 覆盖,关掉以收敛 capture 面到唯一新内核)。tilelang 稀疏消费侧
  (`_forward_tilelang`)本就为 2051 宽设计(pad 到 64 倍数,-1 填充)。

#### H.5.4 第四次 boot 失败定位:tilelang sm70 稀疏核 tail_dim=0 布局崩溃(2026-10-03)

第四次 boot 越过 fused topk(H.5.3 组合生效),崩在
`tilelang_sparse_fwd_sm70` 首编:`tvm InternalError: no available layout found`
(LayoutInference pass)。参数矩阵(p5a-repro 容器,引擎同 env):

| (dim, tail) | 对应模型 | 结果 |
| --- | --- | --- |
| (256, 0) / (512, 0) | tail_dim=0 | **LayoutInference 崩**(零宽 `T.alloc_shared([H,0])` 无法分配寄存器布局) |
| (256, 256) | q_all=512, v_head=256 | OK |
| (512, 64) | DSV 式 q_all=576 | OK(探针 kv 宽度笔误,非内核问题) |

GLM-5.3-Flash 实际参数:`attn_mqa` RadixAttention 以 kv_lora_rank(512)为
v_head_dim、qk_rope_head_dim=0 → d_v=512、tail=0,此前探针只测过 tail=64
(DSV 形状),零尾是首遇。修复:核内 D_tail 以 host 常量在 trace 期分支,
零尾时跳过 `Q_tail_shared`/`K_tail_shared` 分配与 Q-tail copy(K-tail gemm
本就有 `if D_tail > 0`)。GLM 真形状复验:编译 ~11s、vs fp32 参
cos=1.000000(max abs 1.15e-05)、decode bs=4 0.56ms、chunk-1024 5.09ms。

#### H.5.5 A4 验收通过:DSA 稀疏路径点亮(2026-10-03)

第五次 boot(H.5.2 + H.5.3 + H.5.4 三修复齐上)健康:capture 145.9s /
0.40 GB,`fired up and ready to roll`。A4 门(/tmp/p5a_gate.py,对照
/tmp/p4_gate_u2.json u2-triton 臂,同 u2 池权重):

| 项 | 结果 | 判据 |
| --- | --- | --- |
| sanity | "Paris" 在 128 tok 窗内出现(GLM 风格是先复述题面再作答,16 tok 窗误报) | 通过 |
| 24-prompt top-1 vs u2 臂 | **24/24 逐 token 一致**(前 8 top 序列亦逐位一致) | 通过 |
| needle ~1k(全选域,prompt 1542) | **5/5** | 通过 |
| needle ~2.5k(判别 topk 域,prompt 3651 > 2051) | **5/5** | 通过(判别式组 topk 数值正确) |
| prefill ~2k flushed | 1913 tok,1369 tok/s | — |
| **prefill ~8k flushed** | 7619 tok,**1390/1393 tok/s**(u2 基线 717) | **≥800 门:通过(+94%)** |
| prefix cache | 重复请求 `#cached-token: 2304`(仅 3 新 token),文本一致 | 通过(该 build `prompt_tokens_details` 为 null,以服务端 radix 日志为准) |
| decode 单流 | 23.8 tok/s(@330 ctx)/ 22.2 tok/s(@1.5k ctx),无 MTP boot | 记录项 |

判别域通过意味着 indexer → fp8 池写 → fused JIT topk → tilelang 稀疏注意力
整链在 sm70 上数值正确;2k→8k prefill 速率几乎持平(1369→1390)是稀疏选择
的签名(全注意力随长度退化,基线 717@8k)。u2 requant + stage 缓存 boot
~20 min。引擎日志无错误(仅内核 preload 建议),49 请求全 200。

待办:Qwen3.8 NVFP4 回归硬门(下一节记录)→ llama-glm5 CPU 抽查 →
serve 脚本/compose 镜像 `SGLANG_DSA_FUSE_TOPK=1 + --dsa-topk-backend
sgl-kernel + SGLANG_OPT_USE_TOPK_V2=0`。

#### H.5.6 Qwen3.8 NVFP4 回归硬门:通过(三臂 A/B 定案,2026-10-03)

同日同协议三臂(unified 镜像,冻结 qwen38next 配置,bench = flushed
prefill ~4k/8k max_tokens=1 + 384-token 贪心 decode + 服务端 accept):

| 臂 | prefill 8k (tok/s) | decode (tok/s) | accept |
| --- | --- | --- | --- |
| 镜像内 python(基线) | 2878 / 2947 / 2840 | 81.3 / 82.5 / 82.5 | 1.95–2.58 |
| WIP 全树 bind-mount | 2358 / 2556 / **2374**(−15%) | 75.5 / 80.9 / 81.1 | 1.95–2.58 |
| 仅 P5-a 六个 DSA 文件挂载 | **2946 / 2947 / 2877** | 81.7 / 80.6 / **82.7** | 1.95–2.58 |

- **P5-a diff 对 Qwen 零影响,硬门通过。** 六个变更文件(4 改 2 新)全在
  DSA 路径,文件级挂载臂与镜像内臂差 <2.4%(噪声内),accept 逐位一致。
- **全树 bind-mount 的 −15% prefill 是仪器假象,不是代码回归**:dev 检出树
  遮住了镜像内编译工件(rust_extensions `*.so`),且自带本地陈旧
  Marlin `prebuilt/*.so`(树内 02:31/10:52 构建字节级 ≠ 镜像 10:36 构建)。
  教训入库:**Qwen 回归仪用文件级挂载变体**
  (`docker-compose-sglang-v100-qwen38next-p5a-dsaonly.yaml`),基线臂 =
  `-p5a-inimage.yaml`(去挂载);compose 头部已记三臂数据。
- 历史 "decode 93.2" 带与协议绑定(本次协议下两臂都读 ~82),跨协议数字
  不可比;有效对比只有同日同协议 A/B。
- 附注:GLM A4(boot 5)跑在 WIP 树(树内 Marlin .so)上;生产 compose 用
  镜像内 .so,P5-a 数字可能有小幅平移,方向未知——生产化前的 dsa-mtp 验收
  以镜像形态重测为准。

#### H.5.7 llama-glm5 CPU oracle 抽查(2026-10-03)

glm53-f16.gguf llama-server(CPU,56 线程,~3.3 min 载入)对拍:sanity
"Paris" 一致;两条 A4 prompt 的 oracle `reasoning_content` 与引擎臂文本同
族(中国的首都→Beijing 推理锚一致;注意力机制解释的 reasoning 开头近乎
逐句同构)。锚链闭合:llama-glm5 CPU 参考 ↔ u2 臂(P4 门 + 本次)↔ DSA
臂(A4 tops 24/24 + 前 8 top 逐位一致)。注意:该 oracle 模板默认
reasoning-preserve,`content` 为空时答案在 `reasoning_content`。

#### H.5.8 dsa-mtp 组合验收:NEXTN 投机解码跑通 DSA 稀疏路径(2026-10-03)

第六次 boot(boot-5 组合 + 生产 mtp spec 旗标 NEXTN 3/1/4 +
`--enable-linear-replayssm-spec --max-running-requests 4`)一次通过:
target verify 图 119.5s、draft decode 21.6s、draft extend 2.3s 全部捕获
(NextN 头含一个 DSA 层,稀疏路径在 draft 同样执行,零额外修复)。

| 项 | 结果 |
| --- | --- |
| 24-prompt top-1 vs u2/dsa 臂 | **24/24** |
| 24-prompt top-8 | 21/24;3 处分歧全是 meta-preamble 措辞近 tie 翻转("The user asks:" vs "The user is asking"),batched verify 的预期签名 |
| needle 1k / 2.5k | **5/5 / 5/5**(硬门) |
| decode 单流(384 tok essay) | **43.9 tok/s**(target-only 23.8 → +84%) |
| accept len(服务端) | 2.02–3.85 |
| needle 墙钟 | 1k 28.4s → 11.9s |

一致性判据(AGENTS.md:target-only 与 --spec 输出一致性 + accept)满足:
top-1 三臂逐位一致,分歧止步于前 8 token 的同义措辞。P5-a 全链验收完毕。

#### H.5.9 镜像形态复测与生产切换(2026-10-03)

镜像自 P5-a 后的 `dev-ltarcher` 重建(`docker/v100.cn.Dockerfile`,12.9.1
base,重活层全缓存,~18 min),`sglang-v100-unified:latest` 原位替换(旧镜像
id b19d7c586570 留档可回滚)。镜像形态(无 python bind mount)dsa-mtp boot
跑 H.5.8 同款门,与 WIP 树逐项一致:

| 项 | H.5.8(WIP 树) | H.5.9(镜像形态) |
| --- | --- | --- |
| top-1 vs u2/dsa 臂 | 24/24 | **24/24** |
| top-8 | 21/24(近 tie 措辞) | **21/24(同签名)** |
| needle 1k / 2.5k | 5/5 / 5/5 | **5/5 / 5/5** |
| decode 单流 | 43.9 tok/s | **43.9 tok/s** |
| accept len | 2.02–3.85 | **2.08–3.88** |

**生产 compose(`~/vllm-Qwen3.8/docker-compose-sglang-v100-glm53.yaml`)切为
dsa-mtp 模式**:attention-backend 换 dsa 四旗标,新增
`SGLANG_DSA_FUSE_TOPK=1`、`SGLANG_OPT_USE_TOPK_V2=0` 两个 env(镜像未烤,
脚本 dsa 段导出);文件头注明回退方法(四旗标换回 triton + 删两 env)。
decode 12.2 → 43.9 tok/s(+260%),8k prefill 254 → ~1390 tok/s。
