# AGENTS.md

本规范适用于本仓库的所有 AI coding agents 与人类贡献者。

## 目标与范围

**Qwen3.8-Flash-Next 是本 fork 的旗舰与存在理由**(仓库名即它):NVFP4 W4A16、
262k 原生上下文、内置 MTP,在 4× V100-32GB 上已验收(prefill ~3,000 tok/s、
decode ~100 tok/s/流、accept ~2.1)。它是性能与正确性的参照基线——任何内核、量化、
调度、缓存层的基础设施改动,都必须以"不回归 Qwen3.8 验收数字与输出质量"为硬约束;
它是首选的回归金丝雀与新内核验证载体。

**新增使命:GLM-5.3-Flash**(320B/18B 激活,45 层 = 34×KDA 线性注意力 + 11×DSA,
modelopt NVFP4 检查点,182G):在 4× V100-32GB(128G 显存)+ ~120G host RAM + NVMe
目标机上高性能运行,支持 MTP 投机解码与 sglang 式前缀树缓存。

**持续目标**:维持 DeepSeek-V4.1-Flash 引擎线(CSA2 + 专家 spill)可用。

在上述范围内选择数值正确、与现有架构连贯、性能尽可能优的方案;不为了缩小 diff 牺牲
这些目标。分析/设计任务交付设计说明;诊断任务确立原因与证据,仅在要求时实施修复;
实现任务同步受影响的实现、测试与文档。

## 产品与架构

本仓库是上游 sglang 的 Volta(SM70)专用 fork。上游不支持 V100(CUDA 13 丢弃 sm70、
FlashAttention 需 sm80+、Volta 无 bf16);本 fork 用以下手段在 V100 上服务前沿模型:

| 领域 | 事实 |
| --- | --- |
| 量化 | NVFP4 W4A16 经移植 SM70 的 Marlin 内核(寄存器内反量化 FP16);`patches/marlin-v100-*.patch` 由安装器应用 |
| 精度 | Volta 无 bf16:`SGLANG_SM70_FORCE_FP16=1` 机器强制 fp16;新增模型代码必须 fp16-clean |
| 稀疏注意力 | DSV4.1 用 CSA2 + host Engram + MXFP4 专家 spill;sm70 JIT 内核运行时经 ninja/nvcc 编译 |
| 采样/JIT | FlashInfer 固定 revision + `patches/flashinfer-sm70.patch`;sglang-kernel 固定 0.4.6.post1 |
| 服务 | 每机同时只跑一个引擎,Qwen 与 DSV4.1 绑定同一地址 `0.0.0.0:11435`(`SGLANG_V100_HOST/PORT`),不是上游默认 30000 |
| 前缀缓存 | sglang RadixCache + hicache 文件分层(`SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 必须指向磁盘,/tmp 是 tmpfs 会吃 RAM) |

**关键陷阱**:上游 Marlin MoE 内核在 sm80 以下是空壳——缺 V100 内核的服务会正常启动、
正常应答、但专家输出全零。`bash scripts/smoke_v100.sh` 专门检测此事;任何 serve 前必须
先过 smoke,不得把"启动成功/有输出"当作内核正确的证据。

fork 纪律(改动前先分清文件归属):

- **fork 拥有**:`scripts/*_v100*.sh`、`patches/`、`docs/v100/`、本规范。可自由修改。
- **上游文件**(`python/sglang/**` 等):改动保持最小、聚焦、可 rebase;确需偏离上游的
  行为,优先用环境变量开关(`SGLANG_*`,遵循 `env-var-conventions` skill)而不是硬编码
  分叉;patches/ 仅用于第三方依赖源码(FlashInfer/FA/Marlin),不用于 python/ 树。
- 新增模型实现放上游惯例位置(`python/sglang/srt/models/`、`srt/layers/`),便于跟上游。

## 环境与验证

环境由 `bash scripts/install_v100.sh` 一键构建(CUDA 12.8 + Miniconda env
`sglang-v100` + Marlin 约 1 小时 nvcc 编译;源码缓存在 `~/.cache/sglang-v100-sources`)。
逐手册步骤见 **docs/v100/INSTALL.md**(权威文档;失败时按节排查,勿绕过)。

Python 一律用环境内解释器(不要裸 `pip`/系统 `python3`):

```bash
PYTHON="$HOME/miniconda3/envs/sglang-v100/bin/python"   # 或 $SGLANG_V100_PYTHON / venv
$PYTHON -m pytest test/registered/<path> -v             # 测试
bash scripts/smoke_v100.sh                              # 环境自检 + 预热 JIT(必过)
bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh [target|mtp]
bash scripts/serve_dsv41_v100.sh
```

JIT 内核编译的编译器约束(serve 脚本已内置,手工编译时需自设):nvcc host 编译器必须
≤ GCC 14 且带 cc1plus(`CC/CXX/CUDAHOSTCXX/NVCC_PREPEND_FLAGS`);`TORCH_CUDA_ARCH_LIST=7.0`;
venv 的 `bin` 必须在 `PATH`(ninja 是 venv 局部二进制)。

提交前:`pre-commit run --all-files`(ruff + ruff-format + isort + codespell +
clang-format 等;`check-chinese-characters` 只扫 `python/sglang/multimodal_gen/**`)。
`no-commit-to-branch` 钩子保护主分支——在 feature 分支上工作。

## 数值验证纪律

模型移植类变更(GLM 等)必须有独立 oracle,不得只对"看起来合理"的输出:

| 变更 | oracle 与判据 |
| --- | --- |
| 内核/量化/调度/缓存等基础设施 | **先跑 Qwen3.8 NVFP4 回归**:smoke 通过、验收命令数字不回退、生成输出质量不劣化;再谈新模型 |
| GLM-5.3-Flash 模型实现 | `/data/develop/llama-glm5`(llama.cpp fork,glm5next 已合入 + MTP)的 logits/文本;同输入小上下文对齐 |
| NVFP4 反量化语义 | modelopt 公式 `w = fp4_e2m1 × fp8_e4m3_scale × weight_scale_2`;布局参考 `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py`(注意 `swap_weight_nibbles` 与 scales 线性/转置布局) |
| Marlin/MoE kernel | 先 `smoke_v100.sh`(零输出检测),再与 fp16 dense 路径或上游参考数值对比 |
| MTP 接入 | target-only 与 `--spec` 两模式输出一致性 + accept 长度;投机路径改动先读 `speculative-naming` skill |
| 前缀缓存 | 同 prefix 请求 `cached_prompt_tokens` 提升 + 输出与全量 prefill 一致 |

性能声明必须在目标硬件(4×V100)实测,记录硬件、命令与可解读汇总;单卡开发机结论
不得外推为多卡结论。参考基线:Qwen3.8 NVFP4 prefill ~3,000 tok/s、decode ~100 tok/s/流
(MTP accept ~2.1);DSV4.1 warm 8k prefill ~560 tok/s、短代码 ~9 tok/s(重 spill 场景)。

## 参考导航

| 决策 | 入口 |
| --- | --- |
| 构建/环境逐手册 | `docs/v100/INSTALL.md` |
| 引擎能力、硬件、性能表 | `README.md` |
| 启动参数逐项理由 | `scripts/serve_*_v100.sh` 头部注释(OOM 敏感项改动前必读) |
| 公开文档站页面(Mintlify MDX) | `docs/AGENTS.md`(上游规范,勿改;`docs/v100/` 经 `docs/.mintignore` 排除在站点外,归本规范管) |
| 组件改动必读路由 | `.claude/rules/modify-component-must-read.md` |
| 代码风格既有约束 | `.claude/rules/*.md`(no-dataclasses、no-getattr-defensive、comment-style 等) |
| 任务方法 | `.claude/skills/*`(cookbook-add-model / cookbook-migrate-model / write-sglang-test / speculative-naming 等) |
| GLM 架构参考实现 | `/data/develop/llama-glm5`:`src/models/glm5next.cpp` + `conversion/glm5next.py` |
| GLM sglang 侧实现参考 | ktransformers(`doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md` 起步) |
| 本机模型工件 | `/data/models/GLM-5.3-Flash-NVFP4`(modelopt NVFP4 safetensors×120)等 |

## 变更一致性与提交

- 环境变量是本 fork 的主要行为契约:新增/改名 `SGLANG_*` 变量必须走 `environ.py` 约定
  并同步 serve 脚本注释与 `docs/v100/`;废弃变量先留别名一版并记录。
- 性能/精度相关新开关要在 README 对应表格登记条件与数字。
- 仅在用户要求时创建 commit;主题用祈使句、≤72 字符,正文说明动机与验证证据
  (测试命令与结果、对齐的 oracle、实测数字)。
- 完成条件:交付物可用、smoke/相关测试通过、实质声明有证据、无范围内阻塞问题。
