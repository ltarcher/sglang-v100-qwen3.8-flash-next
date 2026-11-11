# Spec: docker-compose-sglang-v100-mtp-8110 环境变量相关代码的单测补齐

## 目标

生产配置 `/home/bso/vllm-Qwen3.8/docker-compose-sglang-v100-mtp-8110.yaml`（注意目录名是 `vllm-Qwen3.8` 连字符）涉及的 26 个 `SGLANG_*`/外部环境变量与 CLI 参数，其"环境变量决策代码"全部有单测覆盖；本次补齐缺口单元，目标函数行覆盖率 ≥90%。

## 覆盖现状（调研结论）

### 已有单测覆盖（无需改动）

| 变量 | 既有测试 |
| --- | --- |
| SGLANG_CUSTOM_AR_ALLOW_PCIE | test/registered/unit/distributed/test_custom_all_reduce_v2_capability.py |
| SGLANG_CUSTOM_ALLREDUCE_ALGO | test/registered/unit/spec/test_v100_speculative.py:702-715 |
| SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS | test_qwen_sparse_sm70_routing.py:135-148 / test_qsa.py:693 |
| SGLANG_MAMBA_SSM_DTYPE（override 优先级） | test_mamba_unittest.py:114/254/322 等 |
| SGLANG_MIXED_PREFILL_STEP_LATENCY_MS | test/registered/unit/managers/test_mixed_prefill_budget.py:623-650 |
| SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION | test/registered/scheduler/test_scheduler_control.py:336 |
| SGLANG_NUMA_BIND_V2 | test/registered/utils/test_numa_utils.py:212/558/584 |
| SGLANG_V100_KV_HOST_TIER / KV_HEAD_LAYOUT / KV_TIER_PHYS_TOKENS | test_kv_cache_configurator_tier.py / test_kv_host_tier_full_capacity_gate.py |
| SGLANG_ENABLE_MM_UNMATCHED_TOKEN_DROP | test/registered/unit/multimodal/test_base_processor_unmatched_tokens.py |
| SGLANG_QWEN4_PLE_STAGED_DIR / REUSE / CHUNK_AHEAD | test_qwen4_exp_ple_staged.py / test_qwen4_exp_ple_prep.py（大量用例） |
| CLI 参数（EAGLE/mamba-strategy/ratio/ple-offload/cuda-graph-prefill/tilelang_fa_v100/linear-attn backends） | test_server_args.py、test_unified_prefill_cuda_graph_gate.py、test_v100_speculative.py、test_page_major_backend_allowlist.py、test_mamba_donated_alloc_ratio.py 等 |

### 外部依赖变量（sglang 代码不读取，明确排除）

- `NCCL_P2P_LEVEL`（NCCL 库）、`FLASHINFER_DISABLE_VERSION_CHECK`（flashinfer 包）、`HF_HUB_OFFLINE`（huggingface_hub）。

### 缺口（本次补齐，共 8 个变量 / 10 个决策单元）

| # | 变量 | 决策代码（文件:行） | 可测性 |
| --- | --- | --- | --- |
| G1 | SGLANG_FORCE_DETERMINISTIC | python/sglang/launch_server.py:29-37（模块级：CUBLAS_WORKSPACE_CONFIG + torch deterministic flags） | subprocess 导入断言 |
| G2 | SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK | srt/utils/common.py:2381-2403 `_should_skip_kernel_pkg_version_check` / `assert_pkg_version` / `check_pkg_version_at_least`（kernel 包白名单 + env 跳过） | 纯逻辑，mock version() |
| G3 | SGLANG_JIT_CACHE_DIR | kernels/jit/utils/compile/cache.py:301-303 `cache_root()`（默认 `~/.cache/sglang/jit` expanduser，env 覆盖） | 纯逻辑 |
| G4 | SGLANG_V100_DECODE_CUDA_BUILD_DIR | srt/layers/attention/tilelang_fa_v100/_decode_cuda.py:59-96 `_load_sm70_cuda_decode_ops()`（默认 `~/.cache/sglang/sm70_longctx_decode`，expanduser，load_inline 失败→None） | mock `load_inline` + 复位模块全局 `_EXT/_OPS_LOAD_ATTEMPTED` |
| G5 | SGLANG_V100_NVFP4_MOE_BUILD_DIR | kernels/ops/moe/sm70_nvfp4_moe_decode.py:36-73 `_load_sm70_nvfp4_moe_decode_ops()`（默认 `~/.cache/sglang/sm70_nvfp4_moe`，失败→None） | 同上（`_EXT/_LOAD_ATTEMPTED`） |
| G6 | SGLANG_SM70_FORCE_FP16 | ① model_executor/model_runner_components/load_model_utils.py:81-105 `maybe_downgrade_dtype_for_legacy_gpu`（capability<8 + env→float16，env=0 保留）；② :108-122 `cast_bf16_parameters_for_sm70`（capability<8 + env→bf16 参数转 fp16）；③ arg_groups/overrides.py:509-537 `_dsa_kv_cache_dtype_default` SM70 分支（major<8+env→float16 KV）；④ configs/mamba_utils.py:110-120 `mamba2_state_dtype` SM70 分支（conv→fp16，SSM env 未设时→fp16，显式 SSM env 优先） | mock `torch.cuda.get_device_capability`/`is_available` + envs override |
| G7 | SGLANG_FORCE_CPU_IMAGE_PREPROCESSING | multimodal/processors/base_processor.py:780-784 `_fast_image_processor_device` FORCE_CPU 分支（最高优先级返回 "cpu"） | 扩展既有 test_processor_device_selection.py |
| G8 | SGLANG_SM70_QSA_GROUPED_PREFILL | layers/attention/qwen_sparse_attn_backend.py:2137 门控 + :2452-2522 `_forward_sm70_grouped_sparse_prefill` 资格判定（breakable graph/MIXED/形状/stride/req_to_token 短路均返回 None；合格→build plan→run kernel） | CPU 张量 + stub backend（`object.__new__`）+ mock `build_grouped_union_plan`/`run_grouped_sparse_prefill`；L2137 门控行本身在 forward 内部，属集成路径，行为由本单测保障 |
| G9 | SGLANG_MAMBA_CONV_DTYPE（补充） | configs/mamba_utils.py:69（conv 默认 bf16、env 映射）+ :96-104（无效 SSM 值告警回退 float32） | 与 G6④ 同一测试文件 |

## 交付物

### 新测试文件（全部放 test/registered/unit/，遵循 write-sglang-test 惯例，register_cpu_ci / register_cuda_ci 注册）

1. `unit/utils/test_kernel_pkg_version_gate.py` — G2：
   - `_should_skip_kernel_pkg_version_check`：非白名单包恒 False；白名单包默认 False；env=1 → True。
   - `assert_pkg_version`：env=1 时版本过旧/未安装均不 raise；env=0 时按版本比较 raise/不 raise（mock `sglang.srt.utils.common.version` 与 `PackageNotFoundError`）。
   - `check_pkg_version_at_least`：env=1 → True 短路。
2. `unit/kernels/test_jit_cache_root.py` — G3：默认路径展开 `~`；env 设置生效；空字符串回退默认。
3. `unit/kernels/test_sm70_ext_build_dirs.py` — G4+G5：
   - 每个 loader：env 设置 → build_directory 传给 load_inline 的值 = expanduser(env)；env 未设 → 默认路径；load_inline 抛异常 → 返回 None 且不炸；`_SRC_PATH.is_file()` False → 返回 None（patch `_SRC_PATH`）。
   - 用 fixture 复位模块全局（`_EXT=None`、`_*_ATTEMPTED=False`），避免污染其他测试。
4. `unit/model_executor/test_sm70_fp16_dtype_gates.py` — G6①② + G6④/G9：
   - `maybe_downgrade_dtype_for_legacy_gpu`：cap≥8 早退；cap<7 raise；env=1 → `model_config.dtype` 变 float16；env=0 → 保留原 dtype（log 分支）。
   - `cast_bf16_parameters_for_sm70`：cuda 不可用/cap≥8 早退；env=0 早退；env=1 + bf16 参数 → 转 fp16 并计数。
   - `mamba2_state_dtype`：CPU 无 cuda → conv=env 值映射（无效值→bf16 默认）；mock cuda+cap(7,0)+env=1 → conv=fp16、SSM env 未设→fp16、SSM env 设了→env 优先；mock cuda+cap(9,0) → 不走 SM70 分支。
   - mock 点：`torch.cuda.is_available`、`torch.cuda.get_device_capability`（按模块引用 patch）。
5. `unit/test_force_deterministic_boot.py` — G1（subprocess 两次）：
   - env=1：`python -c` 导入 `sglang.launch_server`，断言 stdout 含 `[SGLANG_FORCE_DETERMINISTIC]`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`torch.are_deterministic_algorithms_enabled()==True`、`cudnn.deterministic=True`、`cudnn.benchmark=False`、tf32 False。
   - env=0（显式 "0" 与 unset）：均不开 deterministic。
   - subprocess 环境干净（`env=` 白名单传递 PATH+PYTHONPATH），超时保护。
   - register_cuda_ci(est_time≈600)（torch 导入 ×3 慢）。
6. `unit/layers/attention/test_qsa_grouped_prefill_gate.py` — G8：
   - 资格短路矩阵（CPU fp16 张量）：breakable-graph（mock True）→None；MIXED（forward_mode 与 _original_forward_mode 两路）→None；形状不符（R<256/ndim/dtype/int32 topk/k-v shape）→None；stride 不符（非 contiguous）→None；req_to_token None →None。
   - 合格路径：mock `build_grouped_union_plan` 返回 plan → `run_grouped_sparse_prefill` 返回张量 → 输出经 `_pad_extend_output`（stub 为恒等）返回。
   - env off 行为：直接构造调用 `_forward_sm70_grouped_sparse_prefill`（函数本身不含 env 判断，env 门控在调用点 2137，由本测试的存在性 + L2137 行为在 spec 记录为集成契约）。
   - backend 用 `object.__new__(QwenSparseAttnBackend)` + 仅设置被读属性（req_to_token、max_context_len、_sm70_grouped_prefill_logged、_pad_extend_output stub）。

### 既有测试文件扩展

7. `unit/multimodal/test_processor_device_selection.py` — G7 新增 2 用例：
   - env=1 → device=="cpu"（即使 base_gpu_id=3，优先级最高，胜过一切平台判定）。
   - env=0/unset → 走原判定（既有用例已覆盖，补一条显式 "0" 断言防回归）。
   - 用 `envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.override(...)` 上下文。
8. `unit/test_model_overrides.py` — G6③ 在 `test_dsa_kv_cache_dtype_default_pass` 追加 SM70 分支：
   - cap(7,0)+env 默认(True) → auto → float16；env=0 → auto → bfloat16；cap(7,0)+显式 bf16+env=1 → 断言拒绝（assert kv 只允许 fp16/bfloat16/fp8_e4m3，bf16 合法保留）。注意 env=0 分支需 override（EnvBool 默认 True）。

## 覆盖率核算口径

- 目标单元 = 上表 G1-G9 列出的函数 + launch_server.py:29-37 模块块。
- 用 `pytest --cov=<目标模块> --cov-report=json` 在 docker 容器（`sglang-v100-unified:chunk-ahead`，**-w 无效，bash -c 内显式 cd /sg**；fixture/G8 类测试需 GPU，纯逻辑测试 CPU 容器可跑）内运行新测试 + 被扩展的既有文件，从 coverage JSON 按 G1-G9 函数行范围核算行覆盖，要求 ≥90%。
- G1（模块级 import 块）以 subprocess 行为断言为契约；若 pytest-cov 子进程 hook（COV_CORE_*）在容器内生效则一并计入，否则在报告注明。
- L2137 单行门控（forward 内部）不强制计入函数行覆盖，其行为由 G8 资格测试 + 生产 A/B 记录保障。

## 验证步骤

```bash
# 容器内（GPU 供 G8/部分 mamba 测试；纯逻辑用例 CPU 亦可）
docker run --rm --gpus '"device=0"' -v /data/develop/sglang-v100-sxm2-qwen3.8-flash-next:/sg \
  sglang-v100-unified:chunk-ahead bash -c "cd /sg && pip install -q pytest pytest-cov && \
  python -m pytest test/registered/unit/utils/test_kernel_pkg_version_gate.py \
    test/registered/unit/kernels/test_jit_cache_root.py \
    test/registered/unit/kernels/test_sm70_ext_build_dirs.py \
    test/registered/unit/model_executor/test_sm70_fp16_dtype_gates.py \
    test/registered/unit/test_force_deterministic_boot.py \
    test/registered/unit/layers/attention/test_qsa_grouped_prefill_gate.py \
    test/registered/unit/multimodal/test_processor_device_selection.py \
    test/registered/unit/test_model_overrides.py -v --cov=... --cov-report=term-missing"
```

- 全绿 + 目标函数行覆盖 ≥90% + `ruff` 通过（新文件）+ CI 注册调用齐全。

## 约束

- 不改动 python/ 树的生产代码（G4/G5 用 mock 而非重构；environ.py 不动）。
- TDD：新增测试先跑一次确认"红"（缺失覆盖/断言失败的原因清晰），再确认对现状全绿（本任务是"补覆盖"而非修 bug，红灯=用例对当前行为断言前先人工核对预期与实现一致）。
- 不 commit（用户未要求）。
