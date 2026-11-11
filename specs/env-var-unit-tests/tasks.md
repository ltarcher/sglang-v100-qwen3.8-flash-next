# Tasks: env-var 单测补齐（specs/env-var-unit-tests/spec.md）

## T1. G2 — kernel 包版本闸门（SKIP_SGL_KERNEL_VERSION_CHECK）
- [x] 新建 test/registered/unit/utils/test_kernel_pkg_version_gate.py（3 组用例，见 spec §1）
- [x] mock 点：`sglang.srt.utils.common.version`、`PackageNotFoundError`；env 用 `envs.SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK.override`
- [x] register_cpu_ci

## T2. G3 — JIT cache root（JIT_CACHE_DIR）
- [x] 扩展既有 test/registered/unit/kernels/test_jit_cache.py（spec 原计划新建 root 文件，实际该目录已有 jit cache 测试文件，追加 4 个 cache_root 用例）：默认 `~/.cache/sglang/jit` expanduser、env 覆盖、空串回退
- [x] kernels 目录已存在，无需创建

## T3. G4+G5 — 两个 SM70 ext build dir loader
- [x] 新建 test/registered/unit/kernels/test_sm70_ext_build_dirs.py
- [x] 每模块 5 用例：env 覆盖→expanduser 路径传 load_inline；默认路径；load_inline 异常→None 且 attempted 锁存；_SRC_PATH 缺失→静默 None；_EXT 缓存早退；decode 侧另含 SGLANG_V100_DECODE_CUDA 开关与 sm70_cuda_decode_available（(7,0)/(9,0)/驱动异常）
- [x] fixture 复位模块全局：`_decode_cuda._EXT/_OPS_LOAD_ATTEMPTED`、`sm70_nvfp4_moe_decode._EXT/_LOAD_ATTEMPTED`；patch `torch.utils.cpp_extension.load_inline` + patch os.makedirs
- [x] setUp 内延迟导入（消除 tilelang libcudart stub 与 flashinfer ctypes 的 import 顺序 flake，4 次重复批跑稳定通过）

## T4. G6①②+G6④/G9 — SM70 fp16 dtype gates
- [x] 新建 test/registered/unit/model_executor/test_sm70_fp16_dtype_gates.py（18 passed）
- [x] maybe_downgrade_dtype_for_legacy_gpu：cap≥8 早退 / (6,1) raise / (7,0)+env→float16（断言 override 调用）/ env=0→保留
- [x] cast_bf16_parameters_for_sm70：env=1 转 fp16（assertNotIn bf16/assertIn fp16）/ env=0 早退 / cap≥9 早退 / 无 cuda 早退
- [x] mamba2_state_dtype：conv env 映射+无效值回退 / SM70 强制 conv fp16 + SSM 未设→fp16 / SSM env 显式优先 / env=0 保留 / cap(9,0) 忽略 / config、text_config 优先级
- [x] mock `torch.cuda.is_available` / `torch.cuda.get_device_capability`；镜像烘焙的 SGLANG_MAMBA_*_DTYPE=float16 用 override(None) 归零基线

## T5. G1 — FORCE_DETERMINISTIC boot 行为
- [x] 新建 test/registered/unit/test_force_deterministic_boot.py，subprocess -c 导入 sglang.launch_server（2 用例 + 13 subtests passed）
- [x] env=1：marker 行 + CUBLAS_WORKSPACE_CONFIG=:4096:8 + are_deterministic_algorithms_enabled + cudnn.deterministic/benchmark + tf32 两项
- [x] env="0" 与 unset：deterministic 均不开
- [x] 干净 env 白名单 + timeout=600；register_cuda_ci(est_time=600)
- [x] pytest-cov 子进程 hook（COV_CORE_*）在本容器不生效——launch_server 块覆盖用进程内 `SGLANG_FORCE_DETERMINISTIC=1 coverage run` 探针单独度量（L29-37 无缺失行，块覆盖 100%）

## T6. G8 — QSA grouped prefill 资格门控
- [x] 新建 test/registered/unit/layers/attention/test_qsa_grouped_prefill_gate.py（12 passed）
- [x] 短路矩阵：breakable graph / MIXED×2 / R<256 / bf16 / ndim / topk 非 int32 / k-v shape / 非 contiguous / req_to_token None → 各返回 None
- [x] 合格路径：mock build_grouped_union_plan + run_grouped_sparse_prefill（plan 用真实零张量，日志块可解包）
- [x] backend = object.__new__(QwenSparseAttnBackend)，仅设被读属性；CPU fp16 张量
- [x] GPU 容器内运行通过

## T7. G7 — FORCE_CPU_IMAGE_PREPROCESSING 分支
- [x] 扩展 test/registered/unit/multimodal/test_processor_device_selection.py（15 passed）：SGLANG_MM_IMAGE_PROCESSOR_DEVICE 优先级 / FORCE_CPU=1→"cpu" / ="0"→原判定 / 3 个 NPU 分支
- [x] 用 envs.*.override

## T8. G6③ — DSA KV dtype SM70 分支
- [x] 扩展 test/registered/unit/test_model_overrides.py::test_dsa_kv_cache_dtype_default_pass：非 DSA / Llama / npu / xpu / dsa=False / backend warning / (7,0)+env 默认→float16 / env="0"→bfloat16 / 显式 fp16 / learnable_sink+fp16→ValueError（该用例由既有失败转为通过）
- [x] `_view` 补 dtype="auto"（修复 fork 引入的既有 AttributeError）

## T9. 全量验证与报告
- [x] docker GPU 容器内跑全部新/改测试文件：110 passed + 11 subtests（覆盖率批）、83 passed ×4（flake 验证批）、2 passed（G1 批）；test_model_overrides.py 5 个失败经 HEAD 基线比对确认为既有失败（tilelang_fa_v100 白名单/GptOss 后端支持/ple_offload 子测试），本次净修复 1 个（dsa_dtype_pass）
- [x] pytest --cov 目标模块 + coverage JSON 核算：13 个目标函数全部 100%（launch_server L29-37 模块块亦 100%），≥90% 达标
- [x] 新文件 ruff 0.15.1 check + format 全部通过（test_model_overrides.py 26 项违规为既有问题，数量与修改前一致）
- [x] 输出覆盖矩阵报告（变量→测试→行覆盖）；不 commit
