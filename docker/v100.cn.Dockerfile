# syntax=docker/dockerfile:1.7

# China-mirror variant of v100.Dockerfile for NVIDIA V100 (Volta, SM70).
# Build logic, layer ordering, and artifact validation are identical to
# docker/v100.Dockerfile; only the download endpoints change:
#
#   apt         -> mirrors.tuna.tsinghua.edu.cn      (APT_MIRROR)
#   pip / PyPI  -> pypi.tuna.tsinghua.edu.cn         (PIP_INDEX_URL)
#   torch       -> mirrors.aliyun.com pytorch-wheels (PYTORCH_WHEELS_URL)
#   rustup      -> rsproxy.cn
#   git clone   -> ghproxy.net prefix                (GITHUB_PROXY)
#   HF models   -> hf-mirror.com                     (runtime HF_ENDPOINT)
#
# Every mirror is a build arg. For example, to clone GitHub directly instead
# of through the proxy:
#   docker build -f docker/v100.cn.Dockerfile --build-arg GITHUB_PROXY= .
# Or build through compose:
#   docker compose -f docker/v100-compose.yaml -f docker/v100-compose.cn.yaml build
#
# gh-proxy-style mirrors rotate frequently; if the default stops working,
# swap GITHUB_PROXY for another working prefix or pass it empty.

ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYTORCH_WHEELS_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128/
ARG GITHUB_PROXY=https://ghproxy.net/

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR

ENV CUDA_HOME=/usr/local/cuda \
    CUDAHOSTCXX=/usr/bin/g++-12 \
    TORCH_CUDA_ARCH_LIST=7.0 \
    NVCC_THREADS=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Swap the Ubuntu archives for the mirror before the first apt-get call. http
# is deliberate: TLS trust is only guaranteed once ca-certificates (installed
# in this same RUN) is present.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    for sources in /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list; do \
      [ -f "$sources" ] || continue; \
      sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|http://${APT_MIRROR}/ubuntu|g" \
        -e "s|http://security.ubuntu.com/ubuntu|http://${APT_MIRROR}/ubuntu|g" \
        -e "s|http://ports.ubuntu.com/ubuntu-ports|http://${APT_MIRROR}/ubuntu-ports|g" \
        "$sources"; \
    done \
    && apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl ffmpeg git g++-12 libnuma-dev \
      ninja-build patch pkg-config protobuf-compiler python3.12 python3.12-dev \
      python3-pip python3-venv \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && python -m venv "$VIRTUAL_ENV"

FROM base AS builder

ARG MAX_JOBS
ARG PIP_INDEX_URL
ARG PYTORCH_WHEELS_URL
ARG GITHUB_PROXY
ENV MAX_JOBS=${MAX_JOBS} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    RUSTUP_DIST_SERVER=https://rsproxy.cn \
    RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
# No trailing /dist above: rsproxy's rustup-init.sh appends /dist itself, so
# RUSTUP_UPDATE_ROOT=.../rustup/dist would double it and 404 the installer.

# Rust through rsproxy.cn, with crates.io routed to its sparse index too.
RUN curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh \
    | sh -s -- -y --no-modify-path --profile minimal \
    && mkdir -p /root/.cargo \
    && printf '%s\n' \
      '[source.crates-io]' \
      'replace-with = "rsproxy-sparse"' \
      '' \
      '[source.rsproxy-sparse]' \
      'registry = "sparse+https://rsproxy.cn/index/"' \
      '' \
      '[registries.rsproxy]' \
      'index = "sparse+https://rsproxy.cn/index/"' \
      '' \
      '[net]' \
      'git-fetch-with-cli = true' \
      > /root/.cargo/config.toml

# Transparently rewrite every https://github.com/ URL -- including the clone
# inside scripts/setup_v100_marlin.sh -- to the proxy prefix. An empty
# GITHUB_PROXY leaves all clones direct. HTTP/1.1 avoids curl 92 stream
# aborts ("RPC failed") when large clones traverse the proxy.
RUN if [ -n "${GITHUB_PROXY}" ]; then \
      git config --global \
        url."${GITHUB_PROXY}https://github.com/".insteadOf "https://github.com/" \
      && git config --global http.version HTTP/1.1; \
    fi

WORKDIR /opt/sglang
COPY scripts/v100_safe_jobs.sh /usr/local/bin/v100-safe-jobs
RUN chmod +x /usr/local/bin/v100-safe-jobs

# Install SGLang's current runtime dependency list while deliberately excluding
# the four packages replaced below by CUDA 12.8 / SM70 builds. pip goes to
# PIP_INDEX_URL; the torch stack resolves its exact +cu128 wheels from
# PYTORCH_WHEELS_URL (the PyPI default build is also cu128, so the find-links
# mirror only pins provenance).
COPY python/pyproject.toml /tmp/sglang-pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade \
      pip setuptools wheel setuptools-scm setuptools-rust scikit-build-core \
      ninja psutil packaging \
    && python -m pip install \
      torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
      --find-links "${PYTORCH_WHEELS_URL}" \
    && python - <<'PY'
import subprocess
import sys
import tomllib
from packaging.requirements import Requirement

with open("/tmp/sglang-pyproject.toml", "rb") as file:
    project = tomllib.load(file)["project"]

dependencies = [
    *project["dependencies"],
    *project["optional-dependencies"]["diffusion-v100"],
]

replaced = {
    "flashinfer-python",
    # Skipped too: every sgl-deep-ep release on PyPI pins torch==2.13.0, which
    # cannot coexist with the torch 2.9.1 pinned here (and conflicts with
    # compressed-tensors' torch<2.11). sglang imports deep_ep behind
    # try/except (use_deepep / use_deepep_v2 fall back to False), and DeepEP
    # has no SM70 path, so nothing on V100 ever loads it.
    "sgl-deep-ep",
    "sglang-kernel",
    "torch",
    "torchaudio",
    "torchvision",
}
dependencies = [
    dependency
    for dependency in dependencies
    if Requirement(dependency).name.lower().replace("_", "-") not in replaced
]
subprocess.check_call([sys.executable, "-m", "pip", "install", *dependencies])
PY
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install \
      grpcio==1.81.1 grpcio-health-checking==1.81.1 \
      grpcio-reflection==1.81.1 protobuf==6.33.6 tilelang==0.1.8 \
    && python -m pip uninstall -y nvidia-nccl-cu13 || true \
    && python -m pip install --force-reinstall --no-deps \
      nvidia-nccl-cu12==2.27.5

# Patched FlashInfer SM70 source. This is editable because its JIT headers and
# Python sources are both needed at runtime.
COPY patches/flashinfer-sm70.patch \
      /opt/sglang/patches/flashinfer-sm70.patch
RUN git clone https://github.com/haohervchb/flashinfer.git \
      /opt/deps/flashinfer-sm70 \
    && git -C /opt/deps/flashinfer-sm70 checkout --detach \
      c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833 \
    && git -C /opt/deps/flashinfer-sm70 apply \
      /opt/sglang/patches/flashinfer-sm70.patch
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip uninstall -y flashinfer-python flashinfer-cubin || true \
    && python -m pip install --no-deps --no-build-isolation \
      -e /opt/deps/flashinfer-sm70

# TurboMind is the one temporarily retained external SM70 component. Fetch
# only its attributed source directories; never install the 1Cat vLLM or
# attention packages and never place a full 1Cat checkout in the image.
RUN git clone --filter=blob:none --sparse --no-checkout \
      https://github.com/1CatAI/1Cat-vLLM.git \
      /opt/deps/turbomind-sm70-source \
    && git -C /opt/deps/turbomind-sm70-source sparse-checkout set \
      LICENSE csrc/core csrc/sm70_turbomind csrc/moe \
    && git -C /opt/deps/turbomind-sm70-source checkout --detach \
      6ada86ed64af6d1a7b3cb0f34df237fd86f06d48
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git \
      /opt/deps/cutlass-turbomind \
    && git -C /opt/deps/cutlass-turbomind checkout --detach \
      da5e086dab31d63815acafdac9a9c5893b1c69e2

# Build SGLang's adapter for the attributed TurboMind W8A16 block-FP8 and
# FP16 MoE source against the pinned CUTLASS revision used for host validation.
COPY scripts/build_sm70_turbomind.py /opt/sglang/scripts/build_sm70_turbomind.py
COPY python/sglang/kernels/jit/csrc/sm70_turbomind_bindings.cpp \
     /opt/sglang/python/sglang/kernels/jit/csrc/sm70_turbomind_bindings.cpp
COPY python/sglang/kernels/jit/csrc/sm70_fp16_moe_gemm.cu \
     /opt/sglang/python/sglang/kernels/jit/csrc/sm70_fp16_moe_gemm.cu
COPY python/sglang/kernels/jit/csrc/sm70_fp8_e5m2_cache.cu \
     /opt/sglang/python/sglang/kernels/jit/csrc/sm70_fp8_e5m2_cache.cu
RUN --mount=type=cache,target=/root/.cache/torch_extensions,sharing=locked \
    export MAX_JOBS="$(v100-safe-jobs)" \
    && export SGLANG_TURBOMIND_SM70_ROOT=/opt/deps/turbomind-sm70-source \
    && export SGLANG_TURBOMIND_CUTLASS_ROOT=/opt/deps/cutlass-turbomind \
    && python /opt/sglang/scripts/build_sm70_turbomind.py

# Only this source tree invalidates the sglang-kernel layer. The context ignores
# all local .so/build outputs, preventing the stale-binary bug from the host.
COPY python/sglang/kernels/aot /opt/sglang/python/sglang/kernels/aot
RUN python -m pip uninstall -y sglang-kernel || true
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    find /opt/sglang/python/sglang/kernels/aot/python/sgl_kernel \
      -type f -name 'common_ops*.so' -delete \
    && JOBS="$(v100-safe-jobs)" \
    && export MAX_JOBS="${JOBS}" \
    && export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    && export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=1" \
    && python -m pip install --no-deps --no-build-isolation \
      /opt/sglang/python/sglang/kernels/aot

# Marlin uses the dedicated marlin_v100 repository and the exact proven local
# compatibility/tuning patches. Its clone goes through GITHUB_PROXY via the
# insteadOf rewrite above; the script reuses the cache mount on rebuilds. Its
# smoke test is deferred to the next layer.
RUN git clone --depth 1 --branch v4.2.1 \
      https://github.com/NVIDIA/cutlass.git /opt/cutlass
COPY scripts/setup_v100_marlin.sh /opt/sglang/scripts/setup_v100_marlin.sh
COPY patches/marlin-v100-qwen-sm70-tuning.patch \
      /opt/sglang/patches/marlin-v100-qwen-sm70-tuning.patch
COPY patches/marlin-v100-qwen38-nvfp4-tuning.patch \
      /opt/sglang/patches/marlin-v100-qwen38-nvfp4-tuning.patch
COPY patches/marlin-v100-u2-experts.patch \
      /opt/sglang/patches/marlin-v100-u2-experts.patch
RUN --mount=type=cache,target=/opt/deps/marlin-v100,sharing=locked \
    export CUTLASS_DIR=/opt/cutlass \
    && export MARLIN_V100_REPO=/opt/deps/marlin-v100 \
    && export MARLIN_V100_REF=6d72a49939701d26b15b617a4cd2423174adb2d1 \
    && export MARLIN_V100_INSTALL_DIR=/opt/v100-artifacts \
    && export MARLIN_V100_SKIP_SMOKE=1 \
    && export MARLIN_V100_SKIP_BF16_COMPAT=1 \
    && export MAX_JOBS="$(v100-safe-jobs)" \
    && bash /opt/sglang/scripts/setup_v100_marlin.sh

# Application changes do not invalidate any native compilation layer.
# apache-tvm-ffi is re-pinned here (not beside tilelang above) to keep that
# dependency layer's cache: tilelang 0.1.8's bundled TVM breaks its
# TVMDerivedObject shim under apache-tvm-ffi >= 0.1.11, which is what
# sgl-deep-gemm 0.2.0 pins -- but sgl-deep-gemm never imports on SM70
# (ENABLE_JIT_DEEPGEMM requires SM90+), so the tilelang-era 0.1.9 wins.
COPY python /opt/sglang/python
# setup.py reads rust/Cargo.toml, a workspace manifest whose members are
# sglang-grpc, sglang-mm, and sglang-server, so the whole rust/ tree is
# needed -- copying only sglang-grpc fails the editable install.
# SGLANG_BUILD_RUST_EXTS (substring match on extension names) excludes
# sglang-radix-tree: its TreeCore links torch via tch and gates torch to
# 2.11-2.13, while this image pins 2.9.1. That crate is a Unified Radix
# Cache backend that postdates the proven V100 build; the other three
# crates do not link torch and build fine against 2.9.1.
RUN --mount=type=bind,source=rust,target=/mnt/sglang-rust,ro \
    --mount=type=bind,source=proto,target=/mnt/proto,ro \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    mkdir -p /opt/sglang/rust \
    && cp -a /mnt/sglang-rust/. /opt/sglang/rust/ \
    && cp -a /mnt/proto /opt/sglang/proto \
    && export SGLANG_BUILD_RUST_EXTS=grpc,multimodal,server \
    && python -m pip install --no-deps --no-build-isolation -e /opt/sglang/python \
    && python -m pip install cuda-tile==1.5.0 \
    && python -m pip install apache-tvm-ffi==0.1.9 \
    && cp /opt/v100-artifacts/_sm70_marlin_v100_*.abi3.so \
      /opt/sglang/python/sglang/kernels/prebuilt/

# GPU-independent artifact validation is intentionally after every expensive
# compilation RUN, so BuildKit retains those layers if this check ever changes.
RUN python - <<'PY'
import importlib.util
from pathlib import Path
import subprocess

site = Path("/opt/venv/lib/python3.12/site-packages")
common_ops = list((site / "sgl_kernel" / "sm70").glob("common_ops*.so"))
assert len(common_ops) == 1, common_ops
assert common_ops[0].name == "common_ops.abi3.so", common_ops[0]
marlin_dir = Path("/opt/sglang/python/sglang/kernels/prebuilt")
marlin = [
    marlin_dir / "_sm70_marlin_v100_dense.abi3.so",
    marlin_dir / "_sm70_marlin_v100_moe.abi3.so",
]
turbomind = marlin_dir / "_sm70_turbomind_v100.so"
grpc_core = list(
    Path("/opt/sglang/python/sglang/srt/rust_extensions").glob("_grpc*.so")
)
assert len(grpc_core) == 1, grpc_core
assert Path("/opt/deps/flashinfer-sm70/flashinfer/sampling.py").is_file()
assert Path(
    "/opt/sglang/python/sglang/srt/layers/attention/"
    "tilelang_fa_v100/_kernels_paged_decode.py"
).is_file()
for module in ("flash_attn_v100", "flash_attn_v100_cuda", "flash_qla_v100"):
    assert importlib.util.find_spec(module) is None, module

def validate_sm70(binary, required_strings):
    binary = Path(binary)
    assert binary.stat().st_size > 100_000, binary
    cubins = subprocess.check_output(
        ["cuobjdump", "--list-elf", str(binary)], text=True
    )
    assert ".sm_70.cubin" in cubins, (binary, cubins)
    assert all(f".sm_{arch}." not in cubins for arch in (75, 80, 86, 89, 90, 100))
    strings = subprocess.check_output(["strings", str(binary)], text=True)
    for value in required_strings:
        assert value in strings, (binary, value)

# gptq_gemm was removed from the sm70 sgl-kernel by the 2026-09-19 remasure
# commit; GPTQ on V100 goes through the Marlin extension validated below.
validate_sm70(common_ops[0], ["all_reduce", "causal_conv1d_fwd"])
validate_sm70(marlin[0], ["marlin_gemm"])
validate_sm70(marlin[1], ["moe_wna16_marlin_gemm"])
validate_sm70(turbomind, ["fp8_gemm", "f16_moe_gemm", "fp8_e5m2_cache_write"])
print("SM70 build artifacts validated")
PY

FROM base AS runtime

ENV NCCL_P2P_LEVEL=NVL \
    SGLANG_MAMBA_CONV_DTYPE=float16 \
    SGLANG_MAMBA_SSM_DTYPE=float16 \
    SGLANG_SM70_DENSE_GEMV=1 \
    SGLANG_SM70_QWEN_FUSIONS=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com \
    FLASHINFER_WORKSPACE_BASE=/root/sglang-v100-jit \
    TILELANG_CACHE_DIR=/root/sglang-v100-jit/tilelang \
    TRITON_CACHE_DIR=/root/sglang-v100-jit/triton \
    TORCHINDUCTOR_CACHE_DIR=/root/sglang-v100-jit/torchinductor \
    SGLANG_V100_PYTHON=/opt/venv/bin/python

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/deps/flashinfer-sm70 /opt/deps/flashinfer-sm70
COPY --from=builder /opt/sglang/python /opt/sglang/python
COPY scripts/smoke_v100.sh \
      scripts/serve_qwen38_flash_next_nvfp4_v100.sh \
      scripts/serve_glm53_flash_v100.sh \
      scripts/serve_dsv41_v100.sh \
      /opt/sglang/scripts/
COPY docker/v100-entrypoint.sh /usr/local/bin/v100-entrypoint
RUN chmod +x /opt/sglang/scripts/smoke_v100.sh /usr/local/bin/v100-entrypoint

WORKDIR /opt/sglang
EXPOSE 8082

ENTRYPOINT ["/usr/local/bin/v100-entrypoint"]
CMD ["--help"]
