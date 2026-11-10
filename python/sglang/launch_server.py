"""Launch the inference server."""

import asyncio
import os
import sys
import warnings

# Optional deterministic CUDA config.  Default off (perf first).
# On 2026-11-10 we traced a long_prompt (~33k token) staged-arm degradation
# where staged/pinned prefill first-token argmax flipped sporadically across
# boots: the staged PLE gather has a small but intrinsic numeric delta vs the
# pinned path (different memory-access pattern → different reduction order),
# and non-deterministic CUDA (cudnn autotune, split-KV reduction ordering)
# amplifies that delta enough to flip argmax.  When it flips to token 248044
# ("<|im_start|>"), MTP verify rejects every draft token (accept_rate=0.0),
# pure-greedy decode falls into a self-consistent loop attractor.
# Forcing deterministic CUDA (cudnn.deterministic=True, cudnn.benchmark=False,
# CUBLAS_WORKSPACE_CONFIG=:4096:8) stabilises the staged/pinned delta below
# argmax-flip threshold and eliminates the degradation (oracle: 32/32
# token_ids bit-exact match between staged and pinned arms, MTP verify
# first_mismatch_step=4 every round — perfect accept).
# Perf cost on V100 is small (cudnn autotune disabled) but not zero, so this
# is opt-in via env var.  Production can set SGLANG_FORCE_DETERMINISTIC=1
# if reproducibility outweighs marginal throughput.
if os.environ.get("SGLANG_FORCE_DETERMINISTIC") == "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("[SGLANG_FORCE_DETERMINISTIC] CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn.deterministic=True, benchmark=False, tf32=False", flush=True)

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def run_server(server_args):
    """Run the server based on the gRPC flags and server_args.encoder_only."""
    # smg-grpc-servicer reads the raw `server_args.tokenizer_path`, which
    # resolution defaults into the stash but never writes back. Seed it here,
    # before resolve_once() seals the record.
    if (
        server_args.smg_grpc_mode or server_args.grpc_mode
    ) and server_args.tokenizer_path is None:
        server_args.tokenizer_path = server_args.model_path

    # The flags dispatched on below are decided by resolution (`--grpc-mode`
    # folds into `smg_grpc_mode`), and `prepare_server_args` returns raw input.
    server_args.resolve_once()
    cfg = resolving_view(server_args)

    if cfg.encoder_only:
        # For encoder disaggregation
        if cfg.smg_grpc_mode or cfg.grpc_mode:
            from sglang.srt.disaggregation.encoder.grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encoder.http_server import launch_server

            launch_server(server_args)
    elif cfg.smg_grpc_mode:
        # Legacy SMG gRPC server (--smg-grpc-mode, or the deprecated --grpc-mode
        # which __post_init__ folds into smg_grpc_mode). The native Rust gRPC
        # server is a separate path, enabled by --grpc-port, that starts
        # alongside the default HTTP server below.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif cfg.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
