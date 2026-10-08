# Qwen3.8 Flash Next NVFP4: MTP, full context, and concurrent images

Measured September 25, 2026, on four V100-SXM2-32GB GPUs with
`geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4` and
`RadixArk/Qwen3.8-Flash-Next-NVFP4`. The current Docker setup is in the
[README](../../README.md#docker).
The test container used port 8083 and `HF_HUB_OFFLINE=1` with cached weights;
the documented command uses port 8082 and permits downloads.

The changed serving flags were:

```text
--enable-multimodal
--mem-fraction-static 0.84
--context-length 262144
--max-total-tokens 540672
--max-running-requests 4
--max-mamba-cache-size 24
--chunked-prefill-size 2048
--cuda-graph-max-bs 4
--cuda-graph-bs 1 2 4
--mamba-scheduler-strategy extra_buffer
--mamba-full-memory-ratio 0.2
--speculative-algorithm EAGLE
--speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

Startup reported `max_num_reqs=4`, 24 Mamba slots, 540,672 E5M2 KV-token
slots, and 1.80 GB/GPU for MTP draft weights. `--max-total-tokens` is a shared
pool across requests. Two full 262,144-token contexts need 524,288 slots,
leaving 16,384 for other live or cached tokens. Four requests can run at once
when their combined token demand fits the pool. The extra-buffer Mamba
scheduler needs five slots per running request, and 24 covers four.
The [server log](server.log.gz) records startup and all test runs.

| Requests | Input / output tokens each | Result | Full-run aggregate output tok/s | Raw result |
| ---: | ---: | --- | ---: | --- |
| 4 | 1,024 / 512 | 4 completed; four decoded concurrently | 199.66 | [c4_1k.jsonl](c4_1k.jsonl) |
| 4 | 8,192 / 1,024 | 4 completed; fourth briefly queued during prefill | 120.02 | [c4_8k.jsonl](c4_8k.jsonl) |
| 4 | 65,536 / 512 | 4 completed; four active during prefill | 25.25 | [c4_64k.jsonl](c4_64k.jsonl) |
| 2 | 261,120 / 512 | 2 completed; 523,184 live KV tokens reported during decode | 5.83 | [c2_261k.jsonl](c2_261k.jsonl) |

The long requests used 261,632 tokens each, 512 below the per-request context
limit. Their full-run rate includes a long prefill; once both were decoding,
the server logged about 172 aggregate output tokens/s. All rows kept MTP
enabled; the recorded speculative acceptance length was 2.82–3.83 tokens.

Four concurrent image requests with three generated 1536×1024 JPEG images
each returned HTTP 200 and correctly identified the colored rectangle in
their images. Each prompt used 7,644 tokens. GPU 0 peaked at 31,162 MiB used
(1,333 MiB free); the other ranks peaked at 30,330, 30,298, and 30,234 MiB.
`nvidia-smi --query-compute-apps` showed a separate `python -m
sglang.launch_server` process using about 810 MiB on GPU 0 in addition to
the TP0 scheduler. That accounts for most of the measured GPU 0 difference;
a 2–3 GB difference was not reproduced.
The images were synthetic and do not represent every aspect ratio or image
preprocessing path. The [test script](check_images.py) and
[result](images_c4_3each.json) contain the request setup and response metrics.

After flushing the cache, two simultaneous requests each sent three of the
same-size images plus enough text for **259,644 prompt tokens per request**.
Both returned HTTP 200 and the correct image color in two completion tokens.
GPU 0 peaked at 31,478 MiB (1,017 MiB free); the other ranks peaked at
30,646, 30,612, and 30,550 MiB. The full result is
[images_c2_nearfull_3each.json](images_c2_nearfull_3each.json). These prompts
were 2,500 tokens below the context limit, including image tokens, and the
server had 96% of its shared KV pool occupied near completion. Real images
with different sizes and workloads can produce different memory peaks.
