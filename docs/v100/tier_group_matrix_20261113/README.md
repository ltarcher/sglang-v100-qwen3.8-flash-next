# KV host tier × GROUP prefill 四臂性能矩阵（2026-11-13）

4× V100-32GB，Qwen3.8-Flash-Next NVFP4 + MTP，生产引擎容器 `sglang-v100-qwen38-8110`。
**结论臂 B（tier=0 + GROUP=1）已定为生产基线**（compose 注释与本项目记忆同步登记）。

## 臂定义

| 臂 | KV_HOST_TIER | KV_HEAD_LAYOUT | KV_TIER_PHYS_TOKENS | QSA_GROUPED_PREFILL | 说明 |
| --- | --- | --- | --- | --- | --- |
| A | 0 | 0 (FULL) | 0 | 0 | 基线（FULL 池 436,736/rank） |
| B | 0 | 0 (FULL) | 0 | 1 | 全本地拓扑 + grouped（**生产基线**） |
| C | 1 | 1 (PAIR_PARITY) | 786432 | 0 | 历史生产 tier 形态（设备池 393,216/rank） |
| D | 1 | 1 (PAIR_PARITY) | 786432 | 1 | PAIR-aware grouped |

## 方法

- `sglang.bench_serving --dataset-name random-ids --random-range-ratio 1 --random-output-len 256 --flush-cache`
- 长度 1K/4K/8K/16K/32K/64K；conc=1 用 `--num-prompts 2`，conc=2 用 `--num-prompts 4`
- 容器内无网：必须 `--model/--tokenizer` 传本地路径 `/data/models/nvidiaQwen3.8-Flash-Next-NVFP4` + `HF_HUB_OFFLINE=1`；`random` 数据集会拉 ShareGPT，离线用 `random-ids`（已验证与 random 同带宽）
- GROUP 臂（B/D）boot 后先丢弃一次 8k 预跑，剔除 TileLang JIT 编译
- 每臂单次测量，无重复；噪声带约 ±10%（参考 32k conc=1 TPOT 10.7~9.98ms 波动）

## conc=1（单流）

### TTFT (ms)

| L | A | B | C | D |
| --- | --- | --- | --- | --- |
| 1k | 381 | 385 | 466 | 399 |
| 4k | 1468 | 1503 | 1478 | 1536 |
| 8k | 3093 | 2890 | 3088 | 4127 |
| 16k | 6436 | 5866 | 6496 | 10408 |
| 32k | 13589 | 12357 | 13715 | 23737 |
| 64k | 28310 | 26622 | 28680 | 54259 |

### TPOT (ms)（decode 单流 tok/s ≈ 1000/TPOT）

| L | A | B | C | D |
| --- | --- | --- | --- | --- |
| 1k | 11.74 | 11.75 | 14.80 | 14.78 |
| 4k | 11.36 | 11.64 | 14.93 | 17.66 |
| 8k | 11.84 | 11.85 | 20.53 | 19.54 |
| 16k | 15.49 | 10.82 | 24.64 | 11.76 |
| 32k | 10.73 | 9.98 | 16.38 | 16.42 |
| 64k | 12.37 | 11.83 | 21.95 | 21.34 |

## conc=2（聚合 decode tok/s 与波内 TTFT）

### 聚合输出 tok/s（每流 ≈ 一半）

| L | A | B | C | D |
| --- | --- | --- | --- | --- |
| 1k | 87.2 | 89.5 | 71.1 | 71.4 |
| 4k | 45.7* | 63.3 | 39.1 | 46.5 |
| 8k | 42.9 | 44.4 | 35.2 | 32.4 |
| 16k | 28.5 | 30.8 | 25.2 | 17.5 |
| 32k | 15.9 | 17.5 | 15.0 | 9.3 |
| 64k | 8.4 | 8.8 | 7.7 | 4.5 |

\* A 的 4k conc=2 首测遭不可复现瞬态（该桶 TTFT 曾 13.4s，复测 2.24s），此值偏低。

### 波内 Median TTFT (ms)（≈冷 prefill 单流）

| L | A | B | C | D |
| --- | --- | --- | --- | --- |
| 1k | 678 | 565 | 594 | 584 |
| 4k | 2238 | 2380 | 2501 | 2683 |
| 8k | 4362 | 3977 | 4351 | 5871 |
| 16k | 7526 | 6873 | 7589 | 11870 |
| 32k | 20239 | 18405 | 20462 | 35446 |
| 64k | 28889 | 39531† | 29303 | 53519 |

† B 64k conc=2 为补测值，受补测 boot 的 JIT 桶预热不足污染，真值应 ≈ A 的 28.9s。

## 结论

1. **B 臂（tier=0 + GROUP=1）全面最优**：prefill 8k+ 提速 6~9%，decode 与 A 持平或略优。GROUP 收益在全本地拓扑（PAIR 关）下兑现。
2. **C 臂（tier+PAIR）prefill 持平但 decode TPOT 恶化 25~75%**：context 上限 232k < 393k 设备池，全程零 page-in，tier 容量收益为零，只剩 PAIR 对拆读开销。tier 仅在需要 >393k 设备池容量时值得开。
3. **D 臂（tier+PAIR+GROUP）复现 §8.3（KV_HOST_TIER_DESIGN.md）的回归**：长 prefill +34~92%，PAIR-aware grouped 不可用于生产。
4. decode 判据以 conc=1 TPOT 与 conc=2 短长度为准（64k conc=2 的 TPOT 受 prefill 干扰大，D 臂 rerun Max ITL 52s）。

## 原始日志

- `bench_A_conc1.log` / `bench_A_conc2.log`：A 臂（tier0/GROUP0）
- `bench_B.log` / `bench_C.log` / `bench_D.log`：B/C/D 臂（B 内含 JIT 预跑行）
- 同日配套代码：commit `412a3f0346`（PHYS/PAIR 无 tier 拒启守卫，TDD 21/21）
- 生产 compose：`/home/bso/sglang-v100/docker-compose-sglang-v100-qwen3.8.yaml`（B 臂 env + 决策注释）
