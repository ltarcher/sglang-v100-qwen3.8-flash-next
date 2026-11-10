# Debug Session: long-prompt-nondeterminism

**Session ID**: long-prompt-nondeterminism
**PROJECT_ROOT**: /data/develop/sglang-v100-sxm2-qwen3.8-flash-next
**DEBUG_EVIDENCE_DIR**: /tmp/debug-long-prompt-nondeterminism

## 症状

~33k token long_prefill 在 oracle 4×V100 真机上极罕见地进入 248044/248045
自洽循环。退化时 MTP accept_rate=0.0（verify 完全失效、纯 greedy decode），
正常时 accept_rate=0.35-0.53。**退化只出现在 staged 臂、pinned 永远正常**。

## 关键发现（2026-11-10 重算退化率）

**之前文档误判 stat2a 为退化**——stat2a 是连贯输出与 pinned 分叉，
不是 248044/248045 循环退化。正确退化率：

| 批次 | depth | staged 退化 | long_prefill 特征 |
|---|---|---|---|
| stat0a | 0 | ✅ 退化 | 248044/248045 循环 len=31 |
| stat0b | 0 | ❌ 正常 | 连贯输出 |
| bench_d0 | 0 | — | bench 无 long_prefill 用例 |
| stat2a | 2 | ❌ 正常 | 连贯输出（误判已纠正） |
| stat2b | 2 | ❌ 正常 | 连贯输出 |
| d2fix | 2 | ❌ 正常 | 连贯输出 |
| bench_d1/d2 | 1/2 | — | bench 无 long_prefill 用例 |

**退化率**: depth=0 staged 1/3（33%），depth=2 staged 0/4（0%）。
退化仅在 depth=0 出现——可能是 depth=0 staged 的边缘 bug，而非
chunk-ahead 引入的问题。depth=2（chunk-ahead 开启）反而稳定。

## 已拒绝假设

- **H1 CUDA 全局非确定性归约**: REJECTED。如果是全局 CUDA 非确定，
  pinned 臂也应该退化。实际 pinned 永远正常。
- **H5 staged 固有数值差异**: 部分被淡化。如果是 staged 固有差异，
  depth=0 和 depth=2 应该都退化；实际 depth=2 零退化。

## 当前假设

- **H6 (MAIN) MTP verify 与 staged prefill 交互的 depth=0 边缘 bug**
  depth=0 staged 没有 chunk-ahead，所有 prefill chunk 走同步 fill→serve，
  最后一个 partial chunk 走 prefill 主路径。在某种启动时序（kernel JIT
  首次编译顺序、PLE table materialization 完成时机与 decode 图 capture
  的边界）下，decode 起始的 MTP verify 拿到了错误的 prefill KV 状态，
  导致 accept_rate=0.0、纯 greedy 进入自洽循环。**预测**: 增大
  --max-mamba-cache-size、禁用 radix cache、或加 prefill→decode
  barrier 后退化消失；或 depth=0 staged 连跑 10 次退化率下降/消失。
- **H4 mamba state / radix cache 竞态**: INCONCLUSIVE。depth=0 staged
  的 mamba state pool 在首次 decode 前可能存在竞态（pool max=20 紧张），
  而 depth=2 因 chunk-ahead 预填充额外消耗了 state pool、反而"稀释"
  竞态。**预测**: --max-mamba-cache-size 64 后退化消失。
- **H2 kernel JIT 编译时序**: INCONCLUSIVE 但降级。首次 boot 的 JIT
  编译顺序可能不同，但 depth=2 同样有首次编译却不退化。**预测**:
  预热后再跑退化率消失。

## 证据链总结

1. chunk-ahead 与退化无关：退化出现在 depth=0 control arm
2. staged 数据面正确：gate self-check 逐位 + spot-check 逐字节双重证实
3. MTP verify 在退化 boot 失效（accept_rate=0.0）
4. 退化仅 depth=0（无 chunk-ahead）、depth=2（有 chunk-ahead）零退化
5. 退化率低（1/3 depth=0 staged），更像是边缘 bug 而非系统性非确定

## 状态

- progress_status: evidence-sufficient-for-tradeoff
- 退化是独立于 chunk-ahead 的问题，chunk-ahead 已过 gate 且 BENCH PASS。
- 建议：**不阻塞 chunk-ahead 上线**，退化另立 ticket 跟踪。
  但如果要继续追，下一步是 depth=0 staged 连跑 6-10 次看退化率是否稳定，
  或在 mamba state pool / radix cache 加 1-2 个 barrier 后退化是否消失。
