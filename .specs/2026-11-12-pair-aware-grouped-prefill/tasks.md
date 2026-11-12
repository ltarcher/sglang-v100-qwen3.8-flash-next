# Tasks: PAIR 感知 SM70 Grouped Prefill（路线 B）

按依赖排序；1-5 今日完成（CPU 可验证），6 为重启窗口 GPU 验收。

## 1. 内核 + planner：`_kernels_grouped_sparse_prefill.py`
- [x] `main` 签名追加 KPeer/VPeer/Owner；4 处池加载点改 Owner 分支双指针
- [x] planner：`slot_table`/`owner_mask` 参数、owner-bit 拆解、4 元组返回、
      owner 张量构建、docstring
- [x] `run_grouped_sparse_prefill(peer_kv=...)`
- [x] 模块 docstring PAIR 段

## 2. 后端：`qwen_sparse_attn_backend.py`
- [x] `_forward_sm70_grouped_sparse_prefill`：`layer` 参数 + tier3 PAIR 分支
      （非 PAIR 阻断 / PAIR 解析 metadata + pair_peer_kv + imported_as_tensor）
- [x] 调用点门控改为仅 env 判断 + 传 `layer=layer`；L2082 注释更新

## 3. 测试更新与新增（先写测试，红→绿）
- [x] test_qwen_sparse_sm70_routing.py：phase3+非 PAIR 阻断 / phase3+PAIR 放行
- [x] test_grouped_union_plan_cpu.py：4 元组解包 + slot_table/owner 用例
- [x] test_qsa_grouped_prefill_gate.py：断言签名更新（+ 修复 HEAD 上
      `_kv_host_tier` stub 缺失导致的 9 个既有失败）
- [x] test_grouped_sparse_prefill_sm70.py：4 元组解包 + GPU peer 数值用例（延后执行）

## 4. CPU 测试全绿
- [x] 容器内 `/opt/venv/bin/python -m pytest`（宿主无 conda env）：
      54 passed（test_grouped_union_plan_cpu + test_qsa_grouped_prefill_gate
      + test_qwen_sparse_sm70_routing）

## 5. 文档同步 + pre-commit
- [x] PREFILL_LEVERS_PLAN.md（§8.3 新小节 + L2-b 注记）
- [x] KV_HOST_TIER_DESIGN.md §6.1 消费者列表
- [x] README.md 开关条件登记
- [x] lint：pinned 版本（ruff 0.15.1 / ruff-format / isort 7.0.0 / codespell
      2.4.1）对改动文件全过（宿主无 pre-commit 环境，容器内等价执行）

## 6. GPU 验收（重启窗口，2026-11-12 已执行）
- [x] test_grouped_sparse_prefill_sm70.py：2 passed（TileLang 双指针编译 +
      peer 数值对齐；顺带修复 HEAD 上 mask[2,...] 越界的既有断言缺陷）
- [x] smoke_v100.sh 通过（group-prefill 镜像）
- [x] 重建镜像（用户构建 sglang-v100-unified:group-prefill）→ compose 重启 →
      前缀缓存门/needle QA 16k/32k PASS（PAIR 寻址正确）
- [x] A/B 实测：**回归，已回退**——16k conc=1 +62%、conc=2 +136%、8k +29%、
      4k 持平；profiler 定位 = 对家 union 读走 IPC 映射 latency-bound
      （内核 50-95ms/launch vs 全本地估 15ms）；L2-b 锚点为 PAIR 前拓扑
- [x] 生产已回退 GROUPED=0 并验证基线恢复（16k 2-conc TTFT 10.1s ✓）；
      结论与后续选项登记 PREFILL_LEVERS_PLAN.md §8.3 + README
