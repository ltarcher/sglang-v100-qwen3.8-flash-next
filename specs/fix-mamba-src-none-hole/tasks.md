# Tasks: 修复 mamba 匹配 `src=None` 仍服 Full KV 前缀

- [x] T1 红灯测试：在 `test/registered/unit/mem_cache/test_mamba_component_caching_boundary.py` 新增
  `TestMambaMatchClipsToReusableState`（cfg + `_insert` helper + 两个用例 + 状态持有 req 的护栏用例），
  运行确认失败（期望 0/实际 64；期望 704/实际 960）。
- [x] T2 实现：`mamba.py` 新增 `_clip_to_reusable_state`，`finalize_match_result_in_tree_core`
  在 `not enable_hicache` 且请求不持有状态时调用；更新 device-only validator 注释。
- [x] T3 绿：`test_mamba_component_caching_boundary.py` 全绿；`TestMambaCheckpointGrid` /
  `TestMambaFinishedOvershootCheckpoint` / `test_mamba_path_state_cap` /
  `test_mamba_donated_alloc_ratio` / `test_radix_page_reuse_poison` 共 47 例 OK
  （1 个既有 expected failure）。mamba 相关类中仅剩与基线相同的 HiCache 环境失败
  （host 注册，另一容器并发占卡）；`test_mamba_evict_breaks_match` 由红转绿。
  存量用例 `test_finished_insert_lands_in_tree_and_survives_cleanup` 改为按生产路径
  （`cache_finished_req` 先按 effective len 截断 key）构造，原构造把 48 深状态挂在 64 长节点上。
- [x] T4 容器复现（chunk=704）：`src=None` 行全部变为 `serve=0 full_hit=...`；
  `serve>0 && src=None` 计数 0；B1==B2 22/32、A==B1 18/32、长度与 A 同量级（无 2–3× 膨胀）。
- [x] T5 生产回归（chunk 4096）：恢复 compose 并重建容器（重放 10 个实验补丁），
  healthy；smoke 通过；4096 下命中日志仍为 `serve=704 full_hit=704 src=3`（未被误裁），
  `serve>0 && src=None` 计数 0；warm determinism cached 704/768、B1==B2 16/32（历史 11–19/32 区间）。
  性能 A/B（同容器、同配置、改前后各一次 boot）：prefill 8192 tok/s 1794 vs 898，
  decode 16.3 vs 11.2/11.5 tok/s（客户端口径，服务端 accept len 2.25–2.83），未回退；
  绝对数字低于 README 基线是当前机器负载所致，与本次改动无关。
- [x] T6 交付说明：改动文件、测试证据、复现前后日志与性能 A/B 已汇总给用户（未创建 commit）。
