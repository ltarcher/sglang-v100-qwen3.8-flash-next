"""Unit test: host KV tier capacity-extension gates in kv_cache_configurator.

``_tier_extend_logical_capacity`` honors --max-total-tokens beyond the GPU
profile only when the tier is on, a PHYS budget is set, and the kill-switch
(SGLANG_V100_KV_TIER_DISABLE_EXTEND) is clear; the host-memory budget
clamps the request. These gates decide whether boot keeps the profiled
device capacity or pays the excess in host RAM -- wrong gating either
silently shrinks the served context or OOMs the host, so every branch is
pinned here with a stubbed configurator (no GPU, no model).
"""

from types import SimpleNamespace


import sglang.srt.mem_cache.kv_cache_configurator as kkc
import sglang.srt.mem_cache.pool_host.base as pool_base
import sglang.srt.model_executor.pool_configurator as pc
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

PROFILE_TOKENS = 1000
CELL = 2048  # bytes per logical token
AVAILABLE = 4096 * CELL
_UNLIMITED_HOST = 10**9


class _Cfg:
    def __init__(self, base_tokens):
        self.max_total_num_tokens = base_tokens


def _stub(schedule_tokens, tier=1, phys=512, disable=0, host_tokens=_UNLIMITED_HOST):
    """Run _tier_extend_logical_capacity against a stubbed configurator.

    The budget->capacity stub models an affine pool: the device budget
    ``AVAILABLE`` yields the profiled capacity, and each extra CELL bytes
    buys one more logical token.
    """
    env = SimpleNamespace(
        SGLANG_V100_KV_HOST_TIER=SimpleNamespace(get=lambda: tier),
        SGLANG_V100_KV_TIER_PHYS_TOKENS=SimpleNamespace(get=lambda: phys),
        SGLANG_V100_KV_TIER_DISABLE_EXTEND=SimpleNamespace(get=lambda: disable),
    )
    stub = SimpleNamespace(
        config_from_budget=lambda budget: _Cfg(
            PROFILE_TOKENS + max(0, budget - AVAILABLE) // CELL
        )
    )
    ext = KVCacheConfigurator._tier_extend_logical_capacity

    saved = (kkc.get_schedule, kkc.envs, pool_base.host_memory_budget_bytes,
             pc.create_memory_pool_configurator)
    kkc.get_schedule = lambda: SimpleNamespace(max_total_tokens=schedule_tokens)
    kkc.envs = env
    pool_base.host_memory_budget_bytes = lambda: host_tokens * CELL
    pc.create_memory_pool_configurator = lambda self: SimpleNamespace(cell_size=CELL)
    try:
        return ext(stub, _Cfg(PROFILE_TOKENS), AVAILABLE)
    finally:
        (kkc.get_schedule, kkc.envs, pool_base.host_memory_budget_bytes,
         pc.create_memory_pool_configurator) = saved


class TestTierExtendGates:
    def test_no_user_limit_keeps_profile(self):
        assert _stub(None).max_total_num_tokens == PROFILE_TOKENS

    def test_user_limit_within_profile_keeps_profile(self):
        assert _stub(PROFILE_TOKENS).max_total_num_tokens == PROFILE_TOKENS

    def test_tier_off_keeps_profile(self):
        assert _stub(2000, tier=0).max_total_num_tokens == PROFILE_TOKENS

    def test_no_phys_budget_keeps_profile(self):
        assert _stub(2000, phys=0).max_total_num_tokens == PROFILE_TOKENS

    def test_disable_extend_kill_switch_keeps_profile(self):
        assert _stub(2000, disable=1).max_total_num_tokens == PROFILE_TOKENS

    def test_extension_reaches_user_limit(self):
        assert _stub(2000).max_total_num_tokens == 2000

    def test_host_budget_clamps_the_request(self):
        assert _stub(5000, host_tokens=3000).max_total_num_tokens == 3000

    def test_host_budget_below_profile_keeps_profile(self):
        assert _stub(5000, host_tokens=PROFILE_TOKENS).max_total_num_tokens == (
            PROFILE_TOKENS
        )
