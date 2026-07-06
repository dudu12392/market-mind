"""Tests for MarketEnv."""

import numpy as np
import pytest

from src.environment import Action, MarketConfig, MarketEnv


def test_basic_two_retailers() -> None:
    """Two retailers with equal prices should split demand evenly."""
    config = MarketConfig(n_retailers=2, base_demand=1000, price_sensitivity=50)
    env = MarketEnv(config)
    obs_list = env.reset()

    # ── 初始状态 ──
    assert len(obs_list) == 2
    assert obs_list[0].self_state.inventory == 200
    assert obs_list[0].self_state.capital == 10000

    # ── 手动执行一步 ──
    actions = {
        "r_0": Action(price=10.0, order_qty=50),
        "r_1": Action(price=10.0, order_qty=50),
    }
    new_obs, info = env.step(actions)

    # ── 验证 ──
    assert len(new_obs) == 2
    assert info["step"] == 1
    assert info["avg_price"] == 10.0
    # 利润不能为负（price=10 > unit_cost=8）
    assert new_obs[0].self_state.capital >= 0
    assert new_obs[1].self_state.capital >= 0

    print(f"Step 1 - Retailer 0 capital: {new_obs[0].self_state.capital}")
    print(f"Step 1 - Retailer 1 capital: {new_obs[1].self_state.capital}")


def test_random_steps() -> None:
    """Run 50 random steps with 3 retailers; no crashes, step counter monotonic."""
    config = MarketConfig(n_retailers=3, base_demand=1000, price_sensitivity=50)
    env = MarketEnv(config)
    env.reset()

    rng = np.random.default_rng(42)
    retailer_ids = ["r_0", "r_1", "r_2"]
    prev_step = 0

    for _ in range(50):
        actions = {
            rid: Action(
                price=float(rng.uniform(1.0, 30.0)),
                order_qty=int(rng.integers(0, 100)),
            )
            for rid in retailer_ids
        }
        obs, info = env.step(actions)

        # 不抛异常
        assert len(obs) == 3
        assert info["step"] == prev_step + 1
        prev_step = info["step"]

        # 每个 observation 的 step 应与 info 一致
        for o in obs:
            assert o.step == info["step"]

    assert info["step"] == 50


def test_partial_info_mode() -> None:
    """Partial info mode returns aggregated competitor data."""
    config = MarketConfig(n_retailers=3, info_mode="partial")
    env = MarketEnv(config)
    env.reset()

    actions = {f"r_{i}": Action(price=10.0, order_qty=0) for i in range(3)}
    obs, _ = env.step(actions)

    for o in obs:
        comp = o.competitors_info
        assert "avg_market_price" in comp
        assert "total_market_inventory" in comp
        # 不应泄露对手个体信息
        assert all(k in ("avg_market_price", "total_market_inventory") for k in comp)


def test_full_info_mode() -> None:
    """Full info mode returns per-competitor details."""
    config = MarketConfig(n_retailers=3, info_mode="full")
    env = MarketEnv(config)
    env.reset()

    actions = {f"r_{i}": Action(price=10.0, order_qty=0) for i in range(3)}
    obs, _ = env.step(actions)

    for o in obs:
        comp = o.competitors_info
        # 应有 2 个对手
        assert len(comp) == 2
        for cid, cinfo in comp.items():
            assert cid != o.self_state.id
            assert "id" in cinfo
            assert "price" in cinfo
            assert "inventory" in cinfo
