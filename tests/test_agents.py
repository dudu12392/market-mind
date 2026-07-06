"""Tests for rule-based agents."""

import pytest

from src.agents import CostPlusAgent, MatchLowestAgent, RandomAgent
from src.environment import (
    Action,
    MarketConfig,
    MarketEnv,
    Observation,
    RetailerState,
)

UNIT_COST = 8.0


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def env_3() -> MarketEnv:
    return MarketEnv(
        MarketConfig(n_retailers=3, base_demand=1000, price_sensitivity=50)
    )


@pytest.fixture
def full_obs_list(env_3: MarketEnv) -> list[Observation]:
    """Run one step so retailers have non-zero prices and competitor data."""
    env_3.reset()
    actions = {
        "r_0": Action(price=10.0, order_qty=50),
        "r_1": Action(price=12.0, order_qty=30),
        "r_2": Action(price=14.0, order_qty=20),
    }
    obs_list, _ = env_3.step(actions)
    return obs_list


# ── Tests ───────────────────────────────────────────────────────────────────

class TestRandomAgent:
    def test_returns_valid_action(self) -> None:
        agent = RandomAgent("r_0", unit_cost=UNIT_COST)
        obs = Observation(
            self_state=RetailerState(
                id="r_0", inventory=200.0, capital=10000.0
            ),
            step=0,
            competitors_info={},
        )
        action = agent.act(obs)
        assert isinstance(action, Action)
        assert action.price > 0
        assert action.order_qty >= 0

    def test_price_in_range_over_many_calls(self) -> None:
        agent = RandomAgent("r_0", unit_cost=UNIT_COST)
        obs = Observation(
            self_state=RetailerState(
                id="r_0", inventory=200.0, capital=10000.0
            ),
            step=0,
            competitors_info={},
        )
        for _ in range(200):
            action = agent.act(obs)
            assert UNIT_COST <= action.price <= UNIT_COST * 2
            assert 20 <= action.order_qty <= 100


class TestCostPlusAgent:
    def test_default_margin(self) -> None:
        agent = CostPlusAgent("r_1", unit_cost=UNIT_COST)
        obs = Observation(
            self_state=RetailerState(
                id="r_1", inventory=80.0, capital=10000.0
            ),
            step=1,
            competitors_info={},
        )
        action = agent.act(obs)
        # price = unit_cost * (1 + 0.3) = 10.4
        assert action.price == pytest.approx(10.4)
        # inventory 80 < target 100, should order 20
        assert action.order_qty == 20

    def test_custom_margin(self) -> None:
        agent = CostPlusAgent("r_1", unit_cost=UNIT_COST, margin=0.5)
        obs = Observation(
            self_state=RetailerState(
                id="r_1", inventory=200.0, capital=10000.0
            ),
            step=1,
            competitors_info={},
        )
        action = agent.act(obs)
        assert action.price == pytest.approx(12.0)

    def test_no_order_when_inventory_above_target(self) -> None:
        agent = CostPlusAgent("r_1", unit_cost=UNIT_COST)
        obs = Observation(
            self_state=RetailerState(
                id="r_1", inventory=150.0, capital=10000.0
            ),
            step=1,
            competitors_info={},
        )
        action = agent.act(obs)
        assert action.order_qty == 0

    def test_returns_valid_action(self) -> None:
        agent = CostPlusAgent("r_2", unit_cost=UNIT_COST)
        obs = Observation(
            self_state=RetailerState(
                id="r_2", inventory=50.0, capital=10000.0
            ),
            step=1,
            competitors_info={},
        )
        action = agent.act(obs)
        assert isinstance(action, Action)
        assert action.price > 0
        assert action.order_qty >= 0


class TestMatchLowestAgent:
    def test_undercuts_competition(self, full_obs_list: list[Observation]) -> None:
        """r_0 faces r_1(12.0) and r_2(14.0); should price at 12*0.95=11.4."""
        agent = MatchLowestAgent("r_0", unit_cost=UNIT_COST)
        obs_r0 = [o for o in full_obs_list if o.self_state.id == "r_0"][0]
        action = agent.act(obs_r0)

        assert action.price == pytest.approx(11.4)  # 12.0 * 0.95
        assert action.price >= UNIT_COST * 1.02

    def test_price_not_below_cost_floor(self) -> None:
        """Even if competitor prices are very low, never go below cost*1.02."""
        agent = MatchLowestAgent("r_0", unit_cost=UNIT_COST)
        # Simulate a competitor pricing way below cost
        obs = Observation(
            self_state=RetailerState(
                id="r_0", inventory=100.0, capital=10000.0
            ),
            step=1,
            competitors_info={
                "r_1": {"id": "r_1", "price": 2.0, "inventory": 200.0},
            },
        )
        action = agent.act(obs)
        # min_comp * 0.95 = 1.9, but floor is unit_cost * 1.02 = 8.16
        assert action.price == pytest.approx(UNIT_COST * 1.02)

    def test_partial_info_fallback(self) -> None:
        """When no per-competitor prices available, fall back to cost * 1.5."""
        agent = MatchLowestAgent("r_0", unit_cost=UNIT_COST)
        obs = Observation(
            self_state=RetailerState(
                id="r_0", inventory=100.0, capital=10000.0
            ),
            step=1,
            competitors_info={
                "avg_market_price": 11.0,
                "total_market_inventory": 400.0,
            },
        )
        action = agent.act(obs)
        assert action.price == pytest.approx(UNIT_COST * 1.5)

    def test_returns_valid_action(self, full_obs_list: list[Observation]) -> None:
        agent = MatchLowestAgent("r_2", unit_cost=UNIT_COST)
        obs_r2 = [o for o in full_obs_list if o.self_state.id == "r_2"][0]
        action = agent.act(obs_r2)

        assert isinstance(action, Action)
        assert action.price >= UNIT_COST * 1.02
        assert action.order_qty >= 0


# ── Integration: agents + env ───────────────────────────────────────────────

def test_all_agents_in_env_loop(env_3: MarketEnv) -> None:
    """Run 10 steps with all three agent types; no crashes, valid outputs."""
    obs_list = env_3.reset()

    agents = {
        "r_0": RandomAgent("r_0", unit_cost=UNIT_COST),
        "r_1": CostPlusAgent("r_1", unit_cost=UNIT_COST),
        "r_2": MatchLowestAgent("r_2", unit_cost=UNIT_COST),
    }

    for _ in range(10):
        actions = {}
        for rid, agent in agents.items():
            obs_for = [o for o in obs_list if o.self_state.id == rid][0]
            action = agent.act(obs_for)
            # Every action must be valid
            assert action.price > 0
            assert action.order_qty >= 0
            actions[rid] = action

        obs_list, info = env_3.step(actions)
        assert info["step"] > 0
