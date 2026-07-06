"""Market environment with multi-retailer simulation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ── Data Models ────────────────────────────────────────────────────────────

class RetailerState(BaseModel):
    """State of a single retailer."""

    id: str
    inventory: float
    capital: float
    price: float = 0.0
    order_qty: int = 0


class MarketConfig(BaseModel):
    """Configuration for the market environment."""

    name: str = ""
    description: str = ""
    n_retailers: int
    base_demand: float = 1000.0
    price_sensitivity: float = 50.0
    noise_std: float = 10.0
    unit_cost: float = 8.0
    holding_cost: float = 0.1
    stockout_penalty: float = 2.0
    info_mode: str = "full"
    max_steps: int = 200
    decision_interval: int = 1
    brand_noise: float = 0.0


class Action(BaseModel):
    """Action taken by a retailer in a single step."""

    price: float = Field(ge=0)
    order_qty: int = Field(ge=0)


class Observation(BaseModel):
    """Observation returned to a retailer after a step."""

    self_state: RetailerState
    step: int
    competitors_info: dict
    market_total_inventory: float = 0.0
    my_market_share: float = 0.0


# ── MarketEnv ──────────────────────────────────────────────────────────────

class MarketEnv:
    """Multi-retailer market simulation environment."""

    def __init__(self, config: MarketConfig) -> None:
        self.config: MarketConfig = config
        self.step_counter: int = 0
        self.retailers: dict[str, RetailerState] = {}
        self.current_prices: dict[str, float] = {}
        self._last_total_demand: float = 0.0
        self._last_sales: dict[str, float] = {}
        self._init_retailers()
        self._rng: np.random.Generator = np.random.default_rng()

    def _init_retailers(self) -> None:
        """Create initial retailer states."""
        self.retailers = {}
        for i in range(self.config.n_retailers):
            rid = f"r_{i}"
            init_price = self.config.unit_cost * 1.5
            self.retailers[rid] = RetailerState(
                id=rid,
                inventory=200.0,
                capital=10000.0,
                price=init_price,
                order_qty=0,
            )
            self.current_prices[rid] = init_price

    # ── Public API ──────────────────────────────────────────────────────

    def reset(self) -> list[Observation]:
        """Reset environment to initial state and return observations."""
        self.step_counter = 0
        self._rng = np.random.default_rng()
        self._last_total_demand = 0.0
        self._last_sales = {}
        self._init_retailers()
        return self._build_observations()

    def step(
        self, actions: dict[str, Action]
    ) -> tuple[list[Observation], dict]:
        """
        Execute one simulation step.

        Args:
            actions: Mapping from retailer id to Action.

        Returns:
            Tuple of (observations, info).
        """
        # 1. Update prices (only on decision steps) and order_qty (always)
        is_decision_step = (
            self.step_counter % self.config.decision_interval == 0
        )
        for rid, action in actions.items():
            if rid not in self.retailers:
                continue
            if is_decision_step:
                self.current_prices[rid] = action.price
            self.retailers[rid].order_qty = action.order_qty

        # Sync retailer prices from current_prices
        for rid in self.retailers:
            self.retailers[rid].price = self.current_prices[rid]

        # Snapshot inventories before this step (used for profit calc)
        prices = np.array(
            [self.current_prices[rid] for rid in self.retailers]
        )
        inventories_before = np.array(
            [r.inventory for r in self.retailers.values()]
        )
        retailer_ids = list(self.retailers.keys())

        # 2. Calculate total market demand
        avg_price = float(np.mean(prices))
        noise = self._rng.normal(0.0, self.config.noise_std)
        Q_total: float = max(
            0.0,
            self.config.base_demand
            - self.config.price_sensitivity * avg_price
            + noise,
        )

        # 3. Allocate demand via attractiveness (1/price * brand_noise)
        safe_prices = np.maximum(prices, 1e-6)
        base_attract = np.where(inventories_before > 0, 1.0 / safe_prices, 0.0)
        # Random brand preference perturbation
        if self.config.brand_noise > 0:
            brand_factors = self._rng.uniform(
                1.0 - self.config.brand_noise,
                1.0 + self.config.brand_noise,
                size=self.config.n_retailers,
            )
        else:
            brand_factors = np.ones(self.config.n_retailers)
        attractiveness = base_attract * brand_factors
        total_attract = attractiveness.sum()

        if total_attract > 0:
            allocated_demand = Q_total * attractiveness / total_attract
        else:
            allocated_demand = np.zeros(self.config.n_retailers)

        # 4. Actual sales = min(allocated, inventory)
        actual_sales = np.minimum(allocated_demand, inventories_before)
        shortfalls = allocated_demand - actual_sales

        # 5. Profit calculation
        revenue = prices * actual_sales
        cost_of_goods = self.config.unit_cost * actual_sales
        holding_costs = (
            self.config.holding_cost * (inventories_before - actual_sales)
        )
        stockout_penalties = self.config.stockout_penalty * shortfalls
        profits = revenue - cost_of_goods - holding_costs - stockout_penalties

        # 6 & 7. Update inventory and capital
        for idx, rid in enumerate(retailer_ids):
            r = self.retailers[rid]
            r.inventory = float(r.inventory - actual_sales[idx] + r.order_qty)
            r.capital += float(profits[idx])

        # Store sales and total demand for observation
        self._last_sales = {
            rid: float(actual_sales[idx])
            for idx, rid in enumerate(retailer_ids)
        }
        self._last_total_demand = Q_total

        # Increment step counter
        self.step_counter += 1

        # Log
        logger.info(
            "step_completed",
            step=self.step_counter,
            avg_price=round(avg_price, 4),
            total_demand=round(Q_total, 4),
        )

        # 8. Build returns
        observations = self._build_observations()
        sales_dict: dict[str, float] = dict(self._last_sales)
        info: dict = {
            "avg_price": avg_price,
            "total_demand": Q_total,
            "step": self.step_counter,
            "sales": sales_dict,
        }
        return observations, info

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_observations(self) -> list[Observation]:
        """Construct Observation for every retailer based on info_mode."""
        observations: list[Observation] = []
        full_mode = self.config.info_mode == "full"

        # ── Market-level aggregates ──
        total_inventory = float(sum(r.inventory for r in self.retailers.values()))

        if full_mode:
            all_states = {
                r.id: {"id": r.id, "price": r.price, "inventory": r.inventory}
                for r in self.retailers.values()
            }

        for rid, r in self.retailers.items():
            if full_mode:
                competitors_info = {
                    k: v for k, v in all_states.items() if k != rid
                }
            else:
                others = [
                    s for oid, s in self.retailers.items() if oid != rid
                ]
                if others:
                    avg_mkt_price = float(np.mean([s.price for s in others]))
                    total_mkt_inv = float(sum(s.inventory for s in others))
                else:
                    avg_mkt_price = 0.0
                    total_mkt_inv = 0.0
                competitors_info = {
                    "avg_market_price": avg_mkt_price,
                    "total_market_inventory": total_mkt_inv,
                }

            # My market share = last round sales / total demand
            my_sales = self._last_sales.get(rid, 0.0)
            my_share = (
                my_sales / self._last_total_demand
                if self._last_total_demand > 0
                else 0.0
            )

            observations.append(
                Observation(
                    self_state=r,
                    step=self.step_counter,
                    competitors_info=competitors_info,
                    market_total_inventory=total_inventory,
                    my_market_share=my_share,
                )
            )

        return observations
