"""Rule-based agent implementations."""

from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent
from src.environment import Action, Observation


class RandomAgent(BaseAgent):
    """Randomly chooses price and order_qty within fixed bounds."""

    def __init__(self, agent_id: str, unit_cost: float) -> None:
        super().__init__(agent_id)
        self.unit_cost: float = unit_cost
        self._rng: np.random.Generator = np.random.default_rng()

    def act(self, obs: Observation) -> Action:
        price = self._rng.uniform(self.unit_cost, self.unit_cost * 2)
        order_qty = self._rng.integers(20, 101)
        return Action(price=float(price), order_qty=int(order_qty))


class CostPlusAgent(BaseAgent):
    """Prices at unit_cost * (1 + margin); orders to maintain target inventory."""

    def __init__(
        self,
        agent_id: str,
        unit_cost: float,
        margin: float = 0.3,
        target_inventory: float = 100.0,
    ) -> None:
        super().__init__(agent_id)
        self.unit_cost: float = unit_cost
        self.margin: float = margin
        self.target_inventory: float = target_inventory

    def act(self, obs: Observation) -> Action:
        price = self.unit_cost * (1.0 + self.margin)

        current_inv = obs.self_state.inventory
        if current_inv >= self.target_inventory:
            order_qty = 0
        else:
            order_qty = int(self.target_inventory - current_inv)

        return Action(price=float(price), order_qty=order_qty)


class MatchLowestAgent(BaseAgent):
    """Undercuts the lowest competitor price by 5%, floor at unit_cost * 1.02."""

    def __init__(
        self,
        agent_id: str,
        unit_cost: float,
        target_inventory: float = 100.0,
    ) -> None:
        super().__init__(agent_id)
        self.unit_cost: float = unit_cost
        self.target_inventory: float = target_inventory

    def act(self, obs: Observation) -> Action:
        comp = obs.competitors_info

        # Try to extract competitor prices
        competitor_prices: list[float] = []
        for v in comp.values():
            if isinstance(v, dict) and "price" in v:
                competitor_prices.append(float(v["price"]))

        if competitor_prices:
            min_comp_price = min(competitor_prices)
            price = max(min_comp_price * 0.95, self.unit_cost * 1.02)
        else:
            # Partial info mode — no per-competitor prices available
            price = self.unit_cost * 1.5

        # Inventory-reactive ordering
        current_inv = obs.self_state.inventory
        order_qty = max(0, int(self.target_inventory - current_inv))

        return Action(price=float(price), order_qty=order_qty)
