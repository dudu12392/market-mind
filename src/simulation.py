"""Simulation runner: episode loop, config loading, agent factory, demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import structlog
import yaml

from src.agents import BaseAgent, CostPlusAgent, LLMAgent, MatchLowestAgent, RandomAgent
from src.environment import Action, MarketConfig, MarketEnv

logger = structlog.get_logger(__name__)

# ── 1. run_episode ──────────────────────────────────────────────────────────


def run_episode(
    env: MarketEnv,
    agents: list[BaseAgent],
    max_steps: int = 200,
) -> list[dict[str, Any]]:
    """Run a full simulation episode and return per-step logs.

    Returns:
        List of dicts, each with keys:
        step, agent_id, price, order_qty, inventory, sales, profit, capital.
    """
    logs: list[dict[str, Any]] = []
    obs_list = env.reset()

    for step_idx in range(max_steps):
        # ── Build actions from each agent (safe-lookup, no StopIteration) ──
        actions: dict[str, Action] = {}
        capital_before: dict[str, float] = {}

        for agent in agents:
            obs = None
            for o in obs_list:
                if o.self_state.id == agent.agent_id:
                    obs = o
                    break
            if obs is None:
                logger.warning(
                    "agent_observation_missing",
                    agent_id=agent.agent_id,
                    step=step_idx,
                )
                continue
            capital_before[agent.agent_id] = obs.self_state.capital
            actions[agent.agent_id] = agent.act(obs)

        # ── Step with fallback ──
        try:
            obs_list, info = env.step(actions)
        except Exception:
            import traceback

            logger.error(
                "step_crashed",
                step=step_idx,
                traceback=traceback.format_exc(),
            )
            # Conservative fallback for all agents
            fallback_info: dict[str, Any] = {
                "step": env.step_counter + 1,
                "avg_price": env.config.unit_cost * 1.5,
                "total_demand": 0.0,
                "sales": {},
            }
            for agent in agents:
                logs.append(
                    {
                        "step": fallback_info["step"],
                        "agent_id": agent.agent_id,
                        "price": env.config.unit_cost * 1.5,
                        "order_qty": 50,
                        "inventory": 0.0,
                        "sales": 0.0,
                        "profit": 0.0,
                        "capital": capital_before.get(agent.agent_id, 0.0),
                    }
                )
            env.step_counter += 1
            continue

        # ── Record per-agent metrics (safe-lookup) ──
        sales_map: dict[str, float] = info.get("sales", {})
        for agent in agents:
            new_obs = None
            for o in obs_list:
                if o.self_state.id == agent.agent_id:
                    new_obs = o
                    break
            if new_obs is None:
                continue
            state = new_obs.self_state
            profit = state.capital - capital_before[agent.agent_id]

            logs.append(
                {
                    "step": info["step"],
                    "agent_id": agent.agent_id,
                    "price": state.price,
                    "order_qty": state.order_qty,
                    "inventory": state.inventory,
                    "sales": sales_map.get(agent.agent_id, 0.0),
                    "profit": profit,
                    "capital": state.capital,
                }
            )

        logger.info(
            "episode_step",
            step=info["step"],
            avg_price=round(info["avg_price"], 4),
            total_demand=round(info["total_demand"], 4),
        )

    return logs


# ── 2. load_config ──────────────────────────────────────────────────────────


def load_config(path: str | Path) -> MarketConfig:
    """Load MarketConfig from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data: dict = yaml.safe_load(f)
    return MarketConfig(**data)


# ── 3. create_agents ────────────────────────────────────────────────────────


def create_agents(config: MarketConfig, agent_types: list[str]) -> list[BaseAgent]:
    """Factory: instantiate agents by type string.

    Supported types: ``"random"``, ``"cost_plus"``, ``"match_lowest"``, ``"llm"``.
    """
    agents: list[BaseAgent] = []
    for i, atype in enumerate(agent_types):
        agent_id = f"r_{i}"
        if atype == "random":
            agents.append(RandomAgent(agent_id, unit_cost=config.unit_cost))
        elif atype == "cost_plus":
            agents.append(CostPlusAgent(agent_id, unit_cost=config.unit_cost))
        elif atype == "match_lowest":
            agents.append(MatchLowestAgent(agent_id, unit_cost=config.unit_cost))
        elif atype == "llm":
            agents.append(LLMAgent(agent_id, unit_cost=config.unit_cost))
        else:
            raise ValueError(f"Unknown agent type: {atype!r}")
    return agents


# ── 4. __main__ demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 4-Agent Battle: LLM vs Random vs CostPlus vs MatchLowest ──
    AGENT_TYPES = ["llm", "random", "cost_plus", "match_lowest"]
    LABELS = {
        "llm": "LLM (gpt-4o-mini)",
        "random": "Random",
        "cost_plus": "CostPlus",
        "match_lowest": "MatchLowest",
    }
    COLORS = {
        "llm": "#6366f1",
        "random": "#f59e0b",
        "cost_plus": "#10b981",
        "match_lowest": "#ef4444",
    }

    config = MarketConfig(
        n_retailers=len(AGENT_TYPES),
        base_demand=1000,
        price_sensitivity=50,
        noise_std=10,
        unit_cost=8.0,
        info_mode="full",
        max_steps=200,
    )
    agents = create_agents(config, AGENT_TYPES)
    env = MarketEnv(config)

    print(f"Starting 4-agent battle: {', '.join(AGENT_TYPES)}")
    print("LLM model: gpt-4o-mini  |  Steps: 100\n")

    logs = run_episode(env, agents, max_steps=100)

    # ── Plot ──
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    agent_ids = sorted({row["agent_id"] for row in logs})

    for aid in agent_ids:
        rows = [r for r in logs if r["agent_id"] == aid]
        steps = [r["step"] for r in rows]
        capitals = [r["capital"] for r in rows]
        atype = AGENT_TYPES[int(aid.split("_")[1])]
        ax.plot(
            steps,
            capitals,
            label=LABELS[atype],
            color=COLORS[atype],
            linewidth=2,
        )

    ax.axhline(y=10000, color="gray", linestyle="--", alpha=0.5, label="Break-even")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Capital", fontsize=12)
    ax.set_title(
        "4-Agent Battle: Profit Comparison (100 steps)", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    save_path = output_dir / "battle_4agents.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {save_path}")

    # ── Final ranking ──
    final_step = max(r["step"] for r in logs)
    final_capitals: dict[str, str] = {}
    for r in logs:
        if r["step"] == final_step:
            atype = AGENT_TYPES[int(r["agent_id"].split("_")[1])]
            final_capitals[r["agent_id"]] = LABELS[atype]

    ranking = sorted(
        [
            (aid, r["capital"])
            for r in logs
            if r["step"] == final_step
            for aid in [r["agent_id"]]
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\n{'=' * 50}")
    print(f"  Final capital ranking (step {final_step})")
    print(f"{'=' * 50}")
    for rank, (aid, cap) in enumerate(ranking, start=1):
        medal = {1: "[1st]", 2: "[2nd]", 3: "[3rd]", 4: "[4th]"}[rank]
        label = final_capitals.get(aid, aid)
        print(f"  {medal} {rank}. {label}: {cap:,.2f}")
