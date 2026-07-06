"""Final experiment: LLM vs 3 rule-based agents, 150-step long game."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.agents import CostPlusAgent, LLMAgent, MatchLowestAgent, RandomAgent
from src.environment import Action, MarketConfig, MarketEnv, Observation

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = MarketConfig(
    n_retailers=4,
    base_demand=1000,
    price_sensitivity=50,
    noise_std=10,
    unit_cost=8.0,
    holding_cost=0.1,
    stockout_penalty=2.0,
    decision_interval=5,
    brand_noise=0.3,
    max_steps=150,
    info_mode="full",
)

AGENT_LABELS: dict[str, str] = {
    "r_0": "LLM (DeepSeek)",
    "r_1": "Random",
    "r_2": "CostPlus",
    "r_3": "MatchLowest",
}

COLORS: dict[str, str] = {
    "r_0": "#6366f1",
    "r_1": "#f59e0b",
    "r_2": "#10b981",
    "r_3": "#ef4444",
}

# ═══════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════

env = MarketEnv(CONFIG)

llm_agent = LLMAgent("r_0", unit_cost=CONFIG.unit_cost, total_steps=150)
agents = [
    llm_agent,
    RandomAgent("r_1", unit_cost=CONFIG.unit_cost),
    CostPlusAgent("r_2", unit_cost=CONFIG.unit_cost, margin=0.3),
    MatchLowestAgent("r_3", unit_cost=CONFIG.unit_cost),
]

# ═══════════════════════════════════════════════════════════════════════════
# Run simulation
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  LLM vs Rules — 150-step Battle")
print("=" * 60)
print(f"  decision_interval={CONFIG.decision_interval}  brand_noise={CONFIG.brand_noise}")
print(f"  LLM: {llm_agent.model} @ {llm_agent.base_url}")
print()

obs_list: list[Observation] = env.reset()
logs: list[dict[str, Any]] = []
llm_analysis_log: list[str] = []

for step_idx in range(CONFIG.max_steps):
    actions: dict[str, Action] = {}
    capital_before: dict[str, float] = {}

    for agent in agents:
        obs = next(o for o in obs_list if o.self_state.id == agent.agent_id)
        capital_before[agent.agent_id] = obs.self_state.capital
        actions[agent.agent_id] = agent.act(obs)

    obs_list, info = env.step(actions)
    sales_map: dict[str, float] = info.get("sales", {})

    # Record per-agent log
    for agent in agents:
        new_obs = next(
            o for o in obs_list if o.self_state.id == agent.agent_id
        )
        s = new_obs.self_state
        profit = s.capital - capital_before[agent.agent_id]
        logs.append(
            {
                "step": info["step"],
                "agent_id": agent.agent_id,
                "price": s.price,
                "order_qty": s.order_qty,
                "inventory": s.inventory,
                "sales": sales_map.get(agent.agent_id, 0.0),
                "profit": profit,
                "capital": s.capital,
                "market_share": new_obs.my_market_share,
            }
        )

    # Store LLM analysis
    analysis = llm_agent.last_analysis
    llm_analysis_log.append(
        f"[Step {info['step']:>3}] market_total_inv={obs_list[0].market_total_inventory:.1f} "
        f"my_share={obs_list[0].my_market_share:.2%}\n"
        f"  {analysis}\n"
    )

    # Real-time print (every 10 steps)
    if info["step"] % 10 == 0 or info["step"] == 1:
        print(
            f"  Step {info['step']:>3} | "
            f"LLM price={obs_list[0].self_state.price:.2f} "
            f"cap={obs_list[0].self_state.capital:.0f} | "
            f"{analysis[:60]}..."
        )

# ═══════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# ── Final ranking ──
final_step = max(r["step"] for r in logs)
print(f"\n{'=' * 50}")
print(f"  Final capital ranking (step {final_step})")
print(f"{'=' * 50}")
ranking: list[tuple[str, float]] = []
for r in logs:
    if r["step"] == final_step:
        ranking.append((r["agent_id"], r["capital"]))
ranking.sort(key=lambda x: x[1], reverse=True)

for rank, (aid, cap) in enumerate(ranking, start=1):
    label = AGENT_LABELS.get(aid, aid)
    print(f"  {rank}. {label}: {cap:,.2f}")

# ── Performance metrics ──
stats = llm_agent.api_stats
calls = int(stats["calls"])
successes = int(stats["successes"])
total_ms = float(stats["total_ms"])
print(f"\n{'─' * 30}")
print(f"  LLM API Stats")
print(f"{'─' * 30}")
print(f"  Calls:      {calls}")
print(f"  Successes:  {successes}")
print(f"  Failures:   {calls - successes}")
print(f"  Success rate: {successes / calls * 100:.1f}%" if calls > 0 else "  N/A")
print(f"  Avg latency: {total_ms / calls:.0f}ms" if calls > 0 else "  N/A")

# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

agent_ids = sorted({r["agent_id"] for r in logs})
steps_range = sorted({r["step"] for r in logs})

# Helper: extract series per agent
def _series(key: str) -> dict[str, list]:
    out: dict[str, list] = {aid: [] for aid in agent_ids}
    for s in steps_range:
        for aid in agent_ids:
            row = [r for r in logs if r["step"] == s and r["agent_id"] == aid]
            out[aid].append(row[0][key] if row else np.nan)
    return out


# ── Plot 1: Profit comparison ──
fig1, ax1 = plt.subplots(figsize=(12, 7))
for aid in agent_ids:
    rows = [r for r in logs if r["agent_id"] == aid]
    st = [r["step"] for r in rows]
    cap = [r["capital"] for r in rows]
    ax1.plot(st, cap, label=AGENT_LABELS[aid], color=COLORS[aid], linewidth=2)
ax1.axhline(y=10000, color="gray", linestyle="--", alpha=0.5, label="Break-even")
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Capital", fontsize=12)
ax1.set_title("Profit Comparison (150 steps)", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
fig1.savefig(output_dir / "profit_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\n  Saved: output/profit_comparison.png")

# ── Plot 2: Price strategy ──
fig2, ax2 = plt.subplots(figsize=(12, 7))
for aid in agent_ids:
    rows = [r for r in logs if r["agent_id"] == aid]
    st = [r["step"] for r in rows]
    pr = [r["price"] for r in rows]
    ax2.step(st, pr, label=AGENT_LABELS[aid], color=COLORS[aid], linewidth=1.5, where="post")
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Price", fontsize=12)
ax2.set_title("Price Strategy (sticky, interval=5)", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
fig2.savefig(output_dir / "price_strategy.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("  Saved: output/price_strategy.png")

# ── Plot 3: Market share ──
fig3, ax3 = plt.subplots(figsize=(12, 7))
for aid in agent_ids:
    rows = [r for r in logs if r["agent_id"] == aid]
    st = [r["step"] for r in rows]
    ms = [r["market_share"] * 100 for r in rows]
    ax3.plot(st, ms, label=AGENT_LABELS[aid], color=COLORS[aid], linewidth=1.5, alpha=0.85)
ax3.set_xlabel("Step", fontsize=12)
ax3.set_ylabel("Market Share (%)", fontsize=12)
ax3.set_title("Market Share Evolution", fontsize=14, fontweight="bold")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
fig3.savefig(output_dir / "market_share.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("  Saved: output/market_share.png")

# ── Export LLM analysis log ──
analysis_path = output_dir / "llm_analysis_log.txt"
analysis_path.write_text("\n".join(llm_analysis_log), encoding="utf-8")
print(f"  Saved: output/llm_analysis_log.txt  ({len(llm_analysis_log)} entries)")

print(f"\n{'=' * 60}")
print("  Experiment complete!")
print(f"{'=' * 60}")
