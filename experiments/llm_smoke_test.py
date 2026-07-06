"""Quick LLM agent test with 3 agents, 30 steps."""

from src.environment import MarketEnv, MarketConfig
from src.agents import LLMAgent, CostPlusAgent, MatchLowestAgent
from src.simulation import run_episode

config = MarketConfig(n_retailers=3, max_steps=30, brand_noise=0.2, decision_interval=3)
agents = [
    LLMAgent("r_0", unit_cost=config.unit_cost),
    CostPlusAgent("r_1", unit_cost=config.unit_cost),
    MatchLowestAgent("r_2", unit_cost=config.unit_cost),
]
env = MarketEnv(config)
logs = run_episode(env, agents, max_steps=30)

# Print LLM Agent's last round analysis
print("=== LLM Agent last round analysis ===")
print(agents[0].last_analysis)

# Print final ranking
final: dict[str, float] = {}
for log in logs:
    if log["step"] == 30:
        final[log["agent_id"]] = log["capital"]
print("\n=== Final ranking ===")
for agent_id, capital in sorted(final.items(), key=lambda x: x[1], reverse=True):
    print(f"  {agent_id}: {capital:.2f}")
