"""Streamlit real-time simulation dashboard for MarketMind."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from src.agents import BaseAgent, CostPlusAgent, LLMAgent, MatchLowestAgent, RandomAgent
from src.environment import Action, MarketConfig, MarketEnv


# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MarketMind",
    page_icon="📊",
    layout="wide",
)

st.title("📊 MarketMind — 多智能体动态定价博弈")

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — Controls
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 仿真配置")

    selected_types: list[str] = st.multiselect(
        "Agent 类型",
        options=["random", "cost_plus", "match_lowest", "llm"],
        default=["random", "cost_plus", "match_lowest", "llm"],
        key="agent_types",
    )

    n_retailers: int = st.slider("零售商数量", 2, 8, 4, key="n_retailers")
    max_steps: int = st.slider("仿真步数", 50, 300, 150, key="max_steps")
    decision_interval: int = st.slider("定价决策间隔", 1, 10, 5, key="decision_interval")
    brand_noise: float = st.slider("品牌噪声系数", 0.0, 0.5, 0.3, 0.05, key="brand_noise")

    st.divider()
    base_demand: float = st.number_input("基础需求", value=1000.0, key="base_demand")
    price_sensitivity: float = st.number_input("价格敏感度", value=50.0, key="price_sensitivity")
    unit_cost: float = st.number_input("单位成本", value=8.0, key="unit_cost")

    st.divider()
    run_clicked: bool = st.button("🚀 开始仿真", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Agent factory
# ═══════════════════════════════════════════════════════════════════════════

def _build_agents(agent_types: list[str], unit_cost: float) -> list[BaseAgent]:
    agents: list[BaseAgent] = []
    for i in range(n_retailers):
        atype = agent_types[i % len(agent_types)] if agent_types else "random"
        aid = f"r_{i}"
        if atype == "random":
            agents.append(RandomAgent(aid, unit_cost=unit_cost))
        elif atype == "cost_plus":
            agents.append(CostPlusAgent(aid, unit_cost=unit_cost))
        elif atype == "match_lowest":
            agents.append(MatchLowestAgent(aid, unit_cost=unit_cost))
        elif atype == "llm":
            agents.append(LLMAgent(aid, unit_cost=unit_cost))
        else:
            agents.append(RandomAgent(aid, unit_cost=unit_cost))
    return agents


def _get_llm_agent(agents: list[BaseAgent]) -> LLMAgent | None:
    for a in agents:
        if isinstance(a, LLMAgent):
            return a
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Run simulation
# ═══════════════════════════════════════════════════════════════════════════

if run_clicked:
    # Build config
    config = MarketConfig(
        n_retailers=n_retailers,
        base_demand=base_demand,
        price_sensitivity=price_sensitivity,
        unit_cost=unit_cost,
        decision_interval=decision_interval,
        brand_noise=brand_noise,
        max_steps=max_steps,
        info_mode="full",
    )

    agents = _build_agents(selected_types, unit_cost)
    env = MarketEnv(config)

    # Placeholders for charts & table
    price_placeholder = st.empty()
    share_placeholder = st.empty()
    profit_placeholder = st.empty()
    table_placeholder = st.empty()
    status_placeholder = st.empty()
    llm_placeholder = st.sidebar.empty()

    # Data accumulators
    price_df = pd.DataFrame()
    share_df = pd.DataFrame()
    profit_df = pd.DataFrame()
    detail_df = pd.DataFrame()

    st.session_state["sim_running"] = True

    # ── Reset ──
    obs_list = env.reset()
    llm = _get_llm_agent(agents)

    for step_idx in range(max_steps):
        # Build actions
        actions: dict[str, Action] = {}
        for agent in agents:
            obs = next(o for o in obs_list if o.self_state.id == agent.agent_id)
            actions[agent.agent_id] = agent.act(obs)

        obs_list, info = env.step(actions)
        sales_map: dict[str, float] = info.get("sales", {})

        # ── Accumulate data ──
        step_data_price: dict[str, float] = {"step": info["step"]}
        step_data_share: dict[str, float] = {"step": info["step"]}
        step_data_profit: dict[str, float] = {"step": info["step"]}
        detail_rows: list[dict[str, Any]] = []

        for agent in agents:
            new_obs = next(
                o for o in obs_list if o.self_state.id == agent.agent_id
            )
            s = new_obs.self_state
            label = f"{agent.agent_id}"
            step_data_price[label] = s.price
            step_data_share[label] = new_obs.my_market_share * 100
            step_data_profit[label] = s.capital
            detail_rows.append(
                {
                    "Agent": label,
                    "Price": round(s.price, 2),
                    "Inventory": round(s.inventory, 1),
                    "Sales": round(sales_map.get(agent.agent_id, 0.0), 1),
                    "Capital": round(s.capital, 2),
                }
            )

        # Append to DataFrames
        price_df = pd.concat([price_df, pd.DataFrame([step_data_price])], ignore_index=True)
        share_df = pd.concat([share_df, pd.DataFrame([step_data_share])], ignore_index=True)
        profit_df = pd.concat([profit_df, pd.DataFrame([step_data_profit])], ignore_index=True)
        detail_df = pd.DataFrame(detail_rows)

        # ── Update charts (every 3 steps for performance) ──
        if info["step"] % 3 == 0 or info["step"] == max_steps:
            with price_placeholder.container():
                st.subheader("💲 定价策略")
                st.line_chart(
                    price_df.set_index("step"),
                    use_container_width=True,
                    height=250,
                )

            with share_placeholder.container():
                st.subheader("📊 市场份额 (%)")
                st.area_chart(
                    share_df.set_index("step"),
                    use_container_width=True,
                    height=250,
                )

            with profit_placeholder.container():
                st.subheader("💰 利润累积")
                st.line_chart(
                    profit_df.set_index("step"),
                    use_container_width=True,
                    height=280,
                )

            with table_placeholder.container():
                st.subheader("📋 当前状态")
                st.dataframe(
                    detail_df.set_index("Agent"),
                    use_container_width=True,
                    hide_index=False,
                )

        # ── LLM analysis ──
        if llm is not None and llm.last_analysis:
            with llm_placeholder.container():
                st.caption(f"🤖 LLM 分析 (Step {info['step']})")
                st.code(llm.last_analysis, language="text")

        # ── Progress ──
        with status_placeholder.container():
            st.progress(
                info["step"] / max_steps,
                text=f"Step {info['step']} / {max_steps}  |  "
                f"avg_price={info['avg_price']:.2f}  |  "
                f"demand={info['total_demand']:.1f}",
            )

        time.sleep(0.03)

    # ═══════════════════════════════════════════════════════════════════════
    # Final results
    # ═══════════════════════════════════════════════════════════════════════
    final_step = max(int(r["step"]) for _, r in profit_df.iterrows() if not pd.isna(r.iloc[1:]).all())
    final_row = profit_df[profit_df["step"] == max_steps].iloc[0] if len(profit_df) > 0 else None

    if final_row is not None:
        ranking_data: list[dict[str, Any]] = []
        for col in profit_df.columns:
            if col == "step":
                continue
            ranking_data.append(
                {"Agent": col, "Final Capital": round(final_row[col], 2)}
            )
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values("Final Capital", ascending=False)
        ranking_df["Rank"] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df[["Rank", "Agent", "Final Capital"]]

        st.success("🎉 仿真完成！")
        st.subheader("🏆 最终排名")
        st.dataframe(
            ranking_df.set_index("Rank"),
            use_container_width=True,
        )

        # API stats if LLM was used
        if llm is not None:
            stats = llm.api_stats
            calls = int(stats["calls"])
            successes = int(stats["successes"])
            avg_ms = float(stats["total_ms"]) / calls if calls > 0 else 0
            st.caption(
                f"🤖 LLM API: {successes}/{calls} 成功, "
                f"成功率 {successes/calls*100:.0f}%, "
                f"平均延迟 {avg_ms:.0f}ms"
                if calls > 0
                else "🤖 LLM: 未发起 API 调用"
            )

else:
    # ── Idle state ──
    st.info("👈 在侧边栏配置参数后点击 **开始仿真**")
    st.markdown(
        """
        ### 🎮 使用说明
        1. **侧边栏**选择 Agent 类型和仿真参数
        2. 点击 **🚀 开始仿真** 启动多智能体对战
        3. 实时观察定价策略、市场份额和利润曲线变化
        4. 如果选了 LLM Agent，可在侧边栏看到它的实时分析思考
        """
    )
