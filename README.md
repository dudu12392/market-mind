# MarketMind — 多智能体动态定价博弈平台

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Poetry](https://img.shields.io/badge/poetry-package-60a5fa.svg)](https://python-poetry.org)
[![Tests](https://img.shields.io/badge/tests-22%20passed-brightgreen.svg)](./tests)

## 🎯 项目定位

构建可配置的市场仿真环境，让 **LLM Agent** 与 **规则策略 Agent** 在同一市场中博弈，验证 AI 在多主体经济决策中的有效性与局限性。

> 核心问题：AI 能比传统策略更好地定价吗？在什么条件下？

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────┐
│                  仿真主循环                   │
│   MarketEnv ──┬── RandomAgent               │
│               ├── CostPlusAgent              │
│               ├── MatchLowestAgent           │
│               └── LLMAgent (DeepSeek)        │
├─────────────────────────────────────────────┤
│  市场引擎                                     │
│  · 需求弹性 (base_demand - sensitivity×price) │
│  · 吸引力分配 (1/price + brand_noise)         │
│  · 库存持有成本 / 缺货惩罚                     │
│  · 粘性定价 (decision_interval)               │
│  · 信息模式 (full / partial)                  │
├─────────────────────────────────────────────┤
│  可视化 & 分析                                │
│  Streamlit 面板 / matplotlib 图表 / 日志导出    │
└─────────────────────────────────────────────┘
```

---

## 🚀 快速启动

```bash
# 1. 克隆并安装
git clone <your-repo-url>
cd market-mind
poetry install

# 2. 配置 LLM (DeepSeek)
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-chat

# 3. 启动可视化面板
poetry run streamlit run src/visualization.py

# 4. 或运行批量实验
export OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=https://api.deepseek.com LLM_MODEL=deepseek-chat
python experiments/llm_vs_rules_final.py
```

---

## 📊 核心实验

| 实验 | 条件 | LLM 排名 | 关键发现 |
|---|---|---|---|
| [纯价格战](config/scenarios/pure_price_war.yaml) | 每步调价，无噪声 | 第四 | LLM 过度分析，高频决策迷失于简单市场 |
| [品牌差异化](config/scenarios/brand_differentiation.yaml) | 5步调价，30%噪声 | 第四（-14,846） | 随机策略在噪声中意外占优；LLM 无法预测品牌扰动 |
| [信息不对称](config/scenarios/information_asymmetry.yaml) | 部分信息，20%噪声 | 待跑 | 验证信息优势在 Agent 决策中的价值 |

**实验产出** (均在 `output/`)：
- `profit_comparison.png` — 4 条资本累积曲线对比
- `price_strategy.png` — 4 条粘性定价策略图
- `market_share.png` — 市场份额动态演化
- `llm_analysis_log.txt` — LLM 全部 150 步中文决策分析

---

## 🔍 LLM Agent 设计

```
认知循环:  观察 → 分析 → 决策 → 记忆
           │       │       │       │
Observation → Prompt → LLM → Action → MarketEnv.step()
                         │
                    last_analysis (可解释性)
```

- **可解释性**：每轮输出中文分析文本，记录到 `llm_analysis_log.txt`
- **降级机制**：API 失败时回退到 `price=cost×1.5, order_qty=50`
- **记忆管理**：保留最近 10 轮关键指标，追踪市场份额趋势
- **多模型支持**：通过 `OPENAI_BASE_URL` 兼容 DeepSeek / OpenAI / 本地模型

---

## 📁 项目结构

```
market-mind/
├── src/
│   ├── environment.py          # 市场引擎 (pydantic + numpy)
│   ├── simulation.py           # 仿真主循环 + 批量实验
│   ├── visualization.py        # Streamlit 实时监控面板
│   └── agents/
│       ├── base.py             # Agent 抽象基类
│       ├── rule_based.py       # Random / CostPlus / MatchLowest
│       └── llm_agent.py        # LLM Agent (OpenAI SDK + httpx)
├── config/
│   └── scenarios/              # 三组实验 YAML 配置
│       ├── pure_price_war.yaml
│       ├── brand_differentiation.yaml
│       └── information_asymmetry.yaml
├── experiments/
│   └── llm_vs_rules_final.py   # 核心实验脚本
├── tests/
│   ├── test_environment.py     # 市场引擎测试 (4)
│   ├── test_agents.py          # 规则 Agent 测试 (11)
│   └── test_llm_agent.py       # LLM Agent 测试 (7, mocked)
├── output/                     # 实验产出 (图表 + 分析日志)
├── pyproject.toml              # Poetry 依赖管理
└── README.md
```

---

## 🧪 测试

```bash
poetry run pytest tests/ -v
# 22 passed — 覆盖环境引擎和所有 Agent 类型
```

---

## 📊 数据验证：基于 Superstore 数据集校准

MarketMind 的仿真参数并非随意设定，而是对标 Tableau Superstore 真实零售数据（9,994 条订单）。

### 参数映射

| 维度 | Superstore 真实值 | MarketMind 参数 | 匹配度 |
|---|---|---|---|
| 价格离散度 | Furniture CV = 0.62 | `brand_noise=0.2-0.3` | ✅ 良好 |
| 利润率 | 整体 18.2% | CostPlus 23% / LLM 15-25% | ✅ 覆盖真实区间 |
| 竞争格局 | 3 大品类 × 3-4 竞争者 | `n_retailers=4` | ✅ 匹配 |
| 需求波动 | 销售 CV = 0.45-0.62 | `noise_std=10`（累计 150 步接近） | 🟡 部分匹配 |

### 验证方法

```bash
python analysis/superstore_validation.py
# 产出: output/superstore_validation.png + superstore_validation.yaml
```

![Superstore Validation](output/superstore_validation.png)

> **结论**：MarketMind 的价格分布、利润率区间、竞争强度均落在 Superstore 数据的一倍标准差内。仿真环境虽为简化模型，但关键经济参数有真实数据支撑。

---

## 📝 后续改进方向

- **RL 基线**：引入 DQN / PPO Agent 与 LLM 对比
- **多周期库存**：订单提前期、安全库存策略
- **真实数据校准**：基于 Dominick's 零售数据集估计参数
- **多商品扩展**：交叉价格弹性、捆绑定价
- **博弈论分析**：纳什均衡求解 vs LLM 策略收敛

---

## 📄 License

MIT
