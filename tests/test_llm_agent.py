"""Tests for LLMAgent (mocked LLM calls)."""

from unittest.mock import patch


from src.agents.llm_agent import LLMAgent
from src.environment import Action, Observation, RetailerState

UNIT_COST = 8.0


# ── Helper ──────────────────────────────────────────────────────────────────


def _make_obs(step: int = 1) -> Observation:
    return Observation(
        self_state=RetailerState(
            id="r_0", inventory=150.0, capital=10000.0, price=10.0, order_qty=0
        ),
        step=step,
        competitors_info={
            "r_1": {"id": "r_1", "price": 12.0, "inventory": 100.0},
        },
    )


# ── Tests ───────────────────────────────────────────────────────────────────


class TestLLMAgentBasic:
    def test_mock_successful_response(self) -> None:
        """Mock returns valid JSON; agent should parse it correctly."""
        agent = LLMAgent("r_0", unit_cost=UNIT_COST)
        agent.api_key = "sk-mock-key"  # Enable LLM path
        obs = _make_obs()

        mock_response = (
            '{"analysis": "市场平稳，保持现价", "price": 12.0, "order_qty": 80}'
        )

        with patch.object(agent, "_call_llm", return_value=mock_response) as mock:
            action = agent.act(obs)

        mock.assert_called_once()
        assert isinstance(action, Action)
        assert action.price == 12.0
        assert action.order_qty == 80
        assert agent.last_analysis == "市场平稳，保持现价"
        assert len(agent.memory) == 1

    def test_code_block_json(self) -> None:
        """JSON wrapped in ```json fence should still parse."""
        agent = LLMAgent("r_0", unit_cost=UNIT_COST)
        agent.api_key = "sk-mock-key"
        obs = _make_obs()

        mock_response = """```json
{"analysis": "测试", "price": 15.5, "order_qty": 30}
```"""

        with patch.object(agent, "_call_llm", return_value=mock_response):
            action = agent.act(obs)

        assert action.price == 15.5
        assert action.order_qty == 30


class TestLLMAgentFallback:
    def test_invalid_json_falls_back(self) -> None:
        """Invalid JSON → retries fail → conservative fallback."""
        agent = LLMAgent("r_0", unit_cost=UNIT_COST)
        agent.api_key = "sk-mock-key"
        obs = _make_obs()

        # All LLM calls return garbage
        with patch.object(
            agent, "_call_llm", return_value="not valid json {{{}}}"
        ) as mock:
            action = agent.act(obs)

        # 1 initial + 2 correction retries = 3 calls
        assert mock.call_count == 3
        # Conservative: price = unit_cost * 1.5, order_qty = 50
        assert action.price == UNIT_COST * 1.5
        assert action.order_qty == 50

    def test_price_below_floor_triggers_fallback(self) -> None:
        """Price below cost * 0.8 should trigger fallback."""
        agent = LLMAgent("r_0", unit_cost=UNIT_COST)
        agent.api_key = "sk-mock-key"
        obs = _make_obs()

        # price = 5.0 < 8.0 * 0.8 = 6.4 → invalid
        mock_response = '{"analysis": "x", "price": 5.0, "order_qty": 50}'

        with patch.object(agent, "_call_llm", return_value=mock_response):
            action = agent.act(obs)

        assert action.price == UNIT_COST * 1.5
        assert action.order_qty == 50

    def test_empty_llm_response_falls_back(self) -> None:
        """Empty string from LLM → fallback."""
        agent = LLMAgent("r_0", unit_cost=UNIT_COST)
        agent.api_key = "sk-mock-key"
        obs = _make_obs()

        with patch.object(agent, "_call_llm", return_value="") as mock:
            action = agent.act(obs)

        # 1 initial call + 1 retry (second retry short-circuited because "" is falsy)
        assert mock.call_count == 2
        assert action.price == UNIT_COST * 1.5
        assert action.order_qty == 50


class TestLLMAgentMemory:
    def test_memory_accumulates(self) -> None:
        """Multiple act calls should accumulate memory entries."""
        agent = LLMAgent("r_0", unit_cost=UNIT_COST)
        agent.api_key = "sk-mock-key"
        mock_response = '{"analysis": "ok", "price": 10.0, "order_qty": 40}'

        with patch.object(agent, "_call_llm", return_value=mock_response):
            for i in range(5):
                obs = _make_obs(step=i + 1)
                agent.act(obs)

        assert len(agent.memory) == 5
        # Entries should have distinct steps
        steps = [m["step"] for m in agent.memory]
        assert steps == [1, 2, 3, 4, 5]

    def test_memory_caps_at_five(self) -> None:
        """After more than 5 calls, memory should keep only the last 5."""
        agent = LLMAgent("r_0", unit_cost=UNIT_COST)
        agent.api_key = "sk-mock-key"
        mock_response = '{"analysis": "ok", "price": 10.0, "order_qty": 40}'

        with patch.object(agent, "_call_llm", return_value=mock_response):
            for i in range(10):
                obs = _make_obs(step=i + 1)
                agent.act(obs)

        assert len(agent.memory) == 5
        steps = [m["step"] for m in agent.memory]
        assert steps == list(range(6, 11))
