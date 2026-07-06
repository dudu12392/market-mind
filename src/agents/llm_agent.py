"""LLM-powered agent using OpenAI-compatible Chat Completion API."""

from __future__ import annotations

import json
import os
import re
import time

from src.agents.base import BaseAgent
from src.environment import Action, Observation

import structlog

logger = structlog.get_logger(__name__)

# ── Optional OpenAI SDK ─────────────────────────────────────────────────────
try:
    from openai import OpenAI as OpenAISDK

    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False


# ── LLMAgent ────────────────────────────────────────────────────────────────

class LLMAgent(BaseAgent):
    """Agent that delegates pricing decisions to an LLM."""

    def __init__(
        self,
        agent_id: str,
        unit_cost: float,
        model: str = "",
        total_steps: int = 200,
    ) -> None:
        super().__init__(agent_id)
        self.unit_cost: float = unit_cost
        self.total_steps: int = total_steps

        self.api_key: str | None = os.getenv("OPENAI_API_KEY")
        self.base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model: str = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.memory: list[dict] = []
        self.last_analysis: str = ""
        self.api_stats: dict[str, float | int] = {
            "calls": 0,
            "successes": 0,
            "total_ms": 0.0,
        }

        # LLM client
        if HAS_OPENAI_SDK and self.api_key:
            self._client: OpenAISDK | None = OpenAISDK(
                api_key=self.api_key, base_url=self.base_url
            )
        else:
            self._client = None

    # ── Public API ──────────────────────────────────────────────────────

    def act(self, obs: Observation) -> Action:
        # Fast-path: no API key → skip all retries, go straight to conservative
        if not self.api_key:
            self.last_analysis = (
                "未配置 OPENAI_API_KEY 环境变量，采用保守策略"
            )
            self._update_memory(obs, self.unit_cost * 1.5, 50)
            return Action(price=self.unit_cost * 1.5, order_qty=50)

        prompt = self._build_prompt(obs)
        response = self._call_llm(prompt)
        price, order_qty, analysis = self._parse_response(response, obs)
        self.last_analysis = analysis
        self._update_memory(obs, price, order_qty)
        return Action(price=price, order_qty=order_qty)

    def _update_memory(self, obs: Observation, price: float, order_qty: int) -> None:
        self.memory.append(
            {
                "step": obs.step,
                "price": price,
                "order_qty": order_qty,
                "inventory": obs.self_state.inventory,
                "capital": obs.self_state.capital,
                "market_share": obs.my_market_share * 100,
            }
        )
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]

    # ── Prompt builder ──────────────────────────────────────────────────

    def _build_prompt(self, obs: Observation) -> list[dict]:
        """Build the system + user message list for the LLM."""
        system = f"""你是一个零售定价 AI，你的目标是在 {self.total_steps} 轮博弈中最大化累计利润。
市场规则：
- 你以 unit_cost={self.unit_cost} 的成本从供应商进货
- 每轮你需要决定零售价 price 和补货量 order_qty
- 需求随市场均价上升而下降，价格敏感度中等
- 未售出的库存产生持有成本，缺货产生惩罚
- 你的竞争对手也在做同样的事

你需要分析历史趋势，制定明智的定价策略：
- 观察竞争对手的价格水平，判断市场是价格战还是稳定盈利
- 不要盲目跟风最低价——那会毁灭利润
- 适当保持库存以应对需求波动
- 在利润和市场份额之间找平衡

返回格式必须是 JSON：
{{
  "analysis": "你对当前局势的分析（中文，2-3句）",
  "price": 你的定价（浮点数，>= {self.unit_cost * 0.95:.2f}），
  "order_qty": 补货量（整数，0-200）
}}"""

        # ── User message ──
        state = obs.self_state
        # Estimate last-round profit from memory
        last_profit: str = "N/A（首轮）"
        if self.memory:
            prev = self.memory[-1]
            last_profit = f"{state.capital - prev['capital']:.2f}"

        parts: list[str] = []
        parts.append(f"## 当前状态（第 {obs.step} 轮）")
        parts.append(f"- 你的 ID：{self.agent_id}")
        parts.append(f"- 当前库存：{state.inventory:.1f}")
        parts.append(f"- 当前资金：{state.capital:.2f}")
        parts.append(f"- 上轮利润：{last_profit}")

        # ── Historical summary ──
        if self.memory:
            parts.append("\n## 历史摘要")
            for m in self.memory[-5:]:
                parts.append(
                    f"- 第{m['step']}轮：定价{m['price']:.2f}，"
                    f"补货{m['order_qty']}，"
                    f"库存{m['inventory']:.1f}，"
                    f"资金{m['capital']:.2f}"
                )

        # ── Competitor info ──
        comp = obs.competitors_info
        parts.append("\n## 竞争对手情报")
        if "avg_market_price" in comp:
            parts.append(f"- 市场均价：{comp['avg_market_price']:.2f}")
            parts.append(f"- 对手总库存：{comp['total_market_inventory']:.1f}")
        else:
            for cid, cinfo in comp.items():
                parts.append(
                    f"- {cid}：价格 {cinfo['price']:.2f}，"
                    f"库存 {cinfo['inventory']:.1f}"
                )

        # ── Market intelligence (from Observation) ──
        risk_hint = (
            "（高库存，价格战风险大）"
            if obs.market_total_inventory > 600
            else ""
        )
        my_share_pct = obs.my_market_share * 100
        parts.append(f"\n## 市场洞察")
        parts.append(f"- 市场总库存：{obs.market_total_inventory:.1f}{risk_hint}")
        parts.append(f"- 你的市场份额：{my_share_pct:.1f}%（上轮销量/总需求）")

        # Share trend vs previous round
        if len(self.memory) >= 2:
            prev_share = self.memory[-2].get("market_share", 0.0)
            diff = my_share_pct - prev_share
            trend = "上升" if diff > 1 else ("下降" if diff < -1 else "持平")
            parts.append(f"- 市场份额变化：{trend}（{diff:+.1f}%）")

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(parts)},
        ]

    # ── LLM caller ──────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM with retries. Returns response text or conservative fallback."""
        max_attempts = 3
        last_error: str = ""

        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            try:
                if self._client is not None and HAS_OPENAI_SDK:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=400,
                    )
                    content: str = resp.choices[0].message.content or ""
                else:
                    content = self._call_via_httpx(messages)

                elapsed = time.perf_counter() - t0
                self.api_stats["calls"] += 1
                self.api_stats["successes"] += 1
                self.api_stats["total_ms"] += elapsed * 1000
                logger.info(
                    "llm_call_success",
                    agent_id=self.agent_id,
                    attempt=attempt,
                    elapsed_ms=round(elapsed * 1000, 1),
                )
                return content

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.api_stats["calls"] += 1
                self.api_stats["total_ms"] += elapsed * 1000
                last_error = str(exc)
                logger.warning(
                    "llm_call_failed",
                    agent_id=self.agent_id,
                    attempt=attempt,
                    elapsed_ms=round(elapsed * 1000, 1),
                    error=last_error,
                )
                if attempt < max_attempts:
                    time.sleep(1.0 * attempt)  # Linear backoff

        logger.error(
            "llm_all_attempts_failed",
            agent_id=self.agent_id,
            error=last_error,
        )
        return ""  # Signal fallback to _parse_response

    def _call_via_httpx(self, messages: list[dict]) -> str:
        """Fallback LLM caller using httpx directly."""
        import httpx

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 400,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""

    # ── Response parser ─────────────────────────────────────────────────

    def _parse_response(
        self, content: str, obs: Observation
    ) -> tuple[float, int, str]:
        """Extract price, order_qty, analysis from LLM response.

        Falls back to conservative strategy on persistent failure.
        """
        price: float | None = None
        order_qty: int | None = None
        analysis: str = ""

        # Try parsing the response
        if content:
            try:
                price, order_qty, analysis = self._extract_json(content)
            except Exception as exc:
                logger.warning(
                    "llm_parse_failed",
                    agent_id=self.agent_id,
                    error=str(exc),
                    content=content[:200],
                )

        # Validate
        if price is not None and order_qty is not None:
            if price >= self.unit_cost * 0.8 and order_qty >= 0:
                return price, order_qty, analysis

        # Retry with a correction prompt (up to 2 more tries)
        retry_messages = self._build_correction_prompt(content, obs)
        for attempt in range(2):
            retry_content = self._call_llm(retry_messages)
            if not retry_content:
                break
            try:
                price, order_qty, analysis = self._extract_json(retry_content)
                if price >= self.unit_cost * 0.8 and order_qty >= 0:
                    return price, order_qty, analysis
            except Exception:
                continue

        # ── Conservative fallback ──
        logger.warning(
            "llm_fallback_conservative",
            agent_id=self.agent_id,
        )
        return (
            self.unit_cost * 1.5,
            50,
            "解析失败，采用保守策略：定价为成本1.5倍，补货50",
        )

    def _extract_json(self, content: str) -> tuple[float, int, str]:
        """Pull JSON out of a code fence or raw text and return (price, order_qty, analysis)."""
        # Strip ```json ... ``` fences
        cleaned = content.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

        data = json.loads(cleaned)

        price = float(data.get("price", 0))
        order_qty = int(data.get("order_qty", 0))
        analysis = str(data.get("analysis", ""))
        return price, order_qty, analysis

    def _build_correction_prompt(self, prev_content: str, obs: Observation) -> list[dict]:
        """Prompt asking the LLM to fix malformed JSON output."""
        return [
            {
                "role": "system",
                "content": (
                    "你之前的回复格式不正确。请只输出合法的 JSON，包含 analysis、price、order_qty 三个字段。"
                    f"price 必须 >= {self.unit_cost * 0.8}，order_qty 必须是 0-200 的整数。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"你的上一条回复是：\n```\n{prev_content[:500]}\n```\n\n"
                    "请重新给出正确的 JSON 回复。"
                ),
            },
        ]
