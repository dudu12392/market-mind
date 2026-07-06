"""LLM-powered agent using OpenAI-compatible Chat Completion API."""

from __future__ import annotations

import json
import os
import re
import time

import structlog
from src.agents.base import BaseAgent
from src.environment import Action, Observation

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
        brand_noise: float = 0.0,
        temperature: float = 0.3,
    ) -> None:
        super().__init__(agent_id)
        self.unit_cost: float = unit_cost
        self.total_steps: int = total_steps
        self.brand_noise: float = brand_noise
        self.temperature: float = temperature

        self.api_key: str | None = os.getenv("OPENAI_API_KEY")
        self.base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.model: str = model or os.getenv("LLM_MODEL", "deepseek-chat")
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
        if not self.api_key:
            self.last_analysis = "未配置 OPENAI_API_KEY 环境变量，采用保守策略"
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
        if len(self.memory) > 5:
            self.memory = self.memory[-5:]

    # ── Prompt builder ──────────────────────────────────────────────────

    def _build_prompt(self, obs: Observation) -> list[dict]:
        """Build the system + user message list for the LLM."""
        min_price = self.unit_cost * 0.95
        brand_noise_pct = self.brand_noise * 100

        system = f"""你是一个零售定价 AI 经济学家，你的目标是在 {self.total_steps} 轮博弈中最大化累计利润。

## 市场规则
- 你的进货成本为 {self.unit_cost} 元/件
- 每轮你需要决定零售价 price（元）和补货量 order_qty（件）
- 补货立刻到库，order_qty 的有效范围是 0-200
- 市场总需求随所有零售商均价上升而下降
- 未售出库存产生持有成本，缺货产生缺货惩罚
- 存在 {brand_noise_pct}% 的品牌随机扰动，短期销量波动不一定反映真实趋势

## 决策原则
你应该像一个理性的经济学家一样思考：
1. 观察竞争对手的价格水平，判断市场处于价格战还是稳定盈利期
2. 不要盲目跟风最低价——低于成本的销售只会加速亏损
3. 在利润和市场份额之间找平衡，适当保持库存应对需求波动
4. 关注自己的市场份额变化趋势，而非单轮得失

## 禁止行为（违反将导致严重亏损）
- 禁止定价低于成本价的 95%（即低于 {min_price:.2f} 元）
- 禁止在 10 轮内价格波动超过 30%
- 禁止让库存长期为 0（缺货意味着失去顾客）
- 禁止 order_qty 超过 200 或为负数

## 输出格式
必须返回严格 JSON，不要包含注释或额外文本：
{{
  "analysis": "你对当前局势的中文分析（2-3句，包含对竞争格局的判断）",
  "price": 浮点数,
  "order_qty": 整数
}}

## 决策案例（Few-shot）
以下是一个成功和失败的案例，供你参考：

失败案例：对手降价到 8.5，你也跟到 8.5，结果双方都亏损。正确做法是维持 10-12 元，用品牌扰动和库存优势等对手先撑不住。

成功案例：观察到市场均价 11 元，库存充足，对手在涨价。此时定价 10.5-11 元，保持竞争力同时留有利润空间，补货 60-80 件维持安全库存。"""

        # ── User message ──
        state = obs.self_state
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

        # ── Competitor info ──
        comp = obs.competitors_info
        parts.append("\n## 竞争对手情报")
        if "avg_market_price" in comp:
            parts.append(f"- 市场均价：{comp['avg_market_price']:.2f}")
            parts.append(f"- 对手总库存：{comp['total_market_inventory']:.1f}")
        else:
            for cid, cinfo in comp.items():
                parts.append(
                    f"- {cid}：价格 {cinfo['price']:.2f}，库存 {cinfo['inventory']:.1f}"
                )

        # ── Market intelligence ──
        risk_hint = (
            "（高库存，价格战风险大）" if obs.market_total_inventory > 600 else ""
        )
        my_share_pct = obs.my_market_share * 100
        parts.append("\n## 市场洞察")
        parts.append(f"- 市场总库存：{obs.market_total_inventory:.1f}{risk_hint}")
        parts.append(f"- 你的市场份额：{my_share_pct:.1f}%（上轮销量/总需求）")

        if len(self.memory) >= 2:
            prev_share = self.memory[-2].get("market_share", 0.0)
            diff = my_share_pct - prev_share
            trend = "上升" if diff > 1 else ("下降" if diff < -1 else "持平")
            parts.append(f"- 市场份额变化：{trend}（{diff:+.1f}%）")

        # ── Historical summary (with 2000-char truncation) ──
        if self.memory:
            history_lines: list[str] = ["\n## 历史摘要"]
            for m in self.memory:
                history_lines.append(
                    f"- 第{m['step']}轮：定价{m['price']:.2f}，"
                    f"补货{m['order_qty']}，"
                    f"库存{m['inventory']:.1f}，"
                    f"资金{m['capital']:.2f}"
                )

            # Fit within 2000 chars for the whole user message
            base_msg = "\n".join(parts)
            full_history = "\n".join(history_lines)
            combined = base_msg + full_history

            if len(combined) > 2000:
                # Drop oldest history entries until we fit
                trimmed_history_lines = list(history_lines)
                while (
                    len(base_msg + "\n".join(trimmed_history_lines)) > 2000
                    and len(trimmed_history_lines) > 2
                ):
                    trimmed_history_lines.pop(1)  # remove oldest entry after header
                trimmed_history_lines.append("（注：早期历史已省略以控制上下文长度）")
                parts = parts + trimmed_history_lines
            else:
                parts = parts + history_lines

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
                        temperature=self.temperature,
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
                    temperature=self.temperature,
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
                    time.sleep(1.0 * attempt)

        logger.error(
            "llm_all_attempts_failed",
            agent_id=self.agent_id,
            error=last_error,
        )
        return ""

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
                    "temperature": self.temperature,
                    "max_tokens": 400,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""

    # ── Response parser ─────────────────────────────────────────────────

    def _parse_response(self, content: str, obs: Observation) -> tuple[float, int, str]:
        """Extract price, order_qty, analysis from LLM response.

        Falls back to conservative strategy on persistent failure.
        """
        price: float | None = None
        order_qty: int | None = None
        analysis: str = ""

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

        if price is not None and order_qty is not None:
            if price >= self.unit_cost * 0.8 and order_qty >= 0:
                return price, order_qty, analysis

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

        logger.warning("llm_fallback_conservative", agent_id=self.agent_id)
        return (
            self.unit_cost * 1.5,
            50,
            "解析失败，采用保守策略：定价为成本1.5倍，补货50",
        )

    def _extract_json(self, content: str) -> tuple[float, int, str]:
        """Pull JSON out of a code fence or raw text."""
        cleaned = content.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

        data = json.loads(cleaned)
        price = float(data.get("price", 0))
        order_qty = int(data.get("order_qty", 0))
        analysis = str(data.get("analysis", ""))
        return price, order_qty, analysis

    def _build_correction_prompt(
        self, prev_content: str, obs: Observation
    ) -> list[dict]:
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
