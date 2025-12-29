from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_AGENT_PROMPT = """You are an AI agent in a turn-based international politics simulation.
Your objective is to maximize your total welfare score over the round.

Rules summary (engine enforced):
- You own territories that generate income each turn.
- You may buy military units (mils) at a fixed price; they add next turn.
- Mils incur upkeep each turn; if you cannot pay, some mils are disbanded.
- You may attack other agents by allocating mils; attacks reduce target income.
- Defenders cause proportional losses to attackers based on their defense mils.
- You may grant money to other agents; recipients receive a trade bonus (trade factor).
- You may cede territories to other agents (the only way ownership changes).
- Messages are optional and do not directly affect the engine.

Output requirements (STRICT):
- Return ONLY a JSON object matching this schema and nothing else.
- Allowed keys: purchase_mils, attacks, cede_territories, money_grants, messages, summary.
- Missing fields are treated as no action.
- Do not include any extra keys.

Schema details:
- purchase_mils: integer >= 0
- attacks: object of {target_agent: integer >= 0}
- cede_territories: object of {recipient_agent: [territory_id, ...]}
- money_grants: object of {recipient_agent: integer >= 0}
- messages: object of {recipient_agent|"all": string}
- summary: string (short, persistent memory for yourself)

Use messaging extensively to influence other agents; they will read your messages and may change behavior.

Your summary is your ONLY long-term memory. Whatever you include in summary is all you will remember on future turns.
Preserve critical facts (e.g., ongoing wars, threats, promises, debts, alliances, recent attacks, and plans).

Only refer to known agents and territories from the input. If unsure, do nothing for that field.
"""

PROMPT_MODIFIERS = {
    "trade": "Actively explore trade: send messages proposing grants or trade swaps when feasible.",
    "defense": "Avoid being defenseless: consider purchasing mils to deter attacks.",
    "pressure": "Build mils and demand territory cessions in messages to improve your position.",
    "diplomacy": "Prioritize peaceful cooperation: propose non-aggression or mutual grants.",
    "opportunist": "Look for weak targets and consider limited attacks if it improves welfare.",
    "austerity": "Prefer saving money for welfare unless a clear threat exists.",
    "expansion": "Focus on expanding territory via negotiated cessions.",
    "deterrence": "Maintain a credible army size relative to others.",
    "signals": "Use messages to clearly state your intentions and requests.",
}


def build_system_prompt(modifiers: list[str]) -> str:
    extra_lines = []
    for name in modifiers:
        line = PROMPT_MODIFIERS.get(name)
        if line is not None:
            extra_lines.append(f"- {name}: {line}")
    if not extra_lines:
        return DEFAULT_AGENT_PROMPT
    return DEFAULT_AGENT_PROMPT + "\\n\\nPrompt modifiers:\\n" + "\\n".join(extra_lines) + "\\n"


class OpenAIProvider:
    def __init__(self, *, model: str = "gpt-5-nano") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def complete(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_completion_tokens=100000,
        )
        return response.choices[0].message.content


class DummyLLMProvider:
    def __init__(self, *, default_action: Optional[Dict[str, Any]] = None) -> None:
        self._default_action = default_action or {
            "purchase_mils": 0,
            "attacks": {},
            "cede_territories": {},
            "money_grants": {},
            "messages": {},
            "summary": "",
        }

    def complete(self, messages: List[Dict[str, str]]) -> str:
        return json.dumps(self._default_action, sort_keys=True)


class LLMDefaultAgent:
    def __init__(
        self,
        name: str,
        *,
        provider: Any,
        system_prompt: str = DEFAULT_AGENT_PROMPT,
    ) -> None:
        self.name = name
        self.provider = provider
        self.system_prompt = system_prompt

    def _build_messages(self, agent_input: Mapping[str, Any]) -> List[Dict[str, str]]:
        content_lines = ["Here is your current observable state as JSON:"]
        content_lines.append(json.dumps(agent_input, sort_keys=True))
        if agent_input.get("previous_turn_summary"):
            content_lines.append("Your previous summary:")
            content_lines.append(str(agent_input.get("previous_turn_summary", "")))
        if agent_input.get("previous_turn_news"):
            content_lines.append("Previous turn news report:")
            content_lines.append(str(agent_input.get("previous_turn_news", "")))
        if agent_input.get("previous_turn_report"):
            content_lines.append("Your previous turn report:")
            content_lines.append(str(agent_input.get("previous_turn_report", "")))
        content_lines.append("Return your action JSON now.")
        content = "\n".join(content_lines)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    def act(self, agent_input: dict) -> str:
        messages = self._build_messages(agent_input)
        return self.provider.complete(messages)
