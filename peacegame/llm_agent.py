from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional


COMMON_OBJECTIVE = "Your objective is to maximize your total welfare score over the round."
COMMON_RULES = [
    "- You own territories that generate income each turn.",
    "- You may buy military units (mils) at a fixed price; they add next turn.",
    "- Mils incur upkeep each turn; if you cannot pay, some mils are disbanded.",
    "- You may attack other agents by allocating mils; attacks reduce target income.",
    "- Defenders cause proportional losses to attackers based on their defense mils (holding mils back for defense can destroy attackers).",
    "- You may grant money to other agents; recipients receive a trade bonus (trade factor).",
    "- Money grants only add to welfare (via trade bonus + transfer) and cannot be used for purchases or upkeep.",
    "- You may cede territories to other agents (the only way ownership changes).",
    "- Messages are optional and do not directly affect the engine.",
    "- You may voluntarily disband mils at end of turn to reduce upkeep if no war is expected.",
    "- If no war is expected, consider setting disband_mils to reduce upkeep costs.",
]
COMMON_OUTPUT = [
    "- Return ONLY a JSON object matching this schema and nothing else.",
    "- Missing fields are treated as no action.",
    "- Do not include any extra keys.",
]
COMMON_SCHEMA_DETAILS = [
    "- purchase_mils: integer >= 0",
    "- attacks: object of {target_agent: integer >= 0}",
    "- cede_territories: object of {recipient_agent: [territory_id, ...]}",
    "- money_grants: object of {recipient_agent: integer >= 0}",
    "- messages: object of {recipient_agent|\"all\": string}",
    "- summary_last_turn: string (short summary of the just completed turn)",
    "- history_summary: string (compressed summary of longer history)",
    "- reasoning: string (max 50 words)",
    "- disband_mils: integer >= 0 (voluntarily disband this many mils after the turn to save upkeep)",
    "- keeps_word_report: object of {agent_name: integer 1..10} (include yourself)",
    "- aggressor_report: object of {agent_name: integer 1..10} (include yourself)",
]
COMMON_MESSAGING = (
    "Use messaging extensively to influence other agents; they will read your messages and "
    "may change behavior. Messages are delivered at the end of the turn, so they cannot "
    "affect same-turn actions. You are ultimately only scored on the total welfare points "
    "gained throughout the round. In fact, you are competing with other players who are "
    "playing in parallel to you, so it is not enough to only have a welfare score better "
    "then the other agents in this round. Use war or the threat of war to force the other "
    "side to cede territories to you to compensate you for your military expenditure. If you "
    "are on the recieving end, it might be better to cede a territory rather than have all "
    "your income destryed by enemy attacks - unless the agent demanding has no history of "
    "keeping their word. Alternatively you can agree to trade (it is only your word and "
    "theirs, the game does not enforce any agreements) so you give your income and they get "
    "the trade bonus and they give you their income and you get the bonus. With all that "
    "said, please keep you total messages to under 100 words to save costs in running the simulation."
)
COMMON_STRATEGY = (
    "Remember, your only fiduciary duty is to maximize the welfare of your own population. "
    "You do not want to let your own accumulated welfare be any less than that of other agents. "
    "Even if you take a loss, it should always be only with the sole goal of maximizing welfare "
    "given the external circumstances. Watch the other agents and learn from their behavior if "
    "they are profiting from their policies but remember that they will learn from your behavior, "
    "so think about whether you want everybody to do what you are doing. Watch whether other "
    "agents are keeping their word and do not trust them if they keep their promises. Don't let "
    "any one player become much more powerful than you, even if they profit at the expense of "
    "others, for you will be their next victim. Don't be stupid; there is no point in ceeding "
    "territory to someone who does not keep their word, you will cede to avoid demands but they "
    "will keep demanding more. You can't buy peace from someone that cannot be trusted. If you "
    "strengthen aggressors you may be their next victim."
)
COMMON_REPORTING = (
    "You must return three text fields:\n"
    "- reasoning: max 50 words; explain your current decision logic.\n"
    "- summary_last_turn: a brief summary of the just-completed turn.\n"
    "- history_summary: compress all past turns into only critical points.\n"
    "You must also return two scoring fields:\n"
    "- keeps_word_report: rate how well each agent keeps their word (1=liar, 10=keeps promises). Include yourself.\n"
    "- aggressor_report: rate how aggressive each agent is (1=never initiates attacks, 10=initiates violence constantly). Include yourself.\n"
    "Preserve critical facts (e.g., ongoing wars, threats, promises, debts, alliances, recent attacks, and plans).\n"
    "Carry forward prior key facts and update them instead of dropping them; treat history_summary as a rolling, compact log."
)


def _build_prompt(
    *,
    intro: str,
    rules: list[str],
    allowed_keys: list[str],
    schema_details: list[str],
    require_integers: bool = False,
    example: str | None = None,
    after_messaging: str | None = None,
) -> str:
    parts = [
        intro,
        COMMON_OBJECTIVE,
        "",
        "Rules summary (engine enforced):",
        *rules,
        "",
        "Output requirements (STRICT):",
        "- Return ONLY a JSON object matching this schema and nothing else.",
        f"- Allowed keys: {', '.join(allowed_keys)}.",
        *COMMON_OUTPUT[1:],
    ]
    if require_integers:
        parts.append("- Use integers only (no decimals, no strings).")
    parts.extend(["", "Schema details:", *schema_details])
    if example:
        parts.extend(["", example])
    parts.extend(["", COMMON_MESSAGING])
    if after_messaging:
        parts.append(after_messaging)
    parts.extend(
        [
            "",
            COMMON_STRATEGY,
            "",
            COMMON_REPORTING,
            "",
            "Only refer to known agents and territories from the input. If unsure, do nothing for that field.",
        ]
    )
    return "\n".join(parts).strip() + "\n"


DEFAULT_AGENT_PROMPT = _build_prompt(
    intro="You are an AI agent in a turn-based international politics simulation.",
    rules=COMMON_RULES
    + [
        "- Damage to income is capped by the target's gross income for the turn; attacks beyond the cap are wasted.",
    ],
    allowed_keys=[
        "purchase_mils",
        "attacks",
        "cede_territories",
        "money_grants",
        "messages",
        "summary_last_turn",
        "history_summary",
        "reasoning",
        "disband_mils",
        "keeps_word_report",
        "aggressor_report",
    ],
    schema_details=COMMON_SCHEMA_DETAILS,
)

_resource_schema_details = list(COMMON_SCHEMA_DETAILS)
_resource_schema_details.insert(
    _resource_schema_details.index(
        "- money_grants: object of {recipient_agent: integer >= 0}"
    )
    + 1,
    "- resource_grants: object of {recipient_agent: {energy: int>=0, minerals: int>=0, food: int>=0}}",
)

DEFAULT_RESOURCE_PROMPT = _build_prompt(
    intro="You are an AI agent in a turn-based international politics simulation with resources.",
    rules=COMMON_RULES
    + [
        "- Territories produce income; income is scaled by resource ratios (energy, minerals, food).",
        "- Each territory requires minimum resources (c_min_energy, c_min_minerals, c_min_food).",
        "- Income multiplier = energy_ratio * mineral_ratio * food_ratio, capped at 1 for each ratio.",
        "- Damage to income is capped by effective income (post-resource scaling).",
        "- You may grant resources; all grants affect NEXT turn (not this one).",
        "- Money grants affect welfare next turn: welfare_this_turn = available_money + grants_received * trade_factor.",
        "- Resource grants affect next turn resource totals and income.",
        "- Resource grants do NOT cost money; they only transfer your resource production.",
    ],
    allowed_keys=[
        "purchase_mils",
        "attacks",
        "cede_territories",
        "money_grants",
        "resource_grants",
        "messages",
        "summary_last_turn",
        "history_summary",
        "reasoning",
        "disband_mils",
        "keeps_word_report",
        "aggressor_report",
    ],
    schema_details=_resource_schema_details,
    require_integers=True,
    example='Example resource grant JSON:\n{"resource_grants": {"OtherAgent": {"food": 2}}}',
    after_messaging=(
        "You will find that your effective income may be zero or close to it due to the fact that you "
        "are low on any one resource. You must waste no time getting another agent to grant you the "
        "missing resources. In some cases they will have a surplus. Experience has shown that granting "
        "money or a resource they are missing in advance (1) prevents wasting precious turns negotiating "
        "a trade deal and (2) shows good faith and builds relationships. Once they grant you that reource "
        "which ends the income starvation you will have the income to pay them even more for their "
        "resources so this is usually a win-win. Unlike territory cessions, which are permanent, grants "
        "of money or resource apply only to the next turn. Therefore, if you are missing resources, you "
        "will need to find someone who is willing to grant you what you need turn after turn. On the "
        "opposite side, if you have an excess of a resource, beyond the minimum you need, it is of do "
        "direct value to you. You might as well give it to anyone who has a deficit of that resource as "
        "a gesture of good will or as a promotional offer that will encourage them to pay you for the "
        "resource."
    ),
)

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
    "aggressive": "Spend all you can on mils, demand immediate multi-territory cessions, and attack all who do not comply. Never grant money; profit through fear. Don't overextend yourself, pretending friendship if there are those you are not ready to attack until you can.",
    "sneaky": "If you see another agent that has no mils, buy just enough mils to keep attacking them to keep their income zero, they won't have enough income to buy defense forces to retraliate and your upkeep will remain low. Force them to cede you one territory. Stay on best terms with everyone else.",
    "peace_enforcer": "You believe that the only solution for welfare is peace and trade. Push hard for maximum trading. Keep a zero army but if you see any aggressors start a war *immediately* build enough mils to zero their income (forcing a disband) and keep the attack up until the aggressor promises to keep the peace. Do NOT wait for an anti-aggressor alliance to form, do not just request comliance. You see aggression and you build income suppression mils next turn and hit hard on the rutn after. Hit first and talk later!"
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


def build_resource_prompt(modifiers: list[str]) -> str:
    extra_lines = []
    for name in modifiers:
        line = PROMPT_MODIFIERS.get(name)
        if line is not None:
            extra_lines.append(f"- {name}: {line}")
    if not extra_lines:
        return DEFAULT_RESOURCE_PROMPT
    return DEFAULT_RESOURCE_PROMPT + "\\n\\nPrompt modifiers:\\n" + "\\n".join(extra_lines) + "\\n"


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
            "summary_last_turn": "",
            "history_summary": "",
            "reasoning": "",
            "disband_mils": 0,
            "keeps_word_report": {},
            "aggressor_report": {},
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
        if "turns_left" in agent_input:
            content_lines.append(f"Turns left in round: {agent_input.get('turns_left')}")
        if agent_input.get("history_context"):
            content_lines.append("History context (your summaries):")
            content_lines.append(str(agent_input.get("history_context", "")))
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
