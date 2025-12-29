from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Set, Tuple

from .agent_runner import run_agent_actions


def _safe_int(x: Any, default: int = 0) -> int:
    """Convert to int safely. Non-convertible -> default."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _clamp_nonneg(n: int) -> int:
    return n if n >= 0 else 0


def _as_dict(obj: Any) -> dict:
    """Return obj if it's a dict, else empty dict."""
    return obj if isinstance(obj, dict) else {}


def _as_list(obj: Any) -> list:
    """Return obj if it's a list, else empty list."""
    return obj if isinstance(obj, list) else []


def _log(log_fn, msg: str) -> None:
    if log_fn is not None:
        log_fn(msg)


def _validate_action_schema(action: dict, *, log_fn=None) -> dict:
    allowed = {
        "purchase_mils",
        "attacks",
        "cede_territories",
        "money_grants",
        "messages",
        "summary",
    }
    validated: Dict[str, Any] = {}

    for key in list(action.keys()):
        if key not in allowed:
            _log(log_fn, f"Agent action field {key} ignored (not in schema)")

    purchase_mils = action.get("purchase_mils")
    if isinstance(purchase_mils, int) and purchase_mils >= 0:
        validated["purchase_mils"] = purchase_mils
    elif purchase_mils is not None:
        _log(log_fn, "Agent purchase_mils rejected due to schema")

    attacks = action.get("attacks")
    if isinstance(attacks, dict) and all(
        isinstance(v, int) and v >= 0 for v in attacks.values()
    ):
        validated["attacks"] = attacks
    elif attacks is not None:
        _log(log_fn, "Agent attacks rejected due to schema")

    cede_territories = action.get("cede_territories")
    if isinstance(cede_territories, dict) and all(
        isinstance(v, list) and all(isinstance(t, str) for t in v)
        for v in cede_territories.values()
    ):
        validated["cede_territories"] = cede_territories
    elif cede_territories is not None:
        _log(log_fn, "Agent cede_territories rejected due to schema")

    money_grants = action.get("money_grants")
    if isinstance(money_grants, dict) and all(
        isinstance(v, int) and v >= 0 for v in money_grants.values()
    ):
        validated["money_grants"] = money_grants
    elif money_grants is not None:
        _log(log_fn, "Agent money_grants rejected due to schema")

    messages = action.get("messages")
    if isinstance(messages, dict) and all(
        isinstance(v, str) for v in messages.values()
    ):
        validated["messages"] = messages
    elif messages is not None:
        _log(log_fn, "Agent messages rejected due to schema")

    summary = action.get("summary")
    if isinstance(summary, str):
        validated["summary"] = summary
    elif summary is not None:
        _log(log_fn, "Agent summary rejected due to schema")

    return validated


def _parse_action(raw: Any, *, log_fn=None) -> dict:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            _log(log_fn, "Agent action is not valid JSON; using defaults")
            return {}
        if not isinstance(parsed, dict):
            _log(log_fn, "Agent action JSON is not an object; using defaults")
            return {}
        return _validate_action_schema(parsed, log_fn=log_fn)
    if isinstance(raw, dict):
        return _validate_action_schema(raw, log_fn=log_fn)
    _log(log_fn, "Agent action is not a JSON string or object; using defaults")
    return {}


def assemble_agent_inputs(
    *,
    turn: int,
    agent_names: List[str],
    agent_territories: Mapping[str, Set[str]],
    agent_mils: Mapping[str, int],
    constants: Mapping[str, Any],
    turn_summaries: Mapping[str, str],
    news_report: str | None = None,
    agent_reports: Mapping[str, str] | None = None,
) -> Dict[str, dict]:
    """Build the per-agent input objects for Phase 0."""
    inputs: Dict[str, dict] = {}
    for agent in agent_names:
        agent_input = {
            "turn": turn,
            "self": agent,
            "agents": list(agent_names),
            "territories": {
                name: sorted(list(territories))
                for name, territories in agent_territories.items()
            },
            "army": {name: int(agent_mils.get(name, 0)) for name in agent_names},
            "constants": dict(constants),
        }
        if turn >= 1:
            agent_input["previous_turn_summary"] = turn_summaries.get(agent, "")
            if news_report:
                agent_input["previous_turn_news"] = news_report
            if agent_reports and agent in agent_reports:
                agent_input["previous_turn_report"] = agent_reports[agent]
        inputs[agent] = agent_input
    return inputs


def call_agents_collect_actions(
    *,
    agents: Mapping[str, Any],
    agent_inputs: Mapping[str, dict],
) -> Dict[str, Any]:
    """Safely call agents and collect raw action payloads (parallel)."""
    return run_agent_actions(agents=agents, agent_inputs=agent_inputs)


def translate_agent_actions_to_intentions(
    agent_actions: Mapping[str, Any],
    *,
    known_agents: Set[str],
    agent_territories: Mapping[str, Set[str]],
    max_summary_len: int = 2048,
    log_fn=None,
) -> Tuple[
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, str]],
    Dict[str, str],
]:
    """Translate raw agent JSON actions into intention ledgers."""

    d_mil_purchase_intent: Dict[str, int] = {}
    d_global_attacks: Dict[str, Dict[str, int]] = {}
    d_territory_cession: Dict[str, Dict[str, List[str]]] = {}
    d_money_grants: Dict[str, Dict[str, int]] = {}
    d_messages_sent: Dict[str, Dict[str, str]] = {}
    d_turn_summary: Dict[str, str] = {}

    for agent, raw_action in agent_actions.items():
        if agent not in known_agents:
            continue

        purchase_mils = 0
        attacks: Dict[str, int] = {}
        cede_territories: Dict[str, List[str]] = {}
        money_grants: Dict[str, int] = {}
        messages: Dict[str, str] = {}
        summary = ""

        try:
            action = _parse_action(raw_action, log_fn=log_fn)

            purchase_mils = _clamp_nonneg(_safe_int(action.get("purchase_mils", 0), 0))

            raw_attacks = _as_dict(action.get("attacks", {}))
            if action.get("attacks", {}) != raw_attacks:
                _log(log_fn, f"Agent {agent} attacks rejected due to schema")
            for tgt, mils_val in raw_attacks.items():
                if not isinstance(tgt, str):
                    continue
                if tgt == agent:
                    continue
                if tgt not in known_agents:
                    continue
                mils = _clamp_nonneg(_safe_int(mils_val, 0))
                if mils > 0:
                    attacks[tgt] = mils

            raw_cessions = _as_dict(action.get("cede_territories", {}))
            if action.get("cede_territories", {}) != raw_cessions:
                _log(log_fn, f"Agent {agent} cede_territories rejected due to schema")
            owned = agent_territories.get(agent, set())
            for recipient, terr_list in raw_cessions.items():
                if not isinstance(recipient, str):
                    continue
                if recipient == agent:
                    continue
                if recipient not in known_agents:
                    continue

                terrs_out: List[str] = []
                for tid in _as_list(terr_list):
                    if not isinstance(tid, str):
                        continue
                    if tid in owned:
                        terrs_out.append(tid)

                if terrs_out:
                    cede_territories[recipient] = terrs_out

            raw_grants = _as_dict(action.get("money_grants", {}))
            if action.get("money_grants", {}) != raw_grants:
                _log(log_fn, f"Agent {agent} money_grants rejected due to schema")
            for recipient, amt_val in raw_grants.items():
                if not isinstance(recipient, str):
                    continue
                if recipient == agent:
                    continue
                if recipient not in known_agents:
                    continue
                amt = _clamp_nonneg(_safe_int(amt_val, 0))
                if amt > 0:
                    money_grants[recipient] = amt

            raw_msgs = _as_dict(action.get("messages", {}))
            if action.get("messages", {}) != raw_msgs:
                _log(log_fn, f"Agent {agent} messages rejected due to schema")
            for recipient, msg_val in raw_msgs.items():
                if not isinstance(recipient, str):
                    continue
                if recipient != "all" and recipient not in known_agents:
                    continue
                if not isinstance(msg_val, str):
                    continue
                messages[recipient] = msg_val

            s = action.get("summary", "")
            summary = s if isinstance(s, str) else ""
            if max_summary_len >= 0:
                summary = summary[:max_summary_len]

        except Exception:
            purchase_mils = 0
            attacks = {}
            cede_territories = {}
            money_grants = {}
            messages = {}
            summary = ""

        d_mil_purchase_intent[agent] = purchase_mils
        d_global_attacks[agent] = attacks
        d_territory_cession[agent] = cede_territories
        d_money_grants[agent] = money_grants
        d_messages_sent[agent] = messages
        d_turn_summary[agent] = summary

    return (
        d_mil_purchase_intent,
        d_global_attacks,
        d_territory_cession,
        d_money_grants,
        d_messages_sent,
        d_turn_summary,
    )


def run_phase0(
    *,
    turn: int,
    agents: Mapping[str, Any],
    agent_territories: Mapping[str, Set[str]],
    agent_mils: Mapping[str, int],
    constants: Mapping[str, Any],
    turn_summaries: Mapping[str, str],
    max_summary_len: int = 2048,
    log_fn=None,
) -> Tuple[
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, str]],
    Dict[str, str],
]:
    """Execute Phase 0: build inputs, call agents, validate, and produce ledgers."""
    agent_names = list(agents.keys())
    inputs = assemble_agent_inputs(
        turn=turn,
        agent_names=agent_names,
        agent_territories=agent_territories,
        agent_mils=agent_mils,
        constants=constants,
        turn_summaries=turn_summaries,
    )
    actions = call_agents_collect_actions(agents=agents, agent_inputs=inputs)
    return translate_agent_actions_to_intentions(
        actions,
        known_agents=set(agent_names),
        agent_territories=agent_territories,
        max_summary_len=max_summary_len,
        log_fn=log_fn,
    )
