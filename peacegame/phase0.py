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


def _coerce_int(value: Any) -> int | None:
    """Return int if value is integer-like, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


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
        "summary_last_turn",
        "history_summary",
        "reasoning",
        "disband_mils",
        "keeps_word_report",
        "aggressor_report",
        "resource_grants",
    }
    validated: Dict[str, Any] = {}

    for key in list(action.keys()):
        if key not in allowed:
            _log(log_fn, f"Agent action field {key} ignored (not in schema)")

    purchase_mils = action.get("purchase_mils")
    purchase_val = _coerce_int(purchase_mils)
    if purchase_val is not None and purchase_val >= 0:
        validated["purchase_mils"] = purchase_val
    elif purchase_mils is not None:
        _log(log_fn, "Agent purchase_mils rejected due to schema")

    attacks = action.get("attacks")
    if isinstance(attacks, dict):
        clean_attacks: Dict[str, int] = {}
        for target, val in attacks.items():
            coerced = _coerce_int(val)
            if isinstance(target, str) and coerced is not None and coerced >= 0:
                clean_attacks[target] = coerced
        if len(clean_attacks) == len(attacks):
            validated["attacks"] = clean_attacks
        else:
            _log(log_fn, "Agent attacks rejected due to schema")
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
    if isinstance(money_grants, dict):
        clean_grants: Dict[str, int] = {}
        for recipient, val in money_grants.items():
            coerced = _coerce_int(val)
            if isinstance(recipient, str) and coerced is not None and coerced >= 0:
                clean_grants[recipient] = coerced
        if len(clean_grants) == len(money_grants):
            validated["money_grants"] = clean_grants
        else:
            _log(log_fn, "Agent money_grants rejected due to schema")
    elif money_grants is not None:
        _log(log_fn, "Agent money_grants rejected due to schema")

    messages = action.get("messages")
    if isinstance(messages, dict) and all(
        isinstance(v, str) for v in messages.values()
    ):
        validated["messages"] = messages
    elif messages is not None:
        _log(log_fn, "Agent messages rejected due to schema")

    summary_last_turn = action.get("summary_last_turn")
    if isinstance(summary_last_turn, str):
        validated["summary_last_turn"] = summary_last_turn
    elif summary_last_turn is not None:
        _log(log_fn, "Agent summary_last_turn rejected due to schema")

    history_summary = action.get("history_summary")
    if isinstance(history_summary, str):
        validated["history_summary"] = history_summary
    elif history_summary is not None:
        _log(log_fn, "Agent history_summary rejected due to schema")

    reasoning = action.get("reasoning")
    if isinstance(reasoning, str):
        validated["reasoning"] = reasoning
    elif reasoning is not None:
        _log(log_fn, "Agent reasoning rejected due to schema")

    disband_mils = action.get("disband_mils")
    disband_val = _coerce_int(disband_mils)
    if disband_val is not None and disband_val >= 0:
        validated["disband_mils"] = disband_val
    elif disband_mils is not None:
        _log(log_fn, "Agent disband_mils rejected due to schema")

    keeps_word_report = action.get("keeps_word_report")
    if isinstance(keeps_word_report, dict):
        validated["keeps_word_report"] = keeps_word_report
    elif keeps_word_report is not None:
        _log(log_fn, "Agent keeps_word_report rejected due to schema")

    aggressor_report = action.get("aggressor_report")
    if isinstance(aggressor_report, dict):
        validated["aggressor_report"] = aggressor_report
    elif aggressor_report is not None:
        _log(log_fn, "Agent aggressor_report rejected due to schema")

    resource_grants = action.get("resource_grants")
    if isinstance(resource_grants, dict):
        validated["resource_grants"] = resource_grants
    elif resource_grants is not None:
        _log(log_fn, "Agent resource_grants rejected due to schema")

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
    turns_left: int | None = None,
    history_contexts: Mapping[str, str] | None = None,
    legal_inbound_cessions: Mapping[str, Mapping[str, List[str]]] | None = None,
    legal_outbound_cessions: Mapping[str, Mapping[str, List[str]]] | None = None,
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
            if history_contexts and agent in history_contexts:
                agent_input["history_context"] = history_contexts[agent]
            if news_report:
                agent_input["previous_turn_news"] = news_report
            if agent_reports and agent in agent_reports:
                agent_input["previous_turn_report"] = agent_reports[agent]
        if turns_left is not None:
            agent_input["turns_left"] = turns_left
        if legal_inbound_cessions and agent in legal_inbound_cessions:
            agent_input["legal_inbound_cessions"] = legal_inbound_cessions[agent]
        if legal_outbound_cessions and agent in legal_outbound_cessions:
            agent_input["legal_outbound_cessions"] = legal_outbound_cessions[agent]
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
    territory_graph: Mapping[str, Set[str]] | None = None,
    max_summary_len: int = 2048,
    log_fn=None,
) -> Tuple[
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
]:
    """Translate raw agent JSON actions into intention ledgers."""

    d_mil_purchase_intent: Dict[str, int] = {}
    d_global_attacks: Dict[str, Dict[str, int]] = {}
    d_territory_cession: Dict[str, Dict[str, List[str]]] = {}
    d_money_grants: Dict[str, Dict[str, int]] = {}
    d_messages_sent: Dict[str, Dict[str, str]] = {}
    d_summary_last_turn: Dict[str, str] = {}
    d_history_summary: Dict[str, str] = {}
    d_reasoning: Dict[str, str] = {}
    d_mils_disband_intent: Dict[str, int] = {}
    d_keeps_word_report: Dict[str, Dict[str, int]] = {}
    d_aggressor_report: Dict[str, Dict[str, int]] = {}

    for agent, raw_action in agent_actions.items():
        if agent not in known_agents:
            continue

        purchase_mils = 0
        attacks: Dict[str, int] = {}
        cede_territories: Dict[str, List[str]] = {}
        money_grants: Dict[str, int] = {}
        messages: Dict[str, str] = {}
        summary_last_turn = ""
        history_summary = ""
        reasoning = ""
        disband_mils = 0
        keeps_word_report: Dict[str, int] = {}
        aggressor_report: Dict[str, int] = {}

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
                        if territory_graph is not None:
                            neighbors = territory_graph.get(tid, set())
                            if not any(
                                n in agent_territories.get(recipient, set()) for n in neighbors
                            ):
                                _log(
                                    log_fn,
                                    f"Agent {agent} illegal cession ignored: {tid} to {recipient}",
                                )
                                continue
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

            s = action.get("summary_last_turn", "")
            summary_last_turn = s if isinstance(s, str) else ""
            if max_summary_len >= 0:
                summary_last_turn = summary_last_turn[:max_summary_len]

            hs = action.get("history_summary", "")
            history_summary = hs if isinstance(hs, str) else ""
            if max_summary_len >= 0:
                history_summary = history_summary[:max_summary_len]

            r = action.get("reasoning", "")
            reasoning = r if isinstance(r, str) else ""

            disband_mils = _clamp_nonneg(_safe_int(action.get("disband_mils", 0), 0))

            raw_keeps_word = _as_dict(action.get("keeps_word_report", {}))
            if action.get("keeps_word_report", {}) != raw_keeps_word:
                _log(log_fn, f"Agent {agent} keeps_word_report rejected due to schema")
            for target, score_val in raw_keeps_word.items():
                if not isinstance(target, str):
                    continue
                if target not in known_agents:
                    continue
                score = _safe_int(score_val, 0)
                if 1 <= score <= 10:
                    keeps_word_report[target] = score

            raw_aggressor = _as_dict(action.get("aggressor_report", {}))
            if action.get("aggressor_report", {}) != raw_aggressor:
                _log(log_fn, f"Agent {agent} aggressor_report rejected due to schema")
            for target, score_val in raw_aggressor.items():
                if not isinstance(target, str):
                    continue
                if target not in known_agents:
                    continue
                score = _safe_int(score_val, 0)
                if 1 <= score <= 10:
                    aggressor_report[target] = score

        except Exception:
            purchase_mils = 0
            attacks = {}
            cede_territories = {}
            money_grants = {}
            messages = {}
            summary_last_turn = ""
            history_summary = ""
            reasoning = ""
            keeps_word_report = {}
            aggressor_report = {}

        d_mil_purchase_intent[agent] = purchase_mils
        d_global_attacks[agent] = attacks
        d_territory_cession[agent] = cede_territories
        d_money_grants[agent] = money_grants
        d_messages_sent[agent] = messages
        d_summary_last_turn[agent] = summary_last_turn
        d_history_summary[agent] = history_summary
        d_reasoning[agent] = reasoning
        d_mils_disband_intent[agent] = disband_mils
        d_keeps_word_report[agent] = keeps_word_report
        d_aggressor_report[agent] = aggressor_report

    return (
        d_mil_purchase_intent,
        d_global_attacks,
        d_territory_cession,
        d_money_grants,
        d_messages_sent,
        d_summary_last_turn,
        d_history_summary,
        d_reasoning,
        d_mils_disband_intent,
        d_keeps_word_report,
        d_aggressor_report,
    )


def run_phase0(
    *,
    turn: int,
    agents: Mapping[str, Any],
    agent_territories: Mapping[str, Set[str]],
    agent_mils: Mapping[str, int],
    constants: Mapping[str, Any],
    turn_summaries: Mapping[str, str],
    territory_graph: Mapping[str, Set[str]] | None = None,
    max_summary_len: int = 2048,
    log_fn=None,
) -> Tuple[
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
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
        territory_graph=territory_graph,
        max_summary_len=max_summary_len,
        log_fn=log_fn,
    )
