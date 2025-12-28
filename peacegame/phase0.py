from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Set, Tuple


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


def assemble_agent_inputs(
    *,
    turn: int,
    agent_names: List[str],
    agent_territories: Mapping[str, Set[str]],
    agent_mils: Mapping[str, int],
    constants: Mapping[str, Any],
    turn_summaries: Mapping[str, str],
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
        inputs[agent] = agent_input
    return inputs


def call_agents_collect_actions(
    *,
    agents: Mapping[str, Any],
    agent_inputs: Mapping[str, dict],
) -> Dict[str, dict]:
    """Safely call agents and collect raw action dictionaries."""
    actions: Dict[str, dict] = {}
    for agent_name, agent in agents.items():
        try:
            action = agent.act(agent_inputs[agent_name])
        except Exception:
            action = {}
        if not isinstance(action, dict):
            action = {}
        actions[agent_name] = action
    return actions


def translate_agent_actions_to_intentions(
    agent_actions: Mapping[str, Mapping[str, Any]],
    *,
    known_agents: Set[str],
    agent_territories: Mapping[str, Set[str]],
    max_summary_len: int = 2048,
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

    for agent, action in agent_actions.items():
        if agent not in known_agents:
            continue

        purchase_mils = 0
        attacks: Dict[str, int] = {}
        cede_territories: Dict[str, List[str]] = {}
        money_grants: Dict[str, int] = {}
        messages: Dict[str, str] = {}
        summary = ""

        try:
            action = _as_dict(action)

            purchase_mils = _clamp_nonneg(_safe_int(action.get("purchase_mils", 0), 0))

            raw_attacks = _as_dict(action.get("attacks", {}))
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
    )
