from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

from .phase0 import assemble_agent_inputs, call_agents_collect_actions, translate_agent_actions_to_intentions
from .territory_graph import (
    assign_territories_round_robin,
    build_territory_graph,
    compute_legal_cession_lists,
    is_legal_cession,
    load_territory_names,
)
from .visualizations import render_round_metrics


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def build_history_context(
    *,
    history_summary: str,
    turn_summaries: List[tuple[int, str]],
    max_chars: int,
) -> str:
    """Compose history summary plus recent per-turn summaries capped by max_chars."""
    if not history_summary and not turn_summaries:
        return ""

    parts: List[str] = []
    if history_summary:
        parts.append("History summary:")
        parts.append(history_summary)
    if turn_summaries:
        parts.append("Recent turns:")
        for turn, summary in reversed(turn_summaries):
            parts.append(f"turn {turn}: {summary}")

    combined = "\n".join(parts)
    if len(combined) <= max_chars:
        return combined

    recent_lines = [f"turn {turn}: {summary}" for turn, summary in reversed(turn_summaries)]
    recent_block = "\n".join(["Recent turns:"] + recent_lines) if recent_lines else ""
    if recent_block and len(recent_block) <= max_chars:
        if history_summary:
            header = "History summary:\n"
            remaining = max_chars - len(recent_block) - 1
            if remaining > len(header):
                allowed = remaining - len(header)
                return header + history_summary[:allowed] + "\n" + recent_block
        return recent_block

    if recent_lines:
        trimmed = ["Recent turns:"]
        for line in recent_lines:
            candidate = "\n".join(trimmed + [line])
            if len(candidate) > max_chars:
                if len(trimmed) == 1:
                    return line[:max_chars]
                break
            trimmed.append(line)
        return "\n".join(trimmed)

    return history_summary[:max_chars]


def _transpose_attacks(d_global_attacks: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    d_global_attacked: Dict[str, Dict[str, int]] = {}
    for attacker, targets in d_global_attacks.items():
        for target, mils in targets.items():
            d_global_attacked.setdefault(target, {})[attacker] = mils
    return d_global_attacked


def _allocate_losses_proportional(
    attackers: Dict[str, int],
    total_losses: int,
) -> Dict[str, int]:
    if total_losses <= 0 or not attackers:
        return {}
    total_attack = sum(attackers.values())
    if total_attack <= 0:
        return {}

    losses: Dict[str, int] = {}
    remainders: List[tuple[str, int]] = []

    for attacker, mils in attackers.items():
        share_num = mils * total_losses
        base = share_num // total_attack
        rem = share_num % total_attack
        if base > 0:
            losses[attacker] = base
        if rem > 0:
            remainders.append((attacker, rem))

    remaining = total_losses - sum(losses.values())
    if remaining > 0 and remainders:
        for attacker, _ in sorted(remainders, key=lambda x: (-x[1], x[0])):
            losses[attacker] = losses.get(attacker, 0) + 1
            remaining -= 1
            if remaining == 0:
                break

    return losses


def _clamp_attacks_to_mils(
    d_global_attacks: Dict[str, Dict[str, int]],
    agent_mils: Dict[str, int],
    *,
    log_fn=None,
) -> Dict[str, Dict[str, int]]:
    clamped: Dict[str, Dict[str, int]] = {}
    for attacker, targets in d_global_attacks.items():
        total_attack = sum(targets.values())
        available = agent_mils.get(attacker, 0)
        if total_attack <= available:
            clamped[attacker] = dict(targets)
            continue
        if available <= 0 or total_attack <= 0:
            clamped[attacker] = {}
            if log_fn is not None:
                log_fn(f"Agent {attacker} attack mils clamped to 0 (no available mils)")
            continue
        scaled = _allocate_losses_proportional(targets, available)
        clamped[attacker] = scaled
        if log_fn is not None:
            log_fn(
                f"Agent {attacker} attack mils clamped from {total_attack} to {available}"
            )
    return clamped


class SimulationEngine:
    """Full turn engine with logging."""

    def __init__(self, *, run_label: str = "simulation") -> None:
        self.run_label = run_label
        self._ensure_dirs()
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("logs", f"{run_label}_{run_id}.log")
        self._log_fp = open(self.log_path, "w", encoding="utf-8")
        self.agent_territories: Dict[str, Set[str]] = {}
        self.agent_mils: Dict[str, int] = {}
        self.agent_welfare: Dict[str, int] = {}
        self.agent_names: List[str] = []
        self.last_news_report: str = ""
        self.last_agent_reports: Dict[str, str] = {}
        self.last_purchase_price: int = 0
        self.total_turns: int | None = None
        self.last_upkeep_price: int = 0
        self.last_money_per_territory: int = 0
        self.last_damage_per_attack_mil: int = 0
        self.last_defense_destroy_factor: int = 0
        self.per_turn_metrics: Dict[str, Dict[str, List[int]]] = {}
        self.turns_seen: List[int] = []
        self.history_summary: Dict[str, str] = {}
        self.summary_log: Dict[str, List[tuple[int, str]]] = {}
        self.history_max_chars: int = 1000
        self.metric_keys: List[str] = []
        self.territory_graph: Dict[str, Set[str]] = {}
        self.territory_positions: Dict[str, tuple[int, int]] = {}
        self.territory_names: List[str] = []
        self.per_turn_territory_owners: List[List[str | None]] = []
        self.capital_territories: Dict[str, str] = {}

    def close(self) -> None:
        self._log_fp.close()

    def log(self, msg: str) -> None:
        self._log_fp.write(msg + "\n")
        self._log_fp.flush()

    def log_initial_state(
        self,
        *,
        script_name: str,
        agent_territories: Mapping[str, Set[str]],
        agent_mils: Mapping[str, int],
        agent_welfare: Mapping[str, int],
        constants: Mapping[str, Any],
        prompt_modifiers: Mapping[str, str] | None = None,
    ) -> None:
        self.log(f"Script {script_name} start")
        self.log(
            "Initial territories: "
            + str(sorted((a, sorted(list(t))) for a, t in agent_territories.items()))
        )
        self.log(f"Initial mils: {sorted(agent_mils.items())}")
        self.log(f"Initial welfare: {sorted(agent_welfare.items())}")
        self.log("Constants:")
        for key in sorted(constants.keys()):
            val = constants[key]
            if isinstance(val, float):
                self.log(f"  {key}: {_fmt_float(val)}")
            else:
                self.log(f"  {key}: {val}")
        if prompt_modifiers is not None:
            self.log(f"Prompt modifiers: {prompt_modifiers}")

    def log_script_end(self, *, script_name: str) -> None:
        self.log(f"Script {script_name} end")

    def setup_state(
        self,
        *,
        agent_territories: Mapping[str, Set[str]],
        agent_mils: Mapping[str, int],
        agent_welfare: Mapping[str, int],
        territory_seed: int | None = None,
        use_generated_territories: bool = False,
    ) -> None:
        self.agent_territories = {k: set(v) for k, v in agent_territories.items()}
        self.agent_mils = {k: int(v) for k, v in agent_mils.items()}
        self.agent_welfare = {k: int(v) for k, v in agent_welfare.items()}
        agent_names = set(self.agent_territories.keys()) | set(self.agent_mils.keys()) | set(
            self.agent_welfare.keys()
        )
        self.agent_names = sorted(agent_names)
        for agent in self.agent_names:
            self.agent_territories.setdefault(agent, set())
            self.agent_mils.setdefault(agent, 0)
            self.agent_welfare.setdefault(agent, 0)
            self.history_summary.setdefault(agent, "")
            self.summary_log.setdefault(agent, [])

        provided_territories = sorted(
            {t for terrs in self.agent_territories.values() for t in terrs}
        )
        if use_generated_territories:
            desired_count = len(provided_territories)
            territory_names = load_territory_names(Path("names") / "territories.txt")
            if desired_count > 0:
                if len(territory_names) < desired_count:
                    territory_names.extend(
                        [f"Terr{i}" for i in range(len(territory_names), desired_count)]
                    )
                territory_names = territory_names[:desired_count]
        else:
            territory_names = provided_territories
            if not territory_names:
                territory_names = load_territory_names(Path("names") / "territories.txt")
        self.territory_graph, self.territory_positions = build_territory_graph(
            territory_names,
            seed=territory_seed,
        )
        self.territory_names = sorted(self.territory_graph.keys())
        if use_generated_territories or all(len(terrs) == 0 for terrs in self.agent_territories.values()):
            assigned, capitals = assign_territories_round_robin(
                self.agent_names,
                self.territory_graph,
                self.territory_positions,
                seed=territory_seed,
                return_capitals=True,
            )
            self.agent_territories = {k: set(v) for k, v in assigned.items()}
            self.capital_territories = dict(capitals)
        else:
            self.capital_territories = {
                agent: sorted(list(terrs))[0]
                for agent, terrs in self.agent_territories.items()
                if terrs
            }

    def setup_round(self, *, total_turns: int) -> None:
        self.total_turns = int(total_turns)
        self.per_turn_metrics = {}
        self.turns_seen = []
        self.per_turn_territory_owners = []

    def run_turn(
        self,
        *,
        script_name: str,
        turn: int,
        agents: Mapping[str, Any],
        constants: Mapping[str, Any],
        turn_summaries: Dict[str, str],
        max_summary_len: int = 2048,
    ) -> Dict[str, Any]:
        self.log(f"Turn {turn} start for {script_name}")

        agent_territories = self.agent_territories
        agent_mils = self.agent_mils
        agent_welfare = self.agent_welfare

        legal_inbound, legal_outbound = compute_legal_cession_lists(
            agent_territories,
            self.territory_graph,
            capitals=self.capital_territories,
        )
        inputs = assemble_agent_inputs(
            turn=turn,
            agent_names=list(agents.keys()),
            agent_territories=agent_territories,
            agent_mils=agent_mils,
            constants=constants,
            turn_summaries=turn_summaries,
            news_report=self.last_news_report,
            agent_reports=self.last_agent_reports,
            turns_left=self._turns_left(turn),
            history_contexts=self._build_history_contexts(),
            legal_inbound_cessions=legal_inbound,
            legal_outbound_cessions=legal_outbound,
        )
        actions = call_agents_collect_actions(agents=agents, agent_inputs=inputs)
        self.log(f"Raw actions: {actions}")

        (
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
        ) = translate_agent_actions_to_intentions(
            actions,
            known_agents=set(agents.keys()),
            agent_territories=agent_territories,
            territory_graph=self.territory_graph,
            max_summary_len=max_summary_len,
            log_fn=self.log,
        )
        self.last_purchase_price = int(constants["c_mil_purchase_price"])
        self.last_upkeep_price = int(constants["c_mil_upkeep_price"])
        self.last_money_per_territory = int(constants["c_money_per_territory"])
        self.last_damage_per_attack_mil = int(constants["c_damage_per_attack_mil"])
        self.last_defense_destroy_factor = int(constants["c_defense_destroy_factor"])

        original_attacks = {k: dict(v) for k, v in d_global_attacks.items()}
        d_global_attacks = _clamp_attacks_to_mils(
            d_global_attacks, agent_mils, log_fn=self.log
        )
        attack_clamps: Dict[str, tuple[int, int]] = {}
        for attacker, targets in original_attacks.items():
            before = sum(targets.values())
            after = sum(d_global_attacks.get(attacker, {}).values())
            if before != after:
                attack_clamps[attacker] = (before, after)

        d_gross_income: Dict[str, int] = {}
        for agent, territories in agent_territories.items():
            d_gross_income[agent] = len(territories) * int(constants["c_money_per_territory"])

        d_global_attacked = _transpose_attacks(d_global_attacks)

        d_attacking_mils: Dict[str, int] = {}
        for agent, targets in d_global_attacks.items():
            d_attacking_mils[agent] = sum(targets.values())

        d_defense_mils: Dict[str, int] = {}
        for agent, total_mils in agent_mils.items():
            d_defense_mils[agent] = max(total_mils - d_attacking_mils.get(agent, 0), 0)

        d_damage_received: Dict[str, Dict[str, int]] = {}
        for target, attackers in d_global_attacked.items():
            for attacker, mils in attackers.items():
                d_damage_received.setdefault(target, {})[attacker] = (
                    mils * int(constants["c_damage_per_attack_mil"])
                )

        d_total_damage_received: Dict[str, int] = {}
        for target, dmg in d_damage_received.items():
            d_total_damage_received[target] = sum(dmg.values())

        d_available_money: Dict[str, int] = {}
        for agent in d_gross_income:
            d_available_money[agent] = max(
                d_gross_income[agent] - d_total_damage_received.get(agent, 0), 0
            )

        d_upkeep_cost: Dict[str, int] = {}
        d_mils_disbanded_upkeep: Dict[str, int] = {}
        start_mils = dict(agent_mils)
        for agent, mils in agent_mils.items():
            upkeep_price = int(constants["c_mil_upkeep_price"])
            d_upkeep_cost[agent] = mils * upkeep_price
            if d_available_money[agent] >= d_upkeep_cost[agent]:
                d_mils_disbanded_upkeep[agent] = 0
                d_available_money[agent] -= d_upkeep_cost[agent]
            else:
                deficit = d_upkeep_cost[agent] - d_available_money[agent]
                disband = (deficit + upkeep_price - 1) // upkeep_price
                d_mils_disbanded_upkeep[agent] = min(disband, agent_mils[agent])
                reduced_cost = d_upkeep_cost[agent] - d_mils_disbanded_upkeep[agent] * upkeep_price
                d_available_money[agent] = max(d_available_money[agent] - reduced_cost, 0)

        d_mil_purchased: Dict[str, int] = {}
        for agent, requested in d_mil_purchase_intent.items():
            affordable = d_available_money[agent] // int(constants["c_mil_purchase_price"])
            d_mil_purchased[agent] = min(requested, affordable)
            d_available_money[agent] -= d_mil_purchased[agent] * int(
                constants["c_mil_purchase_price"]
            )

        d_mils_lost_by_attacker: Dict[str, int] = {}
        for target, attackers in d_global_attacked.items():
            total_losses = d_defense_mils.get(target, 0) // int(
                constants["c_defense_destroy_factor"]
            )
            losses = _allocate_losses_proportional(attackers, total_losses)
            for attacker, loss in losses.items():
                d_mils_lost_by_attacker[attacker] = (
                    d_mils_lost_by_attacker.get(attacker, 0) + loss
                )

        d_mils_disbanded_total: Dict[str, int] = {}
        for agent in agent_mils:
            d_mils_disbanded_total[agent] = d_mils_disbanded_upkeep.get(agent, 0) + d_mils_lost_by_attacker.get(
                agent, 0
            )

        d_grants_received: Dict[str, Dict[str, int]] = {}
        for giver in sorted(d_money_grants.keys()):
            grants = d_money_grants[giver]
            for receiver in sorted(grants.keys()):
                amount = grants[receiver]
                if amount <= 0:
                    continue
                if d_available_money[giver] <= 0:
                    continue
                paid = min(amount, d_available_money[giver])
                if paid <= 0:
                    continue
                d_available_money[giver] -= paid
                d_grants_received.setdefault(receiver, {})[giver] = paid

        d_trade_bonus: Dict[str, int] = {}
        for agent, grants in d_grants_received.items():
            d_trade_bonus[agent] = int(
                sum(grants.values()) * (float(constants["c_trade_factor"]) - 1)
            )

        d_total_welfare_this_turn: Dict[str, int] = {}
        for agent in d_available_money:
            received = sum(d_grants_received.get(agent, {}).values())
            d_total_welfare_this_turn[agent] = int(
                d_available_money[agent]
                + received * float(constants["c_trade_factor"])
            )

        for agent in agent_welfare:
            agent_welfare[agent] += d_total_welfare_this_turn.get(agent, 0)

        for agent, lost in d_mils_disbanded_total.items():
            agent_mils[agent] = max(agent_mils[agent] - lost, 0)
        for agent, purchased in d_mil_purchased.items():
            agent_mils[agent] = agent_mils.get(agent, 0) + purchased

        d_mils_disbanded_voluntary: Dict[str, int] = {}
        for agent, requested in d_mils_disband_intent.items():
            current = agent_mils.get(agent, 0)
            disband = min(requested, current)
            d_mils_disbanded_voluntary[agent] = disband
            agent_mils[agent] = current - disband

        for giver, cessions in d_territory_cession.items():
            for receiver, terrs in cessions.items():
                for tid in terrs:
                    if tid not in agent_territories.get(giver, set()):
                        continue
                    if not is_legal_cession(
                        tid,
                        receiver,
                        agent_territories=agent_territories,
                        graph=self.territory_graph,
                        giver=giver,
                        capitals=self.capital_territories,
                    ):
                        self.log(
                            f"Illegal cession rejected: {giver} -> {receiver} for {tid}"
                        )
                        continue
                    agent_territories[giver].remove(tid)
                    agent_territories.setdefault(receiver, set()).add(tid)

        d_grants_paid: Dict[str, int] = {}
        for receiver, grants in d_grants_received.items():
            for giver, amt in grants.items():
                d_grants_paid[giver] = d_grants_paid.get(giver, 0) + amt

        for agent in self.agent_names:
            if d_history_summary.get(agent):
                self.history_summary[agent] = d_history_summary[agent]
            summary = d_summary_last_turn.get(agent, "")
            if summary:
                self.summary_log.setdefault(agent, []).append((turn, summary))

        self.last_news_report = self._build_news_report(
            d_global_attacks=d_global_attacks,
            d_mils_lost_by_attacker=d_mils_lost_by_attacker,
            d_total_damage_received=d_total_damage_received,
            d_mils_disbanded_upkeep=d_mils_disbanded_upkeep,
            d_messages_sent=d_messages_sent,
            d_grants_received=d_grants_received,
            d_territory_cession=d_territory_cession,
            d_gross_income=d_gross_income,
            attack_clamps=attack_clamps,
            d_mils_disbanded_voluntary=d_mils_disbanded_voluntary,
        )
        legal_inbound_after, legal_outbound_after = compute_legal_cession_lists(
            agent_territories,
            self.territory_graph,
            capitals=self.capital_territories,
        )
        self.last_agent_reports = self._build_agent_reports(
            d_gross_income=d_gross_income,
            d_total_damage_received=d_total_damage_received,
            d_upkeep_cost=d_upkeep_cost,
            d_mil_purchased=d_mil_purchased,
            d_mils_disbanded_upkeep=d_mils_disbanded_upkeep,
            d_mils_lost_by_attacker=d_mils_lost_by_attacker,
            d_total_welfare_this_turn=d_total_welfare_this_turn,
            d_available_money=d_available_money,
            d_grants_received=d_grants_received,
            d_grants_paid=d_grants_paid,
            d_trade_bonus=d_trade_bonus,
            trade_factor=float(constants["c_trade_factor"]),
            d_defense_mils=d_defense_mils,
            d_mils_disbanded_voluntary=d_mils_disbanded_voluntary,
            start_mils=start_mils,
            end_mils=agent_mils,
            total_welfare=agent_welfare,
            d_reasoning=d_reasoning,
            d_keeps_word_report=d_keeps_word_report,
            d_aggressor_report=d_aggressor_report,
            legal_inbound_cessions=legal_inbound_after,
            legal_outbound_cessions=legal_outbound_after,
        )
        self.log("Previous turn news report:")
        self.log(self.last_news_report)
        self.log("Previous turn agent reports:")
        for agent in sorted(self.last_agent_reports.keys()):
            self.log(f"[{agent}]")
            self.log(self.last_agent_reports[agent])

        self.log(f"Turn {turn} welfare: {d_total_welfare_this_turn}")
        self.log(f"Turn {turn} end mils: {agent_mils}")
        self.log(f"Turn {turn} end territories: {agent_territories}")

        self._record_metrics(
            turn=turn,
            d_total_welfare_this_turn=d_total_welfare_this_turn,
            agent_welfare=agent_welfare,
            d_attacking_mils=d_attacking_mils,
            d_global_attacked=d_global_attacked,
            agent_mils=agent_mils,
            d_mils_lost_by_attacker=d_mils_lost_by_attacker,
            d_mils_disbanded_upkeep=d_mils_disbanded_upkeep,
            d_mils_disbanded_voluntary=d_mils_disbanded_voluntary,
            d_grants_paid=d_grants_paid,
            d_grants_received=d_grants_received,
            trade_factor=float(constants["c_trade_factor"]),
            agent_territories=agent_territories,
        )
        if self.total_turns is not None and turn == self.total_turns - 1:
            self._render_visualization()
            self._write_round_data()

        return {
            "d_mil_purchase_intent": d_mil_purchase_intent,
            "d_global_attacks": d_global_attacks,
            "d_territory_cession": d_territory_cession,
            "d_money_grants": d_money_grants,
            "d_messages_sent": d_messages_sent,
            "d_summary_last_turn": d_summary_last_turn,
            "d_history_summary": d_history_summary,
            "d_reasoning": d_reasoning,
            "d_keeps_word_report": d_keeps_word_report,
            "d_aggressor_report": d_aggressor_report,
            "d_gross_income": d_gross_income,
            "d_attacking_mils": d_attacking_mils,
            "d_defense_mils": d_defense_mils,
            "d_total_damage_received": d_total_damage_received,
            "d_upkeep_cost": d_upkeep_cost,
            "d_mils_disbanded_upkeep": d_mils_disbanded_upkeep,
            "d_mil_purchased": d_mil_purchased,
            "d_mils_lost_by_attacker": d_mils_lost_by_attacker,
            "d_grants_received": d_grants_received,
            "d_trade_bonus": d_trade_bonus,
            "d_total_welfare_this_turn": d_total_welfare_this_turn,
            "agent_mils": dict(agent_mils),
            "agent_welfare": dict(agent_welfare),
            "agent_territories": {k: sorted(list(v)) for k, v in agent_territories.items()},
            "news_report": self.last_news_report,
            "agent_reports": dict(self.last_agent_reports),
        }

    def _build_news_report(
        self,
        *,
        d_global_attacks: Dict[str, Dict[str, int]],
        d_mils_lost_by_attacker: Dict[str, int],
        d_total_damage_received: Dict[str, int],
        d_mils_disbanded_upkeep: Dict[str, int],
        d_messages_sent: Dict[str, Dict[str, str]],
        d_grants_received: Dict[str, Dict[str, int]],
        d_territory_cession: Dict[str, Dict[str, List[str]]],
        d_gross_income: Dict[str, int],
        attack_clamps: Dict[str, tuple[int, int]],
        d_mils_disbanded_voluntary: Dict[str, int],
    ) -> str:
        lines: List[str] = []

        lines.append("Attacks:")
        attack_lines = []
        for attacker in sorted(d_global_attacks.keys()):
            for target in sorted(d_global_attacks[attacker].keys()):
                mils = d_global_attacks[attacker][target]
                if mils > 0:
                    attack_lines.append(f" - {attacker} -> {target}: {mils}")
        lines.extend(attack_lines if attack_lines else [" - none"])

        lines.append("Damage to attackers:")
        dmg_lines = []
        for attacker in sorted(d_mils_lost_by_attacker.keys()):
            dmg_lines.append(f" - {attacker}: {d_mils_lost_by_attacker[attacker]}")
        lines.extend(dmg_lines if dmg_lines else [" - none"])

        lines.append("Income damage:")
        income_lines = []
        for agent in sorted(d_total_damage_received.keys()):
            income_lines.append(f" - {agent}: {d_total_damage_received[agent]}")
        lines.extend(income_lines if income_lines else [" - none"])

        lines.append("Damage cap:")
        cap_lines = []
        for agent in sorted(d_total_damage_received.keys()):
            if d_total_damage_received[agent] > d_gross_income.get(agent, 0):
                cap_lines.append(
                    f" - {agent}: capped at {d_gross_income.get(agent, 0)}"
                )
        lines.extend(cap_lines if cap_lines else [" - none"])

        lines.append("Upkeep disband:")
        disband_lines = []
        for agent in sorted(d_mils_disbanded_upkeep.keys()):
            if d_mils_disbanded_upkeep[agent] > 0:
                disband_lines.append(
                    f" - {agent}: {d_mils_disbanded_upkeep[agent]}"
                )
        lines.extend(disband_lines if disband_lines else [" - none"])

        lines.append("Voluntary disband:")
        voluntary_lines = []
        for agent in sorted(d_mils_disbanded_voluntary.keys()):
            if d_mils_disbanded_voluntary[agent] > 0:
                voluntary_lines.append(
                    f" - {agent}: {d_mils_disbanded_voluntary[agent]}"
                )
        lines.extend(voluntary_lines if voluntary_lines else [" - none"])

        lines.append("Messages:")
        msg_lines = []
        for sender in sorted(d_messages_sent.keys()):
            for recipient in sorted(d_messages_sent[sender].keys()):
                msg = d_messages_sent[sender][recipient]
                msg_lines.append(f" - {sender} -> {recipient}: {msg}")
        lines.extend(msg_lines if msg_lines else [" - none"])

        lines.append("Grants:")
        grant_lines = []
        for receiver in sorted(d_grants_received.keys()):
            for giver in sorted(d_grants_received[receiver].keys()):
                amt = d_grants_received[receiver][giver]
                grant_lines.append(f" - {giver} -> {receiver}: {amt}")
        lines.extend(grant_lines if grant_lines else [" - none"])

        lines.append("Cessions:")
        cession_lines = []
        for giver in sorted(d_territory_cession.keys()):
            for receiver in sorted(d_territory_cession[giver].keys()):
                terrs = d_territory_cession[giver][receiver]
                if terrs:
                    cession_lines.append(
                        f" - {giver} -> {receiver}: {', '.join(sorted(terrs))}"
                    )
        lines.extend(cession_lines if cession_lines else [" - none"])

        lines.append("Attack limits:")
        clamp_lines = []
        for attacker in sorted(attack_clamps.keys()):
            before, after = attack_clamps[attacker]
            clamp_lines.append(f" - {attacker}: {before} -> {after}")
        lines.extend(clamp_lines if clamp_lines else [" - none"])

        return "\n".join(lines)

    def _build_agent_reports(
        self,
        *,
        d_gross_income: Dict[str, int],
        d_total_damage_received: Dict[str, int],
        d_upkeep_cost: Dict[str, int],
        d_mil_purchased: Dict[str, int],
        d_mils_disbanded_upkeep: Dict[str, int],
        d_mils_lost_by_attacker: Dict[str, int],
        d_total_welfare_this_turn: Dict[str, int],
        d_available_money: Dict[str, int],
        d_grants_received: Dict[str, Dict[str, int]],
        d_grants_paid: Dict[str, int],
        d_trade_bonus: Dict[str, int],
        trade_factor: float,
        d_defense_mils: Dict[str, int],
        d_mils_disbanded_voluntary: Dict[str, int],
        start_mils: Dict[str, int],
        end_mils: Dict[str, int],
        total_welfare: Dict[str, int],
        d_reasoning: Dict[str, str],
        d_keeps_word_report: Dict[str, Dict[str, int]],
        d_aggressor_report: Dict[str, Dict[str, int]],
        legal_inbound_cessions: Dict[str, Dict[str, List[str]]],
        legal_outbound_cessions: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, str]:
        reports: Dict[str, str] = {}
        ranked = sorted(total_welfare.items(), key=lambda x: (-x[1], x[0]))
        ranks = {agent: idx + 1 for idx, (agent, _) in enumerate(ranked)}
        total_agents = len(ranked)

        for agent in sorted(d_gross_income.keys()):
            gross = d_gross_income.get(agent, 0)
            damage = d_total_damage_received.get(agent, 0)
            upkeep = d_upkeep_cost.get(agent, 0)
            purchased = d_mil_purchased.get(agent, 0)
            purchase_cost = purchased * self.last_purchase_price
            lost = d_mils_lost_by_attacker.get(agent, 0)
            disbanded = d_mils_disbanded_upkeep.get(agent, 0)
            welfare_this = d_total_welfare_this_turn.get(agent, 0)
            available = d_available_money.get(agent, 0)
            grants_received = sum(d_grants_received.get(agent, {}).values())
            grants_paid = d_grants_paid.get(agent, 0)
            trade_bonus = d_trade_bonus.get(agent, 0)
            defense_mils = d_defense_mils.get(agent, 0)
            voluntary = d_mils_disbanded_voluntary.get(agent, 0)
            total = total_welfare.get(agent, 0)
            rank = ranks.get(agent, total_agents)
            end = end_mils.get(agent, 0)
            start = start_mils.get(agent, 0)
            damage_capped = min(damage, gross)
            damage_wasted = max(damage - gross, 0)
            reasoning = d_reasoning.get(agent, "")
            keeps_word = d_keeps_word_report.get(agent, {})
            aggressor = d_aggressor_report.get(agent, {})
            legal_inbound = legal_inbound_cessions.get(agent, {})
            legal_outbound = legal_outbound_cessions.get(agent, {})

            upkeep_price = self.last_upkeep_price
            purchase_price = self.last_purchase_price
            lines = [
                f"Income: gross={gross} (territories * {self.last_money_per_territory}), damage={damage} (attacks * {self.last_damage_per_attack_mil}), capped={damage_capped}, wasted={damage_wasted}",
                f"Costs: upkeep={upkeep} ({start} units * {upkeep_price}), purchases={purchase_cost} ({purchased} units * {purchase_price})",
                "Grants: received={gr}, paid={gp}, trade_bonus={tb}, trade_factor={tf}".format(
                    gr=grants_received,
                    gp=grants_paid,
                    tb=_fmt_float(float(trade_bonus)),
                    tf=_fmt_float(trade_factor),
                ),
                f"Defense: defense_mils={defense_mils}, attacker_losses=defense_mils/{self.last_defense_destroy_factor}",
                f"Army: start={start}, lost={lost}, disbanded={disbanded}, voluntary_disband={voluntary}, purchased={purchased}, end={end}",
                f"Reasoning (last turn): {reasoning}",
                f"Keeps word report: {keeps_word}",
                f"Aggressor report: {aggressor}",
                f"Legal inbound cessions: {legal_inbound}",
                f"Legal outbound cessions: {legal_outbound}",
                "Welfare: this_turn={w} = gross({gross}) - damage({damage}) - upkeep({upkeep}) - purchases({purchase_cost}) - grants_paid({gp}) + grants_received({g})*trade_factor({tf}) = available_money({a}) + trade_bonus({tb}); total={t}, rank={r}/{n}".format(
                    w=welfare_this,
                    gross=gross,
                    damage=damage,
                    upkeep=upkeep,
                    purchase_cost=purchase_cost,
                    gp=grants_paid,
                    g=grants_received,
                    tf=_fmt_float(trade_factor),
                    a=available,
                    tb=_fmt_float(float(trade_bonus)),
                    t=total,
                    r=rank,
                    n=total_agents,
                ),
            ]
            reports[agent] = "\n".join(lines)

        return reports

    def _turns_left(self, turn: int) -> int | None:
        if self.total_turns is None:
            return None
        remaining = self.total_turns - (turn + 1)
        return max(remaining, 0)

    def _build_history_contexts(self) -> Dict[str, str]:
        contexts: Dict[str, str] = {}
        for agent in self.agent_names:
            contexts[agent] = build_history_context(
                history_summary=self.history_summary.get(agent, ""),
                turn_summaries=self.summary_log.get(agent, []),
                max_chars=self.history_max_chars,
            )
        return contexts

    def _ensure_dirs(self) -> None:
        os.makedirs("logs", exist_ok=True)

    def _record_metrics(
        self,
        *,
        turn: int,
        d_total_welfare_this_turn: Dict[str, int],
        agent_welfare: Dict[str, int],
        d_attacking_mils: Dict[str, int],
        d_global_attacked: Dict[str, Dict[str, int]],
        agent_mils: Dict[str, int],
        d_mils_lost_by_attacker: Dict[str, int],
        d_mils_disbanded_upkeep: Dict[str, int],
        d_grants_paid: Dict[str, int],
        d_grants_received: Dict[str, Dict[str, int]],
        trade_factor: float,
        d_mils_disbanded_voluntary: Dict[str, int],
        agent_territories: Dict[str, Set[str]],
    ) -> None:
        if not self.per_turn_metrics:
            keys = [
                "total_welfare",
                "welfare_this_turn",
                "attacks",
                "attacks_received",
                "army_size",
                "territories",
                "mils_destroyed",
                "mils_disbanded",
                "trade_sent",
                "trade_welfare_received",
            ]
            self.metric_keys = list(keys)
            self.per_turn_metrics = {k: {a: [] for a in self.agent_names} for k in keys}

        self.turns_seen.append(turn)
        for agent in self.agent_names:
            attacks_received = sum(
                d_global_attacked.get(agent, {}).values()
            )
            grants_received = sum(d_grants_received.get(agent, {}).values())
            self.per_turn_metrics["total_welfare"][agent].append(
                agent_welfare.get(agent, 0)
            )
            self.per_turn_metrics["welfare_this_turn"][agent].append(
                d_total_welfare_this_turn.get(agent, 0)
            )
            self.per_turn_metrics["attacks"][agent].append(
                d_attacking_mils.get(agent, 0)
            )
            self.per_turn_metrics["attacks_received"][agent].append(attacks_received)
            self.per_turn_metrics["army_size"][agent].append(agent_mils.get(agent, 0))
            self.per_turn_metrics["territories"][agent].append(
                len(agent_territories.get(agent, set()))
            )
            self.per_turn_metrics["mils_destroyed"][agent].append(
                d_mils_lost_by_attacker.get(agent, 0)
            )
            self.per_turn_metrics["mils_disbanded"][agent].append(
                d_mils_disbanded_upkeep.get(agent, 0)
                + d_mils_disbanded_voluntary.get(agent, 0)
            )
            self.per_turn_metrics["trade_sent"][agent].append(
                d_grants_paid.get(agent, 0)
            )
            self.per_turn_metrics["trade_welfare_received"][agent].append(
                int(grants_received * trade_factor)
            )

        owner_by_territory: Dict[str, str] = {}
        for agent, terrs in agent_territories.items():
            for terr in terrs:
                owner_by_territory[terr] = agent
        owners = [owner_by_territory.get(name) for name in self.territory_names]
        self.per_turn_territory_owners.append(owners)

    def _render_visualization(self) -> None:
        if not self.per_turn_metrics or not self.turns_seen:
            return
        out_dir = Path("visualizations")
        log_stem = Path(self.log_path).stem
        out_path = out_dir / f"{log_stem}.png"
        render_round_metrics(
            output_path=out_path,
            turns=self.turns_seen,
            agents=self.agent_names,
            series=self.per_turn_metrics,
        )

    def _write_round_data(self) -> None:
        if not self.per_turn_metrics or not self.turns_seen:
            return
        ledger_vars = self.metric_keys or list(self.per_turn_metrics.keys())
        data: List[List[List[int]]] = []
        for agent in self.agent_names:
            agent_rows: List[List[int]] = []
            for idx, _turn in enumerate(self.turns_seen):
                row = []
                for key in ledger_vars:
                    values = self.per_turn_metrics.get(key, {}).get(agent, [])
                    row.append(values[idx] if idx < len(values) else 0)
                agent_rows.append(row)
            data.append(agent_rows)

        payload = {
            "agents": self.agent_names,
            "turns": self.turns_seen,
            "ledger_vars": ledger_vars,
            "data": data,
            "territory_names": self.territory_names,
            "territory_positions": {
                name: list(self.territory_positions.get(name, (0, 0)))
                for name in self.territory_names
            },
            "territory_owners": self.per_turn_territory_owners,
        }
        out_dir = Path("round_data")
        log_stem = Path(self.log_path).stem
        out_path = out_dir / f"{log_stem}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
