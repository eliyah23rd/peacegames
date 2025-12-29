from __future__ import annotations

import datetime as dt
import html
import json
import os
from typing import Any, Dict, List, Mapping, Set

from .phase0 import assemble_agent_inputs, call_agents_collect_actions, translate_agent_actions_to_intentions


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
    """Full turn engine with logging and datasheets."""

    def __init__(self, *, run_label: str = "simulation") -> None:
        self.run_label = run_label
        self._ensure_dirs()
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("logs", f"{run_label}_{run_id}.log")
        self.sheet_path = os.path.join("datasheets", f"{run_label}_{run_id}.xls")
        self._log_fp = open(self.log_path, "w", encoding="utf-8")
        self._rows: List[List[str]] = []
        self.agent_territories: Dict[str, Set[str]] = {}
        self.agent_mils: Dict[str, int] = {}
        self.agent_welfare: Dict[str, int] = {}
        self.agent_names: List[str] = []
        self.last_news_report: str = ""
        self.last_agent_reports: Dict[str, str] = {}
        self.last_purchase_price: int = 0
        self.total_turns: int | None = None

    def close(self) -> None:
        headers = ["script", "turn", "agent", "phase", "ledger", "value"]
        self._write_xls(self.sheet_path, headers, self._rows)
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
    ) -> None:
        self.log(f"Script {script_name} start")
        self.log(
            "Initial territories: "
            + str(sorted((a, sorted(list(t))) for a, t in agent_territories.items()))
        )
        self.log(f"Initial mils: {sorted(agent_mils.items())}")
        self.log(f"Initial welfare: {sorted(agent_welfare.items())}")
        self.log(f"Constants: {constants}")

    def log_script_end(self, *, script_name: str) -> None:
        self.log(f"Script {script_name} end")

    def setup_state(
        self,
        *,
        agent_territories: Mapping[str, Set[str]],
        agent_mils: Mapping[str, int],
        agent_welfare: Mapping[str, int],
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

    def setup_round(self, *, total_turns: int) -> None:
        self.total_turns = int(total_turns)

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
        )
        actions = call_agents_collect_actions(agents=agents, agent_inputs=inputs)
        self.log(f"Raw actions: {actions}")

        (
            d_mil_purchase_intent,
            d_global_attacks,
            d_territory_cession,
            d_money_grants,
            d_messages_sent,
            d_turn_summary,
        ) = translate_agent_actions_to_intentions(
            actions,
            known_agents=set(agents.keys()),
            agent_territories=agent_territories,
            max_summary_len=max_summary_len,
            log_fn=self.log,
        )
        self.last_purchase_price = int(constants["c_mil_purchase_price"])

        d_global_attacks = _clamp_attacks_to_mils(
            d_global_attacks, agent_mils, log_fn=self.log
        )

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

        for giver, cessions in d_territory_cession.items():
            for receiver, terrs in cessions.items():
                for tid in terrs:
                    if tid in agent_territories.get(giver, set()):
                        agent_territories[giver].remove(tid)
                        agent_territories.setdefault(receiver, set()).add(tid)

        self.last_news_report = self._build_news_report(
            d_global_attacks=d_global_attacks,
            d_mils_lost_by_attacker=d_mils_lost_by_attacker,
            d_total_damage_received=d_total_damage_received,
            d_mils_disbanded_upkeep=d_mils_disbanded_upkeep,
            d_messages_sent=d_messages_sent,
            d_grants_received=d_grants_received,
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
            d_trade_bonus=d_trade_bonus,
            start_mils=start_mils,
            end_mils=agent_mils,
            total_welfare=agent_welfare,
        )
        self.log("Previous turn news report:")
        self.log(self.last_news_report)
        self.log("Previous turn agent reports:")
        for agent in sorted(self.last_agent_reports.keys()):
            self.log(f"[{agent}]")
            self.log(self.last_agent_reports[agent])

        self._record_phase_rows(
            script_name,
            turn,
            "phase0",
            ("d_turn_summary", d_turn_summary),
            ("d_mil_purchase_intent", d_mil_purchase_intent),
            ("d_global_attacks", d_global_attacks),
            ("d_territory_cession", d_territory_cession),
            ("d_money_grants", d_money_grants),
            ("d_messages_sent", d_messages_sent),
        )
        self._record_phase_rows(script_name, turn, "phase1", ("d_gross_income", d_gross_income))
        self._record_phase_rows(
            script_name,
            turn,
            "phase2",
            ("d_attacking_mils", d_attacking_mils),
            ("d_defense_mils", d_defense_mils),
        )
        self._record_phase_rows(
            script_name,
            turn,
            "phase3",
            ("d_total_damage_received", d_total_damage_received),
        )
        self._record_phase_rows(
            script_name,
            turn,
            "phase4",
            ("d_upkeep_cost", d_upkeep_cost),
            ("d_mils_disbanded_upkeep", d_mils_disbanded_upkeep),
        )
        self._record_phase_rows(
            script_name,
            turn,
            "phase5",
            ("d_mil_purchased", d_mil_purchased),
        )
        self._record_phase_rows(
            script_name,
            turn,
            "phase6",
            ("d_mils_lost_by_attacker", d_mils_lost_by_attacker),
        )
        self._record_phase_rows(
            script_name,
            turn,
            "phase7",
            ("d_total_welfare_this_turn", d_total_welfare_this_turn),
        )
        self._record_phase_rows(
            script_name,
            turn,
            "state",
            ("agent_mils", agent_mils),
            ("agent_welfare", agent_welfare),
        )

        self.log(f"Turn {turn} welfare: {d_total_welfare_this_turn}")
        self.log(f"Turn {turn} end mils: {agent_mils}")
        self.log(f"Turn {turn} end territories: {agent_territories}")

        return {
            "d_mil_purchase_intent": d_mil_purchase_intent,
            "d_global_attacks": d_global_attacks,
            "d_territory_cession": d_territory_cession,
            "d_money_grants": d_money_grants,
            "d_messages_sent": d_messages_sent,
            "d_turn_summary": d_turn_summary,
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

    def _record_phase_rows(
        self,
        script: str,
        turn: int,
        phase: str,
        *ledgers: tuple[str, Dict[str, Any]],
    ) -> None:
        for ledger_name, ledger in ledgers:
            if not ledger:
                continue
            for agent in sorted(ledger.keys()):
                self._rows.append(
                    [
                        script,
                        str(turn),
                        agent,
                        phase,
                        ledger_name,
                        json.dumps(ledger.get(agent), sort_keys=True),
                    ]
                )

    def _build_news_report(
        self,
        *,
        d_global_attacks: Dict[str, Dict[str, int]],
        d_mils_lost_by_attacker: Dict[str, int],
        d_total_damage_received: Dict[str, int],
        d_mils_disbanded_upkeep: Dict[str, int],
        d_messages_sent: Dict[str, Dict[str, str]],
        d_grants_received: Dict[str, Dict[str, int]],
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

        lines.append("Upkeep disband:")
        disband_lines = []
        for agent in sorted(d_mils_disbanded_upkeep.keys()):
            if d_mils_disbanded_upkeep[agent] > 0:
                disband_lines.append(
                    f" - {agent}: {d_mils_disbanded_upkeep[agent]}"
                )
        lines.extend(disband_lines if disband_lines else [" - none"])

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
        d_trade_bonus: Dict[str, int],
        start_mils: Dict[str, int],
        end_mils: Dict[str, int],
        total_welfare: Dict[str, int],
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
            trade_bonus = d_trade_bonus.get(agent, 0)
            total = total_welfare.get(agent, 0)
            rank = ranks.get(agent, total_agents)
            end = end_mils.get(agent, 0)
            start = start_mils.get(agent, 0)

            lines = [
                f"Income: gross={gross}, damage={damage}",
                f"Costs: upkeep={upkeep}, purchases={purchase_cost}",
                f"Grants: received={grants_received}, trade_bonus={trade_bonus}",
                f"Army: start={start}, lost={lost}, disbanded={disbanded}, purchased={purchased}, end={end}",
                "Welfare: this_turn={w} = available_money={a} + grants_received={g} * trade_factor, total={t}, rank={r}/{n}".format(
                    w=welfare_this,
                    a=available,
                    g=grants_received,
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

    def _ensure_dirs(self) -> None:
        os.makedirs("logs", exist_ok=True)
        os.makedirs("datasheets", exist_ok=True)

    def _write_xls(self, path: str, headers: list[str], rows: list[list[str]]) -> None:
        lines = [
            "<html><head><meta charset=\"utf-8\"></head><body>",
            "<table border=\"1\">",
            "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>",
        ]
        for row in rows:
            ledger = row[4] if len(row) > 4 else ""
            value_attr = ""
            if ledger == "agent_welfare":
                value_attr = " bgcolor=\"#e6ffe6\""
            elif ledger == "agent_mils":
                value_attr = " bgcolor=\"#e6f0ff\""
            lines.append(
                "<tr>"
                + "".join(
                    f"<td{value_attr if idx == 5 else ''}>{html.escape(str(cell))}</td>"
                    for idx, cell in enumerate(row)
                )
                + "</tr>"
            )
        lines.append("</table></body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
