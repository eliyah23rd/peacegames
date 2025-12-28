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

    def run_turn(
        self,
        *,
        script_name: str,
        turn: int,
        agents: Mapping[str, Any],
        agent_territories: Dict[str, Set[str]],
        agent_mils: Dict[str, int],
        agent_welfare: Dict[str, int],
        constants: Mapping[str, Any],
        turn_summaries: Dict[str, str],
        max_summary_len: int = 2048,
    ) -> Dict[str, Any]:
        self.log(f"Turn {turn} start for {script_name}")

        inputs = assemble_agent_inputs(
            turn=turn,
            agent_names=list(agents.keys()),
            agent_territories=agent_territories,
            agent_mils=agent_mils,
            constants=constants,
            turn_summaries=turn_summaries,
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
        for agent, mils in agent_mils.items():
            d_upkeep_cost[agent] = mils * int(constants["c_mil_upkeep_price"])
            if d_available_money[agent] >= d_upkeep_cost[agent]:
                d_mils_disbanded_upkeep[agent] = 0
                d_available_money[agent] -= d_upkeep_cost[agent]
            else:
                deficit = d_upkeep_cost[agent] - d_available_money[agent]
                disband = (deficit + int(constants["c_mil_upkeep_price"]) - 1) // int(
                    constants["c_mil_upkeep_price"]
                )
                d_mils_disbanded_upkeep[agent] = min(disband, agent_mils[agent])
                d_available_money[agent] = 0

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
            lines.append(
                "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
            )
        lines.append("</table></body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
