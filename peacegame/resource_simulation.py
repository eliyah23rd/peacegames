from __future__ import annotations

import datetime as dt
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

from .phase0 import assemble_agent_inputs, call_agents_collect_actions
from .simulation import _allocate_losses_proportional, _clamp_attacks_to_mils, _transpose_attacks
from .territory_graph import (
    assign_territories_round_robin,
    compute_legal_cession_lists,
    is_legal_cession,
    load_territory_names,
)


RESOURCE_TYPES = ("energy", "minerals", "food")


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def generate_territory_resources(
    territory_names: List[str],
    graph: Dict[str, Set[str]],
    *,
    peaks_per_resource: Mapping[str, int] | None = None,
    max_value: int = 3,
    resource_adjacent_pct: int = 50,
    resource_one_pct: int = 50,
    seed: int | None = None,
) -> Dict[str, Dict[str, int]]:
    rng = random.Random(seed)
    resources: Dict[str, Dict[str, int]] = {t: {} for t in territory_names}
    if not territory_names:
        return resources

    peaks = dict(peaks_per_resource or {})
    for rtype in RESOURCE_TYPES:
        count = int(peaks.get(rtype, 1))
        if count <= 0:
            continue
        count = min(count, len(territory_names))
        peak_nodes = rng.sample(territory_names, count)

        assigned: Set[str] = set()
        tier_three: Set[str] = set()

        for terr in peak_nodes:
            resources[terr][rtype] = max_value
            assigned.add(terr)
            tier_three.add(terr)

        # Adjacent to peaks: chance to also be max.
        for peak in peak_nodes:
            for neighbor in graph.get(peak, set()):
                if neighbor in assigned:
                    continue
                if rng.randint(1, 100) <= resource_adjacent_pct:
                    resources[neighbor][rtype] = max_value
                    assigned.add(neighbor)
                    tier_three.add(neighbor)

        # Adjacent to any max resource: assign 3 or 2.
        candidates = set()
        for terr in tier_three:
            candidates.update(graph.get(terr, set()))
        for terr in sorted(candidates):
            if terr in assigned:
                continue
            resources[terr][rtype] = max_value if rng.randint(1, 100) <= 50 else max_value - 1
            assigned.add(terr)

        # Random 1s across the remaining map.
        for terr in territory_names:
            if terr in assigned:
                continue
            if rng.randint(1, 100) <= resource_one_pct:
                resources[terr][rtype] = 1
    return resources


def _resource_totals(
    agent_territories: Dict[str, Set[str]],
    territory_resources: Dict[str, Dict[str, int]],
    incoming: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    totals: Dict[str, Dict[str, int]] = {}
    for agent, terrs in agent_territories.items():
        res = {k: 0 for k in RESOURCE_TYPES}
        for terr in terrs:
            for rtype, qty in territory_resources.get(terr, {}).items():
                res[rtype] += int(qty)
        for rtype, qty in incoming.get(agent, {}).items():
            res[rtype] += int(qty)
        totals[agent] = res
    return totals


def _resource_ratios(
    totals: Dict[str, Dict[str, int]],
    agent_territories: Dict[str, Set[str]],
    constants: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    ratios: Dict[str, Dict[str, float]] = {}
    for agent, terrs in agent_territories.items():
        count = len(terrs)
        if count <= 0:
            ratios[agent] = {k: 0.0 for k in RESOURCE_TYPES}
            continue
        r = {}
        r["energy"] = min(1.0, totals[agent]["energy"] / (count * constants["c_min_energy"]))
        r["minerals"] = min(1.0, totals[agent]["minerals"] / (count * constants["c_min_minerals"]))
        r["food"] = min(1.0, totals[agent]["food"] / (count * constants["c_min_food"]))
        ratios[agent] = r
    return ratios


def _resource_multiplier(ratios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for agent, r in ratios.items():
        out[agent] = float(r["energy"] * r["minerals"] * r["food"])
    return out


def _parse_resource_grants(
    actions: Mapping[str, Any],
    *,
    known_agents: Set[str],
    territory_resources: Dict[str, Dict[str, int]],
    agent_territories: Dict[str, Set[str]],
    log_fn=None,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    grants: Dict[str, Dict[str, Dict[str, int]]] = {a: {} for a in known_agents}
    for agent, raw_action in actions.items():
        if agent not in known_agents:
            continue
        if not isinstance(raw_action, str):
            continue
        try:
            parsed = json.loads(raw_action)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        raw_grants = parsed.get("resource_grants", {})
        if not isinstance(raw_grants, dict):
            continue

        # available resources (production only) this turn
        available = {k: 0 for k in RESOURCE_TYPES}
        for terr in agent_territories.get(agent, set()):
            for rtype, qty in territory_resources.get(terr, {}).items():
                available[rtype] += int(qty)

        for recipient, res_map in raw_grants.items():
            if not isinstance(recipient, str):
                continue
            if recipient == agent or recipient not in known_agents:
                continue
            if not isinstance(res_map, dict):
                continue
            clean: Dict[str, int] = {}
            for rtype in RESOURCE_TYPES:
                val = res_map.get(rtype, 0)
                if isinstance(val, int) and val > 0:
                    clean[rtype] = val
            if not clean:
                continue

            # clamp per resource to available
            for rtype, qty in clean.items():
                allowed = min(qty, available.get(rtype, 0))
                if allowed <= 0:
                    continue
                available[rtype] -= allowed
                grants[agent].setdefault(recipient, {})
                grants[agent][recipient][rtype] = grants[agent][recipient].get(rtype, 0) + allowed
    return grants


class ResourceSimulationEngine:
    """Resource-based simulation engine with next-turn grants."""

    def __init__(self, *, run_label: str = "resource_simulation") -> None:
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
        self.total_turns: int | None = None
        self.territory_graph: Dict[str, Set[str]] = {}
        self.territory_positions: Dict[str, Tuple[int, int]] = {}
        self.territory_names: List[str] = []
        self.territory_resources: Dict[str, Dict[str, int]] = {}
        self.pending_resource_grants: Dict[str, Dict[str, int]] = {}
        self.pending_money_grants: Dict[str, int] = {}
        self.per_turn_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.turns_seen: List[int] = []
        self.metric_keys: List[str] = []
        self.per_turn_territory_owners: List[List[str | None]] = []
        self.capital_territories: Dict[str, str] = {}
        self.per_turn_messages: List[Dict[str, Dict[str, str]]] = []
        self.per_turn_reports: List[Dict[str, str]] = []

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
        territory_seed: int | None = 42,
        resource_seed: int | None = 42,
        use_generated_territories: bool = False,
        resource_peaks: Mapping[str, int] | None = None,
        resource_peak_max: int = 3,
        resource_adjacent_pct: int = 50,
        resource_one_pct: int = 50,
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

        from .territory_graph import build_territory_graph

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

        self.territory_resources = generate_territory_resources(
            self.territory_names,
            self.territory_graph,
            peaks_per_resource=resource_peaks,
            max_value=resource_peak_max,
            resource_adjacent_pct=resource_adjacent_pct,
            resource_one_pct=resource_one_pct,
            seed=resource_seed,
        )
        self.pending_resource_grants = {a: {k: 0 for k in RESOURCE_TYPES} for a in self.agent_names}
        self.pending_money_grants = {a: 0 for a in self.agent_names}
        self.per_turn_metrics = {}
        self.turns_seen = []
        self.per_turn_territory_owners = []
        self.per_turn_messages = []
        self.per_turn_reports = []

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
        before_territories = {a: set(t) for a, t in agent_territories.items()}
        money_grants_received = dict(self.pending_money_grants)
        resource_grants_received = {
            a: dict(self.pending_resource_grants.get(a, {})) for a in self.agent_names
        }

        legal_inbound, legal_outbound = compute_legal_cession_lists(
            agent_territories,
            self.territory_graph,
            capitals=self.capital_territories,
        )
        resource_totals = _resource_totals(
            agent_territories, self.territory_resources, self.pending_resource_grants
        )
        ratios = _resource_ratios(resource_totals, agent_territories, constants)
        mult = _resource_multiplier(ratios)

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
            history_contexts={},
            legal_inbound_cessions=legal_inbound,
            legal_outbound_cessions=legal_outbound,
        )
        for agent in inputs.keys():
            inputs[agent]["resource_totals"] = resource_totals.get(agent, {})
            inputs[agent]["resource_ratios"] = ratios.get(agent, {})
            inputs[agent]["resource_minimums"] = {
                "energy": constants["c_min_energy"],
                "minerals": constants["c_min_minerals"],
                "food": constants["c_min_food"],
            }

        actions = call_agents_collect_actions(agents=agents, agent_inputs=inputs)

        from .phase0 import translate_agent_actions_to_intentions

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

        self._log_messages(d_messages_sent)

        d_resource_grants = _parse_resource_grants(
            actions,
            known_agents=set(agents.keys()),
            territory_resources=self.territory_resources,
            agent_territories=agent_territories,
            log_fn=self.log,
        )

        # Resource-based income
        d_gross_income: Dict[str, int] = {}
        d_effective_income: Dict[str, int] = {}
        for agent, territories in agent_territories.items():
            gross = len(territories) * int(constants["c_money_per_territory"])
            d_gross_income[agent] = gross
            d_effective_income[agent] = int(gross * mult.get(agent, 0.0))

        original_attacks = {k: dict(v) for k, v in d_global_attacks.items()}
        d_global_attacks = _clamp_attacks_to_mils(
            d_global_attacks, agent_mils, log_fn=self.log
        )

        d_global_attacked = _transpose_attacks(d_global_attacks)

        d_attacking_mils: Dict[str, int] = {}
        for agent, targets in d_global_attacks.items():
            d_attacking_mils[agent] = sum(targets.values())

        d_total_damage_received: Dict[str, int] = {}
        for target, attackers in d_global_attacked.items():
            total_attack = sum(attackers.values())
            d_total_damage_received[target] = total_attack * int(constants["c_damage_per_attack_mil"])

        for agent in d_total_damage_received:
            d_total_damage_received[agent] = min(
                d_total_damage_received[agent], d_effective_income.get(agent, 0)
            )

        d_upkeep_cost: Dict[str, int] = {}
        for agent, mils in agent_mils.items():
            d_upkeep_cost[agent] = mils * int(constants["c_mil_upkeep_price"])

        d_available_money: Dict[str, int] = {}
        for agent in agent_territories:
            d_available_money[agent] = max(
                d_effective_income.get(agent, 0) - d_total_damage_received.get(agent, 0),
                0,
            )

        # Purchases and upkeep
        d_mil_purchased: Dict[str, int] = {}
        for agent, desired in d_mil_purchase_intent.items():
            price = int(constants["c_mil_purchase_price"])
            max_affordable = d_available_money.get(agent, 0) // price if price > 0 else 0
            purchase = min(desired, max_affordable)
            d_mil_purchased[agent] = purchase
            d_available_money[agent] -= purchase * price

        d_mils_disbanded_upkeep: Dict[str, int] = {}
        for agent, upkeep in d_upkeep_cost.items():
            if d_available_money.get(agent, 0) >= upkeep:
                d_available_money[agent] -= upkeep
                d_mils_disbanded_upkeep[agent] = 0
            else:
                shortfall = upkeep - d_available_money.get(agent, 0)
                d_available_money[agent] = 0
                cost = int(constants["c_mil_upkeep_price"])
                disband = (shortfall + cost - 1) // cost if cost > 0 else 0
                d_mils_disbanded_upkeep[agent] = disband

        d_mils_lost_by_attacker: Dict[str, int] = {}
        for defender, attackers in d_global_attacked.items():
            total_losses = sum(attackers.values()) // int(constants["c_defense_destroy_factor"])
            losses = _allocate_losses_proportional(attackers, total_losses)
            for attacker, loss in losses.items():
                d_mils_lost_by_attacker[attacker] = d_mils_lost_by_attacker.get(attacker, 0) + loss

        # Apply losses and purchases
        for agent, lost in d_mils_disbanded_upkeep.items():
            agent_mils[agent] = max(agent_mils.get(agent, 0) - lost, 0)
        for agent, lost in d_mils_lost_by_attacker.items():
            agent_mils[agent] = max(agent_mils.get(agent, 0) - lost, 0)
        for agent, purchased in d_mil_purchased.items():
            agent_mils[agent] = agent_mils.get(agent, 0) + purchased

        # Voluntary disband
        for agent, requested in d_mils_disband_intent.items():
            current = agent_mils.get(agent, 0)
            disband = min(requested, current)
            agent_mils[agent] = current - disband

        # Territory cessions with adjacency enforcement
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
                        continue
                    agent_territories[giver].remove(tid)
                    agent_territories.setdefault(receiver, set()).add(tid)

        ceded_territories: Dict[str, List[str]] = {}
        received_territories: Dict[str, List[str]] = {}
        for agent in self.agent_names:
            before = before_territories.get(agent, set())
            after = agent_territories.get(agent, set())
            ceded = sorted(before - after)
            received = sorted(after - before)
            if ceded:
                ceded_territories[agent] = ceded
            if received:
                received_territories[agent] = received

        # Money grants affect next turn welfare
        next_pending_money = {a: 0 for a in self.agent_names}
        money_grants_sent = {a: 0 for a in self.agent_names}
        for giver, grants in d_money_grants.items():
            available = d_available_money.get(giver, 0)
            for receiver, amount in grants.items():
                if amount <= 0 or available <= 0:
                    continue
                paid = min(amount, available)
                available -= paid
                next_pending_money[receiver] = next_pending_money.get(receiver, 0) + paid
                money_grants_sent[giver] = money_grants_sent.get(giver, 0) + paid
            d_available_money[giver] = available

        # Resource grants affect next turn resources
        next_pending_resources = {a: {k: 0 for k in RESOURCE_TYPES} for a in self.agent_names}
        resource_grants_sent = {a: {k: 0 for k in RESOURCE_TYPES} for a in self.agent_names}
        for giver, grants in d_resource_grants.items():
            for receiver, res_map in grants.items():
                for rtype, qty in res_map.items():
                    next_pending_resources[receiver][rtype] += qty
                    resource_grants_sent[giver][rtype] = (
                        resource_grants_sent[giver].get(rtype, 0) + qty
                    )

        d_total_welfare_this_turn: Dict[str, int] = {}
        for agent in agent_territories:
            grants_received = self.pending_money_grants.get(agent, 0)
            d_total_welfare_this_turn[agent] = int(
                d_available_money.get(agent, 0)
                + grants_received * float(constants["c_trade_factor"])
            )
            agent_welfare[agent] += d_total_welfare_this_turn[agent]

        self.pending_money_grants = next_pending_money
        self.pending_resource_grants = next_pending_resources

        legal_inbound_after, legal_outbound_after = compute_legal_cession_lists(
            agent_territories,
            self.territory_graph,
            capitals=self.capital_territories,
        )
        self.last_agent_reports = self._build_agent_reports(
            d_gross_income=d_gross_income,
            d_effective_income=d_effective_income,
            d_resource_totals=resource_totals,
            d_resource_ratios=ratios,
            d_total_damage_received=d_total_damage_received,
            d_upkeep_cost=d_upkeep_cost,
            d_mil_purchased=d_mil_purchased,
            d_mils_disbanded_upkeep=d_mils_disbanded_upkeep,
            d_mils_lost_by_attacker=d_mils_lost_by_attacker,
            d_total_welfare_this_turn=d_total_welfare_this_turn,
            d_available_money=d_available_money,
            pending_money_grants=self.pending_money_grants,
            trade_factor=float(constants["c_trade_factor"]),
            d_reasoning=d_reasoning,
            d_keeps_word_report=d_keeps_word_report,
            d_aggressor_report=d_aggressor_report,
            legal_inbound_cessions=legal_inbound_after,
            legal_outbound_cessions=legal_outbound_after,
            constants=constants,
            money_grants_received=money_grants_received,
            money_grants_sent=money_grants_sent,
            resource_grants_received=resource_grants_received,
            resource_grants_sent=resource_grants_sent,
            ceded_territories=ceded_territories,
            received_territories=received_territories,
        )

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
            money_grants_sent=money_grants_sent,
            money_grants_received=money_grants_received,
            resource_grants_sent=resource_grants_sent,
            resource_grants_received=resource_grants_received,
            trade_factor=float(constants["c_trade_factor"]),
            agent_territories=agent_territories,
            d_gross_income=d_gross_income,
            d_effective_income=d_effective_income,
            d_total_damage_received=d_total_damage_received,
            d_available_money=d_available_money,
            d_upkeep_cost=d_upkeep_cost,
            d_mil_purchased=d_mil_purchased,
            d_resource_totals=resource_totals,
            d_resource_ratios=ratios,
            ceded_territories=ceded_territories,
            received_territories=received_territories,
        )
        self.per_turn_messages.append(d_messages_sent)
        self.per_turn_reports.append(dict(self.last_agent_reports))
        self._write_round_data()

        return {
            "d_total_welfare_this_turn": d_total_welfare_this_turn,
            "agent_welfare": dict(agent_welfare),
            "agent_territories": {k: sorted(list(v)) for k, v in agent_territories.items()},
            "d_summary_last_turn": d_summary_last_turn,
        }

    def _build_agent_reports(
        self,
        *,
        d_gross_income: Dict[str, int],
        d_effective_income: Dict[str, int],
        d_resource_totals: Dict[str, Dict[str, int]],
        d_resource_ratios: Dict[str, Dict[str, float]],
        d_total_damage_received: Dict[str, int],
        d_upkeep_cost: Dict[str, int],
        d_mil_purchased: Dict[str, int],
        d_mils_disbanded_upkeep: Dict[str, int],
        d_mils_lost_by_attacker: Dict[str, int],
        d_total_welfare_this_turn: Dict[str, int],
        d_available_money: Dict[str, int],
        pending_money_grants: Dict[str, int],
        trade_factor: float,
        d_reasoning: Dict[str, str],
        d_keeps_word_report: Dict[str, Dict[str, int]],
        d_aggressor_report: Dict[str, Dict[str, int]],
        legal_inbound_cessions: Dict[str, Dict[str, List[str]]],
        legal_outbound_cessions: Dict[str, Dict[str, List[str]]],
        constants: Mapping[str, Any],
        money_grants_received: Dict[str, int],
        money_grants_sent: Dict[str, int],
        resource_grants_received: Dict[str, Dict[str, int]],
        resource_grants_sent: Dict[str, Dict[str, int]],
        ceded_territories: Dict[str, List[str]],
        received_territories: Dict[str, List[str]],
    ) -> Dict[str, str]:
        reports: Dict[str, str] = {}
        for agent in sorted(d_gross_income.keys()):
            gross = d_gross_income.get(agent, 0)
            effective = d_effective_income.get(agent, 0)
            damage = d_total_damage_received.get(agent, 0)
            upkeep = d_upkeep_cost.get(agent, 0)
            purchased = d_mil_purchased.get(agent, 0)
            lost = d_mils_lost_by_attacker.get(agent, 0)
            disbanded = d_mils_disbanded_upkeep.get(agent, 0)
            welfare_this = d_total_welfare_this_turn.get(agent, 0)
            available = d_available_money.get(agent, 0)
            grants_received = pending_money_grants.get(agent, 0)
            res_totals = d_resource_totals.get(agent, {})
            res_ratios = d_resource_ratios.get(agent, {})
            reasoning = d_reasoning.get(agent, "")
            keeps_word = d_keeps_word_report.get(agent, {})
            aggressor = d_aggressor_report.get(agent, {})
            legal_inbound = legal_inbound_cessions.get(agent, {})
            legal_outbound = legal_outbound_cessions.get(agent, {})
            money_received = money_grants_received.get(agent, 0)
            money_sent = money_grants_sent.get(agent, 0)
            res_received = resource_grants_received.get(agent, {})
            res_sent = resource_grants_sent.get(agent, {})
            terrs_received = received_territories.get(agent, [])
            terrs_ceded = ceded_territories.get(agent, [])

            terr_count = max(len(self.agent_territories.get(agent, set())), 1)
            min_energy = float(constants.get("c_min_energy", 1))
            min_minerals = float(constants.get("c_min_minerals", 1))
            min_food = float(constants.get("c_min_food", 1))
            energy_total = res_totals.get("energy", 0)
            minerals_total = res_totals.get("minerals", 0)
            food_total = res_totals.get("food", 0)
            ratios_fmt = {
                "energy": _fmt_float(float(res_ratios.get("energy", 0))),
                "minerals": _fmt_float(float(res_ratios.get("minerals", 0))),
                "food": _fmt_float(float(res_ratios.get("food", 0))),
            }

            lines = [
                f"Income: gross={gross}, effective={effective}, damage={damage}",
                (
                    "Resource ratio details: "
                    f"energy_ratio=min(1, {energy_total}/({terr_count}*{min_energy}))="
                    f"{ratios_fmt['energy']}; "
                    f"minerals_ratio=min(1, {minerals_total}/({terr_count}*{min_minerals}))="
                    f"{ratios_fmt['minerals']}; "
                    f"food_ratio=min(1, {food_total}/({terr_count}*{min_food}))="
                    f"{ratios_fmt['food']}"
                ),
                (
                    "Effective income formula: "
                    f"gross({gross}) * energy_ratio({ratios_fmt['energy']}) * "
                    f"minerals_ratio({ratios_fmt['minerals']}) * "
                    f"food_ratio({ratios_fmt['food']}) = {effective}"
                ),
                f"Resources: totals={res_totals}, ratios={ratios_fmt}",
                "Note: resource grants do not cost money; they only transfer resources.",
                f"Costs: upkeep={upkeep}, purchases={purchased}",
                f"Army: lost={lost}, disbanded={disbanded}",
                f"Grants: money_received={money_received}, money_sent={money_sent}, resources_received={res_received}, resources_sent={res_sent}",
                f"Cessions: received={terrs_received}, ceded={terrs_ceded}",
                f"Reasoning (last turn): {reasoning}",
                f"Keeps word report: {keeps_word}",
                f"Aggressor report: {aggressor}",
                f"Legal inbound cessions: {legal_inbound}",
                f"Legal outbound cessions: {legal_outbound}",
                "Welfare: this_turn={w} = available_money({a}) + grants_received({g})*trade_factor({tf})".format(
                    w=welfare_this,
                    a=available,
                    g=grants_received,
                    tf=_fmt_float(float(trade_factor)),
                ),
            ]
            reports[agent] = "\n".join(lines)
        return reports

    def _log_messages(self, d_messages_sent: Dict[str, Dict[str, str]]) -> None:
        if not d_messages_sent:
            self.log("Messages sent: none")
            return
        self.log("Messages sent:")
        for sender in sorted(d_messages_sent.keys()):
            msgs = d_messages_sent.get(sender, {})
            if not msgs:
                continue
            for recipient, message in msgs.items():
                self.log(f"  {sender} -> {recipient}: {message}")

    def _turns_left(self, turn: int) -> int | None:
        if self.total_turns is None:
            return None
        remaining = self.total_turns - (turn + 1)
        return max(remaining, 0)

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
        money_grants_sent: Dict[str, int],
        money_grants_received: Dict[str, int],
        resource_grants_sent: Dict[str, Dict[str, int]],
        resource_grants_received: Dict[str, Dict[str, int]],
        trade_factor: float,
        agent_territories: Dict[str, Set[str]],
        d_gross_income: Dict[str, int],
        d_effective_income: Dict[str, int],
        d_total_damage_received: Dict[str, int],
        d_available_money: Dict[str, int],
        d_upkeep_cost: Dict[str, int],
        d_mil_purchased: Dict[str, int],
        d_resource_totals: Dict[str, Dict[str, int]],
        d_resource_ratios: Dict[str, Dict[str, float]],
        ceded_territories: Dict[str, List[str]],
        received_territories: Dict[str, List[str]],
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
                "gross_income",
                "effective_income",
                "damage_received",
                "available_money",
                "upkeep_cost",
                "mils_purchased",
                "money_grants_received",
                "money_grants_sent",
                "resource_energy_total",
                "resource_minerals_total",
                "resource_food_total",
                "resource_energy_ratio",
                "resource_minerals_ratio",
                "resource_food_ratio",
                "resource_energy_grants_received",
                "resource_minerals_grants_received",
                "resource_food_grants_received",
                "resource_energy_grants_sent",
                "resource_minerals_grants_sent",
                "resource_food_grants_sent",
                "territories_received",
                "territories_ceded",
            ]
            self.metric_keys = list(keys)
            self.per_turn_metrics = {k: {a: [] for a in self.agent_names} for k in keys}

        self.turns_seen.append(turn)
        for agent in self.agent_names:
            attacks_received = sum(d_global_attacked.get(agent, {}).values())
            grants_received = money_grants_received.get(agent, 0)
            res_totals = d_resource_totals.get(agent, {})
            res_ratios = d_resource_ratios.get(agent, {})
            res_received = resource_grants_received.get(agent, {})
            res_sent = resource_grants_sent.get(agent, {})

            self.per_turn_metrics["total_welfare"][agent].append(agent_welfare.get(agent, 0))
            self.per_turn_metrics["welfare_this_turn"][agent].append(
                d_total_welfare_this_turn.get(agent, 0)
            )
            self.per_turn_metrics["attacks"][agent].append(d_attacking_mils.get(agent, 0))
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
            )
            self.per_turn_metrics["trade_sent"][agent].append(money_grants_sent.get(agent, 0))
            self.per_turn_metrics["trade_welfare_received"][agent].append(
                grants_received * trade_factor
            )
            self.per_turn_metrics["gross_income"][agent].append(d_gross_income.get(agent, 0))
            self.per_turn_metrics["effective_income"][agent].append(
                d_effective_income.get(agent, 0)
            )
            self.per_turn_metrics["damage_received"][agent].append(
                d_total_damage_received.get(agent, 0)
            )
            self.per_turn_metrics["available_money"][agent].append(
                d_available_money.get(agent, 0)
            )
            self.per_turn_metrics["upkeep_cost"][agent].append(d_upkeep_cost.get(agent, 0))
            self.per_turn_metrics["mils_purchased"][agent].append(
                d_mil_purchased.get(agent, 0)
            )
            self.per_turn_metrics["money_grants_received"][agent].append(grants_received)
            self.per_turn_metrics["money_grants_sent"][agent].append(
                money_grants_sent.get(agent, 0)
            )
            self.per_turn_metrics["resource_energy_total"][agent].append(
                res_totals.get("energy", 0)
            )
            self.per_turn_metrics["resource_minerals_total"][agent].append(
                res_totals.get("minerals", 0)
            )
            self.per_turn_metrics["resource_food_total"][agent].append(
                res_totals.get("food", 0)
            )
            self.per_turn_metrics["resource_energy_ratio"][agent].append(
                res_ratios.get("energy", 0)
            )
            self.per_turn_metrics["resource_minerals_ratio"][agent].append(
                res_ratios.get("minerals", 0)
            )
            self.per_turn_metrics["resource_food_ratio"][agent].append(
                res_ratios.get("food", 0)
            )
            self.per_turn_metrics["resource_energy_grants_received"][agent].append(
                res_received.get("energy", 0)
            )
            self.per_turn_metrics["resource_minerals_grants_received"][agent].append(
                res_received.get("minerals", 0)
            )
            self.per_turn_metrics["resource_food_grants_received"][agent].append(
                res_received.get("food", 0)
            )
            self.per_turn_metrics["resource_energy_grants_sent"][agent].append(
                res_sent.get("energy", 0)
            )
            self.per_turn_metrics["resource_minerals_grants_sent"][agent].append(
                res_sent.get("minerals", 0)
            )
            self.per_turn_metrics["resource_food_grants_sent"][agent].append(
                res_sent.get("food", 0)
            )
            self.per_turn_metrics["territories_received"][agent].append(
                len(received_territories.get(agent, []))
            )
            self.per_turn_metrics["territories_ceded"][agent].append(
                len(ceded_territories.get(agent, []))
            )

        owner_by_territory: Dict[str, str] = {}
        for agent, terrs in agent_territories.items():
            for terr in terrs:
                owner_by_territory[terr] = agent
        owners = [owner_by_territory.get(name) for name in self.territory_names]
        self.per_turn_territory_owners.append(owners)

    def _write_round_data(self) -> None:
        if not self.per_turn_metrics or not self.turns_seen:
            return
        ledger_vars = self.metric_keys or list(self.per_turn_metrics.keys())
        data: List[List[List[float]]] = []
        for agent in self.agent_names:
            agent_rows: List[List[float]] = []
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
            "territory_resources": self.territory_resources,
            "messages": self.per_turn_messages,
            "reports": self.per_turn_reports,
        }
        out_dir = Path("round_data")
        log_stem = Path(self.log_path).stem
        out_path = out_dir / f"{log_stem}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
