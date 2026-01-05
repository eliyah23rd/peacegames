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


def generate_territory_resources(
    territory_names: List[str],
    *,
    resource_richness: float = 0.4,
    seed: int | None = None,
) -> Dict[str, Dict[str, int]]:
    rng = random.Random(seed)
    richness = max(0.0, min(1.0, resource_richness))
    p_one = max(0.2, min(0.8, 0.8 - 0.6 * richness))

    resources: Dict[str, Dict[str, int]] = {}
    for terr in territory_names:
        k = 1 if rng.random() < p_one else 2
        types = rng.sample(list(RESOURCE_TYPES), k)
        if richness < 0.5:
            weights = [0.6, 0.3, 0.1]
        else:
            weights = [0.3, 0.4, 0.3]
        quantities = [1, 2, 3]
        terr_res: Dict[str, int] = {}
        for t in types:
            terr_res[t] = rng.choices(quantities, weights=weights, k=1)[0]
        resources[terr] = terr_res
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
            self.log(f"  {key}: {constants[key]}")
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
        resource_richness: float = 0.4,
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
            assigned = assign_territories_round_robin(
                self.agent_names,
                self.territory_graph,
                self.territory_positions,
                seed=territory_seed,
            )
            self.agent_territories = {k: set(v) for k, v in assigned.items()}

        self.territory_resources = generate_territory_resources(
            self.territory_names,
            resource_richness=resource_richness,
            seed=resource_seed,
        )
        self.pending_resource_grants = {a: {k: 0 for k in RESOURCE_TYPES} for a in self.agent_names}
        self.pending_money_grants = {a: 0 for a in self.agent_names}

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

        legal_inbound, legal_outbound = compute_legal_cession_lists(
            agent_territories,
            self.territory_graph,
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
                    ):
                        continue
                    agent_territories[giver].remove(tid)
                    agent_territories.setdefault(receiver, set()).add(tid)

        # Money grants affect next turn welfare
        next_pending_money = {a: 0 for a in self.agent_names}
        for giver, grants in d_money_grants.items():
            available = d_available_money.get(giver, 0)
            for receiver, amount in grants.items():
                if amount <= 0 or available <= 0:
                    continue
                paid = min(amount, available)
                available -= paid
                next_pending_money[receiver] = next_pending_money.get(receiver, 0) + paid
            d_available_money[giver] = available

        # Resource grants affect next turn resources
        next_pending_resources = {a: {k: 0 for k in RESOURCE_TYPES} for a in self.agent_names}
        for giver, grants in d_resource_grants.items():
            for receiver, res_map in grants.items():
                for rtype, qty in res_map.items():
                    next_pending_resources[receiver][rtype] += qty

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
        )

        self.log("Previous turn agent reports:")
        for agent in sorted(self.last_agent_reports.keys()):
            self.log(f"[{agent}]")
            self.log(self.last_agent_reports[agent])

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

            terr_count = max(len(self.agent_territories.get(agent, set())), 1)
            min_energy = float(constants.get("c_min_energy", 1))
            min_minerals = float(constants.get("c_min_minerals", 1))
            min_food = float(constants.get("c_min_food", 1))
            energy_total = res_totals.get("energy", 0)
            minerals_total = res_totals.get("minerals", 0)
            food_total = res_totals.get("food", 0)

            lines = [
                f"Income: gross={gross}, effective={effective}, damage={damage}",
                "Effective income formula: gross * energy_ratio * minerals_ratio * food_ratio",
                (
                    "Resource ratio details: "
                    f"energy_ratio=min(1, {energy_total}/({terr_count}*{min_energy}))="
                    f"{res_ratios.get('energy', 0)}; "
                    f"minerals_ratio=min(1, {minerals_total}/({terr_count}*{min_minerals}))="
                    f"{res_ratios.get('minerals', 0)}; "
                    f"food_ratio=min(1, {food_total}/({terr_count}*{min_food}))="
                    f"{res_ratios.get('food', 0)}"
                ),
                f"Resources: totals={res_totals}, ratios={res_ratios}",
                f"Costs: upkeep={upkeep}, purchases={purchased}",
                f"Army: lost={lost}, disbanded={disbanded}",
                f"Reasoning (last turn): {reasoning}",
                f"Keeps word report: {keeps_word}",
                f"Aggressor report: {aggressor}",
                f"Legal inbound cessions: {legal_inbound}",
                f"Legal outbound cessions: {legal_outbound}",
                f"Welfare: this_turn={welfare_this} = available_money({available}) + grants_received({grants_received})*trade_factor({trade_factor})",
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
