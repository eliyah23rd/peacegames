from __future__ import annotations

import datetime as dt
import os
from typing import Any, Dict, List, Mapping, Set

from .phase0 import assemble_agent_inputs, call_agents_collect_actions, translate_agent_actions_to_intentions


class Phase0Engine:
    """Phase 0 engine wrapper that records logs."""

    def __init__(self, *, run_label: str = "phase0") -> None:
        self.run_label = run_label
        self._ensure_dirs()
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("logs", f"{run_label}_{run_id}.log")
        self._log_fp = open(self.log_path, "w", encoding="utf-8")

    def close(self) -> None:
        self._log_fp.close()

    def log(self, msg: str) -> None:
        self._log_fp.write(msg + "\n")
        self._log_fp.flush()

    def run_turn(
        self,
        *,
        script_name: str,
        turn: int,
        agents: Mapping[str, Any],
        agent_territories: Mapping[str, Set[str]],
        agent_mils: Mapping[str, int],
        constants: Mapping[str, Any],
        turn_summaries: Mapping[str, str],
        max_summary_len: int = 2048,
    ) -> tuple[
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
            max_summary_len=max_summary_len,
            log_fn=self.log,
        )

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

    def log_initial_state(
        self,
        *,
        script_name: str,
        agent_territories: Mapping[str, Set[str]],
        agent_mils: Mapping[str, int],
        constants: Mapping[str, Any],
    ) -> None:
        self.log(f"Script {script_name} start")
        self.log(
            "Initial territories: "
            + str(sorted((a, sorted(list(t))) for a, t in agent_territories.items()))
        )
        self.log(f"Initial mils: {sorted(agent_mils.items())}")
        self.log(f"Constants: {constants}")

    def log_script_end(self, *, script_name: str) -> None:
        self.log(f"Script {script_name} end")

    def _ensure_dirs(self) -> None:
        os.makedirs("logs", exist_ok=True)
