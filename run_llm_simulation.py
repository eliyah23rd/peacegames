from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from peacegame.llm_agent import (
    DummyLLMProvider,
    LLMDefaultAgent,
    OpenAIProvider,
    build_system_prompt,
)
from peacegame.simulation import SimulationEngine


def _load_setup(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python run_llm_simulation.py initial_setup/<file>.json [modifier1 modifier2 ...]"
        )
        return 2

    setup_path = sys.argv[1]
    setup = _load_setup(setup_path)

    constants = setup.get("constants", {})
    initial_state = setup.get("initial_state", {})
    num_turns = int(setup.get("num_turns", 1))
    model = setup.get("model", "gpt-5-nano")
    name = setup.get("name", "llm_simulation")

    if os.environ.get("PEACEGAME_LLM_DUMMY", ""):
        provider = DummyLLMProvider()
    else:
        provider = OpenAIProvider(model=model)

    agent_territories = {
        agent: set(territories)
        for agent, territories in initial_state.get("agent_territories", {}).items()
    }
    agent_mils = {
        agent: int(mils) for agent, mils in initial_state.get("agent_mils", {}).items()
    }
    agent_welfare = {
        agent: int(welfare)
        for agent, welfare in initial_state.get("agent_welfare", {}).items()
    }

    agent_names = sorted(set(agent_territories) | set(agent_mils) | set(agent_welfare))
    modifiers = []
    for raw in sys.argv[2:]:
        modifiers.extend([m for m in raw.split(",") if m])
    agents = {}
    for idx, agent_name in enumerate(agent_names):
        mod = modifiers[idx] if idx < len(modifiers) else ""
        system_prompt = build_system_prompt([mod]) if mod else build_system_prompt([])
        agents[agent_name] = LLMDefaultAgent(
            agent_name, provider=provider, system_prompt=system_prompt
        )

    engine = SimulationEngine(run_label=name)
    engine.log_initial_state(
        script_name=name,
        agent_territories=agent_territories,
        agent_mils=agent_mils,
        agent_welfare=agent_welfare,
        constants=constants,
    )
    engine.setup_state(
        agent_territories=agent_territories,
        agent_mils=agent_mils,
        agent_welfare=agent_welfare,
    )
    engine.setup_round(total_turns=num_turns)

    turn_summaries: Dict[str, str] = {a: "" for a in agent_names}
    for turn in range(num_turns):
        results = engine.run_turn(
            script_name=name,
            turn=turn,
            agents=agents,
            constants=constants,
            turn_summaries=turn_summaries,
            max_summary_len=256,
        )
        turn_summaries = results.get("d_turn_summary", turn_summaries)

    engine.log_script_end(script_name=name)
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
