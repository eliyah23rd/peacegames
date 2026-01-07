from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from peacegame.llm_agent import (
    DummyLLMProvider,
    LLMDefaultAgent,
    OpenAIProvider,
    build_resource_prompt,
    build_system_prompt,
)
from peacegame.simulation import SimulationEngine
from peacegame.resource_simulation import ResourceSimulationEngine


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-round experiments")
    parser.add_argument("config", help="Path to experiment config JSON")
    return parser.parse_args(argv)


def _rank_agents(welfare: Dict[str, int]) -> List[tuple[str, int, int]]:
    ranked = sorted(welfare.items(), key=lambda x: (-x[1], x[0]))
    return [(agent, score, idx + 1) for idx, (agent, score) in enumerate(ranked)]


def _normalize_modifiers(mods: list[str]) -> list[str]:
    out = []
    for raw in mods:
        out.extend([m for m in raw.split(",") if m])
    return out


def _normalize_seed(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    seed = int(raw)
    if seed < 0:
        return None
    return seed


def _resolve_round_seed(base_seed: Any) -> Optional[int]:
    if base_seed is None:
        return None
    seed = int(base_seed)
    if seed < 0:
        import random

        return random.SystemRandom().randint(0, 2**31 - 1)
    return seed


def _format_round_command(
    *,
    setup_path: str,
    model: str,
    modifiers: List[str],
    round_idx: int,
    rounds: int,
    mode: str,
) -> str:
    mods = " ".join(modifiers)
    suffix = f" {mods}" if mods else ""
    runner = "run_llm_resource_simulation.py" if mode == "resources" else "run_llm_simulation.py"
    return (
        f"Round {round_idx + 1}/{rounds}: "
        f"python {runner} {setup_path} --model {model}{suffix}"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python run_experiments.py experiments/<config>.json")
        return 2

    args = _parse_args(sys.argv[1:])
    cfg = _load_json(args.config)

    setup_path = cfg["setup"]
    setup = _load_json(setup_path)
    rounds = int(cfg.get("rounds", 1))
    exp_name = cfg.get("name", "experiment")
    model = cfg.get("model", setup.get("model", "gpt-5-nano"))
    shuffle_mods = bool(cfg.get("shuffle_modifiers", True))
    mode = cfg.get("mode", "base")
    resource_deterministic = bool(cfg.get("resource_deterministic", False))

    constants = setup.get("constants", {})
    initial_state = setup.get("initial_state", {})
    num_turns = int(setup.get("num_turns", 1))

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
    seed_value = setup.get("seed")
    if seed_value is None:
        seed_value = setup.get("territory_seed")
    if seed_value is None:
        seed_value = setup.get("resource_seed")

    modifier_pool = _normalize_modifiers(cfg.get("modifiers", []))
    rounds_config = cfg.get("rounds_config", [])

    modifier_totals: Dict[str, Dict[str, int]] = {}
    round_results: List[Dict[str, Any]] = []

    for round_idx in range(rounds):
        if round_idx < len(rounds_config):
            mods_for_round = _normalize_modifiers(rounds_config[round_idx].get("modifiers", []))
        else:
            mods_for_round = list(modifier_pool)
            if shuffle_mods:
                import random

                if len(modifier_pool) >= len(agent_names):
                    mods_for_round = random.sample(modifier_pool, len(agent_names))
                else:
                    random.shuffle(mods_for_round)

        print(
            _format_round_command(
                setup_path=setup_path,
                model=model,
                modifiers=mods_for_round,
                round_idx=round_idx,
                rounds=rounds,
                mode=mode,
            )
        )

        agents = {}
        mod_map: Dict[str, str] = {}
        for idx, agent_name in enumerate(agent_names):
            mod = mods_for_round[idx] if idx < len(mods_for_round) else ""
            mod_map[agent_name] = mod
            if mode == "resources":
                system_prompt = build_resource_prompt([mod]) if mod else build_resource_prompt([])
            else:
                system_prompt = build_system_prompt([mod]) if mod else build_system_prompt([])
            agents[agent_name] = LLMDefaultAgent(
                agent_name, provider=provider, system_prompt=system_prompt
            )

        run_label = f"{exp_name}_r{round_idx + 1:02d}"
        if mode == "resources":
            engine = ResourceSimulationEngine(run_label=run_label)
        else:
            engine = SimulationEngine(run_label=run_label)
        engine.log_initial_state(
            script_name=run_label,
            agent_territories=agent_territories,
            agent_mils=agent_mils,
            agent_welfare=agent_welfare,
            constants=constants,
            prompt_modifiers=mod_map,
        )
        if mode == "resources":
            if seed_value is not None:
                resolved_seed = _resolve_round_seed(seed_value)
            else:
                resolved_seed = _normalize_seed(setup.get("resource_seed")) if resource_deterministic else None
            engine.setup_state(
                agent_territories=agent_territories,
                agent_mils=agent_mils,
                agent_welfare=agent_welfare,
                use_generated_territories=True,
                seed=resolved_seed,
                resource_peaks=setup.get("resource_peaks"),
                resource_peak_max=setup.get("resource_peak_max", 3),
                resource_adjacent_pct=setup.get("resource_adjacent_pct", 50),
                resource_one_pct=setup.get("resource_one_pct", 50),
            )
        else:
            engine.setup_state(
                agent_territories=agent_territories,
                agent_mils=agent_mils,
                agent_welfare=agent_welfare,
                use_generated_territories=True,
                seed=_resolve_round_seed(seed_value),
            )
        engine.setup_round(total_turns=num_turns)

        turn_summaries: Dict[str, str] = {a: "" for a in agent_names}
        results: Dict[str, Any] = {}
        for turn in range(num_turns):
            results = engine.run_turn(
                script_name=run_label,
                turn=turn,
                agents=agents,
                constants=constants,
                turn_summaries=turn_summaries,
                max_summary_len=512,
            )
            turn_summaries = results.get("d_summary_last_turn", turn_summaries)

        engine.log_script_end(script_name=run_label)
        engine.close()

        welfare = results.get("agent_welfare", {})
        ranked = _rank_agents(welfare)
        round_results.append(
            {
                "round": round_idx + 1,
                "modifiers": mod_map,
                "welfare": welfare,
                "ranking": ranked,
                "log_path": engine.log_path,
            }
        )

        for agent, score, rank in ranked:
            mod = mod_map.get(agent, "none") or "none"
            totals = modifier_totals.setdefault(mod, {"total_welfare": 0, "rounds": 0, "wins": 0})
            totals["total_welfare"] += int(score)
            totals["rounds"] += 1
            if rank == 1:
                totals["wins"] += 1

    summary = {
        "name": exp_name,
        "setup": setup_path,
        "rounds": rounds,
        "num_turns": num_turns,
        "results": round_results,
        "modifier_summary": {},
    }
    for mod, stats in modifier_totals.items():
        rounds_used = max(stats["rounds"], 1)
        summary["modifier_summary"][mod] = {
            "total_welfare": stats["total_welfare"],
            "average_welfare": stats["total_welfare"] / rounds_used,
            "wins": stats["wins"],
            "rounds": stats["rounds"],
        }

    out_dir = "experiments"
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{exp_name}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Experiment summary written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
