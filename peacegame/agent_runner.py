from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Mapping


def _safe_call(agent: Any, agent_input: dict) -> Any:
    try:
        return agent.act(agent_input)
    except Exception:
        return ""


def run_agent_actions(
    *,
    agents: Mapping[str, Any],
    agent_inputs: Mapping[str, dict],
    max_workers: int | None = None,
) -> Dict[str, Any]:
    """Run agent.act calls in parallel and return raw action payloads."""
    actions: Dict[str, Any] = {}
    if not agents:
        return actions

    print(f"Running agent actions in parallel for {len(agents)} agents...")
    worker_count = max_workers or min(32, max(len(agents), 1))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_map = {
            pool.submit(_safe_call, agent, agent_inputs[name]): name
            for name, agent in agents.items()
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                actions[name] = future.result()
            except Exception:
                actions[name] = ""

    print("Agent actions complete.")
    return actions
