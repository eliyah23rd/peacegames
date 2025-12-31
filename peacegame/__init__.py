"""Peacegame simulation package (phase 0 + test utilities)."""

from .agents import DefaultAgent, ScriptedAgent
from .engine import Phase0Engine
from .agent_runner import run_agent_actions
from .llm_agent import DEFAULT_AGENT_PROMPT, DummyLLMProvider, LLMDefaultAgent, OpenAIProvider
from .phase0 import (
    assemble_agent_inputs,
    call_agents_collect_actions,
    run_phase0,
    translate_agent_actions_to_intentions,
)

__all__ = [
    "DefaultAgent",
    "DEFAULT_AGENT_PROMPT",
    "DummyLLMProvider",
    "LLMDefaultAgent",
    "OpenAIProvider",
    "Phase0Engine",
    "run_agent_actions",
    "SimulationEngine",
    "ScriptedAgent",
    "assemble_agent_inputs",
    "call_agents_collect_actions",
    "run_phase0",
    "translate_agent_actions_to_intentions",
]


def __getattr__(name: str):
    if name == "SimulationEngine":
        from .simulation import SimulationEngine

        return SimulationEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
