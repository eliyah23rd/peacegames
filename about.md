# Peacegames - About

Peacegames is a research and prototyping project for building turn-based,
agent-driven strategy simulations that mix economic development, diplomacy,
and military conflict. The long-term goal is to evaluate which policies
and negotiation strategies maximize welfare over time, and to support
experiments that compare peaceful cooperation against defection and aggression.

This repo contains:

- A simulation engine with deterministic phase processing and strict
  validation of agent JSON actions.
- Multiple phases of gameplay including income, trade/grants, attacks,
  upkeep, disbanding, and welfare scoring.
- A deterministic test harness with scripted agents to prevent regressions.
- LLM-based agents with structured prompts, summaries, and reporting.
- Experiment runners to execute multi-round simulations and compare
  strategy modifiers statistically.
- Visualization tooling: turn-by-turn graphs and a UI for browsing round data,
  global news, reports, messages, and maps.
- A territory graph system for adjacency, legal cessions, and capital rules.
- A world map workflow for creating human-friendly territory layouts with
  editable seeds, labels, and overlays.

At a high level, the simulation aims to:

- Encourage agents to make choices under scarcity and uncertainty.
- Preserve a consistent game state across turns.
- Provide interpretable logs, reports, and data exports for analysis.
- Support both deterministic debugging runs and randomized experiments.

Key files and folders:

- `peacegame/`: core engine and agent implementations.
- `tests/` and `test_scripts/`: unit/component tests and scripted scenarios.
- `initial_setup/`: simulation configuration files.
- `ui_app/`: web UI for browsing simulation output.
- `worldmap/`: tools for generating and editing world-style territory maps.
- `experiments/`: experiment configurations and results.

This document is a concise overview; see `README.md` for how to run tests,
simulations, and the UI.
