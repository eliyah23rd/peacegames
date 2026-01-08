# Peacegames

Multi-agent simulation sandbox for exploring cooperation, conflict, and economic trade-offs. This project runs turn-based "rounds" where agents (scripted or LLM-driven) make decisions about military, territory, messaging, and (in the resource mode) multi-resource trade. The core goal is to compare strategies and understand which behaviors maximize welfare over time.

## Quick Start

- Tests: `python run_all_tests.py`
- UI server: `python ui_app/app.py`
- UI in browser: `http://127.0.0.1:8000`

## Concepts

The simulation engine models a set of agents competing or cooperating across a shared map of territories. Each round progresses through phases: agents decide on actions, the engine resolves conflicts and trades, and logs/round data are written for analysis. There are two main modes:

- **Base sim**: money, military, territory, and messaging.
- **Resource sim**: adds energy/minerals/food constraints and trade, affecting effective income.

The project is designed to support reproducible debugging (deterministic seeds) as well as randomized experimentation.

## Running Tests

Use the single wrapper that runs unit, component, and smoke tests:

```bash
python run_all_tests.py
```

## Running Simulations

### Base simulation (LLM agents)

```bash
python -m run_llm_simulation initial_setup/four_agent_violent.json --model gpt-5.1 diplomacy,opportunist,austerity,aggressive
```

### Resource simulation (LLM agents)

```bash
python -m run_llm_resource_simulation initial_setup/resource_demo_4p.json --model gpt-5.1 diplomacy,opportunist,austerity,aggressive
```

### Experiments (multi-round)

```bash
python -m run_experiments initial_setup/four_agent_exp.json --model gpt-5.1
```

Simulation outputs:
- Logs in `logs/`
- Round data in `round_data/` (JSON used by the UI)
- Visuals in `visualizations/`

## UI Viewer

Start the server:

```bash
python ui_app/app.py
```

Then open:
```
http://127.0.0.1:8000
```

The UI lets you explore timelines, per-turn data, map view, global news, individual reports, messages, and experiments. It reads from `round_data/` and renders maps on demand.

## Repository Layout

- `peacegame/`: core simulation engines and utilities
- `initial_setup/`: config files for runs and experiments
- `tests/`: unit and component tests
- `ui_app/`: local UI server + frontend
- `logs/`, `round_data/`, `visualizations/`: generated outputs

