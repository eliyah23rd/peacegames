# Peacegames Command Runbook

All commands assume you run from the repo root: `/home/eli/dev/peacegames`.

## Environment Setup

- Create venv: `python -m venv .venv`
- Activate: `source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- LLM key: `export OPENAI_API_KEY="..."`
- Offline runs: `export PEACEGAME_LLM_DUMMY=1`

## Tests

- Full suite (preferred): `python run_all_tests.py`
- Phase0 tests only: `python -m tests.run_phase0_tests`
- UI smoke test only: `python ui_app/smoke_test.py`

## Simulations

### Base sim (LLM agents)

```
python -m run_llm_simulation initial_setup/four_agent_violent.json --model gpt-5.1
```

Notes:
- You can pass prompt modifiers in agent order (comma-separated allowed).
- Outputs: `logs/` and `round_data/`.

### Resource sim (LLM agents)

```
python -m run_llm_resource_simulation initial_setup/resource_demo_4p.json --model gpt-5.1
```

Notes:
- Same modifier handling as base sim.
- Outputs: `logs/`, `round_data/`, and `visualizations/`.

### Experiments (multi-round)

```
python -m run_experiments initial_setup/four_agent_exp.json --model gpt-5.1
```

Notes:
- Uses `experiments/` configs and writes results to `experiments/`.
- Each round prints the exact command used for that round.

## Experiment Analysis

```
python analyze_experiment.py experiments/<experiment>.json
python analyze_experiment.py experiments/<experiment>.json --write-files
```

Notes:
- `--write-files` writes summary text to `experiments/` and plots to `visualizations/`.

## UI Viewer

- Start server: `python ui_app/app.py`
- Open in browser: `http://127.0.0.1:8000`
- UI reads files from `round_data/` and renders maps on demand.

## World-Builder / Worldmap

### Generate world outline

```
python worldmap/wm_gen.py
```

Notes:
- Requires `worldmap/ne_10m_land/ne_10m_land.shp`.
- Unzip `worldmap/ne_10m_land.zip` if the shapefile is missing.
- Output: `worldmap/world_outline_1600x800.png`.

### Generate territories + labels

```
python worldmap/draw_terrs_v2.py
```

Notes:
- Uses `seed_overrides.json`, `name_overrides.json`, and `label_overrides.json` if present.
- Output: `worldmap/world_map_32_*.png` and `worldmap/world_territories_32.json`.

### Interactive seed editor

```
python worldmap/seed_editor.py --mode move
python worldmap/seed_editor.py --mode name
python worldmap/seed_editor.py --mode label --label-style icons
```

Notes:
- Requires an interactive matplotlib backend (tkinter/PyQt).
- Saves edits to the override JSON files in `worldmap/`.

## Key Files and Folders

- `initial_setup/`: scenario configs for sims and experiments
- `experiments/`: experiment configs and outputs
- `logs/`, `round_data/`, `visualizations/`: generated outputs
- `worldmap/`: world-builder assets and scripts
