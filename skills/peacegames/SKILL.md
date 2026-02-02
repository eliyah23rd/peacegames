---
name: peacegames
description: Repo-specific workflow and runbook for Peacegames. Use for any request in /home/eli/dev/peacegames: running tests, simulations, experiments, analysis, world-builder/worldmap tasks, UI viewer tasks, outputs, or repo rules. Also use when maintaining TODO.md or updating this skill with new rules.
---

# Peacegames

## Overview

Work in `/home/eli/dev/peacegames` with the repo runbook. Keep `TODO.md` updated for every user request. Update this skill whenever new general rules or recurring constraints appear.

## Always-On Rules

- Update `TODO.md` at the start and end of each user request: add tasks, mark completed items, and append brief notes/insights.
- Treat this skill as the default for all Peacegames repo work (keep it front-of-mind).
- When a new rule is provided (by user or learned), update `skills/peacegames/SKILL.md` and note the change in `TODO.md`.
- Follow repo workflow requirements (tests before/after changes, commit + push when done).

## Quick Start

- Tests: `python run_all_tests.py`
- Base sim: `python -m run_llm_simulation initial_setup/four_agent_violent.json --model gpt-5.1`
- Resource sim: `python -m run_llm_resource_simulation initial_setup/resource_demo_4p.json --model gpt-5.1`
- Experiments: `python -m run_experiments initial_setup/four_agent_exp.json --model gpt-5.1`
- UI: `python ui_app/app.py` then open `http://127.0.0.1:8000`

## Core Workflows

- For detailed commands and the world-builder steps, read `skills/peacegames/references/commands.md`.
- Keep outputs organized: `logs/`, `round_data/`, and `visualizations/` are the default targets.
- Use `OPENAI_API_KEY` for real LLM runs or set `PEACEGAME_LLM_DUMMY=1` to run without network calls.

## Notes for Future Sessions

- Always create or update tasks in `TODO.md` as the first and last step of each request.
- If this skill needs additional resources (scripts, references), add them under `skills/peacegames/` and link from this file.
