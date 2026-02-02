# Peacegames TODO

## To Do
- [ ] Add any additional repo workflows if discovered in future requests.

## Done
- [x] Ran `python run_all_tests.py` before changes (2026-02-02).
- [x] Drafted repo skill at `skills/peacegames/SKILL.md`.
- [x] Added command/workflow runbook in `skills/peacegames/references/commands.md`.
- [x] Added tests for skill/todo presence in `tests/test_skill_files.py`.
- [x] Re-ran `python run_all_tests.py` after changes (2026-02-02).
- [x] Re-ran `python run_all_tests.py` after TODO updates (2026-02-02).
- [x] Committed and pushed changes (including latest TODO update).
- [x] Ran `python run_all_tests.py` before updating TODO for sim-viewer reminder (2026-02-02).
- [x] Re-ran `python run_all_tests.py` after TODO update (2026-02-02).
- [x] Ran `python run_all_tests.py` before updating skill file for test policy (2026-02-02).
- [x] Updated `skills/peacegames/SKILL.md` with conditional test-running rule.
- [x] Re-ran `python run_all_tests.py` after skill update (2026-02-02).
- [x] Ran `python run_all_tests.py` before updating TODO for skills/todo location question (2026-02-02).
- [x] Re-ran `python run_all_tests.py` after TODO update (2026-02-02).
- [x] Ran `python run_all_tests.py` before responding to map overlay/name mismatch question (2026-02-02).
- [x] Ran `python run_all_tests.py` before editing map generator for name overrides (2026-02-02).
- [x] Updated map JSON generation to use `name_overrides.json` for UI labels.
- [x] Applied permanent fix and regenerated map outputs.
- [x] Re-ran `python run_all_tests.py` after map generator update (2026-02-02).
- [x] Ran `python run_all_tests.py` before updating TODO for fixed-map usage question (2026-02-02).
- [x] Re-ran `python run_all_tests.py` after TODO update (2026-02-02).
- [x] Confirmed sim run used fixed-map territories from `worldmap/world_territories_32.json`.
- [x] Ran `python run_all_tests.py` before updating TODO for sim map usage question (2026-02-02).
- [x] Re-ran `python run_all_tests.py` after TODO update (2026-02-02).

## Notes & Insights
- Keep this file updated for each user request: add tasks, mark done, and capture notable decisions.
- Skill created at `skills/peacegames/` with runbook in `skills/peacegames/references/commands.md`.
- User asked how to view end-of-round data and world map after a new sim.
- User asked whether the latest sim used fixed-map territories from `worldmap/world_territories_32.json`.
- User ran `python -m run_llm_resource_simulation initial_setup/resource_demo_4p.json --model gpt-5.1 diplomacy,opportunist,austerity,sneaky`.
- User asked where the skills file and TODO are located.
- User requested updating the skill to only require tests when changing .py or data files.
- User reported territory names not matching expected map overlays.
- User noted `world_map_32_internal_labeled.png` has the correct names; likely mismatch between UI JSON cache and updated map assets.
- Root cause: `draw_terrs_v2.py` writes `world_territories_32.json` using default names, not `name_overrides.json`.
- Regenerated `worldmap/world_territories_32.json` after applying name overrides.
