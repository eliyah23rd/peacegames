import json
import os
import unittest

from peacegame.agents import ScriptedAgent
from peacegame.engine import Phase0Engine
from peacegame.simulation import SimulationEngine


TEST_SCRIPTS_DIR = "test_scripts"


def _load_scripts() -> list[dict]:
    scripts = []
    for fname in sorted(os.listdir(TEST_SCRIPTS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(TEST_SCRIPTS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            scripts.append(json.load(f))
    return scripts


class Phase0ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase0_engine = Phase0Engine(run_label="phase0_tests")
        cls.sim_engine = SimulationEngine(run_label="simulation_tests")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.phase0_engine.close()
        cls.sim_engine.close()

    def test_run_all_scripts(self) -> None:
        scripts = _load_scripts()
        self.assertGreater(len(scripts), 0, "No test scripts found")

        for script in scripts:
            with self.subTest(script=script.get("name", "(unnamed)")):
                engine = script.get("engine", "phase0")
                if engine == "simulation":
                    self._run_simulation_script(script)
                else:
                    self._run_phase0_script(script)

    def _run_phase0_script(self, script: dict) -> None:
        name = script.get("name", "(unnamed)")
        constants = script.get("constants", {})
        initial_state = script.get("initial_state", {})
        turns = script.get("turns", [])

        agent_territories = {
            agent: set(territories)
            for agent, territories in initial_state.get("agent_territories", {}).items()
        }
        agent_mils = {
            agent: int(mils) for agent, mils in initial_state.get("agent_mils", {}).items()
        }

        agent_names = set(agent_territories.keys()) | set(agent_mils.keys())
        for turn in turns:
            agent_names |= set(turn.get("actions", {}).keys())
        agent_names = sorted(agent_names)

        actions_by_agent: dict[str, list[str]] = {a: [] for a in agent_names}
        for turn in turns:
            actions = turn.get("actions", {})
            for agent in agent_names:
                action = actions.get(agent, "{}")
                self.assertIsInstance(
                    action, str, f"Script {name} action for {agent} must be JSON string"
                )
                actions_by_agent[agent].append(action)

        agents = {agent: ScriptedAgent(actions=actions_by_agent[agent]) for agent in agent_names}

        self.phase0_engine.log_initial_state(
            script_name=name,
            agent_territories=agent_territories,
            agent_mils=agent_mils,
            constants=constants,
        )

        turn_summaries: dict[str, str] = {a: "" for a in agent_names}

        for idx, turn in enumerate(turns):
            turn_num = int(turn.get("turn", idx))

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
            ) = self.phase0_engine.run_turn(
                script_name=name,
                turn=turn_num,
                agents=agents,
                agent_territories=agent_territories,
                agent_mils=agent_mils,
                constants=constants,
                turn_summaries=turn_summaries,
                max_summary_len=64,
            )

            expected = turn.get("expected", {})
            for ledger_name, ledger_val in expected.items():
                actual = locals()[ledger_name]
                self.assertEqual(actual, ledger_val, f"Mismatch for {ledger_name} in {name}")

            turn_summaries = d_summary_last_turn

        self.phase0_engine.log_script_end(script_name=name)

    def _run_simulation_script(self, script: dict) -> None:
        name = script.get("name", "(unnamed)")
        constants = script.get("constants", {})
        initial_state = script.get("initial_state", {})
        turns = script.get("turns", [])

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

        agent_names = set(agent_territories.keys()) | set(agent_mils.keys()) | set(agent_welfare.keys())
        for turn in turns:
            agent_names |= set(turn.get("actions", {}).keys())
        agent_names = sorted(agent_names)

        for agent in agent_names:
            agent_territories.setdefault(agent, set())
            agent_mils.setdefault(agent, 0)
            agent_welfare.setdefault(agent, 0)

        actions_by_agent: dict[str, list[str]] = {a: [] for a in agent_names}
        for turn in turns:
            actions = turn.get("actions", {})
            for agent in agent_names:
                action = actions.get(agent, "{}")
                self.assertIsInstance(
                    action, str, f"Script {name} action for {agent} must be JSON string"
                )
                actions_by_agent[agent].append(action)

        agents = {agent: ScriptedAgent(actions=actions_by_agent[agent]) for agent in agent_names}

        self.sim_engine.log_initial_state(
            script_name=name,
            agent_territories=agent_territories,
            agent_mils=agent_mils,
            agent_welfare=agent_welfare,
            constants=constants,
        )
        self.sim_engine.setup_state(
            agent_territories=agent_territories,
            agent_mils=agent_mils,
            agent_welfare=agent_welfare,
            territory_seed=42,
        )
        self.sim_engine.setup_round(total_turns=len(turns))

        turn_summaries: dict[str, str] = {a: "" for a in agent_names}

        for idx, turn in enumerate(turns):
            turn_num = int(turn.get("turn", idx))

            results = self.sim_engine.run_turn(
                script_name=name,
                turn=turn_num,
                agents=agents,
                constants=constants,
                turn_summaries=turn_summaries,
                max_summary_len=64,
            )

            expected = turn.get("expected", {})
            expected_state = expected.get("state", {})
            expected_ledgers = expected.get("ledgers", {})
            expected_news = expected.get("news_report")
            expected_reports = expected.get("agent_reports")

            for key, val in expected_ledgers.items():
                self.assertEqual(results.get(key), val, f"Mismatch for {key} in {name}")

            for key, val in expected_state.items():
                self.assertEqual(results.get(key), val, f"Mismatch for {key} in {name}")

            if expected_news is not None:
                self.assertEqual(results.get("news_report"), expected_news, f"Mismatch for news_report in {name}")
            if expected_reports is not None:
                self.assertEqual(results.get("agent_reports"), expected_reports, f"Mismatch for agent_reports in {name}")

            turn_summaries = results.get("d_summary_last_turn", turn_summaries)

        self.sim_engine.log_script_end(script_name=name)


if __name__ == "__main__":
    unittest.main()
