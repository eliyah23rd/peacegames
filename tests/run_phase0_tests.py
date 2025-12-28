import json
import os
import unittest

from peacegame.agents import ScriptedAgent
from peacegame.engine import Phase0Engine


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
        cls.engine = Phase0Engine(run_label="phase0_tests")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.close()

    def test_run_all_scripts(self) -> None:
        scripts = _load_scripts()
        self.assertGreater(len(scripts), 0, "No test scripts found")

        for script in scripts:
            with self.subTest(script=script.get("name", "(unnamed)")):
                self._run_script(script)

    def _run_script(self, script: dict) -> None:
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

        self.engine.log_initial_state(
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
                d_turn_summary,
            ) = self.engine.run_turn(
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

            turn_summaries = d_turn_summary

        self.engine.log_script_end(script_name=name)


if __name__ == "__main__":
    unittest.main()
