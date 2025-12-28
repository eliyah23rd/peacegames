import datetime as dt
import html
import json
import os
import unittest

from peacegame.agents import ScriptedAgent
from peacegame.phase0 import run_phase0


TEST_SCRIPTS_DIR = "test_scripts"
LOG_DIR = "logs"
DATASHEET_DIR = "datasheets"


def _load_scripts() -> list[dict]:
    scripts = []
    for fname in sorted(os.listdir(TEST_SCRIPTS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(TEST_SCRIPTS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            scripts.append(json.load(f))
    return scripts


def _ensure_dirs() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DATASHEET_DIR, exist_ok=True)


def _write_xls(path: str, headers: list[str], rows: list[list[str]]) -> None:
    lines = [
        "<html><head><meta charset=\"utf-8\"></head><body>",
        "<table border=\"1\">",
        "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>",
    ]
    for row in rows:
        lines.append(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        )
    lines.append("</table></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class Phase0ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dirs()
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        cls.log_path = os.path.join(LOG_DIR, f"run_{run_id}.log")
        cls.sheet_path = os.path.join(DATASHEET_DIR, f"run_{run_id}.xls")
        cls._log_fp = open(cls.log_path, "w", encoding="utf-8")
        cls._rows: list[list[str]] = []

    @classmethod
    def tearDownClass(cls) -> None:
        headers = ["script", "turn", "agent", "phase", "ledger", "value"]
        _write_xls(cls.sheet_path, headers, cls._rows)
        cls._log_fp.close()

    def _log(self, msg: str) -> None:
        self._log_fp.write(msg + "\n")
        self._log_fp.flush()

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

        self._log(f"Script {name} start")
        self._log(f"Initial territories: {sorted((a, sorted(list(t))) for a, t in agent_territories.items())}")
        self._log(f"Initial mils: {sorted(agent_mils.items())}")
        self._log(f"Constants: {constants}")

        turn_summaries: dict[str, str] = {a: "" for a in agent_names}

        for idx, turn in enumerate(turns):
            turn_num = int(turn.get("turn", idx))
            self._log(f"Turn {turn_num} start")
            self._log(f"Raw actions: {turn.get('actions', {})}")

            (
                d_mil_purchase_intent,
                d_global_attacks,
                d_territory_cession,
                d_money_grants,
                d_messages_sent,
                d_turn_summary,
            ) = run_phase0(
                turn=turn_num,
                agents=agents,
                agent_territories=agent_territories,
                agent_mils=agent_mils,
                constants=constants,
                turn_summaries=turn_summaries,
                max_summary_len=64,
                log_fn=self._log,
            )

            expected = turn.get("expected", {})
            for ledger_name, ledger_val in expected.items():
                actual = locals()[ledger_name]
                self.assertEqual(actual, ledger_val, f"Mismatch for {ledger_name} in {name}")

            for agent in agent_names:
                self._rows.append(
                    [name, str(turn_num), agent, "phase0", "d_mil_purchase_intent", json.dumps(d_mil_purchase_intent.get(agent, 0), sort_keys=True)]
                )
                self._rows.append(
                    [name, str(turn_num), agent, "phase0", "d_global_attacks", json.dumps(d_global_attacks.get(agent, {}), sort_keys=True)]
                )
                self._rows.append(
                    [name, str(turn_num), agent, "phase0", "d_territory_cession", json.dumps(d_territory_cession.get(agent, {}), sort_keys=True)]
                )
                self._rows.append(
                    [name, str(turn_num), agent, "phase0", "d_money_grants", json.dumps(d_money_grants.get(agent, {}), sort_keys=True)]
                )
                self._rows.append(
                    [name, str(turn_num), agent, "phase0", "d_messages_sent", json.dumps(d_messages_sent.get(agent, {}), sort_keys=True)]
                )
                self._rows.append(
                    [name, str(turn_num), agent, "phase0", "d_turn_summary", d_turn_summary.get(agent, "")]
                )

            turn_summaries = d_turn_summary
            self._log(f"Turn {turn_num} outputs: {d_turn_summary}")

        self._log(f"Script {name} end")


if __name__ == "__main__":
    unittest.main()
