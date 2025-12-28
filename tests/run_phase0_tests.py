import unittest

from peacegame.agents import ScriptedAgent
from peacegame.phase0 import assemble_agent_inputs, run_phase0


class Phase0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.constants = {
            "c_money_per_territory": 10,
            "c_mil_purchase_price": 20,
            "c_mil_upkeep_price": 2,
            "c_defense_destroy_factor": 4,
            "c_trade_factor": 2.0,
        }
        self.agent_territories = {
            "A": {"T1", "T2"},
            "B": {"T3"},
        }
        self.agent_mils = {"A": 5, "B": 3}

    def test_assemble_inputs_turn0_no_summary(self) -> None:
        inputs = assemble_agent_inputs(
            turn=0,
            agent_names=["A", "B"],
            agent_territories=self.agent_territories,
            agent_mils=self.agent_mils,
            constants=self.constants,
            turn_summaries={},
        )
        self.assertNotIn("previous_turn_summary", inputs["A"])
        self.assertEqual(inputs["A"]["turn"], 0)
        self.assertEqual(inputs["A"]["self"], "A")

    def test_phase0_happy_path(self) -> None:
        agents = {
            "A": ScriptedAgent(
                actions=[
                    {
                        "purchase_mils": 2,
                        "attacks": {"B": 1},
                        "cede_territories": {"B": ["T1"]},
                        "money_grants": {"B": 5},
                        "messages": {"all": "hello"},
                        "summary": "turn0",
                    }
                ]
            ),
            "B": ScriptedAgent(actions=[{"summary": "ok"}]),
        }

        (
            d_mil_purchase_intent,
            d_global_attacks,
            d_territory_cession,
            d_money_grants,
            d_messages_sent,
            d_turn_summary,
        ) = run_phase0(
            turn=0,
            agents=agents,
            agent_territories=self.agent_territories,
            agent_mils=self.agent_mils,
            constants=self.constants,
            turn_summaries={},
            max_summary_len=64,
        )

        self.assertEqual(d_mil_purchase_intent["A"], 2)
        self.assertEqual(d_global_attacks["A"], {"B": 1})
        self.assertEqual(d_territory_cession["A"], {"B": ["T1"]})
        self.assertEqual(d_money_grants["A"], {"B": 5})
        self.assertEqual(d_messages_sent["A"], {"all": "hello"})
        self.assertEqual(d_turn_summary["A"], "turn0")
        self.assertEqual(d_turn_summary["B"], "ok")

    def test_phase0_validation_drops_invalid(self) -> None:
        agents = {
            "A": ScriptedAgent(
                actions=[
                    {
                        "purchase_mils": -5,
                        "attacks": {"A": 2, "C": 3, "B": "x"},
                        "cede_territories": {"B": ["T9"], "A": ["T1"]},
                        "money_grants": {"B": -1, "A": 2},
                        "messages": {"C": "nope", "all": 5},
                        "summary": 123,
                    }
                ]
            ),
            "B": ScriptedAgent(actions=[{}]),
        }

        (
            d_mil_purchase_intent,
            d_global_attacks,
            d_territory_cession,
            d_money_grants,
            d_messages_sent,
            d_turn_summary,
        ) = run_phase0(
            turn=1,
            agents=agents,
            agent_territories=self.agent_territories,
            agent_mils=self.agent_mils,
            constants=self.constants,
            turn_summaries={"A": "prev"},
            max_summary_len=64,
        )

        self.assertEqual(d_mil_purchase_intent["A"], 0)
        self.assertEqual(d_global_attacks["A"], {})
        self.assertEqual(d_territory_cession["A"], {})
        self.assertEqual(d_money_grants["A"], {})
        self.assertEqual(d_messages_sent["A"], {})
        self.assertEqual(d_turn_summary["A"], "")

    def test_phase0_truncates_summary(self) -> None:
        agents = {
            "A": ScriptedAgent(actions=[{"summary": "abc" * 10}]),
        }
        (
            _,
            _,
            _,
            _,
            _,
            d_turn_summary,
        ) = run_phase0(
            turn=1,
            agents=agents,
            agent_territories={"A": {"T1"}},
            agent_mils={"A": 0},
            constants=self.constants,
            turn_summaries={"A": "prev"},
            max_summary_len=5,
        )
        self.assertEqual(d_turn_summary["A"], "abcab")


if __name__ == "__main__":
    unittest.main()
