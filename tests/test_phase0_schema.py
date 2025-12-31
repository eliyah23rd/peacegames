import unittest

from peacegame.phase0 import translate_agent_actions_to_intentions


class Phase0SchemaTests(unittest.TestCase):
    def test_new_summary_fields(self) -> None:
        actions = {
            "John": {
                "summary_last_turn": "last",
                "history_summary": "history",
                "reasoning": "short reasoning",
                "disband_mils": 2,
            }
        }
        (
            _,
            _,
            _,
            _,
            _,
            d_summary_last_turn,
            d_history_summary,
            d_reasoning,
            d_mils_disband_intent,
        ) = translate_agent_actions_to_intentions(
            actions,
            known_agents={"John"},
            agent_territories={"John": set()},
            max_summary_len=64,
        )
        self.assertEqual(d_summary_last_turn["John"], "last")
        self.assertEqual(d_history_summary["John"], "history")
        self.assertEqual(d_reasoning["John"], "short reasoning")
        self.assertEqual(d_mils_disband_intent["John"], 2)


if __name__ == "__main__":
    unittest.main()
