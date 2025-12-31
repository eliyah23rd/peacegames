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
                "keeps_word_report": {"John": 9, "Ruth": 3},
                "aggressor_report": {"John": 2, "Ruth": 8},
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
            d_keeps_word_report,
            d_aggressor_report,
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
        self.assertEqual(d_keeps_word_report["John"], {"John": 9})
        self.assertEqual(d_aggressor_report["John"], {"John": 2})

    def test_report_filters_invalid_scores(self) -> None:
        actions = {
            "John": {
                "keeps_word_report": {"John": 0, "Ruth": 11, "Dima": 5},
                "aggressor_report": {"John": "x", "Dima": 7},
            }
        }
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            d_keeps_word_report,
            d_aggressor_report,
        ) = translate_agent_actions_to_intentions(
            actions,
            known_agents={"John", "Dima"},
            agent_territories={"John": set()},
            max_summary_len=64,
        )
        self.assertEqual(d_keeps_word_report["John"], {"Dima": 5})
        self.assertEqual(d_aggressor_report["John"], {"Dima": 7})


if __name__ == "__main__":
    unittest.main()
