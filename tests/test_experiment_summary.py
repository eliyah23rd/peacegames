import unittest

from run_experiments import _format_round_command, _rank_agents


class ExperimentSummaryTests(unittest.TestCase):
    def test_rank_agents(self) -> None:
        welfare = {"B": 5, "A": 5, "C": 1}
        ranked = _rank_agents(welfare)
        self.assertEqual(ranked[0], ("A", 5, 1))
        self.assertEqual(ranked[1], ("B", 5, 2))
        self.assertEqual(ranked[2], ("C", 1, 3))

    def test_format_round_command(self) -> None:
        cmd = _format_round_command(
            setup_path="initial_setup/foo.json",
            model="gpt-test",
            modifiers=["a", "b"],
            round_idx=0,
            rounds=3,
        )
        self.assertEqual(
            cmd,
            "Round 1/3: python run_llm_simulation.py initial_setup/foo.json --model gpt-test a b",
        )


if __name__ == "__main__":
    unittest.main()
