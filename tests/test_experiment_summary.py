import unittest

from analyze_experiment import compute_modifier_stats, summarize_experiment
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

    def test_analyze_experiment_summary(self) -> None:
        with open("tests/fixtures/experiment_sample.json", "r", encoding="utf-8") as f:
            data = __import__("json").load(f)
        stats = compute_modifier_stats(data)
        self.assertEqual(stats["diplomacy"].wins, 2)
        summary = summarize_experiment(data)
        self.assertIn("Experiment: sample_experiment", summary)
        self.assertIn("Round 1: A (diplomacy) welfare=100", summary)


if __name__ == "__main__":
    unittest.main()
