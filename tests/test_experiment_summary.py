import unittest

from run_experiments import _rank_agents


class ExperimentSummaryTests(unittest.TestCase):
    def test_rank_agents(self) -> None:
        welfare = {"B": 5, "A": 5, "C": 1}
        ranked = _rank_agents(welfare)
        self.assertEqual(ranked[0], ("A", 5, 1))
        self.assertEqual(ranked[1], ("B", 5, 2))
        self.assertEqual(ranked[2], ("C", 1, 3))


if __name__ == "__main__":
    unittest.main()
