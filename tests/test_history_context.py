import unittest

from peacegame.simulation import build_history_context


class HistoryContextTests(unittest.TestCase):
    def test_build_history_context_truncates(self) -> None:
        history = "Important long-term history."
        summaries = [
            (0, "t0 summary"),
            (1, "t1 summary"),
            (2, "t2 summary"),
        ]
        ctx = build_history_context(
            history_summary=history,
            turn_summaries=summaries,
            max_chars=50,
        )
        self.assertIn("History summary", ctx)
        # Should include the newest turn if it fits
        self.assertIn("turn 2", ctx)

    def test_build_history_context_empty(self) -> None:
        ctx = build_history_context(history_summary="", turn_summaries=[], max_chars=100)
        self.assertEqual(ctx, "")


if __name__ == "__main__":
    unittest.main()
