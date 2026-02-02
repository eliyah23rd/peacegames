import unittest
from pathlib import Path


class SkillFilesTests(unittest.TestCase):
    def test_skill_file_exists_and_has_frontmatter(self) -> None:
        skill_path = Path(__file__).resolve().parents[1] / "skills" / "peacegames" / "SKILL.md"
        self.assertTrue(skill_path.is_file(), f"Missing skill file: {skill_path}")
        text = skill_path.read_text(encoding="utf-8")
        self.assertIn("name: peacegames", text)
        self.assertIn("description:", text)

    def test_todo_file_exists_and_has_sections(self) -> None:
        todo_path = Path(__file__).resolve().parents[1] / "TODO.md"
        self.assertTrue(todo_path.is_file(), f"Missing TODO file: {todo_path}")
        text = todo_path.read_text(encoding="utf-8")
        self.assertIn("## To Do", text)
        self.assertIn("## Done", text)
        self.assertIn("## Notes & Insights", text)


if __name__ == "__main__":
    unittest.main()
