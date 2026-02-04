import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from peacegame.worldmap_renderer import _draw_notice_legend


class WorldMapRendererTests(unittest.TestCase):
    def test_draw_notice_legend_marks_image(self) -> None:
        img = Image.new("RGB", (200, 200), color="white")
        _draw_notice_legend(
            img,
            notice_box=(0, 0, 200, 200),
            owner_colors={"A": "#ff0000", "B": "#00ff00"},
            pie_colors={"lost": "#111111", "welfare": "#222222"},
            pie_order=["lost", "welfare"],
            icons_dir=Path(__file__).resolve().parents[1] / "icons",
        )
        arr = np.array(img)
        self.assertTrue((arr != 255).any())


if __name__ == "__main__":
    unittest.main()
