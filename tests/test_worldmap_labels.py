import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldmap import draw_terrs_v2 as gen


class WorldMapLabelTests(unittest.TestCase):
    def test_add_name_labels_pie(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        centers = [(10, 10)]
        names = ["TestTerr"]
        out = gen.add_name_labels(
            image,
            centers,
            names,
            resource_style="pie",
            icon_count_per_resource=2,
        )
        self.assertEqual(out.shape, image.shape)

    def test_build_label_layouts_icons(self):
        centers = [(5, 5)]
        names = ["Example"]
        layouts = gen.build_label_layouts(
            centers,
            names,
            font=gen.load_label_font(14),
            icon_size=12,
            icon_count_per_resource=3,
            resource_style="icons",
        )
        self.assertEqual(len(layouts), 1)
        left, _top, right, _bottom = layouts[0]["box"]
        self.assertGreater(right, left)

    def test_random_palette_count(self):
        colors = gen._build_random_palette(5)
        self.assertEqual(len(colors), 5)

    def test_overlay_notice_board(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            board_path = Path(tmp_dir) / "board.png"
            image = np.full((60, 90, 3), 255, dtype=np.uint8)
            board = np.zeros((20, 10, 3), dtype=np.uint8)
            gen.Image.fromarray(board).save(board_path)
            out = gen.overlay_notice_board(image, board_path, (5, 5, 50, 50))
            self.assertEqual(out.shape, image.shape)
