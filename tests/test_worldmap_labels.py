import json
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

    def test_compute_notice_board_box(self):
        land = np.zeros((gen.H, gen.W), dtype=bool)
        box = gen.compute_notice_board_box(land, ratio=2.0, margin=10, ignore_bottom=0)
        self.assertEqual(len(box), 4)
        self.assertGreater(box[2], box[0])
        self.assertGreater(box[3], box[1])

    def test_world_territory_names_follow_overrides(self):
        root = Path(__file__).resolve().parents[1]
        overrides_path = root / "worldmap" / "name_overrides.json"
        data_path = root / "worldmap" / "world_territories_32.json"
        if not overrides_path.exists():
            self.skipTest(f"Missing overrides: {overrides_path}")
        if not data_path.exists():
            self.skipTest(f"Missing territory data: {data_path}")
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        data = json.loads(data_path.read_text(encoding="utf-8"))
        names_by_id = {t["id"]: t.get("name") for t in data.get("territories", [])}
        for tid, name in overrides.items():
            self.assertIn(tid, names_by_id, f"Missing territory id in JSON: {tid}")
            self.assertEqual(names_by_id[tid], name)
