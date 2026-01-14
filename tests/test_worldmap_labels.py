import unittest

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
