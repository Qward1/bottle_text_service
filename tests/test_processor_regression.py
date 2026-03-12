from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.processor import build_grouped_boxes_from_raw, choose_best_date_cluster


def build_false_positive_scene() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h, w = 900, 700
    gray = np.full((h, w), 205, np.uint8)

    # Bottle-like central region with a weak printed date.
    cv2.rectangle(gray, (200, 250), (520, 720), 170, thickness=-1)
    for idx, text in enumerate(("30.03.26", "31.10.25", "10:04")):
        x = 290 if idx < 2 else 325
        y = 500 + (idx * 38)
        cv2.putText(gray, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, 182, 1, cv2.LINE_AA)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.circle(gray, (430, 545), 35, 235, thickness=-1)

    # High-contrast textured background on the right that used to win the score.
    for y in range(210, 700, 26):
        cv2.line(gray, (565, y), (650, y + 12), 120, 2)
        cv2.line(gray, (565, y + 8), (650, y + 20), 235, 1)

    boxes = [
        (280, 470, 395, 515),
        (280, 510, 395, 555),
        (320, 548, 390, 590),
        (555, 210, 658, 280),
        (555, 300, 660, 370),
        (555, 390, 660, 470),
        (555, 500, 660, 590),
    ]
    return gray, boxes


class ProcessorRegressionTests(unittest.TestCase):
    def test_choose_best_date_cluster_prefers_center_date_over_side_texture(self) -> None:
        gray, boxes = build_false_positive_scene()
        glare = np.zeros_like(gray)

        chosen, scored = choose_best_date_cluster(boxes, gray, glare, gray.shape)

        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertGreater(len(scored), 0)

        x1, _, x2, _ = chosen
        self.assertLess(x2, int(gray.shape[1] * 0.72))
        self.assertGreater(x1, int(gray.shape[1] * 0.24))

    def test_build_grouped_boxes_from_raw_skips_tall_vertical_false_groups(self) -> None:
        raw_boxes = [
            (560, 180, 630, 250),
            (562, 270, 632, 340),
            (564, 360, 634, 430),
            (566, 450, 636, 520),
        ]

        grouped = build_grouped_boxes_from_raw(raw_boxes)

        self.assertNotIn((560, 180, 636, 520), grouped)


if __name__ == "__main__":
    unittest.main()
