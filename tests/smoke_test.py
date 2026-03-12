from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from app.processor import encode_image, process_image


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "_out"
OUT_DIR.mkdir(exist_ok=True)


def build_challenging_bottle() -> tuple[bytes, tuple[int, int, int, int]]:
    h, w = 1100, 820
    canvas = np.full((h, w, 3), 232, dtype=np.uint8)

    cv2.rectangle(canvas, (240, 120), (600, 980), (210, 214, 219), thickness=-1)
    cv2.rectangle(canvas, (315, 20), (525, 180), (206, 210, 214), thickness=-1)
    cv2.rectangle(canvas, (265, 360), (575, 730), (242, 242, 242), thickness=-1)

    weak_color = (150, 150, 150)
    cv2.putText(canvas, "LOT 7241", (300, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.0, weak_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "EXP 12.11.2027", (285, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (160, 160, 160), 2, cv2.LINE_AA)
    cv2.putText(canvas, "BATCH 096", (310, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (155, 155, 155), 2, cv2.LINE_AA)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    overlay = canvas.copy()
    cv2.ellipse(overlay, (430, 555), (145, 34), -18, 0, 360, (255, 255, 255), thickness=-1)
    cv2.ellipse(overlay, (455, 535), (120, 20), -18, 0, 360, (250, 250, 250), thickness=-1)
    canvas = cv2.addWeighted(overlay, 0.23, canvas, 0.77, 0)

    shadow = np.zeros_like(canvas)
    cv2.rectangle(shadow, (280, 520), (560, 600), (18, 18, 18), thickness=-1)
    shadow = cv2.GaussianBlur(shadow, (41, 41), 0)
    canvas = cv2.addWeighted(canvas, 1.0, shadow, 0.08, 0)

    rng = np.random.default_rng(11)
    noise = rng.normal(0, 9, size=canvas.shape).astype(np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 1.7, 1.0)
    rotated = cv2.warpAffine(noisy, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    expected_date_region = (270, 505, 585, 610)
    return encode_image(rotated, ext=".jpg", quality=92), expected_date_region


def save_debug_outputs(original_bgr: np.ndarray, metadata: dict) -> None:
    x1 = metadata["crop_box"]["x1"]
    y1 = metadata["crop_box"]["y1"]
    x2 = metadata["crop_box"]["x2"]
    y2 = metadata["crop_box"]["y2"]

    overlay = original_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(str(OUT_DIR / "debug_roi.jpg"), overlay)

    crop = original_bgr[y1:y2, x1:x2]
    if crop.size:
        cv2.imwrite(str(OUT_DIR / "crop_preview.jpg"), crop)


if __name__ == "__main__":
    real_path = os.getenv("BOTTLE_TEST_IMAGE")
    expected = None

    if real_path and Path(real_path).exists():
        original = cv2.imread(real_path)
        content = Path(real_path).read_bytes()
    else:
        content, expected = build_challenging_bottle()
        original = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    result = process_image(content, detector_backend="craft")

    cv2.imwrite(str(OUT_DIR / "improved.jpg"), result.improved_bgr)
    cv2.imwrite(str(OUT_DIR / "bw.png"), result.bw)
    cv2.imwrite(str(OUT_DIR / "high_contrast.jpg"), result.high_contrast)

    meta = result.metadata.__dict__
    save_debug_outputs(original, meta)

    print(meta)
    print("Saved outputs to:", OUT_DIR)

    assert result.improved_bgr.size > 0
    assert result.bw.size > 0
    assert result.high_contrast.size > 0

    if expected is not None:
        x1 = meta["crop_box"]["x1"]
        y1 = meta["crop_box"]["y1"]
        x2 = meta["crop_box"]["x2"]
        y2 = meta["crop_box"]["y2"]
        ex1, ey1, ex2, ey2 = expected
        overlap_w = max(0, min(x2, ex2) - max(x1, ex1))
        overlap_h = max(0, min(y2, ey2) - max(y1, ey1))
        overlap_area = overlap_w * overlap_h
        expected_area = max(1, (ex2 - ex1) * (ey2 - ey1))

        assert meta["crop_found"] is True
        assert meta["detection_confidence"] > 2.0
        assert overlap_area / expected_area > 0.25, (meta["crop_box"], expected)
