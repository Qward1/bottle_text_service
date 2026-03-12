
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Literal, Sequence, Tuple

import os

import cv2
import numpy as np

try:  # pragma: no cover - optional runtime dependency
    from crafter import Crafter  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    Crafter = None


DetectorBackend = Literal["craft", "heuristic"]


@dataclass
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class ProcessMetadata:
    original_width: int
    original_height: int
    crop_box: Dict[str, int]
    crop_found: bool
    upscale_factor: float
    candidate_boxes: int
    rotated_degrees: float
    glare_ratio: float
    detection_confidence: float
    top_candidates: List[Dict[str, float | int | str]]
    detector_requested: str
    detector_backend: str
    detector_fallback_used: bool
    craft_available: bool


@dataclass
class ProcessResult:
    crop_bgr: np.ndarray
    improved_bgr: np.ndarray
    bw: np.ndarray
    high_contrast: np.ndarray
    metadata: ProcessMetadata


@dataclass
class ScoredCandidate:
    box: Tuple[int, int, int, int]
    score: float
    component_count: int
    glare_ratio: float
    area_ratio: float
    aspect_ratio: float
    source: str = "heuristic"


def decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    return image


def encode_image(image: np.ndarray, ext: str = ".jpg", quality: int = 95) -> bytes:
    params = []
    if ext.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, encoded = cv2.imencode(ext, image, params)
    if not success:
        raise ValueError("Не удалось закодировать изображение")
    return encoded.tobytes()


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    means = img.mean(axis=(0, 1))
    gray_mean = float(np.mean(means))
    scale = gray_mean / (means + 1e-6)
    balanced = img * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def resize_with_min_side(
    image: np.ndarray,
    min_side: int,
    max_scale: float = 10.0,
    max_side: int = 2600,
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    current_min = min(h, w)
    current_max = max(h, w)
    if current_min >= min_side:
        return image.copy(), 1.0

    target_scale = float(min_side) / float(max(1, current_min))
    side_cap_scale = float(max_side) / float(max(1, current_max))
    scale = min(target_scale, max_scale, side_cap_scale)
    if scale <= 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return resized, float(scale)


def resize_with_max_side(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return image.copy(), 1.0
    scale = float(max_side) / float(current_max)
    resized = cv2.resize(
        image,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, float(scale)


def ensure_dark_text_on_light(binary_img: np.ndarray) -> np.ndarray:
    white_ratio = float(np.mean(binary_img > 127))
    if white_ratio < 0.5:
        return 255 - binary_img
    return binary_img


def remove_border_artifacts(binary_img: np.ndarray) -> np.ndarray:
    h, w = binary_img.shape[:2]
    inv = (255 - binary_img).copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((inv > 0).astype(np.uint8), connectivity=8)

    max_thin = max(6, int(min(h, w) * 0.015))
    max_area = int(h * w * 0.02)

    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        touches_border = x <= 0 or y <= 0 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1)
        thin = min(bw, bh) <= max_thin
        small = area <= max_area
        if touches_border and (thin or small):
            inv[labels == idx] = 0

    return 255 - inv


def _rect_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0, max(ay1, by1) - min(ay2, by2))
    return gap_x, gap_y


def _overlap_ratio_1d(a1: int, a2: int, b1: int, b2: int) -> float:
    overlap = max(0, min(a2, b2) - max(a1, b1))
    base = max(1, min(a2 - a1, b2 - b1))
    return float(overlap) / float(base)


def clip_box(box: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    return (
        int(max(0, min(w, x1))),
        int(max(0, min(h, y1))),
        int(max(0, min(w, x2))),
        int(max(0, min(h, y2))),
    )


def pad_box(box: Tuple[int, int, int, int], image_shape: Tuple[int, int, int], padding_ratio: float) -> CropBox:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = clip_box(box, image_shape)
    pad_x = int(round((x2 - x1) * padding_ratio))
    pad_y = int(round((y2 - y1) * padding_ratio))
    min_pad = max(4, int(round(min(h, w) * 0.01)))
    return CropBox(
        x1=int(max(0, x1 - max(pad_x, min_pad))),
        y1=int(max(0, y1 - max(pad_y, min_pad))),
        x2=int(min(w, x2 + max(pad_x, min_pad))),
        y2=int(min(h, y2 + max(pad_y, min_pad))),
    )


def scale_box(box: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    if abs(scale - 1.0) < 1e-6:
        return tuple(map(int, box))
    x1, y1, x2, y2 = box
    return (
        int(round(x1 / scale)),
        int(round(y1 / scale)),
        int(round(x2 / scale)),
        int(round(y2 / scale)),
    )


def gaussian_pref(value: float, center: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    return float(np.exp(-((value - center) ** 2) / (2.0 * sigma * sigma)))


def triangle_pref(value: float, low: float, peak: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value == peak:
        return 1.0
    if value < peak:
        return float((value - low) / max(peak - low, 1e-6))
    return float((high - value) / max(high - peak, 1e-6))


def build_glare_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sigma = max(6.0, min(image.shape[:2]) / 28.0)
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    residue = gray.astype(np.float32) - background

    near_clip = (v >= 250) & (s <= 35)
    bright_anomaly = (gray >= 235) & (s <= 70) & (residue >= 22.0)
    mask = np.where(near_clip | bright_anomaly, 255, 0).astype(np.uint8)

    k = max(3, int(round(min(image.shape[:2]) * 0.012)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 2), max(3, k // 2))),
    )

    min_area = max(18, int(image.shape[0] * image.shape[1] * 0.00005))
    max_area = int(image.shape[0] * image.shape[1] * 0.10)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask)
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < min_area or area > max_area:
            continue
        if bw <= 2 or bh <= 2:
            continue
        cleaned[labels == idx] = 255
    return cleaned


def inpaint_for_detection(image: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
    glare_ratio = float(np.mean(glare_mask > 0))
    if glare_ratio <= 0.001:
        return image.copy()
    work_mask = cv2.dilate(glare_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    radius = 4 if glare_ratio < 0.06 else 3
    return cv2.inpaint(image, work_mask, radius, cv2.INPAINT_TELEA)


def normalize_illumination_gray(gray: np.ndarray, clip_limit: float = 2.2) -> np.ndarray:
    gray_f = gray.astype(np.float32) + 1.0
    sigma = max(7.0, min(gray.shape[:2]) / 18.0)
    background = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = gray_f / (background + 1e-3)
    normalized = normalized * (178.0 / (float(np.mean(normalized)) + 1e-3))
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    return normalized


def build_detection_gray(image: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
    suppressed = inpaint_for_detection(image, glare_mask)
    gray = cv2.cvtColor(suppressed, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
    return normalize_illumination_gray(gray, clip_limit=2.0)


def extract_component_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = binary.shape[:2]
    min_area = max(8, int(h * w * 0.00002))
    max_area = int(h * w * 0.035)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []

    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < min_area or area > max_area:
            continue
        if bw < 2 or bh < 4:
            continue
        if bw > int(w * 0.65) or bh > int(h * 0.5):
            continue
        aspect = bw / float(bh + 1e-6)
        fill = area / float(max(1, bw * bh))
        if aspect < 0.08 or aspect > 18.0:
            continue
        if fill < 0.08 or fill > 0.96:
            continue
        boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    return boxes


def build_component_candidate_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    adaptive_dark = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    adaptive_light = cv2.adaptiveThreshold(
        255 - gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))

    masks = []
    for mask in (adaptive_dark, adaptive_light):
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        joined = cv2.bitwise_or(
            cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_h),
            cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_v),
        )
        masks.append(cleaned)
        masks.append(joined)

    boxes: List[Tuple[int, int, int, int]] = []
    for mask in masks:
        boxes.extend(extract_component_boxes(mask))
    return boxes


def build_line_proposals(
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    h, w = image_shape[:2]
    filtered = []
    for box in boxes:
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = float((bw * bh) / max(1.0, h * w))
        aspect = bw / float(bh + 1e-6)
        if area_ratio > 0.02:
            continue
        if bw < 3 or bh < 4:
            continue
        if aspect > 12.0:
            continue
        filtered.append((int(x1), int(y1), int(x2), int(y2)))

    proposals: set[Tuple[int, int, int, int]] = set()
    for seed in filtered:
        sx1, sy1, sx2, sy2 = seed
        seed_height = max(1, sy2 - sy1)
        seed_yc = (sy1 + sy2) / 2.0

        row = []
        for box in filtered:
            bx1, by1, bx2, by2 = box
            bh = max(1, by2 - by1)
            yc = (by1 + by2) / 2.0
            if abs(yc - seed_yc) <= max(12, int(0.85 * max(seed_height, bh))):
                row.append(box)

        row.sort(key=lambda b: b[0])
        if len(row) < 2:
            continue

        group = [row[0]]
        for box in row[1:]:
            prev = group[-1]
            px1, py1, px2, py2 = prev
            bx1, by1, bx2, by2 = box
            prev_h = max(1, py2 - py1)
            box_h = max(1, by2 - by1)
            gap_x = bx1 - px2
            y_overlap = _overlap_ratio_1d(py1, py2, by1, by2)
            if gap_x <= max(22, int(2.0 * max(prev_h, box_h))) and y_overlap >= 0.12:
                group.append(box)
            else:
                if len(group) >= 2:
                    gx1 = min(b[0] for b in group)
                    gy1 = min(b[1] for b in group)
                    gx2 = max(b[2] for b in group)
                    gy2 = max(b[3] for b in group)
                    if (gx2 - gx1) >= 22 and (gx2 - gx1) / float(max(1, gy2 - gy1)) >= 1.1:
                        proposals.add((gx1, gy1, gx2, gy2))
                group = [box]

        if len(group) >= 2:
            gx1 = min(b[0] for b in group)
            gy1 = min(b[1] for b in group)
            gx2 = max(b[2] for b in group)
            gy2 = max(b[3] for b in group)
            if (gx2 - gx1) >= 22 and (gx2 - gx1) / float(max(1, gy2 - gy1)) >= 1.1:
                proposals.add((gx1, gy1, gx2, gy2))

    return sorted(proposals, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def build_morph_candidate_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = gray.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []

    kernel_widths = sorted({max(15, int(w * 0.03)), max(21, int(w * 0.055))})
    for kw in kernel_widths:
        kh = max(3, int(round(kw * 0.25)))
        main_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, main_kernel)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, main_kernel)
        response = cv2.max(blackhat, tophat)

        grad_x = cv2.Sobel(response, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(response, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.75, cv2.convertScaleAbs(grad_y), 0.25, 0)
        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        merged = cv2.bitwise_or(
            cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.06)), 5))),
            cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, max(15, int(h * 0.08))))),
        )
        merged = cv2.dilate(merged, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(120, int(h * w * 0.00018))
        max_area = int(h * w * 0.22)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            aspect = bw / float(bh + 1e-6)
            if area < min_area or area > max_area:
                continue
            if bw < 18 or bh < 10:
                continue
            if aspect < 0.35 or aspect > 18.0:
                continue
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))

    return boxes


def analyze_binary_components(binary_inv: np.ndarray) -> Tuple[int, float, float, float, float]:
    h, w = binary_inv.shape[:2]
    crop_area = max(1, h * w)
    min_area = max(4, int(crop_area * 0.00025))
    max_area = int(crop_area * 0.18)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats((binary_inv > 0).astype(np.uint8), connectivity=8)
    components: List[Tuple[int, int, int, int, int, float, float]] = []
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < min_area or area > max_area:
            continue
        if bw < 2 or bh < 4:
            continue
        if bw > int(w * 0.7) or bh > int(h * 0.95):
            continue
        aspect = bw / float(bh + 1e-6)
        fill = area / float(max(1, bw * bh))
        if aspect < 0.08 or aspect > 12.0:
            continue
        if fill < 0.08 or fill > 0.95:
            continue
        cx, cy = centroids[idx]
        components.append((int(x), int(y), int(bw), int(bh), int(area), float(cx), float(cy)))

    if not components:
        return 0, 0.0, 0.0, 0.0, 0.0

    heights = np.array([c[3] for c in components], dtype=np.float32)
    y_centers = np.array([c[6] for c in components], dtype=np.float32)
    x_centers = np.array([c[5] for c in components], dtype=np.float32)
    widths = np.array([c[2] for c in components], dtype=np.float32)

    mean_height = float(np.mean(heights)) + 1e-6
    alignment = max(0.0, 1.0 - float(np.std(y_centers) / (mean_height * 1.65)))
    size_consistency = max(0.0, 1.0 - float(np.std(heights) / mean_height))

    if len(components) >= 2:
        order = np.argsort(x_centers)
        gaps = np.diff(x_centers[order])
        mean_gap = float(np.mean(gaps)) + 1e-6
        gap_consistency = max(0.0, 1.0 - float(np.std(gaps) / (mean_gap * 1.4)))
    else:
        gap_consistency = 0.0

    coverage = float(np.sum(widths)) / float(max(1, w))
    coverage_score = triangle_pref(coverage, 0.12, 0.52, 1.05)
    line_score = (alignment * 0.4) + (size_consistency * 0.28) + (gap_consistency * 0.18) + (coverage_score * 0.14)
    return len(components), alignment, size_consistency, gap_consistency, line_score



def score_candidate(
    box: Tuple[int, int, int, int],
    gray: np.ndarray,
    glare_mask: np.ndarray,
    image_shape: Sequence[int],
    source: str = "heuristic",
) -> ScoredCandidate:
    h_img, w_img = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = clip_box(box, image_shape)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return ScoredCandidate(
            box=(x1, y1, x2, y2),
            score=-1.0,
            component_count=0,
            glare_ratio=1.0,
            area_ratio=0.0,
            aspect_ratio=0.0,
            source=source,
        )

    bh, bw = crop.shape[:2]
    area_ratio = float((bw * bh) / max(1.0, h_img * w_img))
    aspect_ratio = float(bw / float(bh + 1e-6))
    y_center_ratio = float(((y1 + y2) / 2.0) / max(1.0, h_img))

    binaries = [
        cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7),
        cv2.adaptiveThreshold(255 - crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7),
    ]

    best_components = 0
    best_line_score = 0.0
    best_alignment = 0.0
    best_consistency = 0.0
    best_gap = 0.0
    best_ink_ratio = 0.0
    for binary_inv in binaries:
        comp_count, alignment, consistency, gap_consistency, line_score = analyze_binary_components(binary_inv)
        ink_ratio = float(np.mean(binary_inv > 0))
        if line_score > best_line_score:
            best_components = comp_count
            best_line_score = line_score
            best_alignment = alignment
            best_consistency = consistency
            best_gap = gap_consistency
            best_ink_ratio = ink_ratio

    edge = cv2.Canny(crop, 40, 140)
    edge_density = float(np.mean(edge > 0))
    contrast_span = float(np.percentile(crop, 92) - np.percentile(crop, 8)) / 255.0
    glare_ratio = float(np.mean(glare_mask[y1:y2, x1:x2] > 0)) if glare_mask.size else 0.0

    aspect_score = max(
        triangle_pref(aspect_ratio, 0.65, 3.6, 14.0),
        triangle_pref(aspect_ratio, 0.9, 1.8, 6.5),
    )
    area_score = max(
        triangle_pref(area_ratio, 0.0004, 0.0035, 0.05),
        triangle_pref(area_ratio, 0.0015, 0.008, 0.08),
    )
    edge_score = triangle_pref(edge_density, 0.02, 0.12, 0.28)
    contrast_score = triangle_pref(contrast_span, 0.08, 0.28, 0.75)
    ink_score = triangle_pref(best_ink_ratio, 0.02, 0.17, 0.42)
    component_score = min(1.0, best_components / 12.0)
    position_score = max(
        triangle_pref(y_center_ratio, 0.16, 0.55, 0.82),
        triangle_pref(y_center_ratio, 0.20, 0.48, 0.74),
    )

    bottom_overlay_penalty = 0.0
    if y1 >= int(h_img * 0.82):
        bottom_overlay_penalty += 1.25
    if y1 >= int(h_img * 0.86) and x1 <= int(w_img * 0.40):
        bottom_overlay_penalty += 1.5
    if bh >= int(h_img * 0.22):
        bottom_overlay_penalty += 0.45

    score = (
        best_line_score * 2.0
        + component_score * 0.95
        + aspect_score * 1.05
        + area_score * 0.85
        + edge_score * 0.45
        + contrast_score * 0.35
        + ink_score * 0.35
        + best_alignment * 0.25
        + best_consistency * 0.2
        + best_gap * 0.15
        + position_score * 0.45
        - glare_ratio * 0.85
        - bottom_overlay_penalty
    )
    if aspect_ratio < 0.75:
        score -= 1.15
    if area_ratio > 0.12:
        score -= (area_ratio - 0.12) * 4.0

    return ScoredCandidate(
        box=(x1, y1, x2, y2),
        score=float(score),
        component_count=int(best_components),
        glare_ratio=float(glare_ratio),
        area_ratio=float(area_ratio),
        aspect_ratio=float(aspect_ratio),
        source=source,
    )


def merge_box_pair(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def boxes_are_related(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    gap_x, gap_y = _rect_distance(a, b)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    x_overlap = _overlap_ratio_1d(ax1, ax2, bx1, bx2)
    y_overlap = _overlap_ratio_1d(ay1, ay2, by1, by2)

    same_line = y_overlap >= 0.34 and gap_x <= max(24, int(1.15 * max(aw, bw)))
    stacked = x_overlap >= 0.28 and gap_y <= max(24, int(4.2 * max(ah, bh)))
    nearby = gap_x <= max(12, int(0.35 * max(aw, bw))) and gap_y <= max(12, int(0.35 * max(ah, bh)))
    return same_line or stacked or nearby


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-6)


def dedupe_boxes(boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.82) -> List[Tuple[int, int, int, int]]:
    ordered = sorted(
        {tuple(map(int, b)) for b in boxes},
        key=lambda b: ((b[2] - b[0]) * (b[3] - b[1]), b[0], b[1]),
        reverse=True,
    )
    unique: List[Tuple[int, int, int, int]] = []
    for box in ordered:
        if any(_box_iou(box, other) >= iou_threshold for other in unique):
            continue
        unique.append(box)
    return unique


def tighten_candidate_box(
    gray: np.ndarray,
    box: Tuple[int, int, int, int],
    image_shape: Sequence[int],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = clip_box(box, image_shape)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return x1, y1, x2, y2

    dark = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    light = cv2.adaptiveThreshold(255 - crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    mask = cv2.bitwise_or(dark, light)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))

    inner_boxes = extract_component_boxes(mask)
    if not inner_boxes:
        return x1, y1, x2, y2

    rx1 = min(b[0] for b in inner_boxes)
    ry1 = min(b[1] for b in inner_boxes)
    rx2 = max(b[2] for b in inner_boxes)
    ry2 = max(b[3] for b in inner_boxes)

    crop_h, crop_w = crop.shape[:2]
    rel_area = ((rx2 - rx1) * (ry2 - ry1)) / float(max(1, crop_h * crop_w))
    if rel_area < 0.02 or rel_area > 0.95:
        return x1, y1, x2, y2

    pad_x = max(2, int(round((rx2 - rx1) * 0.08)))
    pad_y = max(2, int(round((ry2 - ry1) * 0.16)))
    tightened = (
        x1 + max(0, rx1 - pad_x),
        y1 + max(0, ry1 - pad_y),
        x1 + min(crop_w, rx2 + pad_x),
        y1 + min(crop_h, ry2 + pad_y),
    )
    return clip_box(tightened, image_shape)


def craft_is_available() -> bool:
    return Crafter is not None


@lru_cache(maxsize=1)
def get_crafter():  # pragma: no cover - model init depends on runtime
    if Crafter is None:
        raise RuntimeError("pycrafter не установлен. Добавьте pycrafter в зависимости.")
    long_size = int(os.getenv("BOTTLE_CRAFT_LONG_SIZE", "1280"))
    text_threshold = float(os.getenv("BOTTLE_CRAFT_TEXT_THRESHOLD", "0.42"))
    link_threshold = float(os.getenv("BOTTLE_CRAFT_LINK_THRESHOLD", "0.18"))
    low_text = float(os.getenv("BOTTLE_CRAFT_LOW_TEXT", "0.18"))
    refiner = os.getenv("BOTTLE_CRAFT_REFINER", "0").lower() in {"1", "true", "yes", "on"}
    return Crafter(
        output_dir=None,
        rectify=False,
        export_extra=False,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        long_size=long_size,
        refiner=refiner,
        crop_type="box",
    )


def normalize_craft_boxes(prediction, image_shape: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for item in prediction.get("boxes", []):
        arr = np.asarray(item, dtype=np.float32).reshape(-1, 2)
        if arr.size == 0:
            continue
        x1, y1 = np.min(arr, axis=0)
        x2, y2 = np.max(arr, axis=0)
        boxes.append(clip_box((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), image_shape))
    return boxes


def filter_craft_boxes(
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Sequence[int],
) -> List[Tuple[int, int, int, int]]:
    h, w = int(image_shape[0]), int(image_shape[1])
    filtered: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        x1, y1, x2, y2 = clip_box(box, image_shape)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = (bw * bh) / float(max(1, h * w))
        aspect = bw / float(bh + 1e-6)
        if bw < 8 or bh < 6:
            continue
        if area_ratio > 0.16:
            continue
        if aspect < 0.18 or aspect > 30.0:
            continue
        filtered.append((x1, y1, x2, y2))
    return dedupe_boxes(filtered, iou_threshold=0.80)


def build_craft_primary_view(image: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
    balanced = gray_world_white_balance(image)
    glare_suppressed = inpaint_for_detection(balanced, glare_mask)
    detection_gray = build_detection_gray(glare_suppressed, glare_mask)
    return cv2.cvtColor(detection_gray, cv2.COLOR_GRAY2RGB)


def build_craft_secondary_view(image: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
    balanced = gray_world_white_balance(image)
    glare_suppressed = inpaint_for_detection(balanced, glare_mask)
    return cv2.cvtColor(glare_suppressed, cv2.COLOR_BGR2RGB)


def detect_craft_candidate_boxes(
    image: np.ndarray,
    glare_mask: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    crafter = get_crafter()
    boxes: List[Tuple[int, int, int, int]] = []

    primary_view = build_craft_primary_view(image, glare_mask)
    prediction = crafter(primary_view)
    boxes.extend(filter_craft_boxes(normalize_craft_boxes(prediction, image.shape), image.shape))

    if len(boxes) < 2:
        secondary_view = build_craft_secondary_view(image, glare_mask)
        prediction = crafter(secondary_view)
        boxes.extend(filter_craft_boxes(normalize_craft_boxes(prediction, image.shape), image.shape))

    return dedupe_boxes(boxes, iou_threshold=0.78)


def build_grouped_boxes_from_raw(
    boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    proposals: List[Tuple[int, int, int, int]] = list(boxes)
    ordered = sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2.0, b[0]))

    for idx, seed in enumerate(ordered):
        group = [seed]
        for jdx, box in enumerate(ordered):
            if idx == jdx:
                continue
            if boxes_are_related(seed, box):
                group.append(box)
        if len(group) >= 2:
            proposals.append(
                (
                    min(b[0] for b in group),
                    min(b[1] for b in group),
                    max(b[2] for b in group),
                    max(b[3] for b in group),
                )
            )

    return dedupe_boxes(proposals, iou_threshold=0.78)


def choose_best_date_cluster(
    boxes: List[Tuple[int, int, int, int]],
    gray: np.ndarray,
    glare_mask: np.ndarray,
    image_shape: Sequence[int],
    source: str = "heuristic",
) -> Tuple[Tuple[int, int, int, int] | None, List[ScoredCandidate]]:
    if not boxes:
        return None, []

    scored = [score_candidate(box, gray, glare_mask, image_shape, source=source) for box in boxes]
    scored = [item for item in scored if item.score > 0.0]
    scored.sort(key=lambda item: item.score, reverse=True)
    if not scored:
        return None, []

    best_box = scored[0].box
    best_score = scored[0].score

    for item in scored[1:12]:
        if item.score < scored[0].score * 0.48:
            continue
        if not boxes_are_related(best_box, item.box):
            continue
        merged = merge_box_pair(best_box, item.box)
        merged = tighten_candidate_box(gray, merged, image_shape)
        merged_score = score_candidate(merged, gray, glare_mask, image_shape, source=f"{source}_merged")
        if merged_score.area_ratio > 0.24:
            continue
        if merged_score.aspect_ratio < 0.72:
            continue
        if merged_score.score >= best_score * 0.84 or merged_score.score > best_score + 0.08:
            best_box = merged
            best_score = max(best_score, merged_score.score)

    image_h = int(image_shape[0])
    for item in scored[1:10]:
        if item.score < scored[0].score * 0.72:
            continue
        merged = merge_box_pair(best_box, item.box)
        mx1, my1, mx2, my2 = merged
        merged_h = max(1, my2 - my1)
        merged_score = score_candidate(merged, gray, glare_mask, image_shape, source=f"{source}_stacked")
        x_overlap = _overlap_ratio_1d(best_box[0], best_box[2], item.box[0], item.box[2])
        if x_overlap < 0.22:
            continue
        if merged_h > int(image_h * 0.34):
            continue
        if merged_score.area_ratio > 0.18:
            continue
        if merged_score.score >= best_score * 0.62 or item.score >= scored[0].score * 0.85:
            best_box = merged
            best_score = max(best_score, merged_score.score)

    best_box = tighten_candidate_box(gray, best_box, image_shape)
    scored.append(score_candidate(best_box, gray, glare_mask, image_shape, source=f"{source}_final"))
    scored.sort(key=lambda item: item.score, reverse=True)

    unique: List[ScoredCandidate] = []
    for item in scored:
        if any(_box_iou(item.box, other.box) >= 0.86 for other in unique):
            continue
        unique.append(item)
        if len(unique) >= 6:
            break

    return best_box, unique


def detect_text_roi(
    image: np.ndarray,
    padding_ratio: float = 0.08,
    detector_backend: DetectorBackend = "craft",
) -> Tuple[CropBox, bool, int, float, List[Dict[str, float | int | str]], float, str, bool, bool]:
    original_h, original_w = image.shape[:2]
    detection_image, det_scale = resize_with_max_side(image, max_side=1800)
    glare_mask = build_glare_mask(detection_image)
    detection_gray = build_detection_gray(detection_image, glare_mask)
    glare_ratio = float(np.mean(glare_mask > 0))

    boxes: List[Tuple[int, int, int, int]] = []
    backend_used = detector_backend
    fallback_used = False
    craft_used = False

    if detector_backend == "craft":
        try:
            craft_boxes = detect_craft_candidate_boxes(detection_image, glare_mask)
            craft_used = bool(craft_boxes)
            if craft_boxes:
                boxes = build_grouped_boxes_from_raw(craft_boxes)
                backend_used = "craft"
            else:
                fallback_used = True
                backend_used = "heuristic_fallback"
        except Exception:
            fallback_used = True
            backend_used = "heuristic_fallback"

    if not boxes:
        component_boxes = build_component_candidate_boxes(detection_gray)
        line_boxes = build_line_proposals(component_boxes, detection_gray.shape)
        morph_boxes = build_morph_candidate_boxes(detection_gray)
        boxes = dedupe_boxes(component_boxes + line_boxes + morph_boxes, iou_threshold=0.78)
        if detector_backend != "craft":
            backend_used = "heuristic"

    candidate_count = len(boxes)
    chosen_box, scored_candidates = choose_best_date_cluster(
        boxes,
        detection_gray,
        glare_mask,
        detection_image.shape,
        source=backend_used,
    )

    if chosen_box is None:
        full = CropBox(0, 0, original_w, original_h)
        return full, False, 0, glare_ratio, [], 0.0, backend_used, fallback_used, craft_used

    chosen_box = tighten_candidate_box(detection_gray, chosen_box, detection_image.shape)
    scaled_box = scale_box(chosen_box, det_scale)
    scaled_box = clip_box(scaled_box, image.shape)
    crop = pad_box(scaled_box, image.shape, padding_ratio=padding_ratio)
    crop_area_ratio = crop.area / float(max(1, original_h * original_w))
    confidence = float(scored_candidates[0].score) if scored_candidates else 0.0

    top_candidates: List[Dict[str, float | int | str]] = []
    for item in scored_candidates[:3]:
        bx1, by1, bx2, by2 = clip_box(scale_box(item.box, det_scale), image.shape)
        top_candidates.append(
            {
                "x1": int(bx1),
                "y1": int(by1),
                "x2": int(bx2),
                "y2": int(by2),
                "score": round(float(item.score), 4),
                "component_count": int(item.component_count),
                "glare_ratio": round(float(item.glare_ratio), 4),
                "area_ratio": round(float(item.area_ratio), 5),
                "aspect_ratio": round(float(item.aspect_ratio), 4),
                "source": item.source,
            }
        )

    if crop_area_ratio < 0.0002 or crop_area_ratio > 0.60:
        full = CropBox(0, 0, original_w, original_h)
        return full, False, candidate_count, glare_ratio, top_candidates, confidence, backend_used, fallback_used, craft_used

    return crop, True, candidate_count, glare_ratio, top_candidates, confidence, backend_used, fallback_used, craft_used



def estimate_rotation_angle(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return 0.0

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = float(rect[-1])
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < 0.8 or abs(angle) > 12.0:
        return 0.0
    return angle


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-6:
        return image.copy()
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def unsharp_mask(image: np.ndarray, sigma: float = 1.2, amount: float = 1.35) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def compress_highlights(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    compressed = np.where(
        gray_f > 0.78,
        0.78 + (gray_f - 0.78) * 0.32,
        gray_f,
    )
    return np.clip(compressed * 255.0, 0, 255).astype(np.uint8)


def prepare_luminance_for_digits(image: np.ndarray) -> np.ndarray:
    balanced = gray_world_white_balance(image)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    l = compress_highlights(l)
    normalized = normalize_illumination_gray(l, clip_limit=2.6)
    blended = cv2.addWeighted(l, 0.42, normalized, 0.58, 0)
    strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blended)
    return strong


def boost_for_digits(image: np.ndarray) -> np.ndarray:
    balanced = gray_world_white_balance(image)
    denoised = cv2.fastNlMeansDenoisingColored(balanced, None, 5, 5, 7, 21)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = compress_highlights(l)
    normalized_l = normalize_illumination_gray(l, clip_limit=2.4)
    blended_l = cv2.addWeighted(l, 0.35, normalized_l, 0.65, 0)
    blended_l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blended_l)
    improved = cv2.cvtColor(cv2.merge([blended_l, a, b]), cv2.COLOR_LAB2BGR)

    improved = cv2.bilateralFilter(improved, d=5, sigmaColor=35, sigmaSpace=35)
    improved = unsharp_mask(improved, sigma=1.0, amount=1.15)
    return improved


def build_high_contrast_variant(image: np.ndarray) -> np.ndarray:
    gray = prepare_luminance_for_digits(image)
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    gray = cv2.createCLAHE(clipLimit=4.2, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.convertScaleAbs(gray, alpha=1.18, beta=-8)
    gray = unsharp_mask(gray, sigma=0.9, amount=0.42)
    return gray


def score_binary_quality(binary_inv: np.ndarray) -> float:
    comp_count, alignment, consistency, gap_consistency, line_score = analyze_binary_components(binary_inv)
    ink_ratio = float(np.mean(binary_inv > 0))
    ink_score = triangle_pref(ink_ratio, 0.02, 0.16, 0.38)
    return (
        (line_score * 1.4)
        + min(1.0, comp_count / 12.0) * 0.45
        + ink_score * 0.3
        + alignment * 0.2
        + consistency * 0.15
        + gap_consistency * 0.1
    )


def build_bw_variant(image: np.ndarray) -> np.ndarray:
    gray = prepare_luminance_for_digits(image)
    gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    dark = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    light = cv2.adaptiveThreshold(
        255 - gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    dark = cv2.medianBlur(dark, 3)
    light = cv2.medianBlur(light, 3)

    chosen = dark if score_binary_quality(dark) >= score_binary_quality(light) else light
    bw = 255 - chosen
    bw = ensure_dark_text_on_light(bw)
    bw = remove_border_artifacts(bw)
    return bw



def process_image(
    file_bytes: bytes,
    crop_padding_ratio: float = 0.08,
    min_side_after_crop: int = 1200,
    max_side_after_crop: int = 2600,
    detector_backend: DetectorBackend = "craft",
) -> ProcessResult:
    image = decode_image(file_bytes)
    original_h, original_w = image.shape[:2]

    crop_box, crop_found, candidate_boxes, glare_ratio, top_candidates, detection_confidence, backend_used, fallback_used, craft_used = detect_text_roi(
        image,
        padding_ratio=crop_padding_ratio,
        detector_backend=detector_backend,
    )
    crop_bgr = image[crop_box.y1:crop_box.y2, crop_box.x1:crop_box.x2].copy()

    if crop_bgr.size == 0:
        crop_bgr = image.copy()
        crop_box = CropBox(0, 0, original_w, original_h)
        crop_found = False

    cropped = crop_bgr.copy()
    cropped, upscale_factor = resize_with_min_side(
        cropped,
        min_side=min_side_after_crop,
        max_side=max_side_after_crop,
    )

    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    angle = estimate_rotation_angle(gray_crop)
    rotated = rotate_bound(cropped, -angle) if abs(angle) > 0.0 else cropped

    improved = boost_for_digits(rotated)
    high_contrast = build_high_contrast_variant(improved)
    bw = build_bw_variant(improved)

    metadata = ProcessMetadata(
        original_width=original_w,
        original_height=original_h,
        crop_box={"x1": int(crop_box.x1), "y1": int(crop_box.y1), "x2": int(crop_box.x2), "y2": int(crop_box.y2)},
        crop_found=crop_found,
        upscale_factor=round(float(upscale_factor), 4),
        candidate_boxes=int(candidate_boxes),
        rotated_degrees=round(float(angle), 3),
        glare_ratio=round(float(glare_ratio), 4),
        detection_confidence=round(float(detection_confidence), 4),
        top_candidates=top_candidates,
        detector_requested=detector_backend,
        detector_backend=backend_used,
        detector_fallback_used=bool(fallback_used),
        craft_available=bool(craft_is_available()),
    )

    return ProcessResult(
        crop_bgr=crop_bgr,
        improved_bgr=improved,
        bw=bw,
        high_contrast=high_contrast,
        metadata=metadata,
    )
