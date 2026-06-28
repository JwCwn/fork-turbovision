"""Gaussian heatmap target generation for TrackNet (numpy/cv2, no torch)."""
from __future__ import annotations

import numpy as np


def gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 3.0) -> np.ndarray:
    """A single 2D Gaussian peak at (cx, cy) on an (h, w) map, values in [0,1].
    Returns all-zeros if the center is None/out of range (ball invisible)."""
    hm = np.zeros((h, w), dtype=np.float32)
    if cx is None or cy is None:
        return hm
    if not (0 <= cx < w and 0 <= cy < h):
        return hm
    # local window for efficiency (±3 sigma)
    r = int(3 * sigma + 1)
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return hm
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    gx = np.exp(-((xs - cx) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((ys - cy) ** 2) / (2 * sigma ** 2))
    hm[y0:y1, x0:x1] = np.outer(gy, gx)
    return hm


def heatmap_peak(hm: np.ndarray, thresh: float = 0.5) -> tuple[float, float, float] | None:
    """Decode a predicted heatmap -> (x, y, confidence) or None if below thresh."""
    if hm.size == 0:
        return None
    idx = int(np.argmax(hm))
    y, x = np.unravel_index(idx, hm.shape)
    conf = float(hm[y, x])
    if conf < thresh:
        return None
    # sub-pixel refinement via local centroid in a small window
    r = 2
    y0, y1 = max(0, y - r), min(hm.shape[0], y + r + 1)
    x0, x1 = max(0, x - r), min(hm.shape[1], x + r + 1)
    patch = hm[y0:y1, x0:x1]
    s = patch.sum()
    if s > 1e-6:
        ys = np.arange(y0, y1)
        xs = np.arange(x0, x1)
        cy = float((patch.sum(1) * ys).sum() / s)
        cx = float((patch.sum(0) * xs).sum() / s)
        return cx, cy, conf
    return float(x), float(y), conf


if __name__ == "__main__":
    hm = gaussian_heatmap(288, 512, 100.0, 50.0, sigma=3.0)
    assert abs(hm.max() - 1.0) < 1e-5
    peak = heatmap_peak(hm)
    assert peak is not None and abs(peak[0] - 100) < 1 and abs(peak[1] - 50) < 1
    assert heatmap_peak(gaussian_heatmap(288, 512, None, None)) is None
    print("heatmap self-check passed; peak=", peak)
