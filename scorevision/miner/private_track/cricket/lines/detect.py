"""Pitch / return-crease LINE detector (occlusion-robust calibration reference).

A single-frame TrackNetV2(out_ch=6) predicts one line-ridge heatmap per crease.
For each frame we PCA-fit the ridge pixels into an endpoint pair, giving the same
2-endpoints-per-line correspondences the hand-labelled de-risk used. These lines
stay visible when the stumps are occluded, so they let geometry() calibrate clips
the stump path has to skip. Deterministic (FP32, deterministic cuDNN, fixed resize)
so it is spotcheck-safe.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import torch

from scorevision.miner.private_track.cricket.tracknet.model import TrackNetV2

# Order MUST match training (Scripts/lines/dataset.LINE_NAMES) -> heatmap channel.
LINE_NAMES = [
    "pitch_left_edge", "pitch_right_edge",
    "return_batter_left", "return_batter_right",
    "return_bowler_left", "return_bowler_right",
]
LINE_INDEX = {n: i for i, n in enumerate(LINE_NAMES)}


def extract_line(hm, sx, sy, thresh=0.4, min_pix=12):
    """Line-ridge heatmap -> (p0, p1) endpoints in ORIGINAL px, sorted by image-y.

    Threshold the ridge, PCA-fit a direction, and take the extreme projections as
    the segment ends. Returns None if too few ridge pixels fire (line absent)."""
    ys, xs = np.where(hm > thresh)
    if len(xs) < min_pix:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    mean = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mean)
    d = vt[0]
    t = (pts - mean) @ d
    p0 = mean + d * t.min()
    p1 = mean + d * t.max()
    e = sorted([p0 * [sx, sy], p1 * [sx, sy]], key=lambda p: p[1])
    return np.array(e[0]), np.array(e[1])


class LineDetector:
    def __init__(self, ckpt, device):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        sd = torch.load(str(ckpt), map_location=device)
        self.K = sd["out_ch"]
        self.m = TrackNetV2(n_frames=1, out_ch=self.K).to(device).eval()
        self.m.load_state_dict(sd["model"])
        self.device = device

    def detect(self, task, thresh=0.4):
        """Folder of *.jpg -> {frame_idx: {line_name: (p0, p1)}} in original px.

        frame_idx is the 0-based position in sorted(task.glob('*.jpg')) — the SAME
        indexing the ball detector uses — so line and ball observations share a
        frame axis without any re-keying."""
        frames = sorted(Path(task).glob("*.jpg"))
        if not frames:
            return {}
        h0, w0 = cv2.imread(str(frames[0])).shape[:2]
        sx, sy = w0 / 512.0, h0 / 288.0
        lines: dict[int, dict] = defaultdict(dict)
        with torch.no_grad():
            for i, fp in enumerate(frames):
                im = cv2.cvtColor(cv2.resize(cv2.imread(str(fp)), (512, 288)),
                                  cv2.COLOR_BGR2RGB)
                x = im.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
                hm = self.m(torch.from_numpy(x).to(self.device))[0].cpu().numpy()
                for name, ci in LINE_INDEX.items():
                    e = extract_line(hm[ci], sx, sy, thresh=thresh)
                    if e is not None:
                        lines[i][name] = e
        return dict(lines)
