"""Camera-invariant width-field regressor (sim-trained).

The bundle reconstructs a full 3D camera per clip and degenerates on single-camera
broadcasts (stump_y comes out as garbage that the range guard nulls -> 0). This
regressor instead learns the monocular field mapping from a physics simulator and
consumes a CAMERA-INVARIANT feature: the ball track transformed into PITCH
coordinates via the 4 pitch-corner homography. That removes the camera variation
that made per-clip solving fragile, so it produces a robust, in-range stump_y on
essentially every clip where the pitch edges are detected (~all of them).

Trained + validated offline (Scripts/sim); here we only run inference. Currently we
trust STUMP_Y (the well-conditioned lateral field, 17% of the score); swing/deviation
are emitted by the bundle until they are independently validated.
"""
from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn as nn

# must match Scripts/sim/featurize.py exactly (same order/scale the net trained on)
CORNER_ORDER = ["pitch_left_far", "pitch_right_far", "pitch_left_near", "pitch_right_near"]
PITCH_WORLD = np.array([[0.0, -1.525], [0.0, 1.525], [20.12, -1.525], [20.12, 1.525]], float)
TARGETS = ["stump_y", "swing_angle", "deviation"]
TARGET_SCALE = {"stump_y": 3.0, "swing_angle": 8.0, "deviation": 8.0}
N_BALL = 16
FEAT_DIM = 2 * N_BALL


def featurize_pr(ball: dict, corners: dict, n_ball: int = N_BALL):
    """Ball track in PITCH coords (camera-invariant). ball: {frame:(u,v)}; corners:
    name->(u,v) for the 4 CORNER_ORDER points. Returns (2*n_ball,) float32 or None."""
    if not all(c in corners and corners[c] is not None for c in CORNER_ORDER):
        return None
    frames = sorted(ball)
    if len(frames) < 2:
        return None
    img = np.array([corners[c] for c in CORNER_ORDER], float)
    H, _ = cv2.findHomography(img, PITCH_WORLD)
    if H is None:
        return None
    uv = np.array([ball[f] for f in frames], float)
    w = (H @ np.hstack([uv, np.ones((len(uv), 1))]).T).T
    z = np.where(np.abs(w[:, 2]) < 1e-9, 1e-9, w[:, 2])
    xy = w[:, :2] / z[:, None]
    xy[:, 0] /= 20.12; xy[:, 1] /= 3.05
    fr = np.array(frames, float)
    grid = np.linspace(fr[0], fr[-1], n_ball)
    bx = np.interp(grid, fr, xy[:, 0]); by = np.interp(grid, fr, xy[:, 1])
    return np.stack([bx, by], 1).ravel().astype(np.float32)


class _MLP(nn.Module):
    def __init__(self, d_in, d_out, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


class WidthRegressor:
    def __init__(self, ckpt, device="cpu"):
        sd = torch.load(str(ckpt), map_location=device)
        self.device = device
        self.net = _MLP(sd.get("feat_dim", FEAT_DIM), len(TARGETS)).to(device).eval()
        self.net.load_state_dict(sd["model"])

    def predict(self, ball: dict, corners: dict):
        """-> {stump_y, swing_angle, deviation} in real units, or None if the pitch
        corners / ball are insufficient for the camera-invariant feature."""
        f = featurize_pr(ball, corners)
        if f is None:
            return None
        with torch.no_grad():
            out = self.net(torch.from_numpy(f)[None].to(self.device))[0].cpu().numpy()
        return {t: float(out[j] * TARGET_SCALE[t]) for j, t in enumerate(TARGETS)}
