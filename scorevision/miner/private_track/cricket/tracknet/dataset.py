"""
TrackNet dataset: turns annotated delivery folders into (frame-stack, heatmaps).

Each sample is a sliding window of `n_frames` consecutive frames from one delivery
task folder (Annotation/<batch>/<video>_d<id>/NNNNNN.jpg) plus the ball heatmaps
for those frames, decoded from a CVAT 'for video 1.1' XML export.

Expected layout per task:
    <task_dir>/000123.jpg ...           (consecutive frames; name = abs frame idx)
    <task_dir>/annotations.xml          (CVAT export; frames are 0-indexed within task)

CVAT exports number frames 0..N-1 in import order. We map them to the sorted
on-disk frames, so CVAT-frame i == i-th sorted image in the task folder.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap import gaussian_heatmap
from .cvat_io import parse_cvat_video_xml


class DeliveryBallDataset(Dataset):
    def __init__(self, batch_dir, n_frames: int = 3, out_hw=(288, 512),
                 sigma: float = 3.0, ann_name: str = "annotations.xml", augment: bool = False):
        self.n = n_frames
        self.H, self.W = out_hw
        self.sigma = sigma
        self.augment = augment
        self.samples: list[tuple[Path, list[Path], list]] = []

        # accept one dir or a list of base dirs; a task = any subdir with an
        # annotations.xml (name-agnostic: works for P1_V12_d6, t20_c059_36, kjs_d0 ...)
        bases = [batch_dir] if isinstance(batch_dir, (str, Path)) else list(batch_dir)
        for base in bases:
            for task in sorted(p for p in Path(base).iterdir() if p.is_dir()):
                ann = task / ann_name
                if not ann.exists():
                    continue  # unlabeled task -> skip (used only for inference)
                frames = sorted(task.glob("*.jpg"))
                if len(frames) < n_frames:
                    continue
                labels = parse_cvat_video_xml(ann)  # {cvat_frame -> BallLabel}
                for i in range(len(frames) - n_frames + 1):
                    self.samples.append((task, frames[i:i + n_frames],
                                         [labels.get(i + k) for k in range(n_frames)]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        task, frame_paths, labels = self.samples[idx]
        chans = []
        scale_done = None
        heatmaps = np.zeros((self.n, self.H, self.W), dtype=np.float32)
        for k, (fp, lb) in enumerate(zip(frame_paths, labels)):
            img = cv2.imread(str(fp))
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (self.W, self.H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chans.append(img)
            if lb is not None and lb.x is not None:
                sx, sy = self.W / w0, self.H / h0
                heatmaps[k] = gaussian_heatmap(self.H, self.W, lb.x * sx, lb.y * sy, self.sigma)
        if self.augment:
            chans = augment_frames(chans)  # same photometric transform for all n frames
        x = np.concatenate([c.transpose(2, 0, 1) for c in chans], axis=0)  # (3*n,H,W)
        return torch.from_numpy(x), torch.from_numpy(heatmaps)


def augment_frames(frames):
    """Strong PHOTOMETRIC augmentation, SAME params across the n frames (temporal
    consistency). No geometry (ball position unchanged). Targets the day/red ->
    night/white domain gap: brightness/gamma/contrast, hue+desaturation+grayscale
    (break colour reliance for white ball), downscale/blur/jpeg/noise (broadcast)."""
    R = np.random
    bright = R.uniform(0.40, 1.45)
    gamma = R.uniform(0.55, 1.9)
    contrast = R.uniform(0.6, 1.5)
    hue_shift = R.uniform(-30, 30)            # degrees
    sat_scale = R.uniform(0.0, 1.3)           # can desaturate toward gray/white
    gains = R.uniform(0.8, 1.2, 3)
    do_gray = R.random() < 0.30
    noise_sd = R.uniform(0.0, 0.045)
    do_blur = R.random() < 0.4; ksize = int(R.choice([3, 5, 7, 9])); angle = R.uniform(0, 180)
    do_jpeg = R.random() < 0.4; q = int(R.randint(28, 80))
    do_down = R.random() < 0.4; ds = R.uniform(0.45, 0.95)
    kern = None
    if do_blur:
        kern = np.zeros((ksize, ksize), np.float32); kern[ksize // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((ksize / 2 - .5, ksize / 2 - .5), angle, 1.0)
        kern = cv2.warpAffine(kern, M, (ksize, ksize)); kern /= kern.sum() + 1e-9
    out = []
    for img in frames:
        a = np.clip(img * bright, 0, 1)
        a = np.clip((a - 0.5) * contrast + 0.5, 0, 1)
        a = np.power(a, gamma, dtype=np.float32)
        hsv = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 360.0
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 1)
        a = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if do_gray:
            a = np.repeat(a.mean(2, keepdims=True), 3, axis=2)
        a = np.clip(a * gains, 0, 1).astype(np.float32)
        if do_down:
            h, w = a.shape[:2]
            a = cv2.resize(cv2.resize(a, (max(1, int(w * ds)), max(1, int(h * ds)))), (w, h))
        if kern is not None:
            a = cv2.filter2D(a, -1, kern)
        if noise_sd > 0:
            a = np.clip(a + R.normal(0, noise_sd, a.shape), 0, 1).astype(np.float32)
        if do_jpeg:
            enc = cv2.imencode(".jpg", (a[..., ::-1] * 255).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, q])[1]
            a = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
        out.append(np.ascontiguousarray(a, dtype=np.float32))
    return out


def weighted_bce_loss(pred: torch.Tensor, target: torch.Tensor, pos_w: float = 50.0):
    """Heatmaps are mostly zero; up-weight positive (ball) pixels."""
    eps = 1e-6
    pred = pred.clamp(eps, 1 - eps)
    w = 1.0 + pos_w * target
    return -(w * (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))).mean()


if __name__ == "__main__":
    import sys
    d = DeliveryBallDataset(sys.argv[1] if len(sys.argv) > 1 else "Annotation/ball_batch1")
    print("samples:", len(d))
    if len(d):
        x, y = d[0]
        print("x", tuple(x.shape), "y", tuple(y.shape), "y.max", float(y.max()))
