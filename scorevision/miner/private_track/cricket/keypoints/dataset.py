"""
Stump+pitch keypoint dataset: single frame -> K Gaussian heatmaps (one per named
keypoint). Trains a detector that auto-locates the calibration keypoints on a
challenge clip at inference (replacing hand-labelling for the physics stage).

Keypoints (K=10): batter-end stumps (3 bases + 3 tops) + pitch side-edges (4).
Photometric augmentation reused (day/red -> night/white robustness).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from scorevision.miner.private_track.cricket.tracknet.heatmap import gaussian_heatmap, heatmap_peak
from scorevision.miner.private_track.cricket.tracknet.dataset import augment_frames
from scorevision.miner.private_track.cricket.check_stumps import load_stumps_per_frame

KP_NAMES = ["bs_left_base", "bs_mid_base", "bs_right_base",
            "bs_left_top", "bs_mid_top", "bs_right_top",
            "pitch_left_far", "pitch_left_near", "pitch_right_far", "pitch_right_near"]
KP_INDEX = {n: i for i, n in enumerate(KP_NAMES)}


class KeypointDataset(Dataset):
    def __init__(self, bases, out_hw=(288, 512), sigma=3.0, augment=False,
                 ann_name="annotations.xml"):
        self.H, self.W = out_hw
        self.sigma = sigma
        self.augment = augment
        self.samples = []  # (frame_path, {kp:(x,y)}, (w0,h0))
        bases = [bases] if isinstance(bases, (str, Path)) else list(bases)
        for base in bases:
            for task in sorted(p for p in Path(base).iterdir() if p.is_dir()):
                ann = task / ann_name
                if not ann.exists():
                    continue
                per_frame = load_stumps_per_frame(ann)
                if not per_frame:
                    continue
                frames = sorted(task.glob("*.jpg"))
                for i, fp in enumerate(frames):
                    kp = per_frame.get(i)  # cvat frame index = sorted position
                    if kp and any(n in kp for n in KP_NAMES):
                        self.samples.append((fp, kp))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, kp = self.samples[idx]
        img = cv2.imread(str(fp)); h0, w0 = img.shape[:2]
        img = cv2.cvtColor(cv2.resize(img, (self.W, self.H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.augment:
            img = augment_frames([img])[0]
        sx, sy = self.W / w0, self.H / h0
        hm = np.zeros((len(KP_NAMES), self.H, self.W), np.float32)
        for n, i in KP_INDEX.items():
            if n in kp:
                x, y = kp[n]
                hm[i] = gaussian_heatmap(self.H, self.W, x * sx, y * sy, self.sigma)
        x = img.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(hm)


def kp_accuracy(pred, target, tol_px=8.0):
    """Per-keypoint localization within tol (argmax vs GT peak). Returns (correct, total)."""
    import numpy as _np
    c = t = 0
    p = pred.detach().cpu().numpy(); g = target.detach().cpu().numpy()
    for b in range(p.shape[0]):
        for k in range(p.shape[1]):
            gt = heatmap_peak(g[b, k], thresh=0.5)
            if gt is None:
                continue
            t += 1
            yy, xx = _np.unravel_index(int(p[b, k].argmax()), p[b, k].shape)
            if ((xx - gt[0]) ** 2 + (yy - gt[1]) ** 2) ** 0.5 <= tol_px:
                c += 1
    return c, t


if __name__ == "__main__":
    import sys
    ds = KeypointDataset(sys.argv[1:] or ["Annotation/wb_ball", "Annotation/ball_batch1"])
    print("keypoint samples (labeled frames):", len(ds))
    if len(ds):
        x, y = ds[0]
        print("x", tuple(x.shape), "y", tuple(y.shape), "kp present in sample0:",
              int((y.amax(dim=(1, 2)) > 0.5).sum()))
