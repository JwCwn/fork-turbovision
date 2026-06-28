"""
TrackNet inference -> per-frame ball positions -> CVAT XML (for human correction).

Runs the model over the sliding 3-frame windows of each task folder, aggregates
the per-frame heatmap predictions (a frame appears in up to n_frames windows;
we average), decodes the peak, and writes a CVAT 'for video 1.1' XML per task so
the predictions can be imported back into CVAT and corrected (active learning).

Deterministic: FP32, deterministic cuDNN, eval mode, fixed resize. Same input ->
same output (spotcheck-safe).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

from .model import TrackNetV2
from .heatmap import heatmap_peak
from .cvat_io import BallLabel, write_cvat_video_xml


def _set_det():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_task(model, task_dir: Path, device: str, n_frames: int,
               out_hw=(288, 512), thresh: float = 0.8) -> tuple[dict[int, BallLabel], int, int]:
    frames = sorted(task_dir.glob("*.jpg"))
    if len(frames) < n_frames:
        return {}, 0, 0
    H, W = out_hw
    img0 = cv2.imread(str(frames[0]))
    h0, w0 = img0.shape[:2]

    # cache resized RGB tensors
    cache = []
    for fp in frames:
        im = cv2.resize(cv2.imread(str(fp)), (W, H))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cache.append(im.transpose(2, 0, 1))

    acc = defaultdict(list)  # cvat_frame -> list of heatmaps
    with torch.no_grad():
        for i in range(len(frames) - n_frames + 1):
            x = np.concatenate(cache[i:i + n_frames], axis=0)[None]
            xt = torch.from_numpy(x).to(device)
            y = model(xt)[0].cpu().numpy()  # (n_frames, H, W)
            for k in range(n_frames):
                acc[i + k].append(y[k])

    labels: dict[int, BallLabel] = {}
    sx, sy = w0 / W, h0 / H
    for cf in range(len(frames)):
        hm = np.mean(acc[cf], axis=0) if acc[cf] else np.zeros((H, W), np.float32)
        peak = heatmap_peak(hm, thresh=thresh)
        if peak is None:
            labels[cf] = BallLabel(cf, None, None, 0)
        else:
            labels[cf] = BallLabel(cf, peak[0] * sx, peak[1] * sy,
                                   1 if peak[2] >= 0.7 else 2)
    n_hit = sum(1 for v in labels.values() if v.x is not None)
    return labels, n_hit, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", required=True, help="dir of *_d* task folders to pseudo-label")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thresh", type=float, default=0.8)
    ap.add_argument("--out-name", default="pseudo.xml")
    args = ap.parse_args()

    _set_det()
    sd = torch.load(args.ckpt, map_location=args.device)
    n_frames = sd.get("n_frames", 3)
    model = TrackNetV2(n_frames=n_frames).to(args.device).eval()
    model.load_state_dict(sd["model"] if "model" in sd else sd)

    tasks = sorted(Path(args.batch).glob("*_d*"))
    tot_hit = tot = 0
    for task in tasks:
        if not task.is_dir():
            continue
        labels, n_hit, n = infer_task(model, task, args.device, n_frames, thresh=args.thresh)
        if not labels:
            continue
        img0 = cv2.imread(str(sorted(task.glob('*.jpg'))[0]))
        h0, w0 = img0.shape[:2]
        write_cvat_video_xml(task / args.out_name, labels, task_name=task.name,
                             width=w0, height=h0, n_frames=n)
        tot_hit += n_hit
        tot += n
        print(f"  {task.name}: {n_hit}/{n} frames with ball -> {args.out_name}")
    print(f"done: {tot_hit}/{tot} frames pseudo-labeled across {len(tasks)} tasks")


if __name__ == "__main__":
    main()
