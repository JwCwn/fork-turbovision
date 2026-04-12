"""Standalone E2E-Spot inference test — POC only.

Run from ~/e2e-spot after cloning https://github.com/jhong93/spot and
https://github.com/jhong93/e2e-spot-models and downloading the
soccer_challenge_rny008gsm_gru_rgb checkpoint.

Usage:
    source spot/.venv/bin/activate
    python e2e_spot_test.py path/to/test_video.mp4
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).parent
SPOT_ROOT = REPO_ROOT / "spot"
sys.path.insert(0, str(SPOT_ROOT))

from model.shift import make_temporal_shift  # noqa: E402


MAX_GRU_HIDDEN_DIM = 768


class FCPrediction(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(
            x.reshape(batch_size * clip_len, -1)
        ).view(batch_size, clip_len, -1)


class GRUPrediction(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True,
        )
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class E2EModelImpl(nn.Module):
    def __init__(self, num_classes, clip_len):
        super().__init__()
        features = timm.create_model("regnety_008", pretrained=False)
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()

        self._require_clip_len = clip_len
        make_temporal_shift(features, clip_len, is_gsm=True)

        self._features = features
        self._feat_dim = feat_dim

        hidden_dim = min(feat_dim, MAX_GRU_HIDDEN_DIM)
        self._pred_fine = GRUPrediction(
            feat_dim, num_classes, hidden_dim, num_layers=1,
        )

    def forward(self, x):
        batch_size, true_clip_len, channels, height, width = x.shape
        clip_len = true_clip_len

        if true_clip_len < self._require_clip_len:
            x = F.pad(
                x, (0,) * 7 + (self._require_clip_len - true_clip_len,),
            )
            clip_len = self._require_clip_len

        im_feat = self._features(
            x.view(-1, channels, height, width),
        ).reshape(batch_size, clip_len, self._feat_dim)

        if true_clip_len != clip_len:
            im_feat = im_feat[:, :true_clip_len, :]

        return self._pred_fine(im_feat)


SOCCERNET_V2_CLASSES = [
    "background",
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card",
]


def extract_frames(video_path, target_height=224):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: fps={fps:.2f} total_frames={total}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = target_height / h
        new_w = int(round(w * scale))
        frame = cv2.resize(frame, (new_w, target_height))
        frames.append(frame)
    cap.release()
    return np.stack(frames), fps


def preprocess(frames):
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def main():
    if len(sys.argv) < 2:
        print("Usage: python e2e_spot_test.py <video.mp4>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Missing video: {video_path}")
        sys.exit(1)

    model_dir = REPO_ROOT / "e2e-spot-models" / "soccer_challenge_rny008gsm_gru_rgb"
    checkpoint_path = model_dir / "checkpoint_149.pt"
    config_path = model_dir / "config.json"
    if not checkpoint_path.exists():
        print(f"Missing checkpoint at {checkpoint_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)
    print(f"Config: {config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = E2EModelImpl(
        num_classes=config["num_classes"] + 1,
        clip_len=config["clip_len"],
    )

    state = torch.load(checkpoint_path, map_location=device)
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "") if k.startswith("module.") else k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    model.to(device)
    model.eval()
    print("Model loaded")

    frames, fps = extract_frames(video_path)
    print(f"Frames shape: {frames.shape}")
    x = preprocess(frames).to(device)
    print(f"Tensor shape: {x.shape}")

    clip_len = config["clip_len"]
    num_frames = x.shape[0]
    all_probs = []

    start = time.time()
    with torch.no_grad():
        for i in range(0, num_frames, clip_len):
            chunk = x[i:i + clip_len].unsqueeze(0)
            logits = model(chunk)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs[0].cpu().numpy())
    elapsed = time.time() - start
    all_probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(all_probs, axis=-1)
    max_probs = np.max(all_probs, axis=-1)

    print(f"Inference time: {elapsed:.2f}s ({num_frames/elapsed:.1f} fps)")
    print(f"Output shape: {all_probs.shape}")

    # Diagnostic 1: top-3 prediction per every 50 frames
    print("\n=== Top-3 per every 50 frames ===")
    for frame_idx in range(0, num_frames, 50):
        row = all_probs[frame_idx]
        top3 = np.argsort(row)[-3:][::-1]
        top3_str = ", ".join(
            f"{SOCCERNET_V2_CLASSES[c] if c < len(SOCCERNET_V2_CLASSES) else c}={row[c]:.3f}"
            for c in top3
        )
        print(f"  Frame {frame_idx}: {top3_str}")

    # Diagnostic 2: peak probability per class (no threshold)
    print("\n=== Per-class peak probability ===")
    peaks = []
    for cls_idx in range(1, len(SOCCERNET_V2_CLASSES)):
        col = all_probs[:, cls_idx]
        peak = float(col.max())
        peak_frame = int(col.argmax())
        peaks.append((peak, peak_frame, SOCCERNET_V2_CLASSES[cls_idx]))
    peaks.sort(reverse=True)
    for peak, peak_frame, name in peaks:
        marker = "***" if peak > 0.1 else ""
        print(f"  {name:25s} peak={peak:.4f} @ frame {peak_frame} {marker}")

    # Diagnostic 3: non-background argmax frames
    nonbg_frames = [
        (i, preds[i], max_probs[i])
        for i in range(num_frames)
        if preds[i] > 0
    ]
    print(f"\n=== Argmax != background: {len(nonbg_frames)}/{num_frames} frames ===")
    for i, cls, conf in nonbg_frames[:30]:
        name = SOCCERNET_V2_CLASSES[cls] if cls < len(SOCCERNET_V2_CLASSES) else f"class_{cls}"
        print(f"  Frame {i}: {name} (conf={conf:.3f})")

    # Diagnostic 4: overall background probability distribution
    bg_probs = all_probs[:, 0]
    print(
        f"\n=== Background prob stats: "
        f"min={bg_probs.min():.3f} avg={bg_probs.mean():.3f} max={bg_probs.max():.3f} ==="
    )


if __name__ == "__main__":
    main()
