"""E2E-Spot SoccerNet-v2 action spotting inference module.

Wraps the pretrained E2E-Spot soccer_challenge_rny008gsm_gru_rgb model
to detect rare high-value events (goal, foul, shot, substitution, etc.)
from video frames sampled at 2 FPS.
"""

import json
import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import timm  # noqa: F401 — needed by model internals
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = getLogger(__name__)

MAX_GRU_HIDDEN_DIM = 768
DEFAULT_MODELS_DIR = Path("/app/models")
SPOT_REPO_DIR = Path("/app/spot")

# ---------------------------------------------------------------------------
# Model definition (must match checkpoint structure exactly)
# ---------------------------------------------------------------------------

class _FCPrediction(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        b, t, _ = x.shape
        return self._fc_out(x.reshape(b * t, -1)).view(b, t, -1)


class _GRUPrediction(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
        )
        self._fc_out = _FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class _E2EModelImpl(nn.Module):
    def __init__(self, num_classes, clip_len):
        super().__init__()
        features = timm.create_model("regnety_008", pretrained=False)
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()

        self._require_clip_len = clip_len

        # GSM temporal shift — import from spot repo
        try:
            from model.shift import make_temporal_shift
            make_temporal_shift(features, clip_len, is_gsm=True)
        except Exception as e:
            logger.warning("GSM temporal shift unavailable (%s); using plain backbone", e)

        self._features = features
        self._feat_dim = feat_dim

        hidden_dim = min(feat_dim, MAX_GRU_HIDDEN_DIM)
        self._pred_fine = _GRUPrediction(feat_dim, num_classes, hidden_dim, num_layers=1)

    def forward(self, x):
        b, true_t, c, h, w = x.shape
        t = true_t

        if true_t < self._require_clip_len:
            x = F.pad(x, (0,) * 7 + (self._require_clip_len - true_t,))
            t = self._require_clip_len

        im_feat = self._features(
            x.view(-1, c, h, w),
        ).reshape(b, t, self._feat_dim)

        if true_t != t:
            im_feat = im_feat[:, :true_t, :]

        return self._pred_fine(im_feat)


# ---------------------------------------------------------------------------
# SoccerNet-v2 → TurboVision class mapping
# ---------------------------------------------------------------------------

SOCCERNET_V2_CLASSES = [
    "background",          # 0
    "Penalty",             # 1
    "Kick-off",            # 2
    "Goal",                # 3
    "Substitution",        # 4
    "Offside",             # 5
    "Shots on target",     # 6
    "Shots off target",    # 7
    "Clearance",           # 8
    "Ball out of play",    # 9
    "Throw-in",            # 10
    "Foul",                # 11
    "Indirect free-kick",  # 12
    "Direct free-kick",    # 13
    "Corner",              # 14
    "Yellow card",         # 15
    "Red card",            # 16
    "Yellow->red card",    # 17
]

# Maps SoccerNet class index → TurboVision action.
# Confidence is the raw model probability (no multiplier).
SOCCERNET_TO_TURBOVISION: dict[int, str] = {
    1:  "shot",              # Penalty → shot
    2:  "pass",              # Kick-off → pass
    3:  "goal",              # Goal → goal
    4:  "substitution",      # Substitution → substitution
    6:  "shot",              # Shots on target → shot
    7:  "shot",              # Shots off target → shot
    8:  "clearance",         # Clearance → clearance
    9:  "ball_out_of_play",  # Ball out of play
    10: "ball_out_of_play",  # Throw-in → ball_out_of_play
    11: "foul",              # Foul → foul
    12: "foul",              # Indirect free-kick → foul
    13: "foul",              # Direct free-kick → foul
}


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpotPrediction:
    """Single action prediction from E2E-Spot."""
    frame_25fps: int   # frame number in 25 FPS space (for validator)
    action: str        # TurboVision action name
    confidence: float  # combined model prob × mapping confidence


# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_MODEL: _E2EModelImpl | None = None
_CONFIG: dict | None = None


def _get_spot_model_dir() -> Path:
    raw = os.environ.get("SPOT_MODELS_DIR")
    if raw:
        return Path(raw).expanduser()
    models_dir = Path(os.environ.get("MINER_MODELS_DIR", str(DEFAULT_MODELS_DIR)))
    spot_dir = models_dir / "e2e-spot"
    if spot_dir.exists():
        return spot_dir
    return models_dir


def _load_spot_model() -> tuple[_E2EModelImpl, dict]:
    global _MODEL, _CONFIG
    if _MODEL is not None and _CONFIG is not None:
        return _MODEL, _CONFIG

    model_dir = _get_spot_model_dir()
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "checkpoint_149.pt"

    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"E2E-Spot model not found at {model_dir}. "
            f"Need config.json and checkpoint_149.pt"
        )

    with open(config_path) as f:
        config = json.load(f)

    # Ensure spot repo is in sys.path for GSM import
    import sys
    spot_repo = Path(os.environ.get("SPOT_REPO_DIR", str(SPOT_REPO_DIR)))
    if spot_repo.exists() and str(spot_repo) not in sys.path:
        sys.path.insert(0, str(spot_repo))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _E2EModelImpl(
        num_classes=config["num_classes"] + 1,
        clip_len=config["clip_len"],
    )

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "") if k.startswith("module.") else k] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()

    _MODEL = model
    _CONFIG = config
    logger.info(
        "E2E-Spot loaded from %s on %s (clip_len=%d, classes=%d)",
        model_dir, device, config["clip_len"], config["num_classes"],
    )
    return _MODEL, _CONFIG


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def predict_with_spot(video_path: Path) -> list[SpotPrediction]:
    """Run E2E-Spot on a video and return mapped TurboVision predictions."""

    min_confidence = _env_float("SPOT_MIN_CONFIDENCE", 0.30)
    sample_fps = _env_float("SPOT_SAMPLE_FPS", 2.0)

    try:
        model, config = _load_spot_model()
    except FileNotFoundError as e:
        logger.warning("E2E-Spot unavailable: %s", e)
        return []

    device = next(model.parameters()).device
    clip_len = config["clip_len"]

    # Extract frames at 2 FPS
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video for E2E-Spot: %s", video_path)
        return []

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    stride = max(1, round(video_fps / sample_fps))

    frames = []
    frame_indices_25fps = []  # original frame numbers in 25fps space
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = 224.0 / h
            new_w = int(round(w * scale))
            frame = cv2.resize(frame, (new_w, 224))
            frames.append(frame)
            # Convert to 25fps frame number for validator
            frame_indices_25fps.append(round(idx * 25.0 / video_fps))
        idx += 1
    cap.release()

    if not frames:
        return []

    # Preprocess
    x = torch.from_numpy(np.stack(frames)).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.to(device)

    # Run inference clip by clip
    all_probs = []
    num_frames = x.shape[0]
    with torch.no_grad():
        for i in range(0, num_frames, clip_len):
            chunk = x[i:i + clip_len].unsqueeze(0)
            logits = model(chunk)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs[0].cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)

    # Extract predictions via mapping — use raw model probability as confidence
    predictions: list[SpotPrediction] = []
    for frame_idx in range(all_probs.shape[0]):
        for sn_cls, tv_action in SOCCERNET_TO_TURBOVISION.items():
            if sn_cls >= all_probs.shape[1]:
                continue
            prob = float(all_probs[frame_idx, sn_cls])
            if prob < min_confidence:
                continue

            frame_25fps = frame_indices_25fps[frame_idx] if frame_idx < len(frame_indices_25fps) else 0
            predictions.append(SpotPrediction(
                frame_25fps=frame_25fps,
                action=tv_action,
                confidence=prob,
            ))

    # Deduplicate: keep highest confidence per action within suppress window
    suppress_frames = 25  # 1 second at 25fps
    deduped: list[SpotPrediction] = []
    predictions.sort(key=lambda p: p.confidence, reverse=True)
    for pred in predictions:
        is_dup = any(
            d.action == pred.action and abs(d.frame_25fps - pred.frame_25fps) <= suppress_frames
            for d in deduped
        )
        if not is_dup:
            deduped.append(pred)

    deduped.sort(key=lambda p: p.frame_25fps)
    logger.info(
        "E2E-Spot: %d frames processed, %d raw predictions, %d after dedup (actions: %s)",
        all_probs.shape[0],
        len(predictions),
        len(deduped),
        {p.action: sum(1 for d in deduped if d.action == p.action) for p in deduped},
    )
    return deduped
