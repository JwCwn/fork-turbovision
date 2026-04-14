"""lRomul's SoccerNet Ball Action Spotting 2023 (1st place) wrapper.

Uses the pretrained MultiDimStacker model to detect PASS and DRIVE events.
These map to TurboVision's pass/pass_received and take_on actions.

Model source: https://github.com/lRomul/ball-action-spotting
Pretrained weights: Google Drive — fold_0/model-006-0.864002.pth (~55MB)

Mount weights at /app/models/lromul/ via host volume at runtime.
Mount the repo's `src/models/` at /app/lromul_src/ for architecture import.
"""

import os
import sys
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import torch

logger = getLogger(__name__)

DEFAULT_MODELS_DIR = Path("/app/models")
LROMUL_SRC_DIR = Path(os.environ.get("LROMUL_SRC_DIR", "/app/lromul_src"))


# ---------------------------------------------------------------------------
# Model defaults — match config ball_finetune_long_004
# ---------------------------------------------------------------------------

MODEL_KWARGS = dict(
    model_name="tf_efficientnetv2_b0",
    num_classes=2,
    num_frames=33,
    stack_size=3,
    num_3d_blocks=4,
    num_3d_features=192,
    num_3d_stack_proj=256,
    index_2d_features=4,
    pretrained=False,
    expansion_3d_ratio=3,
    se_reduce_3d_ratio=24,
    drop_rate=0.2,
    drop_path_rate=0.2,
    act_layer="silu",
)

FRAME_STACK_STEP = 2
FRAME_WIDTH = 1280
FRAME_HEIGHT = 736
CLASS_NAMES = ["PASS", "DRIVE"]
VALIDATOR_FRAME_RATE = 25.0


@dataclass(frozen=True)
class LromulPrediction:
    frame_25fps: int
    action: str
    confidence: float


# ---------------------------------------------------------------------------
# Singleton model
# ---------------------------------------------------------------------------

_MODEL = None
_DEVICE: str | None = None


def _get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _get_checkpoint_path() -> Path:
    base = Path(os.environ.get("LROMUL_MODELS_DIR", str(DEFAULT_MODELS_DIR / "lromul")))
    # Prefer explicit file via env, else pick the first .pth in the directory
    explicit = os.environ.get("LROMUL_CHECKPOINT")
    if explicit:
        return Path(explicit).expanduser()
    candidates = sorted(base.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No lRomul checkpoint in {base}")
    return candidates[0]


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if LROMUL_SRC_DIR.exists() and str(LROMUL_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(LROMUL_SRC_DIR))

    from models.multidim_stacker import MultiDimStacker  # type: ignore

    ckpt_path = _get_checkpoint_path()
    device = _get_device()

    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    params = None
    if isinstance(state, dict):
        if "nn_state_dict" in state:
            weights = state["nn_state_dict"]
            params = state.get("params")
        elif "state_dict" in state:
            weights = state["state_dict"]
        else:
            weights = state
    else:
        weights = state

    model_kwargs = dict(MODEL_KWARGS)
    if params and isinstance(params, dict):
        nn_module_params = None
        nn_val = params.get("nn_module")
        if isinstance(nn_val, dict):
            nn_module_params = nn_val
        elif isinstance(nn_val, (list, tuple)) and len(nn_val) == 2:
            nn_module_params = nn_val[1]
        if nn_module_params:
            for k, v in nn_module_params.items():
                if k in model_kwargs:
                    model_kwargs[k] = v

    model = MultiDimStacker(**model_kwargs).to(device)

    new_state = {}
    for k, v in weights.items():
        for prefix in ("module.", "model.", "nn_module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        logger.warning("lRomul missing keys: %d (first: %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("lRomul unexpected keys: %d (first: %s)", len(unexpected), unexpected[:3])

    model.eval()
    logger.info("lRomul model loaded from %s on %s", ckpt_path, device)
    _MODEL = model
    return _MODEL


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def _resize_with_pad(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize keeping aspect ratio, then center-pad to exact target (matches lRomul)."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0,
    )


def _extract_grayscale_frames(video_path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = _resize_with_pad(f, FRAME_WIDTH, FRAME_HEIGHT)
        frames.append(f)
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8), fps


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Public inference
# ---------------------------------------------------------------------------

def predict_with_lromul(video_path: Path) -> list[LromulPrediction]:
    """Run lRomul on a video, apply Gaussian smoothing + peak detection,
    map PASS/DRIVE peaks to TurboVision predictions.
    """
    # Thresholds — applied to GAUSSIAN-SMOOTHED signal (sigma=3), so lower than raw.
    # Raw peaks of 0.4-0.6 get diluted to ~0.10-0.20 after smoothing.
    pass_peak_height = _env_float("LROMUL_PASS_PEAK_HEIGHT", 0.12)
    drive_peak_height = _env_float("LROMUL_DRIVE_PEAK_HEIGHT", 0.12)
    peak_distance = _env_int("LROMUL_PEAK_DISTANCE", 15)
    max_per_clip = _env_int("LROMUL_MAX_PER_CLIP", 6)
    stride = _env_int("LROMUL_STRIDE", 15)  # window stride in frames
    pass_receive_offset = _env_int("LROMUL_PASS_RECEIVE_OFFSET", 12)
    smooth_sigma = _env_float("LROMUL_SMOOTH_SIGMA", 3.0)

    try:
        model = _load_model()
    except Exception as e:
        logger.warning("lRomul unavailable: %s", e)
        return []

    try:
        frames, fps = _extract_grayscale_frames(video_path)
    except Exception as e:
        logger.error("lRomul frame extraction failed: %s", e)
        return []

    n = len(frames)
    if n == 0:
        return []

    window = MODEL_KWARGS["num_frames"]
    step_in = FRAME_STACK_STEP
    span = (window - 1) * step_in
    half = span // 2
    if n <= span:
        logger.warning("lRomul: video too short (%d frames) for span %d", n, span)
        return []

    device = torch.device(_get_device())
    use_amp = device.type == "cuda"

    indices = list(range(half, n - half, stride))
    centers_arr = np.array(indices, dtype=np.int64)
    probs_arr = np.zeros((len(indices), 2), dtype=np.float32)

    with torch.no_grad():
        for i, center in enumerate(indices):
            idx = [center - half + k * step_in for k in range(window)]
            clip = frames[idx]
            x = torch.from_numpy(clip).float().unsqueeze(0).to(device) / 255.0
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
            else:
                logits = model(x)
            probs_arr[i] = torch.sigmoid(logits).float().cpu().numpy()[0]

    # Postprocessing — Gaussian smoothing + peak detection
    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
    except ImportError:
        logger.warning("scipy unavailable; skipping lRomul postprocessing")
        return []

    predictions: list[LromulPrediction] = []

    # Convert source-fps frame index to validator 25-fps frame index
    def to_validator_frame(src_frame: int) -> int:
        if fps <= 0 or abs(fps - VALIDATOR_FRAME_RATE) < 0.01:
            return int(src_frame)
        return int(round(src_frame * VALIDATOR_FRAME_RATE / fps))

    # Log raw peaks before smoothing — useful for threshold tuning
    raw_pass = probs_arr[:, 0]
    raw_drive = probs_arr[:, 1]
    logger.info(
        "lRomul raw peaks: PASS max=%.3f (frame %d), DRIVE max=%.3f (frame %d)",
        float(raw_pass.max()), int(centers_arr[raw_pass.argmax()]),
        float(raw_drive.max()), int(centers_arr[raw_drive.argmax()]),
    )

    def _rescale_conf(smoothed_peak: float, threshold: float) -> float:
        """Map smoothed peak [threshold, threshold+0.4] → confidence [0.65, 0.95]."""
        span = 0.40
        norm = min(1.0, max(0.0, (smoothed_peak - threshold) / span))
        return 0.65 + norm * 0.30

    # PASS detection
    pass_signal = gaussian_filter1d(raw_pass, sigma=smooth_sigma)
    pass_peaks, pass_props = find_peaks(
        pass_signal, height=pass_peak_height, distance=peak_distance,
    )
    for i, idx in enumerate(pass_peaks):
        center_src = int(centers_arr[idx])
        smoothed = float(pass_props["peak_heights"][i])
        conf = _rescale_conf(smoothed, pass_peak_height)
        v_frame = to_validator_frame(center_src)
        predictions.append(LromulPrediction(
            frame_25fps=v_frame,
            action="pass",
            confidence=conf,
        ))
        predictions.append(LromulPrediction(
            frame_25fps=v_frame + pass_receive_offset,
            action="pass_received",
            confidence=conf,
        ))

    # DRIVE detection (maps to take_on)
    drive_signal = gaussian_filter1d(raw_drive, sigma=smooth_sigma)
    drive_peaks, drive_props = find_peaks(
        drive_signal, height=drive_peak_height, distance=peak_distance,
    )
    for i, idx in enumerate(drive_peaks):
        center_src = int(centers_arr[idx])
        smoothed = float(drive_props["peak_heights"][i])
        conf = _rescale_conf(smoothed, drive_peak_height)
        predictions.append(LromulPrediction(
            frame_25fps=to_validator_frame(center_src),
            action="take_on",
            confidence=conf,
        ))

    # Cap total output; sort by confidence desc, keep top max_per_clip, re-sort by frame
    predictions.sort(key=lambda p: p.confidence, reverse=True)
    predictions = predictions[:max_per_clip]
    predictions.sort(key=lambda p: p.frame_25fps)

    logger.info(
        "lRomul: %d windows, %d pass peaks, %d drive peaks, %d kept (pass_h>=%.2f, drive_h>=%.2f)",
        len(indices),
        len(pass_peaks),
        len(drive_peaks),
        len(predictions),
        pass_peak_height,
        drive_peak_height,
    )
    return predictions
