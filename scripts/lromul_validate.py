"""Validate lRomul's SoccerNet Ball Action Spotting 2023 (1st place)
pretrained weights on TurboVision challenge videos.

POC — checks whether the model usefully detects PASS events on our clip style.

Usage:
    source ~/e2e-spot/spot/.venv/bin/activate
    cd ~/ball-spotting/repo   # lRomul's cloned repo
    python lromul_validate.py <video.mp4> <checkpoint.pth>
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.multidim_stacker import MultiDimStacker  # noqa: E402


MODEL_KWARGS = dict(
    model_name="tf_efficientnetv2_b0.in1k",
    num_classes=2,             # PASS, DRIVE
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


def load_model(ckpt_path: Path) -> MultiDimStacker:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # lRomul uses Argus framework — checkpoint wraps weights in "nn_state_dict".
    # Also may carry 'params' that override default model kwargs.
    params = None
    if isinstance(state, dict):
        if "nn_state_dict" in state:
            weights = state["nn_state_dict"]
            params = state.get("params")
            print(f"Argus checkpoint detected (params keys: {list(params.keys()) if params else 'none'})")
        elif "state_dict" in state:
            weights = state["state_dict"]
        else:
            weights = state
    else:
        weights = state

    # Build model kwargs from saved params when available (ensures architecture match)
    model_kwargs = dict(MODEL_KWARGS)
    if params and isinstance(params, dict):
        nn_module_params = params.get("nn_module") if isinstance(params.get("nn_module"), dict) else None
        # Argus format: ("ModelName", {kwargs}) or just {kwargs}
        if isinstance(params.get("nn_module"), (list, tuple)) and len(params["nn_module"]) == 2:
            nn_module_params = params["nn_module"][1]
        if nn_module_params:
            for k, v in nn_module_params.items():
                if k in model_kwargs:
                    if model_kwargs[k] != v:
                        print(f"Override {k}: {model_kwargs[k]} -> {v}")
                    model_kwargs[k] = v

    model = MultiDimStacker(**model_kwargs).to(device)

    # strip common prefixes if present
    new_state = {}
    for k, v in weights.items():
        for prefix in ("module.", "model.", "nn_module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} — first: {missing[:3]}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} — first: {unexpected[:3]}")
    model.eval()
    return model


def _resize_with_pad(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize keeping aspect ratio, then pad to exact target size (matches lRomul)."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))

    # Pad (center) to exact target size
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0,
    )
    return padded


def extract_grayscale_frames(video_path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: {fps:.2f} fps, {total} frames, {total/fps:.1f}s")

    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = _resize_with_pad(f, FRAME_WIDTH, FRAME_HEIGHT)
        frames.append(f)
    cap.release()
    return np.stack(frames), fps


def sliding_window_predict(
    model: MultiDimStacker,
    frames: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Run inference with a 33-frame window stepped by frame_stack_step=2 centers."""
    device = next(model.parameters()).device
    n = len(frames)
    window = MODEL_KWARGS["num_frames"]               # 33
    step_in = FRAME_STACK_STEP                         # 2 (intra-window spacing)
    span = (window - 1) * step_in                      # 64 frames span per window

    half = span // 2
    if n <= span:
        print(f"Video too short ({n} frames) for {window}-frame window spanning {span}")
        return []

    results: list[tuple[int, np.ndarray]] = []

    # Stride for sliding = every 25 frames (~1s) for reasonable density
    stride = 25
    indices = list(range(half, n - half, stride))

    print(f"Running {len(indices)} windowed inferences (stride={stride}, span={span})...")
    use_amp = device.type == "cuda"

    start = time.time()
    with torch.no_grad():
        for center in indices:
            # Select 33 frames centered at `center`, stepping by 2
            idx = [center - half + i * step_in for i in range(window)]
            clip = frames[idx]  # (33, H, W)
            # Model expects (B, T, H, W) grayscale — normalized to [0, 1]
            x = torch.from_numpy(clip).float().unsqueeze(0).to(device) / 255.0

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
            else:
                logits = model(x)

            probs = torch.sigmoid(logits).float().cpu().numpy()[0]
            results.append((center, probs))
    elapsed = time.time() - start
    print(f"Inference time: {elapsed:.2f}s ({len(indices)/elapsed:.1f} windows/s)")
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python lromul_validate.py <video.mp4> <checkpoint.pth>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    ckpt_path = Path(sys.argv[2])

    print(f"Loading model from {ckpt_path}...")
    model = load_model(ckpt_path)
    print(f"Model loaded on {next(model.parameters()).device}")

    frames, fps = extract_grayscale_frames(video_path)
    print(f"Frames shape: {frames.shape}")

    results = sliding_window_predict(model, frames)
    if not results:
        return

    # Raw per-class peak confidence
    print("\n=== Raw per-class peak confidence ===")
    for cls_idx, name in enumerate(CLASS_NAMES):
        peaks = [(c, p[cls_idx]) for c, p in results]
        peaks.sort(key=lambda x: x[1], reverse=True)
        top5 = peaks[:5]
        print(f"\n{name}:")
        for center, conf in top5:
            t = center / fps
            print(f"  frame {center} ({t:.2f}s): conf={conf:.4f}")

    # Apply lRomul-style postprocessing: Gaussian smooth + peak detection
    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
    except ImportError:
        print("\nInstall scipy to enable postprocessing: pip install scipy")
        return

    centers = np.array([c for c, _ in results])
    prob_array = np.stack([p for _, p in results])  # (N, 2)

    print("\n=== Postprocessed detections (Gaussian sigma=3, peak height=0.15) ===")
    for cls_idx, name in enumerate(CLASS_NAMES):
        signal = prob_array[:, cls_idx]
        smoothed = gaussian_filter1d(signal, sigma=3.0)
        peaks_idx, props = find_peaks(smoothed, height=0.15, distance=15)
        print(f"\n{name}: {len(peaks_idx)} peaks")
        for i, idx in enumerate(peaks_idx):
            center = int(centers[idx])
            t = center / fps
            peak_h = float(props["peak_heights"][i])
            raw = float(signal[idx])
            print(f"  frame {center} ({t:.2f}s): smoothed={peak_h:.3f} raw={raw:.3f}")


if __name__ == "__main__":
    main()
