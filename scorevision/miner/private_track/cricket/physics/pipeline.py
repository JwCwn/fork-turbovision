"""
End-to-end inference physics: a delivery clip (folder of frames) -> 6 fields.

Stages: ball detection (ckpt_wb) + stump/pitch keypoint detection (ckpt_kp) +
kph (from OCR / arg) -> gravity-anchored bundle adjustment -> 6 fields.
All inputs auto-detected (no hand labels). kph is the scale anchor.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import cv2
import torch

from scorevision.miner.private_track.cricket.tracknet.model import TrackNetV2
from scorevision.miner.private_track.cricket.tracknet.infer import infer_task
from scorevision.miner.private_track.cricket.tracknet.heatmap import heatmap_peak
from scorevision.miner.private_track.cricket.keypoints.dataset import KP_NAMES
from scorevision.miner.private_track.cricket.physics import bundle as B


def detect_keypoints(model_kp, task: Path, device, out_hw=(288, 512), thresh=0.4):
    """Per-frame stump+pitch keypoints in original pixels: {frame: {kp:(x,y)}}."""
    H, W = out_hw
    frames = sorted(task.glob("*.jpg"))
    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    sx, sy = w0 / W, h0 / H
    out = {}
    with torch.no_grad():
        for i, fp in enumerate(frames):
            im = cv2.cvtColor(cv2.resize(cv2.imread(str(fp)), (W, H)), cv2.COLOR_BGR2RGB)
            x = im.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            hm = model_kp(torch.from_numpy(x).to(device))[0].cpu().numpy()
            kp = {}
            for k, name in enumerate(KP_NAMES):
                pk = heatmap_peak(hm[k], thresh=thresh)
                if pk:
                    kp[name] = (pk[0] * sx, pk[1] * sy)
            out[i] = kp
    return out, w0, h0


def _resid(A, V, mask, full_A):
    res = np.zeros(full_A.shape[0])
    for c in range(V.shape[1]):
        coef, *_ = np.linalg.lstsq(A[mask], V[mask, c], rcond=None)
        res += (full_A @ coef - V[:, c]) ** 2
    return np.sqrt(res)


def _quad_inliers(ts, vs, thresh):
    """Largest subset consistent with a single quadratic v(t), via deterministic
    minimal-sample (triplet) consensus + refit. vs may be (N,) or (N,2)."""
    from itertools import combinations
    ts = np.asarray(ts, float); vs = np.asarray(vs, float)
    V = vs.reshape(len(ts), -1)
    n = len(ts)
    if n < 4:
        return np.ones(n, bool)
    A = np.vstack([ts**2, ts, np.ones(n)]).T
    triplets = list(combinations(range(n), 3))
    if len(triplets) > 1200:  # cap for large n: consecutive windows
        triplets = [(i, i + 1, i + 2) for i in range(n - 2)]
    best = np.zeros(n, bool)
    for tri in triplets:
        m = np.zeros(n, bool); m[list(tri)] = True
        if np.linalg.matrix_rank(A[m]) < 3:
            continue
        inl = _resid(A, V, m, A) < thresh
        if inl.sum() > best.sum():
            best = inl
    # refit on consensus once
    if best.sum() >= 3:
        best = _resid(A, V, best, A) < thresh
    return best


def clean_ball(ball, thresh=22.0, win=3):
    """Drop GROSS outliers via local leave-one-out quadratic prediction.
    The trajectory is piecewise (pre/post bounce), so a *local* fit (not one
    global parabola) is used so post-bounce points survive."""
    fr = [f for f in sorted(ball) if ball[f].x is not None]
    if len(fr) < 5:
        return ball
    ts = np.array(fr, float)
    xy = np.array([[ball[f].x, ball[f].y] for f in fr])
    keep = []
    for i in range(len(fr)):
        nb = [j for j in range(len(fr)) if j != i and abs(j - i) <= win]
        if len(nb) < 3:
            keep.append(True); continue
        deg = 2 if len(nb) >= 3 else 1
        pred = np.array([np.polyval(np.polyfit(ts[nb], xy[nb, c], deg), ts[i]) for c in range(2)])
        keep.append(np.linalg.norm(pred - xy[i]) < thresh)
    return {f: ball[f] for f, k in zip(fr, keep) if k}


def clean_kps(kps, thresh=12.0):
    """Per-keypoint temporal outlier rejection (static pts move smoothly under pan)."""
    from scorevision.miner.private_track.cricket.keypoints.dataset import KP_NAMES
    out = {f: {} for f in kps}
    for name in KP_NAMES:
        fr = [f for f in sorted(kps) if name in kps[f]]
        if len(fr) < 4:
            for f in fr:
                out[f][name] = kps[f][name]
            continue
        ts = np.array(fr, float)
        xy = np.array([kps[f][name] for f in fr])
        inl = _quad_inliers(ts, xy, thresh)
        for f, k in zip(fr, inl):
            if k:
                out[f][name] = kps[f][name]
    return out


def delivery_window(ball):
    """Restrict ball to the delivery (release->batter): the longest run of frames
    where image-y increases monotonically (ball descending toward the batter under
    the bowler-end camera). Excludes isolated pre-release FPs and post-impact frames."""
    fr = [f for f in sorted(ball) if ball[f].x is not None]
    if len(fr) < 4:
        return ball
    ys = [ball[f].y for f in fr]
    best_s, best_e = 0, 0; s = 0
    for i in range(1, len(fr)):
        if ys[i] >= ys[i - 1] - 6:  # allow tiny non-monotonic jitter
            if i - s > best_e - best_s:
                best_s, best_e = s, i
        else:
            s = i
    win = set(fr[best_s:best_e + 1])
    return {f: ball[f] for f in win}


def build_obs(ball, kps):
    obs = {}
    frames = set(ball) | set(kps)
    for f in sorted(frames):
        o = {}
        kp = kps.get(f, {})
        st = {n: np.array(kp[n]) for n in B.STUMP_3D if n in kp}
        if len(st) >= 4:
            o["stumps"] = st
        pe = {k: np.array(kp[k]) for k in ["pitch_left_far", "pitch_left_near",
                                           "pitch_right_far", "pitch_right_near"] if k in kp}
        if len(pe) >= 1:
            o["pitch"] = pe
        b = ball.get(f)
        if b is not None and b.x is not None:
            o["ball"] = np.array([b.x, b.y])
        if o:
            obs[f] = o
    return obs


def run(task, kph, device=None, fps=25.0,
        ckpt_ball="Scripts/tracknet/ckpt_wb.pt", ckpt_kp="Scripts/keypoints/ckpt_kp.pt"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    task = Path(task)
    frames = sorted(task.glob("*.jpg"))
    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    cx, cy = w0 / 2.0, h0 / 2.0

    sdb = torch.load(ckpt_ball, map_location=device); nfb = sdb.get("n_frames", 3)
    mb = TrackNetV2(n_frames=nfb).to(device).eval(); mb.load_state_dict(sdb["model"])
    ball, _, _ = infer_task(mb, task, device, nfb, thresh=0.8)

    sdk = torch.load(ckpt_kp, map_location=device); K = sdk["out_ch"]
    mk = TrackNetV2(n_frames=1, out_ch=K).to(device).eval(); mk.load_state_dict(sdk["model"])
    kps, _, _ = detect_keypoints(mk, task, device)

    ball = delivery_window(clean_ball(ball))
    kps = clean_kps(kps)
    # time reference = first ball frame (release), so P0 is the release position
    fr0 = min((f for f in ball), default=0)
    obs = build_obs(ball, kps)
    obs = {f - fr0: o for f, o in obs.items()}
    nb = sum("ball" in o for o in obs.values())
    ns = sum("stumps" in o for o in obs.values())
    npi = sum("pitch" in o for o in obs.values())
    print(f"{task.name}: {w0}x{h0}  detected ball={nb} stumps>=4={ns} pitch={npi}  kph={kph}")
    if nb < 4 or ns < 3:
        print("  insufficient detections for physics"); return None

    from scorevision.miner.private_track.cricket.physics.run_delivery import init_params
    sol = B.fit(obs, cx, cy, fps, init_params(cx, cy, kph), kph_obs=kph)
    prm = B.unpack(sol.x)
    r = B.residuals(sol.x, obs, cx, cy, fps, kph_obs=kph)
    f6 = B.fields_from(prm, fps, kph_obs=kph)
    print(f"  BA: rms_resid={np.sqrt(np.mean(r**2)):.1f}px focal={prm['f']:.0f} "
          f"camC=({prm['C'][0]:.1f},{prm['C'][1]:.1f},{prm['C'][2]:.1f})")
    print("  6 fields:")
    for k, v in f6.items():
        print(f"    {k:12s} {v:8.3f}")
    return f6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--kph", type=float, default=None)
    args = ap.parse_args()
    run(args.task, args.kph)
