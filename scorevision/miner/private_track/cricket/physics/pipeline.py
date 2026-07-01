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
from scorevision.miner.private_track.cricket.tracknet.cvat_io import BallLabel
from scorevision.miner.private_track.cricket.keypoints.dataset import KP_NAMES
from scorevision.miner.private_track.cricket.physics import bundle as B


def select_ball_track(cands, w0, h0, max_gap=6, max_step_frac=0.12,
                      y_up_frac=0.06, min_chain=4, min_travel_frac=0.03,
                      conf_floor=0.5, strong_conf=0.7, min_strong=2, accel_frac=0.06):
    """Pick one ball per frame from top-k candidate sets -> {frame: BallLabel}.

    The ball is the CONFIDENT, smooth, descending arc through the candidates --
    not merely the farthest-travelling chain. (Travel maximisation wandered onto
    weak distractors and missed slow real deliveries.) So we maximise the summed
    node reward max(0, conf - conf_floor) along a chain constrained to be smooth
    (bounded velocity), low-acceleration, and mostly descending in image-y, via a
    DP over the candidate DAG ordered by frame. We then accept the highest-scoring
    chain that also actually MOVES (min_travel, rejects static high-conf logos)
    and has real support (>= min_strong nodes above strong_conf).

    Returns a full {frame: BallLabel} dict (None where no ball chosen), so the
    existing clean_ball / delivery_window / build_obs path is unchanged."""
    frames_all = sorted(cands)
    nodes = []  # (frame, x, y, conf)
    by_frame: dict[int, list[int]] = {}
    for f in frames_all:
        for (x, y, c) in cands[f]:
            by_frame.setdefault(f, []).append(len(nodes))
            nodes.append((f, x, y, c))
    out = {f: BallLabel(f, None, None, 0) for f in frames_all}
    if len(nodes) < min_chain:
        return out
    max_step = max_step_frac * w0
    accel_max = accel_frac * w0
    y_up = y_up_frac * h0
    reward = [max(0.0, c - conf_floor) for (_, _, _, c) in nodes]
    dp = list(reward)                # best summed reward of a chain ending here
    trav = [0.0] * len(nodes)        # pure travel (for the move/static guard)
    par = [-1] * len(nodes)
    vel = [(0.0, 0.0)] * len(nodes)  # per-frame velocity of the edge INTO this node
    order = sorted(range(len(nodes)), key=lambda j: nodes[j][0])
    for j in order:
        fj, xj, yj, _ = nodes[j]
        for fi in range(fj - max_gap, fj):
            for i in by_frame.get(fi, ()):
                _, xi, yi, _ = nodes[i]
                if yj < yi - y_up:                  # must not rise more than bounce slack
                    continue
                gap = fj - fi
                dx, dy = xj - xi, yj - yi
                d = (dx * dx + dy * dy) ** 0.5
                if d > max_step * gap:              # smoothness: bounded per-frame velocity
                    continue
                vij = (dx / gap, dy / gap)
                # Acceleration continuity: a real arc changes velocity slowly, so
                # a static-cluster -> ball jump (sudden velocity reversal) is
                # rejected even though its distance is within bounds.
                if par[i] != -1:
                    vx, vy = vel[i]
                    if ((vij[0] - vx) ** 2 + (vij[1] - vy) ** 2) ** 0.5 > accel_max:
                        continue
                if dp[i] + reward[j] > dp[j]:
                    dp[j] = dp[i] + reward[j]
                    trav[j] = trav[i] + d
                    par[j] = i
                    vel[j] = vij
    # Accept the highest-scoring chain that also moves and has strong support;
    # if the top chain is a static high-conf cluster it fails the guards and we
    # fall through to the next-best valid chain instead of giving up.
    min_travel = min_travel_frac * w0
    for end in sorted(range(len(nodes)), key=lambda j: dp[j], reverse=True):
        chain = []
        n = end
        while n != -1:
            chain.append(n)
            n = par[n]
        if len(chain) < min_chain or trav[end] < min_travel:
            continue
        if sum(1 for j in chain if nodes[j][3] >= strong_conf) < min_strong:
            continue
        for j in chain:
            f, x, y, c = nodes[j]
            out[f] = BallLabel(f, x, y, 1 if c >= 0.7 else 2)
        break
    return out


def _validate_stump_template(agg):
    """Drop geometrically-impossible stump keypoints using the known rigid rig.

    The 3 stumps are a short, near-horizontal base line with verticals rising to
    tops directly above. Misidentified keypoints (a 'top' at base height, or
    bases whose left/mid/right order is scrambled) make the bundle solve diverge,
    so we remove them BEFORE the solve -> an honest no-solve instead of garbage."""
    sides = ["left", "mid", "right"]
    bpts = {s: agg[f"bs_{s}_base"] for s in sides if f"bs_{s}_base" in agg}
    tpts = {s: agg[f"bs_{s}_top"] for s in sides if f"bs_{s}_top" in agg}
    bad = set()
    # (1) bases roughly collinear: drop a base whose y is a large outlier vs the
    #     horizontal spread of the base line.
    if len(bpts) >= 2:
        my = float(np.median([p[1] for p in bpts.values()]))
        span = max((abs(p[0] - q[0]) for p in bpts.values() for q in bpts.values()), default=1.0)
        for s, p in bpts.items():
            if abs(p[1] - my) > 0.5 * span + 20:
                bad.add(f"bs_{s}_base")
    # (2) each top must sit clearly ABOVE the base line (smaller image-y).
    good_base_ys = [p[1] for s, p in bpts.items() if f"bs_{s}_base" not in bad]
    if good_base_ys:
        base_y = float(np.median(good_base_ys))
        for s, p in tpts.items():
            if p[1] >= base_y - 20:
                bad.add(f"bs_{s}_top")
    # (3) base left/mid/right must be x-monotonic; if mid isn't between the
    #     others the labels are unreliable -> drop the whole base triple.
    vb = {s: bpts[s] for s in sides if s in bpts and f"bs_{s}_base" not in bad}
    if len(vb) == 3:
        xl, xm, xr = vb["left"][0], vb["mid"][0], vb["right"][0]
        if not (xl < xm < xr or xl > xm > xr):
            bad |= {f"bs_{s}_base" for s in sides}
    return {k: v for k, v in agg.items() if k not in bad}


def detect_keypoints_robust(model_kp, task: Path, device, out_hw=(288, 512),
                            thresh=0.25, min_frac=0.30):
    """Temporal-aggregated keypoints: detect per frame at a LOW threshold, then
    keep only keypoints seen consistently (>= min_frac of frames) and fix each at
    its temporal-median position, broadcast to every frame.

    Stumps are static within the short delivery window, so consolidating over
    time recovers the >=4-5 reliable stumps the bundle needs (per-frame 0.4
    thresholding was dropping to 3) while temporal median rejects sporadic FPs."""
    H, W = out_hw
    frames = sorted(task.glob("*.jpg"))
    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    sx, sy = w0 / W, h0 / H
    per = {n: [] for n in KP_NAMES}
    with torch.no_grad():
        for fp in frames:
            im = cv2.cvtColor(cv2.resize(cv2.imread(str(fp)), (W, H)), cv2.COLOR_BGR2RGB)
            x = im.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            hm = model_kp(torch.from_numpy(x).to(device))[0].cpu().numpy()
            for k, name in enumerate(KP_NAMES):
                pk = heatmap_peak(hm[k], thresh=thresh)
                if pk:
                    per[name].append((pk[0] * sx, pk[1] * sy))
    nfr = len(frames)
    need = max(2, int(min_frac * nfr))
    agg = {}
    for name, pts in per.items():
        if len(pts) >= need:
            arr = np.asarray(pts, float)
            agg[name] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
    agg = _validate_stump_template(agg)
    return {i: dict(agg) for i in range(nfr)}, w0, h0


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
    """Restrict ball to the delivery: the longest run of frames where image-y moves
    MONOTONICALLY (in either direction). The ball descends in image (y increasing)
    under a behind-the-batter camera but ASCENDS (y decreasing) under a behind-the-
    bowler camera; both are valid deliveries, so we take the longest monotonic run
    of whichever sign dominates. Excludes isolated pre-release FPs and post-impact
    frames. (Descending-only rejected the whole arc on bowler-end-camera clips.)"""
    fr = [f for f in sorted(ball) if ball[f].x is not None]
    if len(fr) < 4:
        return ball
    ys = [ball[f].y for f in fr]

    def longest_run(ok):
        best_s, best_e, s = 0, 0, 0
        for i in range(1, len(fr)):
            if ok(ys[i], ys[i - 1]):  # within 6px jitter of the monotone trend
                if i - s > best_e - best_s:
                    best_s, best_e = s, i
            else:
                s = i
        return best_s, best_e

    ds, de = longest_run(lambda a, b: a >= b - 6)   # descending (y increasing)
    as_, ae = longest_run(lambda a, b: a <= b + 6)   # ascending (y decreasing)
    best_s, best_e = (as_, ae) if (ae - as_) > (de - ds) else (ds, de)
    win = set(fr[best_s:best_e + 1])
    return {f: ball[f] for f in win}


def build_obs(ball, kps):
    obs = {}
    frames = set(ball) | set(kps)
    for f in sorted(frames):
        o = {}
        kp = kps.get(f, {})
        st = {n: np.array(kp[n]) for n in B.STUMP_3D if n in kp}
        # >=2 (was 4): the batsman/umpire usually occlude the stumps, so only 2-3
        # keypoints are visible per frame across broadcasts. _validate_stump_template
        # already rejects mis-identified sets (-> 0), and the physical-range output
        # guard nulls any degenerate (e.g. vertical-unconstrained) solve, so it is
        # safe to attempt calibration from as few as 2 consistent stumps + the
        # pitch edges rather than discarding the delivery outright.
        if len(st) >= 2:
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
