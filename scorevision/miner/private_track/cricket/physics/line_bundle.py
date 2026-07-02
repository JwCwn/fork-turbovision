"""Line-based calibration: solve the camera + ball-trajectory bundle from detected
PITCH / RETURN-CREASE lines plus the ball, for clips where the stumps are occluded.

Same 24-parameter model, priors and gravity anchor as physics.bundle; only the
image residual differs — each line contributes its two endpoints reprojected onto
the known 3D crease coordinates instead of stump/pitch points. The batter-end
return creases fix the origin; the bowler-end return creases (visible only in the
opening frames) break the depth gauge; the pitch edges fix width and length. A
solve therefore needs a ball arc AND at least one bowler-end return observation.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import least_squares

from scorevision.miner.private_track.cricket.physics import bundle as B
from scorevision.miner.private_track.cricket.physics.run_delivery import init_params

# world 3D endpoints per line (z=0): [batter/far end, bowler/near end].
PL = 20.12
LINE_3D = {
    "pitch_left_edge":     [(0.0, -1.525, 0.0), (PL,   -1.525, 0.0)],
    "pitch_right_edge":    [(0.0,  1.525, 0.0), (PL,    1.525, 0.0)],
    "return_batter_left":  [(0.0, -1.32, 0.0),  (1.22, -1.32, 0.0)],
    "return_batter_right": [(0.0,  1.32, 0.0),  (1.22,  1.32, 0.0)],
    "return_bowler_left":  [(PL - 1.22, -1.32, 0.0), (PL, -1.32, 0.0)],
    "return_bowler_right": [(PL - 1.22,  1.32, 0.0), (PL,  1.32, 0.0)],
}


def line_residuals(p, lines, ball, cx, cy, fps, fr0, kph, wL=1.0, wB=2.0, wr=0.05):
    """Reproject line endpoints + ball trajectory; append the same physical priors
    physics.bundle.residuals uses (planar bounce, release height, camera height,
    seam-carry damping, no-speed-up-after-bounce)."""
    prm = B.unpack(p)
    V0 = B._v0(prm, kph)
    C, f = prm["C"], prm["f"]
    res = []
    for fr, lns in lines.items():
        t = (fr - fr0) / fps
        rvec = B._rvec_at(prm, t)
        for name, (a, b) in lns.items():
            wa, wb = LINE_3D[name]
            res += list(wL * (B.project(wa, C, rvec, f, cx, cy) - a))
            res += list(wL * (B.project(wb, C, rvec, f, cx, cy) - b))
    for fr, uv in ball.items():
        t = (fr - fr0) / fps
        rvec = B._rvec_at(prm, t)
        P = B.traj_point(prm["P0"], V0, prm["tb"], prm["V1"], t)
        res += list(wB * (B.project(P, C, rvec, f, cx, cy) - uv))
    res += list(wr * prm["w"])
    res += list(wr * prm["a"])
    res.append(wr * (prm["W"] - 1.5))
    g = np.array([0, 0, -B.G])
    Pb = prm["P0"] + V0 * prm["tb"] + 0.5 * g * prm["tb"] ** 2
    res.append(5.0 * Pb[2])
    res.append(0.3 * (prm["P0"][2] - 2.2))
    res.append(0.5 * (prm["C"][2] - 8.0))
    V1 = prm["V1"]
    res.append(0.6 * (V1[0] - 0.78 * V0[0]))
    res.append(2.0 * max(0.0, np.linalg.norm(V1) - np.linalg.norm(V0)))
    return np.array(res)


def solve_lines(lines, ball, cx, cy, fps, kph, max_nfev=200, budget_s=6.0):
    """lines: {frame:{name:(a,b)}} in px; ball: {frame: np.array([x,y])} in px.

    Returns (fields, dbg) on a well-posed solve, else (None, dbg-with-reason).
    Requires a ball arc plus at least one bowler-end return crease observation:
    without the bowler end the pitch is a gauge-free strip and the solve goes
    degenerate (the same failure the pitch-primary stump path had)."""
    n_line_obs = sum(len(v) for v in lines.values())
    n_ball = len(ball)
    frames_with_bowler = sum(
        1 for v in lines.values() if any(n.startswith("return_bowler") for n in v)
    )
    # The camera (-> stump_y/z) is fixed by the lines; the ball only adds the
    # trajectory fields, so a couple of ball points suffice (the de-risk solved
    # cleanly with 2). Require enough lines + both-end coverage (a bowler-end
    # return) to break the depth gauge; too few bowler frames -> degenerate.
    if n_line_obs < 12 or n_ball < 2 or frames_with_bowler < 3:
        return None, dict(reason=f"insufficient lines={n_line_obs} ball={n_ball} "
                                 f"bowler_frames={frames_with_bowler}")
    fr0 = min(ball)
    p0 = np.clip(np.array(init_params(cx, cy, kph), float), B.LB + 1e-6, B.UB - 1e-6)
    # HARD wall-clock deadline (same as bundle.fit): max_nfev alone did NOT bound the
    # time -- the Python line residual is expensive, so 76 line-obs x 200 nfev still
    # ran ~18 s and timed the challenge out. Stop at budget_s and keep the best iterate
    # (a degenerate line solve is nulled by the physical-range guard anyway).
    t0 = time.perf_counter()
    best = {"x": p0, "c": np.inf}

    def _resid(p, *a, **k):
        res = line_residuals(p, *a, **k)
        c = float(np.dot(res, res))
        if c < best["c"]:
            best["c"], best["x"] = c, p.copy()
        if time.perf_counter() - t0 > budget_s:
            raise B._FitBudget()
        return res

    try:
        sol = least_squares(
            _resid, p0, args=(lines, ball, cx, cy, fps, fr0, kph),
            method="trf", bounds=(B.LB, B.UB), loss="soft_l1", f_scale=2.0,
            max_nfev=max_nfev,
        )
    except B._FitBudget:
        sol = B._Sol(); sol.x = best["x"]; sol.success = False; sol.nfev = -1; sol.cost = best["c"]
    prm = B.unpack(sol.x)
    r = line_residuals(sol.x, lines, ball, cx, cy, fps, fr0, kph)
    rms = float(np.sqrt(np.mean(r ** 2)))
    fields = B.fields_from(prm, fps, kph_obs=kph)
    dbg = dict(rms=rms, focal=float(prm["f"]),
               camC=[round(float(v), 1) for v in prm["C"]],
               n_line=n_line_obs, n_ball=n_ball, bowler_frames=frames_with_bowler,
               calib="lines")
    return fields, dbg
