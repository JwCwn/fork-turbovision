"""
Gravity-anchored joint bundle adjustment: camera + 3D ball trajectory.

Single broadcast camera (fixed position, smooth pan/zoom). Absolute scale/focal
that geometry cannot fix is pinned by GRAVITY (g=9.81) + fps + the speed-overlay
kph (hard scale anchor on release speed).

World: origin = batter-end middle-stump base; +x down pitch toward bowler;
+y camera-right; +z up.

Parameters (24):
  C(3) camera centre (fixed)
  r0(3) w(3) a(3)  rotation: rvec_i = r0 + w*t + a*t^2  (quadratic pan)
  f(1) focal px
  P0(3) V0(3)      pre-bounce release pos / velocity (|V0| hard-set from kph)
  tb(1)            bounce time (s)
  V1(3)            post-bounce velocity
  W(1)             pitch half-width (m)
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.optimize import least_squares

G = 9.81
STUMP_H = 0.711
STUMP_DY = 0.1143
STUMP_3D = {
    "bs_left_base": (0., -STUMP_DY, 0.), "bs_mid_base": (0., 0., 0.), "bs_right_base": (0., STUMP_DY, 0.),
    "bs_left_top": (0., -STUMP_DY, STUMP_H), "bs_mid_top": (0., 0., STUMP_H), "bs_right_top": (0., STUMP_DY, STUMP_H),
}
PITCH_L = 20.12  # crease-to-crease pitch length (m); far corners = batter end (x=0)


def pitch_3d(W):
    """4 pitch corners as known 3D points (z=0). far=batter end (x=0, near stumps),
    near=bowler end (x=PITCH_L). y=+/-W half-width. Breaks depth/focal degeneracy."""
    return {"pitch_left_far": (0., -W, 0.), "pitch_right_far": (0., W, 0.),
            "pitch_left_near": (PITCH_L, -W, 0.), "pitch_right_near": (PITCH_L, W, 0.)}

#            C0 C1 C2  r r r   w w w   a a a    f   P P P    V V V  tb  V V V    W
#   P0[0] (release x) lower bound 0 -> 12: the bowler always releases from THEIR end
#   (x~17-22). With stumps absent (pitch-only), the bundle had a gauge ambiguity and
#   flipped the trajectory (release at x=0, bounce off the pitch, V1[0]->0 -> nan);
#   pinning release to the bowler half anchors the orientation the stumps used to give.
LB = np.array([20,-10, 2, -4,-4,-4, -3,-3,-3, -6,-6,-6, 1200, 12,-6,0, -55,-12,-12,0.0, -55,-12,-5, 1.0])
UB = np.array([75, 10,25,  4, 4, 4,  3, 3, 3,  6, 6, 6, 7000,25, 6,6,   0, 12,  8,1.6,   0, 12,22, 2.6])


def _R(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float))[0]


def project(Pw, C, rvec, f, cx, cy):
    Xc = _R(rvec) @ (np.asarray(Pw, float) - np.asarray(C, float))
    if Xc[2] <= 1e-6:
        return np.array([1e6, 1e6])
    return np.array([f * Xc[0] / Xc[2] + cx, f * Xc[1] / Xc[2] + cy])


def project_R(Pw, C, R, f, cx, cy):
    """project() with a precomputed rotation matrix R. rvec is constant within
    a frame, so callers compute R once per frame instead of re-running
    cv2.Rodrigues per point. Bit-identical to project()."""
    Xc = R @ (np.asarray(Pw, float) - np.asarray(C, float))
    if Xc[2] <= 1e-6:
        return np.array([1e6, 1e6])
    return np.array([f * Xc[0] / Xc[2] + cx, f * Xc[1] / Xc[2] + cy])


def traj_point(P0, V0, tb, V1, t):
    g = np.array([0, 0, -G])
    if t <= tb:
        return P0 + V0 * t + 0.5 * g * t * t
    Pb = P0 + V0 * tb + 0.5 * g * tb * tb
    dt = t - tb
    return Pb + V1 * dt + 0.5 * g * dt * dt


def unpack(p):
    return dict(C=p[0:3], r0=p[3:6], w=p[6:9], a=p[9:12], f=p[12],
               P0=p[13:16], V0=p[16:19], tb=p[19], V1=p[20:23], W=p[23])


def _rvec_at(prm, t):
    return prm["r0"] + prm["w"] * t + prm["a"] * t * t


def _v0(prm, kph_obs):
    """V0 with magnitude hard-set from the overlay kph (direction free)."""
    v = prm["V0"]; n = np.linalg.norm(v)
    if kph_obs is None or n < 1e-9:
        return v
    return (kph_obs / 3.6) * v / n


def residuals(p, obs, cx, cy, fps, w=(1.0, 1.0, 2.0, 0.05), kph_obs=None):
    prm = unpack(p)
    V0 = _v0(prm, kph_obs)
    res = []
    ws, wp, wb, wr = w
    C, f = prm["C"], prm["f"]
    for fr, o in obs.items():
        t = fr / fps
        R = _R(_rvec_at(prm, t))  # one Rodrigues per frame, reused for all points
        for name, uv in o.get("stumps", {}).items():
            if name in STUMP_3D:
                res += list(ws * (project_R(STUMP_3D[name], C, R, f, cx, cy) - uv))
        pe = o.get("pitch", {})
        if pe:
            P3 = pitch_3d(prm["W"])
            for k, uv in pe.items():
                if k in P3:
                    res += list(wp * (project_R(P3[k], C, R, f, cx, cy) - uv))
        if "ball" in o:
            P = traj_point(prm["P0"], V0, prm["tb"], prm["V1"], t)
            res += list(wb * (project_R(P, C, R, f, cx, cy) - o["ball"]))
    res += list(wr * prm["w"]); res += list(wr * prm["a"])
    res.append(wr * (prm["W"] - 1.5))
    # physical anchors: the ball is on the ground at the bounce (z=0); plausible
    # release height; these pin the trajectory to the rig's ground plane so the
    # absolute-scaled (kph+gravity) ball acts as a depth ruler.
    g = np.array([0, 0, -G])
    Pb = prm["P0"] + V0 * prm["tb"] + 0.5 * g * prm["tb"] ** 2
    res.append(5.0 * Pb[2])                       # bounce on ground
    res.append(0.3 * (prm["P0"][2] - 2.2))        # release height ~2.2 m
    res.append(0.5 * (prm["C"][2] - 8.0))         # camera height prior ~8 m (broadcast)
    # post-bounce physics: ball can't speed up at the bounce. Horizontal speed
    # drops ~22% (friction); pin V1 horizontal near 0.78*V0 so stump_y/z/deviation
    # are observable from the few post-bounce points instead of running away.
    V1 = prm["V1"]
    res.append(0.6 * (V1[0] - 0.78 * V0[0]))
    res.append(2.0 * max(0.0, np.linalg.norm(V1) - np.linalg.norm(V0)))
    return np.array(res)


def fit(obs, cx, cy, fps, init, kph_obs=None):
    p0 = np.clip(np.array(init, float), LB + 1e-6, UB - 1e-6)
    # max_nfev 8000 -> 2000: a GOOD solve converges in a few hundred evals (clean
    # clips fit in ~3 s). Only a DEGENERATE solve keeps iterating to the cap, and
    # its result is discarded by the physical-range guard anyway -- so the extra
    # evals just burned ~23 s of wall-clock (timeout risk). Cap it low.
    return least_squares(residuals, p0, args=(obs, cx, cy, fps),
                         kwargs={"kph_obs": kph_obs}, method="trf",
                         bounds=(LB, UB), loss="soft_l1", f_scale=2.0, max_nfev=2000)


IMPACT_X = 1.22  # popping crease (m from stumps): default batter interception plane


def _cross_x(Pb, V1, x_target):
    """Post-bounce trajectory crossing of plane x=x_target -> (y, z) or (nan,nan)."""
    g = np.array([0, 0, -G])
    if abs(V1[0]) < 1e-6:
        return float("nan"), float("nan")
    dt = (x_target - Pb[0]) / V1[0]
    Pc = Pb + V1 * dt + 0.5 * g * dt * dt
    return Pc[1], Pc[2]


def fields_from(prm, fps, kph_obs=None):
    P0, tb, V1 = prm["P0"], prm["tb"], prm["V1"]
    V0 = _v0(prm, kph_obs)
    g = np.array([0, 0, -G])
    Pb = P0 + V0 * tb + 0.5 * g * tb * tb
    kph = np.linalg.norm(V0) * 3.6
    swing = np.degrees(np.arctan2(V0[1], -V0[0]))
    deviation = np.degrees(np.arctan2(V1[1], -V1[0])) - swing
    stump_y, stump_z = _cross_x(Pb, V1, 0.0)
    impact_y, impact_z = _cross_x(Pb, V1, IMPACT_X)
    return dict(
        kph=kph, bounce_x=Pb[0], swing_angle=swing, deviation=deviation,
        stump_y=stump_y, stump_z=stump_z,
        release_y=P0[1], release_z=P0[2], bounce_y=Pb[1],
        impact_x=IMPACT_X, impact_y=impact_y, impact_z=impact_z,
        interception_distance=Pb[0] - IMPACT_X,
    )


def _synth_test():
    rng = np.random.RandomState(0); cx, cy, fps = 640.0, 360.0, 25.0
    C_t = np.array([42.0, 1.5, 8.0]); target = np.array([8.0, 0.0, 0.5])
    fwd = target - C_t; fwd /= np.linalg.norm(fwd); up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right); trueup = np.cross(right, fwd)
    r0_t = cv2.Rodrigues(np.vstack([right, -trueup, fwd]))[0].ravel()
    w_t = np.array([0.0, 0.06, 0.0]); a_t = np.zeros(3); f_t = 3600.0; W_t = 1.5
    # ground-bouncing trajectory: choose V0_z so Pb_z = 0 (ball hits the pitch)
    P0_t = np.array([17.0, 0.1, 2.1]); tb_t = 0.42
    v0z = (0.5 * G * tb_t ** 2 - P0_t[2]) / tb_t
    V0_t = np.array([-33.0, 0.4, v0z])
    V1_t = np.array([-31.0, 1.0, 6.0])
    true = np.concatenate([C_t, r0_t, w_t, a_t, [f_t], P0_t, V0_t, [tb_t], V1_t, [W_t]])
    prm_t = unpack(true); kph_t = np.linalg.norm(V0_t) * 3.6
    obs = {}
    for fr in range(30):
        t = fr/fps; rvec = _rvec_at(prm_t, t); o = {}
        if fr <= 6:
            o["stumps"] = {n: project(STUMP_3D[n], C_t, rvec, f_t, cx, cy)+rng.randn(2)*0.3 for n in STUMP_3D}
        pe = {}
        for k, P in pitch_3d(W_t).items():
            pe[k] = project(P, C_t, rvec, f_t, cx, cy) + rng.randn(2) * 0.3
        o["pitch"] = pe
        if 4 <= fr <= 26:
            o["ball"] = project(traj_point(P0_t,V0_t,tb_t,V1_t,t),C_t,rvec,f_t,cx,cy)+rng.randn(2)*0.5
        obs[fr] = o
    init = np.concatenate([[40,0,7], r0_t+rng.randn(3)*0.05, [0,0,0],[0,0,0],[3000],
                           [16,0,2.0],[-30,0,-1],[0.4],[-29,0,5],[1.5]])
    sol = fit(obs, cx, cy, fps, init, kph_obs=kph_t); prm = unpack(sol.x)
    print("synthetic(quad+hardkph): success=%s nfev=%d focal=%.0f" % (sol.success, sol.nfev, prm["f"]))
    ft = fields_from(prm_t, fps, kph_t); fe = fields_from(prm, fps, kph_t)
    for k in ft:
        print("  %-12s true=%8.3f est=%8.3f d=%.3f" % (k, ft[k], fe[k], abs(ft[k]-fe[k])))


if __name__ == "__main__":
    _synth_test()
