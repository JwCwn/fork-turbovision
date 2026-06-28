"""
Per-frame camera calibration for a cricket delivery.

World frame (spec): origin = batter-end middle-stump base; +x down the pitch
toward the bowler; +y to the camera's right; +z up.

Inputs per frame (from CVAT): batter-end stump points (known 3D on the x=0 plane)
+ pitch side-edge points (give the along-pitch vanishing point = x-direction).

Method (robust to the unstable vertical vanishing point):
  1. focal f: 1-D search. For each candidate f, solvePnP the 6 stump points
     (planar, IPPE), then project the world +x direction and compare to the
     observed pitch vanishing point. Pick the f (shared across the delivery)
     minimising that mismatch + stump reprojection.
  2. pose: with f fixed, solvePnP per frame -> R, t, camera center C.
Validation: stump reprojection error, pitch-VP error, camera-center stability.
"""
from __future__ import annotations
import numpy as np
import cv2

STUMP_H = 0.711
STUMP_DY = 0.1143  # mid->outer stump offset in y (m), approx (set width ~0.2286)

# known 3D (X,Y,Z); X=0 stump plane. image-left = -y, image-right = +y.
STUMP_3D = {
    "bs_left_base":  (0.0, -STUMP_DY, 0.0),
    "bs_mid_base":   (0.0,  0.0,      0.0),
    "bs_right_base": (0.0, +STUMP_DY, 0.0),
    "bs_left_top":   (0.0, -STUMP_DY, STUMP_H),
    "bs_mid_top":    (0.0,  0.0,      STUMP_H),
    "bs_right_top":  (0.0, +STUMP_DY, STUMP_H),
}


def _line(p1, p2):
    return np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])


def pitch_vanishing_point(kp):
    """Intersection of the two pitch side-edges -> along-pitch (x) VP, or None."""
    need = ["pitch_left_far", "pitch_left_near", "pitch_right_far", "pitch_right_near"]
    if not all(k in kp for k in need):
        return None
    l = _line(kp["pitch_left_far"], kp["pitch_left_near"])
    r = _line(kp["pitch_right_far"], kp["pitch_right_near"])
    v = np.cross(l, r)
    if abs(v[2]) < 1e-9:
        return None
    return v[:2] / v[2]


def homography_for_frame(kp):
    """DLT homography mapping the x=0 stump plane (y,z) -> image (u,v).
    Returns (H 3x3, reproj_err_px, n_pts) or (None, None, n)."""
    src, dst = [], []
    for name, (X, Y, Z) in STUMP_3D.items():
        if name in kp:
            src.append((Y, Z)); dst.append(kp[name])
    if len(src) < 4:
        return None, None, len(src)
    A = []
    for (y, z), (u, v) in zip(src, dst):
        A.append([-y, -z, -1, 0, 0, 0, u*y, u*z, u])
        A.append([0, 0, 0, -y, -z, -1, v*y, v*z, v])
    _, _, Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3, 3)
    err = 0.0
    for (y, z), (u, v) in zip(src, dst):
        p = H @ [y, z, 1]; p = p[:2] / p[2]
        err += np.hypot(p[0]-u, p[1]-v)
    return H, err/len(src), len(src)


def _stump_corr(kp):
    obj, img = [], []
    for name, XYZ in STUMP_3D.items():
        if name in kp:
            obj.append(XYZ); img.append(kp[name])
    return np.array(obj, np.float64), np.array(img, np.float64)


def _Kinv(f, cx, cy):
    return np.array([[1/f, 0, -cx/f], [0, 1/f, -cy/f], [0, 0, 1.0]])


def solve_pose(kp, f, cx, cy):
    """Pose from the x=0 stump homography (gives y,z axes + origin) plus the
    pitch vanishing point (gives the x axis). Metric, because stump 3D is metric.
    Returns dict or None."""
    from scorevision.miner.private_track.cricket.physics.calib import homography_for_frame  # self
    H, reproj_h, n = homography_for_frame(kp)
    if H is None:
        return None
    vp = pitch_vanishing_point(kp)
    Ki = _Kinv(f, cx, cy)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    g1, g2, g3 = Ki @ h1, Ki @ h2, Ki @ h3
    n1 = np.linalg.norm(g1); n2 = np.linalg.norm(g2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    lam = 2.0 / (n1 + n2)               # scale so |r2|=|r3|~1
    r2, r3, t = lam * g1, lam * g2, lam * g3
    r1 = np.cross(r2, r3)
    # orient x-axis (r1) consistently with the pitch VP direction if available
    if vp is not None:
        vdir = Ki @ np.array([vp[0], vp[1], 1.0]); vdir /= (np.linalg.norm(vdir) + 1e-12)
        if np.dot(r1 / (np.linalg.norm(r1) + 1e-12), vdir) < 0:
            r1 = -r1
    R0 = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    # chirality: origin (stump base) must be in front of camera (depth z_cam>0)
    if t[2] < 0:
        t = -t; R = R @ np.diag([-1, -1, 1])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    C = (-R.T @ t)
    # reproject stumps with this (R,t) to get true reproj error
    obj, img = _stump_corr(kp)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(obj, rvec, t.reshape(3, 1), K, None)
    reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
    return {"f": f, "K": K, "R": R, "t": t, "C": C, "reproj": reproj}


def _vp_alignment_cost(kp, f, cx, cy):
    """How well the homography normal (r2xr3) aligns with the pitch VP dir.
    0 = perfect. Used to solve focal robustly."""
    H, _, _ = homography_for_frame(kp)
    vp = pitch_vanishing_point(kp)
    if H is None or vp is None:
        return None
    Ki = _Kinv(f, cx, cy)
    g1, g2 = Ki @ H[:, 0], Ki @ H[:, 1]
    nrm = np.cross(g1, g2); nn = np.linalg.norm(nrm)
    vdir = Ki @ np.array([vp[0], vp[1], 1.0]); vn = np.linalg.norm(vdir)
    if nn < 1e-12 or vn < 1e-12:
        return None
    return 1.0 - abs(np.dot(nrm / nn, vdir / vn))


def estimate_focal(frames_kp, cx, cy, frange=(400, 4000, 10)):
    """Shared focal: f that best aligns homography normal with the pitch VP."""
    usable = [kp for kp in frames_kp if len(_stump_corr(kp)[0]) >= 4
              and pitch_vanishing_point(kp) is not None]
    if not usable:
        return None
    best_f, best_cost = None, np.inf
    for f in np.arange(*frange):
        costs = [c for kp in usable if (c := _vp_alignment_cost(kp, float(f), cx, cy)) is not None]
        if costs:
            m = float(np.median(costs))
            if m < best_cost:
                best_cost, best_f = m, float(f)
    return best_f


def calibrate_delivery(frames_kp_by_frame, cx, cy):
    """frames_kp_by_frame: {frame:{kp:(u,v)}}. Returns ({frame: pose}, focal)."""
    f = estimate_focal(list(frames_kp_by_frame.values()), cx, cy)
    if f is None:
        return {}, None
    poses = {}
    for fr, kp in frames_kp_by_frame.items():
        pose = solve_pose(kp, f, cx, cy)
        if pose is not None:
            vp_obs = pitch_vanishing_point(kp)
            d = pose["K"] @ pose["R"] @ np.array([1.0, 0.0, 0.0])
            vp_pred = d[:2] / d[2] if abs(d[2]) > 1e-9 else None
            pose["vp_err"] = (None if vp_obs is None or vp_pred is None
                              else float(np.linalg.norm(vp_pred - vp_obs)))
            poses[fr] = pose
    return poses, f


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from scorevision.miner.private_track.cricket.check_stumps import load_stumps_per_frame
    task = sys.argv[1] if len(sys.argv) > 1 else "Annotation/ball_batch1/P1_V12_d6"
    cx, cy = 640.0, 360.0
    pf = load_stumps_per_frame(task + "/annotations.xml")
    poses, f = calibrate_delivery(pf, cx, cy)
    if not poses:
        print("calibration failed (no usable frames)"); sys.exit()
    print(f"{task}: focal={f:.0f}px, calibrated {len(poses)} frames")
    Cs = []
    for fr in sorted(poses):
        p = poses[fr]; C = p["C"]; Cs.append(C)
        ve = "n/a" if p["vp_err"] is None else f"{p['vp_err']:.0f}px"
        print(f"  f{fr:2d}: reproj={p['reproj']:.2f}px  C=({C[0]:6.1f},{C[1]:6.1f},{C[2]:6.1f})  vp_err={ve}")
    Cs = np.array(Cs)
    print(f"\nstump reproj median: {np.median([poses[fr]['reproj'] for fr in poses]):.2f}px")
    print(f"camera center median (m): ({np.median(Cs[:,0]):.1f},{np.median(Cs[:,1]):.1f},{np.median(Cs[:,2]):.1f})")
    print(f"camera center std    (m): ({Cs[:,0].std():.2f},{Cs[:,1].std():.2f},{Cs[:,2].std():.2f})  (small=fixed camera)")
