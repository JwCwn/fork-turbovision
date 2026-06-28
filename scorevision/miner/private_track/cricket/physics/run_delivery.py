"""Run the gravity-anchored BA on one real delivery and report the 6 fields."""
import sys, numpy as np, cv2
from scorevision.miner.private_track.cricket.check_stumps import load_stumps_per_frame
from scorevision.miner.private_track.cricket.tracknet.cvat_io import parse_cvat_video_xml
from scorevision.miner.private_track.cricket.physics import bundle as B
from scorevision.miner.private_track.cricket.physics.calib import STUMP_3D

def build_obs(task):
    pf = load_stumps_per_frame(task + "/annotations.xml")
    ball = parse_cvat_video_xml(task + "/annotations.xml", "ball")
    obs = {}
    frames = sorted(set(pf) | set(f for f in ball if ball[f].x is not None))
    for fr in frames:
        o = {}
        kp = pf.get(fr, {})
        st = {n: np.array(kp[n]) for n in STUMP_3D if n in kp}
        if len(st) >= 4: o["stumps"] = st
        pe = {k: np.array(kp[k]) for k in ["pitch_left_far","pitch_left_near","pitch_right_far","pitch_right_near"] if k in kp}
        if len(pe) == 4: o["pitch"] = pe
        if fr in ball and ball[fr].x is not None:
            o["ball"] = np.array([ball[fr].x, ball[fr].y])
        if o: obs[fr] = o
    return obs

def init_params(cx, cy, kph):
    C = [42., 0., 8.]
    target = np.array([8.,0.,0.5]); Cn=np.array(C); fwd=target-Cn; fwd/=np.linalg.norm(fwd)
    up=np.array([0,0,1.]); right=np.cross(fwd,up); right/=np.linalg.norm(right); tu=np.cross(right,fwd)
    r0 = cv2.Rodrigues(np.vstack([right,-tu,fwd]))[0].ravel()
    v = (kph or 120.0)/3.6
    return np.concatenate([C, r0, [0,0,0], [0,0,0], [3000.],
                           [18,0,2.5], [-v,0,-2.], [0.8], [-v*0.95,0,5.], [1.5]])

if __name__ == "__main__":
    import glob, cv2
    task = sys.argv[1] if len(sys.argv)>1 else "Annotation/ball_batch1/P2_V3_d38"
    kph = float(sys.argv[2]) if len(sys.argv)>2 else None  # None -> no kph anchor
    # auto-detect resolution -> principal point
    jp = sorted(glob.glob(task + "/*.jpg"))
    h, w = cv2.imread(jp[0]).shape[:2]
    cx, cy, fps = w / 2.0, h / 2.0, 25.0
    print(f"resolution {w}x{h} (cx,cy={cx:.0f},{cy:.0f})")
    obs = build_obs(task)
    nb = sum('ball' in o for o in obs.values()); ns = sum('stumps' in o for o in obs.values()); npi = sum('pitch' in o for o in obs.values())
    print(f"{task}: frames={len(obs)} ball={nb} stumps>=4={ns} pitch={npi}  kph_obs={kph}")
    init = init_params(cx, cy, kph)
    sol = B.fit(obs, cx, cy, fps, init, kph_obs=kph)
    prm = B.unpack(sol.x)
    r = B.residuals(sol.x, obs, cx, cy, fps, kph_obs=kph)
    print(f"BA: success={sol.success} cost={sol.cost:.1f} nfev={sol.nfev} rms_resid={np.sqrt(np.mean(r**2)):.2f}")
    print(f"  focal={prm['f']:.0f}  camC=({prm['C'][0]:.1f},{prm['C'][1]:.1f},{prm['C'][2]:.1f})")
    f6 = B.fields_from(prm, fps, kph_obs=kph)
    print("  6 fields:")
    for k,v in f6.items(): print(f"    {k:12s} {v:8.3f}")
