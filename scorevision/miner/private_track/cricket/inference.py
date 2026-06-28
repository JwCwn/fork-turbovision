"""
Cricket delivery miner inference (private track, subnet 44).

Adapted from model-tunning/Scripts/miner.py. Wraps the prepared perception +
physics stack (TrackNet ball ckpt_wb, stump/pitch keypoints ckpt_kp,
gravity-anchored bundle adjustment) plus scorecard OCR into a single
`CricketMiner.predict_video(video_path) -> dict` entrypoint that the private
track predictor maps onto a CricketDeliveryPrediction.

Model weights live alongside this file (tracknet/ckpt_wb.pt, keypoints/ckpt_kp.pt)
so the package is self-contained inside the Docker image.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import cv2
import torch

from scorevision.miner.private_track.cricket.tracknet.model import TrackNetV2
from scorevision.miner.private_track.cricket.tracknet.infer import infer_task
from scorevision.miner.private_track.cricket.physics import bundle as B
from scorevision.miner.private_track.cricket.physics.run_delivery import init_params
from scorevision.miner.private_track.cricket.physics import pipeline as PL

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_BALL = _PKG_DIR / "tracknet" / "ckpt_wb.pt"
_DEFAULT_KP = _PKG_DIR / "keypoints" / "ckpt_kp.pt"
_SCRATCH = Path(tempfile.gettempdir()) / "cricket_miner"

# The 6 core + 7 secondary geometry fields produced by the physics stage.
GEOM_FIELDS = [
    "kph", "bounce_x", "stump_y", "stump_z", "swing_angle", "deviation",
    "release_y", "release_z", "bounce_y", "impact_x", "impact_y",
    "impact_z", "interception_distance",
]


class CricketMiner:
    def __init__(self, device=None, ckpt_ball=_DEFAULT_BALL, ckpt_kp=_DEFAULT_KP):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sdb = torch.load(str(ckpt_ball), map_location=self.device)
        self.nfb = sdb.get("n_frames", 3)
        self.mb = TrackNetV2(n_frames=self.nfb).to(self.device).eval()
        self.mb.load_state_dict(sdb["model"])
        sdk = torch.load(str(ckpt_kp), map_location=self.device)
        self.K = sdk["out_ch"]
        self.mk = TrackNetV2(n_frames=1, out_ch=self.K).to(self.device).eval()
        self.mk.load_state_dict(sdk["model"])
        self._ocr = None

    # ---- perception + physics ------------------------------------------------
    def geometry(self, task: Path, kph, fps=25.0):
        frames = sorted(task.glob("*.jpg"))
        if not frames:
            return None, {}
        h0, w0 = cv2.imread(str(frames[0])).shape[:2]
        cx, cy = w0 / 2.0, h0 / 2.0
        ball, _, _ = infer_task(self.mb, task, self.device, self.nfb, thresh=0.8)
        kps, _, _ = PL.detect_keypoints(self.mk, task, self.device)
        ballw = PL.delivery_window(PL.clean_ball(ball))
        kpc = PL.clean_kps(kps)
        if not ballw:
            return None, dict(reason="no ball window")
        fr0 = min(ballw)
        obs = {f - fr0: o for f, o in PL.build_obs(ballw, kpc).items()}
        nb = sum("ball" in o for o in obs.values())
        ns = sum("stumps" in o for o in obs.values())
        if nb < 4 or ns < 3:
            return None, dict(reason=f"insufficient det ball={nb} stumps={ns}")
        sol = B.fit(obs, cx, cy, fps, init_params(cx, cy, kph), kph_obs=kph)
        prm = B.unpack(sol.x)
        r = B.residuals(sol.x, obs, cx, cy, fps, kph_obs=kph)
        rms = float(np.sqrt(np.mean(r ** 2)))
        f = B.fields_from(prm, fps, kph_obs=kph)
        dbg = dict(rms=rms, focal=float(prm["f"]),
                   camC=[round(float(v), 1) for v in prm["C"]], n_ball=nb, n_stump=ns,
                   res=f"{w0}x{h0}")
        dbg["low_confidence"] = self._low_conf(f, rms, ns)
        return f, dbg

    @staticmethod
    def _low_conf(f, rms, ns):
        """Heuristics that flag an under-constrained solve (for triage only)."""
        reasons = []
        if ns < 5:
            reasons.append(f"stumps={ns}")
        if rms > 60:
            reasons.append(f"rms={rms:.0f}")
        ranges = {"bounce_x": (-2, 25), "stump_y": (-3, 3), "stump_z": (-1, 4),
                  "swing_angle": (-25, 25), "deviation": (-25, 25)}
        for k, (lo, hi) in ranges.items():
            v = f.get(k)
            if v is None or not np.isfinite(v) or not (lo <= v <= hi):
                reasons.append(f"{k}={v}")
        return reasons

    # ---- OCR meta ------------------------------------------------------------
    def ocr_meta(self, video_path: Path, win=(0.2, 0.8)):
        from scorevision.miner.private_track.cricket.scorecard_ocr import (
            extract_band, ocr_tokens, parse_meta, parse_speed,
        )
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        best = {}
        for frac in np.linspace(win[0], win[1], 8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
            ok, fr = cap.read()
            if not ok:
                continue
            toks = ocr_tokens(self._ocr, extract_band(fr, 0.80))
            for k, v in parse_meta(toks).items():
                best.setdefault(k, v)
            s = parse_speed(toks)
            if s and "kph" not in best:
                best["kph"] = s
        cap.release()
        return best

    # ---- full prediction from a raw video -----------------------------------
    def predict_video(self, video_path, fps=25.0):
        video_path = Path(video_path)
        meta = self.ocr_meta(video_path)
        task, seg = self._extract_delivery(video_path, fps)
        geom, dbg = self.geometry(task, meta.get("kph"), fps)
        dbg["seg"] = seg
        return self._assemble(meta, geom), dbg, meta

    def predict_task(self, task, kph=None, meta=None):
        geom, dbg = self.geometry(Path(task), kph)
        m = dict(meta or {})
        if kph is not None:
            m.setdefault("kph", kph)
        return self._assemble(m, geom), dbg, m

    # ---- helpers -------------------------------------------------------------
    def _extract_delivery(self, video_path, fps, min_arc=6, min_path=120.0):
        """Find the real delivery in (possibly multi-delivery) footage.

        Ball-detect every frame, split into contiguous runs, extract each run's
        cleaned monotonic-descending arc, and pick the longest-travelling one
        (a real delivery sweeps across frame; stationary false positives barely
        move). Returns (window_folder, seg_debug)."""
        out = _SCRATCH / ("miner_" + video_path.stem[:12])
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.jpg"):
            old.unlink()
        cap = cv2.VideoCapture(str(video_path))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out / f"{i:06d}.jpg"), fr)
            i += 1
        cap.release()
        ball, _, _ = infer_task(self.mb, out, self.device, self.nfb, thresh=0.8)
        hit = sorted(f for f in ball if ball[f].x is not None)
        seg = dict(n_frames=i, n_ball=len(hit))
        if len(hit) < 5:
            return out, seg
        runs, cur = [], [hit[0]]
        for a, b in zip(hit, hit[1:]):
            if b - a <= 6:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)

        def arc_path(af, arc):
            xs = [arc[f].x for f in af]; ys = [arc[f].y for f in af]
            return sum(((xs[k] - xs[k - 1]) ** 2 + (ys[k] - ys[k - 1]) ** 2) ** 0.5
                       for k in range(1, len(af)))

        best, best_score = None, 0.0
        cands = []
        for run in runs:
            sub = {f: ball[f] for f in run}
            arc = PL.delivery_window(PL.clean_ball(sub))
            af = sorted(f for f in arc if arc[f].x is not None)
            if len(af) < min_arc:
                continue
            path = arc_path(af, arc)
            cands.append((af[0], af[-1], len(af), round(path)))
            if path >= min_path and path > best_score:
                best_score, best = path, af
        seg["candidates"] = sorted(cands, key=lambda c: -c[3])[:5]
        if best is None:
            return out, seg
        seg["chosen"] = (best[0], best[-1], round(best_score))
        lo, hi = best[0] - 2, best[-1] + 2
        win = out.parent / (out.name + "_win")
        win.mkdir(exist_ok=True)
        for old in win.glob("*.jpg"):
            old.unlink()
        frames = sorted(out.glob("*.jpg"))
        for k, fp in enumerate(frames):
            if lo <= k <= hi:
                cv2.imwrite(str(win / fp.name), cv2.imread(str(fp)))
        return win, seg

    @staticmethod
    def _assemble(meta, geom):
        pred = dict(
            match=None, matchid=None, inningsid=None,
            overid=meta.get("overid"), ball_in_over=meta.get("ball_in_over"),
            ballid=None, xlsx_overs=None, scorecard_overs=meta.get("scorecard_overs"),
            runs=meta.get("runs"), wickets=meta.get("wickets"),
        )
        for f in GEOM_FIELDS:
            v = geom.get(f) if geom else None
            pred[f] = round(float(v), 4) if (v is not None and np.isfinite(v) and abs(v) < 1e4) else None
        if geom and meta.get("kph"):
            pred["kph"] = round(float(meta["kph"]), 2)  # trust overlay over BA echo
        return pred
