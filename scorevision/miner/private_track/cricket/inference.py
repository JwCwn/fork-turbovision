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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2
import torch

from scorevision.miner.private_track.logging import logger
from scorevision.miner.private_track.cricket.tracknet.model import TrackNetV2
from scorevision.miner.private_track.cricket.tracknet.infer import (
    frame_to_cache, infer_task_topk, infer_cache_topk,
)
from scorevision.miner.private_track.cricket.physics import bundle as B
from scorevision.miner.private_track.cricket.physics.run_delivery import init_params
from scorevision.miner.private_track.cricket.physics import pipeline as PL

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_BALL = _PKG_DIR / "tracknet" / "ckpt_wb.pt"
_DEFAULT_KP = _PKG_DIR / "keypoints" / "ckpt_kp.pt"
_SCRATCH = Path(tempfile.gettempdir()) / "cricket_miner"
# The live delivery (release -> batter) sits in the opening seconds of every
# challenge clip; the rest is replays / other cameras. Geometry only searches
# this opening window. OCR (kph + metadata) is scanned separately and wider.
# 7 s covers release (~3-4 s) through impact (~6 s) with margin, while staying
# before the first replay; raise only if a clip's live delivery runs later.
DELIVERY_SECS = 7.0

# The 6 core + 7 secondary geometry fields produced by the physics stage.
GEOM_FIELDS = [
    "kph", "bounce_x", "stump_y", "stump_z", "swing_angle", "deviation",
    "release_y", "release_z", "bounce_y", "impact_x", "impact_y",
    "impact_z", "interception_distance",
]

# Physical envelopes (metres / degrees) used to reject degenerate solves: a value
# outside its range is reconstruction garbage, not a real estimate, so it is nulled.
_PHYS_RANGE = {
    "bounce_x": (-2.0, 25.0), "stump_y": (-3.0, 3.0), "stump_z": (-1.0, 4.0),
    "swing_angle": (-25.0, 25.0), "deviation": (-25.0, 25.0),
    "release_y": (-3.0, 3.0), "release_z": (0.0, 4.0), "bounce_y": (-3.0, 3.0),
    "impact_y": (-3.0, 3.0), "impact_z": (-1.0, 4.0),
    "interception_distance": (-25.0, 25.0),
}


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
        tg0 = time.perf_counter()
        # top-k ball candidates -> trajectory selection (robust to bright static
        # distractors); temporal-aggregated keypoints (stable >=4-5 stumps).
        bcands = infer_task_topk(self.mb, task, self.device, self.nfb)
        ball = PL.select_ball_track(bcands, w0, h0)
        tg1 = time.perf_counter()
        kps, _, _ = PL.detect_keypoints_robust(self.mk, task, self.device)
        tg2 = time.perf_counter()
        ballw = PL.delivery_window(PL.clean_ball(ball))
        kpc = PL.clean_kps(kps)
        if not ballw:
            return None, dict(reason="no ball window")
        fr0 = min(ballw)
        obs = {f - fr0: o for f, o in PL.build_obs(ballw, kpc).items()}
        nb = sum("ball" in o for o in obs.values())
        ns = sum("stumps" in o for o in obs.values())
        npi = sum("pitch" in o for o in obs.values())
        # Pitch-primary (solve from pitch when stumps absent) was tried in production
        # and FAILED: with no stumps to anchor the batter-end origin, the elongated
        # pitch quad leaves a gauge ambiguity and the solve goes degenerate (rms>60,
        # release flipped to x=0, V1[0]->0) -> scored 0 AND took ~38 s. So require
        # stumps again; stump-less clips fall back to the (independent) OCR floor.
        if nb < 4 or ns < 3:
            return None, dict(reason=f"insufficient det ball={nb} stumps={ns} pitch={npi}")
        sol = B.fit(obs, cx, cy, fps, init_params(cx, cy, kph), kph_obs=kph)
        tg3 = time.perf_counter()
        logger.info(
            "[timing] geom.ball=%.2fs geom.kp=%.2fs geom.fit=%.2fs (window_frames=%d)",
            tg1 - tg0, tg2 - tg1, tg3 - tg2, len(frames),
        )
        prm = B.unpack(sol.x)
        r = B.residuals(sol.x, obs, cx, cy, fps, kph_obs=kph)
        rms = float(np.sqrt(np.mean(r ** 2)))
        f = B.fields_from(prm, fps, kph_obs=kph)
        dbg = dict(rms=rms, focal=float(prm["f"]),
                   camC=[round(float(v), 1) for v in prm["C"]], n_ball=nb, n_stump=ns,
                   n_pitch=npi, res=f"{w0}x{h0}")
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
    def ocr_meta(self, video_path: Path, step_secs=1.0, max_samples=16):
        """Read scoreboard metadata + the speed-gun kph from the overlay.

        The speed graphic flashes ~14 s in, so OCR walks the opening ~max_samples
        seconds and STOPS once kph is read (the persistent scoreboard metadata is
        captured in the first samples). CPU OCR is ~1 s/frame, so max_samples is kept
        small and the band is downscaled — a whole-clip scan cost 20 s+ (timeouts).
        Missing a late kph only drops to the OCR floor; a timeout scores 0."""
        from scorevision.miner.private_track.cricket.scorecard_ocr import (
            extract_band, ocr_tokens, parse_meta, parse_speed,
        )
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        from collections import Counter, defaultdict
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(step_secs * fps_v))
        # Metadata is voted (most-common) over the DELIVERY window only — runs/overs
        # advance later in the clip, so a whole-clip vote would drift off the ball we
        # are scored on. kph is taken from its first plausible reading anywhere.
        meta_cutoff = int(8.0 * fps_v)
        votes = defaultdict(Counter)
        kph = None
        fi = 0; count = 0
        # Hard wall-clock budget so OCR NEVER causes a timeout regardless of the host's
        # per-frame OCR speed (a whole-clip scan hit 20 s+ on the deployed CPU box).
        t_ocr0 = time.perf_counter()
        while fi < max(n, 1) and count < max_samples and time.perf_counter() - t_ocr0 < 10.0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if not ok:
                break
            band = extract_band(fr, 0.80)
            if band.shape[1] > 1000:  # downscale wide (HD) bands -> ~2x faster OCR
                s = 1000.0 / band.shape[1]
                band = cv2.resize(band, (1000, int(band.shape[0] * s)))
            toks = ocr_tokens(self._ocr, band)
            if fi <= meta_cutoff:
                for k, v in parse_meta(toks).items():
                    votes[k][v] += 1
            if kph is None:
                kph = parse_speed(toks)
            if kph is not None and fi > meta_cutoff:
                break  # have the speed and the delivery-window metadata is covered
            fi += step; count += 1
        cap.release()
        best = {k: c.most_common(1)[0][0] for k, c in votes.items()}
        if kph is not None:
            best["kph"] = kph
        return best

    # ---- full prediction from a raw video -----------------------------------
    def predict_video(self, video_path, fps=25.0):
        video_path = Path(video_path)
        t0 = time.perf_counter()
        # OCR is independent of the perception/physics path, so run it on a
        # worker thread: its ~7s overlaps the extract pass instead of adding to
        # the critical path. geometry() needs meta["kph"], so we join the OCR
        # result before the physics solve (which runs after extract anyway).
        with ThreadPoolExecutor(max_workers=1) as ex:
            ocr_future = ex.submit(self.ocr_meta, video_path)
            task, seg = self._extract_delivery(video_path, fps)
            t_extract = time.perf_counter()
            meta = ocr_future.result()
            t_ocr = time.perf_counter()
        geom, dbg = self.geometry(task, meta.get("kph"), fps)
        t_geom = time.perf_counter()
        logger.info(
            "[timing] extract=%.2fs ocr_wait=%.2fs geometry=%.2fs total=%.2fs",
            t_extract - t0, t_ocr - t_extract, t_geom - t_ocr, t_geom - t0,
        )
        dbg["seg"] = seg
        dbg["timing"] = {
            "extract": round(t_extract - t0, 2),
            "ocr_wait": round(t_ocr - t_extract, 2),
            "geometry": round(t_geom - t_ocr, 2),
        }
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
        # Clear any stale frames from a prior run: the no-window paths return this
        # dir, and geometry() must NOT solve on leftover frames (that produced
        # bogus high-rms "solves"). Pass 1 writes nothing here, so on no-window
        # `out` stays empty -> geometry returns None cleanly.
        for old in out.glob("*.jpg"):
            old.unlink()
        # Pass 1: decode the LIVE DELIVERY ONLY (first DELIVERY_SECS) into the
        # in-memory inference cache. The live ball + batter-end stumps + pitch all
        # appear in the opening few seconds; everything after is replays / slow-mo
        # / other cameras whose different geometry corrupts the solve. Restricting
        # to the opening window both removes that distractor footage AND keeps the
        # search tiny (~125 frames), so it always runs full-res 288x512 (best ball
        # recall) and finishes in seconds instead of scanning the whole clip.
        tx0 = time.perf_counter()
        cap = cv2.VideoCapture(str(video_path))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or fps or 25.0
        max_frames = int(DELIVERY_SECS * fps_v)
        SEARCH_HW = (288, 512)
        cache, dims = [], None
        while len(cache) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            if dims is None:
                dims = fr.shape[:2]
            cache.append(frame_to_cache(fr, out_hw=SEARCH_HW))
        cap.release()
        i = len(cache)
        tx1 = time.perf_counter()
        if dims is None:
            return out, dict(n_frames=0, n_ball=0)
        h0, w0 = dims
        bcands = infer_cache_topk(self.mb, cache, self.device, self.nfb, h0, w0,
                                  out_hw=SEARCH_HW)
        ball = PL.select_ball_track(bcands, w0, h0)
        tx2 = time.perf_counter()
        logger.info(
            "[timing] extract.decode=%.2fs extract.infer=%.2fs (n_frames=%d)",
            tx1 - tx0, tx2 - tx1, i,
        )
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
        # Pass 2: re-decode and write only the ~12 window frames at full res
        # for the geometry stage (keypoints + per-window ball).
        cap = cv2.VideoCapture(str(video_path))
        k = 0
        while k <= hi:
            ok, fr = cap.read()
            if not ok:
                break
            if lo <= k <= hi:
                cv2.imwrite(str(win / f"{k:06d}.jpg"), fr)
            k += 1
        cap.release()
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
            ok = v is not None and np.isfinite(v) and abs(v) < 1e4
            # Physical-range guard: a degenerate solve (e.g. 3 collinear bases, no
            # vertical reference) can fit the image well yet reconstruct absurd 3D
            # (stump_z=-18 m). Null any field outside its physical envelope so a
            # bad solve falls back to "no geometry" instead of emitting garbage
            # that pretends to be an answer.
            lo_hi = _PHYS_RANGE.get(f)
            if ok and lo_hi and not (lo_hi[0] <= v <= lo_hi[1]):
                ok = False
            pred[f] = round(float(v), 4) if ok else None
        # kph is read straight off the broadcast overlay, so emit it whenever OCR
        # found it — even when the physics solve fails. It overrides the BA echo
        # when geometry succeeded, and is the only core field we can score when it
        # didn't (partial credit: emitting is never worse than null).
        if meta.get("kph"):
            pred["kph"] = round(float(meta["kph"]), 2)
        return pred
