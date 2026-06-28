"""
Scorecard / speed overlay OCR  ->  per-delivery meta + kph "ground truth".

Phase-0 offline tool (determinism irrelevant here). Reads the bottom broadcast
band at/around each delivery's contact frame and extracts:
  team, runs, wickets, overid, ball_in_over, scorecard_overs, kph

Reality (verified on samples): the 9 players come from DIFFERENT broadcasts, so
the scoreboard graphic differs per video. Layout is stable WITHIN a video, but
not across. We therefore avoid fixed templates and parse OCR tokens with generic
patterns + disambiguation heuristics. Treat outputs as noisy proxy GT and always
sanity-check with score_simulator before trusting.

Notes on ambiguity:
* Team total "164-5" looks identical to bowler figures "0-38" / "2-29".
  Heuristic: the team score is the `D+-D+` token adjacent to a leading team code
  and WITHOUT trailing parentheses (bowler figures carry "(overs)").
* Overs "54.4" -> over 54 complete + ball 4 of over 55. We expose the raw string
  as scorecard_overs and split overid/ball_in_over with the common convention
  overid = int(part_before_dot), ball_in_over = int(part_after_dot).
  (If the validator GT uses a different convention this is a 1-line change.)
* Speed appears INTERMITTENTLY and only on some balls; we scan a frame window
  after contact and take the first plausible km/h reading.
"""
from __future__ import annotations

import re
import csv
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

FPS = 25

_SCORE_RE = re.compile(r"(?<!\d)(\d{1,3})-(\d{1,2})(?!\d)")
_OVERS_RE = re.compile(r"(?:overs?\s*)(\d{1,3})(?:\.(\d))?", re.I)
_OVERS_BARE_RE = re.compile(r"(?<!\d)(\d{1,3})\.(\d)(?!\d)")  # fallback "54.4"
_SPEED_RE = re.compile(r"(\d{2,3}(?:\.\d)?)\s*(?:km/?h|kmph|kph)", re.I)
_TEAMCODE_RE = re.compile(r"^[A-Z]{2,4}$")


def _norm(t: str) -> str:
    return t.strip()


def extract_band(frame, top_frac: float = 0.80):
    h = frame.shape[0]
    return frame[int(h * top_frac):h, :]


def ocr_tokens(ocr, img) -> list[str]:
    res, _ = ocr(img)
    if not res:
        return []
    return [_norm(t[1]) for t in res]


def parse_speed(tokens: list[str]) -> float | None:
    joined = " ".join(tokens)
    m = _SPEED_RE.search(joined)
    if m:
        try:
            v = float(m.group(1))
            if 40.0 <= v <= 170.0:  # plausible delivery speed
                return v
        except ValueError:
            pass
    return None


def parse_meta(tokens: list[str]) -> dict:
    """Extract team/runs/wickets/overs from a token list. Returns partial dict."""
    out: dict = {}

    # --- team code + team score (disambiguate from bowler figures) -----------
    team = None
    team_score = None  # (runs, wickets)
    for i, tok in enumerate(tokens):
        if _TEAMCODE_RE.match(tok) and tok not in {"SPEED", "KMH", "KPH"}:
            # look at this token and the next for an adjacent score w/o parens
            for j in (i, i + 1):
                if j < len(tokens):
                    cand = tokens[j]
                    if "(" in cand:  # bowler figures like "0-38(15)" -> skip
                        continue
                    m = _SCORE_RE.search(cand)
                    if m and team is None:
                        team = tok
                        team_score = (int(m.group(1)), int(m.group(2)))
                        break
            if team:
                break
    # fallback: first standalone score token without parentheses
    if team_score is None:
        for tok in tokens:
            if "(" in tok:
                continue
            m = _SCORE_RE.fullmatch(tok) or _SCORE_RE.search(tok)
            if m:
                team_score = (int(m.group(1)), int(m.group(2)))
                break

    if team:
        out["team"] = team
    if team_score:
        out["runs"], out["wickets"] = team_score

    # --- overs ---------------------------------------------------------------
    overs_str = None
    for tok in tokens:
        m = _OVERS_RE.search(tok)
        if m:
            overs_str = m.group(1) + (("." + m.group(2)) if m.group(2) else "")
            break
    if overs_str is None:
        for tok in tokens:
            if "(" in tok:
                continue
            m = _OVERS_BARE_RE.fullmatch(tok)
            if m:
                overs_str = f"{m.group(1)}.{m.group(2)}"
                break
    if overs_str is not None:
        out["scorecard_overs"] = overs_str
        if "." in overs_str:
            whole, ball = overs_str.split(".")
            out["overid"] = int(whole)
            out["ball_in_over"] = int(ball)
        else:
            out["overid"] = int(overs_str)

    return out


@dataclass
class MetaRow:
    video: str
    delivery_id: int
    contact_frame: int
    team: str | None = None
    runs: int | None = None
    wickets: int | None = None
    overid: int | None = None
    ball_in_over: int | None = None
    scorecard_overs: str | None = None
    kph: float | None = None
    raw_tokens: str = ""


def process_delivery(ocr, video_path: Path, contact_frame: int,
                     speed_scan=(0, 75, 5), top_frac=0.80) -> tuple[dict, float | None, list[str]]:
    """OCR meta at contact_frame; scan a post-contact window for a speed reading."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, contact_frame)
    ok, frame = cap.read()
    meta, tokens = {}, []
    if ok:
        tokens = ocr_tokens(ocr, extract_band(frame, top_frac))
        meta = parse_meta(tokens)

    kph = parse_speed(tokens)
    if kph is None:
        start, stop, step = speed_scan
        for df in range(start, stop, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, contact_frame + df)
            ok, frame = cap.read()
            if not ok:
                break
            t = ocr_tokens(ocr, extract_band(frame, top_frac))
            kph = parse_speed(t)
            if kph is not None:
                break
    cap.release()
    return meta, kph, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="Scripts/delivery_index.csv")
    ap.add_argument("--videos", default="Videos")
    ap.add_argument("--out", default="Scripts/meta_gt.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = all deliveries")
    ap.add_argument("--per-video", type=int, default=0,
                    help="if >0, only first N deliveries per video (fast survey)")
    ap.add_argument("--top-frac", type=float, default=0.80)
    args = ap.parse_args()

    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    rows = list(csv.DictReader(open(args.index)))
    if args.per_video:
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            c = seen.get(r["video"], 0)
            if c < args.per_video:
                kept.append(r)
                seen[r["video"]] = c + 1
        rows = kept
    if args.limit:
        rows = rows[: args.limit]

    out_rows: list[MetaRow] = []
    got_kph = got_overs = got_score = 0
    for n, r in enumerate(rows, 1):
        vp = Path(args.videos) / f"{r['video']}.mp4"
        if not vp.exists():
            continue
        cf = int(r["contact_frame"])
        meta, kph, tokens = process_delivery(ocr, vp, cf, top_frac=args.top_frac)
        mr = MetaRow(
            video=r["video"], delivery_id=int(r["delivery_id"]), contact_frame=cf,
            team=meta.get("team"), runs=meta.get("runs"), wickets=meta.get("wickets"),
            overid=meta.get("overid"), ball_in_over=meta.get("ball_in_over"),
            scorecard_overs=meta.get("scorecard_overs"), kph=kph,
            raw_tokens=" | ".join(tokens),
        )
        out_rows.append(mr)
        got_kph += kph is not None
        got_overs += meta.get("scorecard_overs") is not None
        got_score += meta.get("runs") is not None
        if n % 25 == 0:
            print(f"  {n}/{len(rows)}  kph={got_kph} overs={got_overs} score={got_score}")

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(out_rows[0]).keys()) if out_rows else [])
        w.writeheader()
        for mr in out_rows:
            w.writerow(asdict(mr))
    print(f"\nprocessed {len(out_rows)} deliveries -> {out}")
    print(f"  score(runs-wkts): {got_score}  overs: {got_overs}  kph: {got_kph}")


if __name__ == "__main__":
    main()
