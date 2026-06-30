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

# both sides up to 3 digits: Australian notation writes wickets-RUNS ("3-103"),
# so the second field can be a 3-digit run total. _disambiguate_score sorts out
# which side is runs vs wickets via the wickets<=10 rule.
_SCORE_RE = re.compile(r"(?<!\d)(\d{1,3})-(\d{1,3})(?!\d)")
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


def _disambiguate_score(a: int, b: int):
    """An 'A-B' score token -> (runs, wickets) using the rule that WICKETS are
    always 0..10. Broadcasts differ: English writes runs-wickets ('103-3'),
    Australian writes wickets-runs ('3-103'), and bowler figures look the same
    ('0-20'). The side that exceeds 10 must be runs; the <=10 side is wickets."""
    if a > 10 and b <= 10:
        return a, b          # runs-wickets (English)
    if b > 10 and a <= 10:
        return b, a          # wickets-runs (Australian)
    if a <= 10 and b <= 10:
        return a, b          # ambiguous early score -> assume runs-wickets
    return None              # both > 10: not a valid score


def parse_meta(tokens: list[str]) -> dict:
    """Extract team/runs/wickets/overs from a token list. Returns partial dict."""
    out: dict = {}

    # --- team score: collect every plausible A-B, disambiguate runs/wickets via
    #     the wickets<=10 rule, then take the candidate with the MOST runs. The
    #     batting team's cumulative total exceeds any single bowler's runs
    #     conceded, so max-runs picks the team score over bowler figures. -------
    team = None
    for tok in tokens:
        if _TEAMCODE_RE.match(tok) and tok not in {"SPEED", "KMH", "KPH"}:
            team = tok
            break
    candidates = []
    for tok in tokens:
        if "(" in tok:  # bowler figures sometimes carry "(overs)" -> skip
            continue
        for m in _SCORE_RE.finditer(tok):
            # Reject bowler figures, which max-runs would otherwise prefer when OCR
            # mangles them into inflated totals: they are glued to the bowler NAME
            # ("RABADA0-19", "NGIDI0-7") or carry a trailing ".overs" ("0-19.2",
            # "0-192.1"). The team total stands alone with no adjacent letter/decimal.
            if m.start() > 0 and tok[m.start() - 1].isalpha():
                continue
            if m.end() < len(tok) and tok[m.end()] == ".":
                continue
            rw = _disambiguate_score(int(m.group(1)), int(m.group(2)))
            if rw is not None:
                candidates.append(rw)
    team_score = max(candidates, key=lambda rw: rw[0]) if candidates else None

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
