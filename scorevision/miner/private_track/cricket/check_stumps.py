"""
Geometric QA for stump keypoint labels (calibration needs clean points).

Checks per delivery, for each end (bs = batter-end, ws = bowler-end):
  - per stump: top above base (top_y < base_y); post ~vertical (|dx| small vs height)
  - across the 3 stumps: bases at similar image-y (same depth), tops similar,
    heights consistent, left_x < mid_x < right_x ordering.
Flags violations so they can be re-labeled before calibration.
"""
from __future__ import annotations
import glob, os
import xml.etree.ElementTree as ET


def load_stumps_per_frame(xml_path):
    """Return {frame: {kp_name: (x,y)}}. Camera pans/zooms within a delivery, so
    geometry must be validated per-frame, not aggregated across frames."""
    root = ET.parse(xml_path).getroot()
    out = {}
    for tr in root.iter("track"):
        if tr.get("label") != "stump_pt":
            continue
        for p in tr.iter("points"):
            if p.get("outside") != "0":
                continue
            name = next((a.text for a in p.iter("attribute") if a.get("name") == "kp"), None)
            if not name:
                continue
            f = int(p.get("frame"))
            out.setdefault(f, {})[name] = tuple(float(v) for v in p.get("points").split(",")[:2])
    return out


def load_stumps(xml_path):
    """Back-compat: merged across frames (do NOT use for geometry checks)."""
    merged = {}
    for kp in load_stumps_per_frame(xml_path).values():
        merged.update(kp)
    return merged


def check_end(kp, end):
    """end='bs' or 'ws'. Returns list of issue strings."""
    issues = []
    cols = ["left", "mid", "right"]
    heights = {}
    for c in cols:
        b = kp.get(f"{end}_{c}_base")
        t = kp.get(f"{end}_{c}_top")
        if b and t:
            h = b[1] - t[1]  # base_y - top_y, should be > 0
            if h <= 0:
                issues.append(f"{end}_{c}: base/top inverted or coincident (base_y={b[1]:.0f} top_y={t[1]:.0f})")
            else:
                heights[c] = h
                if abs(b[0] - t[0]) > 0.6 * h:
                    issues.append(f"{end}_{c}: post not vertical (dx={abs(b[0]-t[0]):.0f} vs h={h:.0f})")
    # bases similar depth (image-y)
    bases = {c: kp.get(f"{end}_{c}_base") for c in cols if kp.get(f"{end}_{c}_base")}
    if len(bases) >= 2:
        bys = [v[1] for v in bases.values()]
        if max(bys) - min(bys) > 0.5 * (max(heights.values()) if heights else 60):
            issues.append(f"{end} bases differ in y by {max(bys)-min(bys):.0f}px (expect ~same depth)")
    # tops similar
    tops = {c: kp.get(f"{end}_{c}_top") for c in cols if kp.get(f"{end}_{c}_top")}
    if len(tops) >= 2:
        tys = [v[1] for v in tops.values()]
        if max(tys) - min(tys) > 0.5 * (max(heights.values()) if heights else 60):
            issues.append(f"{end} tops differ in y by {max(tys)-min(tys):.0f}px")
    # height consistency
    if len(heights) >= 2 and min(heights.values()) > 0:
        if max(heights.values()) / min(heights.values()) > 1.8:
            issues.append(f"{end} stump heights inconsistent: {{" +
                          ", ".join(f'{c}:{h:.0f}' for c, h in heights.items()) + "}")
    # x ordering left<mid<right (use base, fallback top)
    xs = {}
    for c in cols:
        pt = kp.get(f"{end}_{c}_base") or kp.get(f"{end}_{c}_top")
        if pt:
            xs[c] = pt[0]
    order = [xs[c] for c in cols if c in xs]
    if len(order) >= 2 and order != sorted(order):
        issues.append(f"{end} x order not left<mid<right: {{" +
                      ", ".join(f'{c}:{xs[c]:.0f}' for c in cols if c in xs) + "}")
    return issues


def main(batch="Annotation/ball_batch1"):
    anns = sorted(glob.glob(f"{batch}/*/annotations.xml"))
    n_with = 0
    for a in anns:
        task = os.path.basename(os.path.dirname(a))
        per_frame = load_stumps_per_frame(a)
        if not per_frame:
            continue
        n_with += 1
        # validate each labeled frame; report frames with issues
        bad_frames = {}
        for f, kp in sorted(per_frame.items()):
            issues = check_end(kp, "bs") + check_end(kp, "ws")
            if issues:
                bad_frames[f] = issues
        nf = len(per_frame)
        if bad_frames:
            # show a couple of representative bad frames
            sample = list(bad_frames.items())[:2]
            print(f"FLAG {task}: {len(bad_frames)}/{nf} frames flagged. e.g.")
            for f, iss in sample:
                print(f"    frame {f}: {iss[0]}")
        else:
            print(f"OK   {task}: all {nf} labeled frames geometrically consistent")
    print(f"\n(stump deliveries: {n_with})")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "Annotation/ball_batch1")
