"""
CVAT <-> TrackNet label IO (no torch).

Reads 'CVAT for video 1.1' XML exports where the ball is a single point track
labeled 'ball' with an optional 'vis' attribute, and yields per-frame positions.
Also writes the same XML so model pseudo-labels can be imported back into CVAT
for human correction (active-learning loop).

CVAT 'for video 1.1' structure (point track):
  <annotations>
    <track id="0" label="ball">
      <points frame="12" outside="0" occluded="0" points="cx,cy">
        <attribute name="vis">1</attribute>
      </points>
      ...
    </track>
  </annotations>
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BallLabel:
    frame: int
    x: float | None     # None when outside/invisible
    y: float | None
    vis: int            # 0 invisible, 1 clear, 2 blurred


def parse_cvat_video_xml(xml_path: str | Path, label_name: str = "ball") -> dict[int, BallLabel]:
    """Return {frame -> BallLabel} for the given point-track label."""
    root = ET.parse(xml_path).getroot()
    out: dict[int, BallLabel] = {}
    for track in root.iter("track"):
        if track.get("label") != label_name:
            continue
        for pts in track.iter("points"):
            # Only outside=0 points mark a visible ball. outside=1 are CVAT track
            # terminators (and would otherwise clobber the next frame's keyframe);
            # frames with no outside=0 point are simply invisible (absent).
            if pts.get("outside", "0") == "1":
                continue
            xy = pts.get("points", "")
            if not xy:
                continue
            frame = int(pts.get("frame", -1))
            vis = 1
            for attr in pts.iter("attribute"):
                if attr.get("name") == "vis" and attr.text:
                    try:
                        vis = int(attr.text)
                    except ValueError:
                        pass
            cx, cy = (float(v) for v in xy.split(",")[:2])
            out[frame] = BallLabel(frame, cx, cy, vis)
    return out


def write_cvat_video_xml(
    out_path: str | Path,
    labels: dict[int, BallLabel],
    task_name: str = "pseudo",
    width: int = 1920,
    height: int = 1080,
    label_name: str = "ball",
    n_frames: int | None = None,
) -> None:
    """Write a minimal 'CVAT for video 1.1' XML with one point track."""
    ann = ET.Element("annotations")
    ver = ET.SubElement(ann, "version")
    ver.text = "1.1"
    meta = ET.SubElement(ann, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(len(labels))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str((n_frames - 1) if n_frames else (max(labels) if labels else 0))
    orig = ET.SubElement(task, "original_size")
    ET.SubElement(orig, "width").text = str(width)
    ET.SubElement(orig, "height").text = str(height)
    # CVAT needs the label defined in meta (else import: "reading 'attributes'")
    labels_el = ET.SubElement(task, "labels")
    lbl = ET.SubElement(labels_el, "label")
    ET.SubElement(lbl, "name").text = label_name
    ET.SubElement(lbl, "type").text = "points"
    attrs = ET.SubElement(lbl, "attributes")
    at = ET.SubElement(attrs, "attribute")
    ET.SubElement(at, "name").text = "vis"
    ET.SubElement(at, "mutable").text = "True"
    ET.SubElement(at, "input_type").text = "select"
    ET.SubElement(at, "default_value").text = "1"
    ET.SubElement(at, "values").text = "1\n2\n0"

    # Mirror CVAT's export structure: ONE track per visible point (a keyframe at
    # frame N + an "outside" terminator at N+1), with z_order and source=manual.
    # Invisible frames are simply omitted (no track), exactly as CVAT does.
    # last valid frame index: terminator must not exceed it (CVAT errors otherwise)
    last_valid = (n_frames - 1) if n_frames else max(labels)
    tid = 0
    for frame in sorted(labels):
        lb = labels[frame]
        if lb.x is None:
            continue
        track = ET.SubElement(ann, "track", id=str(tid), label=label_name, source="manual")
        tid += 1
        coord = f"{lb.x:.2f},{lb.y:.2f}"
        kf = ET.SubElement(track, "points", frame=str(frame), keyframe="1",
                           outside="0", occluded="0", points=coord, z_order="0")
        ET.SubElement(kf, "attribute", name="vis").text = str(lb.vis)
        # terminator at frame+1, but only if it stays within the clip
        if frame + 1 <= last_valid:
            term = ET.SubElement(track, "points", frame=str(frame + 1), keyframe="1",
                                 outside="1", occluded="0", points=coord, z_order="0")
            ET.SubElement(term, "attribute", name="vis").text = str(lb.vis)
    ET.indent(ann, space="  ")
    ET.ElementTree(ann).write(out_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    import tempfile
    labels = {
        0: BallLabel(0, 100.0, 50.0, 1),
        1: BallLabel(1, 110.5, 48.2, 2),
        2: BallLabel(2, None, None, 0),
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.xml"
        write_cvat_video_xml(p, labels, width=1280, height=720)
        rt = parse_cvat_video_xml(p)
        assert rt[0].x == 100.0 and rt[0].vis == 1
        assert rt[1].vis == 2 and abs(rt[1].x - 110.5) < 1e-6
        assert rt.get(2) is None  # invisible frames are omitted, not stored
    print("cvat_io round-trip self-check passed")
