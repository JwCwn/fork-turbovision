import math
import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

from scorevision.utils.schemas import FramePrediction

logger = getLogger(__name__)

PLAYER_MODEL_NAME = "football-player-detection.pt"
BALL_MODEL_NAME = "football-ball-detection.pt"
DEFAULT_MODELS_DIR = Path("/app/models")

PLAYER_MODEL_BALL_CLASS = 0
PLAYER_MODEL_GOALKEEPER_CLASS = 1
PLAYER_MODEL_PLAYER_CLASS = 2
PLAYER_MODEL_REFEREE_CLASS = 3
BALL_MODEL_BALL_CLASS = 0

_MODELS: tuple[YOLO, YOLO | None] | None = None
_DEVICE: str | None = None


@dataclass(frozen=True)
class Detection:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class FrameDetections:
    frame: int
    width: int
    height: int
    players: list[Detection]
    balls: list[Detection]


@dataclass(frozen=True)
class PossessionPoint:
    frame: int
    owner_x: float
    owner_y: float
    ball_x: float
    ball_y: float


@dataclass(frozen=True)
class PossessionSegment:
    start_frame: int
    end_frame: int
    owner_x: float
    owner_y: float
    ball_x: float
    ball_y: float


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _get_models_dir() -> Path:
    raw = os.environ.get("MINER_MODELS_DIR")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_MODELS_DIR


def _load_models() -> tuple[YOLO, YOLO | None]:
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    models_dir = _get_models_dir()
    player_path = models_dir / PLAYER_MODEL_NAME
    ball_path = models_dir / BALL_MODEL_NAME

    if not player_path.exists():
        raise FileNotFoundError(f"Missing player model: {player_path}")

    player_model = YOLO(player_path)
    use_ball_model = _env_bool("PRIVATE_TRACK_USE_BALL_MODEL", True)
    ball_model = YOLO(ball_path) if use_ball_model and ball_path.exists() else None
    if use_ball_model and ball_model is None:
        logger.warning("Ball model not found at %s; using player model ball class", ball_path)
    logger.info(
        "Loaded YOLO models from %s on %s (dedicated_ball_model=%s)",
        models_dir,
        _get_device(),
        ball_model is not None,
    )
    _MODELS = (player_model, ball_model)
    return _MODELS


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _read_sampled_frames(video_path: Path) -> tuple[list[tuple[int, Any]], float, int]:
    stride = max(1, _env_int("PRIVATE_TRACK_SAMPLE_STRIDE", 5))
    max_frames = max(1, _env_int("PRIVATE_TRACK_MAX_SAMPLED_FRAMES", 180))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    logger.info(
        "Video metadata: fps=%.3f total_frames=%d duration=%.2fs resolution=%dx%d",
        fps,
        total_frames,
        duration,
        width,
        height,
    )

    sampled_frames: list[tuple[int, Any]] = []
    frame_idx = 0

    try:
        while len(sampled_frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % stride == 0:
                sampled_frames.append((frame_idx, frame))
            frame_idx += 1
    finally:
        cap.release()

    return sampled_frames, fps, total_frames


def _parse_detections(result, frame_number: int) -> list[Detection]:
    if not hasattr(result, "boxes") or result.boxes is None:
        return []

    detections: list[Detection] = []
    for box in result.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = box
        detections.append(
            Detection(
                frame=frame_number,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=float(conf),
                cls_id=int(cls_id),
            )
        )
    return detections


def _run_detection_batch(
    frame_batch: list[tuple[int, Any]],
) -> list[FrameDetections]:
    player_model, ball_model = _load_models()
    device = _get_device()
    imgs = [frame for _, frame in frame_batch]

    player_results = player_model.predict(
        imgs,
        device=device,
        verbose=False,
        conf=_env_float("PRIVATE_TRACK_PLAYER_CONF", 0.30),
        imgsz=_env_int("PRIVATE_TRACK_PLAYER_IMGSZ", 960),
        max_det=_env_int("PRIVATE_TRACK_PLAYER_MAX_DET", 48),
    )

    ball_results = None
    if ball_model is not None:
        ball_results = ball_model.predict(
            imgs,
            device=device,
            verbose=False,
            conf=_env_float("PRIVATE_TRACK_BALL_CONF", 0.20),
            imgsz=_env_int("PRIVATE_TRACK_BALL_IMGSZ", 1280),
            max_det=_env_int("PRIVATE_TRACK_BALL_MAX_DET", 8),
        )

    parsed: list[FrameDetections] = []
    for idx, (frame_number, image) in enumerate(frame_batch):
        player_dets = _parse_detections(player_results[idx], frame_number)
        player_candidates = [
            det
            for det in player_dets
            if det.cls_id in (PLAYER_MODEL_PLAYER_CLASS, PLAYER_MODEL_GOALKEEPER_CLASS)
        ]
        fallback_balls = [
            det for det in player_dets if det.cls_id == PLAYER_MODEL_BALL_CLASS
        ]

        dedicated_balls: list[Detection] = []
        if ball_results is not None:
            dedicated_balls = [
                det
                for det in _parse_detections(ball_results[idx], frame_number)
                if det.cls_id == BALL_MODEL_BALL_CLASS
            ]

        parsed.append(
            FrameDetections(
                frame=frame_number,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                players=player_candidates,
                balls=dedicated_balls or fallback_balls,
            )
        )

    return parsed


def _run_detector(video_path: Path) -> tuple[list[FrameDetections], float, int]:
    sampled_frames, fps, total_frames = _read_sampled_frames(video_path)
    if not sampled_frames:
        return [], fps, total_frames

    batch_size = max(1, _env_int("PRIVATE_TRACK_BATCH_SIZE", 12))
    frame_detections: list[FrameDetections] = []
    for start in range(0, len(sampled_frames), batch_size):
        batch = sampled_frames[start : start + batch_size]
        frame_detections.extend(_run_detection_batch(batch))
    return frame_detections, fps, total_frames


def _select_ball(frame_data: FrameDetections) -> Detection | None:
    if not frame_data.balls:
        return None
    return max(frame_data.balls, key=lambda det: (det.conf, -det.area))


def _find_owner(frame_data: FrameDetections, ball: Detection) -> Detection | None:
    if not frame_data.players:
        return None

    frame_diag = math.hypot(frame_data.width, frame_data.height)
    max_owner_distance = frame_diag * _env_float(
        "PRIVATE_TRACK_OWNER_DISTANCE_RATIO", 0.12
    )

    best_player: Detection | None = None
    best_distance = float("inf")
    for player in frame_data.players:
        dist = _distance(player.cx, player.cy, ball.cx, ball.cy)
        if dist < best_distance:
            best_distance = dist
            best_player = player

    if best_player is None or best_distance > max_owner_distance:
        return None
    return best_player


def _build_possession_points(
    detections: list[FrameDetections],
) -> tuple[list[PossessionPoint], list[tuple[int, float]]]:
    possession_points: list[PossessionPoint] = []
    free_ball_points: list[tuple[int, float]] = []
    previous_ball: Detection | None = None
    previous_frame: int | None = None

    for frame_data in detections:
        ball = _select_ball(frame_data)
        if ball is None:
            previous_ball = None
            previous_frame = frame_data.frame
            continue

        owner = _find_owner(frame_data, ball)
        if owner is not None:
            possession_points.append(
                PossessionPoint(
                    frame=frame_data.frame,
                    owner_x=owner.cx,
                    owner_y=owner.cy,
                    ball_x=ball.cx,
                    ball_y=ball.cy,
                )
            )
        elif previous_ball is not None and previous_frame is not None:
            frame_delta = max(1, frame_data.frame - previous_frame)
            ball_speed = _distance(
                ball.cx,
                ball.cy,
                previous_ball.cx,
                previous_ball.cy,
            ) / frame_delta
            free_ball_points.append((frame_data.frame, ball_speed))

        previous_ball = ball
        previous_frame = frame_data.frame

    return possession_points, free_ball_points


def _merge_possessions(points: list[PossessionPoint]) -> list[PossessionSegment]:
    if not points:
        return []

    owner_merge_distance = _env_float("PRIVATE_TRACK_OWNER_MERGE_DISTANCE", 70.0)
    max_gap = max(1, _env_int("PRIVATE_TRACK_SEGMENT_MAX_GAP", 30))

    segments: list[PossessionSegment] = []
    current = PossessionSegment(
        start_frame=points[0].frame,
        end_frame=points[0].frame,
        owner_x=points[0].owner_x,
        owner_y=points[0].owner_y,
        ball_x=points[0].ball_x,
        ball_y=points[0].ball_y,
    )

    for point in points[1:]:
        owner_shift = _distance(
            current.owner_x,
            current.owner_y,
            point.owner_x,
            point.owner_y,
        )
        frame_gap = point.frame - current.end_frame

        if owner_shift <= owner_merge_distance and frame_gap <= max_gap:
            current = PossessionSegment(
                start_frame=current.start_frame,
                end_frame=point.frame,
                owner_x=(current.owner_x + point.owner_x) / 2.0,
                owner_y=(current.owner_y + point.owner_y) / 2.0,
                ball_x=(current.ball_x + point.ball_x) / 2.0,
                ball_y=(current.ball_y + point.ball_y) / 2.0,
            )
            continue

        segments.append(current)
        current = PossessionSegment(
            start_frame=point.frame,
            end_frame=point.frame,
            owner_x=point.owner_x,
            owner_y=point.owner_y,
            ball_x=point.ball_x,
            ball_y=point.ball_y,
        )

    segments.append(current)
    return segments


def _infer_actions(
    possessions: list[PossessionSegment],
    free_ball_points: list[tuple[int, float]],
) -> list[FramePrediction]:
    predictions: list[FramePrediction] = []
    min_owner_change = _env_float("PRIVATE_TRACK_PASS_OWNER_CHANGE", 130.0)
    max_transition_gap = max(1, _env_int("PRIVATE_TRACK_PASS_MAX_GAP", 40))
    shot_speed_threshold = _env_float("PRIVATE_TRACK_SHOT_SPEED_THRESHOLD", 16.0)

    for previous, current in zip(possessions, possessions[1:]):
        owner_change = _distance(
            previous.owner_x,
            previous.owner_y,
            current.owner_x,
            current.owner_y,
        )
        frame_gap = current.start_frame - previous.end_frame

        if owner_change < min_owner_change or frame_gap > max_transition_gap:
            continue

        pass_frame = max(previous.start_frame, previous.end_frame)
        receive_frame = current.start_frame
        predictions.append(
            FramePrediction(frame=int(pass_frame), action="pass", confidence=0.55)
        )
        predictions.append(
            FramePrediction(
                frame=int(receive_frame),
                action="pass_received",
                confidence=0.60,
            )
        )

    for segment in possessions:
        candidate_speeds = [
            speed
            for frame, speed in free_ball_points
            if segment.end_frame < frame <= segment.end_frame + max_transition_gap
        ]
        if not candidate_speeds:
            continue
        if max(candidate_speeds) < shot_speed_threshold:
            continue
        predictions.append(
            FramePrediction(frame=int(segment.end_frame), action="shot", confidence=0.52)
        )

    deduped: dict[tuple[str, int], FramePrediction] = {}
    suppress_gap = max(1, _env_int("PRIVATE_TRACK_ACTION_SUPPRESS_GAP", 25))
    for prediction in sorted(predictions, key=lambda item: (item.action, item.frame)):
        key = (prediction.action, prediction.frame)
        existing = next(
            (
                prev
                for (action, frame), prev in deduped.items()
                if action == prediction.action
                and abs(frame - prediction.frame) <= suppress_gap
            ),
            None,
        )
        if existing is None or prediction.confidence > existing.confidence:
            if existing is not None:
                deduped.pop((existing.action, existing.frame), None)
            deduped[key] = prediction

    return sorted(deduped.values(), key=lambda item: item.frame)


def _limit_predictions(predictions: list[FramePrediction]) -> list[FramePrediction]:
    if not predictions:
        return []

    per_action_defaults = {
        "pass": 8,
        "pass_received": 6,
        "shot": 1,
    }
    kept: list[FramePrediction] = []

    for action, default_limit in per_action_defaults.items():
        limit = max(
            0,
            _env_int(
                f"PRIVATE_TRACK_MAX_{action.upper()}",
                default_limit,
            ),
        )
        action_predictions = [pred for pred in predictions if pred.action == action]
        action_predictions.sort(key=lambda pred: pred.confidence, reverse=True)
        kept.extend(action_predictions[:limit])

    other_predictions = [
        pred for pred in predictions if pred.action not in per_action_defaults
    ]
    kept.extend(other_predictions)

    max_total = max(0, _env_int("PRIVATE_TRACK_MAX_PREDICTIONS", 15))
    kept.sort(key=lambda pred: pred.confidence, reverse=True)
    kept = kept[:max_total]
    return sorted(kept, key=lambda pred: pred.frame)


def _count_by_action(predictions: list[FramePrediction]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for prediction in predictions:
        counts[prediction.action] = counts.get(prediction.action, 0) + 1
    return counts


VALIDATOR_FRAME_RATE = 25.0


def _rescale_frame_to_validator(frame: int, source_fps: float) -> int:
    if source_fps <= 0 or abs(source_fps - VALIDATOR_FRAME_RATE) < 0.01:
        return int(frame)
    seconds = frame / source_fps
    return int(round(seconds * VALIDATOR_FRAME_RATE))


def predict_actions(video_path: Path) -> list[FramePrediction]:
    detections, source_fps, total_frames = _run_detector(video_path)
    if not detections:
        return []

    possession_points, free_ball_points = _build_possession_points(detections)
    possession_segments = _merge_possessions(possession_points)
    raw_predictions = _infer_actions(possession_segments, free_ball_points)
    predictions = _limit_predictions(raw_predictions)

    if source_fps > 0 and abs(source_fps - VALIDATOR_FRAME_RATE) >= 0.01:
        rescaled = [
            FramePrediction(
                frame=_rescale_frame_to_validator(p.frame, source_fps),
                action=p.action,
                confidence=p.confidence,
            )
            for p in predictions
        ]
        logger.info(
            "Rescaled %d predictions from source_fps=%.3f to validator_fps=%.1f",
            len(predictions),
            source_fps,
            VALIDATOR_FRAME_RATE,
        )
        predictions = rescaled

    logger.info(
        "Prediction summary: source_fps=%.3f total_frames=%d sampled_frames=%d "
        "possession_points=%d segments=%d raw_predictions=%d kept_predictions=%d actions=%s",
        source_fps,
        total_frames,
        len(detections),
        len(possession_points),
        len(possession_segments),
        len(raw_predictions),
        len(predictions),
        _count_by_action(predictions),
    )
    return predictions
