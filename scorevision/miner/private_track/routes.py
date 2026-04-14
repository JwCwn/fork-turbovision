import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import HTTPException
from scorevision.miner.private_track.predictor import predict_actions
from scorevision.miner.private_track.video import delete_video, download_video
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse
from scorevision.miner.private_track.logging import logger


def _log_challenge_metadata(
    request: ChallengeRequest,
    predictions: list,
    processing_time: float,
) -> None:
    """Append a JSON line per challenge with metadata + our predictions.

    Used later to pseudo-label collected videos for fine-tuning. Only writes
    when PT_COLLECT_DIR is set.
    """
    collect_dir = os.environ.get("PT_COLLECT_DIR", "").strip()
    if not collect_dir:
        return
    collect_path = Path(collect_dir)
    try:
        collect_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    video_filename = Path(urlparse(request.video_url).path).name
    entry = {
        "challenge_id": request.challenge_id,
        "video_url": request.video_url,
        "video_filename": video_filename,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "processing_time_s": round(processing_time, 3),
        "predictions": [
            {
                "frame": p.frame,
                "action": p.action,
                "confidence": round(p.confidence, 4),
            }
            for p in predictions
        ],
    }

    try:
        with open(collect_path / "challenges.jsonl", "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log challenge metadata: {e}")


async def handle_challenge(request: ChallengeRequest) -> ChallengeResponse:
    logger.info(f"Challenge received: {request.challenge_id}")
    start_time = time.perf_counter()
    video_path = None

    try:
        video_path = await download_video(request.video_url)
        predictions = predict_actions(video_path)
        processing_time = time.perf_counter() - start_time

        logger.info(
            f"Challenge completed: {request.challenge_id}, "
            f"predictions: {len(predictions)}, time: {processing_time:.1f}s"
        )

        _log_challenge_metadata(request, predictions, processing_time)

        return ChallengeResponse(
            challenge_id=request.challenge_id,
            predictions=predictions,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Challenge failed: {request.challenge_id}, error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if video_path:
            delete_video(video_path)


