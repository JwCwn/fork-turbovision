import json
import os
import time
import logging
from scorevision.miner.private_track.security import BLACKLIST_ENABLED, VERIFY_ENABLED
from scorevision.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("scorevision")


def log_challenge_record(challenge_id, video_url, prediction, dbg, meta, processing_time):
    """Emit ONE structured line per challenge for the scoring feedback loop.

    The validator scores each challenge but never returns the score to the miner;
    the score is only visible on the validator side (the per-challenge `Score: N`
    lines). To correlate WHAT we predicted with the score it earned we log a single
    joinable JSON record (by challenge_id / video) covering the full prediction plus
    the solve debug (rms, calib path, detection counts, timing). Goes to stdout (so
    it lands in the container logs alongside the scores) and, if CHALLENGE_LOG is set,
    is also appended to that JSONL file for durable off-box analysis."""
    try:
        pred = (prediction.model_dump() if hasattr(prediction, "model_dump")
                else prediction.dict() if hasattr(prediction, "dict")
                else dict(prediction))
    except Exception:
        pred = {}
    rec = {
        "challenge_id": challenge_id,
        "video": video_url,
        "t": round(time.time(), 1),
        "processing_time": round(processing_time, 2),
        "prediction": pred,
        "dbg": dbg or {},
        "meta": meta or {},
    }
    line = json.dumps(rec, default=str, sort_keys=True)
    logger.info("CHALLENGE_RECORD %s", line)
    path = os.environ.get("CHALLENGE_LOG")
    if path:
        try:
            with open(path, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.warning("challenge-log append failed: %s", e)


def log_startup_config() -> None:
    logger.info(f"Blacklist: {'ENABLED' if BLACKLIST_ENABLED else 'DISABLED'}")
    logger.info(f"Verify: {'ENABLED' if VERIFY_ENABLED else 'DISABLED'}")

    if BLACKLIST_ENABLED or VERIFY_ENABLED:
        logger.info("Security requires: BITTENSOR_WALLET_COLD, HOTKEY, NETUID, MIN_STAKE_THRESHOLD")

    if not BLACKLIST_ENABLED and not VERIFY_ENABLED:
        logger.warning("All security DISABLED - local testing only")
