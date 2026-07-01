from pathlib import Path

from scorevision.miner.private_track.video import get_frame_count
from scorevision.miner.private_track.logging import logger
from scorevision.utils.schemas import CricketDeliveryPrediction, FramePrediction

# Lazily-built singleton so model weights load once per container (not per request).
_CRICKET_MINER = None


def _get_cricket_miner():
    global _CRICKET_MINER
    if _CRICKET_MINER is None:
        from scorevision.miner.private_track.cricket.inference import CricketMiner
        logger.info("Loading cricket models (ckpt_wb + ckpt_kp)...")
        _CRICKET_MINER = CricketMiner()
        logger.info("Cricket models loaded on device=%s", _CRICKET_MINER.device)
    return _CRICKET_MINER


def predict_actions(video_path: Path) -> list[FramePrediction]:
    frame_count = get_frame_count(video_path)

    # TODO: Replace this with your actual prediction logic
    # This example predicts "pass" on every 25th frame
    predictions = []
    for frame in range(0, frame_count, 25):
        predictions.append(FramePrediction(frame=frame, action="pass"))

    return predictions


def predict_cricket_delivery(video_path: Path) -> CricketDeliveryPrediction:
    """Run the prepared ball-tracking + physics pipeline on a downloaded video.

    Returns a CricketDeliveryPrediction with every scored field; geometry comes
    from the physics solve, meta/kph from the broadcast overlay (OCR). Missing
    fields are emitted as None and simply score 0, so a partial solve never hurts.
    """
    miner = _get_cricket_miner()
    try:
        pred, dbg, meta = miner.predict_video(video_path)
        logger.info("Cricket solve: %s", dbg)
    except Exception as e:
        # Never 500 the validator on hard footage; return an all-None row (score 0).
        logger.error("Cricket inference failed, returning empty prediction: %s", e)
        pred, dbg, meta = {}, {}, {}

    # dbg/meta returned too so the caller can log the full challenge record (the
    # miner half of the scoring feedback loop) joined with challenge_id/video_url.
    return CricketDeliveryPrediction(**pred), dbg, meta
