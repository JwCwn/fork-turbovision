import logging
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import cv2
import httpx

logger = logging.getLogger(__name__)


def _collect_dir() -> Path | None:
    raw = os.environ.get("PT_COLLECT_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Cannot create collect dir {path}: {e}")
        return None
    return path


def _collection_filename(url: str) -> str:
    """Use the remote hash filename so the same URL maps to the same local file."""
    name = Path(urlparse(url).path).name
    if name and name.endswith(".mp4"):
        return name
    # Fallback
    return f"chunk-{abs(hash(url))}.mp4"


def _collect_videos_enabled() -> bool:
    return os.environ.get("PT_COLLECT_VIDEOS", "").strip().lower() in ("1", "true", "yes", "on")


async def download_video(url: str) -> Path:
    logger.info(f"Downloading video: {url}")
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(response.content)
        temp_file.close()

        # Optional: also save the video file (disabled by default, enable via
        # PT_COLLECT_VIDEOS=true). URLs are always recorded in challenges.jsonl
        # by routes.py so we can bulk-download later if desired.
        if _collect_videos_enabled():
            collect = _collect_dir()
            if collect is not None:
                dest = collect / _collection_filename(url)
                try:
                    if not dest.exists():
                        shutil.copy(temp_file.name, str(dest))
                        logger.info(f"Collected training video: {dest}")
                except Exception as e:
                    logger.warning(f"Failed to collect video to {dest}: {e}")

        return Path(temp_file.name)


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def delete_video(video_path: Path) -> None:
    try:
        video_path.unlink()
        logger.info(f"Deleted video: {video_path}")
    except Exception as e:
        logger.warning(f"Failed to delete video: {video_path}, error: {e}")
