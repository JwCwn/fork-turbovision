import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from scorevision.miner.private_track.logging import log_startup_config, logger
from scorevision.miner.private_track.routes import handle_challenge
from scorevision.miner.private_track.security import get_security_dependencies, BLACKLIST_ENABLED, VERIFY_ENABLED
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse


def _initial_sync_with_retry(cfg, max_attempts: int = 5, delay_s: float = 5.0) -> bool:
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Performing initial metagraph sync (attempt {attempt}/{max_attempts})...")
            cfg.metagraph.sync_nodes()
            node_count = len(cfg.metagraph.nodes)
            if node_count > 0:
                logger.info(f"Metagraph synced: {node_count} nodes")
                return True
            logger.warning(f"Sync returned 0 nodes on attempt {attempt}")
        except Exception as e:
            logger.warning(f"Sync attempt {attempt} failed: {type(e).__name__}: {e}")
        if attempt < max_attempts:
            time.sleep(delay_s)
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_startup_config()

    if BLACKLIST_ENABLED or VERIFY_ENABLED:
        try:
            from fiber.miner.core import configuration
            cfg = configuration.factory_config()

            if not _initial_sync_with_retry(cfg):
                logger.error(
                    "Initial metagraph sync failed after retries. "
                    "Starting background sync thread anyway — challenges will fail until first sync succeeds."
                )

            sync_thread = threading.Thread(
                target=cfg.metagraph.periodically_sync_nodes,
                daemon=True,
            )
            sync_thread.start()
            logger.info("Background metagraph sync thread started")
        except Exception as e:
            logger.error(f"Failed to start metagraph sync: {e}")

    yield


app = FastAPI(title="Private Track Turbovision Miner", lifespan=lifespan)


@app.post(
    "/challenge",
    response_model=ChallengeResponse,
    dependencies=get_security_dependencies(),
)
async def challenge_endpoint(request: ChallengeRequest) -> ChallengeResponse:
    return await handle_challenge(request)
