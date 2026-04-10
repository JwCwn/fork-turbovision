import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from scorevision.miner.private_track.logging import log_startup_config, logger
from scorevision.miner.private_track.routes import handle_challenge
from scorevision.miner.private_track.security import get_security_dependencies, BLACKLIST_ENABLED, VERIFY_ENABLED
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_startup_config()

    if BLACKLIST_ENABLED or VERIFY_ENABLED:
        try:
            from fiber.miner.core import configuration
            cfg = configuration.factory_config()
            logger.info("Performing initial metagraph sync...")
            cfg.metagraph.sync_nodes()
            logger.info(f"Metagraph synced: {len(cfg.metagraph.nodes)} nodes")

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
