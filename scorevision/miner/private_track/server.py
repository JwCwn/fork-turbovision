import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from scorevision.miner.private_track.logging import log_startup_config, logger
from scorevision.miner.private_track.routes import handle_challenge
from scorevision.miner.private_track.security import BLACKLIST_ENABLED, get_security_dependencies
from scorevision.utils.schemas import ChallengeRequest, ChallengeResponse


def _start_metagraph_sync() -> None:
    """fiber's blacklist_low_stake checks the requesting validator against the
    metagraph, but fiber's factory_config only LOADS nodes.json and never syncs
    from chain. With no syncer running the metagraph stays empty, so every
    validator is rejected with 403. Sync once from chain now (so the very next
    challenge passes), then keep it fresh every 5 min in a daemon thread.

    factory_config() is @lru_cache'd, so this is the same metagraph instance
    that blacklist_low_stake reads at request time.
    """
    from fiber.miner.core.configuration import factory_config

    config = factory_config()
    mg = config.metagraph
    if getattr(mg, "substrate", None) is None:
        logger.warning("Metagraph has no substrate (REFRESH_NODES off?); cannot sync")
        return
    mg.sync_nodes()
    mg.save_nodes()
    logger.info("Initial metagraph sync complete: %d nodes", len(mg.nodes))
    threading.Thread(target=mg.periodically_sync_nodes, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_startup_config()
    if BLACKLIST_ENABLED:
        try:
            _start_metagraph_sync()
        except Exception as e:
            logger.error("Metagraph sync failed at startup: %s", e)
    yield


app = FastAPI(title="Private Track Turbovision Miner", lifespan=lifespan)


@app.post(
    "/challenge",
    response_model=ChallengeResponse,
    dependencies=get_security_dependencies(),
)
async def challenge_endpoint(request: ChallengeRequest) -> ChallengeResponse:
    return await handle_challenge(request)
