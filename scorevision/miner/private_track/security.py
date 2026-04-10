import hashlib
import os
from fastapi import Depends, Header, HTTPException, Request
from substrateinterface import Keypair

BLACKLIST_ENABLED = os.environ.get("BLACKLIST_ENABLED", "true").lower() in ("true", "1", "yes")
VERIFY_ENABLED = os.environ.get("VERIFY_ENABLED", "true").lower() in ("true", "1", "yes")


async def verify_request(
    request: Request,
    validator_hotkey: str = Header(..., alias="Validator-Hotkey"),
    signature: str = Header(..., alias="Signature"),
    nonce: str = Header(..., alias="Nonce"),
):
    body = await request.body()
    payload_hash = hashlib.blake2b(body, digest_size=32).hexdigest()
    message = f"{nonce}{payload_hash}"

    try:
        keypair = Keypair(ss58_address=validator_hotkey)
        if not keypair.verify(message.encode("utf-8"), signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Signature verification failed: {e}")


def get_security_dependencies() -> list:
    deps = []

    if BLACKLIST_ENABLED:
        from fiber.miner.dependencies import blacklist_low_stake
        deps.append(Depends(blacklist_low_stake))

    if VERIFY_ENABLED:
        deps.append(Depends(verify_request))

    return deps
