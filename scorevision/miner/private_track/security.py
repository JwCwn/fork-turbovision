import hashlib
import json
import os
import time

from fastapi import Depends, Header, HTTPException, Request
from fiber.chain import signatures

from scorevision.utils.schemas import ChallengeRequest

BLACKLIST_ENABLED = os.environ.get("BLACKLIST_ENABLED", "true").lower() in ("true", "1", "yes")
VERIFY_ENABLED = os.environ.get("VERIFY_ENABLED", "true").lower() in ("true", "1", "yes")

# Reject a signature whose nonce is too far from now (replay + clock-skew guard).
_NONCE_WINDOW_NS = 180 * 1_000_000_000


async def verify_request(
    request: Request,
    validator_hotkey: str = Header(..., alias="Validator-Hotkey"),
    signature: str = Header(..., alias="Signature"),
    nonce: str = Header(..., alias="Nonce"),
):
    """Verify a validator-signed challenge.

    Must mirror the validator's build_signed_headers() exactly:
        payload_hash = blake2b(payload_bytes, 32).hexdigest()
        message      = f"{nonce}{payload_hash}"
        signature    = validator_keypair.sign(message.encode())

    The validator signs request.model_dump_json() but ships the body via httpx
    (json=model_dump()), so the wire bytes can differ in whitespace/ordering.
    We therefore try the re-serialized canonical form AND the raw body.
    """
    try:
        nonce_ns = int(nonce)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid nonce")
    if abs(time.time_ns() - nonce_ns) > _NONCE_WINDOW_NS:
        raise HTTPException(status_code=401, detail="Stale nonce")

    body = await request.body()
    candidates: list[bytes] = []
    try:
        candidates.append(ChallengeRequest(**json.loads(body)).model_dump_json().encode())
    except Exception:
        pass
    candidates.append(body)

    for payload_bytes in candidates:
        payload_hash = hashlib.blake2b(payload_bytes, digest_size=32).hexdigest()
        message = f"{nonce}{payload_hash}"
        try:
            if signatures.verify_signature(
                message=message,
                signer_ss58_address=validator_hotkey,
                signature=signature,
            ):
                return
        except Exception:
            continue

    raise HTTPException(status_code=401, detail="Invalid signature")


def get_security_dependencies() -> list:
    deps = []

    if BLACKLIST_ENABLED:
        from fiber.miner.dependencies import blacklist_low_stake
        deps.append(Depends(blacklist_low_stake))

    if VERIFY_ENABLED:
        deps.append(Depends(verify_request))

    return deps
