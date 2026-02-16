"""JWT authentication for the RAG API."""

from __future__ import annotations

import time

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from rag_engine.config.settings import Settings
from rag_engine.observability.logger import get_logger

logger = get_logger("auth")

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()


class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.post("/token", response_model=TokenResponse)
async def create_token(
    body: TokenRequest,
    settings: Settings = Depends(_get_settings),
) -> TokenResponse:
    """Exchange an API key for a JWT token."""
    valid_keys = [k.strip() for k in settings.api_keys.split(",") if k.strip()]

    if not valid_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication not configured",
        )

    if body.api_key not in valid_keys:
        logger.warning("invalid_api_key_attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    now = int(time.time())
    expiry = now + settings.jwt_expiry_minutes * 60
    payload = {
        "sub": body.api_key,
        "iat": now,
        "exp": expiry,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    logger.info("token_issued", expiry_minutes=settings.jwt_expiry_minutes)
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expiry_minutes * 60,
    )


async def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """FastAPI dependency: validate JWT from Authorization header."""
    settings: Settings = request.app.state.settings
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
