"""FastAPI middleware for request timing, error handling, and request IDs."""

from __future__ import annotations

import time
from uuid import uuid4

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from rag_engine.observability.logger import get_logger

logger = get_logger("middleware")


class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid4())
        start = time.monotonic()

        # Bind request ID to structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        try:
            response = await call_next(request)
            duration_ms = (time.monotonic() - start) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Duration-MS"] = str(round(duration_ms, 2))
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            return response
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise
