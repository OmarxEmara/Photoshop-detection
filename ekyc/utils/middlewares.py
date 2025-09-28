import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        logger.info(f"Request: {request.method} {request.url.path}")
        response: Response = await call_next(request)
        process_time = time.time() - start_time

        response.headers["X-Process-Time"] = f"{process_time:.2f} econdss"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Log response details
        logger.info(
            f"Response: status_code={response.status_code} processed_in={process_time:.2f}ms"
        )

        return response
