# 06_middleware.py

```python
"""
========================================================
TOPIC 6: Middleware & Interceptors
========================================================

KEY CONCEPTS:
- Middleware wraps EVERY request/response like an onion.
  Order matters: first middleware added = outermost layer.
- Correlation ID → trace a single request across all logs/services.
- CORS           → allow browser-to-API calls across origins.
- GZip           → compress large responses (embeddings, documents).
- Global exception handler → catch ALL unhandled exceptions and return
                              structured JSON instead of raw stack traces.

INTERVIEW ANSWER TIP:
  "I implement custom BaseHTTPMiddleware for cross-cutting concerns like
   request tracing. Each request gets a UUID (Correlation ID) injected
   into request.state and response headers, enabling end-to-end tracing
   across microservices. I combine this with structured JSON logging so
   every log line carries the correlation_id."
"""

import json
import logging
import time
import traceback
import uuid
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

app = FastAPI(title="Middleware Demo")


# ──────────────────────────────────────────────────────────────
# 6A. Correlation ID Middleware
# ──────────────────────────────────────────────────────────────
class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique ID to every request.
    - Reads X-Correlation-ID from incoming request (from upstream gateway).
    - Generates a new UUID if not present.
    - Stores in request.state.correlation_id for use in endpoints/logs.
    - Echoes it back in the response header for client-side tracing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Read from request header or generate new
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))

        # Attach to request state — accessible in any endpoint via request.state
        request.state.correlation_id = correlation_id

        # Process the request
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Inject into response headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        # Structured log (in production, use structlog or python-json-logger)
        log_record = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }
        print(f"[ACCESS] {json.dumps(log_record)}")

        return response


# ──────────────────────────────────────────────────────────────
# 6B. Request Size Limiter Middleware
# ──────────────────────────────────────────────────────────────
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Rejects requests with bodies larger than max_bytes.
    Prevents memory exhaustion from huge payloads.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={"error": f"Request body too large. Max: {self.max_bytes} bytes"},
            )
        return await call_next(request)


# ──────────────────────────────────────────────────────────────
# 6C. Register middleware (ORDER MATTERS — last added = outermost)
# ──────────────────────────────────────────────────────────────

# GZip: compress responses > 1000 bytes (good for large embedding vectors)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS: allow the React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],           # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],           # Authorization, Content-Type, X-Correlation-ID
    expose_headers=["X-Correlation-ID", "X-Response-Time-Ms"],
)

# Request size limit
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=5 * 1024 * 1024)

# Correlation ID (outermost — wraps everything, including CORS)
app.add_middleware(CorrelationIDMiddleware)


# ──────────────────────────────────────────────────────────────
# 6D. Global Exception Handler
# ──────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches ANY unhandled exception and returns structured JSON.
    Without this, FastAPI returns a plain 500 with an HTML traceback.

    In production:
    - Log the full traceback to your logging system.
    - NEVER expose the traceback to the client (security risk).
    - Include correlation_id so you can find the full log.
    """
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    # Log full traceback server-side
    logging.error(
        "Unhandled exception",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "traceback": traceback.format_exc(),
        },
    )

    # Return safe, structured error to client
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "correlation_id": correlation_id,
            # Never include traceback here!
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle specific error types with appropriate HTTP codes."""
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "detail": str(exc)},
    )


# ──────────────────────────────────────────────────────────────
# 6E. Endpoints to demonstrate middleware
# ──────────────────────────────────────────────────────────────

@app.get("/hello")
def hello(request: Request):
    return {
        "message": "Hello!",
        "your_correlation_id": request.state.correlation_id,
    }


@app.get("/large-embedding")
def large_embedding():
    """Returns a large vector — GZip middleware will compress it."""
    return {"embedding": [0.123456] * 1536}  # 1536-dim OpenAI embedding


@app.get("/trigger-error")
def trigger_error():
    """Triggers unhandled exception — caught by global handler."""
    raise RuntimeError("Something went wrong in the ML pipeline!")


@app.get("/trigger-value-error")
def trigger_value_error():
    raise ValueError("Invalid model configuration")


if __name__ == "__main__":
    uvicorn.run("06_middleware:app", host="0.0.0.0", port=8006, reload=True)
```
