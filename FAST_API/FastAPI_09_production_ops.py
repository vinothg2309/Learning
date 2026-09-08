# 09_production_ops.py

```python
"""
========================================================
TOPIC 9: Production Operations
========================================================

KEY CONCEPTS:
- Gunicorn + Uvicorn workers:
    Gunicorn = process manager (handles worker restarts, signals).
    Uvicorn  = ASGI worker (handles async requests).
    Command: gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
    Rule of thumb: workers = 2 × CPU_cores + 1 (for I/O-bound)
                   For ML (GPU): workers = 1 (GPU is the bottleneck)

- Structured JSON Logging:
    Every log line is JSON. Enables log aggregation (Datadog, Splunk, ELK).
    Include: timestamp, level, message, correlation_id, service name.

- Health Check Probes (Kubernetes):
    Liveness  → "Is the app alive? Should I restart it?"
                Fails → Kubernetes kills & restarts the pod.
    Readiness → "Is the app ready to receive traffic?"
                Fails → Kubernetes removes pod from load balancer.
                Use for: model loading in progress, warming up cache.
    Startup   → "Has the app finished initializing?"
                Prevents premature liveness checks during slow startup.

INTERVIEW ANSWER TIP:
  "I deploy FastAPI with gunicorn -k UvicornWorker for process management.
   I expose /healthz/live and /healthz/ready endpoints. The readiness
   probe returns 503 while the model is loading, preventing traffic before
   the model is warm. All logs are JSON-structured with correlation IDs."
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse


# ──────────────────────────────────────────────────────────────
# 9A. Structured JSON Logging
# ──────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON objects.
    Enables log aggregation tools to parse and query logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": "fastapi-ml-service",
            "environment": os.getenv("ENVIRONMENT", "development"),
        }

        # Include extra fields (e.g., correlation_id passed via logging.info(..., extra={}))
        for key in ("correlation_id", "user_id", "model_name", "latency_ms"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        # Include exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logging():
    """Configure root logger to output JSON. Call once at startup."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)

    # Suppress noisy uvicorn access logs (we have our own middleware)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 9B. App state for health tracking
# ──────────────────────────────────────────────────────────────

class AppState:
    """Singleton holding the app's runtime state."""
    is_ready: bool = False       # True after model load
    is_alive: bool = True        # False only if catastrophic failure
    model_loaded: bool = False
    startup_time: str | None = None
    request_count: int = 0
    error_count: int = 0


app_state = AppState()


# ──────────────────────────────────────────────────────────────
# 9C. Lifespan — model loading with readiness tracking
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: model is NOT ready until load completes.
    The readiness probe returns 503 during this window.
    Kubernetes will not route traffic here until is_ready = True.
    """
    logger.info("Starting up ML service", extra={"phase": "startup"})
    app_state.startup_time = datetime.now(timezone.utc).isoformat()

    # Simulate slow model loading (e.g., downloading weights)
    logger.info("Loading ML model...")
    import asyncio
    await asyncio.sleep(2)  # replace with actual model load
    app_state.model_loaded = True
    app_state.is_ready = True
    logger.info("Model loaded. Service is ready.", extra={"model": "llama-3-8b"})

    yield  # serve requests

    # Shutdown
    logger.info("Shutting down ML service")
    app_state.is_ready = False
    app_state.model_loaded = False
    # cleanup: unload model, close connections


app = FastAPI(title="Production Ops Demo", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────
# 9D. Health Check Probes (Kubernetes-compatible)
# ──────────────────────────────────────────────────────────────

@app.get("/healthz/live", tags=["Health"])
def liveness_probe():
    """
    Liveness probe — Kubernetes calls this to decide if the pod should be restarted.
    Should ONLY fail if the app is fundamentally broken (deadlock, OOM, etc.).
    Keep it SIMPLE — never check external dependencies here.
    Response: 200 = alive, 503 = restart me.

    Kubernetes config:
        livenessProbe:
          httpGet:
            path: /healthz/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 3
    """
    if not app_state.is_alive:
        return Response(
            content=json.dumps({"status": "dead", "reason": "catastrophic failure"}),
            status_code=503,
            media_type="application/json",
        )
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/healthz/ready", tags=["Health"])
def readiness_probe():
    """
    Readiness probe — Kubernetes calls this to decide if traffic should be routed here.
    Fails during model loading, warm-up, or when dependencies are down.
    Response: 200 = ready, 503 = not ready (remove from load balancer).

    Kubernetes config:
        readinessProbe:
          httpGet:
            path: /healthz/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          failureThreshold: 3
    """
    checks = {
        "model_loaded": app_state.model_loaded,
        "is_ready": app_state.is_ready,
        # Add more checks: DB connection, Redis connection, etc.
    }

    all_ready = all(checks.values())

    if not all_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": checks,
                "message": "Service is initializing. Retry in a few seconds.",
            },
        )

    return {
        "status": "ready",
        "checks": checks,
        "startup_time": app_state.startup_time,
        "uptime_seconds": _get_uptime(),
    }


@app.get("/healthz/startup", tags=["Health"])
def startup_probe():
    """
    Startup probe — checked only during initial startup.
    Allows slow-starting apps (large model downloads) without
    triggering premature liveness restarts.

    Kubernetes config:
        startupProbe:
          httpGet:
            path: /healthz/startup
            port: 8000
          failureThreshold: 30   # 30 × 10s = 5 minutes max startup time
          periodSeconds: 10
    """
    if app_state.model_loaded:
        return {"status": "started"}
    return JSONResponse(
        status_code=503,
        content={"status": "starting", "message": "Model is still loading..."},
    )


def _get_uptime() -> float:
    """Calculate uptime in seconds since startup."""
    if not app_state.startup_time:
        return 0
    start = datetime.fromisoformat(app_state.startup_time)
    return (datetime.now(timezone.utc) - start).total_seconds()


# ──────────────────────────────────────────────────────────────
# 9E. Metrics endpoint (Prometheus-compatible)
# ──────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["Observability"])
def metrics():
    """
    Basic metrics. In production use: pip install prometheus-fastapi-instrumentator
    and it auto-exposes /metrics in Prometheus format.
    """
    return {
        "requests_total": app_state.request_count,
        "errors_total": app_state.error_count,
        "uptime_seconds": _get_uptime(),
        "model_loaded": app_state.model_loaded,
    }


# ──────────────────────────────────────────────────────────────
# 9F. Request counting middleware
# ──────────────────────────────────────────────────────────────

@app.middleware("http")
async def count_requests(request: Request, call_next):
    app_state.request_count += 1
    response = await call_next(request)
    if response.status_code >= 500:
        app_state.error_count += 1
    return response


# ──────────────────────────────────────────────────────────────
# 9G. Sample endpoint with structured logging
# ──────────────────────────────────────────────────────────────

@app.post("/infer")
async def infer(prompt: str, request: Request):
    correlation_id = getattr(request.state, "correlation_id", "no-id")

    start = time.perf_counter()
    # ... model inference ...
    import asyncio
    await asyncio.sleep(0.1)  # simulate inference
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Inference complete",
        extra={
            "correlation_id": correlation_id,
            "model_name": "llama-3-8b",
            "latency_ms": round(latency_ms, 2),
        },
    )

    return {"response": f"Result for: {prompt}", "latency_ms": latency_ms}


# ──────────────────────────────────────────────────────────────
# 9H. Gunicorn + Uvicorn deployment command
# ──────────────────────────────────────────────────────────────
"""
PRODUCTION DEPLOYMENT:

# Install
pip install gunicorn uvicorn

# Run with 4 workers (tune based on CPU cores & workload)
gunicorn 09_production_ops:app \\
    --workers 4 \\
    --worker-class uvicorn.workers.UvicornWorker \\
    --bind 0.0.0.0:8000 \\
    --timeout 120 \\
    --keep-alive 5 \\
    --access-logfile - \\
    --error-logfile - \\
    --log-level info

# For ML (GPU) workloads: use 1 worker (GPU is the bottleneck)
gunicorn 09_production_ops:app \\
    --workers 1 \\
    --worker-class uvicorn.workers.UvicornWorker \\
    --bind 0.0.0.0:8000 \\
    --timeout 300  # long timeout for inference

# Dockerfile snippet:
# CMD ["gunicorn", "app:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
"""


if __name__ == "__main__":
    uvicorn.run("09_production_ops:app", host="0.0.0.0", port=8009, reload=False)
```
