# 10_scaling_resilience.py

```python
"""
========================================================
TOPIC 10: Scaling & Resilience
========================================================

KEY CONCEPTS:
- Rate Limiting (Redis sliding window):
    Limit requests per client per time window.
    Redis stores per-client counters with TTL.
    Returns 429 Too Many Requests when exceeded.

- Circuit Breaker:
    Prevents cascade failures. When an upstream AI API fails repeatedly,
    the circuit "opens" — subsequent calls fail immediately without
    waiting for timeout. After a cooldown, the circuit "half-opens"
    to test if the service recovered.

- Timeout:
    Long AI API calls must have timeouts. Without them, a slow upstream
    can exhaust your connection pool, crashing the service.
    Use httpx timeout or asyncio.wait_for().

INTERVIEW ANSWER TIP:
  "For rate limiting I use a Redis sliding window counter per IP.
   For resilience against flaky AI APIs (OpenAI, Anthropic), I implement
   a circuit breaker: after 5 consecutive failures, I open the circuit
   for 60 seconds to stop hammering a down service. Timeouts ensure
   we fail fast instead of hanging. This combination — rate limiting +
   circuit breaker + timeout — makes the service production-grade."
"""

import asyncio
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

app = FastAPI(title="Scaling & Resilience Demo")


# ──────────────────────────────────────────────────────────────
# 10A. Rate Limiting — in-memory (for demo; use Redis in production)
# ──────────────────────────────────────────────────────────────

class InMemoryRateLimiter:
    """
    Sliding window rate limiter (in-memory).
    Production version: replace self.store with Redis commands:
        MULTI
        ZADD key <now> <now>
        ZREMRANGEBYSCORE key 0 <window_start>
        ZCARD key
        EXPIRE key <window_seconds>
        EXEC
    """

    def __init__(self):
        # {client_id: [timestamp1, timestamp2, ...]}
        self.store: dict[str, list[float]] = {}

    def is_allowed(self, client_id: str, max_requests: int, window_seconds: int) -> tuple[bool, dict]:
        now = time.time()
        window_start = now - window_seconds

        # Evict requests outside the window (sliding window)
        requests_in_window = [
            t for t in self.store.get(client_id, [])
            if t > window_start
        ]
        requests_in_window.append(now)
        self.store[client_id] = requests_in_window

        count = len(requests_in_window)
        allowed = count <= max_requests

        return allowed, {
            "limit": max_requests,
            "remaining": max(0, max_requests - count),
            "window_seconds": window_seconds,
            "reset_at": int(window_start + window_seconds),
        }


rate_limiter = InMemoryRateLimiter()


# HOW THIS 3-LAYER DECORATOR WORKS
# ----------------------------------
# Usage:
#   @rate_limit(max_requests=5, window_seconds=60)
#   async def get_items(request): ...
#
# Python translates that @ syntax into:
#   get_items = rate_limit(max_requests=5, window_seconds=60)(get_items)
#               └─── Layer 1 ───────────────────────────────┘└─ Layer 2 ┘
#
# Layer 1 — rate_limit(...)  : YOU call this with config. Returns decorator.
# Layer 2 — decorator(func)  : Python calls this automatically, passing your
#                              route function as func. Returns wrapper.
# Layer 3 — wrapper(request) : FastAPI calls this on EVERY HTTP request,
#                              passing the live Request object. Runs the gate logic.
#
# After decoration, get_items IS wrapper.
# func (original get_items) is captured in wrapper's closure — lives there forever.
#
# Timeline:
#   App starts   → rate_limit(5) → decorator(get_items) → wrapper stored as get_items
#   Request hits → FastAPI calls wrapper(request) → checks limit → calls func(request)

def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    # LAYER 1 — Factory: holds the config (max_requests, window_seconds).
    # Returns decorator so Python can call it with the route function next.
    def decorator(func: Callable):
        # LAYER 2 — Receives the original route function (e.g. get_items) as func.
        # Python passes it automatically via the @ syntax — you never call this manually.
        # func is captured in a closure: wrapper can access it on every future request.

        @wraps(func)
        # @wraps copies the original function's name/docstring onto wrapper,
        # so FastAPI's OpenAPI docs show the correct route name — not "wrapper".
        async def wrapper(request: Request, *args, **kwargs):
            # LAYER 3 — Runs on every HTTP request.
            # FastAPI injects the live Request object here automatically.
            # *args/**kwargs carry any other route parameters (path, query, body).

            client_ip = request.client.host if request.client else "unknown"
            allowed, info = rate_limiter.is_allowed(client_ip, max_requests, window_seconds)

            if not allowed:
                # Client exceeded the limit — reject with 429 before touching the route
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Max {max_requests} requests per {window_seconds}s",
                        **info,
                    },
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "Retry-After": str(window_seconds),
                    },
                )

            # Attach rate limit info to request.state so the route can read it if needed
            request.state.rate_limit_info = info

            # Call the original route function (func captured from closure above)
            response = await func(request, *args, **kwargs)
            return response

        return wrapper
        # WHY return wrapper: replaces the original route function with wrapper.
        # From now on, every call to get_items actually calls wrapper first.

    return decorator
    # WHY return decorator: gives Python something to call with the route function.
    # Without this return, Python gets None and crashes with TypeError.


@app.get("/limited-inference")
@rate_limit(max_requests=5, window_seconds=60)  # 5 req/min per IP
async def limited_inference(request: Request, prompt: str = "hello"):
    """
    Rate-limited endpoint. Call it 6 times in 60s to see 429.
    Rate limit info is in response for debugging.
    """
    return {
        "response": f"Result: {prompt}",
        "rate_limit": request.state.rate_limit_info,
    }


# ──────────────────────────────────────────────────────────────
# 10B. Circuit Breaker Pattern
# ──────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"         # Normal operation — requests flow through
    OPEN = "open"             # Too many failures — requests fail immediately
    HALF_OPEN = "half_open"   # Testing if service recovered


class CircuitBreaker:
    """
    State machine:
      CLOSED → (failure_threshold exceeded) → OPEN
      OPEN   → (reset_timeout elapsed)       → HALF_OPEN
      HALF_OPEN → (success)                  → CLOSED
      HALF_OPEN → (failure)                  → OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 60.0,
        success_threshold: int = 2,  # successes needed in HALF_OPEN to close
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout_seconds
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None

    def _try_reset(self):
        """Check if enough time has passed to try half-open."""
        if (
            self.state == CircuitState.OPEN
            and self.last_failure_time
            and (time.time() - self.last_failure_time) >= self.reset_timeout
        ):
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            print(f"[CB] Circuit HALF_OPEN — testing recovery...")

    def call_allowed(self) -> bool:
        """Returns True if the circuit allows a request through."""
        self._try_reset()
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print(f"[CB] Circuit CLOSED — service recovered!")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"[CB] Circuit OPEN — {self.failure_count} failures. Blocking for {self.reset_timeout}s")

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "reset_timeout_seconds": self.reset_timeout,
        }


# One circuit breaker per upstream service
openai_circuit = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=30)


async def call_openai_with_circuit_breaker(prompt: str) -> str:
    """
    Wraps an AI API call with circuit breaker + timeout.
    Falls back gracefully when the circuit is open.
    """
    if not openai_circuit.call_allowed():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "AI service temporarily unavailable",
                "reason": "Circuit breaker is OPEN",
                "circuit": openai_circuit.get_status(),
                "message": "Retry after cooldown period",
            },
        )

    try:
        # asyncio.wait_for adds a timeout — prevents hanging forever
        result = await asyncio.wait_for(
            _call_fake_openai(prompt),
            timeout=10.0,  # 10 second timeout
        )
        openai_circuit.record_success()
        return result

    except asyncio.TimeoutError:
        openai_circuit.record_failure()
        raise HTTPException(
            status_code=504,
            detail={"error": "AI API timeout", "timeout_seconds": 10},
        )
    except Exception as e:
        openai_circuit.record_failure()
        raise HTTPException(
            status_code=502,
            detail={"error": "AI API error", "detail": str(e)},
        )


# Simulates flaky AI API (fails 60% of the time for demo)
_call_count = 0

async def _call_fake_openai(prompt: str) -> str:
    global _call_count
    _call_count += 1
    await asyncio.sleep(0.5)  # simulate latency

    if _call_count % 5 in (1, 2, 3):  # fail 3 out of 5 calls
        raise RuntimeError(f"OpenAI API error 500 on call #{_call_count}")

    return f"OpenAI response to: {prompt} (call #{_call_count})"


@app.get("/ai-infer")
async def ai_infer(prompt: str = "explain attention"):
    """Call AI API with circuit breaker protection."""
    result = await call_openai_with_circuit_breaker(prompt)
    return {"response": result, "circuit": openai_circuit.get_status()}


@app.get("/circuit-status")
def circuit_status():
    """Monitor circuit breaker state."""
    return {
        "openai_circuit": openai_circuit.get_status(),
        "total_calls": _call_count,
    }


# ──────────────────────────────────────────────────────────────
# 10C. Timeout configuration for high-latency AI APIs
# ──────────────────────────────────────────────────────────────

# Reusable httpx client with timeouts (create once, reuse)
# connect=5s: time to establish TCP connection
# read=30s:   time to receive the first byte of response
# write=10s:  time to send the request body
# pool=5s:    time to acquire a connection from pool
HTTP_CLIENT = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
)


@app.get("/call-external-api")
async def call_external_api(url: str = "https://httpbin.org/delay/2"):
    """Demonstrate explicit timeout handling for external AI API calls."""
    try:
        response = await HTTP_CLIENT.get(url)
        return {"status": response.status_code, "body_size": len(response.content)}

    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"Request timed out: {e}")

    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Connection failed: {e}")


# ──────────────────────────────────────────────────────────────
# 10D. Retry with exponential backoff (simple implementation)
# ──────────────────────────────────────────────────────────────

async def with_retry(coro_factory: Callable, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry an async operation with exponential backoff.
    Delay sequence: 1s → 2s → 4s (base * 2^attempt).
    In production: use 'tenacity' library for production-grade retries.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[RETRY] Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

    raise last_error


@app.get("/infer-with-retry")
async def infer_with_retry(prompt: str = "hello"):
    """Inference with retry + circuit breaker + timeout."""
    try:
        result = await with_retry(
            lambda: call_openai_with_circuit_breaker(prompt),
            max_retries=3,
            base_delay=1.0,
        )
        return {"response": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("10_scaling_resilience:app", host="0.0.0.0", port=8010, reload=True)
```
