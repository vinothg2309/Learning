# 01_concurrency_model.py

```python
"""
========================================================
TOPIC 1: FastAPI Concurrency Model
========================================================

KEY CONCEPTS:
- async def  → Non-blocking. FastAPI runs it on the event loop.
                Use for: DB calls, HTTP requests, file I/O.
- def         → Blocking. FastAPI auto-runs it in a threadpool
                so the event loop is NOT blocked.
                Use for: legacy sync code, simple endpoints.
- run_in_executor → Offload CPU-heavy ML work to a ProcessPoolExecutor
                     so the event loop stays free.
- ASGI vs WSGI:
    WSGI = Web Server Gateway Interface (sync, Python standard since PEP 3333)
    ASGI = Asynchronous Server Gateway Interface (async superset of WSGI)

    WSGI (Flask/Django): one request at a time per worker (sync).
    ASGI (FastAPI/Starlette): handles thousands of concurrent connections
    via async I/O — critical for LLM streaming, WebSockets.

- Uvicorn:
    Uvicorn = ASGI server (the actual process that listens on a port).
    It implements the ASGI spec and bridges between OS sockets and your
    FastAPI app. Think of it as the engine room.
    - Single-process, high-performance, built on uvloop + httptools.
    - In production: Gunicorn manages multiple Uvicorn worker processes.
      Gunicorn = process manager (restarts crashed workers, handles signals).
      Uvicorn  = ASGI worker (handles async requests within each process).
    Command: gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker

INTERVIEW ANSWER TIP:
  "FastAPI is ASGI-based. async def endpoints yield control to the event
   loop during I/O waits, allowing other requests to be served. CPU-bound
   ML inference should be offloaded with run_in_executor(ProcessPool)
   to avoid blocking the single-threaded event loop."
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import httpx  # pip install httpx
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Concurrency Demo")

# ──────────────────────────────────────────────────────────────
# 1A. async def — I/O-bound endpoint (correct pattern)
# ──────────────────────────────────────────────────────────────
@app.get("/async-io")
async def async_io_endpoint():
    """
    await suspends THIS coroutine and lets the event loop serve
    other requests while we wait for the HTTP response.
    Never blocks the server.

    When the response IS ready:
      1. OS (epoll/kqueue) detects data on the socket → signals event loop.
      2. Event loop moves this coroutine from SUSPENDED → READY.
      3. Execution resumes exactly at the next line after await.
      4. `response` is fully populated — used normally from here.
    From the code's perspective it looks synchronous; underneath it's non-blocking.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get("https://httpbin.org/delay/1")
    return {"status": response.status_code}


# ──────────────────────────────────────────────────────────────
# 1B. def — sync endpoint (FastAPI handles it safely)
# ──────────────────────────────────────────────────────────────
@app.get("/sync-io")
def sync_io_endpoint():
    """
    FastAPI detects this is a plain `def` and runs it in a
    ThreadPoolExecutor automatically. Safe, but less efficient
    than async for I/O-bound work.
    """
    time.sleep(1)  # simulate blocking I/O
    return {"message": "sync endpoint ran in threadpool"}


# ──────────────────────────────────────────────────────────────
# 1C. CPU-bound ML inference — use ProcessPoolExecutor
# ──────────────────────────────────────────────────────────────

# This runs in a SEPARATE PROCESS, not the event loop process.
# Avoids Python GIL for true parallelism.
def heavy_ml_inference(input_text: str) -> dict:
    """Simulates CPU-heavy work (e.g., running a local transformer)."""
    # In real life: model(tokenizer(input_text)) etc.
    result = sum(i * i for i in range(10_000_000))  # CPU burn
    return {"result": result, "input": input_text}


# Reuse executor — don't create per request (expensive!)
_process_pool = ProcessPoolExecutor(max_workers=2)


@app.get("/cpu-ml-inference")
async def cpu_ml_inference(text: str = "hello"):
    """
    run_in_executor submits the CPU work to a process pool.
    The event loop awaits completion without blocking.
    Other requests are served while inference runs.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_process_pool, heavy_ml_inference, text)
    return result


# ──────────────────────────────────────────────────────────────
# 1D. WRONG PATTERN — never do this in async def!
# ──────────────────────────────────────────────────────────────
@app.get("/wrong-blocking")
async def wrong_blocking():
    """
    ❌ time.sleep() in async def blocks the ENTIRE event loop.
    No other request can be served during this sleep.
    This is the #1 FastAPI performance mistake.
    Fix: use 'await asyncio.sleep(1)' for delays,
         or run_in_executor for CPU work.
    """
    time.sleep(2)  # BLOCKS the event loop — never do this
    return {"oops": "blocked event loop for 2 seconds"}


# ──────────────────────────────────────────────────────────────
# 1E. Correct pattern — async sleep
# ──────────────────────────────────────────────────────────────
@app.get("/correct-async-sleep")
async def correct_async_sleep():
    """✅ asyncio.sleep yields control back to the event loop."""
    await asyncio.sleep(2)
    return {"message": "waited without blocking"}


if __name__ == "__main__":
    uvicorn.run("01_concurrency_model:app", host="0.0.0.0", port=8001, reload=True)
```
