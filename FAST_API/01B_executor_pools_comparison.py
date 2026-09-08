"""
========================================================
TOPIC 1B: ProcessPoolExecutor vs ThreadPoolExecutor
========================================================

KEY CONCEPTS:

1. ProcessPoolExecutor (CPU-bound work)
   ├─ Spawns SEPARATE Python processes (each has own GIL)
   ├─ True parallelism: run on multiple CPU cores simultaneously
   ├─ Memory overhead: ~100MB per process (each loads modules again)
   ├─ Use for: ML inference, crypto, data crunching, heavy computation
   ├─ Risk in Docker: memory explosion (each worker = full Python interpreter)
   └─ Example: heavy_ml_inference(data) with run_in_executor(ProcessPool, func)

2. ThreadPoolExecutor (I/O-bound work)
   ├─ Spawns THREADS within same process (shared memory)
   ├─ Blocked by GIL: only one thread executes Python bytecode at a time
   ├─ Memory overhead: minimal (~1-2MB per thread)
   ├─ Use for: lightweight sync code, legacy libraries, I/O waits
   ├─ Safe in Docker: threads share parent process memory
   └─ Example: slow_api_call() with run_in_executor(ThreadPool, func)

3. FastAPI Default Behavior:
   - plain `def` endpoints → auto-run in ThreadPoolExecutor
   - `async def` endpoints → run on event loop (no executor)
   - CPU-bound in async → must use run_in_executor(ProcessPool, ...)

4. GIL (Global Interpreter Lock):
   ├─ Python mutex that prevents true concurrent Python bytecode execution
   ├─ ProcessPoolExecutor BYPASSES GIL (separate processes = separate locks)
   ├─ ThreadPoolExecutor BLOCKED by GIL (threads share one lock)
   └─ I/O operations RELEASE GIL (so ThreadPool works fine for I/O)

5. Memory Profile (Docker Container, 512MB limit):
   ┌─────────────────────────────────────────────────────────┐
   │ Main Uvicorn Process: 150MB                             │
   │ ├─ FastAPI app, dependencies, models                    │
   │ └─ ONE event loop (async def endpoints run here)        │
   │                                                         │
   │ ThreadPoolExecutor (max_workers=8):                      │
   │ └─ 8 threads sharing parent memory: +0MB (shared)       │
   │   Total: 150MB ✅ Safe                                  │
   │                                                         │
   │ ProcessPoolExecutor (max_workers=4):                     │
   │ ├─ Worker 1: 150MB (duplicate of main + models)        │
   │ ├─ Worker 2: 150MB                                      │
   │ ├─ Worker 3: 150MB                                      │
   │ └─ Worker 4: 150MB                                      │
   │   Total: 150 + (150 × 4) = 750MB ❌ OOM!               │
   └─────────────────────────────────────────────────────────┘

INTERVIEW ANSWER TIP:
  "ProcessPoolExecutor is for CPU-bound work—it creates separate processes
   to bypass GIL. ThreadPoolExecutor is for I/O-bound—it's lighter but
   blocked by GIL for CPU work. In FastAPI + Docker, be careful:
   ProcessPool multiplies memory (each process loads modules), while
   ThreadPool shares memory. For ML in containers, prefer separate
   microservice over ProcessPool."
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import httpx
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Executor Pools Comparison")

# ──────────────────────────────────────────────────────────────
# PART A: ThreadPoolExecutor (I/O-bound, lightweight)
# ──────────────────────────────────────────────────────────────

# Thread pool: minimal overhead, good for I/O waits
thread_pool = ThreadPoolExecutor(max_workers=8)

def sync_api_call(url: str) -> dict:
    """Simulates a blocking I/O call (e.g., legacy sync library)."""
    # In real life: requests.get(url) or database query with sync driver
    time.sleep(1)  # Simulate network latency
    return {"status": 200, "data": f"Response from {url}"}

@app.get("/thread-pool/sync-io")
async def thread_pool_sync_io(url: str = "https://example.com"):
    """
    ThreadPoolExecutor for I/O-bound work (network, disk, database).

    Why ThreadPool here?
    - sync_api_call() is I/O-bound (network wait releases GIL)
    - Multiple threads can wait simultaneously without blocking
    - Memory: all threads share parent process memory (~minimal overhead)
    - Safe in Docker: 8 threads ≈ +5MB total
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(thread_pool, sync_api_call, url)
    return result

# ──────────────────────────────────────────────────────────────
# PART B: ProcessPoolExecutor (CPU-bound, high memory cost)
# ──────────────────────────────────────────────────────────────

# Process pool: spawn separate Python processes
process_pool = ProcessPoolExecutor(max_workers=2)

def cpu_heavy_work(n: int) -> dict:
    """
    CPU-bound work: sum of squares (no I/O, pure computation).
    GIL is a problem: ThreadPool can't parallelism this.
    Solution: ProcessPool (separate processes = separate GILs).
    """
    result = sum(i * i for i in range(n))
    return {"result": result, "n": n}

@app.get("/process-pool/cpu-heavy")
async def process_pool_cpu_heavy(n: int = 10_000_000):
    """
    ProcessPoolExecutor for CPU-bound work (ML, crypto, computation).

    Why ProcessPool here?
    - cpu_heavy_work() is pure CPU, no I/O (GIL locks thread)
    - ThreadPool would block (GIL prevents parallelism)
    - ProcessPool spawns separate processes (separate GILs)
    - Cost: each process loads Python + modules again (~150MB per worker)
    - Risk in Docker: 2 workers × 150MB = 300MB extra memory

    Docker consideration:
    If container limit is 512MB and main app is 200MB:
    ├─ Main: 200MB
    ├─ Worker 1: 200MB
    └─ Worker 2: 200MB
    Total: 600MB > 512MB → OOM KILL ❌

    Fix: reduce max_workers based on container memory
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(process_pool, cpu_heavy_work, n)
    return result

# ──────────────────────────────────────────────────────────────
# PART C: Demonstrating the GIL Problem
# ──────────────────────────────────────────────────────────────

def compute_cpu_bound(iterations: int) -> float:
    """Pure CPU work—no I/O."""
    total = 0.0
    for i in range(iterations):
        total += i * i
    return total

def compute_io_bound(duration: float) -> dict:
    """I/O bound—sleeps without CPU work."""
    time.sleep(duration)
    return {"waited": duration}

@app.get("/comparison/thread-cpu")
async def comparison_thread_cpu():
    """
    ❌ WRONG: ThreadPool for CPU work is inefficient.
    GIL prevents parallelism—threads run sequentially (not parallel).
    """
    loop = asyncio.get_event_loop()
    start = time.time()
    results = [
        await loop.run_in_executor(thread_pool, compute_cpu_bound, 10_000_000),
        await loop.run_in_executor(thread_pool, compute_cpu_bound, 10_000_000),
    ]
    elapsed = time.time() - start
    return {"approach": "ThreadPool CPU", "elapsed_sec": elapsed, "results": results}

@app.get("/comparison/process-cpu")
async def comparison_process_cpu():
    """
    ✅ CORRECT: ProcessPool for CPU work.
    True parallelism—processes run on separate cores simultaneously.
    """
    loop = asyncio.get_event_loop()
    start = time.time()
    results = [
        await loop.run_in_executor(process_pool, compute_cpu_bound, 10_000_000),
        await loop.run_in_executor(process_pool, compute_cpu_bound, 10_000_000),
    ]
    elapsed = time.time() - start
    return {"approach": "ProcessPool CPU", "elapsed_sec": elapsed, "results": results}

@app.get("/comparison/thread-io")
async def comparison_thread_io():
    """
    ✅ CORRECT: ThreadPool for I/O work.
    Threads wait on I/O (GIL released), can do other work.
    """
    loop = asyncio.get_event_loop()
    start = time.time()
    results = [
        await loop.run_in_executor(thread_pool, compute_io_bound, 1),
        await loop.run_in_executor(thread_pool, compute_io_bound, 1),
    ]
    elapsed = time.time() - start
    return {"approach": "ThreadPool I/O", "elapsed_sec": elapsed, "results": results}

# ──────────────────────────────────────────────────────────────
# PART D: Real-world scenario—ML inference with memory concerns
# ──────────────────────────────────────────────────────────────

def load_and_predict(model_path: str, text: str) -> dict:
    """
    Simulates loading a large model and running inference.
    In production: from transformers import pipeline
    model = pipeline("sentiment-analysis")
    result = model(text)
    """
    # Simulate loading a 500MB model
    dummy_model_data = bytearray(50_000_000)  # 50MB dummy data
    prediction = "positive" if len(text) > 5 else "negative"
    return {"text": text, "prediction": prediction}

process_pool_ml = ProcessPoolExecutor(max_workers=2)

@app.get("/ml-inference-process-pool")
async def ml_inference_process_pool(text: str = "hello world"):
    """
    ⚠️ WARNING: ProcessPool for ML in Docker is risky.

    Each worker loads the model independently:
    ├─ Main process (with model): 200MB
    ├─ Worker 1 (loads model again): 200MB
    └─ Worker 2 (loads model again): 200MB
    Total: 600MB

    If container limit is 512MB → OOM KILL

    Better approach: Use separate ML service (see below)
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(process_pool_ml, load_and_predict, "model.bin", text)
    return result

@app.get("/ml-inference-external-service")
async def ml_inference_external_service(text: str = "hello world"):
    """
    ✅ BETTER: Call external ML service (separate container/server).

    Architecture:
    ┌──────────────────┐         ┌──────────────────┐
    │  FastAPI         │         │  ML Service      │
    │  (light, 256MB)  │────────▶│  (heavy, 2GB)    │
    │  Port 8000       │         │  Port 8001       │
    └──────────────────┘         └──────────────────┘

    Benefits:
    - Main app stays light (no model memory)
    - Scale ML service independently (Kubernetes HPA)
    - Isolate failures (if ML crashes, main still runs)
    - Can use GPU on ML service, CPU-only for main
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://ml-service:8001/predict",
            json={"text": text},
            timeout=30,
        )
    return response.json()

# ──────────────────────────────────────────────────────────────
# PART E: Memory monitoring (for Docker health checks)
# ──────────────────────────────────────────────────────────────

try:
    import psutil

    @app.get("/health/memory")
    async def health_memory():
        """
        Monitor memory usage in Docker.
        Use in healthcheck:
          healthcheck:
            test: ["CMD", "curl", "-f", "http://localhost:8000/health/memory"]
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()

        return {
            "status": "ok",
            "process_rss_mb": memory_info.rss // (1024 * 1024),  # Resident set size
            "process_vms_mb": memory_info.vms // (1024 * 1024),  # Virtual memory
            "system_memory_percent": virtual_memory.percent,
            "system_available_mb": virtual_memory.available // (1024 * 1024),
            "warning": "high memory" if virtual_memory.percent > 80 else None,
        }

except ImportError:
    pass

# ──────────────────────────────────────────────────────────────
# SUMMARY TABLE: When to Use Each
# ──────────────────────────────────────────────────────────────

"""
┌──────────────────────────────────────────────────────────────────┐
│          ProcessPoolExecutor vs ThreadPoolExecutor              │
├──────────────┬──────────────────────┬──────────────────────────┤
│ Aspect       │ ProcessPoolExecutor   │ ThreadPoolExecutor      │
├──────────────┼──────────────────────┼──────────────────────────┤
│ Parallelism  │ TRUE (multi-core)     │ PSEUDO (GIL limited)    │
│ GIL Bypass   │ Yes (separate process)│ No (shared GIL)        │
│ Memory Cost  │ HIGH (+150MB/worker)  │ LOW (+1-2MB/worker)    │
│ Best For     │ CPU-bound             │ I/O-bound              │
│              │ ML inference          │ Network calls          │
│              │ Crypto                │ Database queries       │
│              │ Data processing       │ Sync legacy libraries  │
├──────────────┼──────────────────────┼──────────────────────────┤
│ Example Use  │ run_in_executor(      │ run_in_executor(       │
│              │   process_pool,       │   thread_pool,         │
│              │   ml_model.predict    │   sync_api.get_data    │
│              │ )                     │ )                      │
├──────────────┼──────────────────────┼──────────────────────────┤
│ Docker Safe? │ ⚠️ Risky              │ ✅ Safe                │
│              │ (multiplies memory)   │ (shared memory)        │
├──────────────┼──────────────────────┼──────────────────────────┤
│ Startup Time │ Slow (~1-2s)          │ Instant                │
│ Context      │ Heavy (load modules)  │ Light (shared)         │
│ Overhead     │                       │                        │
├──────────────┼──────────────────────┼──────────────────────────┤
│ Debugging    │ Hard (separate proc)  │ Easy (same process)    │
│              │ (use logging)         │ (use debugger)         │
├──────────────┼──────────────────────┼──────────────────────────┤
│ Real-world   │ Transformers model    │ requests library       │
│ Examples     │ PyTorch inference     │ httpx sync calls       │
│              │ numpy/scipy compute   │ psycopg (sync DB)      │
└──────────────┴──────────────────────┴──────────────────────────┘

Rule of Thumb:
├─ Does your function make I/O calls (network, disk)?
│  └─ YES → Use ThreadPoolExecutor (or better: keep it async)
│  └─ NO → Check: is it CPU-intensive?
│
├─ Is it pure CPU (math, ML, crypto)?
│  └─ YES → Use ProcessPoolExecutor (if not in container)
│  └─ NO → Use ThreadPoolExecutor
│
└─ Running in Docker with memory constraints?
   └─ Prefer: separate microservice over ProcessPoolExecutor
   └─ Reason: avoid multiplying memory usage
"""

# ──────────────────────────────────────────────────────────────
# PYTHONUNBUFFERED=1 EXPLANATION
# ──────────────────────────────────────────────────────────────

"""
What is ENV PYTHONUNBUFFERED=1?

SHORT ANSWER:
  Tells Python to flush stdout immediately instead of buffering.

THE PROBLEM (without it):
  ├─ Python buffers output (waits for 4KB-8KB to fill before flushing)
  ├─ In Docker, stdout is a pipe (not terminal) → full buffering
  ├─ Logs delayed or lost if container crashes
  └─ Result: "mysterious" gaps in docker logs on failures

THE SOLUTION (with PYTHONUNBUFFERED=1):
  ├─ Flush output immediately after each print()
  ├─ Docker sees logs in real-time
  ├─ No data loss on crash
  └─ Critical for debugging production issues

REAL EXAMPLE:
  Without: Container crashes at 3:45 PM → docker logs shows last entry at 3:00 PM ❌
  With:    Container crashes at 3:45 PM → docker logs shows error at 3:45 PM ✓

PERFORMANCE: Negligible impact (logging is I/O-bound anyway)

ALWAYS set in Dockerfile for FastAPI + Docker.
"""

# ──────────────────────────────────────────────────────────────
# DOCKER EXAMPLES
# ──────────────────────────────────────────────────────────────

"""
Dockerfile Example 1: ThreadPool only (safe)
────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "01B_executor_pools_comparison:app", "--host", "0.0.0.0", "--port", "8000"]

docker run -m 256m myapp:latest
↳ ThreadPool is safe (light), runs fine in 256MB

Dockerfile Example 2: ProcessPool with memory awareness
──────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn psutil
ENV PYTHONUNBUFFERED=1
ENV MEMORY_LIMIT_MB=512
ENV MAX_PROCESS_WORKERS=2  # ← Tune based on container memory
CMD ["uvicorn", "01B_executor_pools_comparison:app", "--host", "0.0.0.0", "--port", "8000"]

# In code:
import os
max_workers = int(os.getenv("MAX_PROCESS_WORKERS", "2"))
process_pool = ProcessPoolExecutor(max_workers=max_workers)

docker run -m 512m -e MAX_PROCESS_WORKERS=2 myapp:latest
↳ Still risky—monitor memory

Dockerfile Example 3: Separate ML service (best for production)
──────────────────────────────────────────────────────────────
# docker-compose.yml
version: '3.8'
services:
  app:
    image: fastapi-app:latest
    ports:
      - "8000:8000"
    mem_limit: 256m  # Light, no models
    environment:
      ML_SERVICE_URL: http://ml-service:8001
    depends_on:
      - ml-service

  ml-service:
    image: ml-inference-server:latest
    expose:
      - "8001"
    mem_limit: 2g  # ML needs more memory
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

↳ Main app is lightweight, ML is isolated
↳ Can use GPU on ML service, CPU-only for main
↳ Scale independently
"""

if __name__ == "__main__":
    print(__doc__)
    uvicorn.run(
        "01B_executor_pools_comparison:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )

