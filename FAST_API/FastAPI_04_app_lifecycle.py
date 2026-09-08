"""
========================================================
TOPIC 4: App Lifecycle — ML Model Loading
========================================================

KEY CONCEPTS:
- @asynccontextmanager lifespan → NEW way (FastAPI 0.93+).
  Code before yield = startup. Code after yield = shutdown.
  Replaces deprecated @app.on_event("startup").

- Singleton pattern for GPU models:
  Load model ONCE at startup into app.state or a module-level dict.
  Do NOT load the model inside the endpoint (would reload per request!).

- Shared state across workers:
  Each Uvicorn worker is a separate OS process — they do NOT share memory.
  For true sharing: use Redis, a model server (Triton, vLLM), or keep
  model in a single worker with a queue.

INTERVIEW ANSWER TIP:
  "I use lifespan context managers to load ML models into app.state
   at startup. This guarantees the model is warm before the first request
   and is properly unloaded on shutdown to free GPU memory. For multi-
   worker deployments, I use vLLM or Triton as a sidecar model server
   so all workers share one GPU-loaded model."
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request

# ──────────────────────────────────────────────────────────────
# 4A. Simulate a heavy ML model class
# ──────────────────────────────────────────────────────────────
class FakeTransformerModel:
    """Simulates a GPU-resident transformer (e.g., HuggingFace model)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False

    def load(self):
        """In production: model = AutoModelForCausalLM.from_pretrained(...)"""
        print(f"[STARTUP] Loading model '{self.model_name}' onto GPU...")
        # time.sleep(5)  # simulate slow model load
        self.is_loaded = True
        print(f"[STARTUP] Model '{self.model_name}' ready.")

    def predict(self, text: str) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded!")
        return f"[{self.model_name}] Response to: {text}"

    def unload(self):
        """Free GPU memory. In production: del model; torch.cuda.empty_cache()"""
        print(f"[SHUTDOWN] Unloading model '{self.model_name}' from GPU...")
        self.is_loaded = False

# ──────────────────────────────────────────────────────────────
# 4B. Lifespan context manager — the correct pattern
# ──────────────────────────────────────────────────────────────

# Module-level store for models (singleton across requests in ONE process)
# IMPORTANT: _model_store is NOT shared across Uvicorn workers!
#   - Within Worker 1: all requests share the same _model_store ✓
#   - Worker 1 vs Worker 2: each has its own _model_store copy ❌
#   - ProcessPoolExecutor workers: can't access _model_store at all ❌
# See section 4D for details.
_model_store: dict[str, FakeTransformerModel] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything BEFORE yield runs at startup (before first request).
    Everything AFTER yield runs at shutdown (after last request completes).

    FastAPI passes this to the app constructor:
        app = FastAPI(lifespan=lifespan)

    MODULE-LEVEL vs app.state:
      _model_store (module-level):
        ├─ Within same worker: ✓ shared across all requests
        ├─ Across workers: ❌ each worker loads own copy
        ├─ From ProcessPool: ❌ separate processes, can't access
        └─ Lifespan runs once per worker (on startup)

      app.state (request.app.state):
        ├─ Within same worker: ✓ shared across all requests
        ├─ Across workers: ❌ each worker has own instance
        ├─ From ProcessPool: ❌ separate processes, can't access
        └─ Lifespan runs once per worker (on startup)

      Both have IDENTICAL sharing behavior (per-process, not across workers).
      Use whichever makes sense semantically. Module-level is cleaner for singletons.
    """
    # ── STARTUP ──────────────────────────────────────────────
    print("[LIFESPAN] Startup: initializing resources...")

    # Load primary LLM
    llm = FakeTransformerModel("llama-3-8b")
    llm.load()
    _model_store["llm"] = llm

    # Load embedding model
    embedder = FakeTransformerModel("bge-m3-embedding")
    embedder.load()
    _model_store["embedder"] = embedder

    # You can also store in app.state for access via request.app.state
    app.state.db_pool = "fake_connection_pool"  # e.g., asyncpg pool

    print("[LIFESPAN] All resources ready. Accepting requests.")

    yield  # ← server runs here, serving requests

    # ── SHUTDOWN ─────────────────────────────────────────────
    print("[LIFESPAN] Shutdown: releasing resources...")

    for name, model in _model_store.items():
        model.unload()

    _model_store.clear()

    # await app.state.db_pool.close()  ← real pool teardown
    print("[LIFESPAN] Cleanup complete.")

# Pass lifespan to the app
app = FastAPI(title="ML Lifecycle Demo", lifespan=lifespan)

# ──────────────────────────────────────────────────────────────
# 4C. Endpoints that use the pre-loaded models
# ──────────────────────────────────────────────────────────────

@app.post("/generate")
async def generate_text(prompt: str, request: Request):
    """
    Model is already in _model_store — no loading latency per request.
    request.app.state holds app-level state (like DB pool).
    """
    llm = _model_store.get("llm")
    if not llm:
        return {"error": "Model not loaded"}, 503

    # CPU-bound inference should use run_in_executor (see topic 1)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, llm.predict, prompt)

    return {
        "response": result,
        "db_pool": request.app.state.db_pool,
    }

@app.post("/embed")
async def embed_text(text: str):
    """Use the embedding model loaded at startup."""
    embedder = _model_store.get("embedder")
    if not embedder:
        return {"error": "Embedder not loaded"}, 503

    # Simulate returning a vector
    result = embedder.predict(text)
    return {"embedding_preview": result[:50], "dim": 768}

@app.get("/model-status")
def model_status():
    """Health check: report which models are loaded."""
    return {
        name: {"loaded": model.is_loaded, "name": model.model_name}
        for name, model in _model_store.items()
    }

# ──────────────────────────────────────────────────────────────
# 4D. app.state sharing across workers & ProcessPoolExecutor
# ──────────────────────────────────────────────────────────────
"""
CRITICAL: app.state is NOT shared across workers!

SCENARIO 1: uvicorn --workers 3
  ├─ 3 separate OS processes (fork model)
  ├─ Each process has its own FastAPI app instance
  ├─ Each process has its own app.state (independent memory)
  ├─ Result: app.state is NOT shared across workers ❌
  │
  └─ Example:
      Worker 1: app.state.counter = 1
      Worker 2: app.state.counter = 1 (different object!)
      Request hits Worker 1 → increments to 2
      Request hits Worker 2 → still 1 (never saw update)

SCENARIO 2: uvicorn --workers 3 + ProcessPoolExecutor(max_workers=2)
  ├─ 3 Uvicorn workers (separate processes)
  ├─ ProcessPoolExecutor workers (2 separate processes, independent from Uvicorn)
  ├─ ProcessPool workers CANNOT access app.state at all
  │  (they're different processes, different Python interpreters)
  ├─ Result: ProcessPool has zero visibility to app.state ❌
  │
  └─ Why: ProcessPool workers only receive function arguments, not app object
      loop.run_in_executor(process_pool, my_func, arg1, arg2)
      # my_func receives arg1, arg2 ONLY (app.state not passed)

MEMORY LAYOUT:
  ┌─ Uvicorn Master (Gunicorn or Uvicorn manager)
  │
  ├─ Worker 1 (PID 1001)
  │  └─ app.state = {"counter": 1, "db": ...}  ← Process 1 memory
  │
  ├─ Worker 2 (PID 1002)
  │  └─ app.state = {"counter": 1, "db": ...}  ← Process 2 memory (SEPARATE!)
  │
  ├─ Worker 3 (PID 1003)
  │  └─ app.state = {"counter": 1, "db": ...}  ← Process 3 memory (SEPARATE!)
  │
  └─ ProcessPool Worker 1 (PID 2001)
     └─ (no app object at all)

SOLUTIONS FOR SHARED STATE:

1. Redis (best for cross-worker state):
   ├─ app.state.redis = redis.Redis()
   ├─ All workers connect to same Redis instance
   ├─ True shared state (counter, cache, sessions)
   └─ Example: request.app.state.redis.incr("counter")

2. Database (for persistence):
   ├─ app.state.db_pool = asyncpg.create_pool()
   ├─ All workers share connection pool
   ├─ Queries see same data
   └─ Example: await request.app.state.db_pool.fetchval("SELECT counter FROM state")

3. Single worker (--workers 1):
   ├─ No multiprocessing
   ├─ Only one app.state instance (auto-shared across requests)
   ├─ Works for async-only workloads
   └─ Command: uvicorn app:app --workers 1

4. Shared memory block (PyTorch models only):
   ├─ torch.multiprocessing.set_start_method('spawn')
   ├─ model.share_memory()
   ├─ Workers access same model weights (no duplication)
   └─ Advanced, complex, PyTorch-specific

5. Message queue for ProcessPool communication:
   ├─ ProcessPool worker: result = heavy_compute()
   ├─ Send result to main worker via queue.Queue()
   ├─ Main worker stores in app.state + Redis
   └─ Other workers read from Redis

_model_store vs app.state — Which to Use?

COMPARISON TABLE:
  ┌─────────────────────┬──────────────────┬─────────────────────┐
  │ Aspect              │ _model_store     │ app.state           │
  │                     │ (module-level)   │ (request.app.state) │
  ├─────────────────────┼──────────────────┼─────────────────────┤
  │ Shared within       │ ✓ YES (requests) │ ✓ YES (requests)    │
  │ same worker?        │                  │                     │
  ├─────────────────────┼──────────────────┼─────────────────────┤
  │ Shared across       │ ❌ NO (fork)     │ ❌ NO (fork)        │
  │ workers (--workers) │                  │                     │
  ├─────────────────────┼──────────────────┼─────────────────────┤
  │ Access from         │ ❌ NO (separate) │ ❌ NO (separate)    │
  │ ProcessPool?        │ process          │ process             │
  ├─────────────────────┼──────────────────┼─────────────────────┤
  │ Persistence on      │ ❌ Lost (mem)    │ ❌ Lost (mem)       │
  │ app restart?        │                  │                     │
  ├─────────────────────┼──────────────────┼─────────────────────┤
  │ Lifespan runs?      │ ✓ Once/worker    │ ✓ Once/worker       │
  ├─────────────────────┼──────────────────┼─────────────────────┤
  │ Use case            │ Models, caches   │ DB pools, clients   │
  │                     │ (singletons)     │ (request-scoped)    │
  ├─────────────────────┼──────────────────┼─────────────────────┤
  │ Test friendly?      │ ⚠️ Global (hard  │ ✓ Via request obj   │
  │                     │ to mock)         │ (easy to inject)    │
  └─────────────────────┴──────────────────┴─────────────────────┘

WHEN TO USE EACH:

Use _model_store (module-level) for:
  ├─ ML models (load once per worker)
  ├─ Expensive singletons (thread pools, executors)
  ├─ Process-local caches
  └─ Code that doesn't need mocking in tests

Use app.state (request.app.state) for:
  ├─ DB connection pools (shared across requests)
  ├─ Clients (HTTP, Redis, S3)
  ├─ Config objects
  └─ Things you need to inject/mock in tests

EXAMPLE: Both together

  _model_store["llm"] = llm  # Load model once (expensive)
  app.state.db_pool = db_pool  # Share DB pool across requests

  @app.post("/generate")
  async def generate(text: str, request: Request):
      llm = _model_store["llm"]  # Direct access (singleton)
      db = request.app.state.db_pool  # Via request (testable)
      result = llm.predict(text)
      await db.execute("INSERT INTO results VALUES (...)")
      return {"result": result}

MEMORY LAYOUT (uvicorn --workers 2):

  ┌─ Worker 1 (PID 1001)
  │  ├─ Module: _model_store = {"llm": <Model1>}
  │  ├─ app.state.db_pool = <Pool1>
  │  └─ Request 1,2,3... all see same _model_store + app.state
  │
  └─ Worker 2 (PID 1002)
     ├─ Module: _model_store = {"llm": <Model2>}  ← DIFFERENT object!
     ├─ app.state.db_pool = <Pool2>  ← DIFFERENT object!
     └─ Request 4,5,6... see different _model_store + app.state

CRITICAL INSIGHT:
  Both _model_store and app.state are per-process, not shared across workers.
  The ONLY difference: _model_store is module-level (direct access),
  app.state is attached to app (accessed via request.app.state).

REAL-WORLD EXAMPLE:

Without shared state (BAD):
  uvicorn --workers 3
  ├─ Worker 1 loads model (300MB on GPU)
  ├─ Worker 2 loads model (300MB on GPU)
  ├─ Worker 3 loads model (300MB on GPU)
  └─ Total: 900MB (wasteful!)

With shared state (GOOD):
  uvicorn --workers 3 + vLLM server
  ├─ vLLM loads model once (300MB on GPU)
  ├─ All 3 workers call vLLM server HTTP endpoint
  └─ Total: 300MB (efficient!)

CODE EXAMPLE: Redis for shared state

  from fastapi import FastAPI
  import aioredis

  @asynccontextmanager
  async def lifespan(app):
      # STARTUP
      app.state.redis = await aioredis.from_url("redis://localhost")
      yield
      # SHUTDOWN
      app.state.redis.close()

  app = FastAPI(lifespan=lifespan)

  @app.get("/counter")
  async def get_counter(request: Request):
      # All 3 workers access SAME Redis value
      count = await request.app.state.redis.incr("counter")
      return {"count": count}

  Result: counter increments globally (1, 2, 3...) across all workers

INTERVIEW ANSWER:
  "app.state is per-process, NOT shared across workers. With uvicorn --workers 3,
   each worker is a separate process with its own app.state. ProcessPoolExecutor
   workers can't see app.state at all. For true sharing, use Redis (counters,
   cache), database (persistence), or a model server (GPU models). If you need
   isolated state per worker, app.state is fine. Otherwise, delegate to external
   systems."
"""

if __name__ == "__main__":
    uvicorn.run("04_app_lifecycle:app", host="0.0.0.0", port=8004, reload=False)
    # Note: reload=False when using lifespan — reload conflicts with process state
