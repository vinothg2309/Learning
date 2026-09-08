
"""
========================================================
TOPIC 1C: GIL & ThreadPoolExecutor Behavior Deep Dive
========================================================

KEY CONCEPT: What happens when 10 threads run in ThreadPoolExecutor?

SHORT ANSWER:
  ❌ NO, other threads don't ALL wait.

  It depends on what the thread is doing:
  ├─ Thread doing I/O (network, disk)? → RELEASES GIL → others can execute
  └─ Thread doing CPU? → HOLDS GIL → others MUST wait

THE GIL (Global Interpreter Lock):
  ├─ ONE lock per Python process
  ├─ Only ONE thread can execute Python bytecode at a time
  ├─ Released during I/O operations (network, disk, database)
  ├─ Held during CPU work (loops, math, processing)
  └─ Result: threads can't parallelize CPU work, but CAN parallelize I/O

VISUALIZATION:

1️⃣  ThreadPool doing I/O-bound work (10 threads, all making HTTP calls):
    ┌─────────────────────────────────────────────────────────────────┐
    │ GIL RELEASED during I/O waits → multiple threads execute        │
    │                                                                 │
    │ Time: 0s    1s    2s    3s    4s    5s                         │
    │ ──────────────────────────────────────────────────────         │
    │ Thread 1: [waiting on HTTP.....................] ✓ Released GIL│
    │ Thread 2:      [waiting on HTTP.....................] ✓       │
    │ Thread 3:           [waiting on HTTP.....................] ✓   │
    │ Thread 4:                [waiting on HTTP.....................] │
    │ ...                                                             │
    │ Thread 10:                         [waiting on HTTP.........]  │
    │                                                                 │
    │ Result: All 10 threads waiting simultaneously (pseudo-parallel) │
    │ Time to complete: ~5s (not 50s!)                               │
    └─────────────────────────────────────────────────────────────────┘

2️⃣  ThreadPool doing CPU-bound work (10 threads, all computing):
    ┌─────────────────────────────────────────────────────────────────┐
    │ GIL HELD during CPU work → only 1 thread executes at a time    │
    │                                                                 │
    │ Time: 0s  1s   2s   3s   4s   5s   6s   7s   8s   9s  10s     │
    │ ──────────────────────────────────────────────────────────────  │
    │ Thread 1: [CPU work] ✓ (holds GIL)                             │
    │ Thread 2:           [CPU work] ✓ (waits for GIL)               │
    │ Thread 3:                      [CPU work] ✓ (waits)             │
    │ Thread 4:                                  [CPU work] ✓ (waits) │
    │ ...                                                             │
    │ Thread 10:                                         [CPU work]   │
    │                                                                 │
    │ Result: Threads execute SEQUENTIALLY, not parallel              │
    │ Time to complete: ~10s (not 1s with true parallelism!)         │
    │ Speedup from threading: NONE (worse than single-threaded!)     │
    └─────────────────────────────────────────────────────────────────┘

3️⃣  ThreadPool doing MIXED (I/O + CPU):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Thread 1: [GIL HELD during CPU...] then [GIL RELEASED on I/O...]│
    │ Thread 2:                           [can execute during I/O...]  │
    │                                                                 │
    │ Both can make progress (thread switching during I/O waits)      │
    └─────────────────────────────────────────────────────────────────┘

PROOF WITH CODE:
  See examples below
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Thread
import threading

import httpx
from fastapi import FastAPI

app = FastAPI(title="GIL & ThreadPool Behavior")

thread_pool = ThreadPoolExecutor(max_workers=10)
process_pool = ProcessPoolExecutor(max_workers=4)

# ──────────────────────────────────────────────────────────────
# SCENARIO 1: I/O-bound with ThreadPool
# QUESTION: Do all 10 threads wait, or can they execute in parallel?
# ANSWER: They DON'T all wait. GIL is released during I/O.
# ──────────────────────────────────────────────────────────────

def io_bound_work(thread_id: int, delay: float = 1):
    """
    Simulate I/O (network call) that takes 1 second.

    When thread enters httpx.get():
      1. GIL is RELEASED
      2. OS handles network I/O (thread sleeps)
      3. Other threads can now execute Python code
      4. When response arrives, thread wakes up, reacquires GIL
    """
    print(f"[Thread {thread_id}] Starting I/O")
    # Simulate I/O delay
    time.sleep(delay)
    print(f"[Thread {thread_id}] I/O Complete")
    return f"Thread {thread_id} done"

@app.get("/io-bound-demo")
async def io_bound_demo():
    """
    10 threads, each makes 1-second I/O call.

    If GIL blocked ALL threads: would take 10 seconds
    If GIL releases during I/O: takes ~1 second (all parallel)

    Expected: ~1 second ✅
    """
    loop = asyncio.get_event_loop()
    start = time.time()

    # Submit 10 tasks
    tasks = [
        loop.run_in_executor(thread_pool, io_bound_work, i, 1.0)
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    return {
        "approach": "ThreadPool I/O-bound",
        "threads": 10,
        "elapsed_sec": round(elapsed, 2),
        "theory": "~1s (GIL released during I/O)",
        "why": "All threads wait on I/O simultaneously, no blocking",
    }

# ──────────────────────────────────────────────────────────────
# SCENARIO 2: CPU-bound with ThreadPool
# QUESTION: Do other 9 threads wait while 1 thread computes?
# ANSWER: YES. GIL blocks other threads during CPU work.
# ──────────────────────────────────────────────────────────────

def cpu_bound_work(thread_id: int, iterations: int = 10_000_000):
    """
    Pure CPU work (no I/O).

    When thread computes (e.g., loop):
      1. GIL is HELD
      2. Other threads CANNOT execute Python code
      3. Threads must wait for their turn
      4. Only one thread executes at a time (sequential!)
    """
    print(f"[Thread {thread_id}] CPU work starting")
    total = sum(i * i for i in range(iterations))
    print(f"[Thread {thread_id}] CPU work done: {total}")
    return total

@app.get("/cpu-bound-threadpool-demo")
async def cpu_bound_threadpool_demo():
    """
    10 threads, each does CPU-bound work.

    If threads could parallelize (true parallelism): ~1 second on 10-core CPU
    If GIL serializes them: ~10 seconds (one thread at a time)

    Expected: ~10 seconds ❌ (GIL blocks)
    """
    loop = asyncio.get_event_loop()
    start = time.time()

    # Submit 4 CPU tasks (not 10, to avoid overload)
    tasks = [
        loop.run_in_executor(thread_pool, cpu_bound_work, i, 10_000_000)
        for i in range(4)
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    return {
        "approach": "ThreadPool CPU-bound",
        "threads": 4,
        "elapsed_sec": round(elapsed, 2),
        "theory": "~4s (GIL blocks—threads execute sequentially)",
        "why": "CPU work holds GIL; other threads MUST wait for their turn",
        "lesson": "ThreadPool is BAD for CPU work. Use ProcessPool instead.",
    }

# ──────────────────────────────────────────────────────────────
# SCENARIO 3: CPU-bound with ProcessPool
# QUESTION: Do other processes wait?
# ANSWER: NO. Each process has its own GIL (true parallelism).
# ──────────────────────────────────────────────────────────────

@app.get("/cpu-bound-processpool-demo")
async def cpu_bound_processpool_demo():
    """
    4 processes, each does CPU-bound work.

    Each process has its own GIL:
      ├─ Process 1: owns GIL #1 (can compute freely)
      ├─ Process 2: owns GIL #2 (can compute freely, parallel with #1)
      ├─ Process 3: owns GIL #3 (can compute freely)
      └─ Process 4: owns GIL #4 (can compute freely)

    Expected: ~1 second (true parallelism on multi-core CPU) ✅
    """
    loop = asyncio.get_event_loop()
    start = time.time()

    # Submit 4 CPU tasks
    tasks = [
        loop.run_in_executor(process_pool, cpu_bound_work, i, 10_000_000)
        for i in range(4)
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    return {
        "approach": "ProcessPool CPU-bound",
        "processes": 4,
        "elapsed_sec": round(elapsed, 2),
        "theory": "~1s (each process has own GIL—true parallelism)",
        "why": "Separate processes = separate GILs = can compute in parallel",
        "lesson": "ProcessPool is GOOD for CPU work (but memory-heavy).",
    }

# ──────────────────────────────────────────────────────────────
# SCENARIO 4: Detailed GIL contention analysis
# ──────────────────────────────────────────────────────────────

def detailed_io_work(thread_id: int):
    """
    Detailed tracking: which thread gets GIL and when?
    """
    print(f"  [{threading.current_thread().name}] START (has GIL)")
    print(f"  [{threading.current_thread().name}] Making I/O call...")
    time.sleep(0.5)  # I/O (GIL released here)
    print(f"  [{threading.current_thread().name}] I/O complete (reacquired GIL)")
    print(f"  [{threading.current_thread().name}] Processing result...")
    time.sleep(0.1)  # CPU (GIL held)
    print(f"  [{threading.current_thread().name}] DONE")
    return thread_id

@app.get("/gil-contention-demo")
async def gil_contention_demo():
    """
    Show exact order of GIL acquisition with 3 threads doing I/O.
    """
    loop = asyncio.get_event_loop()

    print("\n=== GIL Contention Demo (3 threads, I/O-bound) ===\n")
    start = time.time()

    tasks = [
        loop.run_in_executor(thread_pool, detailed_io_work, i)
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start

    return {
        "elapsed_sec": round(elapsed, 2),
        "observation": "All 3 threads made I/O calls concurrently (GIL released)",
        "why": "During I/O, GIL is released, so threads don't block each other",
        "timeline": "~0.5s (all threads wait on I/O in parallel) + 0.1s (sequential CPU)",
    }

# ──────────────────────────────────────────────────────────────
# SCENARIO 5: The "Thread Starvation" Problem
# ──────────────────────────────────────────────────────────────

def long_cpu_work(thread_id: int):
    """CPU work that takes a long time."""
    print(f"[Thread {thread_id}] CPU work 5 sec...")
    time.sleep(0.01)  # Simulate without actually burning CPU
    total = sum(i * i for i in range(50_000_000))
    print(f"[Thread {thread_id}] CPU work done")
    return total

@app.get("/thread-starvation-demo")
async def thread_starvation_demo():
    """
    ThreadPoolExecutor with 10 workers, but only 2 long CPU tasks.

    Question: Do the other 8 threads help?
    Answer: NO. GIL ensures only 1 thread computes at a time.

    This shows why ThreadPool is bad for CPU-bound work.
    """
    loop = asyncio.get_event_loop()
    start = time.time()

    # 2 CPU-heavy tasks in a pool of 10 threads
    tasks = [
        loop.run_in_executor(thread_pool, long_cpu_work, i)
        for i in range(2)
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start

    return {
        "pool_size": 10,
        "tasks": 2,
        "elapsed_sec": round(elapsed, 2),
        "theory": "~2 seconds (tasks run sequentially due to GIL)",
        "question": "Why don't the other 8 threads help?",
        "answer": "GIL. Only 1 thread can execute Python code at a time.",
        "lesson": "ThreadPoolExecutor is useless for CPU-bound work.",
    }

# ──────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────

"""
┌────────────────────────────────────────────────────────────────────────┐
│ Do Other Threads Wait? (10 threads in ThreadPoolExecutor)            │
├─────────────────────────────┬──────────────┬──────────────────────────┤
│ What the Thread Does        │ GIL Status   │ Other Threads Wait?     │
├─────────────────────────────┼──────────────┼──────────────────────────┤
│ Network I/O (httpx.get)     │ RELEASED ✓   │ NO ❌                  │
│ Disk I/O (file.read)        │ RELEASED ✓   │ NO ❌                  │
│ Database query (asyncpg)    │ RELEASED ✓   │ NO ❌                  │
│ time.sleep()                │ RELEASED ✓   │ NO ❌                  │
│ → Result: All threads       │              │ can execute             │
│   wait on I/O simultaneously │             │ (pseudo-parallel)       │
│                             │              │                        │
├─────────────────────────────┼──────────────┼──────────────────────────┤
│ CPU loop (sum, compute)     │ HELD 🔒      │ YES ❌                 │
│ Math operations             │ HELD 🔒      │ YES ❌                 │
│ String manipulation         │ HELD 🔒      │ YES ❌                 │
│ JSON parsing                │ HELD 🔒      │ YES ❌                 │
│ → Result: Only 1 thread     │              │ executes at a time      │
│   can compute; others        │              │ (no parallelism!)       │
│                             │              │                        │
├─────────────────────────────┼──────────────┼──────────────────────────┤
│ Mixed (I/O + CPU)           │ RELEASED → HELD → RELEASED │            │
│ → Result: Threads take      │              │ turns during I/O,       │
│   turns but don't block     │              │ block during CPU        │
└─────────────────────────────┴──────────────┴──────────────────────────┘

KEY INSIGHT:
  The GIL is DYNAMIC. It's held during CPU, released during I/O.
  ├─ I/O-bound: ✅ ThreadPool works (GIL released during waits)
  └─ CPU-bound: ❌ ThreadPool FAILS (GIL held the whole time)

REAL-WORLD IMPACT:

Scenario A: 10 threads doing 1-second I/O each
  ├─ Without GIL (ProcessPool): ~0.1s (all parallel)
  ├─ With GIL + I/O release (ThreadPool): ~1s (all wait simultaneously)
  └─ Result: ThreadPool is FINE here (I/O dominates)

Scenario B: 10 threads doing 1-second CPU each
  ├─ Without GIL (ProcessPool): ~0.25s (4 processes on 4 cores)
  ├─ With GIL (ThreadPool): ~10s (only 1 thread at a time)
  └─ Result: ThreadPool is 40x SLOWER! ❌

INTERVIEW ANSWER:
  "If one thread in a 10-thread ThreadPool is executing, the other 9 need
   to wait—BUT ONLY if that thread is doing CPU work. If it's doing I/O,
   the GIL is released, and other threads can execute. This is why
   ThreadPool is great for I/O-bound work but terrible for CPU-bound.
   The GIL is the limiting factor."
"""

if __name__ == "__main__":
    import uvicorn

    print(__doc__)
    uvicorn.run(
        "01C_gil_threadpool_behavior:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
    )
