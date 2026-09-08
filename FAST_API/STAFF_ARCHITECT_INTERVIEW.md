STAFF_ARCHITECT_INTERVIEW.md
# FastAPI Staff/Architect Interview Guide

**Level**: Staff/Architect (5+ years backend engineering, Python, distributed systems)  
**Focus**: Production systems, async patterns, type safety, systems design, architectural decisions

---

## Part 1: Async & Concurrency Mastery

### Q1: Explain the difference between concurrency and parallelism in Python, and how FastAPI handles each.

**Expected depth**: 
- Concurrency: logical interleaving (async/await, event loop)
- Parallelism: true simultaneous execution (threads, processes)
- FastAPI: primarily async/concurrency for I/O-bound; thread pool for CPU-bound; worker processes for true parallelism
- Event loop semantics: single-threaded, GIL implications
- When to use `asyncio.run_in_executor()` vs thread pool vs separate workers

**When to use each**:

| Use Case | Method | Why | Example |
|----------|--------|-----|---------|
| **Async I/O** | `async/await` | Native, zero overhead | `await db.query()`, `await httpx.get()` |
| **Sync I/O in async** | `run_in_executor(thread_pool, ...)` | Non-blocking, GIL released during I/O | `loop.run_in_executor(None, requests.get, url)` |
| **CPU-bound in async** | `run_in_executor(process_pool, ...)` | Bypass GIL, true parallelism | `loop.run_in_executor(process_pool, ml_model.predict, data)` |
| **Sync endpoint** | Plain `def` (FastAPI auto-threads) | FastAPI handles threading | `@app.get("/sync")` → auto ThreadPool |
| **Heavy computation** | `gunicorn -w N` workers | Separate processes, scale horizontally | ML inference, data processing |
| **Shared state needed** | Single worker + async | Avoid duplication, simpler | `uvicorn --workers 1` (for models) |

**Code comparison**:

```python
# ❌ WRONG: Blocks entire server
@app.get("/bad")
async def bad():
    time.sleep(1)  # Event loop frozen!
    return "done"

# ✅ RIGHT: Async I/O (no executor needed)
@app.get("/async-io")
async def async_io():
    async with httpx.AsyncClient() as client:
        return await client.get("https://example.com")

# ✅ RIGHT: Sync I/O via ThreadPool executor
thread_pool = ThreadPoolExecutor(max_workers=8)

@app.get("/sync-io")
async def sync_io():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(thread_pool, requests.get, "https://example.com")
    return result

# ✅ RIGHT: CPU-bound via ProcessPool executor
process_pool = ProcessPoolExecutor(max_workers=2)

def heavy_compute(n: int):
    return sum(i*i for i in range(n))

@app.get("/cpu-bound")
async def cpu_bound():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(process_pool, heavy_compute, 10_000_000)
    return {"result": result}

# ✅ RIGHT: Plain sync endpoint (FastAPI auto-threads)
@app.get("/sync-auto")
def sync_auto():
    time.sleep(1)  # Runs in ThreadPool automatically
    return "done"

# ✅ RIGHT: Multiple workers for heavy workloads
# Command: gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
# Each worker gets own lifespan, loads models, etc.
# True parallelism across cores
```

**Decision tree**:

```
Does the task involve I/O (network, disk, DB)?
├─ YES
│  ├─ Can you use async library (asyncpg, httpx, aioredis)?
│  │  ├─ YES → Use async/await (best)
│  │  └─ NO → Use run_in_executor(thread_pool, sync_func)
│  │
│  └─ IO is blocking → auto-run in ThreadPool (plain def endpoint)
│
└─ NO (pure CPU computation)
   ├─ Single-threaded OK? (no parallelism needed)
   │  └─ YES → Just compute in endpoint
   │
   └─ Need parallelism?
      ├─ Within single request → run_in_executor(process_pool, ...)
      └─ Across requests → gunicorn -w N (separate workers)
```

**Real-world scenarios**:

```python
# Scenario 1: Database query (I/O-bound)
# ✅ BEST: Native async
@app.get("/user/{user_id}")
async def get_user(user_id: int):
    return await db.fetchrow("SELECT * FROM users WHERE id=$1", user_id)

# Scenario 2: Legacy requests library (sync I/O)
# ✅ GOOD: ThreadPool executor
@app.get("/external-api")
async def call_external():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, requests.get, "https://api.example.com")

# Scenario 3: ML model inference (CPU-bound)
# ✅ GOOD: ProcessPool executor
@app.get("/predict")
async def predict(text: str):
    loop = asyncio.get_event_loop()
    prediction = await loop.run_in_executor(process_pool, model.predict, text)
    return {"prediction": prediction}

# Scenario 4: Heavy model loading (shared across requests)
# ✅ BEST: Load in lifespan, reuse from memory
@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()  # Once at startup
    yield
    unload_model()

@app.get("/predict")
async def predict(text: str):
    # Model already in memory (no loading overhead)
    return {"prediction": app.state.model.predict(text)}

# Scenario 5: CPU + I/O mixed (e.g., ML inference + DB save)
# ✅ GOOD: ProcessPool for compute, async for DB
@app.post("/predict-and-save")
async def predict_and_save(text: str):
    # CPU: Use ProcessPool
    loop = asyncio.get_event_loop()
    prediction = await loop.run_in_executor(process_pool, model.predict, text)
    
    # I/O: Use async
    await db.execute("INSERT INTO predictions VALUES ($1, $2)", text, prediction)
    
    return {"prediction": prediction}
```

**Memory/Performance implications**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Method              │ Memory Overhead │ Latency │ Use When       │
├─────────────────────┼─────────────────┼─────────┼────────────────┤
│ async/await         │ None (0MB)      │ ~0ms    │ I/O-bound      │
│ ThreadPool          │ Light (1-2MB)   │ ~0ms    │ Sync I/O       │
│ ProcessPool         │ Heavy (150MB)   │ ~10ms   │ CPU-bound      │
│ gunicorn workers    │ High (150×N MB) │ 0ms     │ Scale horiz    │
└─────────────────────┴─────────────────┴─────────┴────────────────┘

In Docker (512MB limit):
├─ async/await: ✅ Safe
├─ ThreadPool: ✅ Safe
├─ ProcessPool(2): ⚠️ Risky (need ~300MB)
└─ gunicorn -w 4: ❌ OOM (each worker ~150MB)

Solution for Docker: Use fewer workers or offload to separate service
```

**Follow-ups**:

**Q: How does FastAPI's dependency injection interact with async context?**

**A**: Dependencies run in the same event loop thread as the endpoint. If a dependency is async, it's awaited before the endpoint runs. If it's sync (plain `def`), FastAPI runs it in the thread pool to avoid blocking.

```python
async def get_db() -> AsyncConnection:
    # Runs in event loop (async context)
    return await asyncpg.connect(dsn)

def get_logger() -> Logger:
    # Runs in thread pool (sync context)
    return Logger()

@app.get("/data")
async def get_data(db: Annotated[AsyncConnection, Depends(get_db)],
                   logger: Annotated[Logger, Depends(get_logger)]):
    # Both dependencies resolved before this runs
    # db: from event loop (async)
    # logger: from thread pool (sync)
    pass
```

**Q: Explain lifespan events and async context managers across worker processes.**

**A**: Lifespan runs ONCE per worker process at startup/shutdown. Each worker has its own lifespan, so resources are duplicated per worker (not shared).

```python
@asynccontextmanager
async def lifespan(app):
    # Runs once when THIS worker starts
    app.state.db_pool = await asyncpg.create_pool(...)
    yield
    # Runs once when THIS worker shuts down
    await app.state.db_pool.close()

# With gunicorn -w 4:
# Lifespan runs 4 times (once per worker)
# Each worker has its own db_pool
# NOT shared across workers
```

**Q: What happens if you block the event loop? How do you debug it?**

**A**: The entire server freezes—no requests processed until the block ends. Debug with:
1. `time.perf_counter()` around each operation
2. `await asyncio.sleep(0)` to yield and check if it resumes
3. `asyncio.timeout()` to fail fast instead of hang
4. Uvicorn logs `tasks pending` on shutdown

```python
@app.get("/debug")
async def debug():
    start = time.perf_counter()
    
    # ❌ BLOCKS (will freeze entire server for 5s)
    time.sleep(5)  # event loop blocked here
    
    elapsed = time.perf_counter() - start
    # If elapsed >> 5s, something is wrong with queuing
    return {"elapsed": elapsed}
```

---

### Q2: Design an async database connection pool for a production FastAPI app.

**Expected**:
- Use `asyncpg` (PostgreSQL) or `motor` (MongoDB), not synchronous drivers
- Connection pool sizing: `(worker_processes × workers_per_process) + overhead`
- Lifecycle management: acquire on startup, release on shutdown
- Connection leak detection and timeouts
- Retry logic with exponential backoff
- Circuit breaker for cascading failures

**Code sketch**:
```python
from contextlib import asynccontextmanager

async def lifespan(app):
    # Startup
    app.db_pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=10, max_size=20,
        timeout=10,
    )
    yield
    # Shutdown
    await app.db_pool.close()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    async with app.db_pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
```

**Follow-ups**:

**Q: How do you handle connection pool exhaustion in production?**

**A**: Monitor pool stats, implement queue timeouts, and use circuit breaker:

```python
async def get_db_with_timeout():
    try:
        async with asyncio.timeout(5):  # Fail fast
            conn = await app.db_pool.acquire()
    except asyncio.TimeoutError:
        logger.error("Pool exhausted")
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        yield conn
    finally:
        await app.db_pool.release(conn)

# Monitor in endpoint
@app.get("/health")
async def health():
    return {
        "pool_size": app.db_pool.get_size(),
        "pool_free": app.db_pool.get_idle_size(),
        "pool_min": app.db_pool.get_min_size(),
        "pool_max": app.db_pool.get_max_size(),
    }
```

**Q: What's the difference between connection pooling and statement pooling?**

**A**: 
- **Connection pooling**: Reuse TCP connections (expensive to create)
- **Statement pooling**: Prepare SQL statements on connection (faster execution)

PostgreSQL: asyncpg does connection pooling + prepared statement cache automatically.

**Q: How do you test async database code without hitting the real DB?**

**A**: Use `unittest.mock.AsyncMock` or `asyncpg.pool.Pool` with test fixtures:

```python
@pytest.fixture
async def db_pool_mock():
    mock = AsyncMock()
    mock.fetchrow = AsyncMock(return_value={"id": 1, "name": "Test"})
    yield mock

@pytest.mark.asyncio
async def test_get_user(db_pool_mock):
    result = await get_user(db_pool_mock, user_id=1)
    assert result["name"] == "Test"
    db_pool_mock.fetchrow.assert_called_once()
```

---

### Q3: Explain structured concurrency in Python and how to implement it safely.

**Expected**:
- Structured concurrency: tasks have defined entry/exit, no orphaned coroutines
- `asyncio.TaskGroup()` (Python 3.11+): cancel all on first exception
- Pre-3.11: use `asyncio.gather()` with exception handling or third-party libs (Trio, AnyIO)
- Resource cleanup guarantees (async with, finally blocks)
- Parent-child relationships, cancellation propagation

**Real scenario**: Multiple API calls to external services—how to handle partial failures?

```python
async with asyncio.TaskGroup() as tg:
    results = []
    for service in services:
        results.append(tg.create_task(call_service(service)))
    # All complete or all cancelled on exception
```

**Follow-ups**:

**Q: How does TaskGroup differ from gather()?**

**A**: TaskGroup cancels all tasks on first exception. gather() continues all tasks.

```python
# TaskGroup: one fails → all cancel
async with asyncio.TaskGroup() as tg:
    tg.create_task(task1())
    tg.create_task(task2())
    # If task1 raises, task2 is cancelled immediately

# gather(): one fails → others continue
results = await asyncio.gather(task1(), task2(), return_exceptions=True)
# If task1 raises, task2 still runs. Results: [Error, result2]
```

**Q: Design a timeout strategy for concurrent sub-tasks.**

**A**: Wrap TaskGroup in `asyncio.timeout()`:

```python
async with asyncio.timeout(30):  # 30s total budget
    async with asyncio.TaskGroup() as tg:
        tg.create_task(api_call_1())  # 10s
        tg.create_task(api_call_2())  # 15s
        tg.create_task(api_call_3())  # 5s
        # All must finish within 30s, or all cancelled
```

**Q: How to implement graceful degradation when some tasks fail?**

**A**: Use `gather(return_exceptions=True)` and handle failures:

```python
@app.get("/search")
async def search(q: str):
    results = await asyncio.gather(
        search_db(q),
        search_cache(q),
        search_external_api(q),
        return_exceptions=True,  # Don't raise, return errors
    )
    
    db_result = results[0] if not isinstance(results[0], Exception) else None
    cache_result = results[1] if not isinstance(results[1], Exception) else None
    api_result = results[2] if not isinstance(results[2], Exception) else None
    
    # Return best available result
    return {
        "db": db_result,
        "cache": cache_result,
        "api": api_result or "degraded (API down)",
    }
```

---

## Part 2: Type Safety & Python Typing

### Q4: How would you enforce type safety across a large FastAPI codebase?

**Expected**:
- Static checkers: `mypy`, `pyright`, `pydantic`
- Type hints everywhere: function signatures, class attributes, return types
- Pydantic v2 for runtime validation + static typing
- Generic types for reusable patterns (TypeVar, Generic, Protocol)
- Strict mode: `strict=True` in pyproject.toml

**Architecture**:
- Use Protocol for duck typing (esp. in dependency injection)
- Discriminated unions for polymorphic endpoints
- Type aliases for domain concepts (UserId = Annotated[int, ...])

**Example**:
```python
from typing import Protocol, TypeVar, Generic
from pydantic import BaseModel

T = TypeVar('T')

class Repository(Protocol[T]):
    async def get(self, id: int) -> T: ...
    async def save(self, item: T) -> None: ...

class User(BaseModel):
    id: int
    email: str

async def get_user(repo: Repository[User]) -> User:
    return await repo.get(1)
```

**Follow-ups**:

**Q: How do you type async generators?**

**A**: Use `AsyncGenerator[YieldType, SendType]`:

```python
from typing import AsyncGenerator

async def fetch_pages(url: str) -> AsyncGenerator[dict, None]:
    """Yields one page at a time."""
    for page in range(1, 10):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}?page={page}")
            yield response.json()

# Usage
async for page in fetch_pages("https://api.example.com"):
    print(page)
```

**Q: What's the difference between `Callable[[int], str]` and `Callable[..., str]`?**

**A**: 
- `Callable[[int], str]`: Takes exactly one `int`, returns `str`
- `Callable[..., str]`: Takes any arguments, returns `str`

```python
def strict(f: Callable[[int], str]) -> str:
    return f(42)

def loose(f: Callable[..., str]) -> str:
    return f(42, extra=True)

# strict(lambda x: str(x))  ✓ OK
# strict(lambda x, y: str(x + y))  ❌ Type error
# loose(lambda x, y: str(x + y))  ✓ OK
```

**Q: How do you test type safety without running the full codebase?**

**A**: Use static type checkers offline:

```bash
# mypy
mypy src/ --strict

# pyright
pyright src/

# pydantic validation
pytest --mypy  # pytest-mypy plugin
```

These catch type errors before runtime.

---

### Q4B: Explain Python typing options (Literal, Union, Optional, TypeVar, Generic, Protocol, Annotated)

**Expected depth**:
- **Literal**: Restrict to specific constant values
- **Union**: Accept multiple types (use `|` in Python 3.10+)
- **Optional**: Shorthand for `Union[T, None]`
- **TypeVar**: Generic type placeholder for reusable code
- **Generic**: Create generic classes/functions
- **Protocol**: Duck typing with static checking
- **Annotated**: Attach metadata (Pydantic, OpenAPI)
- **When to use each** in real FastAPI code

**Typing options comparison**:

```python
from typing import Literal, Union, Optional, TypeVar, Generic, Protocol, Annotated
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# 1. LITERAL: Restrict to specific constant values
# ─────────────────────────────────────────────────────────────

# ✅ GOOD: Type-safe enum-like values
class PaymentRequest(BaseModel):
    method: Literal["card", "bank", "crypto"]  # Only these 3 allowed
    status: Literal["pending", "completed", "failed"]

@app.post("/payment")
async def process_payment(req: PaymentRequest):
    match req.method:
        case "card":
            return await charge_card()
        case "bank":
            return await charge_bank()
        case "crypto":
            return await charge_crypto()
    # TypeChecker guarantees all cases covered

# Pydantic validates at runtime
payment = PaymentRequest(method="card", status="pending")  # ✓ OK
payment = PaymentRequest(method="pp", status="pending")  # ❌ ValidationError

# ─────────────────────────────────────────────────────────────
# 2. UNION: Accept multiple types (Python 3.10+ uses |)
# ─────────────────────────────────────────────────────────────

# ✅ GOOD: Multiple types, type checker knows all possibilities
def process_result(data: Union[str, int, dict]) -> None:
    if isinstance(data, str):
        print(data.upper())  # Type checker knows str methods available
    elif isinstance(data, int):
        print(data * 2)  # Type checker knows int methods
    elif isinstance(data, dict):
        print(data.keys())  # Type checker knows dict methods

# Python 3.10+ shorthand
def process_result(data: str | int | dict) -> None:
    pass

# ✅ GOOD: Discriminated unions for API responses
class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict

class ErrorResponse(BaseModel):
    status: Literal["error"]
    error: str

APIResponse = Union[SuccessResponse, ErrorResponse]

@app.get("/api/data")
async def get_data() -> APIResponse:
    try:
        return SuccessResponse(status="success", data={"id": 1})
    except Exception as e:
        return ErrorResponse(status="error", error=str(e))

# Client knows response type
response = get_data()
if response.status == "success":
    print(response.data)  # Type checker knows 'data' exists
elif response.status == "error":
    print(response.error)  # Type checker knows 'error' exists

# ─────────────────────────────────────────────────────────────
# 3. OPTIONAL: Shorthand for Union[T, None]
# ─────────────────────────────────────────────────────────────

# ❌ OLD: Verbose
def fetch_user(user_id: int) -> Union[User, None]:
    pass

# ✅ NEW: Concise (same meaning)
def fetch_user(user_id: int) -> Optional[User]:
    pass

# ✅ NEWER: Python 3.10+ (most readable)
def fetch_user(user_id: int) -> User | None:
    pass

# In Pydantic models
class Config(BaseModel):
    api_key: str  # Required
    timeout: Optional[int] = None  # Can be None (optional)
    debug: bool = False  # Has default

config = Config(api_key="xyz")  # ✓ OK
config = Config(api_key="xyz", timeout=30)  # ✓ OK
config = Config(api_key="xyz", timeout=None)  # ✓ OK

# ─────────────────────────────────────────────────────────────
# 4. TYPEVAR: Generic placeholder for reusable code
# ─────────────────────────────────────────────────────────────

T = TypeVar('T')  # Can be any type

# ✅ GOOD: Generic function that preserves type
def get_or_default(value: T, default: T) -> T:
    """Return value if truthy, else default. Type is preserved."""
    return value or default

# Type checker preserves type through function
result1: int = get_or_default(5, 0)  # ✓ Returns int
result2: str = get_or_default("hello", "world")  # ✓ Returns str
result3: str = get_or_default(5, "world")  # ❌ Type error (int vs str)

# ✅ GOOD: Constrained TypeVar
NumericT = TypeVar('NumericT', int, float)  # Only int or float allowed

def add(a: NumericT, b: NumericT) -> NumericT:
    return a + b

add(1, 2)  # ✓ Returns int
add(1.5, 2.5)  # ✓ Returns float
add("a", "b")  # ❌ Type error (str not in constraint)

# ─────────────────────────────────────────────────────────────
# 5. GENERIC: Create generic classes
# ─────────────────────────────────────────────────────────────

T = TypeVar('T')

# ✅ GOOD: Generic response wrapper
class Response(BaseModel, Generic[T]):
    status: Literal["success", "error"]
    data: Optional[T] = None
    error: Optional[str] = None

# Type-safe responses
UserResponse = Response[User]
ListResponse = Response[list[User]]

@app.get("/user/{user_id}", response_model=Response[User])
async def get_user(user_id: int) -> Response[User]:
    user = await db.fetch_one(User, user_id)
    return Response(status="success", data=user)

# Type checker knows response.data is User
@app.get("/users", response_model=Response[list[User]])
async def list_users() -> Response[list[User]]:
    users = await db.fetch_all(User)
    return Response(status="success", data=users)

# ✅ GOOD: Generic repository pattern
class Repository(Generic[T]):
    def __init__(self, model_class: type[T]):
        self.model_class = model_class
    
    async def get(self, id: int) -> Optional[T]:
        return await db.fetch_one(self.model_class, id)
    
    async def list(self) -> list[T]:
        return await db.fetch_all(self.model_class)

# Type-safe repositories
user_repo: Repository[User] = Repository(User)
order_repo: Repository[Order] = Repository(Order)

user = await user_repo.get(1)  # Type: Optional[User]
orders = await order_repo.list()  # Type: list[Order]

# ─────────────────────────────────────────────────────────────
# 6. PROTOCOL: Duck typing with static checking
# ─────────────────────────────────────────────────────────────

"""
PURPOSE OF PROTOCOL:

Problem Without Protocol:
  ├─ Duck typing: "If it walks like a duck, quacks like a duck, it's a duck"
  ├─ Works at runtime but NO static type checking
  ├─ Type checker can't verify compatibility before running code
  └─ Result: Silent type errors, hard to refactor

Example (no Protocol):
  def serialize(obj):  # Type checker: "what type is obj?"
      return obj.to_json()  # Does obj have to_json()? Unknown at check time!
  
  serialize(User())  # Works (has to_json)
  serialize("string")  # Runtime error (no to_json) ❌

Why Protocol Solves This:
  ├─ Define interface/contract without inheritance
  ├─ Type checker validates at STATIC check time (before running)
  ├─ Any object matching the interface is valid
  ├─ No need for inheritance or ABC
  └─ Result: Type-safe duck typing

With Protocol:
  class Serializable(Protocol):
      def to_json(self) -> str: ...
  
  def serialize(obj: Serializable):  # Type checker: "obj must have to_json()"
      return obj.to_json()
  
  serialize(User())  # ✓ Type checker verifies User has to_json
  serialize("string")  # ❌ Type error (caught before running!)

When to Use Protocol:
  ├─ Decoupling: Don't depend on concrete classes
  ├─ Dependency injection: Accept any compatible object
  ├─ Testing: Easy to mock without inheritance
  ├─ Plugins: Accept anything with the right interface
  └─ Flexibility: Multiple implementations of same interface

Protocol vs Inheritance:
  ┌──────────────┬────────────────────┬──────────────────────┐
  │ Aspect       │ Inheritance (ABC)  │ Protocol             │
  ├──────────────┼────────────────────┼──────────────────────┤
  │ Coupling     │ Tight (must inherit)│ Loose (any matching) │
  │ Boilerplate  │ Explicit inherit   │ Implicit matching    │
  │ Third-party  │ Hard to use        │ Easy (no change)     │
  │ Type checker │ Yes                │ Yes                  │
  │ Runtime      │ isinstance checks  │ No overhead          │
  └──────────────┴────────────────────┴──────────────────────┘

Example: Protocol advantages with third-party code
  
  # ABC Approach (requires inheritance)
  class Logger(ABC):
      @abstractmethod
      def log(self, msg: str) -> None: pass
  
  # Your code: UserService expects Logger
  class UserService:
      def __init__(self, logger: Logger):  # Must inherit from Logger!
          self.logger = logger
  
  # Third-party library: already has a logger
  class ExternalLibraryLogger:
      def log(self, msg: str) -> None:
          print(msg)
  
  # Problem: Can't use it! Must either:
  #   1. Modify third-party code (can't do)
  #   2. Create wrapper inheriting from Logger
  
  # Protocol Approach (no inheritance needed)
  class LoggerProtocol(Protocol):
      def log(self, msg: str) -> None: ...
  
  class UserService:
      def __init__(self, logger: LoggerProtocol):  # Accept any object with log()
          self.logger = logger
  
  # Works immediately! No changes needed to third-party code
  service = UserService(ExternalLibraryLogger())  # ✓ Type checker: OK!
"""

# ✅ GOOD: Define interface without inheritance
class Serializable(Protocol):
    """Any object with these methods is Serializable."""
    def to_dict(self) -> dict:
        ...
    
    def to_json(self) -> str:
        ...

class User:
    def to_dict(self) -> dict:
        return {"name": self.name, "email": self.email}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

class Order:
    def to_dict(self) -> dict:
        return {"id": self.id, "total": self.total}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

# Both User and Order implement Serializable protocol
def serialize_any(obj: Serializable) -> str:
    """Works with any object that has to_json()."""
    return obj.to_json()

serialize_any(User(...))  # ✓ OK
serialize_any(Order(...))  # ✓ OK
serialize_any("string")  # ❌ Type error (no to_json method)

# ✅ GOOD: Database client protocol (don't depend on specific DB)
class DBClient(Protocol):
    async def execute(self, query: str) -> None: ...
    async def fetch_one(self, query: str) -> dict: ...
    async def fetch_all(self, query: str) -> list[dict]: ...

class UserService:
    def __init__(self, db: DBClient):
        self.db = db  # Can be PostgreSQL, MySQL, MongoDB, mock, etc.
    
    async def get_user(self, user_id: int) -> User:
        row = await self.db.fetch_one(f"SELECT * FROM users WHERE id={user_id}")
        return User(**row)

# Works with any DB that implements DBClient protocol
postgres_service = UserService(PostgresClient())
mongo_service = UserService(MongoDBClient())
test_service = UserService(MockDBClient())

# ─────────────────────────────────────────────────────────────
# 7. ANNOTATED: Attach metadata (Pydantic, OpenAPI)
# ─────────────────────────────────────────────────────────────

# ✅ GOOD: Add validation/documentation without extra classes
class User(BaseModel):
    # Basic: just type
    name: str
    
    # With Pydantic metadata
    email: Annotated[str, Field(regex=r"^[\w\.-]+@[\w\.-]+\.\w+$")]
    age: Annotated[int, Field(ge=0, le=150)]
    bio: Annotated[str, Field(max_length=500, description="User biography")]

# ✅ GOOD: Custom metadata (e.g., logging, tracing)
UserId = Annotated[int, "database primary key"]
Email = Annotated[str, "user email, must be unique"]

class CreateUserRequest(BaseModel):
    user_id: UserId
    email: Email

# ✅ GOOD: Path/Query parameters with metadata
@app.get("/users/{user_id}")
async def get_user(
    user_id: Annotated[int, Path(description="User ID", ge=1)],
    skip: Annotated[int, Query(description="Skip first N records", ge=0)] = 0,
    limit: Annotated[int, Query(description="Max results", le=100)] = 10,
):
    pass

# OpenAPI documentation auto-generated from Annotated metadata

# ─────────────────────────────────────────────────────────────
# 8. COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
```

**Typing options comparison table**:

```
┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Type         │ Purpose          │ Best For         │ Runtime Check    │
├──────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Literal      │ Specific values  │ Enums, flags     │ Pydantic ✓       │
│ Union / |    │ Multiple types   │ Variants, API    │ Pydantic ✓       │
│ Optional     │ May be None      │ Optional fields  │ Pydantic ✓       │
│ TypeVar      │ Generic types    │ Reusable code    │ Type checking    │
│ Generic      │ Generic classes  │ Containers       │ Type checking    │
│ Protocol     │ Duck typing      │ Interfaces       │ Type checking    │
│ Annotated    │ Add metadata     │ Validation, docs │ Pydantic ✓       │
└──────────────┴──────────────────┴──────────────────┴──────────────────┘

Legend: Pydantic ✓ = Runtime validation, Type checking = Static checking only
```

**Decision flowchart**:

```
Is the type from a fixed set of values?
├─ YES (e.g., status, method, environment)
│  └─ Use Literal["value1", "value2"]
│
└─ NO: Can the value be multiple types?
   ├─ YES (but all needed in handler)
   │  └─ Use Union[Type1, Type2] or Type1 | Type2
   │
   └─ NO: Can it be None?
      ├─ YES
      │  └─ Use Optional[Type] or Type | None
      │
      └─ NO: Need reusable generic code?
         ├─ YES (function/class works for any type)
         │  ├─ Need interface duck typing?
         │  │  └─ Use Protocol[T]
         │  └─ NO
         │     ├─ Single type variable? → TypeVar
         │     └─ Multiple type variables? → Generic[T, U]
         │
         └─ NO: Need validation/metadata?
            ├─ YES
            │  └─ Use Annotated[Type, Field(...)]
            │
            └─ NO: Use plain type (int, str, dict, etc.)
```

**Real-world FastAPI examples**:

```python
from typing import Literal, Union, Optional, TypeVar, Generic, Protocol, Annotated
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# Complete REST API with all typing options
# ─────────────────────────────────────────────────────────────

# 1. Literal for fixed values
class OrderStatus(BaseModel):
    status: Literal["pending", "processing", "shipped", "delivered", "cancelled"]

# 2. Annotated for validation
class CreateOrderRequest(BaseModel):
    customer_id: Annotated[int, Field(gt=0)]
    items: Annotated[list, Field(min_items=1, max_items=100)]
    discount_code: Annotated[Optional[str], Field(max_length=20)] = None

# 3. Generic response wrapper with Union
T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    status: Literal["success", "error"]
    data: Optional[list[T]] = None
    error: Optional[str] = None
    pagination: Optional[dict] = None

# 4. Protocol for dependency injection
class PaymentProcessor(Protocol):
    async def charge(self, amount: float) -> dict: ...
    async def refund(self, transaction_id: str) -> dict: ...

# 5. Union for polymorphic responses
class SuccessResponse(BaseModel):
    status: Literal["success"]
    order_id: int

class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str
    code: str

OrderResponse = Union[SuccessResponse, ErrorResponse]

# 6. Complete endpoint with all typing
@app.post("/orders", response_model=OrderResponse)
async def create_order(
    request: CreateOrderRequest,
    payment: Annotated[PaymentProcessor, Depends(get_payment_processor)],
) -> OrderResponse:
    try:
        # Literal type ensures valid status
        result = await payment.charge(order_total)
        return SuccessResponse(status="success", order_id=order_id)
    except Exception as e:
        return ErrorResponse(status="error", message=str(e), code="PAYMENT_FAILED")

@app.get("/orders", response_model=PaginatedResponse[OrderStatus])
async def list_orders(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> PaginatedResponse[OrderStatus]:
    orders = await db.fetch_all(OrderStatus)
    return PaginatedResponse(
        status="success",
        data=orders[skip:skip+limit],
        pagination={"skip": skip, "limit": limit, "total": len(orders)}
    )
```

---

### Q5: Design a validation strategy for deeply nested Pydantic models.

**Expected**:
- Root validators vs field validators (v2 semantics)
- Cross-field validation
- Lazy vs eager validation
- Custom error messages and error enrichment
- Coercion vs rejection for type mismatches
- Performance: validation overhead at scale

**Scenario**: API accepts a complex payment form with conditional fields based on payment method.

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, Union

class CreditCard(BaseModel):
    number: str
    cvv: str
    expiry: str

class BankTransfer(BaseModel):
    account: str
    routing: str

class Payment(BaseModel):
    amount: float
    method: Literal["card", "bank"]
    details: Union[CreditCard, BankTransfer]
    
    @model_validator(mode='after')
    def validate_method_details(self):
        # Ensure details match method type
        pass
```

**Follow-ups**:

**Q: How do you validate against external systems (e.g., Stripe, bank APIs)?**

**A**: Use field validators with async calls in endpoint (not model):

```python
from pydantic import BaseModel

class PaymentMethod(BaseModel):
    card_number: str

@app.post("/charge")
async def charge(payment: PaymentMethod):
    # Validate in endpoint (can be async)
    is_valid = await stripe.verify_card(payment.card_number)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid card")
    
    return await process_charge(payment)
```

**Q: What's the cost of validation on P99 latency?**

**A**: Pydantic v2 validation is ~1-5ms for typical payloads. Deep nesting or custom validators add cost. Profile with:

```python
import time

start = time.perf_counter()
model = MyModel.model_validate(payload)
elapsed = (time.perf_counter() - start) * 1000
print(f"Validation took {elapsed:.2f}ms")
```

For P99: add ~10-20ms buffer for validation overhead.

**Q: How do you version API schemas without breaking backward compatibility?**

**A**: Union types + deprecation warnings:

```python
class UserV1(BaseModel):
    id: int
    name: str

class UserV2(BaseModel):
    id: int
    name: str
    email: str  # New field

@app.get("/users/{user_id}", response_model=Union[UserV1, UserV2])
async def get_user(user_id: int, api_version: int = Query(1)):
    user = await db.get_user(user_id)
    if api_version == 1:
        return UserV1(**user)
    else:
        return UserV2(**user)
```

---

## Part 3: Production Architecture & Scaling

### Q6: Design a FastAPI service for 100k requests/sec with sub-100ms p99 latency.

**Expected approach**:
1. **Load balancing**: NGINX/HAProxy with connection pooling, health checks
2. **Worker processes**: `gunicorn` with `uvicorn` workers, ~(CPU_cores × 2-4)
3. **Concurrency per worker**: `uvicorn --workers 4 --loop uvloop` (16-32 concurrent requests)
4. **Database**: Connection pooling, read replicas, query optimization
5. **Caching**: Redis for hot data, cache-aside pattern
6. **Async I/O**: All external calls (DB, APIs) async
7. **Monitoring**: Prometheus metrics, APM (New Relic, Datadog), distributed tracing
8. **Graceful shutdown**: Drain connections, wait for in-flight requests

**Architecture diagram**:
```
Internet → CloudFlare/WAF
  ↓
Load Balancer (NGINX)
  ↓
[Gunicorn Master]
  ├─ [Uvicorn Worker 1] (async event loop)
  ├─ [Uvicorn Worker 2]
  └─ [Uvicorn Worker N]
  ↓
[Database: Primary + Read Replicas]
[Redis: Cache]
[Message Queue: Celery/RabbitMQ for background tasks]
```

**Follow-ups**:

**Q: How do you handle thundering herd on cache expiration?**

**A**: Probabilistic early expiration + background refresh:

```python
import random

async def get_with_refresh(cache: Redis, key: str):
    cached = await cache.get(key)
    ttl = await cache.ttl(key)
    
    if cached:
        # Probabilistic refresh: if close to expiry, refresh in background
        if ttl < 60 and random.random() < 0.1:  # 10% chance
            asyncio.create_task(refresh_cache(key))
        return cached
    
    return await fetch_fresh_data(key)

async def refresh_cache(key: str):
    """Background refresh (low priority)."""
    data = await fetch_fresh_data(key)
    await cache.setex(key, 300, data)  # Refresh for 5 min
```

**Q: Design request deduplication for idempotent operations.**

**A**: Store request signature + result in cache:

```python
async def deduplicate(request: Request):
    # Create signature from user + operation
    sig = hashlib.sha256(f"{user_id}:{operation}:{request.body}").hexdigest()
    
    # Check if already processed
    cached_result = await cache.get(f"dedup:{sig}")
    if cached_result:
        return json.loads(cached_result)
    
    # Process
    result = await do_operation()
    
    # Store for 24h (deduplication window)
    await cache.setex(f"dedup:{sig}", 86400, json.dumps(result))
    return result
```

**Q: How do you test P99 latency without production load?**

**A**: Use load testing tools with controlled percentiles:

```bash
# Using k6
k6 run --vus 100 --duration 30s script.js
# Reports P50, P95, P99 latencies

# Or: Apache JMeter with plugins
```

---

### Q7: Explain your approach to error handling and observability at scale.

**Expected**:
- Structured logging (JSON, not free-form)
- Correlation IDs across requests
- Exception tracking: error categorization, user-facing vs internal
- Metrics: RED method (Rate, Errors, Duration)
- Distributed tracing: OpenTelemetry
- On-call runbook: how to triage failures

**Code**:
```python
import logging
from pythonjsonlogger import jsonlogger
from opentelemetry import trace, metrics

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)

@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    logger.info("fetching user", extra={"correlation_id": correlation_id, "user_id": user_id})
    
    try:
        user = await db.get_user(user_id)
        if not user:
            logger.warning("user not found", extra={"user_id": user_id})
            raise HTTPException(status_code=404)
        return user
    except Exception as e:
        logger.error("user fetch failed", exc_info=True, extra={"correlation_id": correlation_id})
        raise
```

**Follow-ups**:

**Q: How do you correlate logs across microservices?**

**A**: Pass correlation ID through headers + context vars:

```python
from contextvars import ContextVar

correlation_id: ContextVar[str] = ContextVar("correlation_id")

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    token = correlation_id.set(corr_id)
    
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    
    correlation_id.reset(token)
    return response

# In async calls to other services
async def call_external_service(url: str):
    corr_id = correlation_id.get()
    async with httpx.AsyncClient() as client:
        return await client.get(url, headers={"X-Correlation-ID": corr_id})
```

**Q: Design an alerting strategy: what do you alert on vs. investigate?**

**A**: 
- **Alert**: Error rate > 1%, P99 > 500ms, memory > 80%
- **Log (investigate)**: Individual errors, slow requests, warnings

```yaml
# Prometheus alerts
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
  for: 5m

- alert: HighLatency
  expr: histogram_quantile(0.99, http_request_duration_seconds_bucket) > 0.5
  for: 5m
```

**Q: How do you handle secrets in logs?**

**A**: Use filters to redact:

```python
import logging
import re

class SecretFilter(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r'(password|token|key)=\S+', r'\1=***', str(record.msg))
        return True

logger.addFilter(SecretFilter())
```

---

## Part 4: Advanced FastAPI Patterns

### Q8: Design a generic API versioning strategy.

**Expected**:
- URL versioning (`/v1/`, `/v2/`) vs header versioning vs body field versioning
- Backward compatibility: how long to support old versions?
- Deprecation warnings
- Type-safe migration: coexist old and new schemas

**Approach**:
```python
from fastapi import APIRouter, Header
from typing import Literal

# URL-based versioning
v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

@v1_router.get("/users/{user_id}", response_model=UserV1)
async def get_user_v1(user_id: int):
    pass

@v2_router.get("/users/{user_id}", response_model=UserV2)
async def get_user_v2(user_id: int):
    # UserV2 has additional fields
    pass

app.include_router(v1_router)
app.include_router(v2_router)
```

**Coexistence strategy**: Single handler, conditional response based on version.

**Follow-ups**:

**Q: How do you test API contract between versions?**

**A**: Contract tests ensure compatibility:

```python
@pytest.mark.asyncio
async def test_v1_v2_compatibility():
    """V2 should accept all V1 fields."""
    v1_payload = {"id": 1, "name": "Test"}
    
    # V1 works
    v1_response = await client.get("/v1/users/1")
    assert v1_response.json() == v1_payload
    
    # V2 works with same payload
    v2_response = await client.get("/v2/users/1")
    assert v2_response.json()["id"] == 1
    assert v2_response.json()["name"] == "Test"
```

**Q: Design a gradual migration: how to move clients from v1 to v2?**

**A**: Header-based version with fallback:

```python
from packaging import version

@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    api_version = request.headers.get("API-Version", "1")
    
    if version.parse(api_version) >= version.parse("2.0"):
        return await get_user_v2(user_id)
    else:
        return await get_user_v1(user_id)

# Client migration path:
# Phase 1: Add API-Version: 1 header (explicit)
# Phase 2: Add API-Version: 2 header (new clients)
# Phase 3: Default to v2, remove v1 support
```

**Q: How do you version internal service APIs differently from public APIs?**

**A**: Separate routers + different deprecation policies:

```python
public_v1 = APIRouter(prefix="/api/v1")  # Stable, 2-year support
internal_v1 = APIRouter(prefix="/internal/v1")  # Unstable, 3-month support

@public_v1.get("/users")
async def get_users_public():
    """Stable, backward compatible."""
    pass

@internal_v1.get("/users")
async def get_users_internal():
    """Fast iteration, breaking changes ok."""
    pass
```

---

### Q9: Implement a robust request/response interceptor pattern.

**Expected**:
- Middleware for logging, tracing, authentication
- Exception handlers for custom error responses
- Response compression
- Request size limits
- Rate limiting

**Code**:
```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from time import time
import logging

class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time()
        
        try:
            response = await call_next(request)
        except Exception as exc:
            duration = time() - start_time
            logger.error(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration * 1000,
                    "error": str(exc),
                },
            )
            raise
        
        duration = time() - start_time
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration * 1000,
            },
        )
        response.headers["X-Process-Time"] = str(duration)
        return response

app.add_middleware(RequestTimingMiddleware)
```

**Follow-ups**:

**Q: How do you handle middleware ordering?**

**A**: Add in reverse order (last add = first execute):

```python
app.add_middleware(LoggingMiddleware)      # Executes 3rd
app.add_middleware(AuthMiddleware)         # Executes 2nd
app.add_middleware(CorrelationIDMiddleware)  # Executes 1st
```

**Q: Design request body streaming for large uploads.**

**A**: Use `Request.stream()`:

```python
@app.post("/upload")
async def upload(request: Request):
    max_size = 100 * 1024 * 1024  # 100MB
    total = 0
    
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_size:
            raise HTTPException(status_code=413, detail="File too large")
        await process_chunk(chunk)
    
    return {"status": "ok", "total": total}
```

**Q: How do you implement request decompression?**

**A**: FastAPI/Starlette handles transparently via `Accept-Encoding`:

```python
# Client sends: gzip, deflate, br
# Starlette automatically decompresses
# No code needed—works out of the box

# If you need custom:
from starlette.middleware.gzip import GZIPMiddleware
app.add_middleware(GZIPMiddleware, minimum_size=1000)
```

---

### Q10: Design a dependency injection system for complex service graphs.

**Expected**:
- FastAPI's built-in DI vs. third-party frameworks (injector, dependency-injector)
- Scope management: request-scoped, app-scoped, singleton
- Circular dependencies: how to detect and resolve
- Type-safe dependency resolution
- Testing: mock dependencies

**Scenario**: Inject logger, DB connection, cache, and external service client into a handler.

```python
from typing import Annotated
from fastapi import Depends

class Logger:
    pass

class DatabaseConnection:
    pass

class RedisClient:
    pass

class ExternalServiceClient:
    pass

def get_logger() -> Logger:
    return Logger()

async def get_db() -> DatabaseConnection:
    async with DatabaseConnection() as conn:
        yield conn

def get_redis() -> RedisClient:
    return RedisClient()

def get_external_client(logger: Logger) -> ExternalServiceClient:
    return ExternalServiceClient(logger)

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    logger: Annotated[Logger, Depends(get_logger)],
    db: Annotated[DatabaseConnection, Depends(get_db)],
    redis: Annotated[RedisClient, Depends(get_redis)],
    external: Annotated[ExternalServiceClient, Depends(get_external_client)],
):
    pass
```

**Follow-ups**:

**Q: How do you handle dependency composition at scale?**

**A**: Use factories for complex graphs:

```python
class ServiceFactory:
    def __init__(self, db: Database, cache: Redis):
        self.db = db
        self.cache = cache
    
    def get_user_service(self) -> UserService:
        return UserService(self.db, self.cache)
    
    def get_order_service(self) -> OrderService:
        return OrderService(self.db, self.cache)

async def get_factory() -> ServiceFactory:
    db = await get_db()
    cache = await get_redis()
    return ServiceFactory(db, cache)

@app.get("/users/{user_id}")
async def get_user(user_id: int, factory: Annotated[ServiceFactory, Depends(get_factory)]):
    service = factory.get_user_service()
    return await service.get(user_id)
```

**Q: Design a factory pattern for dependencies.**

**A**: Use `Depends` chaining:

```python
async def get_db() -> Database:
    return await Database.connect()

async def get_user_repository(db: Database = Depends(get_db)) -> UserRepository:
    return UserRepository(db)

@app.get("/users/{user_id}")
async def get_user(user_id: int, repo: UserRepository = Depends(get_user_repository)):
    return await repo.get(user_id)
```

**Q: How do you test endpoints that depend on many services?**

**A**: Override dependencies in test client:

```python
app.dependency_overrides[get_db] = mock_db
app.dependency_overrides[get_cache] = mock_cache

@pytest.fixture
def client():
    return TestClient(app)

def test_get_user(client):
    response = client.get("/users/1")
    assert response.status_code == 200
    
# Cleanup
app.dependency_overrides.clear()
```

---

## Part 5: System Design & Architecture

### Q11: Design a real-time notification system built on FastAPI.

**Requirements**:
- 10M+ users, 1M concurrent connections
- Sub-second latency
- Exactly-once delivery for critical notifications
- Multi-channel: push, email, SMS, in-app

**Architecture**:
```
1. User connects via WebSocket → FastAPI → Redis Pub/Sub
2. Notification service publishes to Redis
3. WebSocket handlers subscribe to user-specific channels
4. Fallback: Celery tasks send email/SMS asynchronously

Scale:
- Redis Cluster for pub/sub (sharding by user_id)
- FastAPI instances behind load balancer
- Stateless: connection state in Redis or DB
```

**Code sketch**:
```python
from fastapi import WebSocket
from redis import Redis

redis = Redis()

@app.websocket("/ws/notifications/{user_id}")
async def websocket_notifications(websocket: WebSocket, user_id: int):
    await websocket.accept()
    pubsub = redis.pubsub()
    pubsub.subscribe(f"notifications:{user_id}")
    
    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                await websocket.send_json(json.loads(message['data']))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        pubsub.close()
        await websocket.close()
```

**Follow-ups**:

**Q: How do you handle connection failures and reconnections?**

**A**: Exponential backoff + heartbeat:

```python
@app.websocket("/ws/notifications/{user_id}")
async def websocket_notifications(websocket: WebSocket, user_id: int):
    await websocket.accept()
    
    async def listen():
        pubsub = redis.pubsub()
        pubsub.subscribe(f"notifications:{user_id}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_json(json.loads(message["data"]))
    
    async def heartbeat():
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    
    try:
        await asyncio.gather(listen(), heartbeat())
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Client reconnects with backoff
    finally:
        await websocket.close()
```

**Q: Design delivery guarantees: at-least-once, exactly-once?**

**A**: 
- **At-least-once**: Store notification + mark sent after delivery
- **Exactly-once**: Deduplication key (idempotent operations)

```python
async def send_notification(user_id: int, notification: dict):
    # Generate idempotent key
    notif_id = hashlib.sha256(str(notification).encode()).hexdigest()
    
    # Check if already sent
    if await cache.get(f"sent:{notif_id}"):
        return
    
    # Send via multiple channels
    try:
        await send_to_websocket(user_id, notification)
        await send_to_email(user_id, notification)
        await send_to_push(user_id, notification)
    except Exception as e:
        await db.insert("notifications_failed", {user_id, notification, error: e})
    
    # Mark as sent
    await cache.setex(f"sent:{notif_id}", 86400, "1")
```

**Q: How do you persist undelivered notifications?**

**A**: Database table + background worker:

```python
class Notification(BaseModel):
    id: int
    user_id: int
    content: str
    delivered: bool = False
    retries: int = 0
    created_at: datetime

# Background task: retry undelivered
async def retry_undelivered():
    while True:
        pending = await db.fetch(
            "SELECT * FROM notifications WHERE delivered = false AND retries < 5"
        )
        
        for notif in pending:
            try:
                await send_to_user(notif.user_id, notif.content)
                await db.execute("UPDATE notifications SET delivered = true WHERE id = $1", notif.id)
            except Exception:
                await db.execute("UPDATE notifications SET retries = retries + 1 WHERE id = $1", notif.id)
        
        await asyncio.sleep(60)  # Retry every minute
```

---

### Q12: Design a multi-tenant SaaS platform on FastAPI.

**Requirements**:
- 100+ tenants, 10M+ users
- Data isolation & compliance (GDPR, SOC2)
- Per-tenant customization (branding, workflows)
- Cost tracking per tenant

**Architecture**:
```
1. Routing: subdomain (acme.saas.com) or path (/acme/api)
2. Middleware: extract tenant_id, inject into context
3. Database: schema-per-tenant or row-level security
4. Cache: namespace by tenant
5. Billing: usage metrics per tenant
```

**Code**:
```python
from contextvars import ContextVar

tenant_context: ContextVar[str] = ContextVar('tenant_id', default=None)

class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID")
        token = tenant_context.set(tenant_id)
        response = await call_next(request)
        tenant_context.reset(token)
        return response

@app.get("/users")
async def list_users(db: DatabaseConnection):
    tenant_id = tenant_context.get()
    return await db.query("SELECT * FROM users WHERE tenant_id = $1", tenant_id)
```

**Follow-ups**:

**Q: How do you handle cross-tenant queries safely?**

**A**: Always filter by tenant in WHERE clause:

```python
@app.get("/users")
async def list_users(request: Request):
    tenant_id = request.state.tenant_id
    
    # ALWAYS filter by tenant—never trust client
    return await db.fetch(
        "SELECT * FROM users WHERE tenant_id = $1",
        tenant_id
    )

# If tenant_id is missing, 403 Forbidden (middleware should catch)
```

**Q: Design per-tenant rate limiting.**

**A**: Key by tenant:

```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda req: req.state.tenant_id)

@app.get("/api/users")
@limiter.limit("1000/minute")
async def list_users(request: Request):
    pass

# Each tenant gets independent 1000 req/min quota
```

**Q: How do you handle data deletion for GDPR compliance?**

**A**: Soft delete + background purge:

```python
class User(BaseModel):
    id: int
    email: str
    deleted_at: Optional[datetime] = None

@app.delete("/users/{user_id}")
async def delete_user(user_id: int, request: Request):
    tenant_id = request.state.tenant_id
    
    # Soft delete
    await db.execute(
        "UPDATE users SET deleted_at = now() WHERE id = $1 AND tenant_id = $2",
        user_id, tenant_id
    )
    
    return {"status": "deleted"}

# Background: hard delete after 30 days
async def purge_deleted():
    while True:
        await db.execute(
            "DELETE FROM users WHERE deleted_at < now() - interval '30 days'"
        )
        await asyncio.sleep(86400)  # Daily
```

---

### Q13: Design a GraphQL API on top of FastAPI.

**Requirements**:
- Type safety
- Complex nested queries
- N+1 query prevention
- Real-time subscriptions

**Approach**:
- Use Strawberry or Ariadne for GraphQL
- DataLoader for batching queries
- Async resolvers for performance

```python
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

@strawberry.type
class User:
    id: strawberry.ID
    name: str

@strawberry.type
class Query:
    @strawberry.field
    async def user(self, id: strawberry.ID) -> User:
        # Resolver logic
        pass

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
```

**Follow-ups**:

**Q: How do you implement authentication in GraphQL?**

**A**: JWT in headers + context:

```python
import strawberry

@strawberry.type
class Query:
    @strawberry.field
    async def me(self, info: strawberry.types.Info) -> User:
        request = info.context["request"]
        token = request.headers.get("Authorization")
        
        if not token:
            raise Exception("Unauthorized")
        
        user = await verify_jwt(token)
        return User(**user)

schema = strawberry.Schema(query=Query)

@app.post("/graphql")
async def graphql(request: Request):
    return await GraphQLRouter(schema)(request)
```

**Q: Design rate limiting per query complexity.**

**A**: Analyze query AST:

```python
from graphql import parse, get_depth

async def graphql_with_rate_limit(request: Request):
    body = await request.json()
    query_str = body.get("query")
    
    query = parse(query_str)
    depth = get_depth(query)
    
    if depth > 10:
        raise HTTPException(status_code=400, detail="Query too deep")
    
    # Process query
    return await execute_graphql(query)
```

**Q: How do you cache GraphQL responses?**

**A**: Cache by query signature:

```python
import hashlib

async def graphql_cached(request: Request):
    body = await request.json()
    query_sig = hashlib.sha256(str(body).encode()).hexdigest()
    
    # Check cache
    cached = await cache.get(f"gql:{query_sig}")
    if cached:
        return json.loads(cached)
    
    # Execute
    result = await execute_graphql(body)
    
    # Cache for 5 minutes
    await cache.setex(f"gql:{query_sig}", 300, json.dumps(result))
    
    return result
```

---

## Part 6: Testing & Quality

### Q14: Design a comprehensive testing strategy for FastAPI.

**Pyramid**:
- **Unit tests** (70%): business logic, validators, serializers
- **Integration tests** (20%): database, external APIs (mocked)
- **E2E tests** (10%): full request/response cycles

**Tools**:
- `pytest` with fixtures
- `httpx.AsyncClient` for testing async endpoints
- `testcontainers` for spinning up databases
- Property-based testing with Hypothesis

```python
import pytest
from httpx import AsyncClient

@pytest.fixture
async def client(app):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_get_user_not_found(client, db):
    response = await client.get("/users/999")
    assert response.status_code == 404
```

**Follow-ups**:

**Q: How do you test async database calls without a real DB?**

**A**: Use `testcontainers` or mock:

```python
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture
async def postgres():
    with PostgresContainer("postgres:15") as postgres:
        yield postgres

@pytest.mark.asyncio
async def test_with_real_db(postgres):
    dsn = postgres.get_connection_url()
    pool = await asyncpg.create_pool(dsn)
    
    # Real database, real queries
    user = await get_user(pool, user_id=1)
    assert user is not None
```

**Q: Design load testing for P99 latency.**

**A**: Use k6 with custom threshold:

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    vus: 100,
    duration: '5m',
    thresholds: {
        'http_req_duration': ['p(99)<500'],  // P99 < 500ms
        'http_req_failed': ['rate<0.01'],    // < 1% errors
    },
};

export default function () {
    let res = http.get('http://localhost:8000/users/1');
    check(res, {
        'status 200': (r) => r.status === 200,
        'latency < 500ms': (r) => r.timings.duration < 500,
    });
    sleep(1);
}
```

**Q: How do you test exception handling?**

**A**: Mock to raise exceptions:

```python
def test_db_failure():
    with patch('asyncpg.create_pool', side_effect=Exception("DB Down")):
        response = client.get("/users/1")
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]
```

---

### Q15: Implement continuous deployment for FastAPI without downtime.

**Deployment strategy**:
1. **Blue-green**: Two production environments, switch traffic
2. **Canary**: 5% → 25% → 100% traffic gradual rollout
3. **Database migrations**: Schema changes backward-compatible
4. **Graceful shutdown**: Drain existing connections, reject new ones

**Code**:
```python
import signal
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup
    app.state.is_shutting_down = False
    yield
    # Shutdown: drain
    app.state.is_shutting_down = True
    await asyncio.sleep(5)  # Wait for in-flight requests

@app.middleware("http")
async def health_check_middleware(request: Request, call_next):
    if app.state.is_shutting_down and request.url.path != "/health":
        return JSONResponse({"status": "shutting_down"}, status_code=503)
    return await call_next(request)

app = FastAPI(lifespan=lifespan)
```

**Follow-ups**:

**Q: How do you handle database schema migrations in a zero-downtime deployment?**

**A**: Expand-Contract pattern:

```sql
-- Phase 1: Add new column (backward compatible)
ALTER TABLE users ADD COLUMN email_v2 VARCHAR(255);

-- Phase 2: Backfill + switch writes
UPDATE users SET email_v2 = email;

-- Phase 3: Switch reads
SELECT email_v2 AS email FROM users;

-- Phase 4: Clean up old column (after old code retired)
ALTER TABLE users DROP COLUMN email;
```

**Q: Design feature flags for safe rollouts.**

**A**: Use feature flag service:

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    user = await fetch_user(user_id)
    
    # New logic only for canary users
    if feature_flags.is_enabled("new_recommendation_engine", user_id):
        user.recommendations = await new_recommender(user)
    else:
        user.recommendations = await old_recommender(user)
    
    return user

# Gradually enable:
# Day 1: 1% of users
# Day 2: 10% of users
# Day 3: 100% of users
```

**Q: How do you detect and rollback bad deployments automatically?**

**A**: Monitor error rate + auto-rollback:

```yaml
# Kubernetes: canary rollout with auto-rollback
apiVersion: fluxcd.io/v1beta1
kind: Kustomization
metadata:
  name: app
spec:
  serviceAccountName: flux
  interval: 5m
  sourceRef:
    kind: GitRepository
    name: app
  path: ./kustomize
  prune: true
  validation: client
  
  # Canary: 10% traffic for 5 min
  # If error rate > 5%, rollback automatically
```

---

## Part 7: Production Challenges & Trade-offs

### Q16: You have a memory leak in production affecting one endpoint. Walk through debugging.

**Approach**:
1. **Identify**: Prometheus memory metrics, `memory_profiler`
2. **Isolate**: Reproduce locally, narrow down endpoint
3. **Profile**: `py-spy`, `guppy3`, tracemalloc
4. **Common causes**:
   - Unbounded cache (Redis, in-memory dicts)
   - Unclosed database connections
   - Circular references in async tasks
   - Third-party libraries holding references

```python
import tracemalloc
tracemalloc.start()

# Run endpoint
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

**Follow-ups**:

**Q: How do you detect memory leaks in CI/CD?**

**A**: Run memory profiler in tests:

```python
import tracemalloc
import pytest

@pytest.fixture(autouse=True)
def memory_check():
    tracemalloc.start()
    yield
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    # Fail if top allocation > 100MB
    if top_stats[0].size > 100 * 1024 * 1024:
        raise MemoryError(f"Large allocation: {top_stats[0]}")
```

**Q: Design monitoring for GC pauses.**

**A**: Track GC stats:

```python
import gc

@app.get("/health/gc")
async def health_gc():
    stats = gc.get_stats()
    
    return {
        "gc_counts": gc.get_count(),  # Objects in each generation
        "gc_stats": [
            {
                "collections": s["collections"],
                "collected": s["collected"],
                "uncollectable": s["uncollectable"],
            }
            for s in stats
        ],
    }
```

**Q: How do you test for resource cleanup?**

**A**: Context manager + assertion:

```python
@pytest.mark.asyncio
async def test_connection_cleanup():
    async with asyncpg.create_pool(dsn) as pool:
        async with pool.acquire() as conn:
            pass
    
    # Verify pool is closed
    with pytest.raises(Exception):
        async with pool.acquire():
            pass
```

---

### Q17: Design a strategy for handling partial outages of external dependencies.

**Expected**:
- Circuit breaker (fail fast when dependency is down)
- Bulkhead pattern (isolate failures)
- Retry logic with exponential backoff
- Graceful degradation (serve stale data)
- Fallback: hardcoded defaults or cached responses

**Code**:
```python
from pybreaker import CircuitBreaker

stripe_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@stripe_breaker
async def charge_user(user_id: int, amount: float):
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.stripe.com/charges", json={...})
        return response.json()

@app.post("/checkout")
async def checkout(order: Order):
    try:
        charge = await charge_user(order.user_id, order.amount)
        return {"status": "success", "charge_id": charge['id']}
    except CircuitBreakerListener:
        # Fallback: mark order for manual review
        return {"status": "pending_review", "message": "Payment processing delayed"}
```

**Follow-ups**:

**Q: How do you test circuit breakers?**

**A**: Mock failures:

```python
def test_circuit_breaker_trips():
    breaker = CircuitBreaker(fail_max=3, reset_timeout=1)
    
    # Simulate 3 failures
    for _ in range(3):
        with pytest.raises(Exception):
            breaker.call(failing_function)
    
    # Circuit open
    with pytest.raises(CircuitBreakerListener):
        breaker.call(any_function)
    
    # Wait for reset
    time.sleep(1.1)
    
    # Circuit half-open, tries again
    breaker.call(working_function)
```

**Q: Design retry budgets: how many retries before giving up?**

**A**: Exponential backoff with max attempts:

```python
async def call_with_retry(url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                return await client.get(url, timeout=5)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(f"Retry {attempt + 1}/{max_retries}, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
```

**Q: How do you communicate dependency outages to clients?**

**A**: Graceful degradation with status:

```python
@app.post("/checkout")
async def checkout(order: Order):
    try:
        charge = await stripe.charge(order.amount)
        return {"status": "success", "charge_id": charge["id"]}
    except Exception:
        # Fallback
        return {
            "status": "pending",
            "message": "Payment processing delayed. We'll confirm via email shortly.",
            "order_id": order.id,
        }
```

---

### Q18: Design a solution for preventing cascade failures in a microservice architecture.

**Expected**:
- Timeout propagation: per-service, per-request
- Load shedding: drop low-priority requests under stress
- Resource limits: connection pools, queue sizes
- Bulkhead isolation: separate thread pools for critical vs. non-critical
- Adaptive throttling: adjust concurrency based on error rate

**Example**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/critical-endpoint")
@limiter.limit("1000/minute")
async def critical_endpoint():
    pass

@app.get("/non-critical")
@limiter.limit("100/minute")
async def non_critical():
    pass
```

**Follow-ups**:

**Q: How do you detect early signs of cascade failure?**

**A**: Monitor these metrics:

```python
# Prometheus queries
- queue_depth > 1000
- error_rate > 5%
- p99_latency > 1000ms
- memory_usage > 80%
- thread_pool_active > pool_size * 0.9
```

**Q: Design adaptive timeouts based on system load.**

**A**: Adjust timeout by system metrics:

```python
async def call_with_adaptive_timeout(url: str):
    # Check system load
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    load_factor = (cpu_percent + memory_percent) / 200  # 0-1
    
    # Base timeout 5s, up to 30s
    timeout = 5 + (load_factor * 25)
    
    async with httpx.AsyncClient() as client:
        return await client.get(url, timeout=timeout)
```

**Q: How do you test cascade failure scenarios?**

**A**: Chaos engineering:

```python
# chaos.py: simulate failures
import random
from unittest.mock import patch

def inject_latency():
    """Inject 100ms latency to all HTTP calls."""
    return asyncio.sleep(0.1)

def inject_errors(rate: float = 0.1):
    """Inject errors to 10% of calls."""
    if random.random() < rate:
        raise Exception("Simulated failure")

# In test
@pytest.fixture
def chaos():
    with patch("httpx.get", side_effect=inject_errors(0.1)):
        yield

def test_cascade_resilience(chaos):
    # System should degrade gracefully
    response = client.get("/search")
    assert response.status_code in [200, 503]  # OK or degraded
```

---

## Part 8: Leadership & Trade-offs

### Q19: You're building a new FastAPI service. What's your tech stack and why?

**Expected to discuss**:
- FastAPI + Uvicorn + Gunicorn
- PostgreSQL vs. NoSQL trade-offs
- Redis for caching/sessions
- Message queue (Celery + RabbitMQ vs. Kafka)
- Deployment: Kubernetes vs. serverless
- Monitoring: Prometheus + Grafana + Datadog
- Async runtime: asyncio vs. uvloop

**Justify each choice**: Cost, scalability, team expertise, latency SLA.

---

### Q20: How do you balance code quality vs. feature velocity?

**Expected**:
- Type safety + linting (mypy, ruff) catches bugs early
- Fast iteration on tests (pytest fixtures, mocking)
- Code review rigor: what to enforce vs. suggest
- Technical debt tracking: pay it down incrementally
- Measure: defect density, on-call load, MTTR (Mean Time To Recovery)

---

## Part 9: Rapid-Fire / Situational

### Q21-Q25: Rapid-fire scenarios (pick 3-4)

**Q21**: How would you optimize a slow GraphQL query hitting 5 different tables?
- DataLoader for batching
- Caching strategies
- Database indexes and query plans
- N+1 problem resolution

**Q22**: Your API latency is normally 50ms P50, but suddenly spikes to 500ms. How do you investigate?
- Check metrics: CPU, memory, network, database
- Correlate with deployments, database changes
- Profile specific endpoints
- Check external dependencies

**Q23**: Design an API for uploading large files (>1GB) reliably.
- Chunked uploads with resumption
- Content-addressable storage (hash-based)
- Virus scanning on chunks
- Cleanup for incomplete uploads

**Q24**: How would you implement OAuth2 with multiple identity providers (Google, GitHub, corporate LDAP)?
- Separate providers, unified user model
- Account linking strategy
- Token refresh and expiration
- Role-based access control

**Q25**: Design a fair usage policy / multi-tenant quota system.
- Per-tenant daily limits
- Burst allowance (token bucket)
- Priority queues for upgrades
- Soft limits with warnings, hard limits with rejection

---

## Scoring Rubric

### Strong (Hire)
- Demonstrates 5+ years backend experience with real production systems
- Thinks about trade-offs: latency, cost, scalability, team knowledge
- Shows deep async/concurrency understanding
- Can design systems that scale to 100k+ req/sec
- Asks clarifying questions before diving in
- Discusses monitoring, observability, on-call burden

### Competent (Maybe)
- Knows FastAPI fundamentals (endpoints, dependencies, Pydantic)
- Can implement CRUD APIs, understand basics of async
- Limited production experience or scaling challenges faced
- Focuses more on feature implementation than architectural concerns

### Not ready (Pass)
- Struggles with async concepts
- No production deployment experience
- Cannot reason about scalability or failure modes
- Treats all endpoints the same (no prioritization of critical paths)

---

## Follow-up Topics (Deep Dives)

If candidate shows strong fundamentals, dig into:
1. **Distributed systems**: Consensus (Raft, Paxos), eventual consistency
2. **Database internals**: Query optimization, index design, transaction isolation
3. **Kubernetes**: Resource requests/limits, RBAC, networking policies
4. **Cost optimization**: Instance sizing, reserved capacity, spot instances
5. **Security**: Input validation, authentication, authorization, encryption
6. **Open-source contributions**: Have they built reusable libraries?

---

## Assessment & Evaluation

**Strong Hire (Staff/Architect Level)**:
- Answers questions with reasoning, not memorized solutions
- Discusses trade-offs: latency vs cost, simplicity vs scale
- Asks clarifying questions before diving in
- Demonstrates real production experience (incidents, scaling, on-call)
- Thinks about observability, monitoring, alerting
- Considers failure modes and edge cases
- Can design systems for 100k+ req/sec

**Competent (Senior Engineer)**:
- Answers basics correctly with some depth
- Limited production scaling experience
- Focuses on feature implementation over architecture
- Understands async fundamentals but limited pattern knowledge

**Not Ready**:
- Struggles with async/await concepts
- No clear production deployment experience
- Treats all endpoints the same (no prioritization)
- Cannot reason about failure modes

---