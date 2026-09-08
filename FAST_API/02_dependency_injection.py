# 02_dependency_injection.py

```python
"""
========================================================
TOPIC 2: Dependency Injection with Depends()
========================================================

KEY CONCEPTS:
- Depends()        → Declares that an endpoint needs something resolved first.
- Sub-dependencies → A dependency can itself depend on other dependencies.
                     FastAPI builds the full dependency graph automatically.
- Class-based Deps → Use classes when the dependency holds config/state.
- Override         → Swap real dependencies with mocks in tests without
                     changing endpoint code. Critical for unit testing.

INTERVIEW ANSWER TIP:
  "FastAPI's DI system is declarative. I declare what I need in the
   function signature; FastAPI resolves, caches (within a request), and
   injects them. For testing I use app.dependency_overrides to swap in
   mocks without touching production code."
"""

from typing import Annotated, Generator

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.testclient import TestClient

app = FastAPI(title="Dependency Injection Demo")


# ──────────────────────────────────────────────────────────────
# 2A. Simple function dependency — query param validation
# ──────────────────────────────────────────────────────────────
def common_pagination(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, le=100),
) -> dict:
    """
    Reusable pagination. Any endpoint can Depends() on this.
    FastAPI injects skip/limit from the query string automatically.
    """
    return {"skip": skip, "limit": limit}


# Annotated[dict, Depends(common_pagination)] — two things in one expression:
#   dict                   → the TYPE of value this dependency returns
#   Depends(common_pagination) → tells FastAPI to CALL common_pagination()
#                                and inject its return value here
#
# Without Annotated (old way — repetitive across every endpoint):
#   def list_items(pagination: dict = Depends(common_pagination)): ...
#   def get_user(pagination: dict = Depends(common_pagination)): ...
#
# With Annotated (new way — define once, reuse as a type alias):
#   PaginationDep = Annotated[dict, Depends(common_pagination)]
#   def list_items(pagination: PaginationDep): ...   ← clean, no repetition
#   def get_user(pagination: PaginationDep): ...     ← same dep, one source
#
# FastAPI reads Annotated metadata at startup — sees Depends() → knows to
# call common_pagination() before the endpoint runs and pass result in.
PaginationDep = Annotated[dict, Depends(common_pagination)]


@app.get("/items")
def list_items(pagination: PaginationDep):
    """
    GET /items?skip=0&limit=5
    FastAPI calls common_pagination() first, then passes result here.
    """
    fake_db = list(range(100))
    p = pagination
    return fake_db[p["skip"] : p["skip"] + p["limit"]]


# ──────────────────────────────────────────────────────────────
# 2B. Class-based dependency — holds config / connection
# ──────────────────────────────────────────────────────────────
class DatabaseSession:
    """
    Simulates a DB session dependency.
    In production: SQLAlchemy Session, Motor AsyncIOMotorClient, etc.
    """

    def __init__(self, db_url: str = "sqlite:///./test.db"):
        self.db_url = db_url
        # self.session = SessionLocal()  ← real pattern

    def query(self, model: str) -> list:
        """Fake DB query."""
        return [{"id": 1, "model": model, "db": self.db_url}]

    def close(self):
        pass  # self.session.close()


def get_db() -> Generator:
    """
    Generator dependency → code AFTER yield runs as teardown.
    FastAPI guarantees teardown even if the endpoint raises an exception.
    Pattern: open resource → yield → close resource.
    """
    db = DatabaseSession()
    try:
        yield db          # ← endpoint receives this
    finally:
        db.close()        # ← always runs (like a context manager)


DbDep = Annotated[DatabaseSession, Depends(get_db)]


@app.get("/users")
def get_users(db: DbDep):
    return db.query("User")


# ──────────────────────────────────────────────────────────────
# 2C. Sub-dependencies (dependency chain)
# ──────────────────────────────────────────────────────────────
def get_api_key(x_api_key: str = Header(...)) -> str:
    """Level 1: extract API key from header."""
    if x_api_key != "secret-key-123":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bad API key")
    return x_api_key


def get_current_user(api_key: str = Depends(get_api_key)) -> dict:
    """
    Level 2: depends on get_api_key.
    FastAPI resolves the full chain: header → api_key → current_user.
    Each dependency is resolved ONCE per request and cached.
    """
    return {"user": "vinoth", "role": "admin", "api_key": api_key}


CurrentUserDep = Annotated[dict, Depends(get_current_user)]


@app.get("/profile")
def get_profile(user: CurrentUserDep):
    """
    This endpoint needs: X-Api-Key header → validated → user dict.
    The chain is automatic; just declare the final dependency.
    """
    return {"profile": user}


# ──────────────────────────────────────────────────────────────
# 2D. Class with __call__ — stateful dependency
# ──────────────────────────────────────────────────────────────
class RoleChecker:
    """
    Parameterized dependency. The class is configured at definition time;
    __call__ is invoked per request.
    """

    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: CurrentUserDep) -> dict:
        """Called per request — receives the user and validates."""
        if user["role"] not in self.allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

# Create an instance with specific config
require_admin = RoleChecker(allowed_roles=["admin", "superuser"])

# FastAPI calls it like: require_admin(user) → triggers __call__
@app.get("/admin-panel")
def admin_panel(user: Annotated[dict, Depends(require_admin)]):
    return {"message": f"Welcome admin {user['user']}"}


# ──────────────────────────────────────────────────────────────
# 2E. Dependency Override — for unit testing (CRITICAL!)
# ──────────────────────────────────────────────────────────────

def test_dependency_override():
    """
    Shows how to swap real dependencies with mocks in tests.
    The endpoint code is UNCHANGED; we only override the dependency.
    """

    # Mock that bypasses real DB
    def mock_get_db():
        class FakeDB:
            def query(self, model):
                return [{"id": 99, "model": model, "db": "MOCK"}]
            def close(self):
                pass
        db = FakeDB()
        try:
            yield db
        finally:
            db.close()

    # Override: swap get_db → mock_get_db for the duration of this test
    app.dependency_overrides[get_db] = mock_get_db

    client = TestClient(app)
    # This call would normally fail without a real DB
    # but succeeds with the mock
    # response = client.get("/users", headers={"x-api-key": "secret-key-123"})

    # Always clean up overrides after tests
    app.dependency_overrides.clear()

    print("Dependency override pattern demonstrated.")


if __name__ == "__main__":
    test_dependency_override()
    uvicorn.run("02_dependency_injection:app", host="0.0.0.0", port=8002, reload=True)
```
