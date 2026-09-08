# 07_security_auth.py

```python
"""
========================================================
TOPIC 7: Security & Authentication
========================================================

KEY CONCEPTS:
- JWT (JSON Web Token): Stateless auth. Server signs a token; client
  sends it with every request. Server validates signature — no DB lookup.
  Parts: header.payload.signature (Base64 encoded).

- OAuth2 with Scopes: Standard protocol. Client requests specific scopes
  (permissions). Token carries the granted scopes. Endpoints verify scope.

- HTTPBearer: FastAPI security scheme that reads 'Authorization: Bearer <token>'.

- API Key: Simpler alternative to JWT. Good for service-to-service calls.

INTERVIEW ANSWER TIP:
  "I implement JWT auth with HTTPBearer. The token contains user_id and
   scopes. Each endpoint declares required scopes; the dependency extracts
   the token, verifies the signature with the secret key, checks expiry,
   and validates the required scope is present. No database hit per request
   — that's the stateless advantage of JWT."
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt  # pip install PyJWT
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    SecurityScopes,
)
from pydantic import BaseModel

app = FastAPI(title="Security & Auth Demo")

# ── Config (use BaseSettings in production) ───────────────────
SECRET_KEY = "super-secret-key-change-in-production"  # 32+ random bytes in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Fake user database
FAKE_USERS_DB = {
    "vinoth": {
        "username": "vinoth",
        "hashed_password": "fakehashedsecret",  # bcrypt in prod
        "scopes": ["read:models", "write:models", "admin"],
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": "fakehashedanalyst",
        "scopes": ["read:models"],   # read-only user
    },
}


# ──────────────────────────────────────────────────────────────
# 7A. JWT Token creation & validation
# ──────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Sign a JWT with the user's data + expiry.
    'sub' (subject) is the standard claim for user identifier.
    'scopes' is a custom claim for permissions.
    'exp' (expiry) is a standard claim; PyJWT validates it automatically.
    """
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    payload["exp"] = expire
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and verify JWT signature + expiry.
    Raises jwt.ExpiredSignatureError if token is expired.
    Raises jwt.InvalidTokenError for any other issue.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ──────────────────────────────────────────────────────────────
# 7B. OAuth2 Password Flow (for user-facing login)
# ──────────────────────────────────────────────────────────────

# This tells FastAPI where the token endpoint is (for Swagger UI)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    scopes={
        "read:models": "Read model information",
        "write:models": "Create and update models",
        "admin": "Full administrative access",
    },
)


@app.post("/auth/token", response_model=Token)
def login_for_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Standard OAuth2 password flow endpoint.
    Client sends: username + password as form data.
    Server returns: JWT access token.

    Test with curl:
        curl -X POST "http://localhost:8007/auth/token" \
             -d "username=vinoth&password=secret"
    """
    user = FAKE_USERS_DB.get(form_data.username)

    # In production: verify bcrypt hash
    # if not user or not bcrypt.checkpw(form_data.password.encode(), user["hashed_password"]):
    if not user or form_data.password != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Only grant scopes the user actually has (don't trust client's scope request blindly)
    granted_scopes = [s for s in form_data.scopes if s in user["scopes"]]

    token = create_access_token(
        data={"sub": user["username"], "scopes": granted_scopes},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=token)


# ──────────────────────────────────────────────────────────────
# 7C. HTTPBearer — extract token from Authorization header
# ──────────────────────────────────────────────────────────────

bearer_scheme = HTTPBearer()


def get_current_user(
    security_scopes: SecurityScopes,          # scopes required by the endpoint
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    Core auth dependency. Does THREE things:
    1. Extracts Bearer token from 'Authorization: Bearer <token>' header.
    2. Decodes and verifies JWT signature + expiry.
    3. Checks that the token has the required scopes for this endpoint.
    """
    payload = decode_token(credentials.credentials)

    username: str = payload.get("sub")
    token_scopes: list[str] = payload.get("scopes", [])

    if not username:
        raise HTTPException(status_code=401, detail="Token missing 'sub' claim")

    # Check each required scope is present in the token
    for required_scope in security_scopes.scopes:
        if required_scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required scope: '{required_scope}'",
                headers={"WWW-Authenticate": f'Bearer scope="{security_scopes.scope_str}"'},
            )

    return {"username": username, "scopes": token_scopes}


# ──────────────────────────────────────────────────────────────
# 7D. Protected endpoints with scope requirements
# ──────────────────────────────────────────────────────────────

# Security() instead of Depends() — enables scope declaration for Swagger UI
@app.get("/models")
def list_models(
    current_user: Annotated[
        dict,
        Security(get_current_user, scopes=["read:models"])  # requires this scope
    ]
):
    """Requires 'read:models' scope. Both vinoth and analyst can access this."""
    return {"models": ["llama-3", "mistral-7b"], "user": current_user["username"]}


@app.post("/models/deploy")
def deploy_model(
    model_name: str,
    current_user: Annotated[
        dict,
        Security(get_current_user, scopes=["write:models"])
    ]
):
    """Requires 'write:models' scope. Only vinoth can access this."""
    return {"deployed": model_name, "by": current_user["username"]}


@app.delete("/models/{model_name}")
def delete_model(
    model_name: str,
    current_user: Annotated[
        dict,
        Security(get_current_user, scopes=["admin"])
    ]
):
    """Requires 'admin' scope. Only vinoth can access this."""
    return {"deleted": model_name, "by": current_user["username"]}


# ──────────────────────────────────────────────────────────────
# 7E. API Key authentication (service-to-service)
# ──────────────────────────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

VALID_API_KEYS = {
    "svc-key-abc123": {"service": "data-pipeline", "permissions": ["read"]},
    "svc-key-xyz789": {"service": "model-trainer", "permissions": ["read", "write"]},
}


def validate_api_key(
    api_key: str = Security(API_KEY_HEADER)
) -> dict:
    """
    Simpler than JWT — good for internal service-to-service calls.
    Store hashed API keys in DB; never store raw keys.
    """
    service = VALID_API_KEYS.get(api_key)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return service


@app.get("/internal/metrics")
def internal_metrics(service: Annotated[dict, Depends(validate_api_key)]):
    """Internal endpoint — uses API key, not JWT."""
    return {
        "caller_service": service["service"],
        "metrics": {"requests_today": 1042, "avg_latency_ms": 234},
    }


if __name__ == "__main__":
    uvicorn.run("07_security_auth:app", host="0.0.0.0", port=8007, reload=True)
```
