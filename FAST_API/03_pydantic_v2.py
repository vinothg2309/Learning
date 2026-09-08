# 03_pydantic_v2.py

```python
"""
========================================================
TOPIC 3: Pydantic V2 Integration
========================================================

KEY CONCEPTS:
- Field(...)         → ... is Python Ellipsis = field is REQUIRED (no default).
                       Lets you add constraints AND mark required together.
                       Field(0.7) or Field(default=0.7) = optional with default.

- field_validator    → Single-field rule. Only that field's value is available.
                       @classmethod required — instance doesn't exist yet.
                       Execution: raw input → field_validator → model_validator.

- model_validator    → Cross-field rule. Needs to compare 2+ fields together.
                       mode="before" → receives raw dict, before field parsing.
                       mode="after"  → receives fully validated model instance (self).

- @classmethod       → Used when method needs the CLASS (cls), not an instance.
                       Pydantic validators use it because no instance exists yet.
                       @staticmethod = needs neither cls nor self (pure utility).

- response_model     → FastAPI filters output to only what the schema exposes.
                       Prevents accidental data leaks (e.g., hashed_password).

- BaseSettings       → Type-safe config from env vars / .env files.
                       The standard pattern for 12-factor app config.

INTERVIEW ANSWER TIP:
  "Pydantic V2 uses Rust-based validation (10x faster than V1).
   I use field_validator for single-field logic, model_validator for
   cross-field checks, and response_model to ensure the API never leaks
   internal fields. BaseSettings gives me type-safe, env-driven config."
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import (
    AliasChoices,
    BaseModel,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict  # pip install pydantic-settings

app = FastAPI(title="Pydantic V2 Demo")


# ──────────────────────────────────────────────────────────────
# 3A. Field metadata — constraints + OpenAPI docs
# ──────────────────────────────────────────────────────────────
class ModelInferenceRequest(BaseModel):
    """Request body for an LLM inference endpoint."""

    prompt: str = Field(
        ...,                          # required (no default)
        min_length=1,
        max_length=4096,
        description="The input prompt for the LLM",
        examples=["Summarize this document:"],
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,          # greater-than-or-equal
        le=2.0,          # less-than-or-equal
        description="Sampling temperature. 0=deterministic, 2=creative",
    )
    max_tokens: int = Field(default=256, gt=0, le=4096)

    # AliasChoices: accept both 'model_name' and 'model' in JSON
    model_name: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("model_name", "model"),
    )


# ──────────────────────────────────────────────────────────────
# 3B. field_validator — single-field validation
# ──────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username: str
    email: EmailStr  # built-in email validation
    password: str
    role: str = "viewer"

    @field_validator("username")
    @classmethod  # required — no instance exists yet during validation
    def username_alphanumeric(cls, v: str) -> str:
        """
        Only 'v' (this field's value) is available — cannot see other fields.
        Return the (possibly modified) value or raise ValueError.
        """
        v = v.strip().lower()
        if not v.isalnum():
            raise ValueError("username must be alphanumeric")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("password must contain an uppercase letter")
        return v  # store as plaintext here; hash it in the endpoint


# ──────────────────────────────────────────────────────────────
# 3C. model_validator — cross-field validation
# ──────────────────────────────────────────────────────────────
class DateRangeRequest(BaseModel):
    start_date: str   # "YYYY-MM-DD"
    end_date: str

    @model_validator(mode="after")
    def end_after_start(self) -> "DateRangeRequest":
        """
        mode="after"  → self has ALL fields. Use to compare 2+ fields.
        mode="before" → cls receives raw dict. Use to rename/transform keys.
        """
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class FineTuneConfig(BaseModel):
    use_lora: bool = False
    lora_rank: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def lora_rank_only_when_lora(self) -> "FineTuneConfig":
        """lora_rank is meaningless when use_lora=False."""
        if not self.use_lora and self.lora_rank != 8:
            raise ValueError("lora_rank has no effect when use_lora=False")
        return self


# ──────────────────────────────────────────────────────────────
# 3D. response_model — control what the API exposes
# ──────────────────────────────────────────────────────────────
class UserInDB(BaseModel):
    """Internal model — has sensitive fields."""
    id: int
    username: str
    email: str
    hashed_password: str       # NEVER expose this
    is_active: bool = True


class UserPublic(BaseModel):
    """Public schema — safe to return to clients."""
    id: int
    username: str
    email: str
    # hashed_password intentionally omitted


# FastAPI serializes the UserInDB but filters it through UserPublic.
# Even if the endpoint returns a UserInDB object, hashed_password
# will NEVER appear in the HTTP response.
@app.post("/users", response_model=UserPublic, status_code=201)
def create_user(user: UserCreate):
    # Simulate DB insert
    db_user = UserInDB(
        id=1,
        username=user.username,
        email=user.email,
        hashed_password="$bcrypt$..." + user.password,
    )
    return db_user  # FastAPI strips hashed_password via response_model


# response_model_exclude_unset=True → omit fields that were not explicitly set.
# Useful for PATCH endpoints where you only want to show changed fields.
@app.get("/model-info", response_model=ModelInferenceRequest, response_model_exclude_unset=True)
def model_info():
    return ModelInferenceRequest(prompt="hello")  # only prompt is set


# ──────────────────────────────────────────────────────────────
# 3E. BaseSettings — type-safe config from environment
# ──────────────────────────────────────────────────────────────
class AppSettings(BaseSettings):
    """
    Reads from environment variables (case-insensitive).
    Falls back to .env file if env var is not set.

    Usage:
        export OPENAI_API_KEY=sk-...
        export DB_URL=postgresql://...
        export DEBUG=false
    """

    model_config = SettingsConfigDict(
        env_file=".env",           # load .env if present
        env_file_encoding="utf-8",
        case_sensitive=False,      # OPENAI_API_KEY == openai_api_key
    )

    openai_api_key: str = Field(..., description="OpenAI API key")
    db_url: str = Field(default="sqlite:///./dev.db")
    debug: bool = Field(default=False)
    max_workers: int = Field(default=4, ge=1, le=32)
    allowed_origins: list[str] = Field(default=["http://localhost:3000"])


# Singleton — instantiated once at startup
# settings = AppSettings()   # uncomment when running with real env vars

# Access via: settings.openai_api_key, settings.db_url, etc.
# FastAPI + Depends pattern for settings injection:
# def get_settings() -> AppSettings:
#     return AppSettings()
# SettingsDep = Annotated[AppSettings, Depends(get_settings)]


# ──────────────────────────────────────────────────────────────
# 3F. Test the validators directly
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test field_validator
    try:
        u = UserCreate(username="vinoth!", email="v@test.com", password="Secret1!")
    except Exception as e:
        print(f"Validation error: {e}")

    u = UserCreate(username="vinoth", email="v@test.com", password="Secret1!")
    print(f"Valid user: {u.username}")

    # Test model_validator
    try:
        d = DateRangeRequest(start_date="2025-01-10", end_date="2025-01-01")
    except Exception as e:
        print(f"Date error: {e}")

    # Test inference request with alias
    req = ModelInferenceRequest(prompt="hello", model="gpt-4o-mini")
    print(f"Model: {req.model_name}")

    uvicorn.run("03_pydantic_v2:app", host="0.0.0.0", port=8003, reload=True)
```
