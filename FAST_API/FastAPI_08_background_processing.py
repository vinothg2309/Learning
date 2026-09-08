# 08_background_processing.py

```python
"""
========================================================
TOPIC 8: Background Processing
========================================================

KEY CONCEPTS:
- BackgroundTasks → Fire-and-forget tasks AFTER the HTTP response is sent.
                    Runs in the SAME process. Good for small I/O (send email,
                    log to DB). NOT for long CPU/GPU tasks.

- Celery + Redis  → Distributed task queue. Tasks run in separate worker
                    processes (can be on different machines). Scales
                    horizontally. Use for: fine-tuning, batch inference,
                    document processing.

- Task State      → Store task status (PENDING→RUNNING→SUCCESS/FAILURE)
                    in Redis so the client can poll for results.

INTERVIEW ANSWER TIP:
  "BackgroundTasks is for quick I/O after responding — like writing an
   audit log. For long-running ML jobs (fine-tuning, batch inference)
   I use Celery with Redis. The API endpoint enqueues the task, returns
   a task_id immediately, and the client polls /tasks/{id}/status.
   This is the async job pattern — critical for LLM workloads that take
   minutes."
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

app = FastAPI(title="Background Processing Demo")

# In-memory task store (use Redis in production)
task_store: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────
# 8A. BackgroundTasks — simple fire-and-forget
# ──────────────────────────────────────────────────────────────

def send_email_notification(email: str, subject: str, body: str):
    """
    Runs AFTER the HTTP response is sent to the client.
    The client doesn't wait for this — they already got their 200 OK.
    Runs in the same thread pool as sync endpoints.
    """
    print(f"[BG] Sending email to {email}: {subject}")
    time.sleep(2)  # simulate SMTP call
    print(f"[BG] Email sent to {email}")


def log_inference_to_db(user_id: str, prompt: str, response: str):
    """Log ML inference for audit trail / fine-tuning data collection."""
    print(f"[BG] Logging inference for user {user_id}")
    # In production: db.execute("INSERT INTO inference_logs ...")
    time.sleep(0.5)
    print(f"[BG] Inference logged.")


class InferenceRequest(BaseModel):
    prompt: str
    user_id: str
    email: str


@app.post("/infer-with-logging")
async def infer_with_logging(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,   # FastAPI injects this automatically
):
    """
    1. Run inference (instant in this demo).
    2. Return response to client immediately.
    3. THEN run background tasks (email + logging).
    Client never waits for background work.
    """
    # Simulate fast inference
    response_text = f"Response to: {request.prompt}"

    # Schedule background tasks — they run AFTER this function returns
    background_tasks.add_task(
        send_email_notification,
        email=request.email,
        subject="Your inference result",
        body=response_text,
    )
    background_tasks.add_task(
        log_inference_to_db,
        user_id=request.user_id,
        prompt=request.prompt,
        response=response_text,
    )

    # This response is sent BEFORE the background tasks run
    return {"response": response_text, "message": "Background tasks scheduled"}


# ──────────────────────────────────────────────────────────────
# 8B. Async task queue — simulating Celery/Redis pattern
# ──────────────────────────────────────────────────────────────
# In production this would be:
#   from celery import Celery
#   celery_app = Celery("tasks", broker="redis://localhost:6379/0")
#   @celery_app.task
#   def run_fine_tuning(config: dict) -> dict: ...

class FineTuneRequest(BaseModel):
    base_model: str
    dataset_path: str
    epochs: int = 3
    learning_rate: float = 2e-5


class TaskStatus(BaseModel):
    task_id: str
    status: str   # PENDING | RUNNING | SUCCESS | FAILURE
    created_at: str
    completed_at: str | None = None
    result: Any = None
    error: str | None = None


async def simulate_fine_tuning(task_id: str, config: dict):
    """
    Simulates a long-running fine-tuning job.
    In production: this is a Celery task running in a worker process.
    """
    task_store[task_id]["status"] = "RUNNING"
    task_store[task_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        print(f"[TASK {task_id}] Starting fine-tuning: {config}")
        # Simulate 5 epoch steps
        for epoch in range(1, config["epochs"] + 1):
            await asyncio.sleep(2)  # simulate GPU training step
            task_store[task_id]["progress"] = f"Epoch {epoch}/{config['epochs']}"
            task_store[task_id]["loss"] = round(1.0 / epoch, 4)
            print(f"[TASK {task_id}] Epoch {epoch} done. Loss: {task_store[task_id]['loss']}")

        # Task succeeded
        task_store[task_id].update({
            "status": "SUCCESS",
            "completed_at": datetime.utcnow().isoformat(),
            "result": {
                "model_path": f"/models/{config['base_model']}-ft-{task_id[:8]}",
                "final_loss": task_store[task_id]["loss"],
            },
        })
        print(f"[TASK {task_id}] Fine-tuning complete!")

    except Exception as e:
        task_store[task_id].update({
            "status": "FAILURE",
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(e),
        })


@app.post("/fine-tune", status_code=202)   # 202 Accepted = async processing started
async def start_fine_tuning(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks,
):
    """
    ASYNC JOB PATTERN:
    1. Client POSTs fine-tune config.
    2. Server creates a task, returns task_id IMMEDIATELY (202 Accepted).
    3. Fine-tuning runs in background.
    4. Client polls GET /tasks/{task_id} to check progress.

    In production: replace background_tasks with celery_app.send_task(...)
    """
    task_id = str(uuid.uuid4())
    config = request.model_dump()

    # Initialize task record
    task_store[task_id] = {
        "task_id": task_id,
        "status": "PENDING",
        "created_at": datetime.utcnow().isoformat(),
        "config": config,
        "progress": None,
        "loss": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }

    # Schedule the long-running task
    background_tasks.add_task(simulate_fine_tuning, task_id, config)

    return {
        "task_id": task_id,
        "status": "PENDING",
        "message": "Fine-tuning started. Poll /tasks/{task_id} for status.",
        "status_url": f"/tasks/{task_id}",
    }


@app.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task_status(task_id: str):
    """
    Polling endpoint. Client calls this every N seconds to check progress.
    Returns current status: PENDING → RUNNING → SUCCESS/FAILURE.

    In production with Celery:
        result = celery_app.AsyncResult(task_id)
        return {"status": result.state, "result": result.result}
    """
    task = task_store.get(task_id)
    if not task:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return TaskStatus(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
        result=task.get("result"),
        error=task.get("error"),
    )


@app.get("/tasks/{task_id}/progress")
def get_task_progress(task_id: str):
    """Detailed progress for running tasks."""
    task = task_store.get(task_id)
    if not task:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress"),
        "current_loss": task.get("loss"),
    }


@app.get("/tasks")
def list_tasks():
    """List all tasks (admin view)."""
    return [
        {"task_id": k, "status": v["status"], "created_at": v["created_at"]}
        for k, v in task_store.items()
    ]


# ──────────────────────────────────────────────────────────────
# 8C. Celery integration sketch (commented — requires Redis)
# ──────────────────────────────────────────────────────────────
"""
# celery_app.py
from celery import Celery

celery_app = Celery(
    "ml_tasks",
    broker="redis://localhost:6379/0",      # task queue
    backend="redis://localhost:6379/1",     # result store
)

@celery_app.task(bind=True)
def run_fine_tuning_celery(self, config: dict) -> dict:
    # self.update_state(state="PROGRESS", meta={"epoch": 1})
    # ... training loop ...
    return {"model_path": "/models/...", "loss": 0.23}


# In FastAPI endpoint:
@app.post("/fine-tune-celery")
def start_fine_tune_celery(request: FineTuneRequest):
    task = run_fine_tuning_celery.delay(request.model_dump())
    return {"celery_task_id": task.id}

@app.get("/celery-tasks/{task_id}")
def celery_task_status(task_id: str):
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)
    return {"state": result.state, "result": result.result}

# Run Celery worker:
#   celery -A celery_app worker --loglevel=info --concurrency=2
"""


if __name__ == "__main__":
    uvicorn.run("08_background_processing:app", host="0.0.0.0", port=8008, reload=True)
```
