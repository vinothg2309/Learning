# A2A Protocol (Agent2Agent)
## Senior Engineer Interview Preparation — 20 Conceptual + 10 Scenario Q&A

---

## 📚 Part 1 — Conceptual Questions (20)

---

### Q1. What is the A2A protocol and why was it created?

**Answer:**
A2A (Agent2Agent) is an open, vendor-neutral communication protocol introduced by Google (April 2025) to standardize how AI agents discover, communicate, and collaborate with each other across organizational and technology boundaries.

**Why it was created:**
- AI ecosystems were fragmenting — every framework (LangChain, CrewAI, AutoGen, Vertex AI) built proprietary agent-to-agent wiring
- No standard existed for an agent built on GPT-4 to safely call an agent built on Claude or Gemini
- Enterprises needed multi-vendor multi-agent pipelines without tight coupling
- It fills the gap that MCP (Model Context Protocol) intentionally left: MCP handles tool/resource exposure from a server to a model; A2A handles agent-to-agent task delegation

**Analogy:** A2A is HTTP for agents — a transport + messaging contract that lets agents interoperate regardless of the model or framework underneath.

---

### Q2. What is an Agent Card and what does it contain?

**Answer:**
An Agent Card is a JSON document that an A2A server exposes at the well-known URL:
```
GET /.well-known/agent.json
```
It is the agent's self-description — a machine-readable "business card" that a client agent reads before initiating communication.

**Key fields:**
| Field | Description |
|---|---|
| `name` | Human-readable agent name |
| `description` | What the agent does |
| `url` | Base endpoint for sending tasks |
| `version` | Agent version string |
| `protocol_version` | A2A spec version the agent implements |
| `capabilities` | Flags like `streaming`, `pushNotifications` |
| `skills` | Array of skill objects (id, name, description, tags, examples) |
| `default_input_modes` | Accepted content types (`text`, `file`, `data`) |
| `default_output_modes` | Output content types |

**Why it matters:** Discovery is automatic — clients don't need out-of-band configuration. An orchestrator can scan Agent Cards across a registry and route tasks dynamically.

---

### Q3. Explain the A2A message structure — Message, Role, and Parts.

**Answer:**
Every A2A payload is a **Message** object composed of:

```
Message
├── role        → "user" | "agent"
├── message_id  → UUID
└── parts[]     → one or more Part objects
    ├── TextPart   { type: "text",  text: "..." }
    ├── FilePart   { type: "file",  file: { mimeType, data | uri } }
    └── DataPart   { type: "data",  data: { ...arbitrary JSON } }
```

- **Role** distinguishes who is speaking — the initiating user-side agent (`user`) or the responding agent (`agent`)
- **Parts** enable multimodal payloads: a single message can combine text instructions, a file attachment, and structured JSON data
- The envelope is transported via **JSON-RPC 2.0** — giving it method dispatch, request IDs, and a standard error contract

---

#### CLIENT — Sending Messages

**Send text:**
```python
from a2a.utils import create_text_message_object
from a2a.utils import get_message_text

message = create_text_message_object("Find Psychiatrists in Austin, TX")
# role=user is set automatically
```

**Send file (PDF / image):**
```python
from a2a.types import Message, Part, FilePart, FileWithBytes, Role
import uuid

message = Message(
    message_id=str(uuid.uuid4()),
    role=Role.user,
    parts=[Part(root=FilePart(file=FileWithBytes(
        name="report.pdf", bytes=pdf_bytes, mime_type="application/pdf"
    )))]
)
```

**Send structured JSON (DataPart):**
```python
from a2a.types import Message, Part, DataPart, Role
import uuid

message = Message(
    message_id=str(uuid.uuid4()),
    role=Role.user,
    parts=[Part(root=DataPart(data={"policy_id": "POL-123", "state": "TX"}))]
)
```

**Read agent response:**
```python
response = await client.send_message(message)   # returns Message or Task
text = get_message_text(response)               # SDK helper — extracts plain str
```

---

#### SERVER (AgentExecutor) — Reading & Replying

**Read user input (text shortcut):**
```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    prompt = context.get_user_input()   # extracts plain str from TextPart
```

**Read any part type manually:**
```python
from a2a.types import TextPart, FilePart, DataPart

for part in context.message.parts:
    if isinstance(part.root, TextPart):   text = part.root.text
    elif isinstance(part.root, FilePart): file = part.root.file
    elif isinstance(part.root, DataPart): data = part.root.data
```

**Reply with text:**
```python
from a2a.utils import new_agent_text_message

await event_queue.enqueue_event(new_agent_text_message("Here are the doctors ..."))
# role=agent is set automatically
```

**Reply with file:**
```python
from a2a.types import Message, Part, FilePart, FileWithBytes, Role
import uuid

reply = Message(
    message_id=str(uuid.uuid4()),
    role=Role.agent,
    parts=[Part(root=FilePart(file=FileWithBytes(
        name="result.pdf", bytes=pdf_bytes, mime_type="application/pdf"
    )))]
)
await event_queue.enqueue_event(reply)
```

**Reply with structured JSON:**
```python
from a2a.types import Message, Part, DataPart, Role
import uuid

reply = Message(
    message_id=str(uuid.uuid4()),
    role=Role.agent,
    parts=[Part(root=DataPart(data={"providers": [...], "count": 5}))]
)
await event_queue.enqueue_event(reply)
```

---

**End-to-end flow:**
```
[Client]  create_text_message_object(prompt)  → Message(role=user,  parts=[TextPart])
[Server]  context.get_user_input()            → plain str
[Agent]   agent.ainvoke({"messages": [...]})  → plain str
[Server]  new_agent_text_message(result)      → Message(role=agent, parts=[TextPart])
[Client]  get_message_text(response)          → plain str
```

---

### Q4. What is a Task in A2A and what are its lifecycle states?

**Answer:**
A **Task** is the unit of work in A2A — but it is **not always created**.

> **Is a Task created for every request?**
>
> | Mode | Task created? | Server returns |
> |---|---|---|
> | **Synchronous** (fast) | ❌ No | `Message` directly |
> | **Asynchronous** (long-running) | ✅ Yes | `Task` with ID |
> | **SSE Streaming** | ✅ Yes (internally) | Event stream |
> | **Push Notification** | ✅ Yes | Callback with Task |
>
> For synchronous mode, if the agent completes immediately, the server returns a `Message` directly — no Task ID, no polling needed.
> A Task is created only when processing is **deferred** (needs time) or the client uses streaming/push mode.

> **Does `AgentCapabilities(streaming=True)` affect task lifecycle?**
> No. It only advertises SSE support (`message/stream`). Task states work identically in sync, async, and streaming modes.

**Lifecycle states:**
```
submitted → working → completed
                   ↘ failed
                   ↘ canceled
                   ↘ input-required   (agent needs clarification from user)
```

| State | Meaning |
|---|---|
| `submitted` | Task received, queued |
| `working` | Agent is actively processing |
| `input-required` | Agent paused, awaiting user input (human-in-the-loop) |
| `completed` | Task done, artifacts available |
| `failed` | Unrecoverable error |
| `canceled` | Canceled by client request |

---

**SERVER — signal state transitions via EventQueue:**

```python
# working → input-required  (pause, ask user a question)
await event_queue.enqueue_event(new_agent_text_message("Which city?"))
await event_queue.enqueue_event(
    TaskStatusUpdateEvent(status=TaskStatus(state=TaskState.input_required))
)

# working → completed  (SDK sets this automatically after execute() returns)
await event_queue.enqueue_event(new_agent_text_message("Here are the results."))

# working → failed  (unrecoverable error)
await event_queue.enqueue_event(
    TaskStatusUpdateEvent(status=TaskStatus(state=TaskState.failed))
)
```

**CLIENT — poll task status:**

> `get_task()` is a single HTTP call — returns current snapshot.
> To poll, call it in a loop until the task reaches a terminal state.

```python
import asyncio

# Poll every 2 seconds until done
while True:
    task = await client.get_task(task_id)

    if task.status.state in ("completed", "failed", "canceled"):
        break                               # terminal state — stop polling

    if task.status.state == "input_required":
        # Agent is waiting for user input — send a follow-up message
        break

    await asyncio.sleep(2)                  # wait before next check

# Read result when completed
if task.status.state == "completed":
    text = get_message_text(task.artifacts[0])
```

**CLIENT — cancel a running task:**

```python
await client.cancel_task(task_id)
# Server's cancel() is called → state moves to "canceled"
```

---

### Q5. What are the four A2A communication modes and when do you choose each?

**Answer:**

> **Sync vs Async use the same client call and same server code.**
> The SDK decides what to return based on how fast `execute()` completes.
> The only difference is how the client handles the response.

| Mode | Client call | Response type | Client handles it by | Server `execute()` change |
|---|---|---|---|---|
| **Synchronous** | `send_message(msg)` | `Message` | Read text directly — done | None |
| **Asynchronous** | `send_message(msg)` | `Task` (ID only) | Poll `get_task(id)` until terminal state | None — SDK auto-creates Task when slow |
| **SSE Streaming** | `send_message_streaming(msg)` | Event stream | `async for event in stream:` loop | Push chunks: `async for chunk in llm.astream():` |
| **Push Notification** | `send_message(msg, push_notification_config=...)` | `Task` (immediately) | Webhook receives POST callback | None — `DefaultRequestHandler` calls back automatically |

**Decision rule:**
- Latency-sensitive → Synchronous
- Long-running, can poll → Asynchronous
- Real-time UX → SSE
- Webhook / serverless client → Push Notification

---

**Mode 1 — Synchronous** (`message/send` → immediate `Message` back)

```python
# Client: send and block until response arrives
from a2a.client import A2AClient
from a2a.utils import create_text_message_object, get_message_text

client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
message = create_text_message_object("Translate: Hello")

response = await client.send_message(message)
# response is a Message (not a Task) — agent completed immediately
print(get_message_text(response))
```

```python
# Server: just push the reply — SDK returns it directly (no Task created)
async def execute(self, context, event_queue):
    result = self.translate(context.get_user_input())
    await event_queue.enqueue_event(new_agent_text_message(result))
```

---

**Mode 2 — Asynchronous** (`message/send` → `Task` ID → client polls)

```python
# Client: send, get Task ID, poll until done
response = await client.send_message(message)
# response is a Task (agent didn't finish immediately)
task_id = response.id

import asyncio
while True:
    task = await client.get_task(task_id)
    if task.status.state in ("completed", "failed", "canceled"):
        break
    await asyncio.sleep(2)          # wait, then check again

print(get_message_text(task.artifacts[0]))
```

```python
# Server: long work — SDK creates Task automatically, execute() runs in background
async def execute(self, context, event_queue):
    result = await self.run_long_analysis(context.get_user_input())  # slow
    await event_queue.enqueue_event(new_agent_text_message(result))
```

---

**Mode 3 — SSE Streaming** (`message/stream` → live event stream)

```python
# Client: open stream, receive events as they arrive (no polling needed)
async with client.send_message_streaming(message) as stream:
    async for event in stream:
        if isinstance(event, Message):
            print(get_message_text(event), end="", flush=True)  # print tokens live
```

```python
# Server: push partial results chunk by chunk
async def execute(self, context, event_queue):
    async for chunk in self.llm.astream(context.get_user_input()):
        await event_queue.enqueue_event(new_agent_text_message(chunk))
```

---

**Mode 4 — Push Notification** (client registers callback URL, disconnects)

```python
# Client: register webhook URL with the task, then disconnect
from a2a.types import PushNotificationConfig

message = create_text_message_object("Run nightly pipeline")
response = await client.send_message(
    message,
    push_notification_config=PushNotificationConfig(
        url="https://my-service.com/webhook/a2a",   # server will POST here when done
    )
)
# Client disconnects — server calls the webhook when task completes

# Webhook handler (receives POST from agent server)
@app.post("/webhook/a2a")
async def handle_callback(event: dict):
    print("Task done:", event["status"]["state"])
    print("Result:", event["artifacts"][0])
```

```python
# Server: no change needed — DefaultRequestHandler handles the callback automatically
# It POSTs TaskStatusUpdateEvent to the registered URL on task completion
```

---

### Q6. How does SSE Streaming work in A2A? What events are streamed?

**Answer:**
SSE (Server-Sent Events) is a standard HTTP/1.1 mechanism where the server sends a continuous `text/event-stream` response over a single persistent connection.

**A2A streaming flow:**
1. Client sends JSON-RPC with method `message/stream`
2. Server keeps the HTTP connection open
3. Server emits discrete events as they occur:
   - **`TaskStatusUpdateEvent`** — state transitions (`working`, `completed`, etc.)
   - **`TaskArtifactUpdateEvent`** — incremental artifact chunks (e.g., LLM tokens)
4. Stream ends with a final event containing `final: true`

**Key difference from polling:**
- Polling = client repeatedly calls `tasks/get` — wasteful, adds latency
- SSE = server pushes the moment state changes — efficient, real-time

**Important:** SSE is server→client only. The client cannot send additional data mid-stream on the same connection.

---

### Q7. How do Push Notifications work in A2A?

**Answer:**
Push Notifications decouple the client from waiting entirely — the client registers a webhook and disconnects.

**Flow:**
1. Client sends task with a `PushNotificationConfig` containing a `url` (callback endpoint) and optional auth headers
2. Server acknowledges and closes the response
3. Agent processes the task asynchronously
4. On completion (or any status change), the server POSTs a `TaskStatusUpdateEvent` or `TaskArtifactUpdateEvent` to the callback URL
5. Client's webhook handler processes the result

**Security considerations:**
- The callback URL should be HTTPS
- Servers should sign payloads (HMAC) so clients can verify authenticity
- Clients should validate the `task_id` matches an outstanding request

**When to use:** Serverless functions (AWS Lambda), mobile clients that go offline, long-running overnight jobs.

---

### Q8. How does A2A differ from MCP (Model Context Protocol)?

**Answer:**

| Dimension | A2A | MCP |
|---|---|---|
| **Purpose** | Agent ↔ Agent task delegation | Model ↔ Tool/Resource exposure |
| **Who talks** | Two autonomous agents | A model client and a tool server |
| **Interaction style** | Long-running tasks, multi-turn | Short-lived tool calls |
| **Discovery** | Agent Card (`/.well-known/agent.json`) | MCP server manifest |
| **Statefulness** | Tasks have full lifecycle & history | Stateless tool calls |
| **Initiated by** | Either agent can initiate | Always model-initiated |
| **Output** | Artifacts (rich, typed results) | Tool return values |

**They are complementary, not competing:**
- MCP exposes tools (search, DB, calculator) to the LLM layer
- A2A delegates entire tasks to specialized agents
- A production system might use both: an orchestrator agent uses A2A to delegate to a specialist agent, which internally uses MCP to call tools

---

### Q9. What is AgentExecutor and what contract must it fulfill?

**Answer:**
`AgentExecutor` is the abstract base class you implement to bridge A2A's request/response lifecycle with your actual agent logic.

**Required interface:**
```python
class AgentExecutor(ABC):
    @abstractmethod
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """Run the agent and push events onto the queue."""

    @abstractmethod
    def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """Handle task cancellation."""
```

**Key responsibilities:**
- Extract user input from `context.get_user_input()`
- Run your business logic (LLM call, tool use, etc.)
- Push results via `await event_queue.enqueue_event(message_or_artifact)`
- Never return a value — all output flows through the event queue
- `execute` must be `async` because `enqueue_event` is a coroutine

---

### Q10. What is the role of RequestContext in A2A?

**Answer:**
`RequestContext` is the read-only view of the inbound request that the `AgentExecutor` receives. It encapsulates:

- **`get_user_input() → str`** — convenience method to extract plain text from the latest user message part
- **`message`** — the full `Message` object with all parts (text, file, data)
- **`task_id`** — the ID of the task being executed
- **`context_id`** — conversation/session identifier for multi-turn interactions
- **`task`** — the full current `Task` object including history

**Why it matters:** RequestContext lets your executor remain stateless — all context needed to process the request is injected, making executors easy to test and horizontally scalable.

---

### Q11. What is an Artifact in A2A and how does it differ from a Message?

**Answer:**

**Message** = a conversational turn (chat). **Artifact** = the final deliverable (file, report, data).

```python
from a2a.utils import new_agent_text_message
from a2a.types import Artifact, Part, TextPart, DataPart

async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    # Message — conversational update (shown to user as chat)
    await event_queue.enqueue_event(
        new_agent_text_message("Searching for cardiologists...")
    )

    doctors = await self.search(context.get_user_input())

    # Artifact — the actual deliverable (structured result)
    await event_queue.enqueue_event(
        Artifact(
            artifact_id="art-001",
            name="doctors.json",
            parts=[Part(root=DataPart(data={"doctors": doctors}))]
        )
    )

    # Message — conversational wrap-up
    await event_queue.enqueue_event(
        new_agent_text_message(f"Found {len(doctors)} cardiologists.")
    )
```

**Client reads them separately:**

```python
task = response.root.result

for msg in task.messages:        # chat turns → display to user
    print(msg.parts[0].root.text)

for art in task.artifacts:       # deliverables → save/process
    data = art.parts[0].root.data["doctors"]
```

| | `Message` | `Artifact` |
|---|---|---|
| **What** | Chat turn | Deliverable |
| **Purpose** | Conversational turn | Deliverable output of a task |
| **Direction** | User→Agent or Agent→User | Agent→Client (result) |
| **Role** | `user` / `agent` | No role — task output |
| **Identity** | `message_id` | `artifact_id` + `index` |
| **Lifecycle** | Exists in message history | Persisted on the Task object |
| **Streaming** | Not streamed | Yes — `TaskArtifactUpdateEvent` |
| **Content** | Parts (text/file/data) | Parts (text/file/data) |
| **Example** | `"Searching..."` | `doctors.json`, `report.pdf` |

---

### Q12. What is InMemoryTaskStore and when should you replace it in production?

**Answer:**
`InMemoryTaskStore` is the default `TaskStore` implementation bundled with the A2A SDK. It stores all task state in a Python dictionary in the process's heap.

**Limitations:**
- State is lost on process restart
- Cannot scale horizontally (each instance has its own in-memory store)
- No TTL / eviction — long-running services leak memory
- No audit trail or observability

**Production replacements:**
| Backend | When |
|---|---|
| Redis | Sub-millisecond lookups, distributed, TTL support |
| PostgreSQL / AlloyDB | Durable, queryable, audit-ready |
| Firestore / DynamoDB | Serverless, managed, global scale |
| Spanner | Multi-region, strong consistency |

---

> **Does TaskStore clear once a task is finished?**
>
> **No** — tasks are **never auto-deleted** after completion.
> They stay in the store until explicitly deleted or evicted.
>
> | Store | What happens after completion |
> |---|---|
> | `InMemoryTaskStore` | Stays in the dict until process restarts or you delete it manually |
> | Redis (with TTL) | Automatically expires after TTL (e.g., 1 hour) — you control this |
> | PostgreSQL | Persists forever — you need a cleanup job or soft-delete |
>
> Completed tasks are kept intentionally so clients can:
> - Re-fetch artifacts later (`tasks/get`)
> - Audit what happened
> - Resume or reference a finished task
>
> `InMemoryTaskStore` has no TTL/eviction — this is a memory leak risk in long-running servers.

---

**Redis TaskStore — implementation:**

```python
import json
import redis.asyncio as redis
from a2a.server.tasks import TaskStore
from a2a.types import Task

class RedisTaskStore(TaskStore):
    """Swap-in replacement for InMemoryTaskStore backed by Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl          # seconds — task auto-expires after this (e.g. 1 hour)

    async def save(self, task: Task) -> None:
        # Serialize Task to JSON, store with TTL
        await self.redis.setex(
            name=f"a2a:task:{task.id}",
            time=self.ttl,
            value=task.model_dump_json()    # Pydantic v2 serialization
        )

    async def get(self, task_id: str) -> Task | None:
        data = await self.redis.get(f"a2a:task:{task_id}")
        if data is None:
            return None
        return Task.model_validate_json(data)

    async def delete(self, task_id: str) -> None:
        await self.redis.delete(f"a2a:task:{task_id}")
```

**Wire it into your server (replace `InMemoryTaskStore`):**

```python
# Before (dev):
request_handler = DefaultRequestHandler(
    agent_executor=ProviderAgentExecutor(),
    task_store=InMemoryTaskStore(),             # lost on restart, no TTL
)

# After (production):
request_handler = DefaultRequestHandler(
    agent_executor=ProviderAgentExecutor(),
    task_store=RedisTaskStore(
        redis_url="redis://redis-host:6379",
        ttl=3600                               # completed tasks expire after 1 hour
    ),
)
```

---

### Q13. How does A2A handle multi-turn conversations and human-in-the-loop?

**Answer:**
A2A supports multi-turn via the `input-required` task state:

1. Agent sets task state to `input-required` and emits a Message asking for clarification
2. Client receives the event (via SSE or polling)
3. Client sends a follow-up message referencing the same `task_id` and `context_id`
4. Agent resumes from the same task context, sees full history via `RequestContext.task.history`
5. Cycle repeats until the agent reaches `completed`

**Key design:** The `context_id` ties the conversation session together across multiple task turns, while `task_id` tracks the unit of work. This separation allows one session to spawn multiple tasks.

**Human-in-the-loop pattern:**
```
Agent detects ambiguity
  → sets status: input-required
  → emits clarification message
  → human/client responds
  → agent resumes
  → sets status: working → completed
```

---

**SERVER — single `execute()` handles both turns:**

> `TaskState.input_required` → string value `"input-required"` in A2A spec
>
> **Why not check `TaskState.submitted`?**
> By the time `execute()` is called, the SDK has already transitioned state to `working`.
> Use `task.history` length to distinguish first turn vs follow-up turn.

```python
from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    task = context.task

    # Turn 1: no prior history → first message, need more info
    if len(task.history or []) <= 1:
        await event_queue.enqueue_event(
            new_agent_text_message("Which city are you looking for providers in?")
        )
        # Signal: pause and wait for user — state becomes "input-required"
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(state=TaskState.input_required)
                #                             ↑ string value: "input-required"
            )
        )
        return      # stop here — execute() will be called again when user replies

    # Turn 2: state is "input-required", user sent a follow-up
    if task.status.state == TaskState.input_required:
        city = context.get_user_input()     # reads user's reply e.g. "Boston, MA"
        result = await self.find_providers(city)
        await event_queue.enqueue_event(new_agent_text_message(result))
        # No explicit state update needed — SDK sets "completed" when execute() returns
```

**CLIENT — Turn 1: send initial message, detect "input-required" pause:**

```python
message = create_text_message_object("Find me a psychiatrist")
response = await client.send_message(message)

# Agent paused → response is a Task with state "input-required"
print(response.status.state)        # "input-required"
task_id    = response.id
context_id = response.context_id    # ties all turns of this conversation together

# Read the agent's question from history
print(get_message_text(response.history[-1]))   # "Which city?"
```

**CLIENT — Turn 2: reply with the answer using same task + context IDs:**

```python
followup = create_text_message_object("Boston, MA")
response2 = await client.send_message(
    followup,
    task_id=task_id,        # resume the same task (agent sees history)
    context_id=context_id   # same conversation session
)

print(response2.status.state)       # "completed"
print(get_message_text(response2))  # final provider list
```

> `task_id` — resumes the specific unit of work
> `context_id` — groups all turns of the same conversation together

---

### Q14. How does A2A handle authentication and security?

**Answer:**
A2A intentionally delegates auth to standard HTTP/transport-layer mechanisms rather than inventing a new auth protocol.

**Supported patterns declared in Agent Card:**
```json
"authentication": {
  "schemes": ["Bearer", "ApiKey", "OAuth2"]
}
```

**3 auth patterns — pick one based on trust level:**

| Pattern | Use when | Header sent |
|---|---|---|
| API Key | Simple service-to-service | `x-api-key: <secret>` |
| Bearer (JWT) | Enterprise, IdP-backed | `Authorization: Bearer <token>` |
| mTLS | Highest trust, both sides cert | TLS handshake |

---

**SERVER — declare auth scheme in AgentCard + validate incoming requests:**

```python
from a2a.types import AgentCard, SecurityScheme, APIKeySecurityScheme

# 1. Declare what auth the server requires (published in agent-card.json)
agent_card = AgentCard(
    ...
    security_schemes={
        "api_key": APIKeySecurityScheme(in_="header", name="x-api-key")
    },
    security=[{"api_key": []}],     # enforce this scheme globally
)

# 2. Validate incoming request (middleware — runs before execute())
from a2a.server.request_handlers import ServerCallContextBuilder
from starlette.requests import Request

class ApiKeyAuthBuilder(ServerCallContextBuilder):
    def build(self, request: Request):
        token = request.headers.get("x-api-key")
        if token == "my-secret-key":
            return ServerCallContext(user=SimpleUser("trusted-client"))
        return ServerCallContext(user=UnauthenticatedUser())  # rejected
```

**CLIENT — store credentials + auto-inject header on every call:**

```python
from a2a.client.auth import InMemoryContextCredentialsStore, AuthInterceptor

# 1. Store the API key (keyed by session + scheme name)
cred_store = InMemoryContextCredentialsStore()
await cred_store.set_credentials(
    session_id="session-1",
    security_scheme_name="api_key",     # must match scheme name in AgentCard
    credential="my-secret-key"
)

# 2. Connect with auth interceptor — injects x-api-key header automatically
client = await ClientFactory.connect(
    "http://localhost:9997",
    interceptors=[AuthInterceptor(cred_store)]
)

# 3. Every send_message() automatically includes x-api-key header
response = await client.send_message(
    message,
    context=ClientCallContext(state={"sessionId": "session-1"})
)
```

**Enterprise recommendation:** Treat every inter-agent call as a zero-trust boundary — validate identity, enforce least-privilege scopes, and log all calls.

---

### Q15. How does A2A agent discovery work at scale?

**Answer:**
Discovery = how a client agent finds and connects to a server agent it has never seen before.

**3 levels of discovery:**

| Level | How | When |
|---|---|---|
| **Direct** | Know the URL upfront, fetch agent card | Dev/local — you own both agents |
| **Registry** | Central list of agent cards, query by tag/skill | Team or org with many agents |
| **Semantic (LLM)** | Embed agent descriptions, vector search by task | Marketplace / dynamic routing |

---

**Level 1 — Direct: fetch agent card from known URL:**

```python
import httpx
from a2a.client import A2ACardResolver, A2AClient

# Client reads agent-card.json from the well-known URL
# Server auto-exposes this at GET /.well-known/agent.json
async with httpx.AsyncClient() as http:
    resolver = A2ACardResolver(http)
    agent_card = await resolver.get_agent_card("http://localhost:9997")
    # agent_card now has: name, skills, url, capabilities

    client = A2AClient(httpx_client=http, agent_card=agent_card)
    response = await client.send_message(message)
```

**Level 2 — Registry: agents self-register on startup, client queries by skill:**

```python
# Agent server: register itself in the registry on startup
import httpx

async def register_on_startup(agent_card: AgentCard, registry_url: str):
    async with httpx.AsyncClient() as http:
        await http.post(f"{registry_url}/register", json={
            "agent_url": agent_card.url,
            "skills":    [s.id for s in agent_card.skills],
            "tags":      ["healthcare", "providers"],
        })

# Client: query registry by skill tag, then connect
async def find_and_call(skill_tag: str, prompt: str):
    async with httpx.AsyncClient() as http:
        # 1. Ask registry which agents have this skill
        results = await http.get(f"{registry_url}/search?tag={skill_tag}")
        agent_url = results.json()[0]["agent_url"]   # pick first match

        # 2. Fetch agent card + connect
        resolver = A2ACardResolver(http)
        agent_card = await resolver.get_agent_card(agent_url)
        client = A2AClient(httpx_client=http, agent_card=agent_card)

        return await client.send_message(create_text_message_object(prompt))
```

> `/.well-known/agent.json` follows the same RFC 8615 standard as OAuth2's
> `/.well-known/openid-configuration` — clients know exactly where to look.

---

### Q16. What is the difference between `message/send` and `message/stream` in the A2A JSON-RPC API?

**Answer:**

| Method | Response type | Use case |
|---|---|---|
| `message/send` | Single JSON-RPC response (sync or Task ID for async) | Short tasks or polling-based async |
| `message/stream` | `text/event-stream` of events | Real-time streaming |
| `tasks/get` | Current Task snapshot | Polling task status |
| `tasks/cancel` | Confirmation | Aborting a running task |
| `tasks/pushNotificationConfig/set` | Confirmation | Register webhook |
| `tasks/pushNotificationConfig/get` | Current config | Inspect webhook setup |

**`message/send` behavior depends on task duration:**
- If the agent completes synchronously → returns a `Message` directly
- If the task needs time → returns a `Task` object with `status: submitted`

---

### Q17. How does A2A enable multi-agent orchestration patterns?

**Answer:**
A2A enables three fundamental orchestration topologies:

**1. Hierarchical (Orchestrator → Specialist)**
```
Orchestrator Agent
├── calls Research Agent  (via A2A)
├── calls Summarizer Agent (via A2A)
└── calls Writer Agent     (via A2A)
```
Each specialist is a fully autonomous A2A server. The orchestrator decomposes the goal and fans out.

**2. Pipeline (Sequential)**
```
Ingestion Agent → Enrichment Agent → Analysis Agent → Report Agent
```
Each agent's artifact becomes the next agent's input message. Loose coupling — each step can be replaced independently.

**3. Peer-to-Peer (Collaborative)**
```
Agent A ↔ Agent B
```
Either agent can initiate a task on the other. Useful for negotiation, verification, or co-authoring patterns.

**Key advantage over monolithic agents:** Each agent can be built with different models, frameworks, and languages. The A2A protocol is the only shared contract.

---

### Q18. What are the tradeoffs between A2A and a direct REST API for agent communication?

**Answer:**

| Dimension | A2A | Plain REST |
|---|---|---|
| **Discovery** | Agent Card auto-discovery | Manual documentation |
| **Task lifecycle** | Built-in (submit/poll/cancel) | Must build from scratch |
| **Streaming** | Native SSE + push notifications | Custom SSE or WebSocket |
| **Multi-turn** | `context_id` + `input-required` state | Session management DIY |
| **Interoperability** | Any A2A-compliant client works | Client must know API schema |
| **Overhead** | JSON-RPC envelope | Lighter, direct |
| **Versioning** | `protocol_version` in Agent Card | API versioning in URL |

**When to prefer REST:**
- Internal microservices where you control both sides
- Extremely latency-sensitive paths (<10ms)
- Simple request-reply with no async need

**When A2A wins:**
- Cross-team or cross-vendor agents
- Long-running tasks requiring lifecycle management
- Marketplace / registry of agents you don't control

---

### Q19. How does the EventQueue pattern work, and why is it used instead of direct returns?

**Answer:**
The `EventQueue` is an async queue that decouples agent execution from response delivery. Instead of returning a value from `execute()`, the agent pushes events:

```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    # Stream partial results
    for chunk in llm.stream(prompt):
        msg = new_agent_text_message(chunk)
        await event_queue.enqueue_event(msg)
```

**Why this design:**
- **Streaming first:** The queue abstraction makes streaming and non-streaming responses identical from the executor's perspective
- **Backpressure:** The queue can apply backpressure if the client is slow
- **Decoupled delivery:** The same executor code works for SSE, polling, and push notifications — the SDK wires the queue to the appropriate transport
- **Testability:** In tests, you can drain the queue and assert events without running a real HTTP server

**Event types you can enqueue:**
- `Message` — conversational reply
- `Artifact` — deliverable output
- `TaskStatusUpdateEvent` — state change signal

---

### Q20. How does A2A integrate with LangGraph agents?

**Answer:**
A LangGraph agent becomes an A2A server by wrapping it in an `AgentExecutor`:

```python
class LangGraphAgentExecutor(AgentExecutor):
    def __init__(self):
        self.graph = build_langgraph_graph()   # your compiled StateGraph

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        # Stream LangGraph node outputs as A2A events
        async for event in self.graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="updates"
        ):
            for node, state in event.items():
                if "messages" in state:
                    last_msg = state["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        a2a_msg = new_agent_text_message(last_msg.content)
                        await event_queue.enqueue_event(a2a_msg)
```

**What this achieves:**
- The LangGraph graph handles all internal reasoning (tools, memory, conditional routing)
- A2A handles all external communication (discovery, task lifecycle, streaming delivery)
- The graph is fully unaware of A2A — clean separation of concerns
- Any A2A client can now call your LangGraph agent without knowing it's LangGraph
---
### Q21. What is `new_agent_text_message()` and when do you use it vs building a `Message` manually?

**Answer:**

`new_agent_text_message()` is a utility from `a2a.utils` that constructs a valid A2A `Message` object with `role=agent`, a single `TextPart`, and an auto-generated `message_id` — in one line.

**What it builds internally:**

```python
# What new_agent_text_message("Hello") returns:
Message(
    role       = Role.agent,
    message_id = str(uuid.uuid4()),   # auto-generated
    parts      = [Part(root=TextPart(text="Hello"))],
    task_id    = None,
    context_id = None,
)
```

**Usage in an executor — the standard pattern:**

```python
from a2a.utils import new_agent_text_message

async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    response = await self.agent.ainvoke(
        {"messages": [{"role": "user", "content": context.get_user_input()}]}
    )
    text = response["messages"][-1].content

    # One-liner — no need to import Message, Part, TextPart, Role, uuid
    await event_queue.enqueue_event(new_agent_text_message(text))
```

**When to build `Message` manually instead:**

Use `new_agent_parts_message()` or construct `Message` directly when the response has multiple part types:

```python
from a2a.utils import new_agent_parts_message
from a2a.types import Part, TextPart, DataPart

# Multiple parts (text + structured data) in one message
await event_queue.enqueue_event(
    new_agent_parts_message(parts=[
        Part(root=TextPart(text=summary)),
        Part(root=DataPart(data={"providers": providers_list})),
    ])
)
```

**Sibling utilities from `a2a.utils`:**

| Function | Returns | Use When |
|---|---|---|
| `new_agent_text_message(text)` | `Message(role=agent, parts=[TextPart])` | Single plain-text reply |
| `new_agent_parts_message(parts)` | `Message(role=agent, parts=[...])` | Mixed parts (text + data + file) |
| `get_message_text(message)` | `str` | Extract text from an incoming user `Message` |

---

### Q22. Is there any possibility to invoke parallelism at the A2A client to reduce latency for multi-session, high-traffic scenarios?

**Answer:**

Yes — the A2A client supports full async parallelism. The key insight is that each `send_message` / `send_message_streaming` call is an independent async coroutine, so you can fan out across multiple agents or multiple sessions concurrently with `asyncio.gather`.

---

**Pattern 1 — Fan-out across multiple A2A agents (parallel specialists)**

Send the same task to `PolicyAgent` and `ProviderAgent` in parallel, merge results:

```python
import asyncio
from a2a.client import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams
from a2a.utils import new_user_text_message

async def parallel_agents(query: str) -> dict:
    async with httpx.AsyncClient() as http:
        policy_client   = await A2AClient.get_client_from_agent_card_url(http, "http://policy-agent:8001")
        provider_client = await A2AClient.get_client_from_agent_card_url(http, "http://provider-agent:8002")

        msg = new_user_text_message(query)
        req = SendMessageRequest(id="1", params=MessageSendParams(message=msg))

        # Both calls run concurrently — total latency = max(t_policy, t_provider)
        policy_resp, provider_resp = await asyncio.gather(
            policy_client.send_message(req),
            provider_client.send_message(req),
        )

    return {
        "policy":   policy_resp.root.result,
        "provider": provider_resp.root.result,
    }
```

---

**Pattern 2 — Fan-out across multiple user sessions (multi-session batch)**

Process N independent user sessions concurrently — each has its own `context_id`:

```python
async def handle_batch(sessions: list[dict]) -> list:
    async with httpx.AsyncClient() as http:
        client = await A2AClient.get_client_from_agent_card_url(http, "http://provider-agent:8002")

        async def one_session(session: dict):
            msg = new_user_text_message(session["query"])
            req = SendMessageRequest(
                id=session["session_id"],
                params=MessageSendParams(
                    message=msg,
                    configuration={"context_id": session["context_id"]},  # isolate per user
                ),
            )
            return await client.send_message(req)

        # All N sessions fly in parallel — bounded by semaphore to avoid overwhelming server
        sem = asyncio.Semaphore(10)  # max 10 concurrent
        async def bounded(s):
            async with sem:
                return await one_session(s)

        return await asyncio.gather(*[bounded(s) for s in sessions])
```

---

**Pattern 3 — Streaming parallel sessions**

For long-running tasks, use `send_message_streaming` so each session streams independently:

```python
async def stream_session(client, session_id: str, query: str):
    req = SendMessageRequest(id=session_id, params=MessageSendParams(message=new_user_text_message(query)))
    async for event in client.send_message_streaming(req):
        yield session_id, event  # yield (session_id, chunk) for multiplexing

# Run multiple streams concurrently with asyncio.TaskGroup (Python 3.11+)
async def multi_stream(sessions: list[dict]):
    async with asyncio.TaskGroup() as tg:
        for s in sessions:
            tg.create_task(stream_session(client, s["id"], s["query"]))
```

---

**Latency comparison:**

| Approach | Latency | Use Case |
|---|---|---|
| Sequential `for` loop | `sum(t1 + t2 + ... + tN)` | Simple scripts only |
| `asyncio.gather` | `max(t1, t2, ..., tN)` | Multi-agent or multi-session fan-out |
| `gather` + `Semaphore(N)` | `max(...)` bounded | High-traffic production — avoid server overload |
| `TaskGroup` + streaming | `max(...)` + streaming | Long-running parallel tasks |

**Best practice for production:**
- Use `asyncio.Semaphore` to cap concurrency (e.g., 10–20 parallel calls)
- Re-use a single `httpx.AsyncClient` across all sessions — connection pooling
- Each session must carry a unique `context_id` to prevent cross-user state bleed in the server's checkpointer

---

### Q23. How do you implement E2E observability from A2A client → server → MCP with a single correlation ID?

**Answer:**

The standard approach is **OpenTelemetry distributed tracing** — one `trace_id` propagated via HTTP headers through every hop. Each service creates child spans under the same root trace, giving a unified waterfall in Grafana Tempo / Jaeger.

---

**Architecture:**

```
A2A Client
  └─ root span: a2a.client.send_message  (trace_id = abc123)
        │   propagated via traceparent header
        ▼
A2A Server (DefaultRequestHandler)
  └─ child span: a2a.server.handle_request
        │   propagated via httpx headers to MCP
        ▼
MCP Server (tool call)
  └─ child span: mcp.tool.list_doctors
```

All three spans share `trace_id = abc123` — one timeline in Grafana Tempo.

---

**Step 1 — Shared OTel setup (all services)**

```python
# observability.py  (same file in client, a2a_server, mcp_server)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import inject, extract

def setup_otel(service_name: str):
    provider = TracerProvider(resource=Resource({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317")))
    trace.set_tracer_provider(provider)

def get_tracer(name: str):
    return trace.get_tracer(name)
```

---

**Step 2 — A2A Client: start root span + inject traceparent header**

```python
from opentelemetry.propagate import inject
from opentelemetry import trace

setup_otel("a2a-client")
tracer = get_tracer("a2a.client")

async def send_with_trace(query: str):
    with tracer.start_as_current_span("a2a.client.send_message") as span:
        span.set_attribute("query.len", len(query))

        # Inject W3C traceparent header so the server continues the same trace
        headers = {}
        inject(headers)   # adds "traceparent": "00-<trace_id>-<span_id>-01"

        async with httpx.AsyncClient() as http:
            # Pass injected headers to the A2A HTTP call
            client = A2AClient(httpx_client=http, agent_card=card)
            resp = await client.send_message(req, headers=headers)

        span.set_attribute("response.status", resp.root.result.status.state)
```

---

**Step 3 — A2A Server: extract traceparent → create child span**

```python
from opentelemetry.propagate import extract
from opentelemetry import trace

setup_otel("a2a-server")
tracer = get_tracer("a2a.server")

class MyAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Extract trace context from incoming HTTP headers
        carrier = dict(context.request.headers)   # headers forwarded by DefaultRequestHandler
        ctx = extract(carrier)                     # restores parent span context

        with tracer.start_as_current_span("a2a.server.handle_request", context=ctx) as span:
            span.set_attribute("task.id", context.task_id)
            span.set_attribute("context.id", context.context_id)

            # When calling MCP — inject again so MCP gets the same trace
            mcp_headers = {}
            inject(mcp_headers)
            result = await self.mcp_client.call_tool("list_doctors", headers=mcp_headers)

            await event_queue.enqueue_event(new_agent_text_message(result))
```

---

**Step 4 — MCP Server: extract + create child span**

```python
setup_otel("a2a-mcp-server")
tracer = get_tracer("mcp.tools")

@mcp.tool()
async def list_doctors(state: str, city: str, request: Request) -> list:
    ctx = extract(dict(request.headers))   # same trace_id arrives here

    with tracer.start_as_current_span("mcp.tool.list_doctors", context=ctx) as span:
        span.set_attribute("search.state", state)
        span.set_attribute("search.city", city)
        results = db.query(state, city)
        span.set_attribute("result.count", len(results))
        return results
```

---

**Step 5 — Correlation ID in logs (structured JSON)**

Inject `trace_id` into every log line so logs and traces are correlated in Grafana:

```python
import logging
from opentelemetry import trace

class TraceIdFilter(logging.Filter):
    def filter(self, record):
        span = trace.get_current_span()
        ctx  = span.get_span_context()
        record.trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else "none"
        record.span_id  = format(ctx.span_id,  "016x") if ctx.is_valid else "none"
        return True

# Attach to root logger — every log line in any service gets trace_id
logging.getLogger().addFilter(TraceIdFilter())
# Log output: {"time": "...", "msg": "...", "trace_id": "abc123...", "span_id": "def456..."}
```

In Grafana: open a trace in Tempo → click "Logs" → filter by `trace_id` → see all log lines from client + server + MCP in one view.

---

**Best practices summary:**

| Practice | Why |
|---|---|
| W3C `traceparent` header (OTel default) | Standard — works across Python, Go, Java, Node |
| One `setup_otel(service_name)` per process | Separate service names in Tempo trace waterfall |
| `inject(headers)` at every outbound call | Propagates context across HTTP hops |
| `extract(headers)` at every receiver | Restores parent context — creates child not orphan span |
| `TraceIdFilter` on root logger | Log + trace correlation in Grafana without extra effort |
| `trace_id` in error responses | Client can reference it when reporting failures to ops |
| Semaphore-bounded parallel calls | Each parallel branch gets its own child span automatically |

---

### Q24. Is there any correlation between `EventQueue` and `task_store`? Whatever is pushed to `EventQueue` will be stored in `task_store`?

**Answer:**

**No correlation — they serve completely different purposes.**

```
EventQueue  →  carries response content  →  streamed to the CLIENT
task_store  →  tracks task lifecycle     →  persisted on the SERVER
```

```python
class DoctorSearchExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        # EventQueue — pushes response CONTENT to client (SSE / polling)
        await event_queue.enqueue_event(
            new_agent_text_message("Searching for doctors...")
        )
        await event_queue.enqueue_event(
            Artifact(artifact_id="art-1", name="doctors.json", parts=[...])
        )

        # task_store — you NEVER call it inside execute()
        # DefaultRequestHandler manages it automatically:
        #   submitted → working → completed
```

**Flow — who writes what:**

```
Client request
      │
      ▼
DefaultRequestHandler
  ├── task_store.save(task, state="working")       ← SDK writes state
  └── executor.execute(context, event_queue)
              │
              ▼  executor enqueues events
        EventQueue  ──►  streamed to client (SSE)
              │
              ▼  after execute() returns
  DefaultRequestHandler
  └── task_store.save(task, state="completed")     ← SDK writes final state
```

| | `EventQueue` | `task_store` |
|---|---|---|
| **Purpose** | Deliver response content to client | Persist task lifecycle state |
| **Who writes** | Your `AgentExecutor` | `DefaultRequestHandler` (SDK internal) |
| **What's stored** | `Message`, `Artifact`, status events | Task status, messages, artifacts |
| **Scope** | Per-request in-memory buffer | Persistent server-side store |
| **Correlation** | None | None |

> `EventQueue` is a pipe to the client. `task_store` is a database of task records. Pushing to the queue does **not** write to the store — the SDK's `DefaultRequestHandler` manages both independently.
---

## 🎯 Part 2 — Scenario-Based Questions (10)

---

### S1. You are designing a multi-agent healthcare system. A triage agent must route patient queries to specialist agents (cardiology, oncology, mental health). How do you architect this using A2A?

**Answer:**

**Architecture:**
```
Patient App
    ↓ (REST/A2A)
Triage Agent (A2A Server + Client)
    ├── GET /.well-known/agent.json  → Cardiology Agent
    ├── GET /.well-known/agent.json  → Oncology Agent
    └── GET /.well-known/agent.json  → Mental Health Agent
```

**Implementation strategy:**

1. **Agent Registry:** Each specialist registers its Agent Card in a central registry tagged with medical specialty skills
2. **Triage logic:** Triage agent uses an LLM to classify the query, then looks up the matching specialist from the registry
3. **Communication mode:** Use SSE streaming so the patient sees real-time responses
4. **Compliance:**
   - All inter-agent calls over mTLS
   - JWT tokens scoped per specialty (cardiology agent cannot call oncology data)
   - Audit log every `task_id` + `context_id` pair for HIPAA trail
5. **Fallback:** If specialist agent returns `failed`, triage escalates to a human-in-the-loop via `input-required`

```python
class TriageAgentExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        query = context.get_user_input()
        specialty = await self.classify_specialty(query)      # LLM call
        agent_url = await self.registry.lookup(specialty)     # Registry lookup

        async with httpx.AsyncClient() as client:
            a2a_client = await ClientFactory.connect(agent_url, ...)
            message = create_text_message_object(content=query)
            async for event in a2a_client.send_message(message):
                await event_queue.enqueue_event(event)        # Proxy events
```

---

### S2. Your A2A agent handles insurance document queries. Under load, you observe P99 latency of 45 seconds. How do you diagnose and fix this?

**Answer:**

**Diagnosis steps:**
1. **Trace task lifecycle** — instrument `submitted→working→completed` timestamps to find where time is lost
2. **Profile the executor** — is the bottleneck in PDF loading, LLM inference, or event queue saturation?
3. **Check concurrency** — is uvicorn running single-worker? A single-threaded server serializes all requests

**Root causes and fixes:**

| Root Cause | Fix |
|---|---|
| PDF loaded on every request | Cache `pdf_base64` at startup in `__init__`, not per-request |
| LLM inference latency | Use streaming response + SSE so user sees tokens immediately |
| Single uvicorn worker | `uvicorn.run(..., workers=4)` or deploy behind gunicorn |
| `InMemoryTaskStore` contention | Replace with `redis` for non-blocking Task Store|
| Synchronous `answer_query` blocking event loop | Wrap blocking LLM call in `asyncio.to_thread()` |

**`asyncio.to_thread` vs `asyncio.gather` — LangGraph context:**

---

**1. `asyncio.to_thread` — PolicyAgent (sync Anthropic `.invoke()`)**

`ChatAnthropic.invoke()` is synchronous — it blocks whichever thread calls it. Wrapping it in `to_thread` pushes it to a worker thread so the A2A event loop stays free to handle other requests.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-opus-4-6")

# Sync function — safe to run in a worker thread
def run_policy_agent(pdf_base64: str, question: str) -> str:
    msg = HumanMessage(content=[
        {"type": "document", "source": {"type": "base64",
         "media_type": "application/pdf", "data": pdf_base64}},
        {"type": "text", "text": question},
    ])
    return llm.invoke([msg]).content    # sync — blocks its thread, not the event loop
```

```python
import asyncio
from a2a.utils import new_agent_text_message

async def execute(self, context, event_queue):
    query = context.get_user_input()

    # ❌ BAD — sync call blocks event loop; all other A2A requests wait
    # response = run_policy_agent(self.pdf_base64, query)

    # ✅ GOOD — offloaded to worker thread; event loop free for other requests
    response = await asyncio.to_thread(run_policy_agent, self.pdf_base64, query)
    await event_queue.enqueue_event(new_agent_text_message(response))
```

---

**2. `asyncio.gather` — ProviderAgent (async LangGraph `ainvoke()`)**

`create_react_agent.ainvoke()` is already async — no thread needed. `gather` runs multiple queries concurrently on the same event loop.

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools=[...])

async def run_provider_query(query: str) -> str:
    result = await agent.ainvoke(           # async — event loop switches during LLM I/O
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content
```

```python
# Handle multiple users concurrently — total time ≈ slowest single call
queries = ["Cardiologist in Boston", "Pediatrician in Austin", "Dermatologist in NYC"]

responses = await asyncio.gather(*[run_provider_query(q) for q in queries])
```

---

**3. Combine both — parallel policy check + provider search**

One A2A request needs both agents. `gather` runs them in parallel: policy in a worker thread, provider on the event loop.

```python
async def handle_request(context, event_queue):
    query = context.get_user_input()

    policy_resp, provider_resp = await asyncio.gather(
        asyncio.to_thread(run_policy_agent, self.pdf_base64, query),  # sync → thread
        run_provider_query(query),                                     # async → event loop
    )
    # Total time ≈ max(policy_time, provider_time), not sum

    combined = f"Coverage:\n{policy_resp}\n\nProviders:\n{provider_resp}"
    await event_queue.enqueue_event(new_agent_text_message(combined))
```

---

```
ChatAnthropic.invoke()  (sync)  → asyncio.to_thread   runs in worker thread
agent.ainvoke()         (async) → asyncio.gather       runs on event loop
both together                   → asyncio.gather(to_thread(sync), ainvoke())
```

---

### S3. You need to build an A2A agent that produces both a summary (text) and a raw data extract (JSON) for the same task. How do you model this?

**Answer:**
A single `Artifact` can hold **multiple parts** of different types. Pack both `TextPart` and `DataPart` into one artifact — one event, one coherent result.

**Server — single artifact with two parts:**

```python
from uuid import uuid4
from a2a.types import Artifact, Part, TextPart, DataPart

async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    query   = context.get_user_input()
    summary, raw_data = await self.agent.analyze(query)

    artifact = Artifact(
        artifact_id = str(uuid4()),
        index       = 0,
        name        = "insurance_analysis",
        parts       = [
            Part(root=TextPart(text=summary)),      # human-readable summary
            Part(root=DataPart(data=raw_data)),     # structured JSON extract
        ]
    )
    await event_queue.enqueue_event(artifact)
```

**Client — read each part by type:**

```python
task = await client.send_message(...)

artifact = task.artifacts[0]                       # single artifact

for part in artifact.parts:
    if isinstance(part.root, TextPart):
        render_to_ui(part.root.text)               # show summary in UI
    elif isinstance(part.root, DataPart):
        ingest_to_db(part.root.data)               # pipe JSON to database
```

**Two artifacts vs one artifact with two parts:**

| | Two Artifacts | One Artifact, Two Parts |
|---|---|---|
| When to use | Independent outputs (different names, indices) | Logically one result in two formats |
| Client code | Iterate `task.artifacts`, match by `name` | Iterate `artifact.parts`, match by type |
| Event count | 2 enqueue calls | 1 enqueue call |

---

### S4. A downstream A2A agent your orchestrator depends on becomes unavailable mid-task. How do you handle this gracefully?

**Answer:**

**Failure detection:**
- HTTP 5xx or connection timeout on `message/send` or SSE stream drop
- Task polls `tasks/get` and status stays `working` past a deadline

**Resilience pattern — Circuit Breaker + Retry with fallback:**

```python
async def delegate_with_resilience(self, agent_url, message, event_queue):
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                client = await ClientFactory.connect(agent_url, ...)
                async for event in client.send_message(message):
                    await event_queue.enqueue_event(event)
            return  # success
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt == 2:
                # Exhausted retries — emit degraded response
                fallback_msg = new_agent_text_message(
                    "Specialist agent unavailable. Please try again shortly."
                )
                await event_queue.enqueue_event(fallback_msg)
            else:
                await asyncio.sleep(2 ** attempt)  # exponential backoff
```

**Additional strategies:**
- Register a secondary agent in the registry as hot-standby
- Cache the last successful artifact for idempotent queries
- Emit `TaskStatusUpdateEvent` with `failed` state so the orchestrator can reroute

---

### S5. How would you implement an A2A agent that requires human approval before completing a long-running financial transaction?

**Answer:**
Use the `input-required` task state as a formal pause point:

```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    task = context.task

    if task.status.state == TaskState.submitted:
        # First turn — extract details and ask for approval
        details = self.extract_transaction(context.get_user_input())

        approval_request = new_agent_text_message(
            f"Please confirm transfer of ${details['amount']} "
            f"to {details['recipient']}. Reply YES to proceed."
        )
        await event_queue.enqueue_event(approval_request)

        # Signal the framework to pause and wait for user response
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(status=TaskStatus(state=TaskState.input_required))
        )

    elif task.status.state == TaskState.input_required:
        # Second turn — user responded
        approval = context.get_user_input().strip().upper()
        if approval == "YES":
            result = await self.execute_transaction(task)
            await event_queue.enqueue_event(new_agent_text_message(f"Done: {result}"))
        else:
            await event_queue.enqueue_event(
                new_agent_text_message("Transaction canceled per your request.")
            )
```

**Key insight:** The `context_id` maintains the session across both turns. The task's `history` gives the second invocation access to the first turn's context — no external session store needed.

---

### S6. You are building an agent marketplace where third-party agents from different vendors register and are callable by enterprise clients. What A2A design concerns do you address?

**Answer:**

**1. Discovery & Registry:**
- Central registry API where vendors POST their Agent Card URL on deploy
- Registry validates the Agent Card schema and reachability before listing
- Tag-based and semantic search for clients to find agents by capability

**2. Security:**
- All agents must support HTTPS with valid certificates — HTTP rejected
- Registry issues short-lived signed tokens (JWT, 1hr TTL) clients use to call agents
- Agents validate tokens against the registry's JWKS endpoint
- Rate limiting per client identity to prevent abuse

**3. Versioning & Compatibility:**
- `protocol_version` in Agent Card — registry rejects agents below minimum supported version
- Agents expose a deprecation date in their card for breaking changes
- Clients pin to a skill `id` — agents can evolve their implementation without breaking clients

**4. Observability:**
- Every task routed through the marketplace is assigned a `correlation_id`
- Centralized logging: `task_id`, `agent_url`, `client_id`, duration, status
- SLA enforcement: if agent's P95 > threshold, registry demotes visibility

**5. Billing / metering:**
- The registry intercepts (or the agent reports) task counts per `client_id`
- Artifacts size and LLM token usage tracked per `task_id`

---

### S7. You need to migrate an existing REST-based agent service to A2A without breaking existing REST clients. How do you approach this?

**Answer:**

**Strategy: Strangler Fig — run both interfaces in parallel**

```
Existing REST clients  ──→  REST Adapter Layer  ──→  A2A AgentExecutor
New A2A clients        ──────────────────────────→  A2A AgentExecutor
```

**Steps:**

1. **Wrap existing logic in AgentExecutor** — the core business logic moves into `execute()` untouched
2. **Add REST shim:** A thin FastAPI/Flask layer that translates REST requests into A2A `RequestContext` objects and calls the executor directly (bypassing the A2A HTTP layer)
3. **Expose Agent Card** at `/.well-known/agent.json` — new A2A clients discover and call natively
4. **Feature flag:** Route traffic via header: `X-Protocol: a2a` goes to the A2A handler, legacy goes to the REST shim
5. **Deprecation timeline:** Monitor REST usage in logs, notify REST clients, sunset after agreed date

```python
# REST shim endpoint
@app.post("/v1/query")  # existing REST path
async def legacy_endpoint(request: LegacyRequest):
    context = adapt_to_request_context(request)   # translate
    queue = InMemoryEventQueue()
    await executor.execute(context, queue)
    events = await queue.drain()
    return adapt_to_rest_response(events)          # translate back
```

---

### S8. An A2A client is receiving SSE events from a specialist agent, but the stream drops mid-way due to a network blip. How do you implement resumption?

**Answer:**
SSE has a built-in resumption mechanism via the `Last-Event-ID` header and `id:` field on each event.

**Server side — emit event IDs:**
```python
async def execute(self, context, event_queue):
    chunks = self.llm.stream(prompt)
    for i, chunk in enumerate(chunks):
        event = new_agent_text_message(chunk)
        event.event_id = f"{context.task_id}-{i}"   # deterministic ID
        await event_queue.enqueue_event(event)
```

**Client side — reconnect with Last-Event-ID:**
```python
async def stream_with_resume(url, last_event_id=None):
    headers = {}
    if last_event_id:
        headers["Last-Event-ID"] = last_event_id

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, ...) as response:
            async for line in response.aiter_lines():
                # parse SSE, track last id seen
                ...
```

**Server must:** store emitted events (in Redis with short TTL) indexed by `event_id` so it can replay missed events for reconnecting clients.

**Fallback:** If resumption is not supported, fall back to `tasks/get` polling to fetch the current artifact state.

---

### S9. You are asked to build a research pipeline: a coordinator agent fans out queries to 5 specialist research agents in parallel, then synthesizes their outputs. How do you implement this efficiently with A2A?

**Answer:**

```python
class CoordinatorExecutor(AgentExecutor):
    SPECIALIST_URLS = [
        "http://finance-agent:8001",
        "http://legal-agent:8002",
        "http://market-agent:8003",
        "http://tech-agent:8004",
        "http://risk-agent:8005",
    ]

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()

        # Fan-out: send to all specialists concurrently
        async def query_specialist(url: str) -> str:
            async with httpx.AsyncClient(timeout=60.0) as http:
                client = await ClientFactory.connect(url, ClientConfig(httpx_client=http))
                message = create_text_message_object(content=query)
                text = ""
                async for event in client.send_message(message):
                    if isinstance(event, Message):
                        text = get_message_text(event)
                return text

        # All 5 agents called in parallel — total time ≈ slowest single agent
        results = await asyncio.gather(
            *[query_specialist(url) for url in self.SPECIALIST_URLS],
            return_exceptions=True
        )

        # Filter failures, synthesize
        valid = [r for r in results if isinstance(r, str)]
        synthesis_prompt = f"Synthesize:\n" + "\n---\n".join(valid)
        final = await asyncio.to_thread(self.llm.invoke, synthesis_prompt)

        await event_queue.enqueue_event(new_agent_text_message(final.content))
```

**Key design decisions:**
- `asyncio.gather` parallelizes all 5 calls — latency = max(individual) not sum
- `return_exceptions=True` prevents one failure from canceling all others
- Each specialist is an independent A2A server — swap any without touching the coordinator
- Add per-specialist timeout to prevent slow agents from holding up synthesis

---

### S10. A security audit flags that your A2A server is vulnerable to SSRF via push notification callback URLs. How do you fix this?

**Answer:**
SSRF (Server-Side Request Forgery) via push notifications: a malicious client registers `http://169.254.169.254/` (AWS metadata service) as a callback URL, causing your server to exfiltrate cloud credentials.

**Fixes:**

**1. Allowlist validation of callback URLs:**
```python
import ipaddress
from urllib.parse import urlparse

BLOCKED_RANGES = [
    ipaddress.ip_network("169.254.0.0/16"),  # AWS/GCP metadata
    ipaddress.ip_network("10.0.0.0/8"),       # Private RFC1918
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
]

def validate_callback_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False  # Force HTTPS only
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        for blocked in BLOCKED_RANGES:
            if ip in blocked:
                return False
    except ValueError:
        pass  # hostname — resolve and check at call time too
    return True
```

**2. Pre-registration:** Only allow callback URLs that were explicitly pre-registered by the client identity during onboarding — reject ad-hoc URLs

**3. HMAC signing:** Sign every outbound push notification payload so the receiver can verify it originated from your server:
```python
import hmac, hashlib
signature = hmac.new(SECRET_KEY, payload_bytes, hashlib.sha256).hexdigest()
headers["X-A2A-Signature"] = f"sha256={signature}"
```

**4. DNS rebinding protection:** Resolve the hostname at registration time AND at call time — block if IP changes between the two (DNS rebinding attack vector)

**5. Egress firewall:** At the infrastructure level, restrict the A2A server's outbound traffic to known HTTPS endpoints only

---

**Where does validation run? — Integration point in A2A Server**

The A2A SDK flow for push notification registration:
```
Client  →  POST tasks/pushNotification/set  →  DefaultRequestHandler
                                                      ↓
                                            PushNotificationConfigStore.set_info()
                                                      ↓
                                             stores URL + token in memory/DB
```

`validate_callback_url()` is called inside a **custom `PushNotificationConfigStore`** by overriding `set_info()` — this is the A2A SDK's hook point that fires before the URL is stored:

```python
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.types import PushNotificationConfig

class SecurePushNotificationConfigStore(InMemoryPushNotificationConfigStore):
    async def set_info(self, task_id: str, config: PushNotificationConfig) -> None:
        # Validation runs HERE — before the URL is ever stored
        if not validate_callback_url(config.url):
            raise ValueError(f"Blocked callback URL: {config.url}")
        await super().set_info(task_id, config)  # only stored if valid

# Wire it into the A2A server:
request_handler = DefaultRequestHandler(
    agent_executor=MyAgentExecutor(),
    task_store=InMemoryTaskStore(),
    push_notification_config_store=SecurePushNotificationConfigStore(),  # ← inject here
)
app = A2AStarletteApplication(agent_card=card, http_handler=request_handler)
```

**Why `set_info()` not the executor?**
- The executor never sees the push notification URL — it only calls `event_queue.enqueue_event()`
- `DefaultRequestHandler` is what calls `push_notification_config_store.set_info()` when it handles `tasks/pushNotification/set`
- Overriding the store is the clean, SDK-idiomatic interception point — no patching of internal handler logic needed

---


