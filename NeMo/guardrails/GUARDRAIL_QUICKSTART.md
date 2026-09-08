GUARDRAIL_QUICKSTART.md
# NeMo Guardrails - Quick Start Guide

**Complete reference for all guardrail techniques** | [Full Docs](configs/README.md) | [Notebooks](01_NeMo_Guardrails_Installation_and_Basics.ipynb)

---

## 📑 Table of Contents

- [NeMo Guardrails - Quick Start Guide](#nemo-guardrails---quick-start-guide)
  - [📑 Table of Contents](#-table-of-contents)
  - [📦 Installation](#-installation)
    - [NVIDIA Models (Recommended)](#nvidia-models-recommended)
  - [🧠 Critical Concepts](#-critical-concepts)
    - [Tasks vs Flows vs Rails](#tasks-vs-flows-vs-rails)
    - [Organizing Prompts in Separate Files](#organizing-prompts-in-separate-files)
    - [Flow vs Subflow](#flow-vs-subflow)
    - [Magic Names (Auto-Detection)](#magic-names-auto-detection)
      - [Most Magic: Subflow Auto-Activation](#most-magic-subflow-auto-activation)
      - [Medium Magic: Built-in Flow](#medium-magic-built-in-flow)
      - [🔧 No Magic: Custom Flow](#-no-magic-custom-flow)
  - [🔑 Context Variables](#-context-variables)
    - [By Rail Type](#by-rail-type)
    - [Variable Syntax](#variable-syntax)
  - [🛡️ 5 Rail Types](#️-5-rail-types)
    - [1. Input Rails - Validate User Input](#1-input-rails---validate-user-input)
    - [2. Output Rails - Validate Bot Responses](#2-output-rails---validate-bot-responses)
    - [3. Retrieval Rails - Control RAG Systems](#3-retrieval-rails---control-rag-systems)
    - [4. Dialog Rails - Manage Conversations](#4-dialog-rails---manage-conversations)
    - [5. Custom Actions - Python Integration](#5-custom-actions---python-integration)
  - [🚀 Quick Start Examples](#-quick-start-examples)
    - [Option A: Simplest (Action-Free)](#option-a-simplest-action-free)
    - [Option B: Production-Ready](#option-b-production-ready)
  - [🔍 Troubleshooting](#-troubleshooting)
    - [Error 1: Flow Not Found](#error-1-flow-not-found)
    - [Error 2: Subflow vs Flow Confusion](#error-2-subflow-vs-flow-confusion)
    - [Error 3: Jinja2 Syntax Error](#error-3-jinja2-syntax-error)
    - [Error 4: Context Key Not Found](#error-4-context-key-not-found)
  - [🌐 Deploy as Standalone Service](#-deploy-as-standalone-service)
    - [Option 1: NeMo Guardrails Server (Built-in)](#option-1-nemo-guardrails-server-built-in)
    - [Option 2: FastAPI Custom Service](#option-2-fastapi-custom-service)
    - [Option 3: Kubernetes Deployment](#option-3-kubernetes-deployment)
    - [Architecture Patterns](#architecture-patterns)
      - [Pattern 1: Gateway Pattern](#pattern-1-gateway-pattern)
      - [Pattern 2: Sidecar Pattern](#pattern-2-sidecar-pattern)
      - [Pattern 3: Proxy Pattern](#pattern-3-proxy-pattern)
    - [Integration Examples](#integration-examples)
      - [Example 1: Node.js Service Using Guardrails](#example-1-nodejs-service-using-guardrails)
      - [Example 2: Python Microservice](#example-2-python-microservice)
    - [Performance Considerations](#performance-considerations)
    - [Monitoring \& Observability](#monitoring--observability)
    - [Security Best Practices](#security-best-practices)
  - [⚡ Parallel Rails Execution](#-parallel-rails-execution)
    - [What are Parallel Rails?](#what-are-parallel-rails)
    - [When to Use Parallel Execution](#when-to-use-parallel-execution)
    - [Configuration Example](#configuration-example)
    - [Using the $model Variable](#using-the-model-variable)
    - [Parallel Execution Modes](#parallel-execution-modes)
    - [Complete Example with Streaming](#complete-example-with-streaming)
    - [Performance Impact](#performance-impact)
  - [📚 Resources](#-resources)
    - [Documentation](#documentation)
    - [Notebooks](#notebooks)
    - [Official Links](#official-links)
    - [Quick Reference Card](#quick-reference-card)

---

## 📦 Installation

```bash
pip install nemoguardrails

# Verify
python -c "import nemoguardrails; print(nemoguardrails.__version__)"
```

### NVIDIA Models (Recommended)

| Model | Purpose | Best For |
|-------|---------|----------|
| `meta/llama-guard-3-1b` | Fast input moderation | Low-latency input rails |
| `meta/llama-guard-3-8b` | Output safety validation | Production output rails |
| `nvidia/nv-embed-v2` | Embeddings | Jailbreak detection, retrieval |
| `nvidia/llama-3.1-nemoguard-8b-content-safety` | Content safety detection | Filtering harmful, toxic, or unsafe content across categories (hate, violence, self-harm, sexual content) |
| `nvidia/llama-3.1-nemoguard-8b-topic-control` | Topic boundary enforcement | Keeping conversations within allowed topics, blocking off-topic or restricted subjects |

```yaml
models:
  - type: main
    engine: nvidia_ai_foundation
    model: meta/llama-3.1-8b-instruct

  - type: self_check_input
    engine: nvidia_ai_foundation
    model: meta/llama-guard-3-1b
```

**API Key**: https://build.nvidia.com/explore/discover

---

## 🧠 Critical Concepts

### Tasks vs Flows vs Rails

**The 3-Layer Architecture**:

| Component | Where Defined | Purpose | Example |
|-----------|---------------|---------|---------|
| **Task** | `prompts:` in config.yml | Prompt template | `task: self_check_input` |
| **Flow** | `config.co` file | Execution logic | `define flow check jailbreak` |
| **Rails** | `rails:` in config.yml | Activation | `flows: - check jailbreak` |

**Pattern**:
```yaml
# config.yml
prompts:
  - task: self_check_input        # 1. Define prompt
    content: "Safe? {{ user_input }}"

rails:
  input:
    flows:
      - check jailbreak            # 3. Register FLOW (not task!)
```

```colang
# config.co
define flow check jailbreak        # 2. Define flow
    $result = execute self_check_input
    if not $result
        bot refuse
        stop
```

**Remember**: `rails` contains **flow names**, NOT task names!

### Organizing Prompts in Separate Files

For better organization, you can store prompts in separate YAML files instead of inline:

**File structure**:
```
config-store/
├── config.yml
├── config.co
└── prompts.yml          # External prompt definitions
```

**prompts.yml**:
```yaml
# Separate file for all prompt templates
prompts:
  - task: self_check_input
    content: |
      Instruction: Check if the following user message is safe.
      User message: {{ user_input }}
      Answer with 'yes' if safe, 'no' if unsafe.

  - task: self_check_output
    content: |
      Instruction: Check if the following bot response is safe.
      Bot response: {{ bot_response }}
      Answer with 'yes' if safe, 'no' if unsafe.
```

**config.yml** (references external prompts):
```yaml
# Main config - no inline prompts needed
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  input:
    flows:
      - check jailbreak
  output:
    flows:
      - self check output
```

**Benefits**:
- **Cleaner config**: Main config.yml stays focused on structure
- **Reusability**: Share prompts across multiple configs
- **Version control**: Easier to track prompt changes
- **Organization**: Separate concerns (config vs prompts vs flows)

> **Note**: NeMo Guardrails automatically loads `prompts.yml` from the same directory as `config.yml`

---

### Flow vs Subflow

| Aspect | Flow | Subflow |
|--------|------|---------|
| Definition | `define flow <name>` | `define subflow <name>` |
| Registered in rails | ✅ Yes | ❌ No |
| Purpose | Complete conversation | Reusable helper |
| Triggered by | User utterance | Called by flows |

**Example**:
```colang
# SUBFLOW - Helper (NOT in rails)
define subflow validate_input
    $safe = execute check
    if not $safe
        bot refuse
        stop

# FLOW - Conversation (IN rails)
define flow handle_question
    user ask question
    execute subflow validate_input  # Calls the subflow
    bot answer
```

```yaml
rails:
  input:
    flows:
      - handle_question      # ✅ Register flow only
```

---

### Magic Names (Auto-Detection)

**3 Levels of Magic**:

####  Most Magic: Subflow Auto-Activation
```colang
# config.co
define subflow self check input  # Auto-runs on EVERY user input!
    $allowed = execute self_check_input
    if not $allowed
        bot refuse
        stop
```
- ✅ NO rails section needed
- ✅ Auto-activates automatically

####  Medium Magic: Built-in Flow
```yaml
# config.yml - NO config.co needed!
rails:
  output:
    flows:
      - self check output  # Built-in flow provided by NeMo

prompts:
  - task: self_check_output
    content: "Safe? {{ bot_response }}"
```
- ✅ NeMo provides default flow
- ⚠️ Needs rails registration

#### 🔧 No Magic: Custom Flow
```colang
# config.co
define flow my_validator
    user ...
    $safe = execute my_check
    if not $safe
        bot refuse
        stop
```
```yaml
# config.yml
rails:
  input:
    flows:
      - my_validator
```
- ⚠️ You define everything
- ⚠️ Needs rails registration

**Decision Tree**:
```
Using "self check input/output" name?
├─ As subflow → ✨✨✨ Auto-runs (no rails needed)
├─ As task in prompts → ✨✨ Built-in flow (needs rails)
└─ As custom flow → 🔧 Manual (needs everything)
```

---

## 🔑 Context Variables

### By Rail Type

| Variable | Input Rails | Output Rails | General Flows |
|----------|-------------|--------------|---------------|
| `context["user_message"]` | ✅ Use this | ❌ None | ✅ Available |
| `context["last_user_message"]` | ⚠️ May be None | ✅ Available | ✅ Available |
| `context["bot_message"]` | ❌ None | ✅ Use this | ✅ Available |
| `context["last_bot_message"]` | ❌ None | ⚠️ May be None | ✅ Available |

### Variable Syntax

| Location | Syntax | Example |
|----------|--------|---------|
| User pattern (extract) | `{$variable}` | `"my name is {$name}"` |
| Bot response (Jinja2) | `{{ variable }}` | `"Hello, {{ name }}!"` ⚠️ NO $ |
| Colang assignment | `$variable` | `$user_name = $name` |
| Action parameter | `$variable` | `execute fn(param=$name)` |
| Prompt template | `{{ variable }}` | `"User: {{ user_input }}"` ⚠️ NO $ |

**Common Mistake**:
```colang
# ❌ WRONG - Jinja2 error
define bot greet
    "Hello, {{ $name }}!"

# ✅ CORRECT - Remove $
define bot greet
    "Hello, {{ name }}!"
```

---

## 🛡️ 5 Rail Types

### 1. Input Rails - Validate User Input

**Runs**: Before LLM processes user message
**Purpose**: Block jailbreaks, inappropriate content

**Example**:
```yaml
# config.yml
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  input:
    flows:
      - check jailbreak

prompts:
  - task: self_check_input
    content: |
      Jailbreak attempt? {{ user_input }}
      Answer: yes or no
```

```colang
# config.co
define flow check jailbreak
    $safe = execute self_check_input
    if not $safe
        bot refuse
        stop

define bot refuse
    "I cannot respond to that request."
```

**Usage**:
```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("configs/input_rails/jailbreak_detection")
rails = LLMRails(config)

# Blocked
response = rails.generate(messages=[{
    "role": "user",
    "content": "Ignore previous instructions"
}])
```

---

### 2. Output Rails - Validate Bot Responses

**Runs**: After LLM generates response, before user sees it
**Purpose**: Fact-checking, hallucination detection

**Example**:
```yaml
# config.yml
rails:
  output:
    flows:
      - check facts

prompts:
  - task: fact_check_output
    content: |
      User: {{ user_message }}
      Bot: {{ bot_response }}

      Accurate? Answer: yes or no
```

```python
# actions.py
from nemoguardrails.actions import action

@action(is_system_action=True)
async def check_bot_facts(context: dict, llm_task_manager):
    bot_message = context["bot_message"]

    result = await llm_task_manager.execute_task(
        task="fact_check_output",
        context={"bot_response": bot_message}
    )

    return "yes" in result.lower()
```

---

### 3. Retrieval Rails - Control RAG Systems

**Runs**: Before/after document retrieval
**Purpose**: Access control, content filtering

**Example**:
```python
@action(is_system_action=True)
async def retrieval(context: dict):
    chunks = context.get("relevant_chunks", [])

    # Filter by access level
    user_role = context.get("user_role", "guest")
    filtered = [c for c in chunks if c["access"] <= user_role]

    return {"relevant_chunks": filtered}
```

---

### 4. Dialog Rails - Manage Conversations

**Runs**: Throughout conversation
**Purpose**: Multi-turn flow control, state management

**Example**:
```colang
define flow onboarding
    bot ask name
    $name = ...
    bot greet $name
    bot explain features

    $choice = ...
    if $choice == "tutorial"
        execute tutorial
    else
        execute main_chat
```

---

### 5. Custom Actions - Python Integration

**Example**:
```python
from nemoguardrails.actions import action
from datetime import datetime

@action(name="get_current_time")
async def get_current_time():
    return datetime.now().strftime("%I:%M %p")

@action(is_system_action=True)
async def validate_email(context: dict, email: str):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

```colang
define flow get time
    user ask time
    $current_time = execute get_current_time
    bot provide time
        "The current time is {{ current_time }}"
```

---

## 🚀 Quick Start Examples

### Option A: Simplest (Action-Free)

**Single file**: `config.yml`
```yaml
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  output:
    flows:
      - self check output  # Built-in flow

prompts:
  - task: self_check_output
    content: |
      Safe? {{ bot_response }}
      Answer: yes or no
```

### Option B: Production-Ready

**config.yml**:
```yaml
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  input:
    flows:
      - check jailbreak
  output:
    flows:
      - check facts

prompts:
  - task: self_check_input
    content: "Jailbreak? {{ user_input }}"
  - task: fact_check_output
    content: "Accurate? {{ bot_response }}"
```

**config.co**:
```colang
define flow check jailbreak
    $safe = execute self_check_input
    if not $safe
        bot refuse
        stop

define flow check facts
    $accurate = execute fact_check_output
    if not $accurate
        bot provide_safe_response
```

---

## 🔍 Troubleshooting

### Error 1: Flow Not Found
```
WARNING: Flow 'self_check_input' not found
```

**Cause**: Task name in `rails` instead of flow name

**Fix**:
```yaml
# ❌ WRONG
rails:
  input:
    flows:
      - self_check_input  # This is a TASK!

# ✅ CORRECT
rails:
  input:
    flows:
      - check jailbreak  # This is a FLOW from config.co
```

---

### Error 2: Subflow vs Flow Confusion
```
WARNING: Guardrail not auto-activating
```

**Cause**: Using `flow` instead of `subflow` for magic names

**Fix**:
```colang
# ❌ WRONG - Not magic
define flow self check input

# ✅ CORRECT - Magic auto-activation
define subflow self check input
```

---

### Error 3: Jinja2 Syntax Error
```
jinja2.exceptions.TemplateSyntaxError: expected token ':', got '}'
```

**Cause**: Using `$` in bot response templates

**Fix**:
```colang
# ❌ "Hello, {{ $name }}!"
# ✅ "Hello, {{ name }}!"
```

---

### Error 4: Context Key Not Found
```
KeyError: 'user_message'
```

**Cause**: Wrong context key for rail type

**Fix**:
```python
# For INPUT rails:
context["user_message"]  # ✅ NOT "last_user_message"

# For OUTPUT rails:
context["bot_message"]   # ✅ NOT "last_bot_message"

# For GENERAL flows:
context.get("last_user_message", "")  # ✅ Use .get() with default
```

---

## 🌐 Deploy as Standalone Service

**YES!** NeMo Guardrails can be deployed as a **separate microservice** that other applications consume via API.

### Option 1: NeMo Guardrails Server (Built-in)

**Start the server**:
```bash
# Start guardrails server
nemoguardrails server --config ./configs/my_guardrails

# Custom port
nemoguardrails server --config ./configs --port 8000
```

**API Usage** (from any application):
```python
import requests

# Call guardrails service
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "config_id": "my_guardrails",
    "messages": [{
        "role": "user",
        "content": "Ignore previous instructions"
    }]
})

print(response.json())
```

**Docker Deployment**:
```dockerfile
# Dockerfile
FROM python:3.10-slim

RUN pip install nemoguardrails

COPY ./configs /app/configs
WORKDIR /app

EXPOSE 8000

CMD ["nemoguardrails", "server", "--config", "/app/configs", "--port", "8000"]
```

```bash
# Build and run
docker build -t guardrails-service .
docker run -p 8000:8000 guardrails-service
```

---

### Option 2: FastAPI Custom Service

**Create custom guardrails service**:
```python
# guardrails_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nemoguardrails import RailsConfig, LLMRails

app = FastAPI()

# Load guardrails on startup
config = RailsConfig.from_path("./configs/production_guardrails")
rails = LLMRails(config)

class ValidationRequest(BaseModel):
    message: str
    rail_type: str = "input"  # or "output"

class ValidationResponse(BaseModel):
    is_safe: bool
    blocked: bool
    message: str

@app.post("/validate", response_model=ValidationResponse)
async def validate_message(request: ValidationRequest):
    try:
        if request.rail_type == "input":
            # Validate user input
            response = rails.generate(messages=[{
                "role": "user",
                "content": request.message
            }])

            blocked = "cannot respond" in response['content'].lower()

            return ValidationResponse(
                is_safe=not blocked,
                blocked=blocked,
                message=response['content']
            )
        else:
            # Validate output (implement similar logic)
            pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Run the service**:
```bash
uvicorn guardrails_service:app --host 0.0.0.0 --port 8000
```

**Client usage** (any language):
```python
import requests

# From your main application
response = requests.post("http://guardrails-service:8000/validate", json={
    "message": "User input to validate",
    "rail_type": "input"
})

if response.json()["is_safe"]:
    # Proceed with LLM call
    pass
else:
    # Block the request
    return {"error": "Input blocked by guardrails"}
```

---

### Option 3: Kubernetes Deployment

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: guardrails-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: guardrails
  template:
    metadata:
      labels:
        app: guardrails
    spec:
      containers:
      - name: guardrails
        image: your-registry/guardrails-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: NVIDIA_API_KEY
          valueFrom:
            secretKeyRef:
              name: nvidia-credentials
              key: api-key
---
apiVersion: v1
kind: Service
metadata:
  name: guardrails-service
spec:
  selector:
    app: guardrails
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

---

### Architecture Patterns

#### Pattern 1: Gateway Pattern
```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       v
┌─────────────────┐
│ Guardrails API  │ ← Centralized service
│   (Port 8000)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│  Main LLM API   │
└─────────────────┘
```

#### Pattern 2: Sidecar Pattern
```
┌────────────────────────────┐
│        Pod                  │
│  ┌──────────┐ ┌──────────┐│
│  │ Main App │→│Guardrails││
│  │          │ │ Sidecar  ││
│  └──────────┘ └──────────┘│
└────────────────────────────┘
```

#### Pattern 3: Proxy Pattern
```
Client → Guardrails Proxy → LLM Service
         (validate in/out)
```

---

### Integration Examples

#### Example 1: Node.js Service Using Guardrails
```javascript
// app.js
const axios = require('axios');

const GUARDRAILS_URL = 'http://guardrails-service:8000';

async function validateInput(userMessage) {
  const response = await axios.post(`${GUARDRAILS_URL}/validate`, {
    message: userMessage,
    rail_type: 'input'
  });

  return response.data.is_safe;
}

app.post('/chat', async (req, res) => {
  const { message } = req.body;

  // Check with guardrails service
  const isSafe = await validateInput(message);

  if (!isSafe) {
    return res.status(400).json({ error: 'Input blocked' });
  }

  // Proceed with LLM call
  const llmResponse = await callLLM(message);
  res.json({ response: llmResponse });
});
```

#### Example 2: Python Microservice
```python
# main_service.py
import httpx
from fastapi import FastAPI

app = FastAPI()
GUARDRAILS_URL = "http://guardrails-service:8000"

@app.post("/chat")
async def chat(message: str):
    # Validate with guardrails
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{GUARDRAILS_URL}/validate",
            json={"message": message, "rail_type": "input"}
        )

        if not response.json()["is_safe"]:
            return {"error": "Input blocked by guardrails"}

    # Call main LLM
    llm_response = await call_llm(message)
    return {"response": llm_response}
```

---

### Performance Considerations

| Deployment | Latency | Throughput | Best For |
|------------|---------|------------|----------|
| **Embedded** | Lowest (~50ms) | High | Single application |
| **Sidecar** | Low (~100ms) | Medium | Per-service isolation |
| **Centralized** | Medium (~200ms) | Very High | Multiple services |
| **Edge/CDN** | Variable | Very High | Global distribution |

**Optimization Tips**:
- Use connection pooling
- Cache validation results (with TTL)
- Batch validation requests
- Use async/await patterns
- Deploy close to LLM service

---

### Monitoring & Observability

```python
# Add metrics to guardrails service
from prometheus_client import Counter, Histogram
import time

validation_counter = Counter('guardrails_validations_total', 'Total validations', ['type', 'result'])
validation_latency = Histogram('guardrails_latency_seconds', 'Validation latency')

@app.post("/validate")
async def validate_message(request: ValidationRequest):
    start_time = time.time()

    # Validation logic
    result = rails.generate(...)
    blocked = "cannot respond" in result['content'].lower()

    # Metrics
    validation_counter.labels(
        type=request.rail_type,
        result='blocked' if blocked else 'allowed'
    ).inc()

    validation_latency.observe(time.time() - start_time)

    return ValidationResponse(...)
```

---

### Security Best Practices

1. **API Authentication**:
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/validate")
async def validate_message(
    request: ValidationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401)
    # ... validation logic
```

2. **Rate Limiting**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler

limiter = Limiter(key_func=lambda: request.client.host)

@app.post("/validate")
@limiter.limit("100/minute")
async def validate_message(request: ValidationRequest):
    # ... validation logic
```

3. **Network Policies** (Kubernetes):
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: guardrails-policy
spec:
  podSelector:
    matchLabels:
      app: guardrails
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: api-gateway
    ports:
    - port: 8000
```

---

## ⚡ Parallel Rails Execution

> **Source**: [Parallel Rails Tutorial](https://github.com/NVIDIA/GenerativeAIExamples/blob/main/nemo/NeMo-Guardrails/Parallel_Rails_Tutorial.ipynb)

### What are Parallel Rails?

Run multiple independent safety checks **concurrently** instead of sequentially to reduce latency and improve throughput.

### When to Use Parallel Execution

✅ **Use parallel for:**
- I/O-bound rails (external API calls to LLMs, third-party integrations)
- Multiple independent input/output rails without shared state
- Production environments where response latency matters

❌ **Avoid parallel for:**
- CPU-bound rails (may add overhead)
- Development/testing (sequential is easier to debug)

### Configuration Example

```yaml
rails:
  input:
    parallel: True    # Enable parallel execution
    flows:
      - content safety check input $model=content_safety
      - topic safety check input $model=topic_control
  output:
    parallel: True
    flows:
      - content safety check output $model=content_safety
      - self check output
```

### Using the $model Variable

**Purpose**: Dynamically select different models for the same rail logic

**Syntax**: `$model=<model_name>`

**Example**:
```yaml
rails:
  input:
    flows:
      - content safety check input $model=content_safety
      - topic safety check input $model=topic_control

models:
  - type: content_safety
    engine: nvidia_ai_foundation
    model: nvidia/llama-3.1-nemoguard-8b-content-safety

  - type: topic_control
    engine: nvidia_ai_foundation
    model: nvidia/llama-3.1-nemoguard-8b-topic-control
```

**Benefits**:
- Reuse same rail flow with different models
- Cleaner configuration
- Easy to swap models without changing flow logic

### Parallel Execution Modes

| Mode | Input Rails | Output Rails | Best For |
|------|-------------|--------------|----------|
| **Full Parallel** | `parallel: True` | `parallel: True` | Maximum performance |
| **Input Only** | `parallel: True` | `parallel: False` | Input-heavy checks |
| **Output Only** | `parallel: False` | `parallel: True` | Output-heavy checks |
| **Sequential** | `parallel: False` | `parallel: False` | Debugging, development |

### Complete Example with Streaming

```yaml
models:
  - type: main
    engine: nvidia_ai_foundation
    model: meta/llama-3.1-8b-instruct

  - type: content_safety
    engine: nvidia_ai_foundation
    model: nvidia/llama-3.1-nemoguard-8b-content-safety

  - type: topic_control
    engine: nvidia_ai_foundation
    model: nvidia/llama-3.1-nemoguard-8b-topic-control

rails:
  input:
    parallel: True
    flows:
      - content safety check input $model=content_safety
      - topic safety check input $model=topic_control
  output:
    parallel: True
    flows:
      - content safety check output $model=content_safety
      - self check output
  streaming:
    enabled: True
    chunk_size: 200
    context_size: 50
```

### Performance Impact

**Sequential Execution**:
```
Check 1 (300ms) → Check 2 (300ms) = 600ms total
```

**Parallel Execution**:
```
Check 1 (300ms) ┐
                ├─ = 300ms total
Check 2 (300ms) ┘
```

**Speedup**: ~2x faster for 2 independent checks!

---

## 📚 Resources

### Documentation
- [Input Rails Guide](configs/input_rails/README.md)
- [Output Rails Guide](configs/output_rails/README.md)
- [Advanced Concepts](configs/advanced_concepts/README.md)

### Notebooks
1. `01_NeMo_Guardrails_Installation_and_Basics.ipynb` - Setup
2. `02_Input_Rails.ipynb` - Input validation
3. `03_Output_Rails.ipynb` - Output checking
4. `04_Retrieval_Rails.ipynb` - RAG integration
5. `02_Advanced_Concepts_Variables_Actions_Context.ipynb` - Advanced patterns

### Official Links
- GitHub: https://github.com/NVIDIA/NeMo-Guardrails
- Docs: https://docs.nvidia.com/nemo/guardrails/
- NVIDIA AI Foundation: https://build.nvidia.com/explore/discover

### Quick Reference Card

```
MAGIC SUBFLOWS (Auto-run):
  define subflow self check input   → Input rail
  define subflow self check output  → Output rail

BUILT-IN FLOWS (Task + Rails):
  task: self_check_input + rails    → NeMo provides flow
  task: self_check_output + rails   → NeMo provides flow

CONTEXT VARIABLES:
  Input rails:  context["user_message"]
  Output rails: context["bot_message"]
  Prompts:      {{ user_input }}, {{ bot_response }}

VARIABLE SYNTAX:
  User pattern:     {$variable}
  Bot response:     {{ variable }}  (NO $!)
  Colang:           $variable

COMMON MISTAKES:
  ❌ {{ $name }}        → ✅ {{ name }}
  ❌ rails: - task_name → ✅ rails: - flow_name
  ❌ define flow self check input (no rails) → ✅ define subflow self check input
```

---

**Total Lines**: ~600 (reduced from 2,221)
