Model Serving Routing.md
# Model Serving & Routing Layer: vLLM, TGI, Cost Optimization

## Table of Contents
- [Model Serving \& Routing Layer: vLLM, TGI, Cost Optimization](#model-serving--routing-layer-vllm-tgi-cost-optimization)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Architecture Pattern](#architecture-pattern)
  - [Part 1: Model Serving Backends](#part-1-model-serving-backends)
    - [1. vLLM (Recommended for Cost)](#1-vllm-recommended-for-cost)
      - [Setup](#setup)
      - [Usage](#usage)
      - [Cost Model](#cost-model)
      - [Advanced: Prompt Caching in vLLM](#advanced-prompt-caching-in-vllm)
    - [2. Hugging Face TGI (Text Generation Inference)](#2-hugging-face-tgi-text-generation-inference)
      - [Setup](#setup-1)
      - [Usage](#usage-1)
    - [3. Ollama (Local CPU Fallback)](#3-ollama-local-cpu-fallback)
      - [Setup](#setup-2)
      - [Usage](#usage-2)
    - [4. Claude \& GPT-4 (API Providers)](#4-claude--gpt-4-api-providers)
  - [Part 2: Intelligent Routing](#part-2-intelligent-routing)
    - [Cost-Aware Router](#cost-aware-router)
  - [Part 3: Caching Strategy](#part-3-caching-strategy)
    - [Prompt Caching (RECOMMENDED)](#prompt-caching-recommended)
  - [Part 4: Cost Tracking \& Observability](#part-4-cost-tracking--observability)
  - [Deployment on Presight Private Cloud](#deployment-on-presight-private-cloud)
    - [Kubernetes Architecture](#kubernetes-architecture)
  - [Key Interview Questions](#key-interview-questions)
  - [Next: RAG \& Memory Systems](#next-rag--memory-systems)

---

## Overview

The **model-serving layer** sits between your agent orchestration framework and the underlying LLM backends. It must:
1. Support multiple backends (open-weights self-hosted + approved APIs)
2. Route requests intelligently (cost, latency, accuracy, model capability)
3. Cache responses to reduce costs
4. Provide fallback mechanisms
5. Track spend per agent/user/tenant
6. Expose metrics for cost optimization

---

## Architecture Pattern

```
┌─────────────────────────┐
│  LangGraph Agent        │
│  (Orchestration)        │
└────────────┬────────────┘
             │ invoke(prompt, model_hint="complex")
             ▼
┌─────────────────────────────────────────┐
│ Model Router & Caching Layer            │
│  ┌──────────────────────────────────┐   │
│  │ 1. Cost Analysis                 │   │
│  │    - Check token budget          │   │
│  │    - Check model capabilities    │   │
│  ├──────────────────────────────────┤   │
│  │ 2. Cache Lookup                  │   │
│  │    - Check prompt cache (Redis)  │   │
│  │    - If hit: return cached result│   │
│  ├──────────────────────────────────┤   │
│  │ 3. Model Selection               │   │
│  │    - Simple: Llama 2 ($0.0005)   │   │
│  │    - Complex: Claude 3 ($0.003)  │   │
│  │    - Fallback if primary down    │   │
│  ├──────────────────────────────────┤   │
│  │ 4. Rate Limiting & Quotas        │   │
│  │    - Check user/tenant quota     │   │
│  │    - Queue if exceeds rate       │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
             │ selected_model, cached_result
             ▼
┌─────────────────────────────────────────┐
│ Backend Selection                       │
│                                         │
│  vLLM  ← Llama 2, Mistral (self-hosted)│
│  TGI   ← Code Llama, Bloom (self-hosted│
│  API   ← Claude, GPT-4 (approved APIs) │
│  Ollama← Local CPU models              │
└─────────────────────────────────────────┘
             │
             ▼ LLM Response + Token Usage
┌─────────────────────────────────────────┐
│ Cost & Metrics Tracking                 │
│  - Cost: $0.00234                       │
│  - Latency: 1230ms                      │
│  - Tokens: 450 input, 120 output        │
│  - Cache hit: yes/no                    │
│  - Model: llama-2-70b                   │
└─────────────────────────────────────────┘
```

---

## Part 1: Model Serving Backends

### 1. vLLM (Recommended for Cost)

**vLLM** is a high-throughput serving engine for open-weight LLMs.

**Best for:** Llama 2, Llama 3, Mistral, Qwen (cost-optimized inference)

#### Setup

```bash
# Install vLLM
pip install vllm

# Start server (8 GPUs, 70B model)
python -m vllm.entrypoints.openai_api_server \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# Or Docker
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 8
```

#### Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="fake-key"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "user", "content": "Analyze this security incident..."}
    ],
    temperature=0.7,
    max_tokens=1000,
    top_p=0.95
)

print(f"Cost: ${response.usage.completion_tokens * 0.0001}")
```

#### Cost Model

| Model | Input (per 1K tokens) | Output (per 1K tokens) | Advantage |
|-------|----------------------|----------------------|-----------|
| Llama 2 70B | $0.0005 | $0.0005 | Cheapest, good quality |
| Llama 3 70B | $0.0008 | $0.0008 | Better reasoning |
| Mistral 7B | $0.0002 | $0.0002 | Cheapest option |
| Qwen 72B | $0.0006 | $0.0006 | Multilingual |

#### Advanced: Prompt Caching in vLLM

vLLM supports prefix caching to reuse encoded context.

```python
# Common security knowledge (reused across many requests)
SECURITY_CONTEXT = """
MITRE ATT&CK Framework:
- T1566: Phishing (Email, etc.)
- T1047: Windows Management Instrumentation
...
D3FEND Countermeasures:
- D3-CS: Cryptographic Signature Detection
...
"""

def query_with_cached_context(incident_details):
    """Reuse security knowledge context across queries"""
    messages = [
        {
            "role": "system",
            "content": SECURITY_CONTEXT,
            "cache_control": {"type": "ephemeral"}  # Cache for this session
        },
        {
            "role": "user",
            "content": incident_details
        }
    ]
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=messages
    )
    
    # Check cache efficiency
    print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
    print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
    print(f"Regular input tokens: {response.usage.prompt_tokens}")
    
    return response
```

---

### 2. Hugging Face TGI (Text Generation Inference)

**TGI** is another high-performance serving engine with good batch support.

**Best for:** Code-specific models (CodeLlama), larger batches

#### Setup

```bash
docker run --gpus all -p 8080:80 \
  -e MODEL_ID=meta-llama/CodeLlama-34b-Instruct-hf \
  ghcr.io/huggingface/text-generation-inference:latest
```

#### Usage

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="http://localhost:8080",
    timeout=60
)

response = client.text_generation(
    "# Analyze this code for security vulnerabilities:\n" + code_snippet,
    max_new_tokens=500,
    details=True  # Include token counts
)

print(f"Generated: {response.generated_text}")
print(f"Tokens: {response.details.tokens}")
```

---

### 3. Ollama (Local CPU Fallback)

**Ollama** runs models on CPU (slower but no GPU needed).

**Best for:** Fallback when GPU exhausted, development/testing

#### Setup

```bash
ollama pull llama2-uncensored:7b
ollama serve --port 11434
```

#### Usage

```python
import requests

def query_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2-uncensored:7b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()['response']
```

---

### 4. Claude & GPT-4 (API Providers)

For **high-complexity tasks** that benefit from reasoning.

```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-...")

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=2000,
    messages=[
        {
            "role": "user",
            "content": "Threat model the entire agent platform..."
        }
    ]
)

# Cost calculation
input_cost = response.usage.input_tokens * 0.015 / 1000
output_cost = response.usage.output_tokens * 0.075 / 1000
total_cost = input_cost + output_cost
print(f"Cost: ${total_cost:.6f}")
```

---

## Part 2: Intelligent Routing

### Cost-Aware Router

```python
from enum import Enum
from dataclasses import dataclass
import time
from functools import lru_cache

class ModelTier(Enum):
    CHEAP = "cheap"        # Llama 2 7B
    MEDIUM = "medium"      # Llama 2 70B
    EXPENSIVE = "expensive"  # Claude Opus
    FALLBACK = "fallback"  # Ollama (CPU)

@dataclass
class ModelConfig:
    """Configuration for each model"""
    model_id: str
    tier: ModelTier
    input_cost_per_1k: float
    output_cost_per_1k: float
    latency_ms: float  # Expected P99
    capabilities: set[str]  # ["reasoning", "code_analysis", "rag", ...]
    max_tokens: int
    available: bool = True
    rate_limit_rpm: int = 1000

# Model registry
MODELS = {
    "llama2-7b": ModelConfig(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        tier=ModelTier.CHEAP,
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0002,
        latency_ms=200,
        capabilities={"basic_qa", "summarization"},
        max_tokens=4096
    ),
    "llama2-70b": ModelConfig(
        model_id="meta-llama/Llama-2-70b-chat-hf",
        tier=ModelTier.MEDIUM,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0005,
        latency_ms=500,
        capabilities={"reasoning", "code_analysis", "rag", "planning"},
        max_tokens=4096
    ),
    "claude-opus": ModelConfig(
        model_id="claude-3-opus-20240229",
        tier=ModelTier.EXPENSIVE,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        latency_ms=1000,
        capabilities={"reasoning", "threat_modeling", "code_analysis", "complex_planning"},
        max_tokens=200000
    ),
    "ollama": ModelConfig(
        model_id="llama2-uncensored:7b",
        tier=ModelTier.FALLBACK,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        latency_ms=5000,
        capabilities={"basic_qa"},
        max_tokens=4096
    )
}

class ModelRouter:
    def __init__(self, cost_budget_usd=100.0):
        self.cost_budget_usd = cost_budget_usd
        self.spent_usd = 0.0
        self.request_count = {}  # Track rate limits
        
    def select_model(
        self,
        task_complexity: str,  # "simple", "moderate", "complex"
        required_capabilities: set[str],
        must_stay_under_latency_ms: int = 5000,
        preferred_tier: str = None
    ) -> str:
        """
        Intelligently select model based on:
        1. Required capabilities
        2. Latency constraints
        3. Cost budget
        4. Availability
        """
        
        # Filter models by required capabilities
        capable_models = [
            (name, config) 
            for name, config in MODELS.items()
            if required_capabilities.issubset(config.capabilities)
            and config.available
            and config.latency_ms <= must_stay_under_latency_ms
        ]
        
        if not capable_models:
            # Fallback to cheapest available
            capable_models = [
                (name, config)
                for name, config in MODELS.items()
                if config.available
            ]
        
        # Sort by cost (cheapest first)
        capable_models.sort(
            key=lambda x: x[1].input_cost_per_1k + x[1].output_cost_per_1k
        )
        
        # Select based on cost budget
        for model_name, config in capable_models:
            estimated_cost = (500 * config.input_cost_per_1k + 500 * config.output_cost_per_1k) / 1000
            
            if self.spent_usd + estimated_cost <= self.cost_budget_usd:
                return model_name
            elif config.tier == ModelTier.FALLBACK:
                # Last resort: use free fallback
                return model_name
        
        # Final fallback
        return "ollama"
    
    def route_request(self, prompt: str, task_type: str, agent_name: str) -> tuple[str, dict]:
        """
        Full routing logic with caching and fallback.
        
        Returns: (model_name, metadata)
        """
        
        # Step 1: Determine requirements based on task
        requirements = self._analyze_task(prompt, task_type)
        
        # Step 2: Check cache first
        cache_hit = self._check_cache(prompt)
        if cache_hit:
            return ("cache", {"cached_response": cache_hit})
        
        # Step 3: Select model
        model_name = self.select_model(
            task_complexity=requirements['complexity'],
            required_capabilities=requirements['capabilities'],
            must_stay_under_latency_ms=5000,
            preferred_tier=requirements.get('preferred_tier')
        )
        
        # Step 4: Check rate limits
        if not self._check_rate_limit(model_name):
            # Fallback to slower model if rate limited
            model_name = "ollama"
        
        config = MODELS[model_name]
        
        return (model_name, {
            "tier": config.tier.value,
            "estimated_cost": (500 * config.input_cost_per_1k + 500 * config.output_cost_per_1k) / 1000,
            "latency_ms": config.latency_ms
        })
    
    def _analyze_task(self, prompt: str, task_type: str) -> dict:
        """Analyze task to determine requirements"""
        complexity = "simple"
        capabilities = {"basic_qa"}
        
        # Heuristic: longer prompts = more complex
        if len(prompt) > 2000:
            complexity = "complex"
            capabilities = {"reasoning", "code_analysis"}
        elif len(prompt) > 500:
            complexity = "moderate"
            capabilities = {"reasoning"}
        
        # Task-specific requirements
        if "code" in task_type.lower() or "vulnerability" in task_type.lower():
            capabilities.add("code_analysis")
        
        if "threat_model" in task_type.lower():
            capabilities = {"reasoning", "threat_modeling"}
            complexity = "complex"
        
        return {
            "complexity": complexity,
            "capabilities": capabilities,
            "preferred_tier": "medium" if complexity == "moderate" else "cheap"
        }
    
    def _check_cache(self, prompt: str):
        """Check Redis/in-memory cache"""
        # Simplified: use lru_cache
        return None  # No cache hit for this example
    
    def _check_rate_limit(self, model_name: str) -> bool:
        """Check if model has exceeded rate limits"""
        config = MODELS[model_name]
        current_count = self.request_count.get(model_name, 0)
        
        # Simple rate limit: max 1000 requests per minute
        return current_count < config.rate_limit_rpm

# Usage in agent
router = ModelRouter(cost_budget_usd=100.0)

model_name, metadata = router.route_request(
    prompt=incident_details,
    task_type="security_analysis",
    agent_name="soc_agent"
)

print(f"Selected: {model_name}, Cost: ${metadata['estimated_cost']}")
```

---

## Part 3: Caching Strategy

### Prompt Caching (RECOMMENDED)

Presight agents will have **highly reusable context**:
- MITRE ATT&CK framework
- D3FEND countermeasures
- Organizational security policies
- Threat intelligence

Cache these **persistent contexts** to reduce costs by 80%.

```python
import hashlib
from functools import lru_cache

class ContextCache:
    """Cache expensive-to-compute contexts"""
    
    # Persistent context (reused across many requests)
    MITRE_CONTEXT = """
    MITRE ATT&CK Framework
    ...
    """
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def compute_retrieval_context(query: str) -> str:
        """
        RAG: retrieve relevant docs from vector DB.
        Cache the result for identical queries.
        """
        # In practice: pgvector query
        return f"Relevant context for: {query}"
    
    @staticmethod
    def build_cached_prompt(incident_details: str) -> list[dict]:
        """Build prompt with cached contexts"""
        return [
            {
                "role": "system",
                "content": ContextCache.MITRE_CONTEXT,
                "cache_control": {"type": "ephemeral"}  # Cache for session
            },
            {
                "role": "user",
                "content": incident_details
            }
        ]

# Result: 80% cost savings on repeated security contexts
```

---

## Part 4: Cost Tracking & Observability

```python
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class ModelUsageMetrics:
    """Track all costs and metrics"""
    timestamp: datetime
    agent_name: str
    tenant_id: str
    model_name: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    cache_hit: bool
    latency_ms: int
    status: str  # "success", "fallback", "error"
    
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost
    
    def to_json(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent_name,
            "tenant": self.tenant_id,
            "model": self.model_name,
            "tokens": f"{self.input_tokens}+{self.output_tokens}",
            "cost": f"${self.total_cost():.6f}",
            "cache_hit": self.cache_hit,
            "latency_ms": self.latency_ms
        })

class CostTracker:
    """Track costs per agent, tenant, model"""
    
    def __init__(self, postgres_conn):
        self.db = postgres_conn
    
    def log_usage(self, metrics: ModelUsageMetrics):
        """Log to database"""
        self.db.execute(
            """
            INSERT INTO model_usage (
                timestamp, agent_name, tenant_id, model_name,
                input_tokens, output_tokens, cost, cache_hit, latency_ms, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                metrics.timestamp,
                metrics.agent_name,
                metrics.tenant_id,
                metrics.model_name,
                metrics.input_tokens,
                metrics.output_tokens,
                metrics.total_cost(),
                metrics.cache_hit,
                metrics.latency_ms,
                metrics.status
            )
        )
    
    def get_cost_by_agent(self, days=30) -> dict:
        """Cost breakdown by agent"""
        result = self.db.query(
            """
            SELECT agent_name, SUM(cost) as total_cost, COUNT(*) as requests
            FROM model_usage
            WHERE timestamp > NOW() - INTERVAL %s DAY
            GROUP BY agent_name
            """,
            (days,)
        )
        return {row['agent_name']: row['total_cost'] for row in result}
    
    def get_cost_by_model(self, days=30) -> dict:
        """Cost breakdown by model"""
        result = self.db.query(
            """
            SELECT model_name, SUM(cost) as total_cost, COUNT(*) as requests
            FROM model_usage
            WHERE timestamp > NOW() - INTERVAL %s DAY
            GROUP BY model_name
            """,
            (days,)
        )
        return {row['model_name']: row['total_cost'] for row in result}
```

---

## Deployment on Presight Private Cloud

### Kubernetes Architecture

```yaml
# presight-serving.yaml

# vLLM Service (GPU-accelerated)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      nodeSelector:
        accelerator: gpu  # Run only on GPU nodes
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.3
        args:
          - --model=meta-llama/Llama-2-70b-chat-hf
          - --tensor-parallel-size=8
          - --gpu-memory-utilization=0.9
        resources:
          limits:
            nvidia.com/gpu: "8"  # 8 GPUs per pod
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---

# Model Router Service (CPU-based routing logic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-router
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-router
  template:
    metadata:
      labels:
        app: model-router
    spec:
      containers:
      - name: router
        image: presight/model-router:latest
        ports:
        - containerPort: 5000
        env:
        - name: VLLM_ENDPOINT
          value: http://vllm-serving:8000
        - name: OLLAMA_ENDPOINT
          value: http://ollama:11434
        - name: COST_BUDGET_USD
          value: "100"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"

---

# Ollama fallback (CPU-only)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-fallback
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      nodeSelector:
        cpu-fallback: "true"
      containers:
      - name: ollama
        image: ollama/ollama:latest
        args: ["serve"]
        resources:
          limits:
            cpu: "8"
            memory: "8Gi"
        ports:
        - containerPort: 11434
```

---

## Key Interview Questions

1. **How would you design model routing for cost optimization?**
   - Answer: Capability-based selection, cost budget tracking, fallback chains

2. **How do you implement caching for security-specific tasks?**
   - Answer: MITRE/D3FEND context caching, prompt caching in vLLM

3. **What's your strategy for handling model unavailability?**
   - Answer: Fallback chain (expensive → medium → cheap → offline)

4. **How would you ensure cost transparency?**
   - Answer: Cost tracking per agent/tenant/model, daily dashboards

---

## Next: RAG & Memory Systems

See `03_RAG_MEMORY_SYSTEMS.md` for how agents retrieve and remember context.

