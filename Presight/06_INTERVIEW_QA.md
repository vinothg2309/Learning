# Presight Interview Prep: 50+ Technical Q&A

## Table of Contents
- [Section 1: Agent Orchestration (10 Q&A)](#section-1-agent-orchestration-10-qa)
  - [Q1-Q10: Agent design, loops, memory, testing, etc.](#agent-orchestration-questions)
- [Section 2: Model Serving & Costs (9 Q&A)](#section-2-model-serving--costs-9-qa)
  - [Q11-Q17: Routing, caching, fallbacks, isolation, cost tracking](#model-serving-questions)
- [Section 3: Security Guardrails (9 Q&A)](#section-3-security-guardrails-9-qa)
  - [Q18-Q26: Injection, authorization, threat modeling](#security-questions)
- [Section 4: System Design & Architecture (6 Q&A)](#section-4-system-design--architecture-6-qa)
  - [Q27-Q31: Full design, scaling, deployment, recovery](#system-design-questions)
- [Section 5: Interview Red-Flags & Common Pitfalls](#section-5-interview-red-flags--common-pitfalls)
  - [Q33-Q42: Mistakes to avoid](#common-mistakes)
- [Final Tips for Interview](#final-tips-for-interview)

---

## Section 1: Agent Orchestration (10 Q&A)

### Q1: Design a multi-agent orchestration framework from scratch. Walk through a SOC incident scenario.

**Answer:**

I would design a **LangGraph-based framework** with planner-executor-critic pattern:

```python
# Agent receives: "Suspicious login from 192.168.1.100 at 2AM"

# Step 1: Planner Agent
# - LLM analyzes incident
# - Creates step-by-step plan
# - Output: "Step 1: Query SIEM for 192.168.1.100"
#           "Step 2: Check geolocation (is 2AM normal?)"
#           "Step 3: Query AD for user details"

# Step 2: Executor Agent
# - Calls tools (SIEM, geolocation, AD)
# - Collects results
# - Updates shared state with findings

# Step 3: Critic Agent
# - Reviews findings
# - Severity assessment
# - Decision: escalate to human? Or close?

# Conditional loop: If confidence low → replan
```

**Key design decisions:**
- **State sharing:** All agents work on same AgentState dict
- **Tool validation:** Every tool call validated before execution
- **Cost awareness:** Conditional routing based on cost budget
- **Determinism:** Same input → reproducible trace (critical for evaluation)

### Q2: How would you implement human-in-the-loop approval for sensitive agent actions?

**Answer:**

```python
# In executor node, before calling dangerous tools:
def validate_sensitive_action(state: AgentState) -> AgentState:
    """Check if action needs human approval"""
    
    SENSITIVE_TOOLS = ["delete_account", "modify_firewall", "shell_exec"]
    
    for tool_call in state['tools_to_call']:
        if tool_call['name'] in SENSITIVE_TOOLS:
            # Raise interrupt to get human approval
            raise NodeInterrupt(
                f"Requires approval: {tool_call['name']}({tool_call['args']})"
            )
    
    return state

# At API level:
for event in agent.stream(state):
    if isinstance(event, NodeInterrupt):
        # Return to frontend, wait for approval
        return {"status": "awaiting_approval", "action": event.message}
    # Continue on approval
```

### Q3: How do you prevent agent loops (endless replanning)?

**Answer:**

```python
class LoopDetector:
    """Prevent infinite loops"""
    
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
    
    def check_loop(self, state: AgentState) -> str:
        """Check if we're looping"""
        
        # Count iterations
        iteration_count = len(state.get('execution_log', []))
        
        if iteration_count >= self.max_iterations:
            return "end"  # Force exit after N iterations
        
        # Check for repeated actions
        recent_actions = [
            action['tool'] 
            for action in state['execution_log'][-5:]
        ]
        if len(recent_actions) == len(set(recent_actions)):
            # Same actions repeating → likely loop
            return "end"
        
        # Normal decision
        if state['critic_decision'] == "revise_plan":
            return "planner"
        else:
            return "end"
```

### Q4: How would you implement cost-aware model switching within an agent?

**Answer:**

```python
class CostAwareRouter:
    def route_model(self, state: AgentState) -> str:
        """Select model based on cost budget"""
        
        remaining_budget = state['budget_usd'] - state['spent_usd']
        
        # Estimate tokens for this step
        estimated_tokens = len(state['current_task']) / 4
        
        # Simple model: $0.0001 per 1K tokens
        # Complex model: $0.001 per 1K tokens
        
        simple_cost = (estimated_tokens * 0.0001) / 1000
        complex_cost = (estimated_tokens * 0.001) / 1000
        
        if complex_cost <= remaining_budget:
            return "claude-3-opus"  # Use best model if affordable
        elif simple_cost <= remaining_budget:
            return "llama-2-70b"  # Fallback to cheaper model
        else:
            return "ollama"  # Free fallback
```

### Q5: How would you implement parallel tool execution in an agent?

**Answer:**

```python
# LangGraph supports parallel execution with reducer functions

class AgentState(TypedDict):
    task: str
    # Tools can be called in parallel
    tool_results: Annotated[list, "extend"]  # Extend = parallel aggregation
    
async def executor_parallel(state: AgentState) -> AgentState:
    """Execute multiple tools in parallel"""
    
    tools = [
        ("query_siem", {"ip": "192.168.1.100"}),
        ("check_geolocation", {"ip": "192.168.1.100"}),
        ("query_threat_intel", {"ip": "192.168.1.100"})
    ]
    
    # Execute all in parallel
    results = await asyncio.gather(*[
        tool_registry[tool_name].invoke(args)
        for tool_name, args in tools
    ])
    
    state['tool_results'].extend([
        {"tool": name, "result": result}
        for (name, _), result in zip(tools, results)
    ])
    
    return state
```

### Q6: Design a memory system for long-running agents (e.g., a security analyst working over days).

**Answer:**

Three-tier memory:
1. **Working memory** (current task) - In AgentState
2. **Session memory** (last 10 findings) - In PostgreSQL
3. **Long-term memory** (patterns, common threats) - In vector DB

```python
class AgentMemory:
    def __init__(self, agent_id: str, db):
        self.agent_id = agent_id
        self.db = db
    
    def save_finding(self, finding: dict):
        """Save to session/long-term memory"""
        # Save to Postgres for recent access
        self.db.execute("""
            INSERT INTO agent_findings (agent_id, timestamp, finding, severity)
            VALUES (%s, NOW(), %s, %s)
        """, (self.agent_id, json.dumps(finding), finding['severity']))
        
        # Embed and save to vector DB for similarity search
        embedding = embed_model.encode(finding['description'])
        self.vector_db.upsert(
            id=finding['id'],
            vector=embedding,
            metadata=finding
        )
    
    def recall_similar_incidents(self, current_incident: str, k=5):
        """Retrieve similar past incidents"""
        query_embedding = embed_model.encode(current_incident)
        
        # Vector similarity search
        similar = self.vector_db.search(query_embedding, k=k)
        
        return similar
```

### Q7: How would you handle agent context length limits?

**Answer:**

```python
class ContextWindowManager:
    """Manage token limits for long-running agents"""
    
    def __init__(self, max_tokens=100000):
        self.max_tokens = max_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using BPE"""
        return len(text) // 4  # Rough estimate
    
    def compress_context(self, state: AgentState) -> AgentState:
        """Compress old findings to fit context"""
        
        total_tokens = sum(
            self.estimate_tokens(json.dumps(item))
            for item in state['execution_log']
        )
        
        if total_tokens > self.max_tokens * 0.8:  # 80% threshold
            # Summarize old findings
            old_findings = state['execution_log'][:-10]
            summary = self._summarize_findings(old_findings)
            
            state['execution_log'] = [
                {"type": "summary", "content": summary},
                *state['execution_log'][-10:]
            ]
        
        return state
    
    def _summarize_findings(self, findings: list) -> str:
        """Summarize findings with LLM"""
        prompt = f"Summarize these security findings: {findings}"
        return llm.invoke(prompt).content
```

### Q8: How do you test agent determinism and reproducibility?

**Answer:**

```python
class DeterminismTest:
    """Verify agent produces same output for same input"""
    
    def test_determinism(self, agent, test_case: dict, num_runs=5):
        """Run agent N times, check consistency"""
        
        results = []
        for i in range(num_runs):
            # Same input, should get same output
            result = agent.invoke(test_case)
            results.append(result)
        
        # Compare findings across runs
        first_findings = set(results[0]['findings'])
        
        for i, result in enumerate(results[1:], 1):
            current_findings = set(result['findings'])
            
            # Allow small variation (LLM temperature), but core should match
            overlap = len(first_findings & current_findings) / len(first_findings)
            
            assert overlap > 0.9, f"Run {i}: findings only {overlap*100}% consistent"
    
    def test_checkpoint_replay(self, agent, thread_id: str):
        """Verify checkpoint snapshots are reproducible"""
        
        # Get checkpoint at step N
        checkpoint = agent.storage.get_checkpoint(thread_id, step=5)
        
        # Resume from checkpoint
        result = agent.invoke(state, config={"configurable": {"thread_id": thread_id}})
        
        # Should produce same final result
        assert result['findings'] == original_result['findings']
```

### Q9: How would you implement agent explainability (showing reasoning)?

**Answer:**

```python
class ExplainableAgent:
    """Agent that outputs reasoning traces"""
    
    def invoke_with_explanation(self, task: str) -> dict:
        """Return both result and reasoning"""
        
        response = {
            'task': task,
            'plan': "",
            'execution_steps': [],
            'findings': [],
            'reasoning': ""
        }
        
        # Capture intermediate states
        for event in self.agent.stream({"input": task}):
            if event['type'] == 'node_execution':
                node_name = event['node']
                node_output = event['state']
                
                # Log reasoning from each node
                response['execution_steps'].append({
                    'node': node_name,
                    'plan': node_output.get('plan', ''),
                    'tools_called': [log['tool'] for log in node_output.get('execution_log', [])],
                    'findings': node_output.get('findings', [])
                })
        
        return response
```

### Q10: Design a framework for A/B testing different prompts/models in production.

**Answer:**

```python
class ABTestingFramework:
    """Test different agent versions in production"""
    
    def __init__(self):
        self.variants = {
            'prompt_v1': {"prompt": "...", "model": "llama"},
            'prompt_v2': {"prompt": "...", "model": "llama"},
            'model_v1': {"prompt": "...", "model": "llama"},
            'model_v2': {"prompt": "...", "model": "mistral"}
        }
    
    def select_variant(self, user_id: str) -> str:
        """Deterministically assign user to variant"""
        # Hash user_id to variant (consistent across requests)
        variant_idx = hash(user_id) % len(self.variants)
        return list(self.variants.keys())[variant_idx]
    
    def record_result(self, user_id: str, variant: str, metric: str, value: float):
        """Track metrics per variant"""
        self.db.execute("""
            INSERT INTO ab_test_results (user_id, variant, metric, value, timestamp)
            VALUES (%s, %s, %s, %s, NOW())
        """, (user_id, variant, metric, value))
    
    def analyze_results(self, days=7) -> dict:
        """Statistical comparison of variants"""
        
        results = self.db.query(f"""
            SELECT variant, metric,
                   AVG(value) as mean,
                   STDDEV(value) as stddev,
                   COUNT(*) as n
            FROM ab_test_results
            WHERE timestamp > NOW() - INTERVAL '{days} days'
            GROUP BY variant, metric
        """)
        
        # Compare statistically significant differences
        return self._compute_statistical_significance(results)
```

---

## Section 2: Model Serving & Costs (8 Q&A)

### Q11: Design a model routing layer for an agent platform with multiple LLM backends.

**Answer:**

See `02_MODEL_SERVING_ROUTING.md` - covers cost-aware routing, caching, fallback chains.

### Q12: How would you implement prompt caching to reduce costs?

**Answer:**

```python
# Identify reusable contexts
MITRE_CONTEXT = """
MITRE ATT&CK Framework:
T1000: Initial Access
...
"""  # Reused across many requests

def query_with_cached_context(incident: str):
    """Reuse cached security knowledge"""
    
    messages = [
        {
            "role": "system",
            "content": MITRE_CONTEXT,
            "cache_control": {"type": "ephemeral"}  # vLLM caches this
        },
        {
            "role": "user",
            "content": incident
        }
    ]
    
    response = client.chat.completions.create(
        model="llama-2-70b",
        messages=messages
    )
    
    # Check cache effectiveness
    if response.usage.cache_read_input_tokens > 0:
        print(f"Cache hit! Saved {response.usage.cache_read_input_tokens} tokens")
    
    return response
```

### Q13: How would you handle model unavailability gracefully?

**Answer:**

```python
class FallbackChain:
    """Fallback from expensive to cheap models"""
    
    CHAIN = [
        {"model": "claude-opus", "cost": 0.075},
        {"model": "llama-2-70b", "cost": 0.0005},
        {"model": "mistral-7b", "cost": 0.0002},
        {"model": "ollama", "cost": 0.0}  # Free, but CPU-based
    ]
    
    async def invoke_with_fallback(self, prompt: str) -> dict:
        """Try each model in order"""
        
        for model_config in self.CHAIN:
            try:
                response = await self._try_model(
                    model_config['model'],
                    prompt,
                    timeout=5.0
                )
                
                return {
                    'response': response,
                    'model': model_config['model'],
                    'cost': model_config['cost'],
                    'fallback': False
                }
            
            except (TimeoutError, ConnectionError):
                # Model unavailable, try next
                continue
            
            except OutOfQuotaError:
                # Rate limited, try next
                continue
        
        # All models exhausted
        raise FatalError("All model backends unavailable")
    
    async def _try_model(self, model: str, prompt: str, timeout: float):
        """Try to invoke model with timeout"""
        
        try:
            response = await asyncio.wait_for(
                self.clients[model].invoke(prompt),
                timeout=timeout
            )
            return response
        
        except asyncio.TimeoutError:
            raise TimeoutError(f"Model {model} timed out")
```

### Q14: How would you optimize batch processing to reduce per-request overhead?

**Answer:**

```python
class BatchProcessor:
    """Group requests to reduce startup overhead"""
    
    def __init__(self, batch_size=32, max_wait_ms=100):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
        self.timers = {}
    
    async def add_request(self, request: dict) -> str:
        """Queue request for batch processing"""
        
        request_id = str(uuid.uuid4())
        self.queue.append((request_id, request))
        
        # Start timer for max wait
        asyncio.create_task(self._wait_and_flush(request_id))
        
        # If batch full, process immediately
        if len(self.queue) >= self.batch_size:
            await self._process_batch()
        
        return request_id
    
    async def _process_batch(self):
        """Process accumulated requests"""
        
        if not self.queue:
            return
        
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        # Batch call (single model inference)
        batch_prompts = [req for _, req in batch]
        batch_results = await self.model.batch_invoke(batch_prompts)
        
        # Return results
        for (request_id, _), result in zip(batch, batch_results):
            self.results[request_id] = result
```

### Q15: How would you implement multi-tenant cost isolation?

**Answer:**

```python
class MultiTenantCostTracking:
    """Track costs per tenant with quotas"""
    
    def invoke_with_cost_isolation(self, tenant_id: str, request: dict):
        """Enforce per-tenant cost limits"""
        
        # Check tenant quota
        quota = self.db.query(
            "SELECT monthly_budget_usd FROM tenant_budgets WHERE tenant_id = %s",
            (tenant_id,)
        )[0]
        
        spent = self.db.query(
            "SELECT SUM(cost_usd) FROM cost_log WHERE tenant_id = %s AND month = CURRENT_MONTH",
            (tenant_id,)
        )[0]['sum'] or 0
        
        remaining = quota['monthly_budget_usd'] - spent
        
        if remaining <= 0:
            raise BudgetExceededError(f"Tenant {tenant_id} out of budget")
        
        # Route to cheaper model if approaching limit
        model = "llama-2-70b" if remaining > 50 else "mistral-7b"
        
        # Execute with tenant isolation
        result = agent.invoke(
            request,
            config={"configurable": {
                "tenant_id": tenant_id,
                "budget_remaining_usd": remaining
            }}
        )
        
        # Log cost
        cost_usd = result['usage']['cost_usd']
        self.db.execute(
            "INSERT INTO cost_log (tenant_id, cost_usd) VALUES (%s, %s)",
            (tenant_id, cost_usd)
        )
        
        return result
```

### Q16: How would you implement cost-aware prompt engineering?

**Answer:**

```python
class CostOptimizedPrompting:
    """Reduce costs through better prompting"""
    
    @staticmethod
    def get_optimized_prompt(task: str, budget_usd: float) -> str:
        """Adapt prompt based on budget"""
        
        if budget_usd > 0.10:
            # Expensive budget: detailed instructions
            return f"""
            Analyze the following security incident comprehensively.
            Consider all possible attack vectors and provide detailed reasoning.
            Task: {task}
            """
        
        elif budget_usd > 0.01:
            # Medium budget: focused instructions
            return f"""
            Analyze this security incident and identify key findings.
            Task: {task}
            """
        
        else:
            # Low budget: minimal instructions
            return f"""
            Analyze: {task}
            Return: Key findings only.
            """
    
    @staticmethod
    def reduce_context(findings: list, max_tokens: int) -> list:
        """Trim findings to fit token budget"""
        
        # Keep highest-severity findings
        findings_sorted = sorted(
            findings,
            key=lambda x: severity_to_score(x['severity']),
            reverse=True
        )
        
        # Keep findings until token limit
        result = []
        token_count = 0
        
        for finding in findings_sorted:
            tokens = len(finding['description']) // 4
            if token_count + tokens <= max_tokens:
                result.append(finding)
                token_count += tokens
        
        return result
```

### Q17: How would you track LLM costs by application and route to cheaper models based on budget?

**Answer:**

**Problem:** Multiple applications use shared LLM infrastructure; need per-app cost visibility and automatic routing to cheaper models when budget constraints hit.

**Solution: Three-tier cost tracking + dynamic model routing**

**Tier 1: Cost Tracking at Invocation**

```python
from datetime import datetime
from decimal import Decimal
import uuid

class CostTracker:
    """Track costs per application with real-time aggregation"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def log_llm_call(self, app_id: str, user_id: str, model: str, 
                     input_tokens: int, output_tokens: int) -> float:
        """Record LLM invocation cost"""
        
        # Cost model per provider
        cost_models = {
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-opus": {"input": 0.015, "output": 0.075},
            "claude-haiku": {"input": 0.00025, "output": 0.00125},
            "llama-2-70b": {"input": 0.0005, "output": 0.0005},
        }
        
        rates = cost_models[model]
        input_cost = (input_tokens * rates["input"]) / 1000
        output_cost = (output_tokens * rates["output"]) / 1000
        total_cost = input_cost + output_cost
        
        # Persist to PostgreSQL
        self.db.execute("""
            INSERT INTO llm_costs 
            (id, app_id, user_id, model, input_tokens, output_tokens, cost_usd, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            str(uuid.uuid4()), app_id, user_id, model,
            input_tokens, output_tokens, Decimal(str(total_cost)), datetime.utcnow()
        ))
        
        return total_cost
```

**Tier 2: Cost Aggregation & Reporting**

```python
class CostAggregator:
    """Query costs by application, model, time period"""
    
    def get_app_cost_summary(self, app_id: str, start_date, end_date) -> dict:
        """Total cost and breakdown by model"""
        
        results = self.db.query("""
            SELECT 
                model,
                COUNT(*) as call_count,
                SUM(input_tokens + output_tokens) as total_tokens,
                SUM(cost_usd) as total_cost,
                AVG(cost_usd) as avg_cost_per_call,
                MIN(cost_usd) as min_cost,
                MAX(cost_usd) as max_cost
            FROM llm_costs
            WHERE app_id = %s AND timestamp BETWEEN %s AND %s
            GROUP BY model
            ORDER BY total_cost DESC
        """, (app_id, start_date, end_date))
        
        return {
            'app_id': app_id,
            'period': {'start': start_date, 'end': end_date},
            'by_model': results,
            'total_cost_usd': sum(r['total_cost'] for r in results),
            'total_calls': sum(r['call_count'] for r in results)
        }
    
    def get_hourly_cost_trend(self, app_id: str, days=30) -> list:
        """Cost breakdown by hour for trend analysis"""
        
        return self.db.query("""
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                SUM(cost_usd) as hourly_cost,
                COUNT(*) as calls
            FROM llm_costs
            WHERE app_id = %s AND timestamp > NOW() - INTERVAL '%d days'
            GROUP BY DATE_TRUNC('hour', timestamp)
            ORDER BY hour DESC
        """, (app_id, days))
    
    def get_all_apps_ranking(self, start_date, end_date) -> list:
        """Rank all applications by total spend"""
        
        return self.db.query("""
            SELECT 
                app_id,
                COUNT(*) as call_count,
                SUM(cost_usd) as total_cost,
                AVG(cost_usd) as avg_cost
            FROM llm_costs
            WHERE timestamp BETWEEN %s AND %s
            GROUP BY app_id
            ORDER BY total_cost DESC
        """, (start_date, end_date))
```

**Tier 3: Cost-Aware Dynamic Routing**

```python
class CostAwareModelRouter:
    """Route to cheaper models based on budget remaining"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.model_config = {
            "gpt-4-turbo": {"cost": 0.04, "latency_ms": 800, "quality": 9.5},
            "claude-opus": {"cost": 0.09, "latency_ms": 900, "quality": 9.3},
            "gpt-3.5-turbo": {"cost": 0.002, "latency_ms": 200, "quality": 7.8},
            "llama-2-70b": {"cost": 0.001, "latency_ms": 400, "quality": 7.5},
            "claude-haiku": {"cost": 0.0015, "latency_ms": 100, "quality": 6.5},
        }
    
    def select_model(self, app_id: str, task_complexity: str, 
                    budget_remaining_usd: float) -> dict:
        """Select model based on complexity, latency SLA, and budget"""
        
        # Task complexity determines minimum quality threshold
        quality_threshold = {
            "high": 9.0,      # Incident analysis, critical decisions
            "medium": 7.5,    # Standard security tasks
            "low": 6.0        # Classification, basic queries
        }[task_complexity]
        
        # Filter models meeting quality threshold
        candidates = [
            (name, cfg) for name, cfg in self.model_config.items()
            if cfg['quality'] >= quality_threshold
        ]
        
        # If budget allows, use best quality
        if budget_remaining_usd >= 0.10:
            return max(candidates, key=lambda x: x[1]['quality'])
        
        # If budget moderate, balance quality vs cost
        elif budget_remaining_usd >= 0.05:
            candidates.sort(key=lambda x: x[1]['quality'] / x[1]['cost'], reverse=True)
            return candidates[0]
        
        # If budget tight, use cheapest meeting threshold
        elif budget_remaining_usd >= 0.01:
            candidates.sort(key=lambda x: x[1]['cost'])
            return candidates[0]
        
        # If budget critical, use cheapest regardless of quality
        else:
            cheapest = min(self.model_config.items(), key=lambda x: x[1]['cost'])
            return cheapest
    
    def get_budget_remaining(self, app_id: str, monthly_budget_usd: float) -> float:
        """Get remaining budget for application this month"""
        
        spent = self.db.query("""
            SELECT SUM(cost_usd) as total
            FROM llm_costs
            WHERE app_id = %s AND EXTRACT(YEAR FROM timestamp) = EXTRACT(YEAR FROM NOW())
              AND EXTRACT(MONTH FROM timestamp) = EXTRACT(MONTH FROM NOW())
        """, (app_id,))[0]
        
        spent_amount = float(spent['total'] or 0)
        return max(0, monthly_budget_usd - spent_amount)
    
    def invoke_with_routing(self, app_id: str, prompt: str, 
                           task_complexity: str, monthly_budget_usd: float):
        """Invoke LLM with automatic cost-aware routing"""
        
        # Check remaining budget
        budget_remaining = self.get_budget_remaining(app_id, monthly_budget_usd)
        
        if budget_remaining <= 0:
            raise BudgetExceededError(
                f"App {app_id} exceeded monthly budget of ${monthly_budget_usd}"
            )
        
        # Select model
        model_name, model_config = self.select_model(
            app_id, task_complexity, budget_remaining
        )
        
        # Call LLM
        response = self.llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        
        # Log cost
        cost = self.cost_tracker.log_llm_call(
            app_id=app_id,
            user_id="system",
            model=model_name,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
        
        return {
            'response': response.choices[0].message.content,
            'model': model_name,
            'cost_usd': cost,
            'budget_remaining': budget_remaining - cost,
            'task_complexity': task_complexity
        }
```

**Integration with Observability:**

```python
class CostObservability:
    """Export costs to Prometheus + Grafana"""
    
    def __init__(self):
        from prometheus_client import Counter, Histogram, Gauge
        
        # Metrics
        self.cost_counter = Counter(
            'llm_cost_usd_total',
            'Total LLM cost in USD',
            ['app_id', 'model']
        )
        
        self.cost_histogram = Histogram(
            'llm_cost_usd_per_call',
            'Cost per LLM call',
            ['app_id', 'model'],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.budget_gauge = Gauge(
            'app_budget_remaining_usd',
            'Budget remaining for application',
            ['app_id']
        )
    
    def record_cost(self, app_id: str, model: str, cost_usd: float):
        """Emit metrics"""
        self.cost_counter.labels(app_id=app_id, model=model).inc(cost_usd)
        self.cost_histogram.labels(app_id=app_id, model=model).observe(cost_usd)
```

**Grafana Dashboard Queries:**

```sql
-- Cost by application (stacked bar, last 30 days)
SELECT 
  DATE_TRUNC('day', timestamp) as day,
  app_id,
  SUM(cost_usd) as daily_cost
FROM llm_costs
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY day, app_id
ORDER BY day DESC

-- Cost per model (pie chart)
SELECT 
  model,
  SUM(cost_usd) as total_cost,
  COUNT(*) as call_count,
  ROUND(AVG(cost_usd)::numeric, 6) as avg_cost
FROM llm_costs
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY model
ORDER BY total_cost DESC

-- Budget vs Actual (gauge per app)
SELECT 
  app_id,
  monthly_budget_usd as budget,
  (SELECT SUM(cost_usd) FROM llm_costs 
   WHERE app_id = tenant_budgets.app_id 
   AND EXTRACT(MONTH FROM timestamp) = EXTRACT(MONTH FROM NOW())) as spent,
  monthly_budget_usd - COALESCE(spent, 0) as remaining
FROM tenant_budgets
```

**Key Design Decisions:**

| Decision | Rationale |
|----------|-----------|
| **Track at invocation** | Real-time visibility; no batch delays |
| **Persistent PostgreSQL store** | Audit trail; historical analysis; alerting queries |
| **Quality thresholds per complexity** | Don't degrade below task requirements |
| **Budget routing** | Prevent overspends without admin intervention |
| **Prometheus metrics** | Integrate with existing observability stack |

**Interview Talking Points:**

✓ **Cost visibility:** Every LLM call tagged with app_id, model, cost  
✓ **Automated routing:** No manual model selection; budget-aware  
✓ **Tenant isolation:** Each app has separate budget; prevents cross-tenant overspend  
✓ **Observability:** Metrics exported to Grafana for real-time monitoring  
✓ **Fallback logic:** If budget tight, gracefully degrade to cheaper model  

---

## Section 3: Security Guardrails (9 Q&A)

### Q18: How would you implement prompt injection defense?

**Answer:**

See `03_SECURITY_GUARDRAILS.md` - covers InputValidator, structural separation, output filtering.

### Q19: Design tool-use authorization for a multi-tenant agent system.

**Answer:**

See `03_SECURITY_GUARDRAILS.md` - role-based permissions with examples.

### Q20: How would you prevent data exfiltration in agent outputs?

**Answer:**

```python
# Pattern-based detection + sanitization (see Q18 in 03_SECURITY_GUARDRAILS.md)
```

### Q21: Map your guardrails to NIST AI RMF.

**Answer:**

```
Govern (G):
- G1: Agent platform risk assessment
- G2: Governance: role-based access, tool authorization
- G3: Policies: acceptable use, audit logging

Map (M):
- M1: Understand use cases (SOC, pentest, code review)
- M2: Identify risks (prompt injection, tool abuse, data exfiltration)

Measure (Ms):
- Ms1: Performance metrics (accuracy, latency, cost)
- Ms2: Safety validation (red-team, guardrail tests)

Manage (Mg):
- Mg1: Risk mitigation (input filtering, output sanitization)
- Mg2: Incident response (alert on security events)
```

### Q22: How would you implement red-team evaluation for a security agent?

**Answer:**

See `04_EVALUATION_FRAMEWORK.md` - RedTeamPrompts and adversarial testing.

### Q23: What is your threat model for the agent platform itself?

**Answer:**

```
Threat 1: Compromised LLM weights
- Impact: Agent behavior altered
- Mitigation: Model signature verification, regular re-verification

Threat 2: Prompt injection via user input
- Impact: Agent bypasses guardrails
- Mitigation: Input validation, semantic anomaly detection

Threat 3: Tool-use bypass
- Impact: Agent calls unauthorized tools
- Mitigation: Role-based authorization, explicit allowlist

Threat 4: Data exfiltration in outputs
- Impact: Sensitive data leaked
- Mitigation: Pattern-based detection, output sanitization

Threat 5: Model supply-chain compromise
- Impact: Malicious model deployed
- Mitigation: Source verification, sandboxed loading

Threat 6: Resource exhaustion attack
- Impact: DoS via token budget exhaustion
- Mitigation: Rate limiting, per-tenant quotas, circuit breakers
```

### Q24: How would you implement audit logging for compliance?

**Answer:**

```python
class AuditLog:
    """Tamper-proof audit trail"""
    
    def log_security_event(self, event: dict):
        """Log event with hash chaining"""
        
        # Add hash chain (tamper detection)
        prev_hash = self.db.query(
            "SELECT event_hash FROM audit_log ORDER BY timestamp DESC LIMIT 1"
        )[0]['event_hash']
        
        event_hash = hashlib.sha256(
            (prev_hash + json.dumps(event)).encode()
        ).hexdigest()
        
        self.db.execute("""
            INSERT INTO audit_log (timestamp, event_type, details, agent_id, tenant_id, event_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            datetime.now(),
            event['type'],  # "tool_call", "injection_detected", "escalation"
            json.dumps(event),
            event.get('agent_id'),
            event.get('tenant_id'),
            event_hash
        ))
        
        # Alert on critical events
        if event['severity'] == 'critical':
            self.send_alert(event)
```

### Q25: How would you ensure agents don't violate organizational security policies?

**Answer:**

```python
class PolicyEnforcer:
    """Enforce organizational security policies"""
    
    POLICIES = {
        'no_credential_output': "Agents must not output passwords, API keys",
        'no_external_communication': "Agents cannot call external APIs",
        'no_data_deletion': "No delete operations on production data",
        'least_privilege_tools': "Only use tools required for task"
    }
    
    def validate_tool_call(self, tool_name: str, tool_args: dict) -> bool:
        """Check if tool call violates policies"""
        
        # Policy: no_credential_output
        if 'password' in tool_name or 'secret' in tool_name:
            return False
        
        # Policy: no_external_communication
        EXTERNAL_TOOLS = ['call_external_api', 'send_email', 'post_to_slack']
        if tool_name in EXTERNAL_TOOLS:
            return False
        
        # Policy: no_data_deletion
        if 'delete' in tool_name or 'drop' in tool_name:
            # Only allow if explicitly approved in tool_args
            if not tool_args.get('approved_by_human'):
                return False
        
        return True
```

### Q26: How would you implement secrets management for agents?

**Answer:**

```python
from hvac import Client

class SecretsManager:
    """Secure secrets using HashiCorp Vault"""
    
    def __init__(self, vault_addr: str, vault_token: str):
        self.client = Client(url=vault_addr, token=vault_token)
    
    def get_secret(self, secret_path: str) -> dict:
        """Retrieve secret from Vault (not stored in code/env)"""
        
        response = self.client.secrets.kv.read_secret_version(path=secret_path)
        return response['data']['data']
    
    def inject_secrets_into_tool(self, tool_config: dict) -> dict:
        """Inject secrets at runtime, not at config time"""
        
        injected = tool_config.copy()
        
        # Replace secret references with actual values
        for key, value in injected.items():
            if isinstance(value, str) and value.startswith("vault://"):
                secret_path = value.replace("vault://", "")
                secret = self.get_secret(secret_path)
                injected[key] = secret
        
        return injected

# Usage
secrets_mgr = SecretsManager(
    vault_addr="http://vault:8200",
    vault_token="s.xxxxxxx"
)

# In agent config:
tool_config = {
    "siem_api_key": "vault://presight/siem/api-key",  # Not actual key
    "tenable_api_token": "vault://presight/tenable/token"
}

# At runtime:
actual_config = secrets_mgr.inject_secrets_into_tool(tool_config)
# Now has actual credentials, but never stored anywhere
```

---

## Section 4: System Design & Architecture (6 Q&A)

### Q27: Design the entire agent platform from scratch. What are your key decisions?

**Answer:**

**Architecture:**
```
┌─────────────────────────────────┐
│ Agent Orchestration (LangGraph) │  → Core: planner-executor-critic
├─────────────────────────────────┤
│ Model Routing & Caching Layer   │  → Cost optimization, fallbacks
├─────────────────────────────────┤
│ Tool Registry & Authorization   │  → Role-based access control
├─────────────────────────────────┤
│ RAG & Memory Systems            │  → Vector DB, knowledge bases
├─────────────────────────────────┤
│ Guardrails & Safety Controls    │  → OWASP/PLOT4AI/NIST compliance
├─────────────────────────────────┤
│ Evaluation & Regression Testing │  → Golden datasets, no silent regressions
├─────────────────────────────────┤
│ Observability & Cost Tracking   │  → OpenTelemetry, Prometheus, alerting
├─────────────────────────────────┤
│ Kubernetes Deployment           │  → Private cloud, multi-tenant
└─────────────────────────────────┘
```

**Key Decisions:**
1. **LangGraph** - Explicit control, determinism, testability
2. **Multi-tier memory** - Session + long-term in vector DB
3. **Cost tracking** - Per-request logging, tenant quotas
4. **Evaluation-first** - Golden datasets before deployment
5. **Security by design** - Guardrails at every layer

### Q28: How would you scale this platform to support 3 different agent types (SOC, pentest, code review)?

**Answer:**

```python
# Shared platform substrate:
class AgentPlatform:
    def __init__(self):
        self.orchestrator = LangGraphOrchestrator()
        self.model_router = ModelRouter()
        self.guardrails = GuardrailsLayer()
        self.evaluator = EvaluationHarness()
        self.vector_db = QdrantVectorDB()

# Domain-specific extensions:
class SOCAgent(BaseAgent):
    TOOLS = ["query_siem", "check_geolocation", "query_threat_intel"]
    GOLDEN_DATASET = "soc_golden.json"
    
    def critique(self, findings):
        # SOC: escalate if critical or unknown
        return severity > "high"

class PentestAgent(BaseAgent):
    TOOLS = ["nmap_scan", "exploit_test", "post_exploitation"]
    GOLDEN_DATASET = "pentest_golden.json"
    
    def critique(self, findings):
        # Pentest: have we found all vulnerabilities?
        return confidence < 0.95

class CodeReviewAgent(BaseAgent):
    TOOLS = ["query_gitlab", "run_sast", "check_dependencies"]
    GOLDEN_DATASET = "code_review_golden.json"
    
    def critique(self, findings):
        # Code review: any exploitable vulns?
        return has_critical_vulns

# All agents share:
- Model routing, cost tracking, guardrails, evaluation
- Infrastructure: K8s, observability, audit logging

# Each agent customizes:
- Domain-specific tools
- Golden datasets
- Critique logic
```

### Q29: How would you implement blue-green deployment for zero-downtime updates?

**Answer:**

```yaml
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-blue
spec:
  replicas: 5
  selector:
    matchLabels:
      version: blue

---

# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-green
spec:
  replicas: 0  # Start with 0 replicas
  selector:
    matchLabels:
      version: green

---

# Service switches traffic
apiVersion: v1
kind: Service
metadata:
  name: agent-platform
spec:
  selector:
    version: blue  # Currently points to blue
  ports:
  - port: 5000

# Deployment steps:
# 1. kubectl apply -f agent-green.yaml  # Create green with 0 replicas
# 2. kubectl scale deployment agent-green --replicas=5  # Warm up
# 3. Run evaluation tests on green
# 4. kubectl patch service agent-platform -p '{"spec":{"selector":{"version":"green"}}}'  # Switch traffic
# 5. Monitor for errors
# 6. kubectl scale deployment agent-blue --replicas=0  # Decommission blue
```

### Q30: How would you ensure reproducibility across agent runs?

**Answer:**

```python
class ReproducibilityManager:
    """Ensure deterministic agent behavior"""
    
    def invoke_reproducibly(self, task: str, random_seed: int = 42):
        """Set all randomness to seed"""
        
        # Pin everything that affects randomness
        import random
        import numpy as np
        import torch
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Disable non-deterministic algorithms
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        
        # Use same model version
        model = load_model("meta-llama/Llama-2-70b-chat-hf", revision="main")
        
        # Use same temperature
        response = model.invoke(task, temperature=0.0)  # Greedy decoding
        
        return response

# Test reproducibility:
result1 = agent.invoke_reproducibly(task, seed=42)
result2 = agent.invoke_reproducibly(task, seed=42)

assert result1['findings'] == result2['findings']
```

### Q31: Design error handling and recovery for agent failures.

**Answer:**

```python
class AgentErrorHandler:
    """Robust error handling"""
    
    def invoke_with_recovery(self, task: str) -> dict:
        """Try with recovery strategies"""
        
        strategies = [
            self._try_with_main_model,
            self._try_with_fallback_model,
            self._try_with_simpler_task,
            self._try_with_human_intervention
        ]
        
        for strategy in strategies:
            try:
                result = strategy(task)
                return {"result": result, "strategy": strategy.__name__}
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        # All strategies exhausted
        return {"result": None, "error": "All recovery strategies exhausted"}
    
    def _try_with_main_model(self, task: str):
        """Try with primary model"""
        return agent.invoke(task, model="llama-2-70b")
    
    def _try_with_fallback_model(self, task: str):
        """Fallback to cheaper/smaller model"""
        return agent.invoke(task, model="mistral-7b")
    
    def _try_with_simpler_task(self, task: str):
        """Break task into simpler subtasks"""
        subtasks = decompose_task(task)
        results = [agent.invoke(subtask) for subtask in subtasks]
        return combine_results(results)
    
    def _try_with_human_intervention(self, task: str):
        """Ask human for help"""
        raise HumanInterventionRequired(f"Agent failed on: {task}")
```

---

## Section 5: Interview Red-Flags & Common Pitfalls

### Q31-Q40: Common Mistakes

1. **Not mentioning cost** - "We'll just use GPT-4" → ignores budget constraints
2. **Ignoring security** - "Agents are fun to build" → misses guardrails requirement
3. **No evaluation strategy** - "We'll test manually" → silent regressions inevitable
4. **Single point of failure** - No fallback models, no redundancy
5. **Not thinking about scale** - "Works for 1 user, what about 1000?"
6. **Ignoring observability** - Can't debug what you can't see
7. **Not separating concerns** - Orchestration logic mixed with business logic
8. **Treating evaluation as afterthought** - "We'll evaluate after building"
9. **Not thinking about failures** - "What if vLLM goes down?"
10. **Over-engineering** - "We need every Kubernetes feature" → maintenance nightmare

---

## Final Tips for Interview

1. **Draw diagrams** - Show architecture clearly
2. **Question assumptions** - "Can you tell me more about the threat model?"
3. **Estimate trade-offs** - "LangGraph is more complex, but gives us determinism"
4. **Mention scale** - "At 1K req/day, we need..."
5. **Show systems thinking** - How do components interact?
6. **Admit uncertainty** - "I'm not sure about X, let me think..."
7. **Ask for feedback** - "Does this align with Presight's needs?"

---

## Resource Links

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NIST AI RMF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.RMF.1.0.pdf)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OpenTelemetry](https://opentelemetry.io/)

Good luck! 🚀
