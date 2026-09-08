# Agent Orchestration Frameworks: LangGraph, CrewAI, AutoGen

## Table of Contents
- [Agent Orchestration Frameworks: LangGraph, CrewAI, AutoGen](#agent-orchestration-frameworks-langgraph-crewai-autogen)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Framework Comparison](#framework-comparison)
  - [1. LangGraph (Recommended)](#1-langgraph-recommended)
    - [Why LangGraph for Presight?](#why-langgraph-for-presight)
    - [Core Concepts](#core-concepts)
      - [1. **State Graph**](#1-state-graph)
      - [2. **Conditional Routing**](#2-conditional-routing)
      - [3. **State Reducer / Aggregator**](#3-state-reducer--aggregator)
      - [4. **Persistence \& Snapshots**](#4-persistence--snapshots)
      - [5. **Streaming**](#5-streaming)
    - [LangGraph Pattern: Planner-Executor-Critic](#langgraph-pattern-planner-executor-critic)
    - [LangGraph Advanced: Human-in-the-Loop](#langgraph-advanced-human-in-the-loop)
    - [Observability \& Tracing](#observability--tracing)
  - [2. CrewAI (Alternative)](#2-crewai-alternative)
    - [When to Consider CrewAI](#when-to-consider-crewai)
    - [CrewAI Pros \& Cons](#crewai-pros--cons)
  - [3. AutoGen (Alternative)](#3-autogen-alternative)
    - [When to Consider AutoGen](#when-to-consider-autogen)
    - [AutoGen Pros \& Cons](#autogen-pros--cons)
  - [Design Decision: LangGraph Recommendation](#design-decision-langgraph-recommendation)
    - [Why LangGraph Wins for Presight](#why-langgraph-wins-for-presight)
  - [Implementation Roadmap](#implementation-roadmap)
    - [Month 1: Core Graph](#month-1-core-graph)
    - [Month 2: Scale \& Guardrails](#month-2-scale--guardrails)
    - [Month 3: Advanced Patterns](#month-3-advanced-patterns)
    - [Months 4-6: Production Hardening](#months-4-6-production-hardening)
  - [Key Interview Questions](#key-interview-questions)
  - [Resources](#resources)
  - [Next: Model Serving \& Routing](#next-model-serving--routing)

---

## Overview

Agent orchestration is the **core backbone** of the Presight platform. It coordinates multiple agents, tools, and LLM interactions in a reliable, deterministic way. The framework choice impacts:
- Complexity of agent logic
- Debugging and observability
- Cost and latency
- Reliability and failure handling
- Extensibility for three security domains

---

## Framework Comparison

| Aspect | LangGraph | CrewAI | AutoGen |
|--------|-----------|--------|---------|
| **Level** | Low-level graph abstraction | High-level multi-agent framework | High-level multi-agent framework |
| **Complexity** | More verbose, more control | Less code, opinionated | Less code, opinionated |
| **Flexibility** | Extremely flexible (any graph) | Good (role-based agents) | Good (conversational agents) |
| **Tool Use** | First-class support | Built-in tool calling | Built-in tool calling |
| **Memory** | You manage explicitly | Built-in memory management | Built-in memory management |
| **Production Ready** | Yes (LangChain ecosystem) | Yes (Crew AI Inc) | Yes (Microsoft Research) |
| **Learning Curve** | Steeper (graph thinking) | Moderate (role-based) | Moderate (conversation-based) |
| **Cost Control** | Excellent (explicit graph control) | Good (built-in token tracking) | Good (built-in tracking) |
| **Testing** | Excellent (deterministic graphs) | Good (agent replay) | Moderate |
| **Presight Fit** | **RECOMMENDED** | Good alternative | Acceptable |

---

## 1. LangGraph (Recommended)

### Why LangGraph for Presight?

**LangGraph is the recommended choice** because:
1. **Explicit control:** Every state transition is visible and testable
2. **Determinism:** Same input → same execution path (critical for evaluation)
3. **Flexibility:** Can express planner-executor-critic patterns natively
4. **Cost visibility:** Loop control enables intelligent caching/fallback routing
5. **Debugging:** State snapshots at every node for troubleshooting
6. **Ecosystem:** Integrates with LangChain, OpenTelemetry, LangSmith

### Core Concepts

#### 1. **State Graph**
A directed acyclic graph where nodes are functions and edges are state transitions.

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    """Shared state across all agent nodes"""
    input: str
    plan: str
    tools_to_call: list[str]
    execution_results: dict
    final_output: str
    cost_usd: float
    latency_ms: float

# Create graph
graph = StateGraph(AgentState)

# Add nodes (agent functions)
graph.add_node("planner", plan_agent)
graph.add_node("executor", execute_tools)
graph.add_node("critic", critic_agent)

# Add edges (conditional routing)
graph.add_edge("planner", "executor")
graph.add_conditional_edges(
    "executor",
    should_continue,  # returns next node or END
    {
        "critic": "critic",
        "planner": "planner",  # loop back if needed
        "end": END
    }
)

agent = graph.compile()
```

#### 2. **Conditional Routing**
Route execution based on state. Critical for:
- Plan validation (is the plan sound?)
- Tool execution loops (do we need more tools?)
- Fallback mechanisms (should we switch models?)

```python
def should_continue(state: AgentState) -> str:
    """Decide next step based on execution results"""
    if state['execution_results'].get('error'):
        # Try fallback model
        return "fallback_executor"
    elif state['critic']['decision'] == "revise_plan":
        return "planner"
    else:
        return "end"
```

#### 3. **State Reducer / Aggregator**
Combine outputs from parallel nodes (e.g., multiple tools called simultaneously).

```python
class AgentState(TypedDict):
    messages: Annotated[list, "append"]  # Append mode for conversations
    tools_called: Annotated[list, "extend"]  # Extend mode for lists
    total_cost: Annotated[float, "add"]  # Sum mode for costs
```

#### 4. **Persistence & Snapshots**
Critical for debugging and replay.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("sqlite://agent.db")
agent = graph.compile(checkpointer=checkpointer)

# Later: retrieve from checkpoint
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke(input_state, config=config)

# Query checkpoints
for checkpoint in checkpointer.list("user-123"):
    print(checkpoint)  # Full state at that point
```

#### 5. **Streaming**
Stream intermediate results as the agent runs (important for cost tracking).

```python
config = {"configurable": {"thread_id": "user-123"}}

for event in agent.stream(input_state, config=config, stream_mode="values"):
    print(f"Node: {event['node']}, Cost: ${event.get('cost_usd', 0)}")
    # Real-time cost visibility
```

### LangGraph Pattern: Planner-Executor-Critic

This is the **core pattern for Presight SOC/Pentest/CodeReview agents**.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class SecurityAgentState(TypedDict):
    """State for security agent (SOC/Pentest/CodeReview)"""
    task: str  # Original request
    plan: str  # Multi-step plan
    plan_confidence: float  # 0.0-1.0 confidence in plan
    current_step: int
    execution_log: list[dict]  # [{"tool": "...", "result": "..."}]
    findings: list[dict]  # [{"severity": "high", "finding": "..."}]
    final_report: str
    total_cost: float
    total_latency_ms: float

# Node: Planner
def planner_node(state: SecurityAgentState) -> SecurityAgentState:
    """LLM creates detailed plan for security analysis"""
    prompt = f"""
    You are a security analyst. Plan steps to investigate:
    {state['task']}
    
    Return:
    - Step 1: [tool] - [description]
    - Step 2: [tool] - [description]
    ...
    """
    response = llm.invoke(prompt)
    state['plan'] = response.content
    state['plan_confidence'] = extract_confidence(response)
    return state

# Node: Executor
async def executor_node(state: SecurityAgentState) -> SecurityAgentState:
    """Execute tools based on plan"""
    steps = parse_plan(state['plan'])
    
    for step_idx, step in enumerate(steps):
        tool_name, tool_args = extract_tool_call(step)
        
        # Call tool (SIEM, Tenable, etc.)
        result = await tools[tool_name].invoke(tool_args)
        
        state['execution_log'].append({
            "step": step_idx,
            "tool": tool_name,
            "args": tool_args,
            "result": result,
            "timestamp": time.time()
        })
    
    return state

# Node: Critic
def critic_node(state: SecurityAgentState) -> SecurityAgentState:
    """LLM reviews findings and decides next step"""
    prompt = f"""
    Review security investigation:
    Task: {state['task']}
    Plan: {state['plan']}
    Findings: {state['execution_log']}
    
    Decision: Should we (1) Revise plan and investigate more, (2) Generate report?
    """
    decision = llm.invoke(prompt).content
    
    # Extract findings
    state['findings'] = extract_findings(state['execution_log'])
    
    # Generate report if enough data
    if "generate report" in decision.lower():
        state['final_report'] = generate_security_report(state['findings'])
    
    return state

# Conditional edge: should we loop?
def should_continue(state: SecurityAgentState) -> str:
    if state['plan_confidence'] < 0.5:
        return "planner"  # Replan
    elif len(state['findings']) < 3 and state['total_cost'] < 10:
        return "executor"  # Run more tools
    else:
        return END  # Report is ready

# Build graph
graph = StateGraph(SecurityAgentState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("critic", critic_node)

graph.add_edge("planner", "executor")
graph.add_edge("executor", "critic")
graph.add_conditional_edges(
    "critic",
    should_continue,
    {"planner": "planner", "executor": "executor", "end": END}
)

graph.set_entry_point("planner")
agent = graph.compile()

# Run with cost tracking
result = agent.invoke({
    "task": "Investigate suspicious login from IP 192.168.1.1",
    "plan": "",
    "execution_log": [],
    "findings": [],
    "final_report": "",
    "total_cost": 0.0,
    "total_latency_ms": 0.0
})

print(f"Report:\n{result['final_report']}")
print(f"Cost: ${result['total_cost']}")
```

### LangGraph Advanced: Human-in-the-Loop

Critical for security agents (human approval of sensitive actions).

```python
def interrupt_before_tool_use(state: SecurityAgentState) -> SecurityAgentState:
    """
    Interrupt execution to get human approval.
    Useful for: deleting data, modifying firewall rules, etc.
    """
    # This would be handled at streaming/API level
    raise NodeInterrupt(
        "Requires approval to execute: shell_exec(rm /data)"
    )

# In your API:
def execute_agent_with_approval(request):
    for event in agent.stream(state):
        if isinstance(event, NodeInterrupt):
            # Return event to frontend, wait for approval
            return {"status": "waiting_approval", "action": event.message}
        else:
            # Continue execution
            pass
```

### Observability & Tracing

```python
from langchain.callbacks import TracingCallbackHandler

# Trace to LangSmith
handler = TracingCallbackHandler(
    project_name="presight-agents",
    tags=["security", "soc"]
)

result = agent.invoke(
    state,
    callbacks=[handler]
)

# Traces available in LangSmith UI
# - Each node execution
# - Each LLM call (tokens, cost)
# - Each tool call
```

---

## 2. CrewAI (Alternative)

### When to Consider CrewAI

CrewAI is good if you prefer a **higher-level abstraction** and don't need fine-grained state control.

```python
from crewai import Agent, Task, Crew

# Define agents with roles
soc_agent = Agent(
    role='Security Operations Center Analyst',
    goal='Investigate security incidents',
    backstory='Expert in SIEM analysis...',
    tools=[siem_tool, tenable_tool],
    llm=claude_model
)

threat_intel_agent = Agent(
    role='Threat Intelligence Analyst',
    goal='Provide context on threats',
    backstory='Expert in MITRE ATT&CK...',
    tools=[mitre_tool, d3fend_tool],
    llm=claude_model
)

# Define tasks
investigation_task = Task(
    description='Investigate the incident: {incident_id}',
    agent=soc_agent,
    expected_output='Detailed incident report'
)

# Crew orchestrates agent collaboration
crew = Crew(
    agents=[soc_agent, threat_intel_agent],
    tasks=[investigation_task],
    verbose=True
)

result = crew.kickoff(inputs={'incident_id': '12345'})
```

### CrewAI Pros & Cons

**Pros:**
- Higher-level, less boilerplate
- Built-in role-based reasoning
- Good for multi-agent collaboration

**Cons:**
- Less control over state flow
- Harder to implement custom loops
- Not ideal for cost-optimized routing

---

## 3. AutoGen (Alternative)

### When to Consider AutoGen

AutoGen is good for **conversational agent patterns** with human-in-the-loop.

```python
from autogen import AssistantAgent, UserProxyAgent

soc_assistant = AssistantAgent(
    name="soc_analyst",
    system_message="You are a security analyst...",
    llm_config={"config_list": [{"model": "claude-opus", "api_key": "..."}]}
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",  # Always ask human
    code_execution_config={"work_dir": "sandbox", "use_docker": True}
)

# Start conversation
user_proxy.initiate_chat(
    soc_assistant,
    message="Investigate incident #12345"
)
```

### AutoGen Pros & Cons

**Pros:**
- Excellent for human collaboration
- Built-in code execution
- Natural conversation flow

**Cons:**
- Harder to deterministic testing
- Less suitable for security (needs explicit tool authorization)
- Not ideal for batch processing

---

## Design Decision: LangGraph Recommendation

### Why LangGraph Wins for Presight

1. **Cost Control:** Conditional routing enables intelligent model switching
   ```python
   def should_escalate_model(state):
       if state['complexity_score'] > 0.8:
           return "gpt4"  # Expensive but accurate
       else:
           return "llama2"  # Cheap alternative
   ```

2. **Evaluation & Regression Testing:** Deterministic state snapshots
   ```python
   checkpoint = agent.get_state(thread_id="test-123", checkpoint_id="abc")
   assert checkpoint['findings'] == expected_findings
   ```

3. **Tool Authorization:** Explicit node for tool-use validation
   ```python
   def validate_tool_use(state):
       """Ensure tool call respects security policy"""
       for tool_call in state['tools_to_call']:
           if not is_authorized(tool_call):
               raise SecurityViolation(f"Unauthorized: {tool_call}")
       return state
   ```

4. **Observability:** Stream every state transition
   ```python
   for event in agent.stream(state):
       log_metrics(event)  # Cost, latency, safety events
   ```

5. **Multi-Tenant Safety:** Isolate agent executions
   ```python
   config = {"configurable": {"user_id": "...", "tenant_id": "..."}}
   result = agent.invoke(state, config=config)  # Isolated state
   ```

---

## Implementation Roadmap

### Month 1: Core Graph
- [ ] Build planner-executor-critic graph
- [ ] Implement tool registry
- [ ] Add basic state persistence
- [ ] Deploy "hello agent" demo

### Month 2: Scale & Guardrails
- [ ] Add tool authorization layer
- [ ] Implement prompt injection filters
- [ ] Add cost tracking to every node
- [ ] Onboard SOC agent (first tenant)

### Month 3: Advanced Patterns
- [ ] Parallel tool execution
- [ ] Human-in-the-loop approval flows
- [ ] Fallback & retry mechanisms
- [ ] Evaluation harness integration

### Months 4-6: Production Hardening
- [ ] Add distributed tracing
- [ ] Implement SLOs and alerting
- [ ] Red-team the graph
- [ ] Scale to 3 agents

---

## Key Interview Questions

1. **How would you implement a planner-executor-critic pattern in LangGraph?**
   - Answer should cover: StateGraph, conditional_edges, state reducers

2. **How do you handle tool-use authorization?**
   - Answer: Separate validation node before tool execution

3. **How would you implement cost-aware model routing?**
   - Answer: Conditional edges checking cost budget and model difficulty

4. **How do you test agent determinism?**
   - Answer: Checkpoint snapshots, compare against golden datasets

5. **How would you handle agent loops (replanning)?**
   - Answer: Conditional edges with loop detection (max retries)

---

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [CrewAI Docs](https://docs.crewai.com/)
- [AutoGen Docs](https://microsoft.github.io/autogen/)
- [LangSmith Tracing](https://smith.langchain.com/)

---

## Next: Model Serving & Routing

See `02_MODEL_SERVING_ROUTING.md` for how to integrate LLMs into your LangGraph agents.
