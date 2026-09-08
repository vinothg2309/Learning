- [LangChain \& LangGraph](#langchain--langgraph)
  - [Advanced Questions \& Answers](#advanced-questions--answers)
- [Part 1 — LangChain](#part-1--langchain)
  - [📚 Concept-Based Questions (20)](#-concept-based-questions-20)
    - [Q1. What is LCEL and how does the pipe (|) operator work internally?](#q1-what-is-lcel-and-how-does-the-pipe--operator-work-internally)
    - [Q2. Explain RunnableSequence, RunnableParallel, and RunnablePassthrough.](#q2-explain-runnablesequence-runnableparallel-and-runnablepassthrough)
    - [Q3. How does LangChain memory work? Compare ConversationBufferMemory, ConversationSummaryMemory, and ConversationTokenBufferMemory.](#q3-how-does-langchain-memory-work-compare-conversationbuffermemory-conversationsummarymemory-and-conversationtokenbuffermemory)
    - [Q4. What is RunnableWithMessageHistory and how does it replace legacy memory?](#q4-what-is-runnablewithmessagehistory-and-how-does-it-replace-legacy-memory)
    - [Q5. How do you build a custom Tool? What do args\_schema and return\_direct do?](#q5-how-do-you-build-a-custom-tool-what-do-args_schema-and-return_direct-do)
    - [Q6. Compare PydanticOutputParser, JsonOutputParser, and StrOutputParser.](#q6-compare-pydanticoutputparser-jsonoutputparser-and-stroutputparser)
    - [Q7. Explain stuff, map\_reduce, refine, and map\_rerank document chains.](#q7-explain-stuff-map_reduce-refine-and-map_rerank-document-chains)
    - [Q8. How does the Retriever interface differ from VectorStore?](#q8-how-does-the-retriever-interface-differ-from-vectorstore)
    - [Q9. What are LangChain callbacks and how do you implement a custom handler?](#q9-what-are-langchain-callbacks-and-how-do-you-implement-a-custom-handler)
    - [Q10. Compare ChatPromptTemplate vs PromptTemplate. How do MessagesPlaceholder and SystemMessage interact?](#q10-compare-chatprompttemplate-vs-prompttemplate-how-do-messagesplaceholder-and-systemmessage-interact)
    - [Q11. How does streaming work? Difference between .stream() and .astream\_events()?](#q11-how-does-streaming-work-difference-between-stream-and-astream_events)
    - [Q12. How do vector stores integrate? Explain similarity\_search vs max\_marginal\_relevance\_search.](#q12-how-do-vector-stores-integrate-explain-similarity_search-vs-max_marginal_relevance_search)
    - [Q13. What are .bind(), .with\_config(), .with\_retry(), and .with\_fallbacks()?](#q13-what-are-bind-with_config-with_retry-and-with_fallbacks)
    - [Q14. How do you implement a RAG chain with context compression in LCEL?](#q14-how-do-you-implement-a-rag-chain-with-context-compression-in-lcel)
    - [Q15. What is LangSmith and how does it integrate for tracing and evaluation?](#q15-what-is-langsmith-and-how-does-it-integrate-for-tracing-and-evaluation)
    - [Q16. How does the agent loop work internally (Plan → Act → Observe)?](#q16-how-does-the-agent-loop-work-internally-plan--act--observe)
    - [Q17. Difference between create\_react\_agent, create\_openai\_tools\_agent, create\_structured\_chat\_agent?](#q17-difference-between-create_react_agent-create_openai_tools_agent-create_structured_chat_agent)
    - [Q18. What is a Multi-Action Agent?](#q18-what-is-a-multi-action-agent)
    - [Q19. How do you handle rate limits, token limits, and cost in production?](#q19-how-do-you-handle-rate-limits-token-limits-and-cost-in-production)
    - [Q20. Explain the document loader ecosystem and how TextSplitters interact with loaders.](#q20-explain-the-document-loader-ecosystem-and-how-textsplitters-interact-with-loaders)
  - [🎯 Scenario-Based Questions (10)](#-scenario-based-questions-10)
    - [Q1. RAG bot answers are irrelevant on a 50K-page manual. How do you fix it?](#q1-rag-bot-answers-are-irrelevant-on-a-50k-page-manual-how-do-you-fix-it)
    - [Q2. Multi-part questions fail ('Compare A and B, recommend cheaper'). Solution?](#q2-multi-part-questions-fail-compare-a-and-b-recommend-cheaper-solution)
    - [Q3. Agent with 8 tools keeps picking the wrong tool. How to fix?](#q3-agent-with-8-tools-keeps-picking-the-wrong-tool-how-to-fix)
    - [Q4. 500 concurrent users with LLM + vector retrieval + DB lookup. Architecture?](#q4-500-concurrent-users-with-llm--vector-retrieval--db-lookup-architecture)
    - [Q5. Route billing vs technical support to different RAG pipelines. Design in LCEL?](#q5-route-billing-vs-technical-support-to-different-rag-pipelines-design-in-lcel)
    - [Q6. Every response must include source citations with page numbers. Implementation?](#q6-every-response-must-include-source-citations-with-page-numbers-implementation)
    - [Q7. Chain fails intermittently with timeout errors in production. Resilience strategy?](#q7-chain-fails-intermittently-with-timeout-errors-in-production-resilience-strategy)
    - [Q8. Add conversation memory to existing stateless RAG chain without rewriting it.](#q8-add-conversation-memory-to-existing-stateless-rag-chain-without-rewriting-it)
    - [Q9. User uploads a PDF, asks questions across multiple turns with memory. Architecture?](#q9-user-uploads-a-pdf-asks-questions-across-multiple-turns-with-memory-architecture)
    - [Q10. Evaluate GPT-4 vs open-source model quality in your RAG pipeline.](#q10-evaluate-gpt-4-vs-open-source-model-quality-in-your-rag-pipeline)
- [Part 2 — LangGraph](#part-2--langgraph)
  - [📚 Concept-Based Questions (20)](#-concept-based-questions-20-1)
    - [Q1. What is LangGraph and how does it differ from LCEL chains?](#q1-what-is-langgraph-and-how-does-it-differ-from-lcel-chains)
    - [Q2. Explain StateGraph, Nodes, Edges, and Conditional Edges.](#q2-explain-stategraph-nodes-edges-and-conditional-edges)
    - [Q3. What is State in LangGraph? How do Annotated fields with reducers work?](#q3-what-is-state-in-langgraph-how-do-annotated-fields-with-reducers-work)
    - [Q4. What is MessagesState and when would you use it vs custom state?](#q4-what-is-messagesstate-and-when-would-you-use-it-vs-custom-state)
    - [Q5. How do Conditional Edges work? What does tools\_condition simplify?](#q5-how-do-conditional-edges-work-what-does-tools_condition-simplify)
  - [🔰 **BEGINNER'S GUIDE: ToolNode \& tools\_condition**](#-beginners-guide-toolnode--tools_condition)
    - [**What are they?**](#what-are-they)
    - [**Step-by-Step: How They Work Together**](#step-by-step-how-they-work-together)
    - [**Example 1: Simple Agent with ToolNode**](#example-1-simple-agent-with-toolnode)
    - [**What Happens Inside?**](#what-happens-inside)
    - [**Example 2: Multiple Tools**](#example-2-multiple-tools)
    - [**Visual Breakdown**](#visual-breakdown)
    - [**Complete Example: Weather Agent**](#complete-example-weather-agent)
    - [**Key Takeaways**](#key-takeaways)
  - [🔰 **Command(resume) vs Command(update)**](#-commandresume-vs-commandupdate)
    - [**Short Answer:**](#short-answer)
    - [**Code Examples:**](#code-examples)
  - [🔰 **How does interrupt know where to resume?**](#-how-does-interrupt-know-where-to-resume)
    - [Q6. Explain checkpointing: MemorySaver vs SqliteSaver vs PostgresSaver.](#q6-explain-checkpointing-memorysaver-vs-sqlitesaver-vs-postgressaver)
    - [Q7. How does human-in-the-loop work? Explain interrupt\_before and interrupt\_after.](#q7-how-does-human-in-the-loop-work-explain-interrupt_before-and-interrupt_after)
    - [Q8. What are dynamic interrupts and how do they differ from static?](#q8-what-are-dynamic-interrupts-and-how-do-they-differ-from-static)
    - [Q9. Explain the Supervisor pattern in multi-agent systems.](#q9-explain-the-supervisor-pattern-in-multi-agent-systems)
    - [Q10. What are Subgraphs and why use them?](#q10-what-are-subgraphs-and-why-use-them)
    - [Q11. How does LangGraph handle error recovery?](#q11-how-does-langgraph-handle-error-recovery)
    - [Q12. What is time-travel debugging?](#q12-what-is-time-travel-debugging)
    - [Q13. Implement a ReAct agent from scratch in LangGraph (without prebuilt).](#q13-implement-a-react-agent-from-scratch-in-langgraph-without-prebuilt)
    - [Q14. How does Send() work for fan-out patterns?](#q14-how-does-send-work-for-fan-out-patterns)
  - [🔰 **Send() vs State Variable**](#-send-vs-state-variable)
  - [🔰 **Send() with Multiple Nodes - How Consolidation Works**](#-send-with-multiple-nodes---how-consolidation-works)
  - [🔰 **Why is it called a "Reducer"?**](#-why-is-it-called-a-reducer)
    - [Q15. Difference between graph.invoke(), graph.stream(), graph.astream\_events()?](#q15-difference-between-graphinvoke-graphstream-graphastream_events)
    - [Q16. How do you manage shared vs private state in multi-agent systems?](#q16-how-do-you-manage-shared-vs-private-state-in-multi-agent-systems)
    - [Q17. Key differences between prebuilt create\_react\_agent and custom graph?](#q17-key-differences-between-prebuilt-create_react_agent-and-custom-graph)
    - [Q18. How does LangGraph ensure thread safety in multi-user deployments?](#q18-how-does-langgraph-ensure-thread-safety-in-multi-user-deployments)
    - [Q19. How do you implement guardrails?](#q19-how-do-you-implement-guardrails)
    - [Q20. What is LangGraph Platform / Cloud vs self-hosting?](#q20-what-is-langgraph-platform--cloud-vs-self-hosting)
  - [🎯 Scenario-Based Questions (10)](#-scenario-based-questions-10-1)
    - [Q1. Deep research agent: search → evaluate → iterate until enough evidence. Design?](#q1-deep-research-agent-search--evaluate--iterate-until-enough-evidence-design)
    - [Q2. Trading system: trades \>$50K need human approval. Design HITL workflow.](#q2-trading-system-trades-50k-need-human-approval-design-hitl-workflow)
    - [Q3. Supervisor coordinating researcher, coder, writer agents. Design?](#q3-supervisor-coordinating-researcher-coder-writer-agents-design)
    - [Q4. Agent enters infinite loops between LLM and tool nodes. Diagnose and fix.](#q4-agent-enters-infinite-loops-between-llm-and-tool-nodes-diagnose-and-fix)
    - [Q5. 1,000 concurrent users, each with conversation thread. Architect state management.](#q5-1000-concurrent-users-each-with-conversation-thread-architect-state-management)
    - [Q6. Process 20 documents in parallel, then synthesise. Fan-out/fan-in?](#q6-process-20-documents-in-parallel-then-synthesise-fan-outfan-in)
    - [Q7. Add rollback feature: undo wrong agent action, retry from previous state.](#q7-add-rollback-feature-undo-wrong-agent-action-retry-from-previous-state)
    - [Q8. External API has 30% failure rate and 10 req/min rate limit. Resilience strategy.](#q8-external-api-has-30-failure-rate-and-10-reqmin-rate-limit-resilience-strategy)
    - [Q9. Coding assistant: LLM writes code, executes in sandbox, iterates if tests fail. Design.](#q9-coding-assistant-llm-writes-code-executes-in-sandbox-iterates-if-tests-fail-design)
    - [Q10. Full auditability required: every LLM call, tool execution, state transition logged. How?](#q10-full-auditability-required-every-llm-call-tool-execution-state-transition-logged-how)
- [Part 4 — Agent Design Patterns](#part-4--agent-design-patterns)
    - [Q: When to use Multi-Agent vs Single Agent?](#q-when-to-use-multi-agent-vs-single-agent)
    - [Q: What are the callback options in LangGraph?](#q-what-are-the-callback-options-in-langgraph)
    - [Q: How to design fault-tolerant and long-running agent workflows?](#q-how-to-design-fault-tolerant-and-long-running-agent-workflows)
    - [Q: How to call MCP (Model Context Protocol) from LangChain and LangGraph?](#q-how-to-call-mcp-model-context-protocol-from-langchain-and-langgraph)
    - [Q: How to effectively handle tool selection, fallback logic, execution and retries within Agent workflow?](#q-how-to-effectively-handle-tool-selection-fallback-logic-execution-and-retries-within-agent-workflow)
    - [Q: How to enable modularity when building agents with multiple tools or context?](#q-how-to-enable-modularity-when-building-agents-with-multiple-tools-or-context)
  - [Q: Why must you use an Async checkpointer for concurrent LangGraph requests, and how does it work?](#q-why-must-you-use-an-async-checkpointer-for-concurrent-langgraph-requests-and-how-does-it-work)


# LangChain & LangGraph
## Advanced Questions & Answers

---

# Part 1 — LangChain

## 📚 Concept-Based Questions (20)

### Q1. What is LCEL and how does the pipe (|) operator work internally?

**Answer:**
LCEL (LangChain Expression Language) is a declarative, composable DSL for building chains. The pipe operator leverages Python's `__or__` dunder on Runnable objects. Writing `prompt | llm | parser` creates a RunnableSequence supporting `invoke()`, `stream()`, `batch()` and async variants from a single declaration.

**Sample Code Examples:**

**1. Basic LCEL Chain**
Compose components using the pipe operator to create a RunnableSequence.
```python
chain = prompt | llm | parser
result = chain.invoke({"topic": "programming"})
```

**2. Streaming**
Stream tokens in real-time as they're generated.
```python
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

**3. Batch Processing**
Process multiple inputs efficiently in a single call.
```python
results = chain.batch([
    {"topic": "databases"},
    {"topic": "cloud computing"}
])
```

**4. Async Operations**
All Runnables support async variants for concurrent execution.
```python
result = await chain.ainvoke({"topic": "async programming"})
```

**5. RunnableParallel**
Execute multiple branches concurrently and merge outputs into a dict.
```
Input: {"question": "What is RAG?"}
           |
    RunnableParallel
      /           \
"context"      "question"
retriever      Passthrough
     |              |
format_docs    (unchanged)
      \           /
       Output: {"context": "...", "question": "What is RAG?"}
                    |
                  prompt | llm | parser
```
```python
chain_with_parallel = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | parser
)
```

**6. RunnablePassthrough**
Forward input unchanged, useful for carrying original data alongside transformations.
```python
chain = RunnableParallel({
    "original": RunnablePassthrough(),
    "transformed": transform_function
})
```

**7. RunnableLambda**
Wrap custom functions to use in LCEL chains.
```python
chain = (
    RunnableLambda(lambda x: x["input"].upper())
    | prompt
    | llm
)
```

**8. RunnableBranch**
Route inputs to different chains based on conditions.
```python
branch = RunnableBranch(
    (lambda x: "code" in x["query"], code_chain),
    (lambda x: "math" in x["query"], math_chain),
    general_chain  # default
)
```

**9. Config & Callbacks**
Pass runtime configuration, callbacks, and metadata.
```python
result = chain.invoke(
    {"topic": "testing"},
    config={
        "callbacks": [custom_handler],
        "tags": ["production"],
        "metadata": {"user_id": "123"}
    }
)
```

---

### Q2. Explain RunnableSequence, RunnableParallel, and RunnablePassthrough.

**Answer:**
- **RunnableSequence**: Chains components in series (A→B→C)
- **RunnableParallel**: Runs branches simultaneously and merges outputs into a dict
- **RunnablePassthrough**: Forwards input unchanged — useful for carrying the original query alongside transformed branches (e.g., passing both query and retrieved docs)

---

### Q3. How does LangChain memory work? Compare ConversationBufferMemory, ConversationSummaryMemory, and ConversationTokenBufferMemory.

**Answer:**
- **BufferMemory**: Stores the full conversation — simple but unbounded token cost
- **SummaryMemory**: Summarises older turns via an LLM — saves tokens but loses detail
- **TokenBufferMemory**: Keeps the most recent N tokens, giving a hard ceiling

Choice depends on cost tolerance, context window, and recall needs.

---

### Q4. What is RunnableWithMessageHistory and how does it replace legacy memory?

**Answer:**
It wraps an LCEL chain and injects/persists chat history from an external store (Redis, Postgres, in-memory). Unlike legacy memory classes coupled to specific chain types, it works with any Runnable, separating persistence from chain logic. You provide a callable returning `BaseChatMessageHistory` keyed by `session_id`.

**Key Points:**
1. **YES** - Stores messages in HumanMessage/AIMessage format
2. **NO** - Does NOT use mem0 or automatic summarization by default
3. Stores **full conversation history** (unbounded by default)
4. You must manually implement summarization if needed

**Message Format - What gets stored:**
```python
from langchain_core.messages import HumanMessage, AIMessage

# After conversation, history contains:
# [
#   HumanMessage(content="What is RAG?"),
#   AIMessage(content="RAG is Retrieval-Augmented Generation..."),
#   HumanMessage(content="How does it work?"),
#   AIMessage(content="It combines retrieval with generation...")
# ]
```

**Sample Code:**
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# In-memory store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap your chain
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# First turn
chain_with_history.invoke(
    {"question": "What is RAG?"},
    config={"configurable": {"session_id": "user_123"}}
)

# Check what's stored
history = store["user_123"]
print(history.messages)
# [
#   HumanMessage(content="What is RAG?"),
#   AIMessage(content="RAG is...")
# ]

# Second turn - history automatically injected
chain_with_history.invoke(
    {"question": "How does it work?"},
    config={"configurable": {"session_id": "user_123"}}
)

# Now history has both turns
print(history.messages)
# [
#   HumanMessage(content="What is RAG?"),
#   AIMessage(content="RAG is..."),
#   HumanMessage(content="How does it work?"),  ← User message
#   AIMessage(content="It combines...")         ← AI message
# ]
```

**Problem: Unbounded Growth**

`RunnableWithMessageHistory` stores **ALL messages** - it grows forever!

**Solution 1: Manual Trimming**
```python
from langchain_core.messages import trim_messages

def get_session_history_with_trim(session_id: str):
    history = store.get(session_id, ChatMessageHistory())

    # Keep only last 10 messages
    if len(history.messages) > 10:
        history.messages = history.messages[-10:]

    return history
```

**Solution 2: Summarization (Manual)**
```python
from langchain_core.prompts import ChatPromptTemplate

def get_session_history_with_summary(session_id: str):
    history = store.get(session_id, ChatMessageHistory())

    # If too long, summarize old messages
    if len(history.messages) > 20:
        old_messages = history.messages[:-10]  # All except last 10
        recent_messages = history.messages[-10:]  # Keep last 10

        # Summarize old messages
        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize this conversation: {messages}"
        )
        summary = llm.invoke(summary_prompt.format(messages=old_messages))

        # Replace with: [Summary] + recent messages
        history.messages = [
            AIMessage(content=f"Previous conversation summary: {summary}"),
            *recent_messages
        ]

    return history
```

**Solution 3: Use mem0 (External Library)**
```python
# RunnableWithMessageHistory does NOT include mem0
# You must integrate mem0 separately

from mem0 import Memory

mem0_client = Memory()

def get_session_history_with_mem0(session_id: str):
    # Fetch from mem0 (returns summarized/relevant memories)
    memories = mem0_client.search(
        query="conversation history",
        user_id=session_id
    )

    # Convert mem0 memories to ChatMessageHistory
    history = ChatMessageHistory()
    for memory in memories:
        history.add_ai_message(memory["text"])

    return history
```

**Comparison:**

| Feature | RunnableWithMessageHistory | mem0 Integration | ConversationSummaryMemory (Legacy) |
|---------|---------------------------|------------------|-----------------------------------|
| Format | HumanMessage/AIMessage | Custom | String |
| Storage | Full history (unbounded) | Summarized/relevant | Summarized |
| Automatic summarization | ❌ No | ✅ Yes | ✅ Yes |
| Manual implementation | ✅ Easy | Requires setup | Built-in (deprecated) |

**Recommended Approach:**

```python
# For short conversations (<50 messages)
# Use RunnableWithMessageHistory with trimming

# For long conversations
# Implement manual summarization OR integrate mem0

# Example: Trim to last 20 messages
from langchain_core.messages import trim_messages

chain_with_trim = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda sid: trim_messages(
        store[sid].messages,
        max_tokens=2000,  # Keep ~2000 tokens of history
        strategy="last"
    )
)
```

---

### Q5. How do you build a custom Tool? What do args_schema and return_direct do?

**Answer:**
Subclass `BaseTool` or use `@tool`, providing name, description, and `_run()`.
- **args_schema** (Pydantic model): Gives the LLM a structured schema of expected arguments
- **return_direct=True**: Tells the agent to return the tool's output directly without further LLM reasoning — useful for deterministic lookups

**Sample Code:**
```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="search query")

class CustomSearchTool(BaseTool):
    name = "search"
    description = "Search for information"
    args_schema = SearchInput
    return_direct = False

    def _run(self, query: str) -> str:
        return f"Results for: {query}"
```

---

### Q6. Compare PydanticOutputParser, JsonOutputParser, and StrOutputParser.

**Answer:**
- **PydanticOutputParser**: Validates and parses into a Pydantic model — strict typing and validation
- **JsonOutputParser**: Extracts JSON without Pydantic overhead
- **StrOutputParser**: Returns the raw string

Each can inject `format_instructions` into the prompt to guide the LLM's output format.

**Sample Code:**

**Method 1: .with_structured_output() (Recommended)**
```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    email: str = Field(description="Email address")

# Direct structured output (uses function calling)
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke("Extract: John is 30 years old, email john@example.com")
# result = Person(name="John", age=30, email="john@example.com")
```

**Method 2: PydanticOutputParser (Legacy)**
```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Person)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract information.\n{format_instructions}"),
    ("user", "{input}")
])

chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser
result = chain.invoke({"input": "John is 30"})
```

**Method 3: JsonOutputParser**
```python
from langchain_core.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()
chain = prompt | llm | json_parser
result = chain.invoke({"input": "Return JSON"})
# result = {"name": "John", "age": 30}
```

**Method 4: StrOutputParser**
```python
from langchain_core.output_parsers import StrOutputParser

str_parser = StrOutputParser()
chain = prompt | llm | str_parser
result = chain.invoke({"input": "Hello"})
# result = "Hello! How can I help?"
```

---

### Q7. Explain stuff, map_reduce, refine, and map_rerank document chains.

**Answer:**
These strategies process the **list of documents retrieved from RAG retrieval** (e.g., top-K chunks from vector store) to generate the final answer.

**RAG Pipeline Flow:**
```
1. Query: "What is Python?"
2. Retriever → VectorStore.similarity_search(k=10)
3. Retrieved docs = [doc1, doc2, ..., doc10]  ← These are the "docs"
4. stuff/map_reduce/refine/map_rerank processes these 10 docs
5. Final Answer
```

**Processing Strategies:**
- **stuff**: Concatenates all 10 docs into one prompt → LLM generates answer. Use when: ≤3-4 docs fit in context.
- **map_reduce**: LLM processes each doc separately (parallel) → combines 10 results. Use when: Many docs.
- **refine**: Starts with doc1, generates answer → refines with doc2 → doc3... sequentially. Use when: Order matters.
- **map_rerank**: LLM generates answer for each doc → scores → picks best. Use when: Want most relevant single answer.

**Sample Code:**
```python
# Step 1: Retrieve docs from vector store
retrieved_docs = retriever.get_relevant_documents("What is Python?")
# retrieved_docs = [Document(...), Document(...), ...]  ← List of 10 chunks

# Step 2: Process retrieved docs with strategy
from langchain.chains.summarize import load_summarize_chain

# Stuff - concatenate all docs
chain = load_summarize_chain(llm, chain_type="stuff")
result = chain.invoke(retrieved_docs)  # Input: list of docs from retrieval

# Map-Reduce - process docs in parallel
chain = load_summarize_chain(llm, chain_type="map_reduce")
result = chain.invoke(retrieved_docs)
```

---

### Q8. How does the Retriever interface differ from VectorStore?

**Answer:**
- **VectorStore**: Handles storage, indexing, and similarity search of embeddings
- **Retriever**: Takes a query string and returns Documents — it may wrap a VectorStore but could also query SQL, APIs, or graphs

Custom Retrievers subclass `BaseRetriever` and implement `_get_relevant_documents()`.

**Sample Code:**
```python
from langchain_core.retrievers import BaseRetriever

# VectorStore to Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Custom Retriever
class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        # Custom logic (SQL, API, etc.)
        return [Document(page_content="...")]
```

---

### Q9. What are LangChain callbacks and how do you implement a custom handler?

**Answer:**
Callbacks are hooks fired during execution. Subclass `BaseCallbackHandler` and override:
- `on_llm_start`
- `on_llm_end`
- `on_chain_start`
- `on_tool_error`, etc.

**Use cases**: logging, token counting, latency tracking, streaming tokens to UI.
Pass via `config={'callbacks': [...]}` or set globally.

**Sample Code:**
```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished: {response}")

# Use it
chain.invoke(
    {"input": "Hello"},
    config={"callbacks": [CustomHandler()]}
)
```

---

### Q10. Compare ChatPromptTemplate vs PromptTemplate. How do MessagesPlaceholder and SystemMessage interact?

**Answer:**
- **PromptTemplate**: Produces a single string
- **ChatPromptTemplate**: Produces a list of typed messages (System, Human, AI)
- **MessagesPlaceholder**: Runtime slot for injecting chat history
- **SystemMessage**: Sets LLM behaviour

Together they enable multi-turn, role-aware prompts.

**Sample Code:**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Invoke with history
chain = prompt | llm
result = chain.invoke({
    "question": "What's next?",
    "chat_history": [("human", "Hi"), ("ai", "Hello!")]
})
```

---

### Q11. How does streaming work? Difference between .stream() and .astream_events()?

**Answer:**
- **`.stream()`**: Yields output chunks (e.g., tokens)
- **`.astream_events()`**: Emits typed events for every lifecycle step across the entire chain — `on_chain_start`, `on_llm_stream`, `on_tool_end`

Ideal for real-time UIs showing intermediate steps, retrieval results, and token-by-token generation.

**Sample Code:**
```python
# .stream() - simple token streaming
for chunk in chain.stream({"input": "Hello"}):
    print(chunk, end="", flush=True)

# .astream_events() - detailed event streaming
async for event in chain.astream_events({"input": "Hello"}):
    if event["event"] == "on_llm_stream":
        print(event["data"]["chunk"])
```

---

### Q12. How do vector stores integrate? Explain similarity_search vs max_marginal_relevance_search.

**Answer:**
Embeddings convert text to vectors; VectorStores index them.
- **similarity_search**: Returns top-k most similar docs
- **MMR (max_marginal_relevance_search)**: Balances relevance with diversity — penalises docs too similar to already-selected ones, reducing redundancy in retrieved context

**Sample Code:**
```python
# Similarity search
docs = vectorstore.similarity_search("query", k=4)

# MMR search - diverse results
docs = vectorstore.max_marginal_relevance_search(
    "query", k=4, fetch_k=20, lambda_mult=0.5
)
```

---

### Q13. What are .bind(), .with_config(), .with_retry(), and .with_fallbacks()?

**Answer:**
- **`.bind()`**: Freezes keyword args on a Runnable
- **`.with_config()`**: Attaches metadata/tags/callbacks
- **`.with_retry()`**: Adds automatic retry on transient errors
- **`.with_fallbacks()`**: Chains alternative Runnables for resilience (e.g., fall back from GPT-4 to GPT-3.5)

**Sample Code:**
```python
# .bind() - freeze parameters
llm_with_stops = llm.bind(stop=["\n\n"])

# .with_config() - add metadata
chain = chain.with_config(tags=["production"])

# .with_retry() - auto retry
chain = chain.with_retry(stop_after_attempt=3)

# .with_fallbacks() - fallback chain
chain = primary_llm.with_fallbacks([backup_llm])
```

---

### Q14. How do you implement a RAG chain with context compression in LCEL?

**Answer:**
The compressor removes irrelevant passages before they reach the LLM.

**Sample Code:**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

chain = (
    RunnableParallel({
        "context": compression_retriever,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | parser
)
```

---

### Q15. What is LangSmith and how does it integrate for tracing and evaluation?

**Answer:**
LangSmith is LangChain's observability platform. Setting `LANGCHAIN_TRACING_V2=true` auto-traces every invocation — inputs, outputs, latencies, token counts per step. It also supports evaluation: define datasets of (input, expected_output) and evaluators (correctness, faithfulness) to benchmark and regression-test.

**Sample Code:**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# Now all chain invocations are auto-traced
result = chain.invoke({"input": "Hello"})

# Evaluation
from langsmith import evaluate
evaluate(
    lambda x: chain.invoke(x),
    data="dataset_name",
    evaluators=[correctness_evaluator]
)
```

---

### Q16. How does the agent loop work internally (Plan → Act → Observe)?

**Answer:**
1. Agent receives input, calls LLM with prompt + tool descriptions
2. LLM outputs either **AgentFinish** (final answer) or **AgentAction** (tool + input)
3. Framework executes the tool → Observation
4. (Action, Observation) appended to scratchpad; LLM called again
5. Loop until AgentFinish or max iterations

**Scratchpad Example:**
```
Iteration 1:
  Prompt to LLM: "Question: What's the weather in NYC? Tools: [search, calculator]"
  LLM Output: Action: search, Input: "NYC weather"
  Tool Execution → Observation: "72°F, sunny"

Iteration 2:
  Prompt to LLM: "Question: What's the weather in NYC?
                  Scratchpad:
                  Action: search, Input: 'NYC weather'
                  Observation: 72°F, sunny"
  LLM Output: AgentFinish: "The weather in NYC is 72°F and sunny"
```

**Sample Code:**
```python
from langchain.agents import AgentExecutor, create_react_agent

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,
    verbose=True  # Shows scratchpad updates
)

result = agent_executor.invoke({"input": "Search for X"})
```

---

### Q17. Difference between create_react_agent, create_openai_tools_agent, create_structured_chat_agent?

**Answer:**
- **create_react_agent**: Uses text-based ReAct parsing (model-agnostic)
- **create_openai_tools_agent**: Uses OpenAI's native function/tool-calling API — more reliable, supports parallel tool calls
- **create_structured_chat_agent**: Uses JSON-based action specification

OpenAI tools agent preferred with OpenAI models.

**Sample Code:**
```python
from langchain.agents import create_react_agent, create_openai_tools_agent

# ReAct agent - text-based
agent = create_react_agent(llm, tools, prompt)

# OpenAI tools agent - native function calling
agent = create_openai_tools_agent(llm, tools, prompt)

# Both used with AgentExecutor
executor = AgentExecutor(agent=agent, tools=tools)
```

---

### Q18. What is a Multi-Action Agent?

**Answer:**
- **Single-action agent**: Produces one tool call per LLM turn
- **Multi-action agent**: (e.g., via OpenAI parallel tool calling) outputs multiple tool calls in one response, executed concurrently — reducing round-trips and latency

**Sample Code:**
```python
# OpenAI models support parallel tool calling
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Single invoke can trigger multiple tools in parallel
result = executor.invoke({"input": "Search X and calculate Y"})
# LLM may call both search_tool and calculator_tool simultaneously
```

---

### Q19. How do you handle rate limits, token limits, and cost in production?

**Answer:**
- Use `.with_retry()` with exponential backoff
- Count tokens via callbacks (tiktoken), set `max_tokens`
- Use TokenBufferMemory or context compression
- Track costs in callbacks, set budget alerts
- Use cheaper fallbacks and cache repeated calls (InMemoryCache, Redis)

**Sample Code:**
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Caching
set_llm_cache(InMemoryCache())

# Retry with fallback
chain = (
    primary_llm
    .with_retry(stop_after_attempt=3)
    .with_fallbacks([cheaper_llm])
)

# Token limits
llm = ChatOpenAI(model="gpt-4", max_tokens=500)
```

---

### Q20. Explain the document loader ecosystem and how TextSplitters interact with loaders.

**Answer:**
- **Loaders** (PyPDFLoader, WebBaseLoader, CSVLoader, etc.): Ingest content into Document objects with `page_content` + `metadata`
- **TextSplitters** (RecursiveCharacterTextSplitter, TokenTextSplitter, SemanticChunker): Chunk Documents for embedding

**Sample Code:**
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(docs)
```

---

## 🎯 Scenario-Based Questions (10)

### Q1. RAG bot answers are irrelevant on a 50K-page manual. How do you fix it?

**Answer:**
Enable LangSmith tracing to inspect retrieved docs. Adjust chunking (RecursiveCharacterTextSplitter ~500 tokens, 50 overlap). Switch to MMR search. Add ContextualCompressionRetriever + reranker (Cohere/cross-encoder). Refine system prompt to say 'I don't know' when context is insufficient.

---

### Q2. Multi-part questions fail ('Compare A and B, recommend cheaper'). Solution?

**Answer:**
Add query decomposition: LLM breaks complex query into sub-queries. Retrieve docs per sub-query via RunnableParallel, merge contexts, then pass combined context + original question to LLM for synthesis.

**Sample Code:**
```python
# Decomposition chain
decompose = prompt_decompose | llm | JsonOutputParser()
sub_queries = decompose.invoke({"query": "Compare A and B"})

# Parallel retrieval
chain = (
    RunnableParallel({
        f"context_{i}": retriever
        for i, q in enumerate(sub_queries)
    })
    | synthesis_prompt
    | llm
)
```

---

### Q3. Agent with 8 tools keeps picking the wrong tool. How to fix?

**Answer:**
Improve tool descriptions — clear, mutually exclusive, with example inputs. Switch to create_openai_tools_agent for structured function calling. Add few-shot examples of correct tool selection. Group related tools under a router tool. Set max_iterations.

---

### Q4. 500 concurrent users with LLM + vector retrieval + DB lookup. Architecture?

**Answer:**
Use async throughout (ainvoke, astream). Deploy behind FastAPI with async endpoints. RunnableParallel for concurrent retrieval + DB. LLM streaming. Redis caching. Connection pooling. .with_retry() for failures. LangSmith monitoring. Horizontal scaling behind load balancer.

**Architecture Diagram:**
```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                             │
│                    (NGINX/AWS ALB/GCP LB)                        │
└──────────────────┬──────────────┬──────────────┬────────────────┘
                   │              │              │
          ┌────────▼──────┐  ┌────▼──────┐  ┌───▼────────┐
          │   FastAPI     │  │  FastAPI  │  │  FastAPI   │
          │   Server 1    │  │  Server 2 │  │  Server 3  │
          │  (async/await)│  │ (async)   │  │  (async)   │
          └───────┬───────┘  └─────┬─────┘  └──────┬─────┘
                  │                │                │
          ┌───────▼────────────────▼────────────────▼─────┐
          │         Redis Cache (Shared)                  │
          │   - LLM response cache                        │
          │   - Embedding cache                           │
          └───────────────────────────────────────────────┘
                  │
          ┌───────▼─────────────────────────────────────┐
          │    LCEL Chain (Per Request - Async)        │
          │                                             │
          │  User Query                                 │
          │       │                                     │
          │       ▼                                     │
          │  RunnableParallel (Concurrent Execution)    │
          │  ┌─────────────┬──────────────┬──────────┐  │
          │  │             │              │          │  │
          │  ▼             ▼              ▼          │  │
          │ Vector       DB Query      Session      │  │
          │ Retrieval    (asyncpg)     Data         │  │
          │ (async)      Pool: 20      (Redis)      │  │
          │  │             │              │          │  │
          │  └─────────────┴──────────────┘          │  │
          │       │                                  │  │
          │       ▼                                  │  │
          │  Merge Results                          │  │
          │       │                                  │  │
          │       ▼                                  │  │
          │  LLM (with_retry + with_fallbacks)      │  │
          │  Stream tokens (astream)                │  │
          │       │                                  │  │
          │       ▼                                  │  │
          │  Response to Client (SSE/WebSocket)     │  │
          └─────────────────────────────────────────┘
                  │
          ┌───────▼─────────────────────────────────┐
          │   External Services (Connection Pools)  │
          │                                         │
          │  ┌────────────┐  ┌────────────┐        │
          │  │ Vector DB  │  │  Postgres  │        │
          │  │  (Pinecone/│  │   Pool:20  │        │
          │  │   Qdrant)  │  │ Connections│        │
          │  └────────────┘  └────────────┘        │
          │                                         │
          │  ┌────────────┐  ┌────────────┐        │
          │  │  OpenAI    │  │ LangSmith  │        │
          │  │   API      │  │  Tracing   │        │
          │  │ (Retry:3)  │  │ Monitoring │        │
          │  └────────────┘  └────────────┘        │
          └─────────────────────────────────────────┘
```

**Concrete Example - Why Run in Parallel?**

**User Query:** "What's my spending limit on AWS services?"

**Sequential Execution (600ms total):**
```
Step 1: Vector search (200ms) → Finds docs: "AWS spending limits policy..."
Step 2: DB query (200ms) → Gets user data: {role: "engineer", team: "backend"}
Step 3: Session fetch (200ms) → Gets history: ["User asked about Azure yesterday"]
Total: 600ms
```

**Parallel Execution (200ms total):**
```
                 ┌─ Vector search (200ms) → "AWS spending limits policy..."
User Query ──────┼─ DB query (200ms) ───────→ {role: "engineer", team: "backend"}
                 └─ Session fetch (200ms) ──→ ["Asked about Azure yesterday"]
                              │
                              ▼
                    All complete at 200ms
                              │
                              ▼
                     Merge all 3 results
                              │
                              ▼
               Send to LLM with combined context:
               "Context: [AWS policy docs]
                User: engineer on backend team
                History: asked about Azure yesterday
                Question: What's my spending limit?"
                              │
                              ▼
                    LLM generates answer
```

**Why They're Independent:**
1. **Vector search** needs only: query text → searches knowledge base
2. **DB query** needs only: user_id → fetches user profile
3. **Session fetch** needs only: session_id → gets conversation history

**None depends on the others' results** → Safe to run concurrently → 3x faster!


**Component Breakdown:**

**1. Load Balancer**
- Distributes 500 concurrent requests across multiple FastAPI servers
- Health checks, auto-scaling triggers
- SSL termination

**2. FastAPI Servers (Horizontal Scaling)**
- Async endpoints using `async def`
- Each server handles ~150-200 concurrent requests
- Auto-scales based on CPU/memory usage

**3. Redis Cache (Shared)**
- Caches LLM responses (same query → instant response)
- Caches embeddings (avoid re-computing)
- TTL: 1 hour for responses, 24 hours for embeddings

**How `set_llm_cache()` works:**
```python
from langchain.globals import set_llm_cache
from langchain.cache import RedisCache

set_llm_cache(RedisCache(redis_client))

# What gets cached?
# Cache Key = hash(prompt_text + model_name + temperature + max_tokens + ...)
# Cache Value = LLM response text

# Example flow:
# Request 1: User asks "What is Python?"
#   → Cache miss (not in cache)
#   → Calls OpenAI API (costs $0.002, takes 800ms)
#   → Response: "Python is a programming language..."
#   → Saves to Redis: key="hash_abc123" → value="Python is a programming language..."

# Request 2: Different user asks "What is Python?" (exact same prompt)
#   → Cache hit! (found in Redis)
#   → Returns from cache (costs $0, takes 5ms)
#   → Response: "Python is a programming language..." (instant!)

# Request 3: User asks "What is Java?" (different prompt)
#   → Cache miss (different hash)
#   → Calls OpenAI API again
```

**What's included in cache key:**
- Prompt text (exact match required)
- Model name (gpt-4, gpt-3.5-turbo, etc.)
- Temperature, max_tokens, top_p
- System messages, few-shot examples

**Important:** Even tiny prompt differences create different cache keys!
```python
"What is Python?"      → Cache key: hash_abc123
"What is Python? "     → Cache key: hash_xyz789 (extra space!)
"what is python?"      → Cache key: hash_def456 (different case!)
```

**Cache Flow Diagram:**
```
Request arrives: "What is Python?"
         │
         ▼
  Generate cache key
  hash(prompt + model + params)
         │
         ▼
    Check Redis
         │
    ┌────┴────┐
    │         │
Cache HIT  Cache MISS
    │         │
    ▼         ▼
Return    Call OpenAI API
cached    (costs $, slow)
response       │
(free,         ▼
instant)   Save response
    │       to Redis
    │          │
    └──────────┘
         │
         ▼
    Return to user
```

**Cache backends available:**
- **InMemoryCache**: Fast but lost on restart, single-server only
- **RedisCache**: Persistent, shared across servers (production choice)
- **SQLiteCache**: File-based, single-server
- **UpstashRedisCache**: Serverless Redis

**Real-world impact (500 concurrent users):**
- Without cache: 500 requests × $0.002 = $1.00, avg latency 800ms
- With 50% cache hit rate: 250 API calls × $0.002 = $0.50 (50% cost savings!)
- Cached requests: <10ms response time
- API rate limits: Reduced by 50%, less throttling

**4. LCEL Chain (Per Request)**
- **RunnableParallel**: Runs independent operations concurrently, then merges results
- **Why parallel?**: Each operation is independent - they don't need each other's results
- **What each does**:
  - **Vector Search**: "What docs are relevant to this query?" → Knowledge base context
  - **DB Query**: "What do I know about this user?" → User profile, preferences, history
  - **Session Fetch**: "What did we talk about earlier?" → Conversation history from Redis
- **After parallel execution**: All 3 results merged → sent together to LLM
- **Time saved**: If each takes 200ms, parallel = 200ms total vs sequential = 600ms total
- **Streaming**: Tokens streamed to client via SSE/WebSocket

**5. Connection Pools**
- **VectorDB**: Connection pooling to Pinecone/Qdrant
- **Postgres**: asyncpg pool with 20 connections per server
- **Redis**: Connection pool for cache operations

**6. Retry & Fallback**
- LLM calls: `.with_retry(stop_after_attempt=3)`
- Fallback: GPT-4 → GPT-3.5 on failure
- Circuit breaker pattern for failing services

**7. Monitoring**
- **LangSmith**: Traces every request (latency, tokens, costs)
- **Metrics**: Prometheus/Grafana for system health
- **Alerts**: Slack/PagerDuty for errors/latency spikes

**Sample Code:**
```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache
import asyncpg
import redis.asyncio as redis

app = FastAPI()

# Redis cache setup - Caches LLM prompt → response mappings
redis_client = redis.Redis(host="redis", decode_responses=True)
set_llm_cache(RedisCache(redis_client))

# How it works:
# 1. First request: prompt="What is Python?" → Calls OpenAI API → Response cached
# 2. Same request: prompt="What is Python?" → Returns from cache (no API call!)
# 3. Cache key = hash(prompt + model + parameters)
# 4. Saves: API costs, latency, rate limits

# Postgres connection pool
db_pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        "postgresql://...",
        min_size=10,
        max_size=20
    )

# Async functions for parallel execution (INDEPENDENT operations)
async def vector_search(query: str):
    """Searches knowledge base - needs only query text"""
    # Input: "What's my AWS spending limit?"
    # Output: [Doc("AWS limits: $500/month for engineers...")]
    results = await vector_store.asimilarity_search(query, k=5)
    return results

async def db_lookup(user_id: str):
    """Fetches user profile - needs only user_id"""
    # Input: user_id="u_123"
    # Output: {role: "engineer", team: "backend", tier: "standard"}
    async with db_pool.acquire() as conn:
        result = await conn.fetch("SELECT * FROM users WHERE id=$1", user_id)
    return result

async def session_history(session_id: str):
    """Gets conversation history - needs only session_id"""
    # Input: session_id="sess_456"
    # Output: ["Asked about Azure limits yesterday"]
    history = await redis_client.lrange(f"session:{session_id}", 0, -1)
    return history

# LCEL chain with RunnableParallel
llm = ChatOpenAI(model="gpt-4", streaming=True)
llm_with_fallback = llm.with_retry(
    stop_after_attempt=3
).with_fallbacks([ChatOpenAI(model="gpt-3.5-turbo")])

# RunnableParallel executes all 3 functions CONCURRENTLY
# Each function is independent - doesn't need others' results
# Results are merged into a dict after all complete
chain = (
    RunnableParallel({
        "context": lambda x: vector_search(x["query"]),      # ← Runs in parallel
        "user_data": lambda x: db_lookup(x["user_id"]),      # ← Runs in parallel
        "history": lambda x: session_history(x["session_id"]), # ← Runs in parallel
        "query": RunnablePassthrough()  # Just passes query through
    })
    # ↓ After parallel execution, all results merged:
    # {
    #   "context": [Doc("AWS limits...")],
    #   "user_data": {role: "engineer"...},
    #   "history": ["Asked about Azure..."],
    #   "query": "What's my spending limit?"
    # }
    | prompt  # ← Prompt receives all merged data
    | llm_with_fallback
    | parser
)

# Async endpoint with streaming
@app.post("/chat")
async def chat_endpoint(query: str, user_id: str):
    from fastapi.responses import StreamingResponse

    async def generate():
        async for chunk in chain.astream({
            "query": query,
            "user_id": user_id
        }):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# Health check for load balancer
@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Performance Optimization Strategies:**

1. **Async Throughout**: All I/O operations (DB, vector store, LLM) use async
2. **Parallel Execution**: RunnableParallel runs vector search + DB query concurrently
3. **Connection Pooling**: Reuse connections, avoid connection overhead
4. **Caching**: Redis caches frequent queries, reducing LLM calls by 40-60%
5. **Streaming**: Stream tokens to reduce perceived latency
6. **Horizontal Scaling**: Add more servers as load increases
7. **Retry Logic**: Auto-retry transient failures (rate limits, timeouts)
8. **Fallback Models**: Use cheaper models on primary failure
9. **Monitoring**: LangSmith tracks slow queries, token usage, costs

**Expected Performance:**
- **Throughput**: 500 concurrent users across 3 servers
- **Latency**: p50 ~2s, p95 ~5s (with streaming, first token <500ms)
- **Cost**: Cache hit rate 50% → 50% reduction in LLM costs
- **Availability**: 99.9% with retry + fallback + health checks

---

### Q5. Route billing vs technical support to different RAG pipelines. Design in LCEL?

**Answer:**
Classifier chain: prompt | llm | StrOutputParser → 'billing'/'technical'. RunnableBranch: if billing → billing_retriever | billing_prompt | llm; if technical → tech_retriever | tech_prompt | llm; default → general chain. Each branch has its own vector store and prompt.

**Sample Code:**
```python
from langchain_core.runnables import RunnableBranch

# Routing chain
branch = RunnableBranch(
    (
        lambda x: "billing" in x["category"],
        billing_retriever | billing_prompt | llm
    ),
    (
        lambda x: "technical" in x["category"],
        tech_retriever | tech_prompt | llm
    ),
    general_chain  # default
)

full_chain = classifier_chain | branch | parser
```

---

### Q6. Every response must include source citations with page numbers. Implementation?

**Answer:**
Force LLM to cite sources by including them in prompt and parsing output.

**What metadata is included by default?**

Different loaders include different metadata automatically:

**PyPDFLoader** - YES, includes by default:
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("manual.pdf")
docs = loader.load()

# Check metadata
print(docs[0].metadata)
# {'source': 'manual.pdf', 'page': 0}  ← YES, automatic!

# Each page is a separate Document
print(len(docs))  # 50 (if PDF has 50 pages)
```

**DirectoryLoader** - YES, includes filename:
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

print(docs[0].metadata)
# {'source': 'docs/file1.txt'}  ← YES, automatic!
```

**TextLoader** - NO page numbers (text files don't have pages):
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("document.txt")
docs = loader.load()

print(docs[0].metadata)
# {'source': 'document.txt'}  ← Filename YES, page NO
```

**UnstructuredLoader** - YES, more metadata:
```python
from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("report.pdf")
docs = loader.load()

print(docs[0].metadata)
# {
#   'source': 'report.pdf',
#   'page_number': 1,  ← YES!
#   'filename': 'report.pdf',
#   'file_directory': '/path/to/dir'
# }
```

**Important:** After splitting, you must preserve metadata!

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load with metadata
loader = PyPDFLoader("manual.pdf")
docs = loader.load()  # Each doc has metadata

# Split - metadata is PRESERVED by default
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_documents(docs)

# Verify metadata preserved
print(chunks[0].metadata)
# {'source': 'manual.pdf', 'page': 0}  ← Still there!
print(chunks[10].metadata)
# {'source': 'manual.pdf', 'page': 2}  ← Page updated correctly
```

**Summary - What's included by default:**

| Loader | Source/Filename | Page Number | Other Metadata |
|--------|-----------------|-------------|----------------|
| PyPDFLoader | ✅ Yes | ✅ Yes (per page) | - |
| DirectoryLoader | ✅ Yes | Depends on inner loader | - |
| TextLoader | ✅ Yes | ❌ No (no pages in .txt) | - |
| UnstructuredLoader | ✅ Yes | ✅ Yes | File directory, etc. |
| CSVLoader | ✅ Yes | ❌ No | Row number |
| WebBaseLoader | ✅ Yes (URL) | ❌ No | Title, language |

**Key Point:** Most loaders include source/filename by default, but **you must check** what's included for your specific loader.

**How to verify:**
```python
# Always check metadata after loading
loader = PyPDFLoader("manual.pdf")
docs = loader.load()
print(docs[0].metadata)  # See what's included

# After splitting, verify metadata preserved
chunks = splitter.split_documents(docs)
print(chunks[0].metadata)  # Should still have source & page
```

**If metadata is missing, add it manually:**
```python
# Add custom metadata
for i, doc in enumerate(docs):
    doc.metadata["chunk_id"] = i
    doc.metadata["document_type"] = "manual"
    doc.metadata["version"] = "v2.0"
```

**Complete RAG example with citations:**

**Code:**
```python
# 1. Load with metadata (automatic for PyPDFLoader)
loader = PyPDFLoader("manual.pdf")
docs = loader.load()
print(docs[0].metadata)  # {'source': 'manual.pdf', 'page': 0} ✓

# 2. Format docs with metadata
def format_with_metadata(docs):
    return "\n\n".join([
        f"[{doc.metadata['source']}, page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in docs
    ])

# 3. Prompt instructs citation format
prompt = ChatPromptTemplate.from_template("""
Answer using only context. Cite as [source, page X].

Context: {context}
Question: {question}
Answer with citations:
""")

# 4. Chain
chain = (
    {"context": retriever | format_with_metadata, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Output: "The limit is $500 [manual.pdf, page 12]"
```

---

### Q7. Chain fails intermittently with timeout errors in production. Resilience strategy?

**Answer:**
Add retry, fallback, and timeout handling.

**Code:**
```python
# 1. Retry with exponential backoff
llm_with_retry = llm.with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1000  # 1s, 2s, 4s
)

# 2. Fallback to cheaper model
llm_with_fallback = llm_with_retry.with_fallbacks([
    ChatOpenAI(model="gpt-3.5-turbo")  # Faster fallback
])

# 3. Set timeout
llm = ChatOpenAI(model="gpt-4", request_timeout=30)

# 4. Use in chain
chain = retriever | prompt | llm_with_fallback | parser

# Flow: GPT-4 timeout → retry 3x → fail → GPT-3.5
```

---

### Q8. Add conversation memory to existing stateless RAG chain without rewriting it.

**Answer:**
Wrap existing chain with `RunnableWithMessageHistory` - no chain changes needed.

**Code:**
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Existing chain (unchanged)
chain = retriever | prompt | llm | parser

# 1. Add history placeholder to prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 2. Session history getter
def get_session_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://localhost")

# 3. Wrap chain
chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# 4. Use with session
chain_with_memory.invoke(
    {"question": "What is RAG?"},
    config={"configurable": {"session_id": "user_123"}}
)
```

---

### Q9. User uploads a PDF, asks questions across multiple turns with memory. Architecture?

**Answer:**
Create per-session vector store + conversational RAG chain.

**Flow:**
```
Upload PDF → Load → Split → FAISS per session
     ↓
Ask questions → Retrieve from session FAISS → LLM with history
```

**Code:**
```python
from langchain_community.vectorstores import FAISS

session_stores = {}  # {session_id: FAISS}

# 1. Upload handler
def process_upload(pdf_path: str, session_id: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    chunks = splitter.split_documents(docs)

    # Per-session vector store
    session_stores[session_id] = FAISS.from_documents(chunks, embeddings)

# 2. Chat handler
def chat(question: str, session_id: str):
    retriever = session_stores[session_id].as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_with_history
        | llm
    )

    chain_with_memory = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=lambda sid: get_redis_history(sid),
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    return chain_with_memory.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

# 3. Cleanup
def cleanup(session_id: str):
    del session_stores[session_id]
    redis_client.delete(f"session:{session_id}")
```

---

### Q10. Evaluate GPT-4 vs open-source model quality in your RAG pipeline.

**Answer:**
Create test dataset, run both models, compare metrics in LangSmith.

**Code:**
```python
# Imports
from langsmith import Client, evaluate
from langchain.evaluation import load_evaluator  # ← From LangChain, NOT LangSmith!
from langchain_openai import ChatOpenAI

client = Client()

# 1. Create dataset (100+ examples)
dataset = client.create_dataset("rag-eval")
client.create_examples(
    dataset_id=dataset.id,
    inputs=[{"question": "What is the return policy?"}],
    outputs=[{"answer": "30 days", "source": "policy.pdf, page 5"}]
)

# 2. Define evaluators (from langchain.evaluation)
evaluators = [
    # QA Evaluator - checks correctness
    load_evaluator("qa", llm=ChatOpenAI(model="gpt-4")),

    # Criteria Evaluator - checks faithfulness
    load_evaluator("criteria", criteria="faithfulness"),

    # Custom evaluator - checks for citations
    lambda run, example: {
        "has_citation": "[" in run.outputs.get("answer", "")
    }
]

# 3. Evaluate GPT-4
gpt4_results = evaluate(
    lambda inputs: gpt4_chain.invoke(inputs),
    data=dataset.name,
    evaluators=evaluators,
    experiment_prefix="gpt4-rag"
)

# 4. Evaluate Llama
llama_results = evaluate(
    lambda inputs: llama_chain.invoke(inputs),
    data=dataset.name,
    evaluators=evaluators,
    experiment_prefix="llama-rag"
)

# 5. Compare in LangSmith UI
# Metrics: accuracy, latency, cost, failures
```

**Where does `load_evaluator` come from?**

```python
# ✅ CORRECT - from LangChain
from langchain.evaluation import load_evaluator

# ❌ WRONG - NOT from LangSmith
# from langsmith import load_evaluator  # This doesn't exist!
```

**Available Evaluators:**

```python
# 1. QA Evaluator - LLM judges answer correctness
qa_evaluator = load_evaluator(
    "qa",
    llm=ChatOpenAI(model="gpt-4")
)

# 2. Criteria Evaluator - judges custom criteria
criteria_evaluator = load_evaluator(
    "criteria",
    criteria="faithfulness"  # or "relevance", "coherence", "harmfulness"
)

# 3. Labeled Criteria - with reference answers
labeled_criteria = load_evaluator(
    "labeled_criteria",
    criteria="correctness"
)

# 4. String Distance - exact match scoring
string_distance = load_evaluator(
    "string_distance",
    distance="levenshtein"
)

# 5. Embedding Distance - semantic similarity
embedding_distance = load_evaluator(
    "embedding_distance",
    embeddings=OpenAIEmbeddings()
)

# 6. Regex Match - pattern matching
regex_match = load_evaluator(
    "regex_match",
    flags=re.IGNORECASE
)
```

**Complete Example with All Imports:**

```python
# Step 1: Import everything
from langsmith import Client, evaluate  # LangSmith for platform
from langchain.evaluation import load_evaluator  # LangChain for evaluators
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Step 2: Initialize
client = Client()
llm = ChatOpenAI(model="gpt-4")

# Step 3: Create evaluators
evaluators = [
    # Correctness (LLM-as-judge)
    load_evaluator("qa", llm=llm),

    # Faithfulness (grounded in context?)
    load_evaluator("criteria", criteria="faithfulness"),

    # Semantic similarity to expected answer
    load_evaluator("embedding_distance", embeddings=OpenAIEmbeddings()),

    # Custom Python function
    lambda run, example: {
        "word_count": len(run.outputs["answer"].split()),
        "has_source": "source" in run.outputs
    }
]

# Step 4: Run evaluation on LangSmith platform
results = evaluate(
    lambda inputs: my_chain.invoke(inputs),
    data="my-dataset",
    evaluators=evaluators  # Pass LangChain evaluators to LangSmith
)
```

**Relationship:**

```
┌─────────────────────────────────────────┐
│           LangChain                     │
│  (Evaluation Functions)                 │
│                                         │
│  • load_evaluator()                     │
│  • QAEvalChain                          │
│  • CriteriaEvalChain                    │
│  • Custom evaluators                    │
└─────────────────┬───────────────────────┘
                  │
                  │ Used by
                  ▼
┌─────────────────────────────────────────┐
│          LangSmith                      │
│   (Evaluation Platform)                 │
│                                         │
│  • evaluate() function                  │
│  • Dataset management                   │
│  • Result visualization                 │
│  • Comparison UI                        │
└─────────────────────────────────────────┘
```

**Summary:**
- ✅ `load_evaluator` is from **`langchain.evaluation`**
- ✅ `evaluate` function is from **`langsmith`**
- LangChain provides the **evaluator functions**
- LangSmith provides the **platform** to run and visualize evaluations

**Results:**
```
Model       Accuracy  Latency  Cost
GPT-4       92%       2.1s     $0.06
Llama-70B   85%       1.2s     $0.01
```

---

# Part 2 — LangGraph

## 📚 Concept-Based Questions (20)

### Q1. What is LangGraph and how does it differ from LCEL chains?

**Answer:**
LangGraph builds stateful, multi-actor apps as graphs with cycles. LCEL is for linear DAGs.

**Key Differences:**
| Feature | LCEL | LangGraph |
|---------|------|-----------|
| Flow | Linear/branching (DAG) | Cycles allowed |
| State | Stateless | Persistent state |
| HITL | Not supported | Built-in interrupts |
| Use case | RAG chains | Agents, workflows |

**Code:**
```python
# LCEL - Linear chain (no cycles)
chain = retriever | prompt | llm | parser

# LangGraph - Cycles allowed
from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("tools", "agent")  # ← Cycle back!
graph.add_conditional_edges("agent", should_continue)
```

---

### Q2. Explain StateGraph, Nodes, Edges, and Conditional Edges.

**Answer:**
Core building blocks of LangGraph.

**Code:**
```python
from langgraph.graph import StateGraph
from typing import TypedDict

# State schema
class State(TypedDict):
    messages: list
    count: int

# Nodes - functions that receive & update state
def node_a(state: State):
    return {"count": state["count"] + 1}

def node_b(state: State):
    return {"count": state["count"] * 2}

# Conditional edge - routing logic
def router(state: State):
    if state["count"] > 10:
        return "end"
    return "node_b"

# Build graph
graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)

# Regular edge (always goes to node_a)
graph.add_edge("node_b", "node_a")

# Conditional edge (routes based on state)
graph.add_conditional_edges(
    "node_a",
    router,  # Function that returns next node name
    {"node_b": "node_b", "end": END}
)

graph.set_entry_point("node_a")
app = graph.compile()
```

---

### Q3. What is State in LangGraph? How do Annotated fields with reducers work?

**Answer:**
State carries data through the graph. Reducers define how to merge updates instead of overwriting.

**Code:**
```python
from typing import Annotated
import operator

def add_messages(left: list, right: list):
    """Custom reducer: append new messages"""
    return left + right

class State(TypedDict):
    # Default behavior: overwrite
    count: int

    # With reducer: merge using add_messages
    messages: Annotated[list, add_messages]

    # Built-in reducer: addition
    total: Annotated[int, operator.add]

# How it works:
initial_state = {"count": 5, "messages": ["hi"], "total": 10}

# Node returns update
update = {"count": 10, "messages": ["bye"], "total": 5}

# Result after merge:
# {
#   "count": 10,           # Overwritten (no reducer)
#   "messages": ["hi", "bye"],  # Merged with add_messages
#   "total": 15            # Merged with operator.add (10 + 5)
# }
```

---

### Q4. What is MessagesState and when would you use it vs custom state?

**Answer:**
MessagesState is a pre-built state for simple chatbots. Use custom state for production.

**Code:**
```python
from langgraph.graph import MessagesState

# MessagesState - simple chatbot
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

graph = StateGraph(MessagesState)
# Good for: Simple chat, quick prototypes

# Custom State - production apps
class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: list  # RAG context
    user_id: str  # User tracking
    iteration_count: int  # Loop guard
    tool_results: dict  # Tool outputs

graph = StateGraph(CustomState)
# Good for: RAG, multi-step agents, HITL workflows
```

---

### Q5. How do Conditional Edges work? What does tools_condition simplify?

**Answer:**
Conditional edges route based on state. `tools_condition` is a built-in router for ReAct agents.

**Code:**
```python
from langgraph.prebuilt import tools_condition

# Custom conditional edge
def my_router(state: State):
    if state["count"] > 10:
        return "finish"
    elif state["error"]:
        return "retry"
    return "continue"

graph.add_conditional_edges(
    "process",
    my_router,
    {
        "finish": END,
        "retry": "process",
        "continue": "next_step"
    }
)

# tools_condition - built-in for agents
graph.add_conditional_edges(
    "agent",
    tools_condition,  # Checks if LLM returned tool_calls
    {
        "tools": "tools",  # Has tool_calls → execute tools
        END: END           # No tool_calls → finish
    }
)

# tools_condition logic (simplified):
def tools_condition(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

---

## 🔰 **BEGINNER'S GUIDE: ToolNode & tools_condition**

### **What are they?**

**ToolNode** = A node that executes tools (functions) that the LLM wants to call
**tools_condition** = A router that decides: "Should we run tools or finish?"

They work together to create the classic **ReAct agent loop**.

---

### **Step-by-Step: How They Work Together**

**The Problem They Solve:**

Your agent needs to:
1. Let the LLM decide which tools to use
2. Actually execute those tools
3. Send results back to the LLM
4. Repeat until the task is done

**The Solution:**

```
User Question
     ↓
  [Agent Node] ← LLM decides: "I need to use the calculator tool"
     ↓
  [tools_condition] ← Checks: "Did LLM ask for tools?"
     ↓
    YES → [ToolNode] ← Executes calculator(5 + 3)
     ↓
  [Agent Node] ← LLM sees result: "8", generates final answer
     ↓
  [tools_condition] ← Checks: "Did LLM ask for tools?"
     ↓
    NO → [END] ← Done!
```

---

### **Example 1: Simple Agent with ToolNode**

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. Define a simple tool
@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tools = [calculator]

# 2. Create LLM with tools
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools(tools)

# 3. Agent node - LLM decides what to do
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 4. Build the graph
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))  # ← ToolNode executes tools

# Add edges
graph.set_entry_point("agent")
graph.add_edge("tools", "agent")  # After tools, go back to agent

# The magic: tools_condition decides the route
graph.add_conditional_edges(
    "agent",
    tools_condition,  # ← Checks if LLM wants to use tools
    {
        "tools": "tools",  # If yes → go to ToolNode
        END: END          # If no → we're done
    }
)

app = graph.compile()

# 5. Run it!
result = app.invoke({"messages": [("user", "What's 5 + 3?")]})
print(result["messages"][-1].content)
# "The result is 8"
```

---

### **What Happens Inside?**

**Turn 1: LLM Decides to Use Tool**

```python
# User asks: "What's 5 + 3?"

# Agent node runs:
response = llm_with_tools.invoke([("user", "What's 5 + 3?")])

# LLM response looks like:
# AIMessage(
#     content="",
#     tool_calls=[{
#         "name": "calculator",
#         "args": {"a": 5, "b": 3},
#         "id": "call_123"
#     }]
# )

# tools_condition checks:
if response.tool_calls:  # ✅ Yes, there are tool_calls
    return "tools"  # → Go to ToolNode

# ToolNode executes:
result = calculator(a=5, b=3)  # Returns 8

# ToolNode creates:
# ToolMessage(
#     content="8",
#     tool_call_id="call_123"
# )

# Goes back to agent...
```

**Turn 2: LLM Gives Final Answer**

```python
# Agent node runs again with:
# [
#   ("user", "What's 5 + 3?"),
#   AIMessage(tool_calls=[...]),
#   ToolMessage(content="8")
# ]

response = llm_with_tools.invoke(messages)

# LLM response:
# AIMessage(
#     content="The result is 8",
#     tool_calls=[]  # ← No more tool calls!
# )

# tools_condition checks:
if response.tool_calls:  # ❌ No, empty
    return END  # → Finish!
```

---

### **Example 2: Multiple Tools**

```python
@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

tools = [calculator, search]

# ToolNode automatically handles both tools!
graph.add_node("tools", ToolNode(tools))

# LLM can choose which tool to use:
result = app.invoke({"messages": [("user", "Search for Python and add 5 + 3")]})

# Flow:
# 1. Agent: "I'll use search tool"
# 2. ToolNode: executes search("Python")
# 3. Agent: "Now I'll use calculator"
# 4. ToolNode: executes calculator(5, 3)
# 5. Agent: "Here are the results: ..."
```

---

### **Visual Breakdown**

**What ToolNode Does:**

```python
# ToolNode is like this:
class ToolNode:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    def __call__(self, state):
        last_message = state["messages"][-1]

        # Get all tool calls from LLM
        tool_results = []
        for tool_call in last_message.tool_calls:
            # Find the right tool
            tool = self.tools[tool_call["name"]]

            # Execute it!
            result = tool.invoke(tool_call["args"])

            # Create result message
            tool_results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": tool_results}
```

**What tools_condition Does:**

```python
# tools_condition is like this:
def tools_condition(state):
    last_message = state["messages"][-1]

    # Check if LLM asked for tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"  # Yes → execute tools
    else:
        return END  # No → we're done
```

---

### **Complete Example: Weather Agent**

```python
from typing import Literal

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    # Simulated - in reality, call an API
    return f"The weather in {city} is sunny, 72°F"

@tool
def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    return f"Current time in {timezone}: 3:30 PM"

tools = [get_weather, get_time]

# Create graph
graph = StateGraph(MessagesState)
graph.add_node("agent", lambda s: {"messages": [llm_with_tools.invoke(s["messages"])]})
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", tools_condition)

app = graph.compile()

# Test it!
result = app.invoke({
    "messages": [("user", "What's the weather in NYC and what time is it there?")]
})

# Flow:
# 1. Agent: "I need get_weather and get_time"
# 2. tools_condition: "Tool calls found!" → ToolNode
# 3. ToolNode: Executes both tools in parallel
# 4. Agent: Receives results, formats answer
# 5. tools_condition: "No more tool calls" → END
```

---

### **Key Takeaways**

1. **ToolNode** = Executes the tools LLM wants to use
2. **tools_condition** = Decides if we need to execute tools or finish
3. Together they create the **ReAct loop**: Think → Act → Observe → Repeat
4. You just provide the tools, LangGraph handles the rest!

**The Magic:**
- ✅ LLM decides which tools to use (you don't write if/else)
- ✅ ToolNode executes them automatically
- ✅ tools_condition handles the routing
- ✅ Loop continues until LLM says "I'm done"

---

## 🔰 **Command(resume) vs Command(update)**

### **Short Answer:**

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `Command(resume=value)` | Resume with approval value | After `interrupt()` asks for input |
| `Command(update={...})` | Update state AND resume | When you need to modify state before continuing |

### **Code Examples:**

**Scenario 1: Command(resume) - Simple Approval**

```python
from langgraph.types import interrupt, Command

def approval_node(state):
    if state["amount"] > 1000:
        # Pause and ask for approval
        approved = interrupt("Approve this payment?")
        if not approved:
            return {"status": "rejected"}
    return {"status": "approved"}

# Run - pauses at interrupt
config = {"configurable": {"thread_id": "t1"}}
app.invoke({"amount": 5000}, config)
# Execution paused, waiting for human

# Resume with approval (pass config separately!)
app.invoke(Command(resume=True), config)  # ✅ Approved
# OR
app.invoke(Command(resume=False), config)  # ❌ Rejected

# Config is passed as 2nd parameter, NOT inside Command!
```

**Scenario 2: Command(update) - Modify State**

```python
# Run - pauses at interrupt
app.invoke({"email": "Draft email..."}, config)
# Execution paused

# Human reviews and modifies the email
app.invoke(
    Command(update={"email": "Edited by human"}),  # Update state
    config  # Config passed separately
)
# Resumes with modified state
```

**Key Differences:**

```python
# Command(resume) - Just continues execution
Command(resume=True)  # Simple value passed to interrupt()

# Command(update) - Modifies state THEN continues
Command(update={"field": "new value"})  # Dict of state updates

# Both require config passed separately:
app.invoke(Command(...), config)  # ← Config is 2nd param
```

**Complete Example:**

```python
from langgraph.types import interrupt, Command

def process_payment(state):
    amount = state["amount"]

    if amount > 10000:
        # Dynamic interrupt - asks for approval
        decision = interrupt(f"Approve ${amount}?")

        if decision == "approve":
            return {"status": "processing"}
        elif decision == "edit":
            # Will be updated via Command(update)
            return {"status": "waiting_for_edit"}
        else:
            return {"status": "rejected"}

    return {"status": "processing"}

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session_1"}}

# Initial run - pauses
app.invoke({"amount": 15000}, config)

# Option 1: Just resume with approval
app.invoke(Command(resume="approve"), config)

# Option 2: Update amount and resume
app.invoke(
    Command(update={"amount": 5000}),  # Reduce amount
    config
)

# Option 3: Reject
app.invoke(Command(resume="reject"), config)
```

**Summary:**
- `Command(resume=value)` → Pass value to `interrupt()`, continue
- `Command(update={...})` → Update state, then continue
- Config ALWAYS passed separately: `app.invoke(Command(...), config)`

---

## 🔰 **How does interrupt know where to resume?**

**Answer:** Checkpoint saves the node name + execution position.

**Code:**
```python
def payment_node(state):
    approved = interrupt("Approve?")  # ← Pauses here
    # Checkpoint saves: node="payment_node", line=after interrupt

app.invoke({"amount": 5000}, config)  # Pauses

# Check where it will resume
state = app.get_state(config)
print(state.next)  # ('payment_node',) ← Resume here

# Resume - loads checkpoint, continues from saved position
app.invoke(Command(resume=True), config)
```

**That's it:** Checkpoint stores `{node_name, position}` → Resume loads it → Continues.

---

### Q6. Explain checkpointing: MemorySaver vs SqliteSaver vs PostgresSaver.

**Answer:**
Checkpointing saves state after every node for persistence and HITL.

**Code:**
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# MemorySaver - dev/testing (lost on restart)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# SqliteSaver - single-server production
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=checkpointer)

# PostgresSaver - multi-server production
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
app = graph.compile(checkpointer=checkpointer)

# Usage - state persists per thread
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke({"messages": ["Hello"]}, config)

# Resume from checkpoint
result2 = app.invoke({"messages": ["Continue"]}, config)
# Continues from where thread_id="user_123" left off!
```

**Comparison:**
| Checkpointer | Storage | Multi-server | Use case |
|--------------|---------|--------------|----------|
| MemorySaver | RAM | No | Dev/test |
| SqliteSaver | File | No | Single server |
| PostgresSaver | Postgres | Yes | Production |

---

### Q7. How does human-in-the-loop work? Explain interrupt_before and interrupt_after.

**Answer:**
HITL pauses execution for human review before or after specific nodes.

**Code:**
```python
# Compile with interrupts
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["send_email"],  # Pause before sending
    interrupt_after=["generate_code"]  # Pause after code gen
)

# Run until interrupt
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke({"task": "Send email to client"}, config)
# Pauses before "send_email" node

# Inspect state
state = app.get_state(config)
print(state.values)  # See what's about to be sent

# Modify state if needed
app.update_state(config, {"email_body": "Edited by human"})

# Resume execution
result = app.invoke(None, config)  # None = resume
# Continues from where it paused
```

**Flow:**
```
agent → generate_code → [PAUSE] → human reviews → resume → send_email → [PAUSE] → human approves → resume → END
```

---

### Q8. What are dynamic interrupts and how do they differ from static?

**Answer:**
Static interrupts always pause at specific nodes. Dynamic interrupts pause conditionally at runtime.

**Code:**
```python
from langgraph.types import interrupt

# Static interrupt - always pauses
app = graph.compile(interrupt_before=["send_email"])

# Dynamic interrupt - conditional pause
def process_payment(state: State):
    amount = state["amount"]

    # Only pause if amount > $10,000
    if amount > 10000:
        approval = interrupt("Large payment needs approval")
        # Execution pauses here, returns control to user
        if not approval:
            return {"status": "rejected"}

    # Process payment
    return {"status": "completed"}

# Usage
config = {"configurable": {"thread_id": "txn_123"}}
result = app.invoke({"amount": 15000}, config)
# Pauses with message "Large payment needs approval"

# Human approves
result = app.invoke(Command(resume=True), config)
# Continues execution
```

**Comparison:**
| Type | When pauses | Set at | Flexibility |
|------|-------------|--------|-------------|
| Static | Always | Compile time | Fixed nodes |
| Dynamic | Conditionally | Runtime | Based on state |

---

### Q9. Explain the Supervisor pattern in multi-agent systems.

**Answer:**
Supervisor routes tasks to specialist agents in a hub-and-spoke pattern.

**Code:**
```python
# Supervisor node - LLM decides next agent
def supervisor(state: State):
    prompt = f"""
    Task: {state['task']}
    Results so far: {state['results']}

    Which specialist should work next?
    - researcher: gather information
    - coder: write code
    - analyst: analyze data
    - FINISH: task complete
    """

    response = llm.invoke(prompt)
    return {"next_agent": response.content}

# Specialist nodes
def researcher(state: State):
    # Research logic
    return {"results": state["results"] + ["research done"]}

def coder(state: State):
    # Coding logic
    return {"results": state["results"] + ["code written"]}

# Routing from supervisor
def route_supervisor(state: State):
    return state["next_agent"]

# Build graph
graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)

# All specialists return to supervisor
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")

# Supervisor routes to next agent or END
graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "coder": "coder",
        "FINISH": END
    }
)

graph.set_entry_point("supervisor")
```

**Flow:**
```
Task → Supervisor → Researcher → Supervisor → Coder → Supervisor → FINISH
           ↑______________|          ↑________|
          (hub-and-spoke)
```

---

### Q10. What are Subgraphs and why use them?

**Answer:**
Subgraphs are nested graphs used as nodes. Enable modularity and reusability.

**Code:**
```python
# Subgraph - research workflow
research_graph = StateGraph(ResearchState)
research_graph.add_node("search", search_node)
research_graph.add_node("summarize", summarize_node)
research_graph.add_edge("search", "summarize")
research_graph.set_entry_point("search")

# Compile subgraph
research_subgraph = research_graph.compile()

# Parent graph uses subgraph as a node
main_graph = StateGraph(MainState)
main_graph.add_node("plan", plan_node)
main_graph.add_node("research", research_subgraph)  # ← Subgraph as node
main_graph.add_node("write", write_node)

main_graph.add_edge("plan", "research")
main_graph.add_edge("research", "write")

app = main_graph.compile()
```

**Benefits:**
- **Modularity**: Each subgraph is self-contained
- **Reusability**: Use same research graph in multiple workflows
- **Testing**: Test subgraph independently
- **Clean interface**: Parent only sees subgraph input/output

---

### Q11. How does LangGraph handle error recovery?

**Answer:**
Checkpointing + error-handling nodes enable robust recovery.

**Code:**
```python
def api_call_node(state: State):
    try:
        result = external_api.call(state["query"])
        return {"result": result, "error": None}
    except Exception as e:
        return {"error": str(e), "retry_count": state.get("retry_count", 0) + 1}

def should_retry(state: State):
    if state.get("error") and state["retry_count"] < 3:
        return "retry"
    elif state.get("error"):
        return "fallback"
    return "success"

graph.add_node("api_call", api_call_node)
graph.add_node("fallback", fallback_node)
graph.add_conditional_edges("api_call", should_retry, {
    "retry": "api_call",  # Try again
    "fallback": "fallback",  # Give up, use fallback
    "success": "next_step"
})

# If crash happens, checkpoint allows resume from last successful node
```

---

### Q12. What is time-travel debugging?

**Answer:**
Inspect and replay from any checkpoint to debug agent behavior.

**Code:**
```python
# Run agent
config = {"configurable": {"thread_id": "debug_123"}}
app.invoke({"task": "Research topic"}, config)

# List all checkpoints
checkpoints = app.get_state_history(config)
for checkpoint in checkpoints:
    print(f"Step {checkpoint.step}: {checkpoint.values}")

# Load specific checkpoint (e.g., step 3)
state_at_step_3 = app.get_state(config, checkpoint_id=checkpoint_id_3)
print(state_at_step_3.values)  # See what agent "thought" at step 3

# Modify and replay from that point
app.update_state(config, {"query": "modified query"}, as_node="agent")
result = app.invoke(None, config)  # Continues from modified state

# Use case: Agent made wrong decision at step 3
# → Load step 3, modify state, replay to test different path
```

---

### Q13. Implement a ReAct agent from scratch in LangGraph (without prebuilt).

**Answer:**
Build custom ReAct agent with agent → tools loop.

**Code:**
```python
from langgraph.prebuilt import ToolNode, tools_condition

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Agent node
def agent_node(state: State):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Conditional: agent → tools (if tool_calls) or END
graph.add_conditional_edges("agent", tools_condition)

# Loop: tools → agent
graph.add_edge("tools", "agent")

graph.set_entry_point("agent")
app = graph.compile(checkpointer=MemorySaver())

# Usage
app.invoke({"messages": [("user", "What's 5 + 3?")]}, config)
```

---

### Q14. How does Send() work for fan-out patterns?

**Answer:**
Send() spawns parallel node executions for map-reduce patterns.

**Code:**
```python
from langgraph.types import Send
import operator

class State(TypedDict):
    documents: list
    summaries: Annotated[list, operator.add]  # Reducer merges all

def map_docs(state: State):
    # Fan-out: create parallel tasks
    return [
        Send("summarize", {"doc": doc})
        for doc in state["documents"]
    ]

def summarize(state: dict):
    # Each runs in parallel
    summary = llm.invoke(f"Summarize: {state['doc']}")
    return {"summaries": [summary]}

def reduce(state: State):
    # All summaries merged via reducer
    final = llm.invoke(f"Combine: {state['summaries']}")
    return {"final_report": final}

graph.add_node("map_docs", map_docs)
graph.add_node("summarize", summarize)
graph.add_node("reduce", reduce)
graph.add_conditional_edges("map_docs", map_docs)  # Returns list of Send()
graph.add_edge("summarize", "reduce")

# Flow: map_docs → [summarize, summarize, ...] → reduce
```

---

## 🔰 **Send() vs State Variable**

**Quick Answer:**

| Approach | Execution | Use Case |
|----------|-----------|----------|
| **Send()** | Parallel - spawns multiple node instances | Process items independently |
| **State variable** | Sequential - single node execution | Store data for later |

**Code Comparison:**

**Approach 1: State Variable (Sequential)**
```python
class State(TypedDict):
    documents: list
    summaries: list

def process_docs(state: State):
    summaries = []
    # ❌ Sequential - one at a time
    for doc in state["documents"]:  # 20 docs = 20 seconds
        summary = llm.invoke(f"Summarize: {doc}")  # 1 second each
        summaries.append(summary)
    return {"summaries": summaries}

# Total time: 20 seconds (sequential)
```

**Approach 2: Send() (Parallel)**
```python
from langgraph.types import Send

def map_docs(state: State):
    # ✅ Parallel - spawns 20 nodes at once
    return [
        Send("summarize", {"doc": doc})
        for doc in state["documents"]  # 20 docs = 1 second!
    ]

def summarize(state: dict):
    return {"summaries": [llm.invoke(f"Summarize: {state['doc']}")]}

# Total time: 1 second (all parallel)
```

**Visual Difference:**

```
State Variable (Sequential):
doc1 → process → doc2 → process → doc3 → process
       1s              1s              1s
Total: 3 seconds

Send() (Parallel):
doc1 → process ┐
doc2 → process ├→ All done in 1 second
doc3 → process ┘
```

**When to Use Each:**

```python
# Use State Variable when:
state["user_id"] = "123"  # Store data
state["count"] += 1       # Track something
state["results"].append(x)  # Collect results sequentially

# Use Send() when:
# - Process list items independently
# - Need parallel execution
# - Map-reduce patterns
return [Send("process", {"item": i}) for i in items]
```

**Key Difference:** Send() creates **multiple parallel executions**, state variable just **stores data**.

---

## 🔰 **Send() with Multiple Nodes - How Consolidation Works**

**Your Question:** If Send() spawns 10 parallel executions in node1, do they stay parallel through node2 and node3? How is output consolidated?

**Answer:**

1. **Send() only spawns the target node** - not all downstream nodes
2. **Reducer consolidates** outputs after all parallel executions complete
3. **Next node waits** for consolidation before running

**Visual Flow:**

```
Node1 (fan-out)
    │
    └─→ Send("node2", {item: 1}) ─┐
    └─→ Send("node2", {item: 2}) ─┤
    └─→ Send("node2", {item: 3}) ─┤  ← 10 parallel executions
    └─→ ... (10 total)            │
                                  ▼
                        [Wait for all to complete]
                                  │
                        [Reducer consolidates outputs]
                                  │
                                  ▼
                              Node3 (single execution)
                                  │
                                  ▼
                                 END
```

**Code Example:**

```python
from langgraph.types import Send
import operator

class State(TypedDict):
    items: list
    processed: Annotated[list, operator.add]  # ← Reducer merges results
    final_result: str

# Node 1: Fan-out (spawns 10 parallel executions)
def node1_fanout(state: State):
    return [
        Send("node2_process", {"item": item})
        for item in state["items"]  # 10 items
    ]

# Node 2: Process (runs 10 times in parallel)
def node2_process(state: dict):
    result = f"Processed: {state['item']}"
    # Each execution returns its result
    return {"processed": [result]}  # ← List because of reducer

# Node 3: Consolidate (runs once after all complete)
def node3_consolidate(state: State):
    # state["processed"] now has ALL 10 results merged
    final = ", ".join(state["processed"])
    return {"final_result": final}

# Build graph
graph = StateGraph(State)
graph.add_node("node1_fanout", node1_fanout)
graph.add_node("node2_process", node2_process)
graph.add_node("node3_consolidate", node3_consolidate)

graph.set_entry_point("node1_fanout")

# node1 returns Send() → spawns parallel node2
graph.add_conditional_edges("node1_fanout", node1_fanout)

# All node2 executions → node3 (after consolidation)
graph.add_edge("node2_process", "node3_consolidate")

app = graph.compile()

# Run
result = app.invoke({"items": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
print(result["final_result"])
# "Processed: 1, Processed: 2, ..., Processed: 10"
```

**Step-by-Step Execution:**

```python
# Step 1: node1_fanout runs
# Returns: [Send("node2_process", {item: 1}), Send("node2_process", {item: 2}), ...]

# Step 2: 10 parallel executions of node2_process
# Execution 1: {"processed": ["Processed: 1"]}
# Execution 2: {"processed": ["Processed: 2"]}
# ...
# Execution 10: {"processed": ["Processed: 10"]}

# Step 3: Reducer (operator.add) consolidates ALL results
# state["processed"] = ["Processed: 1", "Processed: 2", ..., "Processed: 10"]

# Step 4: node3_consolidate runs ONCE with consolidated state
# Has access to ALL 10 results in state["processed"]
```

**How Consolidation Works:**

**Without Reducer (WRONG):**
```python
# If processed field has NO reducer
processed: list  # ← Last write wins!

# Result: state["processed"] = ["Processed: 10"]  # Only last one!
```

**With Reducer (CORRECT):**
```python
# With reducer
processed: Annotated[list, operator.add]  # ← Merges all!

# Result: state["processed"] = ["Processed: 1", ..., "Processed: 10"]
```

---

## 🔰 **Why is it called a "Reducer"?**

**Answer:** It **reduces** (combines) multiple state updates into one final value.

**The Problem:**
```python
# 10 parallel nodes all update same field
# Without reducer: ❌ Last write wins
summaries: list  # → only ["Summary 10"]

# With reducer: ✅ Combines all
summaries: Annotated[list, operator.add]
# → ["Summary 1", "Summary 2", ..., "Summary 10"]
```

**How It Works:**
```python
# Reducer merges: current + new → final
operator.add(["Sum 1", "Sum 2"], ["Sum 3"])
# → ["Sum 1", "Sum 2", "Sum 3"]

# Like Python's reduce():
reduce(add, [1,2,3,4,5])  # → 15 (many → one)
```

**Common Reducers:**
```python
# Lists: concatenate
summaries: Annotated[list, operator.add]
# [1,2] + [3,4] → [1,2,3,4]

# Ints: sum
total: Annotated[int, operator.add]
# 5 + 10 → 15

# Custom
def merge_dicts(old, new):
    return {**old, **new}
```

**Why "Reducer":** Reduces `many updates → one value` (like Python's `reduce()`)

---

**Timeline:**

```
Time 0s: node1_fanout starts
Time 0.1s: 10 parallel node2_process start
Time 1s: All 10 node2_process complete
Time 1s: Reducer consolidates outputs
Time 1s: node3_consolidate starts (waits for all)
Time 2s: node3_consolidate completes
```

**Key Points:**

1. ✅ **Send() spawns only target node** (node2), not all downstream
2. ✅ **Parallel executions run simultaneously**
3. ✅ **Graph waits** for ALL parallel executions to complete
4. ✅ **Reducer merges** all outputs into state
5. ✅ **Next node (node3) runs once** with consolidated state

**What if you want node3 to also run in parallel?**

```python
# Chain Send() calls
def node2_process(state: dict):
    result = process(state["item"])
    # Spawn another parallel execution!
    return {
        "processed": [result],
        "next": [Send("node3_parallel", {"data": result})]
    }

# Now node3 also runs 10 times in parallel
```

**Summary:**
- Send() = **Fan-out** (1 → many parallel)
- Reducer = **Fan-in** (many → 1 consolidated)
- Next node waits for fan-in before running

---

### Q15. Difference between graph.invoke(), graph.stream(), graph.astream_events()?

**Answer:**
Different execution modes for different use cases.

**Return Types:**

| Method | Returns | Type |
|--------|---------|------|
| `invoke()` | Final state | `dict` |
| `stream()` | Iterator of node updates | `Iterator[dict]` |
| `astream_events()` | Async iterator of events | `AsyncIterator[dict]` |

**Code:**
```python
# invoke() - blocking, returns final result
result = app.invoke({"messages": ["Hello"]}, config)
# Returns: {'messages': [...], 'final_answer': '...'}
```

**app.stream() - Returns Iterator[dict]**

Each dict = `{node_name: state_update}`

```python
for event in app.stream({"messages": ["Hello"]}, config):
    print(event)

# Output (example):
# {'agent': {'messages': [AIMessage(content='', tool_calls=[...])]}}
# {'tools': {'messages': [ToolMessage(content='8', tool_call_id='...')]}}
# {'agent': {'messages': [AIMessage(content='The answer is 8')]}}
# {'__end__': {'messages': [...]}}  # Final state

# Each event structure:
# {
#   'node_name': {
#     'field1': value1,
#     'field2': value2
#   }
# }
```

**app.astream_events() - Returns AsyncIterator[dict]**

Each dict = detailed event with `event`, `name`, `data` fields

```python
async for event in app.astream_events({"messages": ["Hello"]}, config):
    print(event)

# Output (example events):

# Event 1: Chain starts
{
  'event': 'on_chain_start',
  'name': 'LangGraph',
  'run_id': 'abc-123',
  'tags': [],
  'metadata': {},
  'data': {'input': {'messages': ['Hello']}}
}

# Event 2: Agent node starts
{
  'event': 'on_chain_start',
  'name': 'agent',
  'run_id': 'def-456',
  'parent_ids': ['abc-123'],
  'tags': ['langGraph:step:1'],
  'data': {}
}

# Event 3: LLM starts
{
  'event': 'on_chat_model_start',
  'name': 'ChatOpenAI',
  'run_id': 'ghi-789',
  'data': {
    'input': {
      'messages': [HumanMessage(content='Hello')]
    }
  }
}

# Event 4: LLM token streaming
{
  'event': 'on_chat_model_stream',
  'name': 'ChatOpenAI',
  'run_id': 'ghi-789',
  'data': {
    'chunk': AIMessageChunk(content='The')  # First token
  }
}

# Event 5: More tokens
{
  'event': 'on_chat_model_stream',
  'run_id': 'ghi-789',
  'data': {
    'chunk': AIMessageChunk(content=' answer')  # Second token
  }
}

# Event 6: LLM ends
{
  'event': 'on_chat_model_end',
  'name': 'ChatOpenAI',
  'run_id': 'ghi-789',
  'data': {
    'output': AIMessage(content='The answer is 8')
  }
}

# Event 7: Agent node ends
{
  'event': 'on_chain_end',
  'name': 'agent',
  'run_id': 'def-456',
  'data': {
    'output': {'messages': [AIMessage(content='The answer is 8')]}
  }
}

# Event 8: Graph ends
{
  'event': 'on_chain_end',
  'name': 'LangGraph',
  'run_id': 'abc-123',
  'data': {
    'output': {'messages': [...]}
  }
}
```

**Filtering astream_events:**

```python
async for event in app.astream_events({"messages": ["Hello"]}, config):
    # Only token streams
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
        # Output: "The answer is 8" (streamed)

    # Only node completions
    if event["event"] == "on_chain_end":
        node_name = event["name"]
        print(f"✓ {node_name} completed")
        # Output:
        # ✓ agent completed
        # ✓ tools completed
        # ✓ LangGraph completed
```

**Summary:**

```python
# invoke() → dict
result = app.invoke(...)  # One final dict

# stream() → Iterator of {node: state_update}
for chunk in app.stream(...):  # Dict per node
    print(chunk)  # {'agent': {...}}

# astream_events() → AsyncIterator of detailed events
async for event in app.astream_events(...):  # Dict per event
    print(event)  # {'event': 'on_chat_model_stream', 'data': {...}}
```

**Use Cases:**
- **invoke()**: Get final result only
- **stream()**: Show node-by-node progress (coarse-grained)
- **astream_events()**: Token-level streaming, detailed lifecycle (fine-grained)

---

### Q16. How do you manage shared vs private state in multi-agent systems?

**Answer:**
**Shared state** = Fields that exist in BOTH parent and subgraph
**Private state** = Fields that exist ONLY in subgraph

**Key Rule:** `Only matching field names get synced between parent and subgraph.`

**Example: Research Team**

```python
# Parent State - SHARED across all agents
class TeamState(TypedDict):
    task: str           # ✓ Shared - matches subgraph field
    final_report: str   # ✓ Shared - matches subgraph field

# Researcher Subgraph
class ResearcherState(TypedDict):
    task: str          # ✓ SHARED - matches parent (input from parent)
    queries: list      # ✗ PRIVATE - NOT in parent
    sources: list      # ✗ PRIVATE - NOT in parent
    final_report: str  # ✓ SHARED - matches parent (output to parent)

# Coder Subgraph
class CoderState(TypedDict):
    task: str          # ✓ SHARED - matches parent (input from parent)
    code_drafts: list  # ✗ PRIVATE - NOT in parent
    test_results: list # ✗ PRIVATE - NOT in parent
    final_report: str  # ✓ SHARED - matches parent (output to parent)

# Build subgraphs
researcher = StateGraph(ResearcherState).compile()
coder = StateGraph(CoderState).compile()

# Parent graph
team_graph = StateGraph(TeamState)
team_graph.add_node("researcher", researcher)  # Researcher can't see coder's private state
team_graph.add_node("coder", coder)            # Coder can't see researcher's private state

app = team_graph.compile()

# Run
result = app.invoke({"task": "Build a calculator"})

# Parent sees:
print(result)
# {
#   'task': 'Build a calculator',       ← Shared
#   'final_report': '...'               ← Shared
# }

# Parent does NOT see:
# - researcher's queries, sources       ← Private to researcher
# - coder's code_drafts, test_results   ← Private to coder
```

**Visual:**

```
Parent State (SHARED):
┌─────────────────────┐
│ task: "..."         │ ← All agents see this
│ final_report: "..." │ ← All agents see this
└─────────────────────┘
         │
    ┌────┴────┐
    │         │
Researcher  Coder
    │         │
    ▼         ▼
Private:    Private:
queries     code_drafts
sources     test_results
```

**How It Works:**

```python
# Step 1: Parent calls researcher
# Parent passes: {'task': 'Build calculator'}
# Researcher receives: {'task': 'Build calculator'}

# Step 2: Researcher works (private state)
# Researcher state: {
#   'task': 'Build calculator',
#   'queries': ['how to build calculator'],  ← PRIVATE
#   'sources': ['wikipedia.com'],             ← PRIVATE
#   'final_report': 'Calculator needs +,-,*,/'
# }

# Step 3: Researcher returns to parent
# Parent receives: {'final_report': 'Calculator needs +,-,*,/'}
# Parent does NOT get: queries, sources ← Stayed private!

# Step 4: Parent updates shared state
# Parent state: {
#   'task': 'Build calculator',
#   'final_report': 'Calculator needs +,-,*,/'  ← From researcher
# }
```

**Key Rules:**

1. **Shared**: Fields in parent state + matching fields in subgraph
2. **Private**: Extra fields in subgraph state NOT in parent
3. **Input**: Parent → Subgraph (matching fields only)
4. **Output**: Subgraph → Parent (matching fields only)

**Why Use Private State?**
- Keep implementation details hidden
- Prevent state pollution
- Each agent has its own workspace
- Cleaner interfaces between agents

---

### Q17. Key differences between prebuilt create_react_agent and custom graph?

**Answer:**
Prebuilt for quick start, custom for production control.

**Code:**
```python
# Prebuilt - simple, fast setup
from langgraph.prebuilt import create_react_agent
app = create_react_agent(llm, tools, checkpointer=checkpointer)
# Pros: 5 lines of code
# Cons: Fixed state, no HITL, no multi-agent

# Custom - full control
graph = StateGraph(CustomState)
graph.add_node("agent", agent_with_custom_logic)
graph.add_node("tools", tools_with_validation)
graph.add_node("human_review", human_review_node)
graph.add_conditional_edges("agent", custom_routing)
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # HITL
)
# Pros: Full control, HITL, custom state, error handling
# Cons: More code

# Use prebuilt: Prototypes, simple chatbots
# Use custom: Production, complex workflows, HITL
```

---

### Q18. How does LangGraph ensure thread safety in multi-user deployments?

**Answer:**
Thread isolation via thread_id + PostgresSaver.

**Code:**
```python
# Each user gets unique thread_id
user_1_config = {"configurable": {"thread_id": "user_1"}}
user_2_config = {"configurable": {"thread_id": "user_2"}}

# Concurrent requests - completely isolated
app.invoke({"messages": ["Hi"]}, user_1_config)  # User 1's state
app.invoke({"messages": ["Hello"]}, user_2_config)  # User 2's state

# States are independent
state_1 = app.get_state(user_1_config)  # Only user 1's messages
state_2 = app.get_state(user_2_config)  # Only user 2's messages

# With PostgresSaver - safe across multiple servers
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
app = graph.compile(checkpointer=checkpointer)
# Multiple FastAPI servers can handle requests concurrently
# PostgresSaver ensures no state collision
```

---

### Q19. How do you implement guardrails?

**Answer:**
Validation nodes + conditional edges for safety checks.

**Code:**
```python
def content_safety_check(state: State):
    message = state["messages"][-1].content
    if has_sensitive_data(message):
        return {"safe": False, "reason": "Contains PII"}
    if is_harmful(message):
        return {"safe": False, "reason": "Harmful content"}
    return {"safe": True}

def route_safety(state: State):
    return "proceed" if state.get("safe") else "block"

graph.add_node("safety_check", content_safety_check)
graph.add_node("block", lambda s: {"error": s["reason"]})
graph.add_node("process", process_node)

graph.add_conditional_edges("safety_check", route_safety, {
    "proceed": "process",
    "block": END
})

# Budget guardrail
def token_budget_check(state: State):
    if state["tokens_used"] > state["token_limit"]:
        return interrupt("Budget exceeded")  # HITL
    return {"approved": True}
```

---

### Q20. What is LangGraph Platform / Cloud vs self-hosting?

**Answer:**
Platform is managed, self-hosting is DIY.

**Comparison:**
| Feature | Platform/Cloud | Self-Hosting |
|---------|----------------|--------------|
| Infrastructure | Managed | You manage |
| Checkpointing | Built-in Postgres | Setup PostgresSaver |
| Scaling | Auto | Manual |
| Monitoring | Built-in Studio UI | Custom dashboards |
| Cron jobs | Built-in | Implement yourself |
| Cost | Usage-based fee | Infrastructure only |

**Code - Self-hosting:**
```python
from fastapi import FastAPI
from langgraph.checkpoint.postgres import PostgresSaver

app = FastAPI()
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
graph_app = graph.compile(checkpointer=checkpointer)

@app.post("/invoke")
async def invoke(request: dict):
    result = await graph_app.ainvoke(
        request["input"],
        {"configurable": {"thread_id": request["thread_id"]}}
    )
    return result

# Deploy: uvicorn, docker, k8s, monitoring, scaling - all manual
```

**When to use:**
- **Platform**: Fast launch, managed ops, teams without DevOps
- **Self-hosting**: Full control, on-prem requirements, cost optimization

---

## 🎯 Scenario-Based Questions (10)

### Q1. Deep research agent: search → evaluate → iterate until enough evidence. Design?

**Answer:**
Iterative search with LLM-based evaluation loop.

**Code:**
```python
class State(TypedDict):
    query: str
    search_results: Annotated[list, operator.add]
    iteration_count: int
    sufficient: bool

def search(state: State):
    results = web_search(state["query"])
    return {"search_results": results, "iteration_count": state["iteration_count"] + 1}

def evaluate(state: State):
    prompt = f"Do we have enough evidence? Results: {state['search_results']}"
    response = llm.invoke(prompt)
    return {"sufficient": "yes" in response.content.lower()}

def should_continue(state: State):
    if state["sufficient"] or state["iteration_count"] >= 5:
        return "synthesize"
    return "search"

graph.add_node("search", search)
graph.add_node("evaluate", evaluate)
graph.add_node("synthesize", synthesize_report)
graph.add_conditional_edges("evaluate", should_continue)
graph.add_edge("search", "evaluate")
```

---

### Q2. Trading system: trades >$50K need human approval. Design HITL workflow.

**Answer:**
Dynamic interrupt for large trades.

**Code:**
```python
from langgraph.types import interrupt

def prepare_trade(state: State):
    amount = state["amount"]

    if amount > 50000:
        # Pause for human approval
        approval = interrupt(f"Approve ${amount} trade?")
        if not approval:
            return {"status": "rejected"}

    return {"status": "ready", "trade_details": {...}}

def execute_trade(state: State):
    # Execute the trade
    return {"status": "executed"}

graph.add_node("prepare", prepare_trade)
graph.add_node("execute", execute_trade)
graph.add_edge("prepare", "execute")

app = graph.compile(checkpointer=PostgresSaver(...))

# Small trade - no pause
app.invoke({"amount": 10000}, config)  # Executes immediately

# Large trade - pauses
app.invoke({"amount": 75000}, config)  # Pauses for approval
# Human reviews, then resumes
app.invoke(Command(resume=True), config)
```

---

### Q3. Supervisor coordinating researcher, coder, writer agents. Design?

**Answer:**
See Q9 in conceptual section - same pattern with subgraphs.

**Code:**
```python
# Subgraphs for specialists
researcher_graph = StateGraph(ResearchState)
# ... build researcher graph ...
researcher = researcher_graph.compile()

coder_graph = StateGraph(CodeState)
coder = coder_graph.compile()

# Supervisor
def supervisor_node(state: State):
    prompt = f"Task: {state['task']}, Results: {state['results']}"
    next_agent = llm.invoke(prompt).content
    return {"next": next_agent}

graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)

graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next"])
```

---

### Q4. Agent enters infinite loops between LLM and tool nodes. Diagnose and fix.

**Answer:**
Add iteration guard + loop detection.

**Code:**
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int
    tool_history: list  # Track tool calls

def agent(state: State):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1
    }

def tools_node(state: State):
    # Check for duplicate tool calls
    last_msg = state["messages"][-1]
    if last_msg.tool_calls in state["tool_history"]:
        return {"messages": [("system", "Already called this tool")]}

    result = execute_tools(last_msg.tool_calls)
    return {
        "messages": [result],
        "tool_history": state["tool_history"] + [last_msg.tool_calls]
    }

def should_continue(state: State):
    if state["iteration_count"] >= 10:  # Max iterations
        return "fallback"
    return tools_condition(state)

graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "fallback": "fallback",
    END: END
})
```

---

### Q5. 1,000 concurrent users, each with conversation thread. Architect state management.

**Answer:**
PostgresSaver + unique thread_id per user.

**Code:**
```python
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
app = graph.compile(checkpointer=checkpointer)

# Each user gets unique thread_id
@fastapi_app.post("/chat")
async def chat(user_id: str, message: str):
    config = {"configurable": {"thread_id": f"user_{user_id}"}}
    result = await app.ainvoke({"messages": [message]}, config)
    return result

# Cleanup old threads (cron job)
def cleanup_expired():
    # Delete checkpoints older than 30 days
    db.execute("DELETE FROM checkpoints WHERE created_at < NOW() - INTERVAL '30 days'")
```

---

### Q6. Process 20 documents in parallel, then synthesise. Fan-out/fan-in?

**Answer:**
See Q14 in conceptual section - same Send() pattern.

**Code:**
```python
def fan_out(state: State):
    return [Send("summarize", {"doc": doc}) for doc in state["documents"]]

def summarize(state: dict):
    return {"summaries": [llm.invoke(f"Summarize: {state['doc']}")]}

class State(TypedDict):
    documents: list
    summaries: Annotated[list, operator.add]  # Reducer merges all

graph.add_conditional_edges("fan_out", fan_out)
# All 20 summarize nodes run in parallel, results merged by reducer
```

---

### Q7. Add rollback feature: undo wrong agent action, retry from previous state.

**Answer:**
Use checkpoint time-travel to rollback.

**Code:**
```python
config = {"configurable": {"thread_id": "session_123"}}

# Agent makes mistake at step 5
app.invoke({"task": "Send email"}, config)

# List checkpoints
history = app.get_state_history(config)
for checkpoint in history:
    print(f"Step {checkpoint.metadata['step']}: {checkpoint.values}")

# Rollback to step 4 (before mistake)
checkpoint_id_4 = list(history)[4].config["configurable"]["checkpoint_id"]
rollback_config = {"configurable": {"thread_id": "session_123", "checkpoint_id": checkpoint_id_4}}

# Modify state
app.update_state(rollback_config, {"email": "corrected email"})

# Resume from corrected state
app.invoke(None, config)  # Continues from step 4 with correction
```

---

### Q8. External API has 30% failure rate and 10 req/min rate limit. Resilience strategy.

**Answer:**
Retry + rate limiting + fallback.

**Enterprise Rate Limiting Strategy:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE RATE LIMITING                      │
└─────────────────────────────────────────────────────────────────┘

         ┌──────────────┐
         │   Request    │
         │    Queue     │
         └──────┬───────┘
                │
                ▼
    ┌────────────────────────┐
    │   Token Bucket Filter  │──── 10 tokens/min (API limit)
    │   (Sliding Window)     │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  Exponential Backoff   │──── Retry 3 times (30% failure)
    │   + Circuit Breaker    │──── Open after 5 failures
    └────────┬───────────────┘
             │
             ├─── Success ──────► Return Result
             │
             └─── All Failed ───► Fallback Strategy
                                  ├─ Cached Response
                                  ├─- Default Response
                                  └─- Alternative API
```

**Rate Limiting Techniques:**

**1. Token Bucket (Preferred for Enterprise)**
- Bucket capacity = 10 tokens (API limit)
- Refill rate = 10 tokens/60 seconds
- Request consumes 1 token
- If bucket empty → Queue or reject

**2. Sliding Window Counter**
- Track timestamps of last 10 requests
- Before new request: Remove timestamps > 60 seconds old
- If count < 10 → Allow
- If count ≥ 10 → Calculate wait time

**Code:**
```python
import time
from collections import deque
from functools import wraps

# Token Bucket Rate Limiter
class TokenBucket:
    def __init__(self, capacity=10, refill_rate=10/60):  # 10 req/min
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()

    def consume(self, tokens=1):
        """Try to consume tokens. Returns True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def wait_time(self):
        """Calculate seconds to wait for next token."""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.refill_rate

# Circuit Breaker Pattern
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Initialize
rate_limiter = TokenBucket(capacity=10, refill_rate=10/60)
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

def api_node(state: State):
    # 1. Rate Limiting - Token Bucket
    if not rate_limiter.consume():
        wait_time = rate_limiter.wait_time()
        time.sleep(wait_time)
        rate_limiter.consume()

    # 2. Circuit Breaker + Retry with Exponential Backoff
    for attempt in range(3):
        try:
            result = circuit_breaker.call(external_api.call, state["query"])
            return {"result": result, "error": None}
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
            else:
                # All retries failed
                return {"error": str(e), "use_fallback": True}

# 3. Fallback Strategy
def route(state: State):
    if state.get("use_fallback"):
        return "fallback"
    return "success"

def fallback_node(state: State):
    # Try cached response or default
    cached = cache.get(state["query"])
    if cached:
        return {"result": cached, "source": "cache"}
    return {"result": "Default response", "source": "fallback"}

graph.add_node("api", api_node)
graph.add_node("fallback", fallback_node)
graph.add_conditional_edges("api", route)
```

**Why This Works in Enterprises:**

1. **Token Bucket** → Smooth traffic, prevents burst overload
2. **Circuit Breaker** → Stop hitting failing API, prevent cascade failures
3. **Exponential Backoff** → Give API time to recover (1s → 2s → 4s)
4. **Fallback** → Always return something (cached/default)

**Rate Limit: 10 req/min = Request every 6 seconds max**

---

### Q9. Coding assistant: LLM writes code, executes in sandbox, iterates if tests fail. Design.

**Answer:**
Iterative code generation with test feedback loop.

**Code:**
```python
class State(TypedDict):
    task: str
    code: str
    test_result: str
    iteration_count: int

def write_code(state: State):
    prompt = f"Task: {state['task']}\nPrevious error: {state.get('test_result', '')}"
    code = llm.invoke(prompt).content
    return {"code": code, "iteration_count": state["iteration_count"] + 1}

def execute(state: State):
    try:
        result = sandbox.run(state["code"])
        return {"test_result": "pass" if result.success else result.error}
    except Exception as e:
        return {"test_result": str(e)}

def should_retry(state: State):
    if state["test_result"] == "pass":
        return "finish"
    elif state["iteration_count"] >= 5:
        return "give_up"
    return "write_code"

graph.add_node("write_code", write_code)
graph.add_node("execute", execute)
graph.add_conditional_edges("execute", should_retry)
graph.add_edge("write_code", "execute")
```

---

### Q10. Full auditability required: every LLM call, tool execution, state transition logged. How?

**Answer:**
Callbacks + checkpointing + audit log in state.

**Code:**
```python
from langchain.callbacks.base import BaseCallbackHandler
import json

class AuditLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        log_entry = {
            "type": "llm_start",
            "timestamp": time.time(),
            "prompts": prompts,
            "model": serialized.get("name")
        }
        elasticsearch.index("audit", log_entry)

    def on_llm_end(self, response, **kwargs):
        log_entry = {
            "type": "llm_end",
            "timestamp": time.time(),
            "response": response.generations,
            "tokens": response.llm_output.get("token_usage")
        }
        elasticsearch.index("audit", log_entry)

# Add to state
class State(TypedDict):
    messages: list
    audit_log: Annotated[list, operator.add]

def node_with_logging(state: State):
    result = llm.invoke(state["messages"], callbacks=[AuditLogger()])
    return {
        "messages": [result],
        "audit_log": [{"node": "agent", "timestamp": time.time()}]
    }

# Checkpoints store full state snapshots
app = graph.compile(checkpointer=PostgresSaver(...))

# LangSmith for visual audit
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

---
---

# Part 4 — Agent Design Patterns

---

### Q: When to use Multi-Agent vs Single Agent?

**Answer:**

**Single Agent** — use when:
- Task fits in one context window
- Linear sequential steps that share full context
- < 5–7 tools (agent stays focused)
- Low latency is critical (no coordination overhead)
- Simplicity in debugging is preferred

**Multi-Agent** — switch when hitting any of:
- **Context overflow** — 500-page doc + history + tools > context limit
- **Parallelism** — independent sub-tasks that can run simultaneously (3x speedup)
- **Specialization** — different domains need different system prompts + tools
- **Fault isolation** — one agent fails without killing others
- **Human-in-the-loop at boundaries** — natural pause points between agents

```
Single Agent:  One coherent task → linear steps → answer
Multi-Agent:   Orchestrator → [Research Agent] + [Code Agent] + [Writer Agent]
                              (parallel)         (parallel)      (waits for both)
```

**Decision Rule:** Start with single agent. Switch to multi-agent only when hitting context limits, parallelism needs, specialization needs, or fault isolation needs.

| Signal | Single | Multi |
|--------|:------:|:-----:|
| Task fits in one context window | ✅ | |
| Independent parallel sub-tasks | | ✅ |
| Different system prompt per domain | | ✅ |
| > 10 tools | | ✅ |
| Low latency required | ✅ | |
| Fault isolation per sub-task | | ✅ |

---

### Q: What are the callback options in LangGraph?

**Answer:** Three mechanisms at different levels:

**1. `RunnableConfig` callbacks** — injected at invoke time, fires on every LLM/tool/node event:
```python
class MyHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Node '{serialized['name']}' start")
    def on_llm_end(self, response, **kwargs):
        print(f"Tokens: {response.llm_output.get('token_usage')}")
    def on_tool_error(self, error, **kwargs):
        print(f"Tool ERROR: {error}")

app.invoke(input, config={"callbacks": [MyHandler()]})
```

**2. `astream_events()`** — LangGraph native structured event stream, most powerful:
```python
async for event in app.astream_events(input, version="v2"):
    if event["event"] == "on_chain_start":
        print(f"Node: {event['name']}")
    elif event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")  # real-time tokens
    elif event["event"] == "on_tool_start":
        print(f"Tool: {event['name']}")
```

**3. `with_listeners()`** — graph-level pre/post hooks on compiled graph:
```python
app = graph.compile().with_listeners(
    on_start=lambda run: print(f"Graph start: {run.id}"),
    on_end=lambda run: print(f"Graph done: {run.status}"),
    on_error=lambda run: print(f"Graph FAILED: {run.error}")
)
```

| Mechanism | Scope | Token Streaming | Use Case |
|-----------|-------|:---------------:|----------|
| `RunnableConfig callbacks` | LLM + tool + node | Yes | Logging, APM, tracing |
| `astream_events()` | Every layer + node name | Yes (per token) | Real-time UI, observability |
| `with_listeners()` | Whole graph only | No | SLA timing, audit, alerting |

---

### Q: How to design fault-tolerant and long-running agent workflows?

**Answer:** Seven layers of resilience:

**1. Checkpointing — Resume from crash:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("./workflow.db")
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "task-001"}}
app.invoke(initial_state, config=config)   # first run
app.invoke(None, config=config)            # resume after crash
```

**2. Retry with exponential backoff:**
```python
def with_retry(max_attempts=3, backoff_base=2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(state)
                except Exception as e:
                    if attempt == max_attempts: raise
                    time.sleep(backoff_base ** (attempt - 1))
        return wrapper
    return decorator

@with_retry(max_attempts=3)
def call_llm_node(state): ...
```

**3. Idempotency — Safe to retry:**
```python
def send_email_node(state):
    key = f"send_email_{state['task_id']}"
    if key in state["completed_steps"]:
        return {}                          # skip if already done
    send_email(...)
    return {"completed_steps": state["completed_steps"] | {key}}
```

**4. Error routing — Fallback nodes:**
```python
def route(state):
    if state.get("error") and state["retry_count"] < 3:
        return "retry"
    elif state.get("error"):
        return "fallback"
    return "next"
```

**5. Timeout — No infinite hangs:**
```python
async def node(state):
    return await asyncio.wait_for(llm.ainvoke(...), timeout=30.0)
```

**6. Human-in-the-loop — Pause at risky steps:**
```python
app = graph.compile(checkpointer=checkpointer, interrupt_before=["publish"])
app.invoke(input, config=config)                          # pauses at publish
app.invoke({"approved": True}, config=config)             # resume after human
```

**Architecture summary:**
```
Long-Running Workflow
├── Checkpointing     → Crash recovery, resume
├── Idempotency       → Safe retries, no double execution
├── Retry + Backoff   → Transient failure handling
├── Error Routing     → Fallback nodes, graceful degradation
├── Timeouts          → No stuck nodes
└── Human-in-Loop     → Pause at risky decisions
```

---

### Q: How to call MCP (Model Context Protocol) from LangChain and LangGraph?

**Answer:**

```bash
pip install langchain-mcp-adapters mcp langgraph
```

**LangChain with MCP:**
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    },
    "remote-server": {
        "url": "http://localhost:8000/sse",
        "transport": "sse"
    }
}) as client:
    tools = await client.get_tools()          # MCP tools → LangChain tools
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke([HumanMessage("List /tmp files")])
```

**LangGraph ReAct Agent with MCP:**
```python
from langgraph.prebuilt import create_react_agent

async with MultiServerMCPClient({...}) as client:
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)    # drop-in — ToolNode handles execution
    result = await agent.ainvoke({"messages": [HumanMessage("...")]})
```

**Custom MCP Server (Python):**
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("my-server")

@app.list_tools()
async def list_tools():
    return [types.Tool(name="get_weather", description="...", inputSchema={...})]

@app.call_tool()
async def call_tool(name, arguments):
    if name == "get_weather":
        return [types.TextContent(type="text", text=f"22°C sunny")]
```

**Key pattern:**
```
MultiServerMCPClient       # manages MCP server subprocess lifecycle
    └── get_tools()        # MCP tools → LangChain BaseTool objects
            └── ToolNode   # LangGraph executes them automatically
```

`async with` is required — starts the MCP subprocess on entry, terminates cleanly on exit.

| Transport | When | Config |
|-----------|------|--------|
| `stdio` | Local process (npx, python) | `command` + `args` |
| `sse` | Remote HTTP server | `url` |

---

---

### Q: How to effectively handle tool selection, fallback logic, execution and retries within Agent workflow?

**Answer:**

**1. Tool Selection — Write precise descriptions (LLM reads these to choose):**
```python
@tool
def search_web(query: str) -> str:
    """Search internet for CURRENT events, live prices, recent news.
    Use when: today's data needed. Do NOT use for: historical facts or math."""
    return DuckDuckGoSearchRun().run(query)

@tool
def query_database(sql: str) -> str:
    """Query internal SQL DB for company orders, users, revenue.
    Use when: internal business data needed. Do NOT use for: external info."""
    return db.execute(sql)
```

**2. Rule-based pre-selection (filter tools before LLM sees them):**
```python
import re

RULES = [
    (r"\b(today|latest|live|current)\b", ["search_web"]),
    (r"\b(sql|order|customer|revenue)\b", ["query_database"]),
    (r"\b(calculate|math|\d+[\+\-\*/])\b", ["calculate"]),
]

def select_tools(query: str) -> list:
    matched = set()
    for pattern, tools in RULES:
        if re.search(pattern, query, re.I):
            matched.update(tools)
    return [ALL_TOOLS[n] for n in matched] or list(ALL_TOOLS.values())
```

**3. Fallback chain — primary → secondary → cached:**
```python
import functools

def with_fallback(*fallbacks):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for candidate in [fn] + list(fallbacks):
                try:
                    return candidate(*args, **kwargs)
                except Exception as e:
                    print(f"[FALLBACK] {candidate.__name__} failed: {e}")
            raise RuntimeError("All fallbacks exhausted")
        return wrapper
    return decorator

@tool
@with_fallback(duckduckgo_search, cached_search)   # tries each in order
def search_web(query: str) -> str:
    """Search the web."""
    return brave_search(query)   # primary
```

**4. Retry with exponential backoff:**
```python
import time, functools

def with_retry(max_attempts=3, backoff=2.0):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts: raise
                    sleep = backoff ** (attempt - 1)   # 1s → 2s → 4s
                    print(f"[RETRY] attempt {attempt} failed, retrying in {sleep}s")
                    time.sleep(sleep)
        return wrapper
    return decorator

@with_retry(max_attempts=3)
def call_llm_node(state): ...
```

**5. LangGraph fallback routing:**
```python
def route_after_tool(state) -> str:
    error = state.get("tool_error")
    if not error:            return "agent"       # success — loop back
    if state["retries"] < 2: return "retry"       # transient — retry
    return "fallback"                             # exhausted — use fallback

def fallback_node(state):
    """Answer from model knowledge when all tools fail."""
    response = llm.invoke([*state["messages"],
        HumanMessage("Answer from your training knowledge only.")])
    return {"messages": [response]}

graph.add_conditional_edges("tools", route_after_tool, {
    "agent":    "agent",
    "retry":    "increment_retry",
    "fallback": "fallback"
})
```

**Decision flow:**
```
Tool called
    ├── Success                       → return result
    ├── Transient error (network)     → retry with backoff (1s → 2s → 4s)
    ├── Repeated failure              → fallback tool chain
    └── All fallbacks exhausted       → graceful degrade (LLM knowledge only)
```

| Layer | Mechanism | Protects Against |
|-------|-----------|-----------------|
| **Selection** | Rules / descriptions | Wrong tool chosen |
| **Fallback chain** | `with_fallback(*fallbacks)` | Single tool down |
| **Retry** | Exponential backoff | Transient errors |
| **Graph routing** | Conditional edges | Cascading failures |
| **Graceful degrade** | LLM-only answer | All tools unavailable |

---

### Q: How to enable modularity when building agents with multiple tools or context?

**Core idea**: Separate concerns into registry, context, and subgraph layers so each piece can evolve independently.

**Pattern 1 — Tool Registry (dynamic loading)**
```python
from langchain_core.tools import tool

ALL_TOOLS: dict[str, BaseTool] = {}

def register(fn):
    t = tool(fn)
    ALL_TOOLS[t.name] = t
    return t

@register
def search_web(query: str) -> str:
    """Search the web for current information."""
    return f"Results for: {query}"

@register
def query_database(sql: str) -> str:
    """Run a SQL query against the database."""
    return f"DB results for: {sql}"

# At runtime, inject only relevant tools
def get_tools_for_domain(domain: str) -> list:
    mapping = {
        "research": ["search_web"],
        "analytics": ["query_database"],
    }
    return [ALL_TOOLS[n] for n in mapping.get(domain, list(ALL_TOOLS))]
```

**Pattern 2 — AgentContext (shared state object)**
```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentContext:
    user_id: str
    domain: str
    memory: dict[str, Any] = field(default_factory=dict)
    tools: list = field(default_factory=list)

    def add_memory(self, key: str, value: Any):
        self.memory[key] = value

# Pass context through LangGraph state
class AgentState(TypedDict):
    messages: list
    ctx: AgentContext
```

**Pattern 3 — Plug-in Subgraphs**
```python
from langgraph.graph import StateGraph, END

def build_research_subgraph():
    g = StateGraph(AgentState)
    g.add_node("search", search_node)
    g.add_node("summarize", summarize_node)
    g.add_edge("search", "summarize")
    g.add_edge("summarize", END)
    return g.compile()

def build_analytics_subgraph():
    g = StateGraph(AgentState)
    g.add_node("query", query_node)
    g.add_node("chart", chart_node)
    g.add_edge("query", "chart")
    g.add_edge("chart", END)
    return g.compile()

# Compose into a parent graph
main = StateGraph(AgentState)
main.add_node("research", build_research_subgraph())
main.add_node("analytics", build_analytics_subgraph())
main.add_conditional_edges("router", route_fn, {
    "research": "research",
    "analytics": "analytics",
})
```

**Summary table**

| Layer | Pattern | Benefit |
|---|---|---|
| Tools | `ToolRegistry` | Add/remove tools without changing agent core |
| State | `AgentContext` | Shared memory and config across nodes |
| Graph | Plug-in subgraphs | Swap domain logic independently |
| Routing | Conditional edges | Dynamic dispatch to correct subgraph |


---

## Q: Why must you use an Async checkpointer for concurrent LangGraph requests, and how does it work?

**Answer:**

LangGraph's `ainvoke` is an async coroutine — it runs on Python's `asyncio` event loop. When the checkpointer writes or reads state, it must also be non-blocking, otherwise it **freezes the event loop** and all other concurrent requests stall.

| Checkpointer | Async? | DB Client | Problem |
|---|---|---|---|
| `MemorySaver` | No | Python `dict` | Blocks event loop — dev only |
| `AsyncSqliteSaver` | Yes | `aiosqlite` | Non-blocking, single machine |
| `AsyncPostgresSaver` | Yes | `asyncpg` | Non-blocking, multi-pod prod |
| `AsyncRedisSaver` | Yes | `aioredis` | Non-blocking + TTL expiry |

---

**How it works:**

Each `ainvoke` with the same `thread_id` loads that user's latest checkpoint from the DB, runs the graph, then saves the updated state — all async:

```
Request arrives → async load checkpoint (thread_id=alice) → run nodes → async save checkpoint
                  ↑ event loop free during DB I/O — 100 other requests run concurrently
```

---

**Concrete example — 3 users, 3 pods, 1 shared Postgres:**

```python
import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Build once at startup — shared checkpointer across all workers
async def build_agent():
    cp = AsyncPostgresSaver.from_conn_string("postgresql://user:pass@db:5432/lg")
    await cp.setup()                    # creates tables if needed
    return create_react_agent(ChatOpenAI(model="gpt-4o"), tools=[], checkpointer=cp)

async def handle(agent, user_id: str, msg: str) -> str:
    config = {"configurable": {"thread_id": user_id}}   # isolates each user
    resp = await agent.ainvoke({"messages": [{"role": "user", "content": msg}]}, config=config)
    return resp["messages"][-1].content

async def main():
    agent = await build_agent()

    # Turn 1 — 3 users concurrently (event loop juggles all 3)
    await asyncio.gather(
        handle(agent, "alice", "My order #123 is missing"),
        handle(agent, "bob",   "I need a refund for #456"),
        handle(agent, "carol", "Track my shipment #789"),
    )

    # Turn 2 — can arrive on a DIFFERENT pod
    # Each ainvoke loads ONLY that user's history from Postgres
    await asyncio.gather(
        handle(agent, "alice", "It was supposed to arrive Monday"),
        handle(agent, "bob",   "I paid by credit card"),
        handle(agent, "carol", "It's been 5 days"),
    )
    # alice remembers "order #123" | bob remembers "#456" | carol remembers "#789"
```

**What Postgres stores (per user, per node):**

```
thread_id   checkpoint_id   node       state_snapshot
─────────   ─────────────   ────       ──────────────────────────────────
alice       chk-001         __start__  {messages: [Turn 1 msg]}
alice       chk-002         agent      {messages: [Turn 1 msg + LLM reply]}
alice       chk-003         agent      {messages: [Turn 2 msg + LLM reply]}
bob         chk-001         __start__  {messages: [Turn 1 msg]}
carol       chk-001         __start__  {messages: [Turn 1 msg]}
```

**Environment-based setup:**

```python
async def get_checkpointer():
    env = os.environ.get("ENV", "dev")
    if env == "dev":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        cp = AsyncSqliteSaver.from_conn_string("checkpoints.db")   # local file
    else:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        cp = AsyncPostgresSaver.from_conn_string(
            os.environ["POSTGRES_URL"],
            pool_size=20        # match uvicorn worker count
        )
    await cp.setup()
    return cp
```