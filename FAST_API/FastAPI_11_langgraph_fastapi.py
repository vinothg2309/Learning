# 11_langgraph_fastapi.py


"""
========================================================
TOPIC 11: Serving LangGraph in FastAPI
========================================================

KEY CONCEPTS:
- graph.compile() is EXPENSIVE — builds the state machine, validates
  edges, wires up nodes. Do it ONCE at startup via lifespan.

- graph.invoke() / graph.astream() is CHEAP — just runs the compiled
  graph with a given input. Safe to call per request, concurrently.

- Compiled graph is STATELESS between invocations — it holds no
  conversation memory itself. Pass state (messages, thread_id) in
  the input dict each time.

- For multi-turn memory: use a Checkpointer (MemorySaver / Redis).
  Pass config={"configurable": {"thread_id": session_id}} per request.
  The checkpointer stores & restores state between turns automatically.

PATTERN:
  lifespan → compile graph → store in app.state
  endpoint → app.state.graph.ainvoke(input, config)

INTERVIEW ANSWER TIP:
  "I compile the LangGraph once at startup and store the compiled graph
   in app.state. Each request calls ainvoke() with the user's input and
   a thread_id for memory isolation. The checkpointer (Redis in prod)
   persists conversation state between turns without any manual state
   management in the endpoint."
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── LangGraph imports ──────────────────────────────────────────
# pip install langgraph langchain-openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # in-process checkpointer
# from langgraph.checkpoint.redis import RedisSaver  # production checkpointer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import os, sys
RAG_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', 'RAG'))
if RAG_PATH not in sys.path:
    sys.path.insert(0, RAG_PATH)
from llm_config import get_llm

# ──────────────────────────────────────────────────────────────
# 11A. Define Graph State & Nodes
# ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The state passed between graph nodes.
    add_messages is a reducer: new messages are APPENDED, not replaced.
    This preserves the full conversation history across turns.
    """
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph() -> StateGraph:
    """
    Define nodes and edges ONCE.
    Separating build from compile makes testing easier.
    """


    llm = get_llm(model="gpt-4o-mini", temperature=0.7)

    # ── Nodes ──────────────────────────────────────────────────
    async def call_llm(state: AgentState) -> dict:
        """Single node: pass messages to LLM, stream response chunks."""
        # Use ainvoke for async support (collect all chunks into final response)
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}   # add_messages appends this

    # ── Graph wiring ───────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_edge(START, "llm")
    graph.add_edge("llm", END)

    return graph


# ──────────────────────────────────────────────────────────────
# 11B. Lifespan — compile graph ONCE at startup
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    compile() is called here — ONCE per process lifetime.
    All requests share the same compiled graph object (it's thread-safe
    for reading; state is isolated per invocation via input dict).
    """
    print("[STARTUP] Compiling LangGraph...")

    # MemorySaver: stores checkpoints in-process (dev/single-worker only).
    # For production / multi-worker: use RedisSaver so all workers share state.
    #   checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")
    checkpointer = MemorySaver()

    compiled = build_graph().compile(checkpointer=checkpointer)
    app.state.graph = compiled          # store for all endpoints to use

    print("[STARTUP] LangGraph ready.")
    yield
    print("[SHUTDOWN] Cleaning up.")
    # No explicit cleanup needed for in-memory checkpointer.
    # For Redis: await checkpointer.aclose()


app = FastAPI(title="LangGraph FastAPI", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────
# 11C. Request / Response schemas
# ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"     # thread_id for memory isolation


class ChatResponse(BaseModel):
    reply: str
    session_id: str


# ──────────────────────────────────────────────────────────────
# 11D. Single-turn invoke endpoint
# ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """
    Invoke the compiled graph once per request.

    config["configurable"]["thread_id"]:
      - Tells the checkpointer which conversation thread to load/save.
      - Different session_ids = isolated conversation histories.
      - Same session_id across requests = multi-turn memory.

    graph.ainvoke() returns the FINAL state after all nodes complete.
    """
    graph = request.app.state.graph   # compiled graph from lifespan

    config = {"configurable": {"thread_id": body.session_id}}

    final_state = await graph.ainvoke(
        {"messages": [HumanMessage(content=body.message)]},
        config=config,
    )

    # Last message in state is the AI's reply
    reply = final_state["messages"][-1].content

    return ChatResponse(reply=reply, session_id=body.session_id)


# ──────────────────────────────────────────────────────────────
# 11E. Streaming endpoint — yield tokens as they are generated
# ──────────────────────────────────────────────────────────────

async def stream_graph_tokens(graph, user_message: str, session_id: str) -> AsyncIterator[str]:
    """
    astream_events() yields fine-grained events including individual tokens.
    Filter for 'on_chat_model_stream' to get tokens as they arrive.
    """
    config = {"configurable": {"thread_id": session_id}}

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
        version="v2",                 # use v2 event schema
    ):
        kind = event["event"]

        # on_chat_model_stream fires for each token chunk from the LLM
        if kind == "on_chat_model_stream":
            chunk_content = event["data"]["chunk"].content
            if chunk_content:
                yield chunk_content   # send token to client immediately


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """
    Streams LLM tokens back to the client as they are generated.
    Client sees partial responses immediately — same as ChatGPT UI.

    Test with curl:
        curl -N -X POST "http://localhost:8011/chat/stream" \\
             -H "Content-Type: application/json" \\
             -d '{"message": "explain transformers", "session_id": "user-1"}'
    """
    graph = request.app.state.graph

    return StreamingResponse(
        stream_graph_tokens(graph, body.message, body.session_id),
        media_type="text/plain",
    )



async def raw_stream_graph_tokens(graph, user_message: str, session_id: str) -> AsyncIterator[str]:
    """
    Use astream_events() to capture 'on_chat_model_stream' for token-by-token streaming.
    Yields individual tokens as they arrive from the LLM.
    """
    config = {"configurable": {"thread_id": session_id}}

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        ):
            # Filter for LLM streaming events (token-by-token)
            if event.get("event") == "on_chat_model_stream":
                # Structure: {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content="token")}}
                chunk = event.get("data", {}).get("chunk")
                if chunk:
                    # Extract token from the chunk
                    token = getattr(chunk, "content", "")
                    if token:  # only yield non-empty tokens
                        yield token
                        await asyncio.sleep(0.005)  # small delay for smooth streaming
    except Exception as e:
        yield f"\n[ERROR] {str(e)}\n"

@app.post("/chat/raw_stream")
async def chat_stream(body: ChatRequest, request: Request):
    """
    Streams LLM tokens back to the client as they are generated.
    Client sees partial responses immediately — same as ChatGPT UI.

    Test with curl:
        curl -N -X POST "http://localhost:8011/chat/raw_stream" \\
             -H "Content-Type: application/json" \\
             -d '{"message": "explain transformers", "session_id": "user-1"}'
    """
    graph = request.app.state.graph

    return StreamingResponse(
        raw_stream_graph_tokens(graph, body.message, body.session_id),
        media_type="text/plain",
    )

# ──────────────────────────────────────────────────────────────
# 11F. Session history endpoint — inspect stored state
# ──────────────────────────────────────────────────────────────

@app.get("/chat/{session_id}/history")
async def get_history(session_id: str, request: Request):
    """
    Retrieve the full conversation history for a session.
    The checkpointer stores this; we just fetch and return it.
    """
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": session_id}}

    # get_state returns the latest checkpoint for this thread
    state_snapshot = graph.get_state(config)

    if not state_snapshot or not state_snapshot.values:
        return {"session_id": session_id, "messages": []}

    messages = state_snapshot.values.get("messages", [])
    return {
        "session_id": session_id,
        "turn_count": len([m for m in messages if isinstance(m, HumanMessage)]),
        "messages": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant",
             "content": m.content}
            for m in messages
        ],
    }


# ──────────────────────────────────────────────────────────────
# 11G. Key patterns summary (as comments for quick reference)
# ──────────────────────────────────────────────────────────────

"""
COMPILE vs INVOKE — always separate:

    # STARTUP (once):
    compiled = graph.compile(checkpointer=checkpointer)
    app.state.graph = compiled

    # PER REQUEST (many times):
    result = await app.state.graph.ainvoke(input, config)


MEMORY ISOLATION via thread_id:

    session-A → {"configurable": {"thread_id": "session-A"}}  → own history
    session-B → {"configurable": {"thread_id": "session-B"}}  → own history


CHECKPOINTER CHOICE:

    Dev   → MemorySaver()                  (in-process, lost on restart)
    Prod  → RedisSaver / PostgresSaver     (persistent, shared across workers)


MULTI-WORKER GOTCHA:
    MemorySaver is in-process — different workers have separate memory.
    With gunicorn -w 4, session history is NOT shared across workers.
    Fix: use RedisSaver so all workers share one Redis instance.
"""


if __name__ == "__main__":
    uvicorn.run("11_langgraph_fastapi:app", host="0.0.0.0", port=8011, reload=False)
    # reload=False — lifespan + compiled graph conflicts with hot-reload

