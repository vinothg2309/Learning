MCP.md
# Model Context Protocol (MCP)

> **MCP is an open standard that defines how AI models communicate with external tools, data sources, and services.**

---
## 1. Why MCP Exists

### The Problem: M × N Integration Hell

Before MCP, every AI application had to build custom integrations for every tool it needed.

```
WITHOUT MCP — M×N custom integrations:

  Claude ──────── GitHub connector
  Claude ──────── Slack connector
  Claude ──────── Postgres connector
  GPT-4  ──────── GitHub connector   ← duplicate work
  GPT-4  ──────── Slack connector    ← duplicate work
  GPT-4  ──────── Postgres connector ← duplicate work
  Gemini ──────── GitHub connector   ← duplicate work again
  ...

  3 models × 3 tools = 9 custom connectors to maintain

WITH MCP — M+N integrations:

  Claude ─┐
  GPT-4  ─┤─── MCP ───┬─── GitHub Server
  Gemini ─┘           ├─── Slack Server
                      └─── Postgres Server

  3 models + 3 MCP servers = 6 things to maintain
```

### Key Motivations

| Pain Point | MCP Solution |
|---|---|
| Every app rebuilds tool integrations from scratch | One MCP server → works with any MCP client |
| No standard for tool discovery | Servers declare their capabilities at runtime |
| Context lost between tool calls | Persistent server state across the session |
| Security is ad-hoc per integration | Standardized transport + permission model |
| Hard to swap AI providers | Clients and servers are decoupled |

### Analogy

```
USB-C is to hardware   what   MCP is to AI tooling

USB-C: one cable standard → any device works with any charger
MCP:   one protocol standard → any AI works with any tool server
```
---
## 2. Core MCP Concepts

MCP defines **four primitives** that servers can expose:

```
┌─────────────────────────────────────────────────────┐
│                  MCP SERVER                         │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌─────┐ │
│  │  TOOLS   │  │RESOURCES │  │ PROMPTS │  │ROOTS│ │
│  │          │  │          │  │         │  │     │ │
│  │ Functions│  │Read-only │  │Reusable │  │File │ │
│  │ AI can   │  │ data the │  │ prompt  │  │sys  │ │
│  │ invoke   │  │ AI reads │  │templates│  │scope│ │
│  └──────────┘  └──────────┘  └─────────┘  └─────┘ │
└─────────────────────────────────────────────────────┘
```

### 2.1 Tools — Functions the AI can call

Tools are the primary way AI models take actions in the world.
```python
# Tool schema structure — what the client sees
tool_schema = {
    "name": "search_web",
    "description": "Search the web and return top results",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

print("Tool name:", tool_schema["name"])
print("Required params:", tool_schema["inputSchema"]["required"])
print("Optional params:", [k for k in tool_schema["inputSchema"]["properties"] 
                           if k not in tool_schema["inputSchema"]["required"]])
```
```python
# Minimal working MCP server with FastMCP
# pip install mcp

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def greet(name: str, formal: bool = False) -> str:
    """Generate a greeting for the given name."""
    if formal:
        return f"Good day, {name}."
    return f"Hey {name}!"

# FastMCP auto-generates JSON schema from type hints + docstrings
# No manual schema definition needed
print("Server name:", mcp.name)
print("Registered tools:", [t for t in dir(mcp) if not t.startswith('_')][:5])
```
### 2.2 Resources — Read-only data the AI can access

Resources are like files or database records — the AI can read them but **cannot modify** them through the resource interface.

```
┌───────────────────────────────────────────────────────────────────────┐
│                    RESOURCE ACCESS FLOW                               │
│                                                                       │
│  SERVER side:                                                         │
│    @mcp.resource("users://{user_id}/profile")  ← register URI        │
│    def get_profile(user_id: str) -> str: ...                          │
│                                                                       │
│  CLIENT side:                                                         │
│    Step 1: resources/list  → discover available URIs                  │
│    Step 2: resources/read  → fetch content by URI                     │
│    Step 3: resources/subscribe  → watch for changes (optional)        │
│                                                                       │
│  URI types:                                                           │
│    Static:   config://app/settings          ← fixed URI              │
│    Template: users://{user_id}/profile      ← parameterised URI      │
└───────────────────────────────────────────────────────────────────────┘
```
```python
# SERVER SIDE: Register resources with @mcp.resource

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="resource-demo")

# ── Static resource — fixed URI, same content every time ──────────────────
@mcp.resource("config://app/settings")
def get_settings() -> str:
    """Return application configuration."""
    return json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 8096,
        "temperature": 0.7
    }, indent=2)

# ── Dynamic resource template — URI contains a variable {user_id} ─────────
# The variable is automatically extracted from the URI and passed as argument
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Return user profile data for a given user_id."""
    profiles = {
        "alice": {"name": "Alice", "role": "admin",  "email": "alice@example.com"},
        "bob":   {"name": "Bob",   "role": "viewer", "email": "bob@example.com"},
    }
    return json.dumps(profiles.get(user_id, {"error": "user not found"}))

# ── List resource template ─────────────────────────────────────────────────
@mcp.resource("reports://daily/{date}")
def get_daily_report(date: str) -> str:
    """Return a daily summary report for YYYY-MM-DD format date."""
    return json.dumps({
        "date": date,
        "transactions": 14823,
        "volume_usd": 2_341_000,
        "fraud_blocked": 7,
    })

import json
print("Resources registered on server:")
print("  Static:   config://app/settings")
print("  Template: users://{user_id}/profile")
print("  Template: reports://daily/{date}")
```
```python
# CLIENT SIDE Step 1+2: resources/list → resources/read
# Uses the MCP Python SDK ClientSession

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def demonstrate_resource_access():
    server_params = StdioServerParameters(
        command="python",
        args=["my_resource_server.py"]  # ← the file containing mcp.run()
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── Step 1: Discover all available resources ──────────────────
            resources_result = await session.list_resources()
            print("Available resources:")
            for r in resources_result.resources:
                print(f"  URI:         {r.uri}")
                print(f"  Name:        {r.name}")
                print(f"  Description: {r.description}")
                print(f"  MIME type:   {r.mimeType}")
                print()

            # ── Step 2a: Read a static resource by its exact URI ──────────
            config_result = await session.read_resource("config://app/settings")
            print("Static resource content:")
            for content in config_result.contents:
                print(content.text)

            # ── Step 2b: Read a template resource — substitute {user_id} ──
            # The client fills in the template variable in the URI itself
            profile_result = await session.read_resource("users://alice/profile")
            print("\nTemplate resource content (user=alice):")
            for content in profile_result.contents:
                print(content.text)

            # ── Step 2c: Another template substitution ────────────────────
            report = await session.read_resource("reports://daily/2026-02-23")
            print("\nDaily report (2026-02-23):")
            for content in report.contents:
                print(content.text)


# asyncio.run(demonstrate_resource_access())

print("Resource client pattern:")
print("""
  session.list_resources()            → lists all URIs the server exposes
  session.read_resource(uri)          → fetches content at that URI
  session.list_resource_templates()   → lists parameterised URI templates

  Template rule:
    Server declares: users://{user_id}/profile
    Client calls:    session.read_resource("users://alice/profile")
    MCP SDK extracts user_id="alice" and calls get_user_profile("alice")
""")
```
```python
# CLIENT SIDE Step 3: list_resource_templates — discover parameterised URIs
# Useful when you want to know WHICH templates exist before constructing a URI

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def list_templates_example():
    server_params = StdioServerParameters(command="python", args=["my_resource_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover URI templates (parameterised resources)
            templates = await session.list_resource_templates()
            print("URI Templates exposed by server:")
            for t in templates.resourceTemplates:
                print(f"  Template:    {t.uriTemplate}")
                print(f"  Name:        {t.name}")
                print(f"  Description: {t.description}")
                print()

            # Subscribe to resource change notifications (optional)
            # Server will push notifications/resources/updated when content changes
            await session.subscribe_resource("config://app/settings")
            print("Subscribed to config resource — server will notify on changes")

# asyncio.run(list_templates_example())

# ── Full API surface for resources ────────────────────────────────────────────
resource_api = {
    "resources/list":              "session.list_resources()             → all static URIs",
    "resources/templates/list":    "session.list_resource_templates()    → all URI templates",
    "resources/read":              "session.read_resource(uri)           → fetch content",
    "resources/subscribe":         "session.subscribe_resource(uri)      → watch for changes",
    "resources/unsubscribe":       "session.unsubscribe_resource(uri)    → stop watching",
    "notifications/resources/"
    "updated (server→client)": "server pushes when subscribed resource changes",
}

print("\nComplete MCP Resource API:")
print(f"{'MCP method':<40} {'SDK call / description'}")
print("-" * 90)
for method, desc in resource_api.items():
    print(f"  {method:<38} {desc}")

# ── How resources differ from tools ─────────────────────────────────────────
print("""
Resources vs Tools:
  Tools      → Claude calls them to take actions or compute results (write OK)
  Resources  → Claude reads them for context/data (read-only, no side effects)

When to use Resources:
  ✓ Config files, documentation, reference data
  ✓ User profiles, account snapshots the AI needs as context
  ✓ Database records Claude should read before deciding what tool to call
  ✗ Don't use for real-time data that changes per-request → use Tools instead
""")
```
### 2.3 Prompts — Reusable prompt templates

Prompts let servers publish pre-built instruction templates that clients (and end users) can invoke by name.
```python
from mcp.server.fastmcp import FastMCP
from mcp.types import GetPromptResult, PromptMessage, TextContent

mcp = FastMCP(name="prompt-demo")

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Prompt template for reviewing code."""
    return f"""Review this {language} code for bugs, security issues, and style:

```{language}
{code}
```

Provide specific, actionable feedback."""

@mcp.prompt()
def summarize(text: str, max_words: int = 100) -> str:
    """Summarize text in a given number of words."""
    return f"Summarize the following text in under {max_words} words:\n\n{text}"

# Simulating what the generated prompt looks like
sample_prompt = code_review("python", "x = lambda: None; x()")
print(sample_prompt)
```
### 2.4 Sampling — Server asks the AI for completions

This is the *reverse* direction: the server can ask the connected LLM to generate text, enabling agentic loops that don't require the client to manage every step.

```
Normal flow:     Client ──request──▶ Server ──result──▶ Client

Sampling flow:   Server ──sampling request──▶ Client (LLM)
                                              Client completes
                 Server ◀──completion──────── Client
                 Server continues processing…
```
```python
# Conceptual illustration of a sampling request payload
sampling_request = {
    "method": "sampling/createMessage",
    "params": {
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Classify this email as spam or not spam: 'You won $1M!'"
                }
            }
        ],
        "modelPreferences": {
            "hints": [{"name": "claude-haiku-4-5"}],
            "speedPriority": 0.9,
            "costPriority": 0.8
        },
        "maxTokens": 50
    }
}

import json
print("Sampling request from SERVER → CLIENT:")
print(json.dumps(sampling_request, indent=2))
```
---
## 3. MCP Architecture

### 3.1 The Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        HOST PROCESS                         │
│  (Claude Desktop, VS Code, custom app, Jupyter, etc.)       │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    MCP CLIENT                          │ │
│  │  • Maintains 1:1 connection with each server           │ │
│  │  • Sends requests, receives results                    │ │
│  │  • Passes tool results back to the LLM                 │ │
│  └────────────┬────────────────────────────────┬──────────┘ │
│               │ transport                      │ transport  │
└───────────────┼────────────────────────────────┼────────────┘
                │                                │
    ┌───────────▼────────┐          ┌────────────▼───────────┐
    │    MCP SERVER A    │          │    MCP SERVER B        │
    │   (local process)  │          │  (remote HTTP server)  │
    │                    │          │                        │
    │  Tools:            │          │  Tools:                │
    │  • read_file       │          │  • query_database      │
    │  • write_file      │          │  • run_query           │
    │                    │          │                        │
    │  Resources:        │          │  Resources:            │
    │  • file://...      │          │  • db://tables/...     │
    └────────────────────┘          └────────────────────────┘
```

### 3.2 Session Lifecycle

```
CLIENT                          SERVER
  │                               │
  │──── initialize ─────────────▶│  Send protocol version + capabilities
  │◀─── initialized ─────────────│  Respond with server capabilities
  │                               │
  │──── notifications/initialized▶│  Client signals ready
  │                               │
  │  ←── OPERATION PHASE ──────── │
  │                               │
  │──── tools/list ─────────────▶│  Discover available tools
  │◀─── [tool schemas] ───────────│
  │                               │
  │──── resources/list ─────────▶│  Discover available resources
  │◀─── [resource URIs] ──────────│
  │                               │
  │──── tools/call ─────────────▶│  Invoke a tool
  │◀─── tool result ──────────────│
  │                               │
  │  ←── SHUTDOWN PHASE ───────── │
  │──── close ──────────────────▶│
```
### 3.3 Transports

Transport is how the client and server communicate. MCP supports two transports:

```
┌──────────────────────────────────────────────────────────────┐
│                    TRANSPORT OPTIONS                         │
│                                                              │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │     stdio (local)       │  │   Streamable HTTP        │  │
│  │                         │  │   (remote) ← PREFERRED   │  │
│  │  Client spawns server   │  │                          │  │
│  │  as a child process.    │  │  Server runs as an HTTP  │  │
│  │  JSON-RPC messages over │  │  service. Client sends   │  │
│  │  stdin/stdout pipes.    │  │  POST /mcp requests,     │  │
│  │                         │  │  server streams results  │  │
│  │  ✓ Simple setup         │  │  via SSE.                │  │
│  │  ✓ Strong isolation     │  │                          │  │
│  │  ✗ Same machine only    │  │  ✓ Runs anywhere         │  │
│  │  ✗ One client only      │  │  ✓ Multiple clients      │  │
│  └─────────────────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

#### stdio Transport

```
HOST PROCESS
┌──────────────────────────────────────┐
│  MCP Client                          │
│      │ spawns                        │
│      ▼                               │
│  [python research_server.py]         │
│      │                               │
│  stdin  ◀──── JSON-RPC requests ───  │
│  stdout ────▶ JSON-RPC responses ──▶ │
│  stderr ────▶ logs (ignored by MCP)  │
└──────────────────────────────────────┘
```

#### Streamable HTTP Transport

`Server-Sent Events (SSE)` transport has been `officially deprecated` in the Model Context Protocol (MCP) specification, starting with version 2025-03-26

```
CLIENT                          SERVER (any machine)
  │                               │
  │── POST /mcp (initialize) ───▶ │
  │◀── 200 OK + Mcp-Session-Id ───│
  │                               │
  │── POST /mcp (tools/call) ────▶│
  │                               │  long-running → stream
  │◀── SSE stream of events ──────│
  │   data: {partial result}      │
  │   data: {partial result}      │
  │   data: [DONE]                │
  │                               │
  │── DELETE /mcp (close) ───────▶│
```
```python
# Running with stdio transport (local process)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="stdio-demo")

@mcp.tool()
def echo(message: str) -> str:
    """Echo back the message."""
    return f"Echo: {message}"

if __name__ == "__main__":
    # Transport selected at startup — swap 'stdio' for 'streamable-http'
    mcp.run(transport="stdio")   # ← local, spawned by client
    # mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)  # ← remote
```
```python
# Running with Streamable HTTP transport (remote)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="http-demo")

@mcp.tool()
def current_time() -> str:
    """Return the current UTC time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

# if __name__ == "__main__":
#     mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
#     # Client connects via: http://localhost:8000/mcp

print("Streamable HTTP server would listen on http://0.0.0.0:8000/mcp")
```
### 3.4 JSON-RPC Message Format

All MCP communication uses **JSON-RPC 2.0** — a simple request/response protocol over whatever transport is chosen.
```python
import json

# 1. Client → Server: initialize handshake
initialize_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {
            "sampling": {},          # client can handle sampling requests
            "roots": {"listChanged": True}
        },
        "clientInfo": {"name": "my-app", "version": "1.0.0"}
    }
}

# 2. Server → Client: capability declaration
initialize_response = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2025-03-26",
        "capabilities": {
            "tools":     {"listChanged": True},
            "resources": {"listChanged": True},
            "prompts":   {"listChanged": True},
            "logging":   {}
        },
        "serverInfo": {"name": "research-server", "version": "0.1.0"}
    }
}

# 3. Client → Server: call a tool
tool_call_request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "search_paper",
        "arguments": {"topic": "large language models", "max_result": 3}
    }
}

# 4. Server → Client: tool result
tool_call_response = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "content": [
            {
                "type": "text",
                "text": '["2301.00234v1", "2302.13971v1", "2303.08774v2"]'
            }
        ],
        "isError": False
    }
}

for label, msg in [("INIT REQUEST", initialize_request),
                   ("INIT RESPONSE", initialize_response),
                   ("TOOL CALL", tool_call_request),
                   ("TOOL RESULT", tool_call_response)]:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print('='*50)
    print(json.dumps(msg, indent=2))
```
### 3.5 End-to-End Flow: Claude + MCP Server

```
USER                 CLAUDE (HOST)               MCP SERVER
 │                       │                           │
 │──"Search for LLM  ───▶│                           │
 │   papers"             │                           │
 │                       │── tools/list ────────────▶│
 │                       │◀─ [search_paper,           │
 │                       │    extract_info]            │
 │                       │                           │
 │                       │  Claude decides to call    │
 │                       │  search_paper              │
 │                       │                           │
 │                       │── tools/call ────────────▶│
 │                       │   search_paper(           │
 │                       │     topic="LLMs",         │
 │                       │     max_result=5)          │
 │                       │                           │  hits arxiv API
 │                       │                           │  saves JSON
 │                       │◀─ ["2301.00234", ...] ────│
 │                       │                           │
 │                       │  Claude may call          │
 │                       │  extract_info next        │
 │                       │── tools/call ────────────▶│
 │                       │   extract_info(           │
 │                       │     paper_id="2301.00234")│
 │                       │◀─ {title, authors, ...} ──│
 │                       │                           │
 │◀── "Here are 5 ───────│                           │
 │    papers on LLMs…"   │                           │
```
```python
# Full end-to-end: Claude + MCP client calling a local server
# This uses the mcp Python SDK's ClientSession

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic

async def run_mcp_agent(user_query: str, server_script: str):
    """Connect to an MCP server and run a Claude agent loop."""
    
    anthropic_client = anthropic.Anthropic()

    server_params = StdioServerParameters(
        command="python",
        args=[server_script],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # 1. Handshake
            await session.initialize()
            print("✓ Connected to MCP server")

            # 2. Discover tools
            tools_result = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema
                }
                for t in tools_result.tools
            ]
            print(f"✓ Discovered {len(tools)} tools: {[t['name'] for t in tools]}")

            # 3. Agentic loop
            messages = [{"role": "user", "content": user_query}]

            while True:
                response = anthropic_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    tools=tools,
                    messages=messages
                )

                # Collect assistant message
                messages.append({"role": "assistant", "content": response.content})

                # Check stop condition
                if response.stop_reason == "end_turn":
                    break

                # Process tool calls
                if response.stop_reason == "tool_use":
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"  → calling {block.name}({block.input})")
                            result = await session.call_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result.content[0].text if result.content else ""
                            })
                    messages.append({"role": "user", "content": tool_results})

            # 4. Extract final text
            final = next(
                (b.text for b in response.content if hasattr(b, "text")),
                "No response."
            )
            return final

# To run:
# result = asyncio.run(run_mcp_agent(
#     user_query="Find 3 papers about transformer attention mechanisms",
#     server_script="code/research_server.py"
# ))
# print(result)

print("Agent loop defined. Call run_mcp_agent() to execute.")
```
### 3.6 MCP Configuration (claude_desktop_config.json)

MCP servers are registered in a config file so the host knows which servers to launch:
```python
import json

# ~/.config/claude/claude_desktop_config.json
claude_desktop_config = {
    "mcpServers": {
        # stdio server — launched as a local subprocess
        "research": {
            "command": "python",
            "args": ["/path/to/research_server.py"],
            "env": {
                "PAPER_DIR": "/tmp/papers"
            }
        },
        # Another server using uv (common for Python MCP servers)
        "filesystem": {
            "command": "uvx",
            "args": ["mcp-server-filesystem", "/home/user/docs"],
        },
        # Remote HTTP server
        "weather-api": {
            "url": "https://weather-mcp.example.com/mcp",
            "headers": {
                "Authorization": "Bearer ${WEATHER_API_KEY}"
            }
        }
    }
}

print(json.dumps(claude_desktop_config, indent=2))
```
### 3.7 Low-level vs FastMCP

FastMCP wraps the low-level MCP SDK. Here's what FastMCP hides:
```python
# LOW-LEVEL MCP Server (verbose, full control)
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio

server = Server(name="low-level-demo")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Called when client does tools/list."""
    return [
        types.Tool(
            name="add",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict
) -> list[types.TextContent]:
    """Called when client does tools/call."""
    if name == "add":
        result = arguments["a"] + arguments["b"]
        return [types.TextContent(type="text", text=str(result))]
    raise ValueError(f"Unknown tool: {name}")

# async def main():
#     async with stdio_server() as (read, write):
#         await server.run(read, write, server.create_initialization_options())

print("Low-level server defined.")
print()
print("Equivalent FastMCP code:")
print("""
  mcp = FastMCP("demo")

  @mcp.tool()
  def add(a: float, b: float) -> float:
      \"\"\"Add two numbers.\"\"\"
      return a + b

  mcp.run(transport='stdio')
""")
```
### 3.8 Complete Reference: MCP Capabilities Matrix

```
┌───────────────┬────────────────┬────────────────────────────────────────┐
│   Primitive   │   Direction    │   Purpose                              │
├───────────────┼────────────────┼────────────────────────────────────────┤
│ Tools         │ Client→Server  │ Actions & computations the AI invokes  │
│ Resources     │ Client→Server  │ Read-only data the AI consumes         │
│ Prompts       │ Client→Server  │ Reusable prompt templates              │
│ Sampling      │ Server→Client  │ Server asks LLM to generate text       │
│ Roots         │ Client→Server  │ File system scope boundaries           │
│ Logging       │ Server→Client  │ Structured log messages                │
│ Progress      │ Server→Client  │ Long-running task progress updates     │
│ Cancellation  │ Client→Server  │ Cancel an in-progress request          │
└───────────────┴────────────────┴────────────────────────────────────────┘

┌───────────────┬────────────────────────────────────────────────────────┐
│   Transport   │   Best For                                             │
├───────────────┼────────────────────────────────────────────────────────┤
│ stdio         │ Local tools, dev/testing, CLI-based servers            │
│ Streamable    │ Production, multi-client, remote/cloud deployments     │
│ HTTP          │                                                        │
└───────────────┴────────────────────────────────────────────────────────┘
```
```python
# Summary: MCP in 5 lines

summary = """
MCP at a glance
═══════════════

1. WHY:    Eliminates M×N integration hell between AI models and tools.

2. WHAT:   Open standard (JSON-RPC 2.0) for AI ↔ tool communication.
           Primitives: Tools | Resources | Prompts | Sampling

3. HOW:    Client (AI host) discovers and calls server capabilities.
           Transport: stdio (local) or Streamable HTTP (remote).

4. BUILD:  FastMCP: @mcp.tool() → auto-schema from type hints + docstrings.
           Low-level SDK: full control over list_tools / call_tool handlers.

5. RUN:    mcp.run(transport='stdio')  ← local
           mcp.run(transport='streamable-http', port=8000)  ← remote
"""

print(summary)
```
---
## 4. Building MCP Servers

### Patterns at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│               MCP SERVER BUILDING BLOCKS                         │
│                                                                  │
│  FastMCP (recommended)          Low-level SDK                    │
│  ────────────────────           ──────────────                   │
│  @mcp.tool()        ←──────→   @server.call_tool()              │
│  @mcp.resource()    ←──────→   @server.read_resource()          │
│  @mcp.prompt()      ←──────→   @server.get_prompt()             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  SERVER INTERNALS                                          │  │
│  │                                                            │  │
│  │  Context object  ──▶  logging, progress, sampling         │  │
│  │  Lifespan hook   ──▶  DB pool, HTTP session, cache init   │  │
│  │  Dependencies    ──▶  injected via mcp.get_context()      │  │
│  │  Error handling  ──▶  McpError with structured payload    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```
```python
# 4.1 Lifespan Management — initialise shared resources once, clean up on shutdown
# The lifespan context is available to all tools via mcp.get_context()

from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from typing import AsyncIterator
import httpx
import asyncio

# Shared state container
class AppState:
    def __init__(self, http_client: httpx.AsyncClient, cache: dict):
        self.http_client = http_client
        self.cache: dict = cache

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    """Runs once: before first client connects → until server shuts down."""
    print("🟢 Starting up: creating HTTP client + cache")
    async with httpx.AsyncClient(timeout=10.0) as client:
        state = AppState(http_client=client, cache={})
        yield state              # tools receive this via ctx.request_context.lifespan_context
    print("🔴 Shutting down: resources released")

mcp = FastMCP(name="stateful-server", lifespan=app_lifespan)

@mcp.tool()
async def fetch_url(url: str, ctx=None) -> str:
    """Fetch a URL using the shared HTTP client (connection-pooled)."""
    # Access the shared lifespan state
    state: AppState = ctx.request_context.lifespan_context
    
    # Check cache first
    if url in state.cache:
        return f"[CACHED] {state.cache[url][:200]}"
    
    resp = await state.http_client.get(url)
    body = resp.text[:500]
    state.cache[url] = body
    return body

print("Stateful server with lifespan defined.")
print("Pattern: lifespan creates resources → tools use them via context")
```
```python
# 4.2 Structured Error Handling
# MCP errors are returned as isError=True with a text content block

from mcp.server.fastmcp import FastMCP
from mcp import McpError
from mcp.types import ErrorData
import re

mcp = FastMCP(name="error-handling-demo")

class ValidationError(Exception):
    pass

def validate_transaction_id(txn_id: str) -> None:
    """Validate -style transaction ID format."""
    pattern = r'^[A-Z0-9]{17}$'
    if not re.match(pattern, txn_id):
        raise ValidationError(
            f"Invalid transaction ID '{txn_id}'. Expected 17 uppercase alphanumeric chars."
        )

@mcp.tool()
def get_transaction(transaction_id: str) -> str:
    """
    Look up a transaction by ID.
    
    Args:
        transaction_id: 17-character alphanumeric transaction ID
    """
    try:
        validate_transaction_id(transaction_id)
    except ValidationError as e:
        # Return structured error — Claude will handle/retry gracefully
        raise McpError(
            ErrorData(code=-32602, message=str(e))
        )
    
    # Simulate DB lookup
    mock_db = {"5TY68390XB789012A": {"amount": 99.99, "currency": "USD", "status": "COMPLETED"}}
    result = mock_db.get(transaction_id)
    
    if result is None:
        raise McpError(
            ErrorData(code=-32001, message=f"Transaction {transaction_id} not found")
        )
    
    return str(result)

# Test validation
try:
    validate_transaction_id("INVALID")
except ValidationError as e:
    print(f"Caught: {e}")

try:
    validate_transaction_id("5TY68390XB789012A")
    print("Valid transaction ID ✓")
except ValidationError as e:
    print(f"Error: {e}")

print("\nError flow:")
print("  Tool raises McpError → MCP SDK catches it → sends isError=True response")
print("  Claude sees error text → decides to retry / ask user / give up")
```
```python
# 4.3 Progress Notifications for Long-Running Tools
# Server pushes progress events to client during execution

from mcp.server.fastmcp import FastMCP, Context
import asyncio

mcp = FastMCP(name="progress-demo")

@mcp.tool()
async def bulk_process(items: list[str], ctx: Context) -> str:
    """
    Process a large batch of items with progress reporting.
    Client sees real-time progress while tool runs.
    """
    total = len(items)
    results = []

    for i, item in enumerate(items):
        # Report progress — client receives notifications/progress events
        await ctx.report_progress(
            progress=i,
            total=total,
            # message=f"Processing {item}"  # optional in some SDK versions
        )

        # Simulate work
        await asyncio.sleep(0.1)
        results.append(f"processed:{item}")

    await ctx.report_progress(progress=total, total=total)
    return f"Done. Processed {total} items: {results}"

# Progress flow:
print("Progress notification flow:")
print("""
  Tool starts
      ↓
  ctx.report_progress(0, 10)  ──▶  notifications/progress {progress:0,  total:10}
  ctx.report_progress(5, 10)  ──▶  notifications/progress {progress:5,  total:10}
  ctx.report_progress(10, 10) ──▶  notifications/progress {progress:10, total:10}
      ↓
  Tool returns result
""")
```
---
## 5. Production Engineering

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PRODUCTION MCP SERVER                             │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │
│  │   LATENCY   │  │     COST     │  │   SECURITY   │  │  OBS.  │  │
│  │─────────────│  │──────────────│  │──────────────│  │────────│  │
│  │ • Cache     │  │ • Result     │  │ • Input      │  │• Trace │  │
│  │   results   │  │   compress   │  │   validate   │  │• Metric│  │
│  │ • Conn pool │  │ • Token      │  │ • Rate limit │  │• Log   │  │
│  │ • Parallel  │  │   budget     │  │ • OAuth2/JWT │  │• Alert │  │
│  │   tool exec │  │ • Batching   │  │ • Sandboxing │  │        │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  └────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │   RESILIENCE: retries + circuit breaker + health check       │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```
```python
# 5.1 Latency: TTL Cache decorator for expensive tool calls
import functools
import hashlib
import json
import time
from typing import Any, Callable

def ttl_cache(ttl_seconds: int = 300):
    """Decorator: cache tool results with TTL. Reduces repeated API calls."""
    cache: dict[str, tuple[Any, float]] = {}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name + args
            key_data = {"fn": func.__name__, "args": args, "kwargs": kwargs}
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()

            now = time.time()
            if cache_key in cache:
                result, ts = cache[cache_key]
                if now - ts < ttl_seconds:
                    print(f"  [CACHE HIT] {func.__name__} → returning cached result")
                    return result

            result = func(*args, **kwargs)
            cache[cache_key] = (result, now)
            print(f"  [CACHE MISS] {func.__name__} → computed + cached for {ttl_seconds}s")
            return result
        return wrapper
    return decorator


# Apply to any tool function
@ttl_cache(ttl_seconds=60)
def fetch_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Fetch live exchange rate (cached 60s to avoid hammering forex API)."""
    # In production: call forex API
    rates = {"USD_EUR": 0.92, "USD_GBP": 0.79, "USD_JPY": 149.5}
    return rates.get(f"{from_currency}_{to_currency}", 1.0)


# Demo
print("First call:")
rate = fetch_exchange_rate("USD", "EUR")
print(f"  Rate: {rate}")

print("\nSecond call (same args, within TTL):")
rate = fetch_exchange_rate("USD", "EUR")
print(f"  Rate: {rate}")
```
```python
# 5.2 Resilience: Retry with exponential backoff + Circuit Breaker

import time
import random
from enum import Enum

# ── Retry with exponential backoff ──────────────────────────────────────────
def retry(max_attempts: int = 3, base_delay: float = 0.5, backoff: float = 2.0):
    """Decorator: retry on failure with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    jitter = random.uniform(0, delay * 0.1)
                    print(f"  Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay + jitter)
                    delay *= backoff
        return wrapper
    return decorator


# ── Circuit Breaker ──────────────────────────────────────────────────────────
class CircuitState(Enum):
    CLOSED = "CLOSED"        # normal operation
    OPEN = "OPEN"            # failing, reject all calls
    HALF_OPEN = "HALF_OPEN"  # testing if service recovered

class CircuitBreaker:
    """
    Prevents cascading failures by stopping calls to a failing downstream.
    
    CLOSED ──(N failures)──▶ OPEN ──(timeout)──▶ HALF_OPEN
      ▲                                                │
      └──────────────(success)────────────────────────┘
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: float = 0

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                print(f"  Circuit → HALF_OPEN (testing recovery)")
            else:
                raise Exception(f"Circuit OPEN — downstream unavailable. Retry in {self.recovery_timeout - elapsed:.0f}s")

        try:
            result = func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print("  Circuit → CLOSED (recovered)")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"  Circuit → OPEN after {self.failure_count} failures")
            raise


# Demo
cb = CircuitBreaker(failure_threshold=3)

def unreliable_api(x):
    if x < 3:
        raise ConnectionError("Service unavailable")
    return f"OK({x})"

for i in range(5):
    try:
        result = cb.call(unreliable_api, i)
        print(f"  Call {i}: {result}")
    except Exception as e:
        print(f"  Call {i}: ERROR — {e}")
```
```python
# 5.3 Security: Input validation + Rate limiting + API key auth

import re
import time
from collections import defaultdict
from mcp.server.fastmcp import FastMCP

# ── Input Sanitisation ────────────────────────────────────────────────────────
ALLOWED_SQL_PATTERN = re.compile(r'^[a-zA-Z0-9_\s,=<>!\'\"%.()-]+$')

def sanitize_sql_param(value: str, field_name: str) -> str:
    """Block SQL injection in tool parameters."""
    if not ALLOWED_SQL_PATTERN.match(value):
        raise ValueError(f"Invalid characters in '{field_name}'. Possible injection attempt.")
    if len(value) > 500:
        raise ValueError(f"'{field_name}' exceeds max length of 500.")
    return value.strip()

# ── Token Bucket Rate Limiter ─────────────────────────────────────────────────
class RateLimiter:
    """
    Token bucket: each client gets `capacity` tokens, refills at `rate` tokens/sec.
    
    Bucket:  [●●●●●●●●●●]  capacity=10
    Request: consume 1 token
    Refill:  +rate tokens per second up to capacity
    """
    def __init__(self, rate: float = 5.0, capacity: float = 10.0):
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, tuple[float, float]] = defaultdict(
            lambda: (capacity, time.time())
        )

    def allow(self, client_id: str) -> tuple[bool, float]:
        tokens, last_refill = self._buckets[client_id]
        now = time.time()
        # Refill
        tokens = min(self.capacity, tokens + (now - last_refill) * self.rate)
        if tokens >= 1.0:
            self._buckets[client_id] = (tokens - 1.0, now)
            return True, tokens - 1.0
        else:
            self._buckets[client_id] = (tokens, now)
            return False, tokens

limiter = RateLimiter(rate=2.0, capacity=5.0)

# Simulate requests
for i in range(8):
    allowed, remaining = limiter.allow("client-alice")
    status = "✓ ALLOWED" if allowed else "✗ RATE LIMITED"
    print(f"  Request {i+1}: {status} (tokens remaining: {remaining:.2f})")
    time.sleep(0.1)

# ── Security: Allowlist-based tool permissions ────────────────────────────────
print("\nTool permission matrix:")
permissions = {
    "read_transaction":   ["analyst", "admin", "support"],
    "refund_transaction": ["admin"],
    "delete_account":     ["admin"],
    "get_balance":        ["analyst", "admin", "support", "customer"],
}

for tool, roles in permissions.items():
    print(f"  {tool:<25} allowed: {roles}")
```
```python
# 5.4 Observability: Structured logging + OpenTelemetry tracing

import logging
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

# ── Structured JSON Logger ────────────────────────────────────────────────────
class StructuredLogger:
    """Emits JSON log lines — easy to ingest into Splunk / Datadog / CloudWatch."""
    def __init__(self, service: str):
        self.service = service

    def log(self, level: str, event: str, **fields):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "service": self.service,
            "event": event,
            **fields,
        }
        print(json.dumps(record))

    def tool_call(self, tool: str, args: dict, result: Any, duration_ms: float, error: str = None):
        self.log(
            level="ERROR" if error else "INFO",
            event="tool_call",
            tool=tool,
            args=args,
            result_preview=str(result)[:100] if result else None,
            duration_ms=round(duration_ms, 2),
            error=error,
            trace_id=str(uuid.uuid4())[:8],
        )


logger = StructuredLogger(service="-mcp-server")

# ── Timing decorator using structured logger ──────────────────────────────────
def traced_tool(func):
    """Decorator: log every tool call with timing and args."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        error = None
        result = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000
            logger.tool_call(
                tool=func.__name__,
                args=kwargs or (args[0] if args else {}),
                result=result,
                duration_ms=duration_ms,
                error=error,
            )
    return wrapper


# Demo
@traced_tool
def lookup_user(user_id: str, include_balance: bool = False) -> dict:
    time.sleep(0.01)  # simulate DB query
    return {"id": user_id, "name": "Alice", "balance": 500.00 if include_balance else None}

lookup_user(user_id="u_12345", include_balance=True)
lookup_user(user_id="u_99999")
```
```python
# 5.5 Cost Control: Token budget enforcement + result truncation

import tiktoken  # pip install tiktoken

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in a string using tiktoken."""
    try:
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback: approximate 4 chars per token
        return len(text) // 4

def truncate_to_token_budget(text: str, max_tokens: int = 1000) -> str:
    """
    Truncate tool output to stay within token budget.
    Prevents huge tool responses from blowing up context window.
    """
    token_count = count_tokens(text)
    if token_count <= max_tokens:
        return text
    
    # Binary search for right truncation point
    chars = len(text)
    target_chars = int(chars * (max_tokens / token_count))
    truncated = text[:target_chars]
    return truncated + f"\n\n[... TRUNCATED: {token_count - max_tokens} tokens removed ...]"


# Token budget policy per tool type
TOKEN_BUDGETS = {
    "search_results":     2000,   # search tools return max 2000 tokens
    "document_content":   4000,   # document reading tools
    "database_query":     1500,   # DB results
    "api_response":        800,   # external API calls
}

# Demo
sample_output = "transaction data " * 500  # 9000+ chars
print(f"Original: {count_tokens(sample_output)} tokens")

truncated = truncate_to_token_budget(sample_output, max_tokens=50)
print(f"Truncated: {count_tokens(truncated)} tokens")
print(f"Preview: {truncated[:100]}...")
```
---
## 6. Advanced MCP Topics

### 6.1 Multi-Server Orchestration

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI HOST (Claude)                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   MCP CLIENT MANAGER                    │   │
│  │                                                         │   │
│  │   sessions = {                                          │   │
│  │     "payments":   ClientSession(payments_server),       │   │
│  │     "fraud":      ClientSession(fraud_server),          │   │
│  │     "compliance": ClientSession(compliance_server),     │   │
│  │   }                                                     │   │
│  │                                                         │   │
│  │   tools = merge(payments.tools + fraud.tools + ...)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │               │                    │               │
│           ▼               ▼                    ▼               │
│   ┌──────────────┐ ┌─────────────┐ ┌──────────────────────┐   │
│   │ Payments MCP │ │  Fraud MCP  │ │   Compliance MCP     │   │
│   │ Server       │ │  Server     │ │   Server             │   │
│   └──────────────┘ └─────────────┘ └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```
```python
# 6.1 Multi-Server Orchestration — aggregating tools from N servers

import asyncio
import json
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic

@dataclass
class ServerConfig:
    name: str
    command: str
    args: list[str]
    tool_prefix: str = ""  # e.g. "payments_" to namespace tool names

class MultiServerMCPClient:
    """
    Manages connections to multiple MCP servers and presents
    a unified tool catalog to Claude.
    """
    def __init__(self, servers: list[ServerConfig]):
        self.servers = servers
        self._sessions: dict[str, ClientSession] = {}
        self._tool_map: dict[str, tuple[str, dict]] = {}  # tool_name → (server_name, schema)

    async def connect_all(self):
        """Initialize all server connections in parallel."""
        await asyncio.gather(*[self._connect(s) for s in self.servers])
        print(f"Connected to {len(self._sessions)} servers")
        print(f"Total tools available: {len(self._tool_map)}")

    async def _connect(self, config: ServerConfig):
        """Connect to a single server and register its tools."""
        params = StdioServerParameters(command=config.command, args=config.args)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                for tool in tools.tools:
                    # Prefix tool names to avoid collisions
                    namespaced = f"{config.tool_prefix}{tool.name}" if config.tool_prefix else tool.name
                    self._tool_map[namespaced] = (config.name, {
                        "name": namespaced,
                        "description": f"[{config.name}] {tool.description}",
                        "input_schema": tool.inputSchema,
                    })
                self._sessions[config.name] = session
                print(f"  ✓ {config.name}: {len(tools.tools)} tools")

    def get_all_tools(self) -> list[dict]:
        """Return unified tool list for Claude's tools= parameter."""
        return [schema for _, schema in self._tool_map.values()]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Route tool call to the correct server."""
        if tool_name not in self._tool_map:
            return f"Error: tool '{tool_name}' not found"
        server_name, _ = self._tool_map[tool_name]
        session = self._sessions[server_name]
        # Strip prefix to get original tool name
        original_name = tool_name.replace(f"{server_name}_", "")
        result = await session.call_tool(original_name, arguments)
        return result.content[0].text if result.content else ""


# Configuration for 's multi-server setup
server_configs = [
    ServerConfig(name="payments",   command="python", args=["servers/payments_server.py"],   tool_prefix="pay_"),
    ServerConfig(name="fraud",      command="python", args=["servers/fraud_server.py"],       tool_prefix="fraud_"),
    ServerConfig(name="compliance", command="python", args=["servers/compliance_server.py"],  tool_prefix="kyc_"),
]

print("Multi-server config:")
for cfg in server_configs:
    print(f"  {cfg.name}: prefix='{cfg.tool_prefix}', cmd={cfg.command} {cfg.args}")

print("\nUsage:")
print("""
  async with MultiServerMCPClient(server_configs) as client:
      await client.connect_all()
      tools = client.get_all_tools()
      # Pass tools to Claude → Claude picks pay_refund, fraud_score, etc.
""")
```
```python
# 6.2 Schema Versioning & Evolution
# How to version MCP server APIs without breaking existing clients

import json
from enum import Enum

class SchemaVersion(str, Enum):
    V1 = "1.0"
    V2 = "2.0"

# ── Version-aware tool that maintains backward compatibility ──────────────────
def get_tool_schema(version: SchemaVersion) -> dict:
    """Return the appropriate schema for the requested API version."""
    
    base_schema = {
        "name": "search_transactions",
        "description": "Search transactions by various criteria",
    }

    if version == SchemaVersion.V1:
        # V1: simple date range filter
        base_schema["inputSchema"] = {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                "end_date":   {"type": "string", "description": "ISO date YYYY-MM-DD"},
            },
            "required": ["start_date", "end_date"]
        }
    elif version == SchemaVersion.V2:
        # V2: added merchant filter, cursor-based pagination, amount range
        # V1 params still supported for backward compat
        base_schema["description"] += " (v2: adds merchant filter, pagination, amount range)"
        base_schema["inputSchema"] = {
            "type": "object",
            "properties": {
                "start_date":  {"type": "string"},
                "end_date":    {"type": "string"},
                # NEW in V2:
                "merchant_id": {"type": "string", "description": "Filter by merchant"},
                "min_amount":  {"type": "number", "description": "Minimum transaction amount"},
                "max_amount":  {"type": "number", "description": "Maximum transaction amount"},
                "cursor":      {"type": "string", "description": "Pagination cursor"},
                "page_size":   {"type": "integer", "default": 50},
            },
            "required": ["start_date", "end_date"]  # same required as V1
        }

    return base_schema


# ── Version negotiation during initialize handshake ───────────────────────────
def negotiate_version(client_versions: list[str], server_versions: list[str]) -> str | None:
    """Pick highest mutually-supported version."""
    client_set = set(client_versions)
    for v in sorted(server_versions, reverse=True):
        if v in client_set:
            return v
    return None


# Demo
v1_schema = get_tool_schema(SchemaVersion.V1)
v2_schema = get_tool_schema(SchemaVersion.V2)

print("V1 schema required fields:", v1_schema["inputSchema"]["required"])
print("V1 schema properties:", list(v1_schema["inputSchema"]["properties"].keys()))
print()
print("V2 schema required fields:", v2_schema["inputSchema"]["required"])
print("V2 schema properties:", list(v2_schema["inputSchema"]["properties"].keys()))
print()

negotiated = negotiate_version(
    client_versions=["1.0", "2.0"],
    server_versions=["2.0", "3.0"]
)
print(f"Negotiated version: {negotiated}")

print("""
Evolution rules:
  ✓ Add new optional fields (backward compatible)
  ✓ Add new tools (existing tools untouched)
  ✗ Remove required fields (breaking change → increment major version)
  ✗ Rename fields (breaking change)
  ✓ Deprecate via description: "DEPRECATED: use X instead"
""")
```
---
## 7. Integration with Agent Frameworks

### LangChain + LangGraph + MCP

```
┌──────────────────────────────────────────────────────────────────┐
│                    LANGCHAIN MCP INTEGRATION                     │
│                                                                  │
│  langchain-mcp-adapters (pip install langchain-mcp-adapters)     │
│                                                                  │
│  MCP Server ──▶ MCPToolkit ──▶ LangChain BaseTool ──▶ Agent     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   LANGGRAPH + MCP                           │ │
│  │                                                             │ │
│  │   StateGraph                                                │ │
│  │     ├── agent_node   (Claude decides which tool to call)    │ │
│  │     └── tools_node   (executes MCP tools via toolkit)      │ │
│  │                                                             │ │
│  │   Routing: agent → tools → agent → tools → … → END         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```
```python
# 7.1 LangChain + MCP via langchain-mcp-adapters
# pip install langchain-mcp-adapters langchain-anthropic langgraph

import asyncio
import sys
import os
from dotenv import load_dotenv
from mcp import StdioServerParameters

load_dotenv("/Users/vinotganesan/Learning/LLM & AGENTS/RAG/.env")

api_key = os.environ['UNIFIED_LLM_KEY']
base_url = ""

# langchain_mcp_adapters converts MCP tools → LangChain BaseTool
# so they work seamlessly with any LangChain agent or chain

async def langchain_mcp_example():
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,   # uses the active conda env Python — avoids ModuleNotFoundError
        args=["code/create_mcp_server_client/research_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # load_mcp_tools converts MCP tools → LangChain BaseTool objects
            tools = await load_mcp_tools(session)
            print(f"Loaded {len(tools)} LangChain tools from MCP:")
            for t in tools:
                print(f"  • {t.name}: {t.description[:60]}")

            llm = ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=api_key,
                base_url=base_url
            )

            # create_react_agent builds a ReAct loop: think → act → observe
            agent = create_react_agent(llm, tools)

            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Find 3 papers on RAG systems"}]
            })
            return result["messages"][-1].content

# asyncio.run(langchain_mcp_example())
print("LangChain + MCP pattern:")
print("""
  1. sys.executable ensures the subprocess uses the same conda env (avoids import errors)
  2. load_mcp_tools(session) → list of BaseTool
  3. ChatAnthropic(api_key=..., base_url=...) → connects to internal LLM proxy
  4. create_react_agent(llm, tools) → ReAct loop handles tool calls automatically
""")
```
```python
# 7.2 LangGraph StateGraph + MCP Tools (custom graph with persistence)
# pip install langgraph langchain-anthropic langchain-mcp-adapters

from typing import Annotated, TypedDict, Literal
import operator

graph_code = '''
import sys
import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv("/Users/vinotganesan/Learning/LLM & AGENTS/RAG/.env")
api_key = os.environ["UNIFIED_LLM_KEY"]
base_url = ""

# ── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_count: int

# ── Nodes ────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    api_key=api_key,
    base_url=base_url
)

def agent_node(state: AgentState) -> AgentState:
    """Call Claude — decides whether to use a tool or respond."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "tool_call_count": state["tool_call_count"]}

def tools_node(state: AgentState) -> AgentState:
    """Execute all tool calls requested by Claude."""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tool_map[tool_call["name"]]
        output = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    return {
        "messages": results,
        "tool_call_count": state["tool_call_count"] + len(results)
    }

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route: if last message has tool calls → tools node, else → END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

# ── Graph ─────────────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")  # after tools → back to agent
app = graph.compile()
'''

print("LangGraph + MCP StateGraph structure:")
print("""
  START
    │
    ▼
  [agent]  ─── has tool calls? ──▶ [tools]
    ▲                                  │
    │◀──────────────────────────────────┘
    │
    └─── no tool calls ──▶ END

  Key: graph.compile() returns a Pregel app that can be:
    • streamed:   app.stream({"messages": [...]})
    • invoked:    app.invoke({"messages": [...]})
    • persisted:  app.compile(checkpointer=MemorySaver())
""")
```
```python
# 7.3 LangSmith Tracing for MCP Tool Calls
# LangSmith gives full visibility into every Claude call + tool result

import os
import json
from datetime import datetime, timezone

# LangSmith auto-traces when LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY set
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "-mcp-agent"
# os.environ["LANGCHAIN_API_KEY"] = "ls__..."

# ── Manual span wrapping for non-LangChain MCP calls ─────────────────────────
def create_langsmith_run(name: str, inputs: dict, run_type: str = "tool") -> dict:
    """Create a LangSmith run metadata dict (illustrative)."""
    return {
        "id": str(__import__("uuid").uuid4()),
        "name": name,
        "run_type": run_type,
        "inputs": inputs,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "project_name": os.getenv("LANGCHAIN_PROJECT", "default"),
        "tags": ["mcp", "production"],
        "metadata": {
            "server": "-mcp-server",
            "transport": "stdio",
        }
    }

def finish_langsmith_run(run: dict, outputs: dict, error: str = None) -> dict:
    run["end_time"] = datetime.now(timezone.utc).isoformat()
    run["outputs"] = outputs
    run["error"] = error
    run["status"] = "error" if error else "success"
    return run


# Demo — what a traced MCP tool call looks like in LangSmith
run = create_langsmith_run(
    name="search_paper",
    inputs={"topic": "transformer attention", "max_results": 5},
    run_type="tool"
)
run = finish_langsmith_run(
    run,
    outputs={"paper_ids": ["2301.00234v1", "2302.13971v1"]},
)
print("LangSmith trace record:")
print(json.dumps(run, indent=2))

print("""
LangSmith gives you:
  • Full trace: user query → Claude thinking → tool call → tool result → response
  • Latency breakdown per step
  • Token usage per LLM call
  • Dataset creation from traces for evaluation
  • A/B testing different prompts / models / tool descriptions
""")
```
```python
# 7.4 Streamable HTTP Transport (MCP Protocol 2025-03-26 / mcp SDK >= 1.2)
# Replaces SSE with a single bidirectional POST endpoint — simpler and cloud-friendly
#
# Server-side change (research_server.py):
#   mcp.run(transport='streamable-http', port=8007)   # exposes POST /mcp
#
# Inspector:
#   npx @modelcontextprotocol/inspector  → set Transport=HTTP, URL=http://localhost:8007/mcp

import asyncio
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv("/Users/vinotganesan/Learning/LLM & AGENTS/RAG/.env")

api_key = os.environ['UNIFIED_LLM_KEY']
base_url = ""

async def streamable_http_example():
    """Connect to a remote MCP server via Streamable HTTP transport."""
    # Requires: python research_server.py  (with transport='streamable-http')
    async with streamablehttp_client("http://localhost:8007/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.list_tools()
            tools = [{
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema
            } for t in response.tools]
            print(f"Connected via Streamable HTTP. Tools: {[t['name'] for t in tools]}")

            client = Anthropic(api_key=api_key, base_url=base_url)
            messages = [{"role": "user", "content": "Search for 2 papers on LLM alignment"}]

            llm_response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                tools=tools,
                messages=messages
            )

            # Collect all tool results before next API call
            tool_results = []
            for block in llm_response.content:
                if block.type == "tool_use":
                    result = await session.call_tool(block.name, arguments=block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.content
                    })
                    print(f"  Tool '{block.name}' called → {len(result.content)} results")

            return tool_results

# asyncio.run(streamable_http_example())

print("""
SSE vs Streamable HTTP
┌─────────────────────┬────────────────────────────┬────────────────────────────┐
│                     │ SSE (legacy)               │ Streamable HTTP (new)      │
├─────────────────────┼────────────────────────────┼────────────────────────────┤
│ Protocol version    │ 2024-11-05                 │ 2025-03-26                 │
│ Endpoints           │ GET /sse + POST /messages  │ POST /mcp  (single)        │
│ Direction           │ Server push                │ Bidirectional HTTP         │
│ Cloud / load-bal.   │ Needs sticky sessions      │ Stateless-compatible       │
│ Inspector transport │ transportType=sse          │ transportType=http         │
│ SDK import          │ mcp.client.sse             │ mcp.client.streamable_http │
│ Server arg          │ transport='sse'            │ transport='streamable-http'│
└─────────────────────┴────────────────────────────┴────────────────────────────┘

Key difference: SSE opens a long-lived GET stream + a separate POST endpoint.
Streamable HTTP uses a single POST /mcp for everything — easier to proxy and deploy.
""")
```
---
## 8. Testing & Evaluation

```
┌──────────────────────────────────────────────────────────────────┐
│                     MCP TESTING PYRAMID                          │
│                                                                  │
│                      ┌─────────────┐                            │
│                      │   E2E Tests │  (Claude + real MCP server) │
│                      │  (slowest)  │                            │
│                    ┌─┴─────────────┴─┐                          │
│                    │ Integration Tests│  (MCP client ↔ server)  │
│                  ┌─┴─────────────────┴─┐                        │
│                  │     Unit Tests       │  (tool functions)      │
│                  │     (fastest)        │                        │
│                  └─────────────────────┘                        │
│                                                                  │
│  pytest + pytest-asyncio  ←  standard test stack for MCP        │
└──────────────────────────────────────────────────────────────────┘
```
```python
# 8.1 Unit Tests — test tool functions directly (no MCP server needed)
# Save as tests/test_tools.py and run: pytest tests/ -v

import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock, patch


# ── Functions under test (from research_server.py) ────────────────────────────
def search_paper_impl(topic: str, max_results: int, paper_dir: str) -> list:
    """Pure function extracted from MCP tool for testability."""
    path = os.path.join(paper_dir, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "papers_info.json")
    
    try:
        with open(file_path, "r") as f:
            return list(json.load(f).keys())
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def extract_info_impl(paper_id: str, paper_dir: str) -> str:
    """Pure function for paper lookup."""
    for item in os.listdir(paper_dir):
        item_path = os.path.join(paper_dir, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if paper_id in data:
                            return json.dumps(data[paper_id], indent=2)
                except Exception:
                    continue
    return f"There's no saved information related to paper {paper_id}."


# ── Test Cases ────────────────────────────────────────────────────────────────
class TestSearchPaper:

    def test_returns_empty_list_for_unknown_topic(self, tmp_path):
        result = search_paper_impl("quantum_computing_xyz", 5, str(tmp_path))
        assert result == []

    def test_creates_directory_for_topic(self, tmp_path):
        search_paper_impl("machine learning", 5, str(tmp_path))
        expected_dir = tmp_path / "machine_learning"
        assert expected_dir.exists()

    def test_spaces_replaced_with_underscores(self, tmp_path):
        search_paper_impl("deep learning transformers", 3, str(tmp_path))
        assert (tmp_path / "deep_learning_transformers").exists()


class TestExtractInfo:

    def setup_method(self):
        """Set up fixture data."""
        self.paper_data = {
            "2301.00234v1": {
                "title": "Attention is All You Need",
                "authors": ["Vaswani et al."],
                "summary": "Transformer architecture paper",
                "pdf_url": "https://arxiv.org/pdf/2301.00234v1",
                "published": "2023-01-01"
            }
        }

    def test_finds_existing_paper(self, tmp_path):
        # Setup
        topic_dir = tmp_path / "transformers"
        topic_dir.mkdir()
        with open(topic_dir / "papers_info.json", "w") as f:
            json.dump(self.paper_data, f)

        result = extract_info_impl("2301.00234v1", str(tmp_path))
        parsed = json.loads(result)
        assert parsed["title"] == "Attention is All You Need"

    def test_returns_error_for_missing_paper(self, tmp_path):
        result = extract_info_impl("nonexistent_id", str(tmp_path))
        assert "no saved information" in result.lower()

    def test_searches_all_topic_directories(self, tmp_path):
        # Paper in a different topic directory
        other_dir = tmp_path / "nlp"
        other_dir.mkdir()
        with open(other_dir / "papers_info.json", "w") as f:
            json.dump(self.paper_data, f)

        result = extract_info_impl("2301.00234v1", str(tmp_path))
        assert "Attention is All You Need" in result


# Run the tests
import sys
import io

# Simulate pytest output
test_results = []
for cls in [TestSearchPaper, TestExtractInfo]:
    for method_name in [m for m in dir(cls) if m.startswith("test_")]:
        try:
            obj = cls()
            if hasattr(obj, "setup_method"):
                obj.setup_method()
            with tempfile.TemporaryDirectory() as tmp:
                class FakePath:
                    def __init__(self, p): self._p = p
                    def __truediv__(self, other): return FakePath(os.path.join(self._p, other))
                    def mkdir(self, **kw): os.makedirs(self._p, exist_ok=True)
                    def exists(self): return os.path.exists(self._p)
                    def __str__(self): return self._p
                getattr(obj, method_name)(FakePath(tmp))
                test_results.append(f"  PASS  {cls.__name__}::{method_name}")
        except Exception as e:
            test_results.append(f"  FAIL  {cls.__name__}::{method_name}: {e}")

print(f"{'='*60}")
print(f"  MCP Tool Unit Tests")
print(f"{'='*60}")
for r in test_results:
    print(r)
print(f"\n{len([r for r in test_results if 'PASS' in r])}/{len(test_results)} passed")
```
```python
# 8.2 Integration Test — in-memory MCP client-server roundtrip
# Tests the full MCP protocol without spawning a subprocess

import asyncio
import json
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp import types

# ── Build a test server ────────────────────────────────────────────────────────
test_mcp = FastMCP(name="test-server")

@test_mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@test_mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

# ── In-memory transport test harness ─────────────────────────────────────────
async def run_in_memory_tests():
    """
    Use mcp.server.memory_server for fast in-process testing.
    Falls back to direct tool function calls if SDK version doesn't support it.
    """
    from mcp.server.fastmcp import FastMCP as _FastMCP

    results = []

    # Test 1: Tool list discovery
    # (Simulate what happens during initialize)
    tools = test_mcp._tool_manager.list_tools()
    tool_names = [t.name for t in tools]
    assert "add" in tool_names, "add tool should be registered"
    assert "divide" in tool_names, "divide tool should be registered"
    results.append(f"  PASS  tools/list → {tool_names}")

    # Test 2: Successful tool invocation
    add_tool = next(t for t in tools if t.name == "add")
    add_result = await test_mcp._tool_manager.call_tool("add", {"a": 5, "b": 3})
    assert add_result.content[0].text == "8", f"Expected 8, got {add_result.content[0].text}"
    results.append(f"  PASS  add(5, 3) → {add_result.content[0].text}")

    # Test 3: Tool with error
    try:
        div_result = await test_mcp._tool_manager.call_tool("divide", {"a": 10, "b": 0})
        # Should have isError=True
        if div_result.isError:
            results.append("  PASS  divide(10, 0) → isError=True")
        else:
            results.append("  FAIL  divide(10, 0) should return error")
    except Exception as e:
        results.append(f"  PASS  divide(10, 0) → raised {type(e).__name__}")

    # Test 4: Schema validation
    add_schema = next(t for t in tools if t.name == "add")
    assert "a" in add_schema.inputSchema["properties"]
    assert "b" in add_schema.inputSchema["properties"]
    results.append("  PASS  add schema has correct properties")

    return results

# Run integration tests
results = asyncio.run(run_in_memory_tests())
print("="*60)
print("  MCP Integration Tests (in-memory)")
print("="*60)
for r in results:
    print(r)
print(f"\n{len([r for r in results if 'PASS' in r])}/{len(results)} passed")
```
```python
# 8.3 LLM-as-Judge Evaluation for Tool Output Quality
# Evaluates whether Claude's tool usage was correct and helpful

import json
import anthropic

def evaluate_tool_call(
    user_query: str,
    tool_name: str,
    tool_args: dict,
    tool_result: str,
    final_response: str,
    model: str = "claude-haiku-4-5-20251001"  # cheap model for evals
) -> dict:
    """
    Use a judge LLM to evaluate the quality of a tool call.
    Returns scores for: relevance, correctness, completeness.
    """
    client = anthropic.Anthropic()

    judge_prompt = f"""Evaluate this AI agent's tool usage. Score each dimension 1-5.

        USER QUERY: {user_query}

        TOOL CALLED: {tool_name}
        TOOL ARGUMENTS: {json.dumps(tool_args, indent=2)}
        TOOL RESULT: {tool_result[:500]}
        FINAL RESPONSE: {final_response[:500]}

        Evaluate:
        1. tool_relevance (1-5): Was this the right tool to call for the query?
        2. args_correctness (1-5): Were the arguments well-formed and appropriate?
        3. response_quality (1-5): Did the final response correctly use the tool result?

        Respond in JSON: {{"tool_relevance": X, "args_correctness": X, "response_quality": X, "reasoning": "..."}}
    """

    # NOTE: Uncomment to run with real API key
    # response = client.messages.create(
    #     model=model,
    #     max_tokens=300,
    #     messages=[{"role": "user", "content": judge_prompt}]
    # )
    # return json.loads(response.content[0].text)

    # Mock response for demonstration
    return {
        "tool_relevance": 5,
        "args_correctness": 4,
        "response_quality": 5,
        "reasoning": "search_paper was the correct tool; topic was well-specified; response cited specific papers"
    }


# ── Batch evaluation across a test dataset ────────────────────────────────────
test_cases = [
    {
        "user_query": "Find papers about RAG systems",
        "tool_name": "search_paper",
        "tool_args": {"topic": "retrieval augmented generation", "max_results": 5},
        "tool_result": '["2312.10997v1", "2302.12345v2"]',
        "final_response": "I found 2 papers on RAG: [2312.10997v1] Survey of RAG methods..."
    },
    {
        "user_query": "What is the weather today?",
        "tool_name": "search_paper",   # WRONG tool selected
        "tool_args": {"topic": "weather", "max_results": 3},
        "tool_result": '[]',
        "final_response": "I couldn't find weather information through the research tool."
    },
]

print("="*60)
print("  Tool Call Quality Evaluation")
print("="*60)
total_scores = []
for i, case in enumerate(test_cases):
    scores = evaluate_tool_call(**case)
    avg = sum([scores["tool_relevance"], scores["args_correctness"], scores["response_quality"]]) / 3
    total_scores.append(avg)
    print(f"\nTest Case {i+1}: {case['user_query'][:50]}")
    print(f"  Tool relevance:    {scores['tool_relevance']}/5")
    print(f"  Args correctness:  {scores['args_correctness']}/5")
    print(f"  Response quality:  {scores['response_quality']}/5")
    print(f"  Average score:     {avg:.1f}/5")
    print(f"  Reasoning: {scores['reasoning'][:80]}")

print(f"\nOverall average: {sum(total_scores)/len(total_scores):.2f}/5")
```
---
## 9. Real-World Use Cases: 

###  MCP Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      AI AGENT PLATFORM                           │
│                                                                        │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │  Customer Support │    │  Risk & Fraud    │    │  Ops Dashboard  │  │
│  │  Agent (Claude)  │    │  Agent (Claude)  │    │  Agent (Claude) │  │
│  └────────┬─────────┘    └────────┬─────────┘    └───────┬─────────┘  │
│           │                       │                       │           │
│           └───────────────────────┴───────────────────────┘           │
│                                   │                                   │
│              ┌────────────────────▼─────────────────────┐             │
│              │           MCP CLIENT MANAGER              │             │
│              └──┬──────────────┬──────────────┬──────────┘             │
│                 │              │              │                        │
│    ┌────────────▼──┐ ┌─────────▼──┐ ┌────────▼─────────┐             │
│    │  Payments MCP │ │ Fraud MCP  │ │  Compliance MCP  │             │
│    │  Server       │ │ Server     │ │  Server          │             │
│    │               │ │            │ │                  │             │
│    │ • get_txn     │ │ • risk_    │ │ • kyc_check      │             │
│    │ • refund      │ │   score    │ │ • sanctions_     │             │
│    │ • search_txns │ │ • flag_txn │ │   check          │             │
│    │ • get_balance │ │ • get_     │ │ • aml_screen     │             │
│    │ • dispute     │ │   alerts   │ │                  │             │
│    └───────────────┘ └────────────┘ └──────────────────┘             │
│           │                  │                   │                    │
│    ┌──────▼──────┐   ┌───────▼──────┐   ┌────────▼────────┐          │
│    │ Payment DB  │   │ ML Risk Model│   │ Sanctions Lists │          │
│    │ (Postgres)  │   │ (real-time)  │   │ + KYC Database  │          │
│    └─────────────┘   └──────────────┘   └─────────────────┘          │
└────────────────────────────────────────────────────────────────────────┘
```
```python
# 9.1  Payments MCP Server
# Exposes transaction lookup, refund initiation, balance queries

from mcp.server.fastmcp import FastMCP
from typing import Optional
import json
import re
from datetime import datetime, timezone

payments_mcp = FastMCP(name="-payments-server")

# ── Simulated payment database ────────────────────────────────────────────────
TRANSACTIONS_DB = {
    "5TY68390XB789012A": {
        "id": "5TY68390XB789012A", "amount": 149.99, "currency": "USD",
        "status": "COMPLETED", "payer": "alice@example.com",
        "merchant": "Amazon", "timestamp": "2026-02-20T14:32:00Z",
        "description": "Electronics purchase"
    },
    "3BK12345YC098765B": {
        "id": "3BK12345YC098765B", "amount": 29.99, "currency": "USD",
        "status": "PENDING", "payer": "bob@example.com",
        "merchant": "Spotify", "timestamp": "2026-02-23T09:15:00Z",
        "description": "Monthly subscription"
    },
    "7QM99887ZD112233C": {
        "id": "7QM99887ZD112233C", "amount": 500.00, "currency": "EUR",
        "status": "DISPUTED", "payer": "carol@example.com",
        "merchant": "Unknown Merchant", "timestamp": "2026-02-22T18:00:00Z",
        "description": "Unknown charge"
    },
}

ACCOUNTS_DB = {
    "alice@example.com": {"balance": 1250.50, "currency": "USD", "status": "ACTIVE"},
    "bob@example.com":   {"balance": 89.30,   "currency": "USD", "status": "ACTIVE"},
    "carol@example.com": {"balance": 0.00,     "currency": "USD", "status": "FROZEN"},
}


@payments_mcp.tool()
def get_transaction(transaction_id: str) -> str:
    """
    Retrieve full details of a  transaction.

    Args:
        transaction_id: 17-character  transaction ID (e.g. 5TY68390XB789012A)

    Returns:
        JSON with transaction details including amount, status, payer, merchant
    """
    # Validate format
    if not re.match(r'^[A-Z0-9]{17}$', transaction_id):
        return json.dumps({"error": "Invalid transaction ID format. Expected 17 uppercase alphanumeric chars."})

    txn = TRANSACTIONS_DB.get(transaction_id)
    if not txn:
        return json.dumps({"error": f"Transaction {transaction_id} not found"})
    return json.dumps(txn, indent=2)


@payments_mcp.tool()
def search_transactions(
    payer_email: str,
    status: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    limit: int = 10,
) -> str:
    """
    Search transactions for a payer, optionally filtered by status or amount range.

    Args:
        payer_email: Email of the payer
        status: Filter by status (COMPLETED, PENDING, DISPUTED, REFUNDED)
        min_amount: Minimum transaction amount
        max_amount: Maximum transaction amount
        limit: Maximum number of results (default 10, max 100)
    """
    results = [
        txn for txn in TRANSACTIONS_DB.values()
        if txn["payer"] == payer_email
        and (status is None or txn["status"] == status)
        and (min_amount is None or txn["amount"] >= min_amount)
        and (max_amount is None or txn["amount"] <= max_amount)
    ][:min(limit, 100)]

    return json.dumps({
        "total": len(results),
        "transactions": results
    }, indent=2)


@payments_mcp.tool()
def initiate_refund(transaction_id: str, amount: Optional[float] = None, reason: str = "") -> str:
    """
    Initiate a full or partial refund for a completed transaction.

    Args:
        transaction_id: The transaction to refund
        amount: Refund amount (None = full refund)
        reason: Reason for the refund (required for audit)
    """
    txn = TRANSACTIONS_DB.get(transaction_id)
    if not txn:
        return json.dumps({"error": "Transaction not found"})
    if txn["status"] != "COMPLETED":
        return json.dumps({"error": f"Cannot refund transaction in status: {txn['status']}"})
    if not reason.strip():
        return json.dumps({"error": "Refund reason is required"})

    refund_amount = amount or txn["amount"]
    if refund_amount > txn["amount"]:
        return json.dumps({"error": f"Refund amount {refund_amount} exceeds original {txn['amount']}"})

    return json.dumps({
        "refund_id": f"REF-{transaction_id[:8]}",
        "transaction_id": transaction_id,
        "refund_amount": refund_amount,
        "currency": txn["currency"],
        "status": "REFUND_INITIATED",
        "reason": reason,
        "initiated_at": datetime.now(timezone.utc).isoformat(),
        "expected_completion": "3-5 business days"
    }, indent=2)


@payments_mcp.tool()
def get_account_balance(email: str) -> str:
    """
    Get the current  balance for an account.

    Args:
        email: Account email address
    """
    account = ACCOUNTS_DB.get(email)
    if not account:
        return json.dumps({"error": f"Account {email} not found"})
    return json.dumps(account, indent=2)


# Demo all tools
print("===  Payments MCP Server Demo ===\n")

print("1. get_transaction:")
print(payments_mcp._tool_manager.list_tools()[0])

print("\n2. search_transactions for alice:")
# Direct function call for demo
results = [t for t in TRANSACTIONS_DB.values() if t["payer"] == "alice@example.com"]
print(json.dumps(results, indent=2))

print("\n3. initiate_refund:")
refund_result = json.loads(initiate_refund("5TY68390XB789012A", reason="Customer request"))
print(json.dumps(refund_result, indent=2))
```
```python
# 9.2  Fraud Detection MCP Server
# Real-time risk scoring using a rule-based model (replace with ML in prod)

from mcp.server.fastmcp import FastMCP
from typing import Optional
import json
import hashlib
from datetime import datetime, timezone

fraud_mcp = FastMCP(name="-fraud-server")

def compute_risk_score(features: dict) -> tuple[float, list[str]]:
    """
    Rule-based risk scoring (production: replace with XGBoost / neural net).
    Returns (score 0-100, list of triggered risk signals).
    """
    score = 0.0
    signals = []

    # High-value transaction
    if features.get("amount", 0) > 1000:
        score += 25
        signals.append("HIGH_VALUE_TRANSACTION")

    # New device or IP
    if features.get("new_device", False):
        score += 20
        signals.append("NEW_DEVICE")

    # Geo mismatch (user's usual country vs transaction country)
    if features.get("geo_mismatch", False):
        score += 30
        signals.append("GEO_MISMATCH")

    # Velocity: many transactions in short time
    if features.get("txn_velocity_1h", 0) > 5:
        score += 20
        signals.append("HIGH_VELOCITY")

    # Known merchant vs unknown
    if not features.get("known_merchant", True):
        score += 15
        signals.append("UNKNOWN_MERCHANT")

    # Account age
    if features.get("account_age_days", 365) < 30:
        score += 10
        signals.append("NEW_ACCOUNT")

    return min(score, 100), signals


@fraud_mcp.tool()
def score_transaction_risk(
    transaction_id: str,
    amount: float,
    payer_email: str,
    merchant_id: str,
    country_code: str,
    device_fingerprint: str,
    ip_address: str,
) -> str:
    """
    Compute real-time fraud risk score for a transaction (0-100).
    Score interpretation: 0-30=LOW, 31-60=MEDIUM, 61-80=HIGH, 81-100=CRITICAL.

    Args:
        transaction_id: Transaction being evaluated
        amount: Transaction amount in USD
        payer_email: Payer's email address
        merchant_id: Merchant identifier
        country_code: ISO country code of the transaction origin
        device_fingerprint: Hash of device characteristics
        ip_address: Client IP address (used for geo lookup)
    """
    # In production: look up payer history, device history, geo data
    payer_hash = hashlib.md5(payer_email.encode()).hexdigest()
    
    features = {
        "amount": amount,
        "new_device": device_fingerprint not in ["fp_known_001", "fp_known_002"],
        "geo_mismatch": country_code not in ["US", "GB", "DE"],  # payer's home countries
        "txn_velocity_1h": 2,  # In prod: query DB for recent txns
        "known_merchant": merchant_id.startswith("M_VERIFIED_"),
        "account_age_days": 180,
    }

    score, signals = compute_risk_score(features)

    risk_level = (
        "CRITICAL" if score >= 81 else
        "HIGH"     if score >= 61 else
        "MEDIUM"   if score >= 31 else
        "LOW"
    )

    recommendation = {
        "CRITICAL": "BLOCK — do not process, flag for manual review",
        "HIGH":     "CHALLENGE — require 2FA before proceeding",
        "MEDIUM":   "MONITOR — process but track closely",
        "LOW":      "APPROVE — proceed normally",
    }[risk_level]

    return json.dumps({
        "transaction_id": transaction_id,
        "risk_score": round(score, 1),
        "risk_level": risk_level,
        "triggered_signals": signals,
        "recommendation": recommendation,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2)


@fraud_mcp.tool()
def get_fraud_alerts(
    severity: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Get current fraud alerts for the operations team.

    Args:
        severity: Filter by CRITICAL, HIGH, MEDIUM (None = all)
        limit: Max alerts to return
    """
    alerts = [
        {"id": "ALT-001", "severity": "CRITICAL", "type": "CARD_TESTING",
         "description": "Multiple small transactions detected on account u_445566",
         "accounts_affected": 1, "flagged_at": "2026-02-23T07:00:00Z"},
        {"id": "ALT-002", "severity": "HIGH", "type": "GEO_ANOMALY",
         "description": "Transactions from 3 countries within 30 minutes",
         "accounts_affected": 1, "flagged_at": "2026-02-23T08:30:00Z"},
        {"id": "ALT-003", "severity": "MEDIUM", "type": "VELOCITY_SPIKE",
         "description": "15 transactions in past hour from same IP",
         "accounts_affected": 5, "flagged_at": "2026-02-23T09:00:00Z"},
    ]

    if severity:
        alerts = [a for a in alerts if a["severity"] == severity.upper()]

    return json.dumps({"total": len(alerts[:limit]), "alerts": alerts[:limit]}, indent=2)


# Demo
print("=== Fraud Risk Score Demo ===\n")

# Low-risk transaction
result = json.loads(score_transaction_risk(
    transaction_id="5TY68390XB789012A",
    amount=49.99,
    payer_email="alice@example.com",
    merchant_id="M_VERIFIED_AMAZON",
    country_code="US",
    device_fingerprint="fp_known_001",
    ip_address="192.168.1.1"
))
print(f"Low-risk txn: score={result['risk_score']} level={result['risk_level']}")

# High-risk transaction
result_high = json.loads(score_transaction_risk(
    transaction_id="7QM99887ZD112233C",
    amount=1500.00,
    payer_email="carol@example.com",
    merchant_id="UNKNOWN_MERCH",
    country_code="RU",   # geo mismatch
    device_fingerprint="fp_new_xyz",   # new device
    ip_address="10.0.0.1"
))
print(f"High-risk txn: score={result_high['risk_score']} level={result_high['risk_level']}")
print(f"  Signals: {result_high['triggered_signals']}")
print(f"  Action: {result_high['recommendation']}")
```
```python
# 9.3  Compliance MCP Server
# KYC verification, sanctions screening, AML checks

from mcp.server.fastmcp import FastMCP
from typing import Optional
import json
import re
from datetime import datetime, timezone

compliance_mcp = FastMCP(name="-compliance-server")

# Mock sanctions list (in prod: OFAC SDN list, EU consolidated list, UN list)
SANCTIONS_LIST = {
    "names": ["SANCTIONED PERSON A", "BLOCKED ENTITY LLC"],
    "countries": ["KP", "IR", "SY", "CU"],  # North Korea, Iran, Syria, Cuba
    "entities": ["E_BLOCKED_001", "E_BLOCKED_002"],
}

# Mock KYC status database
KYC_DATABASE = {
    "alice@example.com": {
        "status": "VERIFIED",
        "tier": 3,
        "verified_at": "2024-01-15",
        "documents": ["PASSPORT", "PROOF_OF_ADDRESS"],
        "daily_limit_usd": 10000,
    },
    "bob@example.com": {
        "status": "BASIC",
        "tier": 1,
        "verified_at": "2025-06-01",
        "documents": ["EMAIL"],
        "daily_limit_usd": 500,
    },
    "carol@example.com": {
        "status": "PENDING_REVIEW",
        "tier": 0,
        "verified_at": None,
        "documents": [],
        "daily_limit_usd": 0,
    },
}


@compliance_mcp.tool()
def check_kyc_status(email: str, required_tier: int = 1) -> str:
    """
    Check KYC (Know Your Customer) verification status for an account.
    
    KYC Tiers:
      0 = Unverified, 1 = Basic (email only), 2 = Standard (ID), 3 = Enhanced (ID + address)

    Args:
        email: Account email address
        required_tier: Minimum KYC tier required for the operation
    """
    record = KYC_DATABASE.get(email)
    if not record:
        return json.dumps({"error": f"No KYC record for {email}"})

    meets_requirement = record["tier"] >= required_tier
    return json.dumps({
        "email": email,
        "kyc_status": record["status"],
        "tier": record["tier"],
        "meets_tier_requirement": meets_requirement,
        "required_tier": required_tier,
        "daily_limit_usd": record["daily_limit_usd"],
        "documents_on_file": record["documents"],
        "action_required": (
            None if meets_requirement
            else f"Account must complete KYC Tier {required_tier} verification"
        ),
    }, indent=2)


@compliance_mcp.tool()
def screen_sanctions(
    full_name: str,
    country_code: str,
    entity_id: Optional[str] = None,
) -> str:
    """
    Screen an individual or entity against global sanctions lists (OFAC, EU, UN).

    Args:
        full_name: Full legal name of the individual or entity
        country_code: ISO 3166-1 alpha-2 country code
        entity_id: Optional internal entity ID
    """
    hits = []

    # Name screening (simplified fuzzy match)
    name_upper = full_name.upper()
    for sanctioned_name in SANCTIONS_LIST["names"]:
        if sanctioned_name in name_upper or name_upper in sanctioned_name:
            hits.append({"list": "OFAC_SDN", "match_type": "NAME", "matched_value": sanctioned_name})

    # Country screening
    if country_code.upper() in SANCTIONS_LIST["countries"]:
        hits.append({"list": "OFAC_COUNTRY", "match_type": "COUNTRY", "matched_value": country_code})

    # Entity screening
    if entity_id and entity_id in SANCTIONS_LIST["entities"]:
        hits.append({"list": "EU_CONSOLIDATED", "match_type": "ENTITY_ID", "matched_value": entity_id})

    status = "BLOCKED" if hits else "CLEAR"
    return json.dumps({
        "screening_status": status,
        "hits": hits,
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "recommendation": (
            "DO NOT PROCESS — Report to compliance team within 24h" if hits
            else "Proceed with transaction"
        ),
    }, indent=2)


@compliance_mcp.tool()
def check_transaction_limits(
    email: str,
    transaction_amount: float,
    transaction_currency: str = "USD",
) -> str:
    """
    Verify if a transaction is within the account's regulatory limits.

    Args:
        email: Account email
        transaction_amount: Amount to check
        transaction_currency: Currency code (default USD)
    """
    record = KYC_DATABASE.get(email)
    if not record:
        return json.dumps({"error": f"Account {email} not found"})

    # Simplified: use USD equivalent
    amount_usd = transaction_amount * (0.92 if transaction_currency == "EUR" else 1.0)

    # FinCEN CTR threshold: $10,000+ requires Currency Transaction Report
    ctr_required = amount_usd >= 10000

    within_limit = amount_usd <= record["daily_limit_usd"]

    return json.dumps({
        "within_daily_limit": within_limit,
        "daily_limit_usd": record["daily_limit_usd"],
        "transaction_amount_usd": round(amount_usd, 2),
        "ctr_required": ctr_required,
        "ctr_threshold_usd": 10000,
        "action": (
            "BLOCK — exceeds account daily limit" if not within_limit
            else "FILE_CTR — file FinCEN CTR before processing" if ctr_required
            else "PROCEED"
        ),
    }, indent=2)


# Demo
print("=== Compliance MCP Server Demo ===\n")

print("KYC Check (alice, tier 2 required):")
print(check_kyc_status("alice@example.com", required_tier=2))

print("\nKYC Check (carol, tier 1 required):")
kyc = json.loads(check_kyc_status("carol@example.com", required_tier=1))
print(f"  Status: {kyc['kyc_status']}, Meets requirement: {kyc['meets_tier_requirement']}")
print(f"  Action: {kyc['action_required']}")

print("\nSanctions screening (clear):")
screen = json.loads(screen_sanctions("Alice Johnson", "US"))
print(f"  Result: {screen['screening_status']}")

print("\nTransaction limit check ($600 for bob - basic KYC):")
limit = json.loads(check_transaction_limits("bob@example.com", 600, "USD"))
print(f"  Within limit: {limit['within_daily_limit']}, Action: {limit['action']}")
```
```python
# 9.4  Customer Support Agent — full agentic loop with Claude
# Combines all three MCP servers into one unified support experience

import anthropic
import json

def run__support_agent(user_query: str) -> str:
    """
    Simulated  support agent using Claude + mock tool execution.
    In production: replace mock_call_tool with real MCP server calls.
    """
    client = anthropic.Anthropic()

    # All tools from payments + fraud + compliance servers
    tools = [
        {
            "name": "get_transaction",
            "description": "Retrieve full details of a  transaction by ID",
            "input_schema": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string", "description": "17-char transaction ID"}
                },
                "required": ["transaction_id"]
            }
        },
        {
            "name": "search_transactions",
            "description": "Search transactions for a payer, filtered by status or amount",
            "input_schema": {
                "type": "object",
                "properties": {
                    "payer_email":  {"type": "string"},
                    "status":       {"type": "string", "enum": ["COMPLETED", "PENDING", "DISPUTED", "REFUNDED"]},
                    "min_amount":   {"type": "number"},
                    "max_amount":   {"type": "number"},
                },
                "required": ["payer_email"]
            }
        },
        {
            "name": "initiate_refund",
            "description": "Initiate a full or partial refund for a completed transaction",
            "input_schema": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                    "amount":         {"type": "number"},
                    "reason":         {"type": "string"},
                },
                "required": ["transaction_id", "reason"]
            }
        },
        {
            "name": "score_transaction_risk",
            "description": "Get fraud risk score (0-100) for a transaction",
            "input_schema": {
                "type": "object",
                "properties": {
                    "transaction_id":     {"type": "string"},
                    "amount":             {"type": "number"},
                    "payer_email":        {"type": "string"},
                    "merchant_id":        {"type": "string"},
                    "country_code":       {"type": "string"},
                    "device_fingerprint": {"type": "string"},
                    "ip_address":         {"type": "string"},
                },
                "required": ["transaction_id", "amount", "payer_email",
                             "merchant_id", "country_code", "device_fingerprint", "ip_address"]
            }
        },
        {
            "name": "check_kyc_status",
            "description": "Check KYC verification status and limits for an account",
            "input_schema": {
                "type": "object",
                "properties": {
                    "email":         {"type": "string"},
                    "required_tier": {"type": "integer"},
                },
                "required": ["email"]
            }
        },
    ]

    def mock_call_tool(name: str, args: dict) -> str:
        """Route tool calls to the right mock implementation."""
        if name == "get_transaction":
            return get_transaction(**args)
        elif name == "search_transactions":
            results = [t for t in TRANSACTIONS_DB.values()
                      if t["payer"] == args.get("payer_email")
                      and (not args.get("status") or t["status"] == args["status"])]
            return json.dumps({"total": len(results), "transactions": results}, indent=2)
        elif name == "initiate_refund":
            return initiate_refund(**args)
        elif name == "score_transaction_risk":
            return score_transaction_risk(**args)
        elif name == "check_kyc_status":
            return check_kyc_status(**args)
        return json.dumps({"error": f"Unknown tool: {name}"})

    system = """You are a  customer support agent with access to real-time tools.
Be helpful, professional, and concise. Always verify facts using tools before stating them.
Never reveal internal risk scores or sanctions details to customers.
For refunds, always confirm the reason before initiating."""

    messages = [{"role": "user", "content": user_query}]

    print(f"Query: {user_query}")
    print("-" * 60)

    for turn in range(5):  # max 5 tool call rounds
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool] {block.name}({json.dumps(block.input)[:80]})")
                    result = mock_call_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

    final = next((b.text for b in response.content if hasattr(b, "text")), "No response.")
    print(f"\nAgent response:\n{final}")
    return final


# Run the support agent
# NOTE: Requires ANTHROPIC_API_KEY in environment
try:
    run__support_agent(
        "Hi, I'm alice@example.com. Can you show me my recent completed transactions?"
    )
except anthropic.AuthenticationError:
    print("(Set ANTHROPIC_API_KEY to run the live agent)")
    print("\nMock demo — tools that would be called:")
    print("  1. search_transactions(payer_email='alice@example.com', status='COMPLETED')")
    print("  → Returns: 1 transaction (5TY68390XB789012A, $149.99 Amazon)")
    print("  2. Claude formats response for customer")
except Exception as e:
    print(f"Error: {e}")
```
---
## Quick Reference Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        MCP COMPLETE CHEAT SHEET                            │
│                                                                            │
│  INSTALL                                                                   │
│    pip install mcp anthropic langchain-mcp-adapters langchain-anthropic    │
│                                                                            │
│  BUILD A SERVER (FastMCP)                                                  │
│    mcp = FastMCP(name="server", lifespan=my_lifespan)                      │
│    @mcp.tool()   def fn(x: str) -> str: ...  ← action                     │
│    @mcp.resource("uri://{id}") def r(id) -> str: ...  ← read-only data    │
│    @mcp.prompt() def p(x: str) -> str: ...  ← reusable template           │
│    mcp.run(transport="stdio")                                              │
│    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)         │
│                                                                            │
│  CONNECT A CLIENT                                                          │
│    async with stdio_client(StdioServerParameters(...)) as (r,w):           │
│        async with ClientSession(r, w) as session:                          │
│            await session.initialize()                                      │
│            tools = await session.list_tools()                              │
│            result = await session.call_tool("name", {"arg": "val"})        │
│                                                                            │
│  LANGCHAIN INTEGRATION                                                     │
│    tools = await load_mcp_tools(session)       ← mcp → LangChain BaseTool │
│    agent = create_react_agent(llm, tools)      ← ReAct agent              │
│                                                                            │
│  PRODUCTION CHECKLIST                                                      │
│    ✓ TTL cache on expensive tool calls                                     │
│    ✓ Input validation + sanitization                                       │
│    ✓ Rate limiting (token bucket)                                          │
│    ✓ Retry with exponential backoff                                        │
│    ✓ Circuit breaker for downstream APIs                                   │
│    ✓ Structured JSON logging + trace IDs                                   │
│    ✓ Token budget enforcement on tool outputs                              │
│    ✓ McpError for structured error responses                               │
│    ✓ ctx.report_progress() for long-running tools                          │
│                                                                            │
│   USE CASES                                                          │
│    payments-server:    get_transaction, search_transactions,               │
│                        initiate_refund, get_account_balance                │
│    fraud-server:       score_transaction_risk, get_fraud_alerts            │
│    compliance-server:  check_kyc_status, screen_sanctions,                 │
│                        check_transaction_limits                            │
└────────────────────────────────────────────────────────────────────────────┘
```