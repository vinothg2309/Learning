# 05_advanced_responses.py

```python
"""
========================================================
TOPIC 5: Advanced Responses — Streaming, SSE, WebSocket
========================================================

KEY CONCEPTS:
- StreamingResponse  → Stream LLM tokens as they are generated.
                       Client receives partial responses immediately.
- Server-Sent Events → One-way push from server to client over HTTP.
                       Simpler than WebSocket for read-only streams.
- FileResponse       → Serve files efficiently from disk.
- WebSocket          → Full-duplex, persistent connection.
                       Used for real-time chat, live inference sessions.

INTERVIEW ANSWER TIP:
  "For LLM token streaming I use StreamingResponse with an async generator.
   The generator yields each token as it arrives from the model. The client
   receives a chunked HTTP response and can render tokens in real time —
   exactly how ChatGPT's UI works. For multi-turn interactive sessions
   I use WebSocket."
"""

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

app = FastAPI(title="Advanced Responses Demo")


# ──────────────────────────────────────────────────────────────
# 5A. StreamingResponse — LLM token streaming
# ──────────────────────────────────────────────────────────────

async def fake_llm_token_generator(prompt: str):
    """
    Simulates an async LLM token stream.
    In production this would be:
        async for chunk in openai_client.chat.completions.create(..., stream=True):
            yield chunk.choices[0].delta.content or ""
    """
    words = f"The answer to '{prompt}' is: FastAPI is awesome for ML serving!".split()

    for word in words:
        yield word + " "          # yield one token at a time
        await asyncio.sleep(1)  # simulate model generation latency


@app.get("/stream-text")
async def stream_text(prompt: str = "what is FastAPI"):
    """
    Client receives tokens as they are generated — no waiting for full response.

    Test with:
        curl -N "http://localhost:8005/stream-text?prompt=hello"
    The -N flag disables curl's buffering so you see tokens live.
    """
    return StreamingResponse(
        fake_llm_token_generator(prompt),
        media_type="text/plain",
    )


# ──────────────────────────────────────────────────────────────
# 5B. Server-Sent Events (SSE) — standard format
# ──────────────────────────────────────────────────────────────

async def sse_llm_generator(prompt: str):
    """
    SSE format: each event is 'data: <payload>\n\n'
    The browser's EventSource API parses this automatically.
    """
    words = f"SSE stream for: {prompt}".split()

    for i, word in enumerate(words):
        # SSE event format
        event_data = json.dumps({"token": word, "index": i})
        yield f"data: {event_data}\n\n"   # ← double newline ends the event
        await asyncio.sleep(0.15)

    # Signal stream completion
    yield "data: [DONE]\n\n"


@app.get("/stream-sse")
async def stream_sse(prompt: str = "explain transformers"):
    """
    SSE endpoint. Browser can connect via:
        const es = new EventSource('/stream-sse?prompt=hello');
        es.onmessage = (e) => console.log(JSON.parse(e.data));

    Test with curl:
        curl -N -H "Accept: text/event-stream" \
             "http://localhost:8005/stream-sse?prompt=hello"
    """
    return StreamingResponse(
        sse_llm_generator(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ──────────────────────────────────────────────────────────────
# 5C. FileResponse — serve generated reports / model outputs
# ──────────────────────────────────────────────────────────────

@app.get("/download-report")
async def download_report():
    """
    Serve a file from disk. FastAPI handles:
    - ETag / Last-Modified caching headers
    - Range requests (resume partial downloads)
    - Efficient file streaming (no loading entire file into memory)
    """
    # Create a dummy file for demo
    report_path = Path("/tmp/ml_report.txt")
    report_path.write_text("Model Accuracy: 94.2%\nF1 Score: 0.91\n")

    return FileResponse(
        path=str(report_path),
        filename="ml_report.txt",           # sets Content-Disposition header
        media_type="text/plain",
        # background=BackgroundTask(cleanup, report_path)  ← delete after send
    )


# ──────────────────────────────────────────────────────────────
# 5D. WebSocket — full-duplex real-time communication
# ──────────────────────────────────────────────────────────────

class ConnectionManager:
    """Manage multiple active WebSocket connections (e.g., chat rooms)."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)
        print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def send_to_client(self, message: str, ws: WebSocket):
        await ws.send_text(message)

    async def broadcast(self, message: str):
        """Send a message to ALL connected clients."""
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for interactive LLM chat.

    Test with wscat:
        wscat -c ws://localhost:8005/ws/chat/session-abc

    Flow:
    1. Client connects → server accepts.
    2. Client sends a message (prompt).
    3. Server streams LLM tokens back one by one.
    4. Client disconnects → server cleans up.
    """
    await manager.connect(websocket)

    try:
        while True:
            # Wait for the next message from this client
            user_message = await websocket.receive_text()
            print(f"[WS] Session {session_id} sent: {user_message}")

            # Stream response tokens back
            words = f"[AI] You said '{user_message}'. Here is my response!".split()
            for word in words:
                await websocket.send_text(word + " ")
                await asyncio.sleep(0.08)

            # Signal end of this turn
            await websocket.send_text("\n[TURN_DONE]")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # Optionally notify other clients
        await manager.broadcast(f"Session {session_id} has left the chat.")


@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    Pure streaming WebSocket — server pushes data without waiting
    for client messages. Useful for live model monitoring dashboards.
    """
    await websocket.accept()
    try:
        counter = 0
        while True:
            counter += 1
            payload = json.dumps({
                "session": session_id,
                "tick": counter,
                "gpu_util": 72 + (counter % 10),  # fake metric
            })
            await websocket.send_text(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        print(f"[WS] Stream {session_id} disconnected at tick {counter}")


if __name__ == "__main__":
    uvicorn.run("05_advanced_responses:app", host="0.0.0.0", port=8005, reload=True)
```
