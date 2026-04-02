"""
FastAPI wrapper para llama.cpp server.
Proxea requests al servidor de llama.cpp con formato OpenAI-compatible.

Uso:
    # Primero levantar llama-server en puerto 8080
    # Después:
    uvicorn deploy.server:app --host 0.0.0.0 --port 8000
"""

import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Emirati Family Chatbot")

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 512


class ChatResponse(BaseModel):
    response: str
    tokens_generated: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint de chat que proxea a llama.cpp server."""
    payload = {
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"llama.cpp server error: {e}")

    data = resp.json()
    choice = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return ChatResponse(
        response=choice["content"],
        tokens_generated=usage.get("completion_tokens", 0),
    )
