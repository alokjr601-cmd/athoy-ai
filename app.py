import os
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Athoy AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

@app.get("/")
def root():
    return {"status": "ok", "service": "Athoy AI Backend"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/api/chat")
def chat(req: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured yet.")

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    input_messages = []
    for item in req.history[-12:]:
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            input_messages.append({"role": role, "content": content})

    input_messages.append({"role": "user", "content": message})

    try:
        response = client.responses.create(
            model=model,
            instructions=(
                "You are Athoy AI, a helpful, accurate and friendly AI assistant. "
                "Answer clearly and naturally. If you are unsure, say so instead of inventing facts."
            ),
            input=input_messages,
        )
        return {"reply": response.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")
