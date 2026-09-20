import os
import json
import urllib.request
import urllib.error

from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    return {
        "status": "ok",
        "service": "Athoy AI Backend",
        "ai": "Gemini"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured yet."
        )

    message = req.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message is empty."
        )

    contents = []

    for item in req.history[-12:]:
        role = item.get("role")
        content = item.get("content", "").strip()

        if not content:
            continue

        if role == "user":
            gemini_role = "user"
        elif role == "assistant":
            gemini_role = "model"
        else:
            continue

        contents.append({
            "role": gemini_role,
            "parts": [
                {"text": content}
            ]
        })

    contents.append({
        "role": "user",
        "parts": [
            {"text": message}
        ]
    })

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are Athoy AI, a helpful, accurate and friendly "
                        "AI assistant. Answer clearly and naturally. "
                        "If you are unsure, say so instead of inventing facts."
                    )
                }
            ]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

        candidates = result.get("candidates", [])

        if not candidates:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned no response."
            )

        parts = candidates[0].get("content", {}).get("parts", [])

        reply = "".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        ).strip()

        if not reply:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty response."
            )

        return {"reply": reply}

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")

        raise HTTPException(
            status_code=500,
            detail=f"Gemini API error: {error_body}"
        )

    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Connection error: {str(e)}"
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI request failed: {str(e)}"
        )
