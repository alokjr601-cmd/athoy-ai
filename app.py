import os
import json
import time
import urllib.request
import urllib.error

from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Athoy AI Backend",
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Request Model
# =========================================================

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


# =========================================================
# Basic Routes
# =========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Athoy AI Backend",
        "ai": "Gemini",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# Gemini Helpers
# =========================================================

def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured."
        )

    return api_key


def get_preferred_model():
    return os.getenv(
        "GEMINI_MODEL",
        "gemini-3.6-flash"
    )


def gemini_request(api_key, model, contents):
    """
    Sends a generateContent request to Gemini.
    """

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are Athoy AI, a helpful, accurate and friendly "
                        "AI assistant. Answer clearly and naturally. "
                        "Be concise when appropriate and provide useful "
                        "details when necessary. "
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

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        + model
        + ":generateContent?key="
        + api_key
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

    with urllib.request.urlopen(
        request,
        timeout=60
    ) as response:

        return json.loads(
            response.read().decode("utf-8")
        )


# =========================================================
# Find Available Gemini Models
# =========================================================

def get_available_models(api_key):
    """
    Ask Gemini which models are currently available
    for this API key.
    """

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models?key="
        + api_key
    )

    request = urllib.request.Request(
        url,
        headers={
            "Content-Type": "application/json"
        },
        method="GET"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        models = []

        for model in result.get("models", []):

            name = model.get("name", "")

            methods = model.get(
                "supportedGenerationMethods",
                []
            )

            if (
                name.startswith("models/")
                and "generateContent" in methods
            ):
                clean_name = name.replace(
                    "models/",
                    "",
                    1
                )

                models.append(clean_name)

        return models

    except Exception:
        return []


# =========================================================
# Choose Fallback Model
# =========================================================

def choose_fallback_model(api_key, current_model):
    """
    Select an available Gemini model if the preferred model
    is temporarily unavailable.
    """

    models = get_available_models(api_key)

    if not models:
        return None

    # Do not immediately choose the same model.
    models = [
        m for m in models
        if m != current_model
    ]

    if not models:
        return None

    # Prefer Flash models because Athoy AI is a chat application.
    flash_models = [
        m for m in models
        if "flash" in m.lower()
    ]

    if flash_models:

        # Prefer newer-looking model names.
        flash_models.sort(
            key=lambda x: x.lower(),
            reverse=True
        )

        return flash_models[0]

    # If no Flash model is available,
    # use any model supporting generateContent.
    models.sort(
        key=lambda x: x.lower(),
        reverse=True
    )

    return models[0]


# =========================================================
# Extract Gemini Response
# =========================================================

def extract_reply(result):

    candidates = result.get(
        "candidates",
        []
    )

    if not candidates:
        return ""

    content = candidates[0].get(
        "content",
        {}
    )

    parts = content.get(
        "parts",
        []
    )

    text_parts = []

    for part in parts:

        text = part.get(
            "text",
            ""
        )

        if text:
            text_parts.append(text)

    return "".join(text_parts).strip()


# =========================================================
# Chat Endpoint
# =========================================================

@app.post("/api/chat")
def chat(req: ChatRequest):

    api_key = get_api_key()

    message = req.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message is empty."
        )


    # -----------------------------------------------------
    # Prepare conversation history
    # -----------------------------------------------------

    contents = []

    for item in req.history[-12:]:

        role = item.get("role")

        content = item.get(
            "content",
            ""
        ).strip()

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
                {
                    "text": content
                }
            ]
        })


    contents.append({
        "role": "user",
        "parts": [
            {
                "text": message
            }
        ]
    })


    # -----------------------------------------------------
    # Preferred model
    # -----------------------------------------------------

    preferred_model = get_preferred_model()


    # -----------------------------------------------------
    # Attempt 1 + Retry
    # -----------------------------------------------------

    last_error = None

    for attempt in range(2):

        try:

            result = gemini_request(
                api_key,
                preferred_model,
                contents
            )

            reply = extract_reply(result)

            if reply:
                return {
                    "reply": reply,
                    "model": preferred_model
                }

            last_error = "Gemini returned an empty response."

        except urllib.error.HTTPError as e:

            error_body = e.read().decode(
                "utf-8",
                errors="ignore"
            )

            last_error = error_body

            # 503 = temporary service unavailable.
            # Wait and retry.
            if e.code == 503:

                if attempt == 0:
                    time.sleep(3)
                    continue

            # 429 = rate limit.
            if e.code == 429:

                if attempt == 0:
                    time.sleep(3)
                    continue

            # 404 means model is unavailable.
            if e.code == 404:
                break

            break

        except Exception as e:

            last_error = str(e)

            if attempt == 0:

                time.sleep(2)

                continue

            break


    # -----------------------------------------------------
    # Fallback model
    # -----------------------------------------------------

    fallback_model = choose_fallback_model(
        api_key,
        preferred_model
    )


    if fallback_model:

        try:

            result = gemini_request(
                api_key,
                fallback_model,
                contents
            )

            reply = extract_reply(result)

            if reply:

                return {
                    "reply": reply,
                    "model": fallback_model
                }

        except urllib.error.HTTPError as e:

            fallback_error = e.read().decode(
                "utf-8",
                errors="ignore"
            )

            last_error = fallback_error

        except Exception as e:

            last_error = str(e)


    # -----------------------------------------------------
    # Final Error
    # -----------------------------------------------------

    raise HTTPException(
        status_code=503,
        detail=(
            "Gemini is temporarily unavailable. "
            "The backend retried the request and "
            "also checked for an available fallback model. "
            "Please try again shortly. "
            f"Details: {last_error}"
        )
        )
