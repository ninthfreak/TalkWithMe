"""Streaming client for a locally running llama.cpp server.

The llama.cpp server exposes an OpenAI-compatible API at /v1/chat/completions.
We stream tokens via SSE for the main chat flow, and do a quick non-streaming
call for the persona router.
"""

import json
import logging
from typing import AsyncGenerator, Dict, List, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


async def stream_chat(
    messages: List[Dict[str, str]],
) -> AsyncGenerator[str, None]:
    """Stream tokens from the LLM's /v1/chat/completions endpoint.

    Yields individual token strings as they arrive.
    """
    settings = get_settings()
    url = f"{settings.llm.base_url}/v1/chat/completions"

    payload = {
        "model": settings.llm.model,
        "messages": messages,
        "max_tokens": settings.llm.max_tokens,
        "temperature": settings.llm.temperature,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    token = chunk["choices"][0].get("delta", {}).get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError) as exc:
                    logger.warning("Malformed SSE chunk from LLM: %s", exc)
                    continue


async def chat_completion(messages: List[Dict[str, str]], max_tokens: int = 64) -> str:
    """Non-streaming LLM call. Used for the persona router.

    Returns the full response text, or empty string on failure.
    """
    settings = get_settings()
    url = f"{settings.llm.base_url}/v1/chat/completions"

    payload = {
        "model": settings.llm.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,  # Low temperature for deterministic routing
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
            return body["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("LLM non-streaming call failed: %s", exc)
        return ""
