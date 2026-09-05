"""Streaming client for a locally running llama.cpp server.

The llama.cpp server exposes an OpenAI-compatible API at /v1/chat/completions
and /v1/completions. We stream tokens via SSE for the main chat flow, and do a
quick non-streaming call for the persona router.

**Two ways to ask for a persona's turn**, and the difference is most of what
separates this app's dialogue from a dedicated roleplay front-end:

* ``PromptFormat.CHAT`` sends roles to /v1/chat/completions. The backend
  wraps them in the model's instruct template, which is a request for an
  *assistant's answer* — complete, tidy, resolved, and addressed to a user.
  No amount of persona prompt fully undoes that framing, because it is
  applied after the prompt and closer to the generation point.
* ``PromptFormat.TRANSCRIPT`` (the default) sends one flat script to
  /v1/completions, ending at the responding persona's own name:

      [Tony]: what about the harbour?
      [Alex]:

  The model is not being asked for an answer; it is continuing a
  conversation it can already see the rhythm of. Same model, same persona,
  different job.

Transcript mode falls back to chat mode by itself when the backend has no
completion endpoint, and tool calling has no completion-endpoint
equivalent, so a persona with tools enabled always takes the chat path.

Tool calling: stream_chat_with_tools() runs a fully agentic loop — when the
LLM answers with tool_calls, we invoke each tool (built-in tools from
app/services/builtin.py first, everything else on its owning MCP server
via app/services/tool_registry.py), feed the results back, and repeat until
the LLM produces a plain text answer.
"""

import json
import logging
import uuid
from typing import AsyncGenerator, Dict, List, Optional

import httpx

from app.config import Persona, PromptFormat, get_settings
from app.services import builtin, mcp_client
from app.services.tool_registry import get_server_for_tool

logger = logging.getLogger(__name__)


def _base_payload(
    messages: List[dict],
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
) -> dict:
    """Common /v1/chat/completions payload fields (model, sampling, streaming).

    *max_tokens* overrides the configured ceiling for this call — the chat
    flow derives it from the room/persona length tier. *stop* carries the
    other room personas' speaker prefixes so the backend stops before
    continuing someone else's turn.
    """
    settings = get_settings()
    payload = {
        "model": settings.llm.model,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else settings.llm.max_tokens,
        "temperature": settings.llm.temperature,
        "stream": True,
    }
    if stop:
        payload["stop"] = stop
    return payload


def render_transcript(messages: List[dict], persona_name: str) -> str:
    """The messages list as one flat script, primed for *persona_name*.

    The system message becomes a header, every turn keeps the "[Name]: "
    tagging the room preamble already explains, and the whole thing ends
    on the responding persona's own tag with nothing after it — that empty
    line is the entire mechanism. A model handed it has one obvious job:
    say the next thing this person says.

    The persona's own past turns are untagged in the messages list (they
    are its "assistant" role there) and are re-tagged here, because in a
    script every line needs a speaker or the transcript stops reading as
    one.
    """
    header: List[str] = []
    turns: List[str] = []
    for msg in messages:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        role = msg.get("role")
        if role == "system":
            header.append(content)
        elif role == "assistant":
            turns.append(f"[{persona_name}]: {content}")
        else:
            # Already tagged by build_llm_messages, human and personas alike.
            turns.append(content)

    # One newline between turns, a blank line only after the header: the
    # transcript has to look like a conversation, and lines separated by
    # blank lines read as separate blocks of writing instead.
    body = "\n".join(turns)
    prompt = "\n\n".join(part for part in ("\n\n".join(header), body) if part)
    # No trailing space after the colon: the model emits its own leading
    # space, and a space we add is a token boundary we chose for it.
    return f"{prompt}\n[{persona_name}]:" if prompt else f"[{persona_name}]:"


def _completion_payload(
    prompt: str,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
) -> dict:
    """/v1/completions payload — the transcript-mode twin of _base_payload."""
    settings = get_settings()
    payload = {
        "model": settings.llm.model,
        "prompt": prompt,
        "max_tokens": max_tokens if max_tokens is not None else settings.llm.max_tokens,
        "temperature": settings.llm.temperature,
        "stream": True,
    }
    if stop:
        payload["stop"] = stop
    return payload


class CompletionEndpointMissing(Exception):
    """The backend has no usable /v1/completions — fall back to chat mode."""


async def _iter_text_chunks(payload: dict) -> AsyncGenerator[dict, None]:
    """Yield one `choices[0]` dict per SSE line from /v1/completions.

    Same shape as _iter_completion_chunks, except the token lives in
    "text" rather than "delta.content". Raises CompletionEndpointMissing
    when the endpoint is not there, so the caller can retry in chat mode
    rather than failing a turn over a backend difference.
    """
    settings = get_settings()
    url = f"{settings.llm.base_url}/v1/completions"

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code in (404, 405, 501):
                await resp.aread()
                raise CompletionEndpointMissing(
                    f"{url} returned {resp.status_code}"
                )
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    yield chunk["choices"][0]
                except (json.JSONDecodeError, KeyError, IndexError) as exc:
                    logger.warning("Malformed SSE chunk from LLM: %s", exc)
                    continue


async def _iter_completion_chunks(payload: dict) -> AsyncGenerator[dict, None]:
    """Yield one `choices[0]` dict per SSE data line from the LLM.

    Each dict carries the "delta" and, on the final line, "finish_reason".
    Malformed lines are logged and skipped rather than aborting the stream.
    """
    settings = get_settings()
    url = f"{settings.llm.base_url}/v1/chat/completions"

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
                    yield chunk["choices"][0]
                except (json.JSONDecodeError, KeyError, IndexError) as exc:
                    logger.warning("Malformed SSE chunk from LLM: %s", exc)
                    continue


async def _stream_transcript(
    messages: List[Dict[str, str]],
    persona_name: str,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
) -> AsyncGenerator[dict, None]:
    """Stream a persona's turn as a continuation of a flat transcript.

    Yields the same events as stream_chat(). Raises
    CompletionEndpointMissing before yielding anything if the backend has
    no completion endpoint, which is what makes the caller's fallback safe
    — nothing has reached the user at that point.
    """
    prompt = render_transcript(messages, persona_name)
    payload = _completion_payload(prompt, max_tokens=max_tokens, stop=stop)
    finish_reason: Optional[str] = None
    # The prompt ends at "[Name]:" with no trailing space, so the model's
    # first token usually carries one. Stripping it here rather than in the
    # frontend keeps every consumer (bubble, TTS, persistence, the guard)
    # working on the same text.
    seen_text = False
    chunks = _iter_text_chunks(payload)
    try:
        async for choice in chunks:
            token = choice.get("text") or ""
            if token and not seen_text:
                token = token.lstrip()
                if not token:
                    continue
                seen_text = True
            if token:
                yield {"type": "token", "token": token}
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    finally:
        await chunks.aclose()
    yield {"type": "finish", "reason": finish_reason}


async def stream_chat(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    persona_name: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Stream a reply from the LLM's /v1/chat/completions endpoint.

    Yields event dicts, matching the shape stream_chat_with_tools() uses so
    the chat router can treat both paths identically:

      {"type": "token", "token": str}
      {"type": "finish", "reason": str | None}   — exactly one, last

    The trailing "finish" event is what makes truncation detectable:
    reason == "length" means the reply was cut off at max_tokens, and the
    caller must mark it so the next persona is not handed a dangling
    sentence to complete.

    With *persona_name* given and the transcript format configured, the
    turn goes to /v1/completions as a flat script instead (see the module
    docstring). A backend without that endpoint falls back here, once per
    call and silently to the user: a different backend is not a reason to
    lose somebody's turn.
    """
    settings = get_settings()
    if persona_name and settings.llm.prompt_format is PromptFormat.TRANSCRIPT:
        try:
            async for event in _stream_transcript(
                messages, persona_name, max_tokens=max_tokens, stop=stop
            ):
                yield event
            return
        except CompletionEndpointMissing as exc:
            logger.warning(
                "No completion endpoint (%s); using chat format for this turn. "
                "Set llm.prompt_format to 'chat' to stop trying.", exc,
            )

    payload = _base_payload(messages, max_tokens=max_tokens, stop=stop)
    finish_reason: Optional[str] = None
    # Hold a handle on the inner generator so it can be closed explicitly.
    # `async for` does NOT close it: when a caller stops early (the reply
    # guard cutting at a speaker prefix) and closes *this* generator,
    # GeneratorExit lands here, the loop unwinds, and the inner generator
    # is left suspended inside `async with httpx.AsyncClient(...)` with the
    # HTTP response still open. Its cleanup then waits on the event loop's
    # asyncgen finalisation hook, so the abandoned request keeps holding a
    # llama.cpp slot and the *next* persona's call queues behind it — which
    # is what made the chat stall after a cut reply.
    chunks = _iter_completion_chunks(payload)
    try:
        async for choice in chunks:
            token = (choice.get("delta") or {}).get("content") or ""
            if token:
                yield {"type": "token", "token": token}
            # Intermediate chunks carry null; keep the last non-null value.
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    finally:
        await chunks.aclose()
    yield {"type": "finish", "reason": finish_reason}


# The router picks a name in a handful of tokens; anything slower than this
# is a server problem, not a slow answer.
ROUTER_TIMEOUT = 15.0
# Prose — a suggested message, a persona draft — can legitimately take a
# local model a minute or more. Matches the streaming path's ceiling.
PROSE_TIMEOUT = 120.0


async def _text_completion(
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Optional[str]:
    """One non-streaming /v1/completions call.

    Returns the text, "" on any failure the caller should treat as an
    empty answer, or None when the endpoint is not there — which is the
    caller's cue to retry in chat format rather than to give up.
    """
    settings = get_settings()
    url = f"{settings.llm.base_url}/v1/completions"
    payload = {
        "model": settings.llm.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code in (404, 405, 501):
                return None
            resp.raise_for_status()
            return (resp.json()["choices"][0].get("text") or "").strip()
    except Exception as exc:
        logger.warning("LLM completion call failed: %s", exc)
        return ""


async def chat_completion(
    messages: List[Dict[str, str]],
    max_tokens: int = 64,
    temperature: Optional[float] = None,
    timeout: float = ROUTER_TIMEOUT,
    persona_name: Optional[str] = None,
) -> str:
    """Non-streaming LLM call. Used for the persona router and suggestions.

    *temperature* defaults to 0.1, which is what the router wants — it is
    picking a name, not writing. Callers producing prose (the suggested
    player message, a persona draft) should pass the configured sampling
    temperature — and PROSE_TIMEOUT. The default timeout is sized for the
    router's sixteen tokens; a persona draft asked to write a hundred-odd
    words hit it every time on a local model, came back as "", and the
    server then finished generating into a closed connection.

    With *persona_name* given and the transcript format configured, the
    call goes to /v1/completions as a flat script, exactly as a real turn
    would — a persona auditioned in one format and played in another is a
    preview of something the app never runs, which is the same reason the
    preview builds the real room preamble.

    Returns the full response text, or empty string on failure.
    """
    settings = get_settings()
    url = f"{settings.llm.base_url}/v1/chat/completions"

    payload = {
        "model": settings.llm.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1 if temperature is None else temperature,
        "stream": False,
    }

    if persona_name and settings.llm.prompt_format is PromptFormat.TRANSCRIPT:
        text = await _text_completion(
            render_transcript(messages, persona_name),
            max_tokens=max_tokens,
            temperature=payload["temperature"],
            timeout=timeout,
        )
        if text is not None:
            return text
        logger.warning(
            "No completion endpoint; using chat format for this call. "
            "Set llm.prompt_format to 'chat' to stop trying."
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
            # "content" is null, not absent, when the model returns a tool
            # call or is stopped at zero tokens. Callers here all treat the
            # result as a string ("".strip(), truthiness), so a None would
            # crash the router and the suggestion endpoint alike.
            return body["choices"][0]["message"]["content"] or ""
    except Exception as exc:
        logger.warning("LLM non-streaming call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Agentic tool-call loop
# ---------------------------------------------------------------------------

def _merge_tool_call_delta(pending: Dict[int, dict], delta: dict) -> None:
    """Accumulate a streamed tool-call delta into the per-index accumulator.

    Tool calls arrive in fragments: the first chunk carries id/type/name,
    later chunks append to function.arguments (and occasionally name).
    Backends that re-send the full name in later deltas are handled by
    resyncing to the latest copy rather than concatenating it.
    """
    index = delta.get("index", 0)
    entry = pending.setdefault(
        index, {"type": "function", "function": {"name": "", "arguments": ""}}
    )
    if delta.get("id"):
        entry["id"] = delta["id"]
    if delta.get("type"):
        entry["type"] = delta["type"]
    fn = delta.get("function") or {}
    if fn.get("name"):
        current_name = entry["function"]["name"]
        if current_name and current_name in fn["name"]:
            # The incoming name CONTAINS what we already have — this is the
            # signature of a backend re-sending the name (in full) in a later
            # delta; concatenating would yield "get_timeget_time". The check
            # is directional on purpose: a genuine name fragment is almost
            # never a superstring of the accumulated name, so normal
            # fragment streaming is unaffected.
            entry["function"]["name"] = fn["name"]
        else:
            entry["function"]["name"] = current_name + fn["name"]
    if fn.get("arguments"):
        entry["function"]["arguments"] += fn["arguments"]


def _normalize_tool_call(tc: dict) -> dict:
    """Fill in the fields a tool call needs before it is sent anywhere."""
    normalized = dict(tc)
    # Some backends omit the id; the follow-up "tool" message must reference
    # it, so synthesize one when missing.
    normalized["id"] = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
    normalized.setdefault("type", "function")
    function = dict(normalized.get("function") or {})
    function["name"] = function.get("name") or ""
    function["arguments"] = function.get("arguments") or ""
    normalized["function"] = function
    return normalized


def _try_parse_arguments(raw: str) -> Optional[dict]:
    """Parse the JSON argument string the LLM produced.

    Returns None when the string is not valid JSON — typically a call cut
    off mid-stream at max_tokens. Callers must refuse to execute such a
    call rather than falling back to guessed/empty arguments.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON tool arguments: %.200s", raw)
        return None
    return parsed if isinstance(parsed, dict) else {"value": parsed}


async def stream_chat_with_tools(
    messages: List[dict],
    tools: List[dict],
    persona: Persona,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
) -> AsyncGenerator[dict, None]:
    """Stream a persona reply, running an agentic tool-call loop.

    Yields event dicts (not SSE strings; the chat router formats them):
      {"type": "token", "token": str}
      {"type": "tool_call", "tool_name": str, "arguments": dict,
       "result": str, "failed": bool}
      {"type": "finish", "reason": str | None}   — exactly one, last

    ``persona`` is required (not defaulted): built-in tools execute
    against it (e.g. add_memory writes to the persona's directory), so a
    caller that omits it is a bug, not a "no built-ins" case.

    The loop continues while the LLM answers with tool_calls, up to
    mcp.max_tool_iterations tool rounds. The FINAL round is sent without
    the tools list so the LLM is forced to produce text — this guarantees
    a non-empty response even when the iteration cap is exhausted.

    Tool calls whose arguments do not parse as JSON (usually a call
    truncated at max_tokens, i.e. finish_reason "length") are NOT
    executed; the LLM receives an "Error:" result explaining why, and
    can retry with a smaller request or answer without the tool.
    """
    settings = get_settings()
    max_iterations = settings.mcp.max_tool_iterations
    tool_list = tools or []

    conversation = list(messages)
    for round_num in range(max_iterations + 1):
        is_final_round = round_num == max_iterations
        payload = _base_payload(conversation, max_tokens=max_tokens, stop=stop)
        if tool_list and not is_final_round:
            payload["tools"] = tool_list
        if is_final_round and tool_list:
            logger.debug(
                "Persona memory: LLM round %d/%d for persona '%s': FINAL round — "
                "tools withheld to force a text answer",
                round_num + 1, max_iterations + 1, persona.name,
            )
        else:
            offered = payload.get("tools", [])
            logger.debug(
                "Persona memory: LLM round %d/%d for persona '%s': %d tool(s) in payload: %s",
                round_num + 1, max_iterations + 1, persona.name,
                len(offered), [t["function"]["name"] for t in offered],
            )

        content_parts: List[str] = []
        pending_tool_calls: Dict[int, dict] = {}
        finish_reason: Optional[str] = None
        # Explicit close for the same reason as stream_chat: a caller that
        # stops early must not strand the HTTP response and its llama.cpp slot.
        chunks = _iter_completion_chunks(payload)
        try:
            async for choice in chunks:
                delta = choice.get("delta") or {}
                token = delta.get("content") or ""
                if token:
                    content_parts.append(token)
                    yield {"type": "token", "token": token}
                for tc_delta in delta.get("tool_calls") or []:
                    _merge_tool_call_delta(pending_tool_calls, tc_delta)
                # Intermediate chunks carry null; keep the last non-null value.
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
        finally:
            await chunks.aclose()

        if not pending_tool_calls:
            # Key diagnostic line: if the round above shows add_memory in
            # the payload but THIS line follows, the wiring is intact and
            # the model simply chose not to call the tool (a prompt/model
            # issue, not an app issue).
            logger.debug(
                "Persona memory: persona '%s' answered with plain text on round %d (no tool calls)",
                persona.name, round_num + 1,
            )
            # Plain text response — the loop is done.
            yield {"type": "finish", "reason": finish_reason}
            return

        # Pathological case: the model emitted tool calls even though no
        # tools were offered on the final round. Executing hallucinated
        # calls would be worse than stopping, so we don't.
        if is_final_round:
            logger.warning(
                "LLM still emitted tool calls on the final (tool-less) round; stopping"
            )
            yield {"type": "finish", "reason": finish_reason}
            return

        tool_calls = [
            _normalize_tool_call(pending_tool_calls[i]) for i in sorted(pending_tool_calls)
        ]
        logger.info(
            "Tool call round %d/%d: %s",
            round_num + 1, max_iterations,
            [tc["function"]["name"] for tc in tool_calls],
        )

        # Append the assistant's tool-call message, then each tool result,
        # in the exact pairing the OpenAI-compatible API expects.
        conversation.append({
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": tool_calls,
        })
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            arguments = _try_parse_arguments(tc["function"]["arguments"])
            if arguments is None:
                # Unparseable arguments — almost always a call truncated by
                # max_tokens (finish_reason "length"). Executing it with
                # guessed/empty args would be worse than refusing: the LLM
                # sees the refusal and can retry smaller or answer directly.
                logger.warning(
                    "Not executing tool '%s': arguments not valid JSON (finish_reason=%s)",
                    tool_name, finish_reason,
                )
                result = (
                    f"Error: the call to '{tool_name}' was not executed because "
                    "its arguments were not valid JSON"
                    + (
                        " (the response hit max_tokens mid-call). Retry with a "
                        "smaller request or answer without the tool."
                        if finish_reason == "length"
                        else ". Retry with valid arguments or answer without the tool."
                    )
                )
                arguments = {}
            else:
                if builtin.is_builtin_tool(tool_name):
                    # Built-ins win name collisions (tool_registry never
                    # registers an MCP tool under a built-in name), and
                    # they run locally — no server lookup, no network.
                    result = builtin.call_builtin_tool(persona, tool_name, arguments)
                else:
                    server = get_server_for_tool(tool_name)
                    if server is None:
                        available = [t["function"]["name"] for t in tool_list]
                        result = (f"Error: unknown tool '{tool_name}'. "
                                  f"Available tools: {available}")
                    else:
                        result = await mcp_client.call_tool(server, tool_name, arguments)
            conversation.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
            yield {
                "type": "tool_call",
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                # Explicit failure flag for the frontend — the client should
                # not re-derive failure by sniffing the "Error:" prefix.
                "failed": result.startswith(mcp_client.ERROR_PREFIX),
            }
