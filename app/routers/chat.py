"""Chat router — the core SSE streaming endpoint.

Receives a user message, decides which persona should respond (router, random,
or explicit selection), streams tokens back via SSE, and appends the full
response to session history. Messages are persisted to disk per chat room.
"""

import json
import logging
import random
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import (
    TYPICAL_LENGTH_SPECS,
    ChatRoom,
    Persona,
    PlayerProfile,
    TypicalLength,
    derive_max_tokens,
    get_chatrooms,
    get_personas,
    get_settings,
    resolve_typical_length,
)
from app.models import ChatRequest
from app.session import session
from app.services.llm import chat_completion, stream_chat, stream_chat_with_tools
from app.services.reply_guard import ReplyGuard, stop_sequences
from app.services.tool_registry import get_all_tools

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# Shown when a room demands a player profile it does not have yet. The
# frontend matches on this to pop the profile editor open.
PROFILE_REQUIRED_MESSAGE = "This room needs your character profile before you can chat."

# Shown when every persona's reply was cut for writing as somebody else, so
# the turn produced nothing. Silence here looks identical to a hang.
NO_USABLE_REPLY_MESSAGE = (
    "No one managed a reply in their own voice — they kept answering as each other. "
    "Try sending again."
)


# ---------------------------------------------------------------------------
# Persona pool resolution — derive eligible personas from chat room config
# ---------------------------------------------------------------------------

def _resolve_room_personas(chat_room: str) -> list[str]:
    """Return the list of persona names eligible for the given chat room.

    "default" room (or any room not found in config) includes all personas.
    Named rooms are limited to their assigned persona_names.
    This is the authoritative source of truth for persona eligibility —
    no longer dependent on the frontend-maintained session.active_personas.
    """
    all_names = [p.name for p in get_personas().personas]

    if chat_room.lower() == "default":
        return all_names

    chatrooms_config = get_chatrooms()
    room = next(
        (r for r in chatrooms_config.chat_rooms if r.name.lower() == chat_room.lower()),
        None,
    )
    if room is not None:
        # Only include personas that actually exist in the config (empty list means "no one is here")
        return [n for n in room.persona_names if n in all_names]

    # Unknown room — fall back to all personas rather than blocking the chat
    return all_names


def _find_room(chat_room: str) -> Optional[ChatRoom]:
    """The configured room, or None for "default" / an unknown name.

    None means "no room-level config" — the implicit "default" room has no
    chatrooms.yaml entry, so it falls back to the global settings.
    """
    if chat_room.lower() == "default":
        return None
    return next(
        (r for r in get_chatrooms().chat_rooms if r.name.lower() == chat_room.lower()),
        None,
    )


# ---------------------------------------------------------------------------
# Room preamble — who is here, and the rules of the room
# ---------------------------------------------------------------------------

def _user_label(room: Optional[ChatRoom]) -> str:
    """What the human is called in the transcript and the prompt.

    The player's character name when the room has one, else a neutral
    "User". Every consumer takes it from here: the preamble, the "[Name]: "
    tags in history, the stop strings, and the reply guard. If they
    disagreed, a persona could be told not to speak as "Kira" while the
    transcript tagged them "User".
    """
    if room is not None and room.player_profile.name.strip():
        return room.player_profile.name.strip()
    return "User"


def _player_lines(player: PlayerProfile, speaker: str) -> list[str]:
    """The block describing who the human is, for the persona to react to."""
    if not (player.description.strip() or player.appearance.strip()):
        return []

    lines = ["", f"You are talking with {speaker}."]
    if player.description.strip():
        lines.append(f"Who they are: {player.description.strip()}")
    if player.appearance.strip():
        # The "picture", as text — see PlayerProfile.appearance.
        lines.append(f"What they look like: {player.appearance.strip()}")
    lines.append(
        f"Treat {speaker} as that character: react to who they are and how they "
        "look, and address them by name. Never write their lines for them."
    )
    return lines


def _build_room_preamble(
    persona: Persona,
    chat_room: str,
    eligible: list[str],
    length: TypicalLength,
    player: Optional[PlayerProfile] = None,
) -> str:
    """The app-generated block appended to a persona's system prompt.

    Three things depend on it. The roster is what makes "never invent a
    character" enforceable — you cannot forbid inventing people without
    saying who exists. The rules name the two observed failure modes
    explicitly, including continuing a cut-off message, because a truncated
    line in the history reads to a model as a prompt to complete. And the
    player block is how a persona knows who it is talking to.

    The length line is the *only* thing shaping reply length. The derived
    token cap is a runaway guard, not a style control — that distinction is
    the whole point of the tier.
    """
    # A named player character is a better thing to address than "the user",
    # and it is what the personas are told to call them.
    speaker = player.name.strip() if player and player.name.strip() else "the user"

    by_name = {p.name: p for p in get_personas().personas}
    others = [by_name[n] for n in eligible if n in by_name and n != persona.name]
    if others:
        roster = ", ".join(
            f"{p.name} ({p.description})" if p.description else p.name for p in others
        )
        who = f"The only people here are: {roster}, and {speaker}. There is nobody else."
    else:
        who = f"You are the only one here, besides {speaker}. There is nobody else."

    lines = [
        f'You are {persona.name}, in a group chat called "{chat_room}".',
        who,
        "",
        # Every voice in the transcript is tagged, including the human's.
        # Leaving the human untagged made "untagged text" the model's only
        # example of how they write, and personas answering third or fourth
        # copied it.
        'Every message you can see is tagged with who said it, as "[Name]: text" — '
        f"{speaker}'s included. Your own reply is the one untagged voice: write "
        f"only what {persona.name} says, with no tag.",
    ]

    if player is not None:
        lines.extend(_player_lines(player, speaker))

    lines += [
        "",
        f"- Write only as {persona.name}. Never write a line, a reply, or a name "
        "prefix for anyone else.",
        f"- You are not {speaker}. Never speak or write as {speaker}, never answer "
        f"on their behalf, and never write {speaker}'s next message — not even to "
        "move the conversation along.",
        "- Never invent a new character or speak as one.",
        "- Never continue, complete, or rewrite someone else's message, even if it "
        "looks cut off. Respond to it as it stands.",
        "- Do not begin your reply with your own name.",
    ]

    spec = TYPICAL_LENGTH_SPECS[length]
    if spec.words:
        lines.append(
            f"- This is a chat room, not an essay: aim for about {spec.phrasing} "
            f"(~{spec.words} words). Go longer only when the thought genuinely "
            "needs it. Stop at a natural end — never break off mid-word."
        )

    # Stated last so it is the final thing before the transcript. A persona
    # replying third or fourth has seen no assistant turn in this
    # conversation at all, so it has no in-context anchor for its own voice.
    lines += ["", f"It is {persona.name}'s turn. Reply as {persona.name}, and no one else."]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persona router — asks the LLM to pick the best responder
# ---------------------------------------------------------------------------

def _build_router_prompt(user_message: str, chat_room: str) -> list[dict]:
    """Build a minimal prompt that asks the LLM to pick a persona by name."""
    personas_config = get_personas().personas
    eligible = _resolve_room_personas(chat_room)
    active_personas = [p for p in personas_config if p.name in eligible]
    persona_choices = ", ".join(p.name for p in active_personas)

    # Build router hints block — only for personas actually eligible in this room
    hints = "\n".join(
        f"- {p.name}: {p.router_hints}" for p in active_personas
    )

    # Include last N conversation turns for context
    max_context = get_settings().general.max_turns_for_context
    recent = session.history[-max_context:]
    context_lines = []
    for msg in recent:
        if msg.role == "user":
            context_lines.append(f"User: {msg.content}")
        else:
            context_lines.append(f"{msg.persona}: {msg.content}")
    context = "\n".join(context_lines)

    system = (
        "You are a conversation router. Your ONLY job is to pick the best "
        "persona to respond to the user's latest message.\n\n"
        f"Available personas:\n{hints}\n\n"
        f"Recent conversation:\n{context}\n\n"
        f"User's latest message: {user_message}\n\n"
        "Respond with ONLY the name of the best persona. Choose from: "
        f"{persona_choices}. Do not add any explanation."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": "Pick one persona."}]


async def _pick_persona(who_answers: str, user_message: str, chat_room: str) -> str:
    """Determine which persona should respond.

    - "router": ask the LLM to decide
    - "random": pick randomly from eligible room personas
    - explicit name: use that persona directly
    - anything else: fall back to random
    """
    eligible = _resolve_room_personas(chat_room)

    if not eligible:
        raise ValueError(f"No eligible personas for room '{chat_room}'")

    if who_answers == "random":
        return random.choice(eligible)

    if who_answers == "router":
        try:
            prompt = _build_router_prompt(user_message, chat_room)
            result = await chat_completion(prompt, max_tokens=16)
            chosen = result.strip().strip("\"'")
            # Validate the LLM actually returned an eligible name
            if chosen in eligible:
                return chosen
            logger.info("Router returned unknown name '%s', falling back to random", chosen)
        except Exception as exc:
            logger.warning("Router call failed (%s), falling back to random", exc)
        return random.choice(eligible)

    # Explicit persona name — validate it's in this room
    if who_answers in eligible:
        return who_answers

    # Unknown value — fall back to random
    logger.info("Unrecognized who_answers='%s', falling back to random", who_answers)
    return random.choice(eligible)


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

async def _chat_stream(req: ChatRequest) -> AsyncIterator[str]:
    """Generator that yields SSE-formatted JSON lines."""
    # Switch to the requested chat room for persistence
    session.set_current_room(req.chat_room)

    # Resolve eligible personas from the chat room config — the authoritative source
    config = get_personas()
    eligible = _resolve_room_personas(req.chat_room)

    if not eligible:
        yield f'data: {json.dumps({"type": "error", "message": "No eligible personas for this room"})}\n\n'
        yield f'data: {json.dumps({"type": "complete"})}\n\n'
        return

    # Room-level config: None for "default" (no chatrooms.yaml entry), in
    # which case every room setting falls back to the global values.
    room = _find_room(req.chat_room)

    # A room that requires a player profile refuses messages until it has
    # one. Checked here, not only in the frontend, for the same reason
    # persona eligibility is: the server is the authority. Bail before the
    # user message is recorded, so nothing half-happens.
    if room is not None and room.require_player_profile and not room.player_profile.is_complete:
        yield f'data: {json.dumps({"type": "error", "message": PROFILE_REQUIRED_MESSAGE})}\n\n'
        yield f'data: {json.dumps({"type": "complete"})}\n\n'
        return

    settings = get_settings()
    max_replies = min(settings.general.max_persona_replies, len(eligible))

    # Pick the first persona using the configured strategy
    first_persona_name = await _pick_persona(req.who_answers, req.message, req.chat_room)

    # Use frontend-provided message ID or generate one
    user_message_id = req.message_id or str(uuid.uuid4())

    # Add user message to history (persisted automatically)
    session.add_user_message(req.message, user_message_id)

    echo_enabled = room.echo_chamber if room else False
    user_label = _user_label(room)

    # Echo chamber overrides max_replies — only one persona echoes the user.
    # Multiple identical echoes from different personas would be pointless noise.
    if echo_enabled:
        max_replies = 1

    # Two lists, not one: a persona whose reply the guard cuts to nothing
    # has been *tried* but has not *replied*, and must not use up one of the
    # requested reply slots. Conflating them is why a cut reply could leave
    # the user with fewer answers than they asked for — or none at all.
    attempted_personas: list[str] = []
    replied_personas: list[str] = []

    while len(replied_personas) < max_replies:
        if not attempted_personas:
            persona_name = first_persona_name
        else:
            remaining = [n for n in eligible if n not in attempted_personas]
            if not remaining:
                break
            persona_name = random.choice(remaining)

        persona = next((p for p in config.personas if p.name == persona_name), None)
        if not persona:
            yield f'data: {json.dumps({"type": "error", "message": f"Persona {persona_name} not found"})}\n\n'
            return

        attempted_personas.append(persona_name)

        # Generate the assistant message ID BEFORE emitting "start". The
        # frontend stamps it onto every TTS item enqueued during this
        # response, so audio is associated with the correct message no
        # matter when each fetch resolves. Generating it after the stream
        # (and backfilling later) is how audio got misattributed across turns.
        assistant_message_id = str(uuid.uuid4())

        # Emit start event — include the user's message_id so frontend can track it,
        # and this response's message_id so streaming TTS audio can be associated
        # with the correct message from the first token onward.
        yield f'data: {json.dumps({"type": "start", "persona": persona_name, "user_message_id": user_message_id, "message_id": assistant_message_id})}\n\n'

        truncated = False

        if echo_enabled:
            # Echo chamber: bypass the LLM entirely and return the user's message verbatim.
            # No preamble, no length tier, no guard — nothing was generated.
            full_text = req.message
            yield f'data: {json.dumps({"type": "token", "persona": persona_name, "token": full_text})}\n\n'
        else:
            # Length is shaped by the preamble; the derived cap only guards
            # against a runaway, and can never exceed settings.llm.max_tokens.
            length = resolve_typical_length(persona, room, settings.general.typical_length)
            max_tokens = derive_max_tokens(length, settings.llm.max_tokens)

            messages = session.build_llm_messages(
                system_prompt=persona.system_prompt,
                responding_persona=persona_name,
                max_turns_for_context=settings.general.max_turns_for_context,
                room_preamble=_build_room_preamble(
                    persona, req.chat_room, eligible, length,
                    player=room.player_profile if room else None,
                ),
                user_label=user_label,
            )

            # Layer 1: stop before the backend generates another persona's
            # turn at all. Only catches names that exist — the guard below
            # is what catches invented ones. The human counts as a speaker
            # here: a persona answering *as the user* is the same failure.
            other_voices = eligible + [user_label]
            stop = stop_sequences(persona_name, other_voices)
            guard = ReplyGuard(persona_name, other_voices)

            stream = (
                # Agentic path: the LLM may invoke MCP tools mid-reply. The
                # loop runs regardless of show_tool_calls; that flag only
                # controls whether tool_call SSE events are emitted.
                stream_chat_with_tools(
                    messages, get_all_tools(), max_tokens=max_tokens, stop=stop
                )
                if persona.allow_tool_calls
                else stream_chat(messages, max_tokens=max_tokens, stop=stop)
            )

            full_text = ""
            try:
                try:
                    async for event in stream:
                        if event["type"] == "token":
                            # The guard may hold text back briefly at line
                            # starts while it decides whether a speaker
                            # prefix is forming. Only emit what it releases.
                            safe = guard.feed(event["token"])
                            if safe:
                                full_text += safe
                                yield f'data: {json.dumps({"type": "token", "persona": persona_name, "token": safe})}\n\n'
                            if guard.stopped:
                                logger.info(
                                    "Cut %s's reply at speaker prefix %r",
                                    persona_name, guard.cut_at,
                                )
                                break
                        elif event["type"] == "tool_call" and settings.general.show_tool_calls:
                            yield f'data: {json.dumps({"type": "tool_call", "persona": persona_name, "tool_name": event["tool_name"], "arguments": event["arguments"], "result": event["result"], "failed": event["failed"]})}\n\n'
                        elif event["type"] == "finish":
                            # A reply cut off at max_tokens ends mid-sentence.
                            # Marking it keeps the next persona from being
                            # handed a dangling thought to complete.
                            truncated = event.get("reason") == "length"
                finally:
                    # Break leaves the generator suspended; closing it here
                    # releases the upstream HTTP stream promptly.
                    await stream.aclose()

                tail = guard.flush()
                if tail:
                    full_text += tail
                    yield f'data: {json.dumps({"type": "token", "persona": persona_name, "token": tail})}\n\n'
            except Exception as exc:
                logger.error("Streaming error: %s", exc)
                # Release anything the guard was still holding, so text the
                # model did produce is not swallowed by the failure.
                held = guard.flush()
                if held:
                    yield f'data: {json.dumps({"type": "token", "persona": persona_name, "token": held})}\n\n'
                yield f'data: {json.dumps({"type": "error", "message": str(exc)})}\n\n'
                return

        if not full_text.strip():
            # The guard cut the whole reply — the persona opened by writing
            # as someone else and produced nothing of its own. Persisting an
            # empty turn would show a blank bubble and feed the next persona
            # a meaningless "[Name]: " line. Skip it; the frontend reuses the
            # untouched row for whoever speaks next.
            logger.info(
                "%s produced no reply of its own (cut at %r); trying the next persona",
                persona_name, guard.cut_at,
            )
            continue

        # Persist — subsequent personas will see this in history
        session.add_assistant_message(
            full_text, persona_name, assistant_message_id, truncated=truncated
        )

        replied_personas.append(persona_name)

        yield f'data: {json.dumps({"type": "done", "persona": persona_name, "text": full_text, "message_id": assistant_message_id})}\n\n'

    if not replied_personas:
        # Every persona was cut. Without this the turn ends with a start
        # event and nothing after it: an empty bubble and no explanation,
        # which reads as the app having hung.
        logger.warning(
            "No persona produced a usable reply in room '%s' (tried %s)",
            req.chat_room, ", ".join(attempted_personas) or "nobody",
        )
        yield f'data: {json.dumps({"type": "error", "message": NO_USABLE_REPLY_MESSAGE})}\n\n'

    yield f'data: {json.dumps({"type": "complete"})}\n\n'


@router.post("")
async def chat(req: ChatRequest):
    """Accept a user message and return an SSE stream of the AI response."""
    return StreamingResponse(
        _chat_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
