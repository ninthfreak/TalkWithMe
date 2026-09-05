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

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import (
    TYPICAL_LENGTH_SPECS,
    ChatRoom,
    Persona,
    TypicalLength,
    derive_max_tokens,
    get_chatrooms,
    get_personas,
    get_player,
    get_settings,
    resolve_typical_length,
)
from app.models import (
    ChatRequest,
    SpeakAsRequest,
    SuggestReplyRequest,
    SuggestReplyResponse,
)
from app import persistence
from app.session import recent_exchanges, session
from app.services import builtin, persona_store
from app.services.llm import PROSE_TIMEOUT, chat_completion, stream_chat, stream_chat_with_tools
from app.services.reply_guard import ReplyGuard, stop_sequences
from app.services.tool_registry import get_all_tools

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# Shown when a room demands the player has adopted a persona and they have
# not. The frontend matches on this to open the character picker.
PERSONA_REQUIRED_MESSAGE = (
    "This room needs you to be playing as someone. Pick a character first."
)

# Shown when every persona's reply was cut for writing as somebody else, so
# the turn produced nothing. Silence here looks identical to a hang.
# Extra attempts allowed per turn to replace replies the guard cut. Without
# slack, a room whose size equals max_persona_replies could never reach the
# requested count once a single reply was cut.
MAX_CUT_RETRIES = 3

NO_USABLE_REPLY_MESSAGE = (
    "No one managed a reply in their own voice — they kept answering as each other. "
    "Try sending again."
)


# ---------------------------------------------------------------------------
# Persona pool resolution — derive eligible personas from chat room config
# ---------------------------------------------------------------------------

def _resolve_room_personas(chat_room: str, exclude_adopted: bool = True) -> list[str]:
    """Return the list of persona names eligible for the given chat room.

    "default" room (or any room not found in config) includes all personas.
    Named rooms are limited to their assigned persona_names.
    This is the authoritative source of truth for persona eligibility —
    no longer dependent on the frontend-maintained session.active_personas.
    """
    all_names = [p.name for p in get_personas().personas]
    # The persona the player has adopted is the player, so it must not also
    # answer as an AI — you would be talking to yourself.
    if exclude_adopted:
        played = get_player().adopted(set(all_names))
        if played:
            all_names = [n for n in all_names if n != played]

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

def _adopted_persona() -> Optional[Persona]:
    """The persona the player is currently playing, or None.

    Resolved against the live persona list every time: the adopted persona
    can be deleted or renamed after the fact, and a dangling reference must
    degrade to "playing as themselves" rather than half-applying.
    """
    personas = {p.name: p for p in get_personas().personas}
    name = get_player().adopted(personas.keys())
    return personas.get(name) if name else None


def _user_label() -> str:
    """What the human is called in the transcript and the prompt.

    The adopted persona's name when there is one, else a neutral "User".
    Every consumer takes it from here: the preamble, the "[Name]: " tags in
    history, the stop strings, and the reply guard. If they disagreed, a
    persona could be told not to speak as "Kira" while the transcript
    tagged them "User".
    """
    adopted = _adopted_persona()
    return adopted.name if adopted else "User"


def _player_lines(player: Optional[Persona], speaker: str) -> list[str]:
    """The block describing who the human is, for the persona to react to.

    The player adopts a configured persona, so this is that persona's own
    description and prompt — the same text that would drive them if the
    LLM were playing them.

    It is also the most contaminating text in the whole system message,
    and the reason is structural: a persona prompt is written as second
    person instructions ("You find everything boring and you say so"), so
    pasting it in raw hands another character a set of instructions
    addressed to "you". An earlier version did exactly that, at full
    length, and it explains the worst version of the complaint personas
    draw: everyone in the room drifting towards the player's manner, and
    still drifting in a brand new room with only one persona in it,
    because the adopted player is global and follows you everywhere.

    So it is quoted, capped, and explicitly labelled as a note about
    somebody else. The cap matters twice over: an 8KB player prompt used
    to be pasted into every persona's system message, drowning their own
    voice by the same arithmetic that made short persona prompts useless.
    """
    if player is None:
        return []

    lines = ["", f"You are talking with {speaker}."]
    if player.description.strip():
        lines.append(f"Who they are: {player.description.strip()}")
    sketch = _trimmed_prompt(player.system_prompt, _PLAYER_SKETCH_CHARS)
    if sketch:
        lines.append(
            f"This is the note {speaker} is played from. It describes THEM, and it "
            f"is not an instruction to you — read it the way you would read a "
            f'note about someone you are about to meet: "{sketch}"'
        )
    lines.append(
        f"Treat {speaker} as that character: react to who they are, and address "
        f"them by name. Never write their lines for them, and never take on their "
        f"manner, opinions or turns of phrase — you are not them."
    )
    return lines


# How much of the persona's own prompt to repeat at the end of the
# preamble. Long enough to carry a voice, short enough that a persona with
# an 8KB prompt does not double the system message.
_VOICE_REMINDER_CHARS = 600

# How much of the *player's* prompt to show the personas. Deliberately
# shorter than the voice reminder: it is someone else's character, it is
# already the most contaminating text in the message (see _player_lines),
# and every character of it is a character not spent on the persona's own
# voice.
_PLAYER_SKETCH_CHARS = 300


def _trimmed_prompt(text: str, limit: int) -> str:
    """A prompt flattened to one line and cut to *limit* characters.

    Cut on a sentence boundary where there is one in range: half a
    sentence reads as a truncation the model may try to complete, which is
    the failure mode the whole containment layer exists to avoid.
    """
    prompt = " ".join(text.split())
    if not prompt or len(prompt) <= limit:
        return prompt
    cut = prompt[:limit]
    for end in (". ", "! ", "? "):
        idx = cut.rfind(end)
        if idx > limit // 2:
            return cut[: idx + 1]
    return cut.rstrip() + "\u2026"


def _voice_reminder(persona: Persona) -> str:
    """The persona's own prompt again, trimmed for the recency position."""
    return _trimmed_prompt(persona.system_prompt, _VOICE_REMINDER_CHARS)


def _build_room_preamble(
    persona: Persona,
    chat_room: str,
    eligible: list[str],
    length: TypicalLength,
    player: Optional[Persona] = None,
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
        # Named shape by shape because the abstract rule above was not
        # enough on its own: models that would never write "Luna:" write
        # "**Luna:**" happily, having learnt it as formatting rather than
        # as taking someone else's turn.
        "- Never start a line with a name and a colon, in any form — not "
        "\"Luna:\", not \"**Luna:**\", not \"[Luna]:\", not \"### Luna\". Each of "
        "those opens somebody else's turn.",
        "- You are writing one message, not a transcript. It has one speaker, "
        "you, and it ends when you stop talking.",
        "- Do not begin your reply with your own name.",
        # The failure this one exists for is not authorship but influence:
        # nobody writes another persona's line, and yet a room where one
        # character is bored ends up with everybody bored. A model
        # continues a transcript, so whatever is salient in it — a mood, a
        # grievance, a favourite word — is the likeliest next thing to
        # generate, and the rules above all permit it. Naming the moods
        # rather than saying "stay in character" is deliberate: the
        # abstract version reads as advice, the concrete one as a rule.
        f"- The others are not you. Their moods, opinions, obsessions and turns "
        f"of phrase are theirs: never drift into echoing them. If someone here "
        f"is bored, or angry, or fixated on one subject, that is a fact about "
        f"them, not about you — react to it as {persona.name}, who may well feel "
        f"nothing of the kind.",
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
    #
    # The persona's *character* is repeated here, not just its name. The
    # system prompt is assembled as "<persona prompt>\n\n<this preamble>",
    # and this preamble runs to ~255 words of text that is identical for
    # everyone in the room — so a typical 15-word persona prompt was about
    # 5% of what the model read, sitting as far from the generation point
    # as it is possible to be. Every persona then sounded like the
    # preamble, which is to say like every other persona. Restating the
    # voice last costs a few dozen tokens and puts it where
    # instruction-following is strongest.
    voice = _voice_reminder(persona)
    if voice:
        lines += [
            "",
            f"It is {persona.name}'s turn. Reply as {persona.name} and no one else, "
            f"in {persona.name}'s own voice rather than a neutral one:",
            voice,
        ]
    else:
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

    # Same windowing as the reply prompt: whole exchanges, so the router
    # never sees answers whose question has been cut away.
    max_context = get_settings().general.max_turns_for_context
    recent = recent_exchanges(session.history, max_context)
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
# Memory injection (docs/feature_persona_memory.md)
# ---------------------------------------------------------------------------

def _system_prompt_with_memories(persona, settings) -> str:
    """The persona's system prompt, with saved memories appended if eligible.

    Qualifying conditions: the global enable_persona_memories flag, a
    non-zero memory_size, and a memories.txt that exists and is not
    blank. Note that allow_tool_calls is deliberately NOT part of this
    gate: a persona that may not call tools can still benefit from
    memories it saved earlier (injection and adding are independent).

    The memory budget is enforced on the read path as well as the write
    path: the file may have been edited by an external process (the
    README explicitly encourages it), so an over-limit file is purged
    oldest-first to the persona's memory_size before injection, rather
    than being handed to the LLM verbatim.
    """
    if not (settings.general.enable_persona_memories and persona.memory_size > 0):
        logger.debug(
            "Persona memory: NOT injecting saved memories for '%s' "
            "(enable_persona_memories=%s, memory_size=%d)",
            persona.name, settings.general.enable_persona_memories, persona.memory_size,
        )
        return persona.system_prompt
    if persona.persona_dir is None:
        return persona.system_prompt
    # Cheap no-op when the file is already within budget; repairs the
    # on-disk file as a side effect when it isn't (e.g. an external
    # writer ignored the persona's budget).
    persona_store.purge_memories_to_limit(persona.persona_dir, persona.memory_size)
    memories = persona_store.read_memories(persona.persona_dir)
    if not memories.strip():
        return persona.system_prompt
    memory_lines = [line for line in memories.splitlines() if line.strip()]
    logger.debug(
        "Persona memory: injecting %d saved memory line(s) into the system prompt of '%s'",
        len(memory_lines), persona.name,
    )
    return (
        persona.system_prompt
        + "\n\nYou have the following memories related to the user:\n"
        + memories
    )


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

async def _chat_stream(req: ChatRequest) -> AsyncIterator[str]:
    """Generator that yields SSE-formatted JSON lines."""
    # Switch to the requested chat room for persistence. The name becomes a
    # directory, so an invalid one is refused here rather than reaching the
    # filesystem.
    try:
        session.set_current_room(req.chat_room)
    except persistence.UnsafeRoomName as exc:
        yield f'data: {json.dumps({"type": "error", "message": str(exc)})}\n\n'
        yield f'data: {json.dumps({"type": "complete"})}\n\n'
        return

    # Every write this turn makes goes to the room the turn started in.
    # Reading session.current_room at persist time instead meant a room
    # switch mid-reply filed the reply in the room the user moved to.
    turn_room = session.current_room

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
    if room is not None and room.require_player_persona and _adopted_persona() is None:
        yield f'data: {json.dumps({"type": "error", "message": PERSONA_REQUIRED_MESSAGE})}\n\n'
        yield f'data: {json.dumps({"type": "complete"})}\n\n'
        return

    settings = get_settings()
    requested_replies = settings.general.max_persona_replies
    max_replies = min(requested_replies, len(eligible))
    if max_replies < requested_replies:
        # The commonest reason "max persona replies" appears not to work:
        # the setting is global but the room is smaller than it. Say so,
        # rather than leaving the user to guess why 6 produced 4.
        logger.info(
            "Room '%s' has %d persona(s), so at most %d can reply "
            "(max_persona_replies is %d)",
            req.chat_room, len(eligible), max_replies, requested_replies,
        )

    # Pick the first persona using the configured strategy
    first_persona_name = await _pick_persona(req.who_answers, req.message, req.chat_room)

    # Use frontend-provided message ID or generate one
    user_message_id = req.message_id or str(uuid.uuid4())

    # Add user message to history (persisted automatically)
    session.add_user_message(req.message, user_message_id, room=turn_room)

    user_label = _user_label()

    # A cut reply costs an attempt but not a slot. Tracking attempts per
    # persona (rather than a flat "already tried" list) lets a persona whose
    # reply was cut be re-rolled once everyone untried has had a go — which
    # matters most when max_replies equals the room size, where a single cut
    # would otherwise make the requested count unreachable. A persona that
    # actually replied is never asked again in the same turn.
    replied_personas: list[str] = []
    attempts: dict[str, int] = {}
    attempt_budget = max_replies + MAX_CUT_RETRIES

    while len(replied_personas) < max_replies and sum(attempts.values()) < attempt_budget:
        if not attempts:
            persona_name = first_persona_name
        else:
            candidates = [n for n in eligible if n not in replied_personas]
            if not candidates:
                break
            # Untried personas first; only re-roll a cut one when nobody
            # fresh is left.
            fewest = min(attempts.get(n, 0) for n in candidates)
            persona_name = random.choice(
                [n for n in candidates if attempts.get(n, 0) == fewest]
            )

        persona = next((p for p in config.personas if p.name == persona_name), None)
        if not persona:
            yield f'data: {json.dumps({"type": "error", "message": f"Persona {persona_name} not found"})}\n\n'
            return

        attempts[persona_name] = attempts.get(persona_name, 0) + 1

        # Diagnostic trail (DEBUG): the three inputs the add_memory feature
        # gates on, exactly as the runtime sees them (post-cache, post-parse).
        logger.debug(
            "Persona memory: persona '%s' decision inputs: allow_tool_calls=%s, "
            "memory_size=%d, enable_persona_memories=%s, persona_dir=%s",
            persona_name, persona.allow_tool_calls, persona.memory_size,
            settings.general.enable_persona_memories, persona.persona_dir,
        )

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

        # Length is shaped by the preamble; the derived cap only guards
        # against a runaway, and can never exceed settings.llm.max_tokens.
        length = resolve_typical_length(persona, room, settings.general.typical_length)
        max_tokens = derive_max_tokens(length, settings.llm.max_tokens)

        messages = session.build_llm_messages(
            # The persona's own prompt, with its saved memories appended
            # when the feature is on for it (docs/feature_persona_memory.md).
            system_prompt=_system_prompt_with_memories(persona, settings),
            responding_persona=persona_name,
            max_turns_for_context=settings.general.max_turns_for_context,
            room_preamble=_build_room_preamble(
                persona, req.chat_room, eligible, length,
                player=_adopted_persona(),
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

        # Agentic path: the LLM may invoke MCP tools AND the built-in
        # tools (add_memory) mid-reply. The loop runs regardless of
        # show_tool_calls; that flag only controls whether tool_call SSE
        # events are emitted.
        if persona.allow_tool_calls:
            tools = get_all_tools() + builtin.get_builtin_tools_for(persona, settings)
            logger.debug(
                "Persona memory: persona '%s' — agentic path, %d tool(s) supplied to LLM: %s",
                persona_name, len(tools), [t["function"]["name"] for t in tools],
            )
            stream = stream_chat_with_tools(
                messages, tools, persona, max_tokens=max_tokens, stop=stop
            )
        else:
            logger.debug(
                "Persona memory: persona '%s' — allow_tool_calls is False, plain "
                "streaming path taken; NO tools of any kind supplied to LLM",
                persona_name,
            )
            stream = stream_chat(messages, max_tokens=max_tokens, stop=stop)

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
            full_text, persona_name, assistant_message_id,
            truncated=truncated, room=turn_room,
        )

        replied_personas.append(persona_name)

        yield f'data: {json.dumps({"type": "done", "persona": persona_name, "text": full_text, "message_id": assistant_message_id})}\n\n'

    if not replied_personas:
        # Every persona was cut. Without this the turn ends with a start
        # event and nothing after it: an empty bubble and no explanation,
        # which reads as the app having hung.
        logger.warning(
            "No persona produced a usable reply in room '%s' (tried %s)",
            req.chat_room, ", ".join(attempts) or "nobody",
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


# ---------------------------------------------------------------------------
# Suggested player message
# ---------------------------------------------------------------------------
#
# The inverse of everything the reply guard enforces: here the LLM *is*
# asked to write as the player. That is fine because the player asked for
# it and the result lands in the input box for them to edit — it is a
# drafting aid, not an auto-reply. Nothing is sent or persisted until they
# press send.

# How many of the player's own past messages to show as a voice sample.
_VOICE_SAMPLE_MESSAGES = 8


def _player_voice_sample(user_label: str) -> list[str]:
    """The player's own recent messages, oldest first.

    This is the "how they write" half of the prompt: vocabulary, sentence
    shape, how much they say. Their own words carry that better than any
    description of them could.
    """
    said = [m.content for m in session.history if m.role == "user"]
    return said[-_VOICE_SAMPLE_MESSAGES:]


def _build_suggestion_prompt(chat_room: str) -> list[dict]:
    """Ask the LLM for the player's next message, in the player's voice.

    Two different jobs, kept apart on purpose:

    * the **character description** decides *what* to say — the manner,
      what this character cares about, how they would react;
    * the **voice sample** (their own past messages) decides *how* to say
      it — vocabulary, sentence shape, how much they usually write.

    An earlier version mentioned the description in a single line while
    giving the sample a labelled section and an explicit "match this"
    instruction, so drafts came back sounding right and behaving like
    nobody in particular. Both now get a section and an instruction.

    Written in the second person throughout: the model is the character,
    not a writer working on their behalf.
    """
    room = _find_room(chat_room)
    user_label = _user_label()
    eligible = _resolve_room_personas(chat_room)
    settings = get_settings()
    length = resolve_typical_length(None, room, settings.general.typical_length)
    profile = _adopted_persona()

    named = profile is not None
    if named:
        lines = [f'You are {user_label}, in a group chat called "{chat_room}".']
    else:
        lines = [
            f'You are the human in a group chat called "{chat_room}", shown in '
            f'the transcript as "{user_label}".'
        ]

    # Deliberately NOT _player_lines(): that block is addressed to personas
    # talking *to* the player and ends "never write their lines for them",
    # which is the exact opposite of this task.
    if profile and profile.description.strip():
        lines += ["", "Who you are:", profile.description.strip()]
    if profile and profile.system_prompt.strip():
        lines += ["", "How you are written:", profile.system_prompt.strip()]

    if profile is not None:
        lines += [
            "",
            "Stay in character. What you say should follow from who you are — "
            "your manner, what you care about, and how someone like you would "
            "react to what was just said.",
        ]

    if eligible:
        lines += ["", f"The others here are: {', '.join(eligible)}."]

    sample = _player_voice_sample(user_label)
    if sample:
        lines += [
            "",
            "How you write — match this voice, vocabulary and typical length "
            "(these are your own earlier messages):",
        ]
        lines += [f"- {s}" for s in sample]

    recent = recent_exchanges(session.history, settings.general.max_turns_for_context)
    if recent:
        lines += ["", "The conversation so far:"]
        for msg in recent:
            speaker = user_label if msg.role == "user" else msg.persona
            lines.append(f"[{speaker}]: {msg.content}")

    spec = TYPICAL_LENGTH_SPECS[length]
    length_line = (
        f" Aim for about {spec.phrasing} (~{spec.words} words)." if spec.words else ""
    )
    lines += [
        "",
        "Write your next message. Output only the message itself — no name "
        "prefix, no quotation marks, no narration, and no explanation of what "
        f"you wrote.{length_line}",
    ]

    return [
        {"role": "system", "content": "\n".join(lines)},
        {"role": "user", "content": "Write your next message."},
    ]


@router.post("/suggest", response_model=SuggestReplyResponse)
async def suggest_reply(req: SuggestReplyRequest):
    """Draft the player's next message for them to review and edit."""
    room = _find_room(req.chat_room)
    settings = get_settings()
    length = resolve_typical_length(None, room, settings.general.typical_length)

    text = await chat_completion(
        _build_suggestion_prompt(req.chat_room),
        max_tokens=derive_max_tokens(length, settings.llm.max_tokens),
        # Prose, not routing: use the configured sampling temperature so a
        # suggestion does not come out flat and repetitive, and give it
        # the time prose takes.
        temperature=settings.llm.temperature,
        timeout=PROSE_TIMEOUT,
    )

    if not text.strip():
        raise HTTPException(
            status_code=503,
            detail="The LLM did not return a suggestion. Is the server running?",
        )

    # The same guard the personas get, pointed the other way: strip a
    # "[Tony]: " prefix the model added, and cut it off if it carries on
    # into a persona's reply.
    user_label = _user_label()
    guard = ReplyGuard(user_label, _resolve_room_personas(req.chat_room))
    cleaned = (guard.feed(text) + guard.flush()).strip()

    if not cleaned:
        # The guard cut the draft to nothing — it opened as somebody else.
        # Falling back to the raw text put a "[Luna]: ..." line in the input
        # box for the player to send as themselves, which is exactly the
        # nested-transcript pattern the guard exists to keep out of history.
        raise HTTPException(
            status_code=503,
            detail="The draft came back in someone else's voice. Try again.",
        )
    return SuggestReplyResponse(text=cleaned)


# ---------------------------------------------------------------------------
# Speaking as a persona
# ---------------------------------------------------------------------------
#
# The player writes a line and a persona says it. No LLM is involved: the
# text is the player's, attributed to the persona they picked. This
# replaces the old "echo chamber" room mode, which achieved the same thing
# by bouncing your own message back at you and left the room stuck in that
# mode until you remembered to switch it off.
#
# It streams the same SSE shape as a normal reply so the frontend renders
# the bubble, persists it and speaks it through TTS with no special casing.

async def _speak_stream(req: SpeakAsRequest) -> AsyncIterator[str]:
    """Emit one persona message authored by the player."""
    try:
        session.set_current_room(req.chat_room)
    except persistence.UnsafeRoomName as exc:
        yield f'data: {json.dumps({"type": "error", "message": str(exc)})}\n\n'
        yield f'data: {json.dumps({"type": "complete"})}\n\n'
        return

    turn_room = session.current_room

    # The room's own cast, the persona the player has adopted included:
    # speaking as your own character is the player writing, not the LLM.
    # Restricted to the room for the same reason replies are — a line from
    # someone the room has never heard of contradicts the roster the next
    # persona is given ("the only people here are…"), and an unexplained
    # voice in the transcript is how invented characters start.
    cast = _resolve_room_personas(req.chat_room, exclude_adopted=False)
    persona = next(
        (p for p in get_personas().personas if p.name == req.persona and p.name in cast),
        None,
    )
    if persona is None:
        yield f'data: {json.dumps({"type": "error", "message": f"{req.persona} is not in this room."})}\n\n'
        yield f'data: {json.dumps({"type": "complete"})}\n\n'
        return

    text = req.text.strip()
    if not text:
        yield f'data: {json.dumps({"type": "error", "message": "Nothing to say."})}\n\n'
        yield f'data: {json.dumps({"type": "complete"})}\n\n'
        return

    message_id = req.message_id or str(uuid.uuid4())
    yield f'data: {json.dumps({"type": "start", "persona": persona.name, "user_message_id": None, "message_id": message_id})}\n\n'
    yield f'data: {json.dumps({"type": "token", "persona": persona.name, "token": text})}\n\n'

    session.add_assistant_message(text, persona.name, message_id, room=turn_room)
    logger.info("Player spoke as %s in room '%s'", persona.name, turn_room)

    yield f'data: {json.dumps({"type": "done", "persona": persona.name, "text": text, "message_id": message_id})}\n\n'
    yield f'data: {json.dumps({"type": "complete"})}\n\n'


@router.post("/speak")
async def speak_as(req: SpeakAsRequest):
    """Say something as a persona. Returns the same SSE stream as a reply."""
    return StreamingResponse(
        _speak_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
