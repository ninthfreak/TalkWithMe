"""Writing memories without asking the model to call a tool.

Whether two characters have met is a fact the app can observe: they took
turns in the same room, so ``met.txt`` is written directly and needs no
cooperation from the model at all. What is worth *remembering* about
somebody is a judgement, and only the model can make it — which is the
one real asymmetry between the two halves of persona memory.

Needing the model, though, is not the same as needing a **tool**. A tool
call is one way to ask a model a question and, here, the worst one
available: it has to happen in the middle of a reply, it drags the
persona off the transcript prompt format onto the instruct format (see
``stream_chat_with_tools``, which builds a chat-completions payload and
never renders a transcript), and small local models emit malformed calls
often enough that the answer is unreliable even when everything else
lines up. On top of that it is gated behind ``allow_tool_calls``, which
is off by default and reads, in the editor, as a setting about MCP
servers.

So this asks the same question as a plain completion, once, after the
conversation has finished: *what did you learn about the people in it?*
That works in transcript mode, works with a model that cannot call tools
at all, and asks for the judgement at the moment it is easiest to make —
looking back at a whole scene rather than interrupting a sentence.

``add_memory`` is left in place for personas that do have tools: this is
another way to write the same file, not a replacement.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import AppSettings, Persona
from app.models import ChatMessage
from app.services import persona_store
from app.services.llm import PROSE_TIMEOUT, chat_completion

logger = logging.getLogger(__name__)


# How many new memories one reflection may file per persona.
#
# A cap rather than a limit the model is asked to respect: a model that
# answers a "what did you learn" question with twelve lines has usually
# started narrating the conversation back, and the first few are the ones
# worth having either way. It also bounds the damage a runaway does to a
# persona's byte budget, which purges oldest-first.
MAX_MEMORIES_PER_REFLECTION = 3

# Enough for the cap plus the model clearing its throat.
_REFLECTION_MAX_TOKENS = 220

# Not the room's sampling temperature. This is an extraction task — read
# the scene, report what was in it — and the room's temperature is set
# for performance, where surprising word choices are the point. Here a
# surprising word choice is an invented fact.
_REFLECTION_TEMPERATURE = 0.2

# The model's way of saying there was nothing worth keeping. Matched
# generously because it is the answer we most want it to feel free to
# give: a conversation that taught nobody anything is the common case,
# and a model that thinks it has to produce a line will invent one.
_NOTHING = re.compile(r"^\W*(nothing|none|n/?a|no memories|nothing to add)\b", re.I)


@dataclass
class Reflection:
    """What one persona took away from one conversation."""

    persona: str
    saved: List[str] = field(default_factory=list)
    # Lines the model produced that were not filed, and why. Kept because
    # "the model said nothing useful" and "the model named somebody who
    # was not there" are different problems with different fixes, and
    # neither is visible from the memories file.
    skipped: List[str] = field(default_factory=list)

    @property
    def learned_anything(self) -> bool:
        return bool(self.saved)


def render_conversation(history: Sequence[ChatMessage], user_label: str) -> str:
    """The finished conversation as a flat, fully tagged script.

    Every line carries a speaker, the human's included. Unlike the
    transcript sent for a *reply* this is not primed for anybody: nothing
    is being continued, so it ends where the conversation ended.
    """
    lines = []
    for message in history:
        text = (message.content or "").strip()
        if not text:
            continue
        speaker = user_label if message.role == "user" else (message.persona or "")
        lines.append(f"[{speaker}]: {text}" if speaker else text)
    return "\n".join(lines)


def build_reflection_prompt(
    persona: Persona, present: Sequence[str], conversation: str,
) -> List[Dict[str, str]]:
    """The one question this whole module asks.

    Deliberately short, and deliberately not written as a list of
    prohibitions. Two rules learned the hard way elsewhere in this app
    apply here too: naming a mood suggests it, and a prompt that is mostly
    "do not" produces a grudging answer. So it says what to write, gives
    one example of the shape, and makes "nothing" an ordinary answer
    rather than a failure — most conversations really do teach nobody
    anything, and a model that believes it must produce a line will
    invent one.

    The stored format is asked for directly ("[Name] text") because it is
    also the format the memory file uses, so there is no second
    representation to keep in sync.
    """
    others = ", ".join(present)
    return [
        {
            "role": "system",
            "content": (
                f"You are {persona.name}. {persona.system_prompt.strip()}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"This conversation has finished:\n\n{conversation}\n\n"
                f"You are {persona.name}. Write down anything you learned "
                f"about the other people in it that you would still want to "
                f"know the next time you meet them — what they want, what "
                f"they fear, something that happened to them, a strong "
                f"opinion, something they asked you to remember.\n\n"
                f"One line each, starting with whose it is in brackets:\n"
                f"[Tony] Tony has never been on a boat and does not intend "
                f"to start.\n\n"
                f"Write only about these people: {others}. Most "
                f"conversations leave you with one or two lines, and plenty "
                f"leave you with none — if nothing came up that you would "
                f"still want to know, answer with the single word: nothing"
            ),
        },
    ]


def parse_reflection(
    text: str, persona_name: str, present: Sequence[str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """The model's answer as (subject, memory) pairs, plus what was dropped.

    Filtering is the point, not tidiness. A memory filed about somebody
    who was not in the room is unreachable — injection only shows the
    people present — and one filed about the persona itself is the
    original trait-bleed bug in slow motion: a note saying what somebody
    else is like, stored unattributed, comes back in every later
    conversation as though it were true of the persona holding it.
    """
    allowed = {n.casefold(): n for n in present if n and n.strip()}
    allowed.pop(persona_name.casefold(), None)

    saved: List[Tuple[str, str]] = []
    skipped: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or _NOTHING.match(line):
            continue
        subject, memory = persona_store.split_memory_line(line)
        if not subject or not memory:
            skipped.append(line)
            continue
        canonical = allowed.get(subject.casefold())
        if canonical is None:
            # Either somebody who was not here, or the persona itself.
            skipped.append(line)
            continue
        saved.append((canonical, memory))
    return saved, skipped


async def reflect(
    persona: Persona,
    present: Sequence[str],
    history: Sequence[ChatMessage],
    settings: AppSettings,
    user_label: str,
) -> Reflection:
    """Ask one persona what it learned, and file the answer.

    Never raises. A reflection that fails costs the room some memories of
    one conversation, which is the same as the state before this existed;
    an exception escaping here would take out whatever triggered it —
    starting a new chat, or switching rooms.
    """
    result = Reflection(persona=persona.name)
    if persona.persona_dir is None or persona.memory_size <= 0:
        return result
    if not settings.general.enable_persona_memories:
        return result

    others = [n for n in present if n and n.casefold() != persona.name.casefold()]
    if not others:
        return result

    conversation = render_conversation(history, user_label)
    if not conversation.strip():
        return result

    try:
        answer = await chat_completion(
            build_reflection_prompt(persona, others, conversation),
            max_tokens=_REFLECTION_MAX_TOKENS,
            temperature=_REFLECTION_TEMPERATURE,
            timeout=PROSE_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 — see the docstring
        logger.warning("Reflection failed for persona '%s': %s", persona.name, exc)
        return result

    if not answer.strip():
        logger.debug("Reflection: persona '%s' returned nothing", persona.name)
        return result

    memories, skipped = parse_reflection(answer, persona.name, others)
    result.skipped = skipped

    for subject, memory in memories[:MAX_MEMORIES_PER_REFLECTION]:
        outcome = persona_store.append_memory(
            persona.persona_dir, subject, memory, persona.memory_size,
        )
        if outcome.startswith("Error:"):
            logger.debug(
                "Reflection: persona '%s' could not save a memory about %s: %s",
                persona.name, subject, outcome,
            )
            result.skipped.append(f"[{subject}] {memory}")
        elif outcome == "The memory was already saved.":
            # Expected and uninteresting: a reflection re-reads the same
            # conversation, so it re-derives what it derived last time.
            logger.debug(
                "Reflection: persona '%s' already knew '%s' about %s",
                persona.name, memory, subject,
            )
        else:
            result.saved.append(f"[{subject}] {memory}")

    if result.saved:
        logger.info(
            "Reflection: persona '%s' saved %d memory line(s)",
            persona.name, len(result.saved),
        )
    return result


def spoke_in(history: Sequence[ChatMessage]) -> List[str]:
    """Every persona with a turn in this conversation, in first-seen order.

    Only these reflect. A persona that was in the room but never spoke has
    nothing to look back on, and asking it would mean paying for a whole
    completion to be told "nothing" — with a real chance of being told
    something invented instead.
    """
    seen: List[str] = []
    for message in history:
        name = (message.persona or "").strip()
        if message.role == "assistant" and name and name not in seen:
            seen.append(name)
    return seen


async def reflect_on_conversation(
    history: Sequence[ChatMessage],
    personas: Sequence[Persona],
    settings: AppSettings,
    user_label: str,
    room: Optional[str] = None,
) -> List[Reflection]:
    """Run the pass for everybody who spoke. Never raises.

    Sequential rather than gathered: the backend serves one slot, so
    firing these in parallel would not finish sooner and would make the
    queue behind them unpredictable.
    """
    if not settings.general.enable_persona_memories:
        return []

    by_name = {p.name.casefold(): p for p in personas}
    speakers = spoke_in(history)
    if not speakers:
        return []

    # Everybody with a voice in the conversation, the human included under
    # whatever name they were playing: a persona learns about the person
    # it was talking to, not only about the other characters.
    cast = speakers + [user_label]

    results = []
    for name in speakers:
        persona = by_name.get(name.casefold())
        if persona is None:
            # Renamed or deleted since they spoke.
            continue
        present = [n for n in cast if n.casefold() != name.casefold()]
        results.append(
            await reflect(persona, present, history, settings, user_label)
        )

    saved = sum(len(r.saved) for r in results)
    logger.info(
        "Reflection on room '%s': %d persona(s) looked back, %d memory line(s) saved",
        room or "?", len(results), saved,
    )
    return results
