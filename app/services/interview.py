"""Interview rooms — taking notes during a conversation, not after it.

The reflection pass asks "what did you learn about these people" once, at
the end, and files at most three lines. That is the right size for a
social memory and the wrong size for an interview, where the thing being
produced *is* the record and the conversation runs for hours.

So an interview room does three things differently, all of them hanging
off ``ChatRoom.interview``:

  * **the question changes.** Reflection asks for impressions — what
    somebody wants, what they fear. An interview asks for facts: names,
    dates, places, sequence, in the interviewee's own terms.
  * **the cadence changes.** Notes are taken every few exchanges rather
    than at the end, over a window that overlaps the last one, so a pass
    that fails costs nothing that the next one cannot recover.
  * **the store changes.** Notes go to a dossier (app/services/dossier.py)
    rather than memories.txt, because a life story outgrows a file that
    is injected whole.

What does **not** change is the format. A note is written in the same
``[Subject] #tags @when text`` line the dossier stores, parsed by the same
function, so there is one representation rather than two — the same rule
the reflection prompt follows for ``(assumed)``.

Nothing here runs for an ordinary room. Every entry point takes the room
and returns early when it is not an interview.
"""

import logging
import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from app.config import AppSettings, Persona
from app.models import ChatMessage
from app.services import dossier
from app.services.dossier import Note
from app.services.llm import PROSE_TIMEOUT, chat_completion
from app.services.reflection import render_conversation

logger = logging.getLogger(__name__)


# How many exchanges pass between one note-taking pass and the next.
#
# Three is a compromise between two costs that pull opposite ways. Every
# pass is a completion against a backend that serves one request at a
# time, so the user's next message queues behind it; but the reply window
# is six exchanges, so waiting much longer means the pass is writing down
# a conversation that has already fallen out of the prompt — which is the
# forgetting this whole feature exists to fix.
NOTES_EVERY_EXCHANGES = 3

# The window a pass reads, in exchanges. One more than the cadence, so
# consecutive passes overlap by an exchange: a pass that fails, or that
# was skipped because the previous one was still running, is covered by
# the next. The overlap is free — append_notes refuses a note it already
# holds, and the prompt is shown what has been written down.
NOTES_WINDOW_EXCHANGES = NOTES_EVERY_EXCHANGES + 1

# How many notes one pass may file. Larger than a reflection's three
# because an interview is the one conversation where the transcript is
# dense with things worth keeping — and because the dossier has room for
# them, which memories.txt does not.
MAX_NOTES_PER_PASS = 10

# Enough for the cap plus the open questions and the model clearing its
# throat: notes are one line each, questions are shorter.
_NOTES_MAX_TOKENS = 60 * MAX_NOTES_PER_PASS + 200

# Not the room's sampling temperature. This is an extraction task — read
# what was said, write it down — and a surprising word choice here is an
# invented fact. Same value and same reasoning as the reflection pass.
_NOTES_TEMPERATURE = 0.2

# The model's way of saying this stretch added nothing. Matched
# generously for the same reason the reflection pass matches it
# generously: it is the answer we most want it to feel free to give, and
# in an interview a model that thinks it must produce a line invents a
# fact about somebody's life.
_NOTHING_RE = re.compile(r"^\W*(nothing|none|n/?a|nothing new|no new)\b", re.I)

# How many open questions the interviewer carries. They are the agenda,
# they are injected every turn, and a list of twenty is not an agenda.
MAX_OPEN_QUESTIONS = 6


class NotesResult(NamedTuple):
    """What one note-taking pass did."""

    persona: str
    subject: str
    filed: List[str]
    questions: List[str]
    duplicates: int = 0
    merges: Sequence[Tuple[str, str]] = ()
    # Lines the model produced that were not filed. Counted because
    # "the model had nothing to add" and "the model wrote about the
    # wrong person" are different problems with different fixes, and
    # neither is visible from the dossier.
    dropped: Sequence[str] = ()


# ---------------------------------------------------------------------------
# When a pass runs
# ---------------------------------------------------------------------------

def exchanges_so_far(history: Sequence[ChatMessage]) -> int:
    """How many times the human has spoken in this conversation.

    The unit is the human's turn, not the message count: a room where
    three personas answer every message would otherwise take notes three
    times as often in the same conversation.
    """
    return sum(1 for m in history if m.role == "user")


def due_for_notes(
    history: Sequence[ChatMessage], every: int = NOTES_EVERY_EXCHANGES,
) -> bool:
    """Whether this turn is one of the ones that writes notes."""
    spoken = exchanges_so_far(history)
    return spoken > 0 and spoken % max(1, every) == 0


def recent_conversation(
    history: Sequence[ChatMessage],
    user_label: str,
    exchanges: int = NOTES_WINDOW_EXCHANGES,
) -> str:
    """The last few exchanges as a flat script, for the note pass to read.

    Anchored on the human's turns, like the reply window: an exchange is
    a message and the replies it drew, and cutting between them would
    hand the pass an answer with no question above it.
    """
    starts = [i for i, m in enumerate(history) if m.role == "user"]
    if len(starts) > exchanges:
        history = history[starts[-exchanges]:]
    return render_conversation(history, user_label)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

def build_notes_prompt(
    persona: Persona,
    subject: str,
    goal: str,
    conversation: str,
    index: Sequence[dossier.Topic] = (),
    open_questions: Sequence[Note] = (),
) -> List[Dict[str, str]]:
    """Ask the interviewer what to write down, and what it still wants.

    Four things go in, and each of them fixes something specific:

    *The goal*, so the pass knows what counts. "Write down what matters"
    means nothing on its own; "his working life, 1978 to retirement"
    decides whether the name of a foreman is worth a line.

    *The topics already in use*, which is the half of tag consistency no
    string comparison can do. ``dossier.snap_tag`` merges "schools" into
    "school" because they are spellings of one word; it will never merge
    "job" into "work", because they share no letters. Showing the model
    the index and asking it to reuse an entry is what keeps those
    together — the snapping underneath is the backstop, not the mechanism.

    *The open questions*, returned rather than deleted. Nothing in this
    app asks a model to remove a line: the pass is shown the questions
    that stand and writes back the ones that still stand, and a question
    the conversation answered disappears because it was left out.

    *The format*, which is the dossier's own line format, so what comes
    back is parsed by the same function that reads the file.

    Written as instructions to do things rather than to avoid them, and
    "nothing new" is an ordinary answer — a model that believes it must
    produce a line will invent one, and in an interview an invented line
    becomes a fact about somebody's life.
    """
    parts = [f"This is the conversation so far:\n\n{conversation}\n\n"]

    if goal.strip():
        parts.append(
            f"You are {persona.name}, and you are drawing {subject} out "
            f"about this: {goal.strip()}\n\n"
        )
    else:
        parts.append(
            f"You are {persona.name}, and you are drawing {subject} out "
            f"about their life.\n\n"
        )

    if index:
        topics = ", ".join(t.tag for t in index)
        parts.append(
            f"Your notes are filed under these topics: {topics}.\n"
            f"Use one of these where the note belongs under it, so "
            f"everything about one subject stays together.\n\n"
        )

    parts.append(
        f"Write down what {subject} told you in this stretch of the "
        f"conversation. One fact per line, in their terms, starting with "
        f"their name in brackets. Add one or two topics with #, and the "
        f"year or the period it is about with @ when they gave you one. "
        f"Begin a line with (assumed) where you worked it out rather than "
        f"being told, and with (sensitive) where they said they would "
        f"rather leave it:\n"
        f"[{subject}] #work #vickers @1978 Started at the yard straight "
        f"from school.\n"
        f"[{subject}] (assumed) #work Took the move to management as a "
        f"demotion.\n\n"
    )

    if open_questions:
        standing = "\n".join(f"  {n.text}" for n in open_questions)
        parts.append(
            f"You were also waiting to ask these:\n{standing}\n\n"
            f"Write back the ones you are still waiting on, and add any "
            f"this stretch opened up. "
        )
    else:
        parts.append("Then add what you want to ask next. ")

    parts.append(
        f"Each one on its own line, beginning (open):\n"
        f"[{subject}] (open) #work What did he do between 1986 and 1989?\n\n"
        f"If this stretch told you nothing you had not already written "
        f"down, answer with the single word: nothing"
    )

    return [
        {"role": "system", "content": f"You are {persona.name}. {persona.system_prompt.strip()}"},
        {"role": "user", "content": "".join(parts)},
    ]


def parse_notes(text: str, subject: str) -> Tuple[List[Note], List[Note], List[str]]:
    """The model's answer as (facts, questions, dropped).

    Parsed with the dossier's own line parser, so a line the model writes
    is a line the file could hold. Two things are dropped rather than
    filed, and both are counted so "the model said nothing useful" stays
    distinguishable from "the model wrote about the wrong person":

      * a line about somebody who is not the interviewee. A dossier is
        one person's; a fact about their mother is a fact about them,
        filed under their name.
      * a line with no text left after the markers.
    """
    facts: List[Note] = []
    questions: List[Note] = []
    dropped: List[str] = []

    for raw in text.splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if not line:
            continue
        if _NOTHING_RE.match(line):
            continue

        note = dossier.parse_note_line(line)
        if not note.text:
            dropped.append(line)
            continue
        if note.subject and note.subject.casefold() != subject.casefold():
            dropped.append(line)
            continue

        settled = note._replace(subject=subject)
        if settled.open_question:
            questions.append(settled)
        else:
            facts.append(settled)

    return facts, questions, dropped


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

async def take_notes(
    persona: Persona,
    subject: str,
    goal: str,
    history: Sequence[ChatMessage],
    settings: AppSettings,
    user_label: str,
    *,
    whole_conversation: bool = False,
) -> Optional[NotesResult]:
    """Read the last stretch of the interview and write it down.

    Never raises. A pass that fails costs one stretch of notes, and the
    window overlaps so the next pass covers it; an exception escaping
    here would take out the reply that triggered it.

    *whole_conversation* is for the pass that runs when the interview
    ends, where there is no next pass to recover anything.
    """
    if persona.persona_dir is None:
        return None
    if not settings.general.enable_persona_memories:
        return None

    conversation = (
        render_conversation(history, user_label) if whole_conversation
        else recent_conversation(history, user_label)
    )
    if not conversation.strip():
        return None

    dossier.reindex(persona.persona_dir, subject)
    held = dossier.read_notes(persona.persona_dir, subject)
    standing = [n for n in held if n.open_question]

    try:
        answer = await chat_completion(
            build_notes_prompt(
                persona, subject, goal, conversation,
                index=dossier.topics(held), open_questions=standing,
            ),
            max_tokens=_NOTES_MAX_TOKENS,
            temperature=_NOTES_TEMPERATURE,
            timeout=PROSE_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 — see the docstring
        logger.warning("Interview notes failed for '%s': %s", persona.name, exc)
        return None

    if not answer.strip():
        # chat_completion swallows a backend error and returns "", so this
        # is either "nothing new" or a failed call. Said at INFO rather
        # than DEBUG because in an interview the difference matters and
        # the log is the only place it shows.
        logger.info("Interview notes: '%s' returned nothing for %s", persona.name, subject)
        return None

    facts, questions, dropped = parse_notes(answer, subject)
    if dropped:
        # At INFO, not DEBUG. A pass whose lines were all about somebody
        # else writes nothing and looks exactly like a quiet stretch of
        # conversation — and in an interview, notes that never arrive
        # are the failure the whole feature exists to prevent.
        logger.info(
            "Interview notes: '%s' wrote %d line(s) that were not about %s "
            "and were not filed; first was %r",
            persona.name, len(dropped), subject, dropped[0],
        )

    try:
        result = dossier.append_notes(
            persona.persona_dir, subject, facts[:MAX_NOTES_PER_PASS],
        )
        if questions:
            dossier.replace_open_questions(
                persona.persona_dir, subject, questions[:MAX_OPEN_QUESTIONS],
            )
    except OSError as exc:
        logger.warning("Interview notes: could not write %s's dossier on %s: %s",
                       persona.name, subject, exc)
        return None

    if result.filed or questions:
        logger.info(
            "Interview notes: '%s' wrote %d line(s) about %s and is waiting on %d question(s)",
            persona.name, len(result.filed), subject, len(questions),
        )
    return NotesResult(
        persona=persona.name,
        subject=subject,
        filed=[n.stored() for n in result.filed],
        questions=[n.text for n in questions[:MAX_OPEN_QUESTIONS]],
        duplicates=len(result.duplicates),
        merges=result.merges,
        dropped=dropped,
    )


# ---------------------------------------------------------------------------
# What reaches the reply prompt
# ---------------------------------------------------------------------------

def preamble_lines(persona: Persona, subject: str, goal: str) -> List[str]:
    """The interview's lines in the room preamble.

    Two sentences at most, stated as facts about what is happening. The
    preamble is held to about 200 words for a measured reason — at 330 it
    was 96% of what the model read and the character could not outvote it
    — so an interview gets one line for what it is and one for the goal,
    and the notes block carries everything else.
    """
    # Capitalised for the sentence start: with nobody adopted the
    # subject is the literal string "the user", and the preamble's
    # addressed-to line already has to do the same.
    named = subject[0].upper() + subject[1:] if subject else "They"
    lines = [
        f"- This is an interview. {named} is talking about their life, "
        f"and {persona.name} is drawing them out and writing down what "
        f"they say.",
    ]
    if goal.strip():
        lines.append(f"- What you are here for: {goal.strip()}")
    return lines


def notes_block(persona: Persona, subject: str, query: str) -> str:
    """What this interviewer has on the person in front of them.

    Two levels, which is the whole reason a dossier can outgrow a prompt:
    the index of everything, always, and the contents of the topics that
    match what is being discussed. See app/services/dossier.py.
    """
    if persona.persona_dir is None:
        return ""
    dossier.reindex(persona.persona_dir, subject)
    notes = dossier.read_notes(persona.persona_dir, subject)
    if not notes:
        return ""
    return dossier.render_block(dossier.select(notes, query), subject)
