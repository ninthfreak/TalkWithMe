"""Rewriting a persona's memories as fewer lines that say the same thing.

Three separate lines —

    [Brad] Brad is 43 years old.
    [Brad] Brad is a banker.
    [Brad] Brad is a tall, handsome man.

— are one sentence's worth of information stored as three, and every one
of them is re-read into the prompt on every turn Brad is in the room.
Condensed:

    [Brad] Brad is a tall, handsome, 43 year old man. He is a banker.

That is the saving this exists for, and it needs a model: deciding that
those three belong together is exactly the judgement no similarity
threshold can make safely.

**It is run by hand, and it shows its work first.** The pass is lossy in
a way the rest of the memory system is not — everything else only ever
adds a line or drops an exact duplicate, while this rewrites sentences
the persona will act on. So nothing is written until the proposed result
has been looked at. What gets applied is exactly what was shown, not a
second generation that might differ.

A guess is never merged into a line of facts — folded into one, it
*becomes* a fact, which is the distinction the (assumed) marker exists to
keep. But a guess the facts have since settled is dropped outright: if a
persona once supposed Brad was about fifty and later learned he is 43,
carrying both wastes the space this pass exists to reclaim, and leaves
the persona holding a belief it has already been corrected on. This is
the one place that culling can happen, because it is the only place that
sees a person's guesses and facts side by side.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from app.config import Persona
from app.services import persona_store
from app.services.llm import PROSE_TIMEOUT, chat_completion

logger = logging.getLogger(__name__)

# Long enough to rewrite a full 8KB memory file; the answer is by
# definition shorter than the input.
_CONDENSE_MAX_TOKENS = 1200

# Rewriting, not composing. A surprising word choice here is an invented
# fact about somebody the persona will then act on.
_CONDENSE_TEMPERATURE = 0.1


@dataclass
class CondensePlan:
    """What a condense pass proposes, before anything is written."""

    persona: str
    before: List[str] = field(default_factory=list)
    after: List[str] = field(default_factory=list)
    # Byte-identical lines removed on the way in. Counted separately
    # because that part needs no model and is never in doubt.
    duplicates_removed: int = 0
    # Why the proposal is the same as the input, when it is.
    note: str = ""
    # People whose notes were kept exactly as written because the rewrite
    # would have lost a hard detail, with the detail named. Shown in the
    # preview: a silent refusal looks like the pass not working.
    protected: List[str] = field(default_factory=list)

    @property
    def before_bytes(self) -> int:
        return _bytes(self.before)

    @property
    def after_bytes(self) -> int:
        return _bytes(self.after)

    @property
    def saved_bytes(self) -> int:
        return max(0, self.before_bytes - self.after_bytes)

    @property
    def worth_applying(self) -> bool:
        return bool(self.after) and self.after != self.before


def _bytes(lines: Sequence[str]) -> int:
    return len("".join(line + "\n" for line in lines).encode("utf-8"))


# Words that are capitalised for grammar rather than because they name
# something. Anything else capitalised mid-sentence is taken to be a name,
# a place, a brand — a specific, and one the rewrite has to keep.
_NOT_A_NAME = {
    "i", "the", "a", "an", "he", "she", "they", "it", "we", "you", "his",
    "her", "their", "its", "our", "your", "this", "that", "these", "those",
    "and", "but", "or", "so", "if", "when", "while", "then", "there", "here",
    "not", "no", "yes", "also", "still", "now", "once", "because", "although",
    "however", "mr", "mrs", "ms", "dr",
}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")


def specifics(text: str) -> set:
    """The hard details in a memory: every number, and every proper noun.

    This is the part of "keep every fact" that can be checked without a
    model. A rewrite that turns "Brad is 43 years old" into "Brad is a
    middle-aged banker" has kept the gist and lost the fact, and no
    similarity score would notice — the sentence about Brad is still very
    like the sentence about Brad. But "43" is either in the result or it
    is not, and so is "Volvo", "Leeds", "Thursday".

    Numbers are any token containing a digit. Proper nouns are words
    capitalised somewhere other than the start of a sentence, less a short
    list of words English capitalises for grammar. Deliberately generous:
    a false alarm here keeps a line as it was, which costs nothing; a miss
    loses a fact, which is the thing this exists to prevent.

    Everything is casefolded, so "Banker" and "banker" agree.
    """
    found = set()
    at_sentence_start = True
    for match in _WORD_RE.finditer(text):
        token = match.group(0)
        lowered = token.casefold()
        if any(ch.isdigit() for ch in token):
            found.add(lowered)
        elif token[0].isupper() and not at_sentence_start and lowered not in _NOT_A_NAME:
            found.add(lowered)
        # Cheap sentence detection: this token was followed by . ! or ?
        # (by position, not by text.find(), which would return the first
        # occurrence of a repeated word rather than this one).
        end = match.end()
        at_sentence_start = end < len(text) and text[end] in ".!?"
    return found


def _words(text: str) -> set:
    return {t.casefold() for t in _WORD_RE.findall(text)}


def lost_details(before: Sequence[str], after: Sequence[str]) -> set:
    """Details present in *before* that no line of *after* still carries."""
    wanted = set()
    for line in before:
        wanted |= specifics(line)
    have = set()
    for line in after:
        have |= _words(line)
    return wanted - have


def build_condense_prompt(persona_name: str, memories: Sequence[str]) -> List[Dict[str, str]]:
    """Ask for the same information in fewer lines.

    Written as a rewriting job rather than a writing one, and the
    difference matters: a model asked to "summarise what Alex knows about
    Brad" produces a paragraph about Brad, complete with the things it
    assumes about bankers. Asked to merge lines that are about the same
    thing, it merges lines.

    The subject tags come back on every line because they are how the
    memory is filed. A rewrite that dropped them would file everything
    under nobody.
    """
    listing = "\n".join(memories)
    return [
        {
            "role": "system",
            "content": (
                "You tidy up notes. You never add anything that is not "
                "already in them, and the only thing you ever drop is a "
                "guess that a later note has settled."
            ),
        },
        {
            "role": "user",
            "content": (
                f"These are {persona_name}'s notes about people they know. "
                f"Each line starts with whose it is, in brackets.\n\n"
                f"{listing}\n\n"
                f"Rewrite them using fewer lines. Merge lines that are "
                f"about the same person and the same sort of thing, so "
                f"that three notes like\n"
                f"  [Brad] Brad is 43 years old.\n"
                f"  [Brad] Brad is a banker.\n"
                f"  [Brad] Brad is a tall, handsome man.\n"
                f"become one:\n"
                f"  [Brad] Brad is a tall, handsome, 43 year old man. He "
                f"is a banker.\n\n"
                f"Keep every fact. Every number, age, date, name and place "
                f"in the notes must still be in the rewrite, written the "
                f"same way — 43 stays 43, not \"in his forties\". Keep "
                f"each line about one person, and keep the name in "
                f"brackets at the start. Leave a note alone when it has "
                f"nothing to merge with.\n\n"
                f"A line marked (assumed) is something guessed rather "
                f"than known. Keep that marking, and never merge a guess "
                f"into a line of facts. But where a fact settles a guess, "
                f"the guess has served its purpose: drop it. If one line "
                f"says (assumed) Brad is about fifty and another says "
                f"Brad is 43 years old, keep only the second — he is 43, "
                f"and what somebody once supposed is no longer worth "
                f"carrying.\n\n"
                f"Answer with the rewritten notes and nothing else."
            ),
        },
    ]


def parse_condensed(
    answer: str, allowed_subjects: Sequence[str],
) -> List[persona_store.Memory]:
    """The model's answer as memory rows, or [] if it cannot be trusted.

    Every guard here answers the same question: could applying this lose
    or invent something? A subject that was not in the input means a
    person the persona never met; an unparseable line means a memory with
    no owner. Either one makes the whole answer suspect, and the safe
    response to a suspect rewrite is to keep what is already on disk.
    """
    allowed = {s.casefold() for s in allowed_subjects}
    rows: List[persona_store.Memory] = []
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        memory = persona_store.parse_memory_line(line)
        if not memory.subject or not memory.text:
            logger.info("Condense: untagged line in the answer, discarding the rewrite")
            return []
        if memory.subject.casefold() not in allowed:
            logger.info(
                "Condense: answer mentions '%s', who is not in the notes; "
                "discarding the rewrite", memory.subject,
            )
            return []
        rows.append(memory)
    return rows


async def plan(persona: Persona) -> CondensePlan:
    """Work out what condensing this persona's memories would produce.

    Writes nothing. The exact-duplicate trim *is* applied first, because
    it needs no judgement and cannot remove anything the file does not
    already contain twice — leaving it out would mean the model spent its
    attention on lines a plain function can delete for free.
    """
    result = CondensePlan(persona=persona.name)
    if persona.persona_dir is None:
        result.note = "This persona has no directory on disk."
        return result

    result.duplicates_removed = persona_store.dedupe_memories(persona.persona_dir)

    before = [
        line for line in persona_store.read_memories(persona.persona_dir).splitlines()
        if line.strip()
    ]
    result.before = before
    result.after = list(before)
    if len(before) < 2:
        result.note = "Nothing to merge."
        return result

    subjects = [persona_store.parse_memory_line(line).subject for line in before]
    subjects = [s for s in subjects if s]

    try:
        answer = await chat_completion(
            build_condense_prompt(persona.name, before),
            max_tokens=_CONDENSE_MAX_TOKENS,
            temperature=_CONDENSE_TEMPERATURE,
            timeout=PROSE_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 — a failed tidy-up is not a failure
        logger.warning("Condense failed for persona '%s': %s", persona.name, exc)
        result.note = "The model could not be reached."
        return result

    rows = parse_condensed(answer, subjects)
    if not rows:
        result.note = "The rewrite came back unusable, so nothing is proposed."
        return result

    condensed, protected = assemble(before, rows)
    result.protected = protected

    # A rewrite that is longer than what it replaces has not condensed
    # anything, whatever else it did — and is the shape a model takes when
    # it starts embroidering rather than merging.
    if _bytes(condensed) >= _bytes(before):
        result.note = "Nothing could be merged without making it longer."
        result.after = list(before)
        return result

    result.after = condensed
    return result


def assemble(
    before: Sequence[str], rows: Sequence[persona_store.Memory],
) -> "tuple[List[str], List[str]]":
    """Take the rewrite person by person, keeping anyone it would shortchange.

    Per subject rather than all-or-nothing: one bad merge about Brad is
    no reason to throw away a good one about Cora.

    The check is on **known** lines only. Every number and proper noun in
    what the persona was told about somebody has to survive in the
    rewrite about them, or their notes are kept exactly as they were and
    the preview says which detail would have gone. Guesses are left to
    the model — they are the one permitted loss, culled when a fact
    settles them, and the standard being enforced here is that a *fact*
    is never traded for a paraphrase of itself.

    Returns (lines, protected) where *protected* is a human-readable
    entry per person kept as written.
    """
    original: "dict[str, List[persona_store.Memory]]" = {}
    order: List[str] = []
    for line in before:
        memory = persona_store.parse_memory_line(line)
        key = memory.subject.casefold()
        if key not in original:
            original[key] = []
            order.append(key)
        original[key].append(memory)

    rewritten: "dict[str, List[persona_store.Memory]]" = {}
    for row in rows:
        rewritten.setdefault(row.subject.casefold(), []).append(row)

    lines: List[str] = []
    protected: List[str] = []
    for key in order:
        theirs = original[key]
        proposed = rewritten.get(key)
        if proposed is None:
            # The rewrite said nothing about them at all. Silence is not a
            # merge; keep what is there.
            lines.extend(m.stored() for m in theirs)
            continue

        known_before = [m.text for m in theirs if not m.assumed]
        known_after = [m.text for m in proposed if not m.assumed]
        lost = lost_details(known_before, known_after)
        if lost:
            name = theirs[0].subject
            lines.extend(m.stored() for m in theirs)
            protected.append(
                f"{name}: kept as written — the rewrite would have lost "
                + ", ".join(sorted(lost))
            )
            logger.info("Condense: keeping %s's notes; rewrite lost %s", name, sorted(lost))
            continue

        lines.extend(m.stored() for m in proposed)

    return lines, protected


def apply(persona: Persona, memories: Sequence[str]) -> int:
    """Write a reviewed set of memories over the persona's file.

    Takes the lines back from the caller rather than regenerating them:
    what gets written is exactly what was shown, and a second call to the
    model could not promise that.

    Returns the number of lines written. Raises OSError if the write
    fails, because this one was asked for explicitly and a silent no-op
    would leave the user believing the file had changed.
    """
    if persona.persona_dir is None:
        raise OSError(f"Persona '{persona.name}' has no directory on disk")

    lines = [line.strip() for line in memories if line and line.strip()]
    persona_store.write_memories(persona.persona_dir, lines)
    logger.info(
        "Condensed persona '%s' to %d memory line(s)", persona.name, len(lines),
    )
    return len(lines)
