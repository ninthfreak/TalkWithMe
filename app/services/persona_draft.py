"""Draft a persona with the LLM, and say what actually made it distinct.

The problem this exists for: hand-written personas come out sounding the
same. The usual cause is that they are written as *topics* ("philosophy,
emotions, art") and *adjective piles* ("thoughtful, curious, friendly"),
neither of which changes what a model does with a turn. Two personas that
differ only in subject matter produce the same sentences about different
nouns.

What does change behaviour is listed in ``LEVERS`` below. The drafting
prompt is built around it, and the draft comes back with notes saying
which levers the brief supplied and which had to be invented — so the
guidance transfers to personas written by hand afterwards.

Output is parsed from labelled blocks rather than JSON. Local models
follow "NAME: ..." far more reliably than they emit valid JSON, and a
half-written brace costs the whole draft where a missing block costs one
field.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import LengthBias, Persona

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# What actually differentiates a persona
# ---------------------------------------------------------------------------
#
# Ordered roughly by how much they change a reply. The frontend shows this
# list next to the brief box, so it doubles as the teaching material.

@dataclass(frozen=True)
class Lever:
    key: str
    title: str
    hint: str


LEVERS: List[Lever] = [
    Lever(
        "stance",
        "What they do with a turn",
        "Assert, ask, deflect, correct, tell an anecdote, negotiate. This is "
        "the single biggest differentiator and almost nobody writes it down.",
    ),
    Lever(
        "register",
        "How the sentences are built",
        "Length, vocabulary, contractions, jargon, profanity, whether they "
        "finish their thoughts. Two characters with identical opinions read "
        "as different people if the prose is shaped differently.",
    ),
    Lever(
        "signature",
        "A verbal tic",
        "One repeatable construction — how they open, a word they overuse, a "
        "comparison they keep reaching for. Recognisable within a line.",
    ),
    Lever(
        "agenda",
        "What they want",
        "What they are pushing for, defending, or selling in the conversation. "
        "A character with a stake argues; a character without one comments.",
    ),
    Lever(
        "negative",
        "What they never do",
        "Refusals and avoidances. Negative space differentiates harder than "
        "anything positive, because it cuts off the generic reply.",
    ),
    Lever(
        "relationships",
        "What they think of the others",
        "A named opinion about another persona in the room gives the model "
        "something to play that a solo description cannot.",
    ),
    Lever(
        "flaw",
        "Where they are wrong",
        "A blind spot, an overconfidence, an out-of-date belief. Perfect "
        "characters converge on the assistant voice.",
    ),
    Lever(
        "mood",
        "The mood they arrive in",
        "Impatient, delighted, wary, bored. The default emotional register "
        "before anything is said to them.",
    ),
]

# Things people reliably write that do NOT differentiate, named explicitly
# so the model does not produce them and the user learns to stop.
ANTI_PATTERNS = [
    "topic lists (\"philosophy, art, emotions\") — those route a question, "
    "they do not change a voice",
    "adjective piles (\"thoughtful, curious, friendly\") — every model reads "
    "all of them as \"helpful assistant\"",
    "\"You are a helpful X\" framing — it collapses straight back to the "
    "default assistant register",
    "biography with no behavioural consequence — a backstory only matters "
    "if it changes how they answer",
]


# ---------------------------------------------------------------------------
# Field limits, mirrored from the create/update form
# ---------------------------------------------------------------------------

MAX_NAME = 25
MAX_DESCRIPTION = 30
MAX_ROUTER_HINTS = 256
MAX_SYSTEM_PROMPT = 8192

# Long enough to carry every lever, short enough to stay inside the room's
# length budget. Wildly over-long prompts drown the room preamble instead,
# which is the same failure pointing the other way.
TARGET_PROMPT_WORDS = 120


@dataclass
class PersonaDraft:
    """A drafted persona plus the reasoning that produced it."""

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    router_hints: str = ""
    length_bias: LengthBias = LengthBias.MATCH
    avatar_color: str = "#4A90D9"
    # One line per lever, saying what was used and where it came from.
    notes: List[str] = field(default_factory=list)
    # How this persona is meant to differ from the existing cast.
    contrast: str = ""

    def is_usable(self) -> bool:
        """Enough to populate the form: a name and something to say."""
        return bool(self.name.strip() and self.system_prompt.strip())


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _cast_summary(existing: List[Persona]) -> str:
    """The current cast, in enough detail to be written *against*.

    The whole system prompt of each is too much (a room of six would bury
    the instructions), but a name and a description alone is too little to
    contrast with — the description is capped at 30 characters. The opening
    of each prompt is the useful middle.
    """
    if not existing:
        return "There are no other personas yet; this is the first."
    lines = []
    for p in existing:
        opening = " ".join(p.system_prompt.split())[:220]
        lines.append(f"- {p.name} ({p.description or 'no description'}): {opening}")
    return "\n".join(lines)


def build_draft_prompt(brief: str, existing: List[Persona]) -> List[dict]:
    """The messages that ask the LLM for a persona.

    Written as an instruction to a *casting director*, not to an assistant
    filling in a form: the framing matters, because "fill in these fields"
    produces field-shaped filler and "make this person different from those
    people" produces a character.
    """
    lever_block = "\n".join(f"- {lv.title}: {lv.hint}" for lv in LEVERS)
    anti_block = "\n".join(f"- Avoid {a}" for a in ANTI_PATTERNS)

    system = f"""You write characters for a group chat where several of them talk to one human and to each other. You are given a short brief and the cast that already exists. Your job is to produce one new character who is unmistakably NOT one of the others.

The cast so far:
{_cast_summary(existing)}

What actually makes a character behave differently (use as many as the brief allows, and invent the rest):
{lever_block}

What does not work, and must not appear in your output:
{anti_block}

Write the system prompt in the second person, addressed to the character ("You interrupt when..."). Around {TARGET_PROMPT_WORDS} words: long enough to carry a voice, short enough not to drown the room's own instructions. Every sentence must constrain behaviour — if a sentence could be deleted without changing a single reply, delete it yourself.

Reply in exactly this format, with these labels, and nothing else:

NAME: <up to {MAX_NAME} characters, no slashes>
DESCRIPTION: <up to {MAX_DESCRIPTION} characters, shown in the room roster>
ROUTER_HINTS: <comma-separated topics this character should be picked for>
LENGTH_BIAS: <one of: much_shorter, shorter, match, longer, much_longer>
AVATAR_COLOR: <a hex colour like #4A90D9>
CONTRAST: <one sentence on how this character differs from the cast above>
NOTES:
- <one line per lever you used, naming the lever and what you did with it; say which ones the brief already supplied and which you invented>
SYSTEM_PROMPT:
<the prompt itself, second person, no name prefix, no quotes>"""

    user = f"The brief: {brief.strip()}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LABELS = (
    "NAME", "DESCRIPTION", "ROUTER_HINTS", "LENGTH_BIAS",
    "AVATAR_COLOR", "CONTRAST", "NOTES", "SYSTEM_PROMPT",
)
_LABEL_RE = re.compile(rf"^\s*({'|'.join(_LABELS)})\s*:\s*(.*)$", re.IGNORECASE)
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _split_blocks(text: str) -> Dict[str, str]:
    """Labelled blocks to a dict. Unknown lines join the block above them.

    Deliberately forgiving: a model that wraps the reply in a code fence,
    adds a preamble, or drops one label still yields everything else.
    """
    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() in ("```", "```markdown", "```text"):
            continue
        match = _LABEL_RE.match(line)
        if match:
            current = match.group(1).upper()
            blocks[current] = [match.group(2)] if match.group(2).strip() else []
            continue
        if current is not None:
            blocks[current].append(line)
    return {k: "\n".join(v).strip() for k, v in blocks.items()}


def _clean_one_line(value: str, limit: int) -> str:
    value = " ".join(value.split())
    # Models like to quote a value back; the quotes are not part of it.
    value = value.strip('"“”\'')
    return value[:limit].strip()


def _parse_notes(block: str) -> List[str]:
    notes = []
    for line in block.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if line:
            notes.append(line)
    return notes


def parse_draft(text: str) -> PersonaDraft:
    """Turn a model reply into a draft, salvaging whatever is well formed."""
    blocks = _split_blocks(text)
    draft = PersonaDraft()

    draft.name = _clean_one_line(blocks.get("NAME", ""), MAX_NAME)
    # A slash makes the persona unreachable on /api/personas/{name}/...
    draft.name = draft.name.replace("/", " ").replace("\\", " ").strip()
    draft.description = _clean_one_line(blocks.get("DESCRIPTION", ""), MAX_DESCRIPTION)
    draft.router_hints = _clean_one_line(blocks.get("ROUTER_HINTS", ""), MAX_ROUTER_HINTS)
    draft.contrast = _clean_one_line(blocks.get("CONTRAST", ""), 400)
    draft.notes = _parse_notes(blocks.get("NOTES", ""))

    prompt = blocks.get("SYSTEM_PROMPT", "").strip().strip("`").strip()
    draft.system_prompt = prompt[:MAX_SYSTEM_PROMPT]

    raw_bias = _clean_one_line(blocks.get("LENGTH_BIAS", ""), 32).lower()
    try:
        draft.length_bias = LengthBias(raw_bias)
    except ValueError:
        if raw_bias:
            logger.info("Draft returned an unusable length_bias %r; using 'match'", raw_bias)

    colour = _clean_one_line(blocks.get("AVATAR_COLOR", ""), 7)
    if _HEX_RE.match(colour):
        draft.avatar_color = colour.upper()
    elif colour:
        logger.info("Draft returned an unusable avatar_color %r; keeping the default", colour)

    return draft


# ---------------------------------------------------------------------------
# Post-draft critique
# ---------------------------------------------------------------------------
#
# Checked locally rather than asked of the model: these are the failures
# the model itself is most likely to commit, so it is the wrong judge.

_ANTI_PATTERN_WORDS = (
    "helpful", "friendly", "thoughtful", "curious", "knowledgeable",
    "insightful", "engaging", "assistant", "ai companion",
)


def critique(draft: PersonaDraft) -> List[str]:
    """Warnings about a draft, in the user's terms rather than the model's."""
    warnings: List[str] = []
    prompt = draft.system_prompt
    words = prompt.split()

    if len(words) < 40:
        warnings.append(
            f"The prompt is only {len(words)} words. Short prompts lose to the "
            "room's shared instructions — aim for something nearer "
            f"{TARGET_PROMPT_WORDS}."
        )
    found = sorted({w for w in _ANTI_PATTERN_WORDS if w in prompt.lower()})
    if found:
        warnings.append(
            "Contains generic assistant vocabulary (" + ", ".join(found) + "). "
            "Those words pull every model back towards its default voice."
        )
    if "never" not in prompt.lower() and "refus" not in prompt.lower():
        warnings.append(
            "Nothing here says what this character will not do. Negative "
            "space differentiates harder than anything positive."
        )
    if not re.search(r"\byou\b", prompt.lower()):
        warnings.append(
            "The prompt is not addressed to the character in the second "
            "person, which is the form the rest of the app assumes."
        )
    return warnings
