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

Diversity comes from the levers, not from the cast. An earlier version
sent every existing persona to the model and asked for someone unlike
them. That was the wrong mechanism: it made the prompt grow with the cast
(slow past a handful of personas), and it defined a new character by what
the others were rather than by what it was. A character built from strong
levers is distinct on its own.

Output is parsed from labelled blocks rather than JSON. Local models
follow "NAME: ..." far more reliably than they emit valid JSON, and a
half-written brace costs the whole draft where a missing block costs one
field.
"""

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from app.config import LengthBias

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
    # For the dialog: explains to a person why this matters.
    hint: str
    # For the drafting prompt: the same thing said as an instruction.
    # A model does not need to be told why a lever works, and every word
    # spent explaining is a word competing with the specification above
    # it. Defaults to the hint so a new lever cannot silently lose text.
    prompt_hint: str = ""
    # The dial or detail field that now sets this directly, if any. Such a
    # lever is still shown in the dialog as teaching material, but it is
    # kept OUT of the drafting prompt: repeating "how the sentences are
    # built" as free advice next to an explicit Register instruction
    # invites the model to overrule the setting it was just given.
    superseded_by: str = ""


LEVERS: List[Lever] = [
    Lever(
        "stance",
        "What they do with a turn",
        "Assert, ask, deflect, correct, tell an anecdote, negotiate. This is "
        "the single biggest differentiator and almost nobody writes it down.",
        superseded_by="stance",
    ),
    Lever(
        "register",
        "How the sentences are built",
        "Length, vocabulary, contractions, jargon, profanity, whether they "
        "finish their thoughts. Two characters with identical opinions read "
        "as different people if the prose is shaped differently.",
        superseded_by="vocabulary",
    ),
    Lever(
        "signature",
        "A verbal tic",
        "One repeatable construction — how they open, a word they overuse, a "
        "comparison they keep reaching for. Recognisable within a line.",
        superseded_by="tic",
    ),
    Lever(
        "agenda",
        "What they want",
        "What they are pushing for, defending, or selling in the conversation. "
        "A character with a stake acts; a character without one comments.",
        superseded_by="wants",
    ),
    Lever(
        "negative",
        "What they never do",
        "Refusals and avoidances. Negative space differentiates harder than "
        "anything positive, because it cuts off the generic reply.",
        superseded_by="never",
    ),
    Lever(
        "relationships",
        "What they think of the others",
        "A named opinion — fond, wary, exasperated — about another persona in "
        "the room gives the model something to play that a solo "
        "description cannot.",
        prompt_hint="a named opinion about someone else here — fond, wary, exasperated",
    ),
    Lever(
        "flaw",
        "Where they are wrong",
        "A blind spot, an overconfidence, an out-of-date belief. Perfect "
        "characters converge on the assistant voice.",
        superseded_by="wrong",
    ),
    Lever(
        "mood",
        "The mood they arrive in",
        "Delighted, impatient, content, wary. The default emotional register "
        "before anything is said to them.",
        prompt_hint="delighted, impatient, content, wary — before anyone speaks to them",
    ),
]

# Things people reliably write that do NOT differentiate, named explicitly
# so the model does not produce them and the user learns to stop.
ANTI_PATTERNS = [
    "topic lists (\"philosophy, art, emotions\") — those route a question, "
    "they do not change a voice",
    "a pile of adjectives as the whole character (\"thoughtful, curious, "
    "friendly\") — any one of those can be true of someone; the list is not "
    "a person",
    "\"You are a helpful X\" framing — it collapses straight back to the "
    "default assistant register",
    "biography for its own sake — a past is worth a line when it shows in "
    "how they answer, and worth none when it does not",
]


# ---------------------------------------------------------------------------
# The specification: dials and details
# ---------------------------------------------------------------------------
#
# A single free-text brief made every word a *global* dial. "A crude
# harbourmaster" gave the model one adjective and nothing to attach it to,
# so "crude" coloured word choice, disposition and cooperativeness at once
# and the result was a belligerent character who was bad at conversation.
#
# Each axis is now its own field with its own fixed vocabulary, and each
# option carries the instruction the prompt actually uses — "coarse" alone
# is as vague as the brief was. A dropdown cannot be read as a global
# intensity dial, which is the entire trick.
#
# Every dial has an "" option meaning *unspecified*: it is left out of the
# prompt and the model invents it. Nothing here is compulsory taxonomy.

UNSPECIFIED = ""


@dataclass(frozen=True)
class DialOption:
    value: str
    label: str
    # What the prompt says when this option is chosen. Carries the whole
    # weight: the label is for the human, this is for the model.
    instruction: str


@dataclass(frozen=True)
class Dial:
    key: str
    title: str
    hint: str
    group: str
    options: List[DialOption]
    default: str

    def option(self, value: str) -> Optional[DialOption]:
        return next((o for o in self.options if o.value == value), None)


SPEECH = "How they talk"
ENGAGEMENT = "What they do with a turn"


# Four, not seven, and every one of them silent until it is set. The
# reasoning, since it decides what belongs here and what does not:
#
#   * Instructions compete. Seven simultaneous style constraints get
#     averaged into a generically "stylised" voice; one constraint gets
#     applied. A dial sitting at a neutral default ("ordinary sentence
#     lengths, varied") says nothing and dilutes everything else, and
#     seven of those buried a twelve-word brief under ninety-four words
#     the user never chose.
#   * Models caricature disposition labels. "Blunt" does not produce
#     blunt, it produces rude, because in the training distribution blunt
#     characters are rude. Two attempts to hold that line with prose
#     ("WORD CHOICE ONLY: this does not make them hostile") failed, and
#     the second one made things worse by putting *hostile* in front of
#     the model.
#   * So: a dial may cover the MECHANICS OF SPEECH, which a brief
#     expresses badly. Disposition — politeness, temper, certainty, warmth
#     — belongs in "Who they are", where the user's own words carry it and
#     nothing has to be caricatured to be understood.
#
# Dropped on that rule: Register (its lexical half lives in Vocabulary
# now; politeness is relational, which is why Warmth went the same way),
# Temperament (existed to stop Register bleeding into temper, so it went
# with it), and Certainty.

DIALS: List[Dial] = [
    Dial(
        "vocabulary", "Vocabulary",
        "Which words they reach for. The one thing a brief says badly.",
        SPEECH,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("blunt_everyday", "Blunt everyday",
                       "short, common words; contractions; no jargon and no abstractions"),
            DialOption("trade", "Trade talk",
                       "the working vocabulary of their job — practical nouns, tools, procedures"),
            DialOption("plain_literate", "Plain literate",
                       "clear and unshowy; complete sentences, ordinary words, no flourish"),
            DialOption("bookish", "Bookish",
                       "reaches for the precise word; the occasional allusion or uncommon term"),
            DialOption("ornate", "Ornate",
                       "long clauses, metaphor, rhetorical shape"),
            DialOption("technical", "Technical",
                       "domain jargon used precisely, accessibility second"),
            # Profanity is a vocabulary, and only a vocabulary. It used to
            # live on a politeness dial next to "courteous" and "blunt",
            # where the model read the whole axis as how they treat people.
            DialOption("crude", "Crude",
                       "crude turns of phrase and mild profanity, with everyone alike"),
            DialOption("foul_mouthed", "Foul-mouthed",
                       "swears constantly and without thinking about it, at people they "
                       "like as much as anyone"),
        ],
        UNSPECIFIED,
    ),
    Dial(
        "sentences", "Sentence shape",
        "How the prose is built, whatever the words in it are.",
        SPEECH,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("clipped", "Clipped",
                       "fragments, often no verb; stops as soon as the point is made"),
            DialOption("short", "Short",
                       "short complete sentences, one idea in each"),
            DialOption("flowing", "Flowing",
                       "longer sentences whose subordinate clauses connect ideas"),
            DialOption("rambling", "Rambling",
                       "runs on and digresses; arrives at the point late, or not at all"),
        ],
        UNSPECIFIED,
    ),
    Dial(
        "abstraction", "Abstraction",
        "Whether they argue from cases or from principles.",
        SPEECH,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("concrete", "Concrete",
                       "talks about specific things, people and events; examples rather "
                       "than principles"),
            DialOption("theoretical", "Theoretical",
                       "reaches for principles, systems and generalisations"),
        ],
        UNSPECIFIED,
    ),
    Dial(
        "stance", "Stance",
        "What they do with a turn — mechanics, not mood, and the biggest "
        "single differentiator.",
        ENGAGEMENT,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("asks", "Asks",
                       "answers with a question more often than with a statement"),
            DialOption("responds", "Responds",
                       "listens, then addresses what was actually said"),
            DialOption("asserts", "Asserts",
                       "leads with their own position whether or not it was asked for"),
            DialOption("corrects", "Corrects",
                       "picks up errors, including small ones"),
            DialOption("tells", "Tells a story",
                       "answers by way of something that happened to them or to someone "
                       "they know"),
        ],
        UNSPECIFIED,
    ),
]

DIALS_BY_KEY: Dict[str, Dial] = {d.key: d for d in DIALS}

# Ordered groups for the form. Built here rather than grouped in the
# template so that the UI cannot drift from the prompt: adding a dial to
# DIALS puts it on screen and in the prompt in the same edit.
DIAL_GROUPS: List[Tuple[str, List[Dial]]] = [
    (group, [d for d in DIALS if d.group == group])
    for group in (SPEECH, ENGAGEMENT)
    if any(d.group == group for d in DIALS)
]


@dataclass(frozen=True)
class DetailField:
    key: str
    label: str
    placeholder: str
    hint: str


# Free text, all optional. Blank means "invent it" — and the draft's notes
# say which were given and which were invented, so the difference between
# a thin brief and a full one is visible rather than mysterious.
# The placeholders deliberately describe a *warm* character while the brief
# box above them describes a prickly one. Every example in this dialog used
# to be the same suspicious harbourmaster, and a page of examples in one
# register is itself an instruction — to the model when it reaches the
# prompt, and to the person writing the brief.
DETAILS: List[DetailField] = [
    DetailField("wants", "What they want",
                "everyone fed, whether or not they can pay today",
                "What they are after in a conversation. A character with a stake acts; "
                "one without a stake comments."),
    DetailField("never", "What they never do",
                "never lets anyone leave empty-handed",
                "Refusals and avoidances. The strongest single differentiator, because "
                "it cuts off the generic reply."),
    DetailField("wrong", "Where they are wrong",
                "certain the new place on the corner will not last the winter",
                "A blind spot or an out-of-date belief. Characters with no flaws "
                "converge on the assistant voice."),
    DetailField("tic", "A verbal tic",
                "asks after your mother before she answers anything",
                "One repeatable thing you would recognise in a single line."),
    DetailField("background", "Background",
                "took the bakery over from her mother; knows everyone's order",
                "Occupation or history — but only the parts that change how they answer."),
]

DETAILS_BY_KEY: Dict[str, DetailField] = {d.key: d for d in DETAILS}

MAX_DETAIL_CHARS = 400


@dataclass
class PersonaSpec:
    """Everything the user filled in: the brief, the dials, the details."""

    brief: str = ""
    dials: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_request(cls, brief: str, dials: Dict[str, str], details: Dict[str, str]):
        """Build a spec, dropping anything that is not a known key or value.

        Unknown dial values fall back to *unspecified* rather than being
        passed through: an option this build does not know cannot have an
        instruction, so sending the bare word would reintroduce exactly
        the vagueness the dials exist to remove.
        """
        clean_dials: Dict[str, str] = {}
        for key, value in (dials or {}).items():
            dial = DIALS_BY_KEY.get(key)
            if dial is None:
                continue
            clean_dials[key] = value if dial.option(value) else UNSPECIFIED
        clean_details = {
            k: " ".join(str(v).split())[:MAX_DETAIL_CHARS]
            for k, v in (details or {}).items()
            if k in DETAILS_BY_KEY and str(v).strip()
        }
        return cls(brief=brief, dials=clean_dials, details=clean_details)

    def instruction_for(self, key: str) -> Optional[str]:
        """The prompt line for one dial, or None when unspecified."""
        dial = DIALS_BY_KEY[key]
        option = dial.option(self.dials.get(key, dial.default))
        if option is None or not option.instruction:
            return None
        return f"- {dial.title}: {option.label} — {option.instruction}"


# ---------------------------------------------------------------------------
# Field limits, mirrored from the create/update form
# ---------------------------------------------------------------------------

MAX_NAME = 25
MAX_DESCRIPTION = 30
MAX_ROUTER_HINTS = 256
MAX_SYSTEM_PROMPT = 8192

# A sketch, not a specification. 120 words of "you do X, you never Y"
# produces a character who does X and never does Y — reliably, every
# turn, whatever is actually said to them, which reads as heavy-handed
# and static because it is. An actor improvises from a few lines; a
# decision table only gets executed.
#
# The old number existed for an arithmetic reason that has since gone
# away: a persona had to be bulky to compete with a 330-word room
# preamble. The preamble is 160 words now, so sixty words of character is
# already more than a quarter of what the model reads.
TARGET_PROMPT_WORDS = 60


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

    def is_usable(self) -> bool:
        """Enough to populate the form: a name and something to say."""
        return bool(self.name.strip() and self.system_prompt.strip())


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Language shared by drafting and refining. Both are the same job under
# different starting conditions, and the two failures below happen in both:
# a free-text instruction read as a global intensity dial, and a prompt
# written as an essay about the character rather than instructions to an
# actor. Kept as constants so a fix to one path cannot miss the other.

WRITING_RULES = (
    "Second person, addressed to the character (\"You keep the good glue for jobs "
    "nobody is paying for\").\n\n"
    "Write who they are and what they care about — not a list of rules to follow. "
    "An actor improvises from a sketch; a decision table only gets executed, the "
    "same way every time, whatever is actually being said. Leave gaps for them to "
    "fill, and trust that what they would do in a situation you have not thought "
    "of follows from who they are.\n\n"
    "Plain language: a note to an actor, not an essay, and not a demonstration of "
    "the character's own vocabulary — an ornate character still gets a "
    "plainly-written note.\n\n"
    # Aimed at the model's essayist house style. Kept to vocabulary: an
    # earlier version asked for "concrete examples" too, and got prompts
    # that were nothing but examples.
    "Ordinary words rather than an essayist's, unless the specification says "
    "otherwise.\n\n"
    "Distinct is not the same as difficult. Warm, kind, delighted, loyal and "
    "generous are specific ways to be; write someone unpleasant only if you were "
    "asked for one."
)


def _anti_pattern_block() -> str:
    return "\n".join(f"- Avoid {a}" for a in ANTI_PATTERNS)


def build_draft_prompt(spec: PersonaSpec) -> List[dict]:
    """The messages that ask the LLM for a persona.

    Written as an instruction to a *casting director*, not to an assistant
    filling in a form: the framing matters, because "fill in these fields"
    produces field-shaped filler and "write this person so they could not
    be mistaken for anyone" produces a character.
    """
    # Only the levers nothing on the form sets. The rest are already in
    # the block above as instructions, and restating them as advice makes
    # the model treat a setting as a suggestion.
    open_levers = [lv for lv in LEVERS if not lv.superseded_by]
    lever_block = "\n".join(
        f"- {lv.title}: {lv.prompt_hint or lv.hint}" for lv in open_levers
    )
    anti_block = _anti_pattern_block()

    set_lines = [line for line in
                 (spec.instruction_for(d.key) for d in DIALS) if line]
    open_dials = [d.title for d in DIALS if spec.instruction_for(d.key) is None]

    # An unset dial contributes nothing but the one line below naming it as
    # open. This is the whole point of the redesign: fill in only the brief
    # and the brief is what the model reads.
    dial_block = "\n".join(set_lines)
    if open_dials:
        if dial_block:
            dial_block += "\n"
        dial_block += ("- Not set, so choose for yourself and say what you chose: "
                       + ", ".join(open_dials))

    # Blanks collapse into one line rather than five "NOT GIVEN" ones: the
    # model needs to know which are open, not to be told five times how to
    # fill one in, and the prompt is competing for attention with itself.
    detail_lines = [
        f"- {d.label}: {spec.details[d.key].strip()}"
        for d in DETAILS if spec.details.get(d.key, "").strip()
    ]
    blank = [d.label for d in DETAILS if not spec.details.get(d.key, "").strip()]
    if blank:
        detail_lines.append(
            "- Not given. Invent only what earns its place, and leave the rest "
            "out: " + ", ".join(blank)
        )

    system = f"""You write characters for a group chat where several of them talk to one human and to each other. Write ONE character from this specification.

WHO THEY ARE — the whole point, and everything below is subordinate to it
{spec.brief.strip()}

{chr(10).join(detail_lines)}

HOW THEY SPEAK
{dial_block}

WORTH HAVING IF THERE IS ROOM
{lever_block}

WHAT DOES NOT WORK, AND MUST NOT APPEAR IN YOUR OUTPUT
{anti_block}

WRITING THE SYSTEM PROMPT
Around {TARGET_PROMPT_WORDS} words. {WRITING_RULES}

Reply in exactly this format, with these labels, and nothing else:

NAME: <up to {MAX_NAME} characters, no slashes>
DESCRIPTION: <up to {MAX_DESCRIPTION} characters, shown in the room roster>
ROUTER_HINTS: <comma-separated topics this character should be picked for>
LENGTH_BIAS: <one of: much_shorter, shorter, match, longer, much_longer>
AVATAR_COLOR: <a hex colour like #4A90D9>
NOTES:
- <one line per choice you made: which settings you followed, which details you invented, and what you chose for anything left open>
SYSTEM_PROMPT:
<the prompt itself, second person, no name prefix, no quotes>"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Write the character."},
    ]


# ---------------------------------------------------------------------------
# Refining a persona that already exists
# ---------------------------------------------------------------------------
#
# A different job from drafting, and the difference is conservation: the
# character already works, and the ask is usually one axis of it. A model
# handed a prompt and told to "make him warmer" will cheerfully rewrite
# the whole thing in its own register and hand back a stranger — so the
# instruction to leave everything else alone has to be as loud as the
# change itself, and the notes have to say what was preserved.

MAX_REFINE_INSTRUCTION = 1000


def build_refine_prompt(current: PersonaDraft, instruction: str) -> List[dict]:
    """The messages that ask the LLM to revise an existing persona.

    ``current`` is what is in the editor form, not what is on disk: the
    user is looking at the form, and refining anything else would revise a
    persona they cannot see.
    """
    system = f"""You revise characters for a group chat. You are given one that already exists and one instruction about what to change. Make that change and leave everything else alone.

THE CHARACTER AS IT STANDS
Name: {current.name}
Description: {current.description}
Router hints: {current.router_hints}
Reply length vs the room: {current.length_bias.value}
System prompt:
{current.system_prompt.strip()}

THE CHANGE ASKED FOR
{instruction.strip()}

HOW TO MAKE IT
Change what the instruction asks for and nothing else. Everything the instruction does not touch stays as it is, in the same words wherever those words still work — this is a revision, not a rewrite, and it must still be recognisably the same character afterwards.

Do not change the name. Keep the description, router hints and reply length as they are unless the change makes them wrong.

Read the instruction narrowly, and change one thing with it. A word about how they SPEAK changes their word choice and nothing else. A word about how they FEEL toward one person says nothing about how they treat everyone.

If the instruction is vague, apply it to the smallest part of the character it could reasonably mean, and say in your notes what you took it to mean.

WHAT DOES NOT WORK, AND MUST NOT APPEAR IN YOUR OUTPUT
{_anti_pattern_block()}

WRITING THE SYSTEM PROMPT
Keep it about as long as it already is unless the instruction asks for more or less. {WRITING_RULES}

Reply in exactly this format, with these labels, and nothing else. Omit a label entirely if that field is unchanged:

DESCRIPTION: <up to {MAX_DESCRIPTION} characters, shown in the room roster>
ROUTER_HINTS: <comma-separated topics this character should be picked for>
LENGTH_BIAS: <one of: much_shorter, shorter, match, longer, much_longer>
NOTES:
- <what you changed, and what you deliberately left alone>
SYSTEM_PROMPT:
<the revised prompt, second person, no name prefix, no quotes>"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Revise the character."},
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LABELS = (
    "NAME", "DESCRIPTION", "ROUTER_HINTS", "LENGTH_BIAS",
    "AVATAR_COLOR", "NOTES", "SYSTEM_PROMPT",
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


def parse_draft(text: str, base: Optional[PersonaDraft] = None) -> PersonaDraft:
    """Turn a model reply into a draft, salvaging whatever is well formed.

    ``base`` is what an omitted or empty block falls back to. Drafting
    passes nothing and gets the field defaults; refining passes the
    persona as it stands, which is what makes "omit a label if that field
    is unchanged" safe to ask for — without it a reply that sensibly left
    DESCRIPTION out would blank the description, and a missing
    LENGTH_BIAS would quietly reset a laconic persona to "match".
    """
    blocks = _split_blocks(text)
    draft = replace(base) if base is not None else PersonaDraft()
    # Notes are about this reply, never inherited from the previous one.
    draft.notes = _parse_notes(blocks.get("NOTES", ""))

    name = _clean_one_line(blocks.get("NAME", ""), MAX_NAME)
    # A slash makes the persona unreachable on /api/personas/{name}/...
    name = name.replace("/", " ").replace("\\", " ").strip()
    if name:
        draft.name = name

    description = _clean_one_line(blocks.get("DESCRIPTION", ""), MAX_DESCRIPTION)
    if description:
        draft.description = description

    router_hints = _clean_one_line(blocks.get("ROUTER_HINTS", ""), MAX_ROUTER_HINTS)
    if router_hints:
        draft.router_hints = router_hints

    prompt = blocks.get("SYSTEM_PROMPT", "").strip().strip("`").strip()
    if prompt:
        draft.system_prompt = prompt[:MAX_SYSTEM_PROMPT]

    raw_bias = _clean_one_line(blocks.get("LENGTH_BIAS", ""), 32).lower()
    if raw_bias:
        try:
            draft.length_bias = LengthBias(raw_bias)
        except ValueError:
            logger.info(
                "Draft returned an unusable length_bias %r; keeping %r",
                raw_bias, draft.length_bias.value,
            )

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

# Words that ARE the default assistant, whatever else the prompt says.
_ASSISTANT_WORDS = ("assistant", "ai companion", "helpful and")

# Words that are only a problem in a heap. Any one of them can be true of
# a person — a character is allowed to be kind — and flagging a single one
# taught the opposite lesson: it read as "warmth is a mistake", which is
# how a cast ends up uniformly unpleasant. Three or more is a pile, and a
# pile is the failure the check is actually for.
_BLAND_WORDS = (
    "friendly", "thoughtful", "curious", "knowledgeable", "insightful",
    "engaging", "helpful", "warm", "kind", "caring", "empathetic",
)
_BLAND_PILE = 3

# Past this, a prompt has stopped describing someone and started
# specifying them.
_RULEBOOK_WORDS = 110


def critique(draft: PersonaDraft) -> List[str]:
    """Warnings about a draft, in the user's terms rather than the model's."""
    warnings: List[str] = []
    prompt = draft.system_prompt
    words = prompt.split()

    if len(words) < 20:
        warnings.append(
            f"The prompt is only {len(words)} words — probably too little to be "
            "anyone in particular."
        )
    elif len(words) > _RULEBOOK_WORDS:
        # The warning used to point the other way, at anything under 40
        # words, because a persona had to be bulky to compete with a
        # 330-word room preamble. The preamble is 160 words now, and the
        # real failure has changed ends: a long prompt is a list of rules,
        # and a character given rules executes them identically every turn
        # instead of reacting to what was said.
        warnings.append(
            f"At {len(words)} words this is closer to a rulebook than a "
            "character. Long prompts get performed the same way every turn — "
            f"nearer {TARGET_PROMPT_WORDS} words leaves them room to react."
        )
    lowered = prompt.lower()
    assistant = sorted({w for w in _ASSISTANT_WORDS if w in lowered})
    if assistant:
        warnings.append(
            "Contains assistant vocabulary (" + ", ".join(assistant) + "). "
            "Those words pull every model back towards its default voice."
        )
    bland = sorted({w for w in _BLAND_WORDS if w in lowered})
    if len(bland) >= _BLAND_PILE:
        warnings.append(
            "Leans on a pile of adjectives (" + ", ".join(bland) + ") rather than "
            "on things this character does. Any one of them is fine; several "
            "together describe nobody in particular."
        )
    if not re.search(r"\byou\b", prompt.lower()):
        warnings.append(
            "The prompt is not addressed to the character in the second "
            "person, which is the form the rest of the app assumes."
        )
    return warnings
