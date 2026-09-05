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
from dataclasses import dataclass, field
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
    hint: str
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
        superseded_by="register",
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
        "A character with a stake argues; a character without one comments.",
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
        "A named opinion about another persona in the room gives the model "
        "something to play that a solo description cannot.",
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
ENGAGEMENT = "How they engage"


DIALS: List[Dial] = [
    Dial(
        "vocabulary", "Vocabulary",
        "Named registers rather than a vague scale — \"plain\" is exactly the "
        "kind of adjective a model reads loosely.",
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
        ],
        # Defaults deliberately below the model's house style. Left to
        # itself it writes everyone as an essayist; this pushes back
        # unless you ask for otherwise.
        "plain_literate",
    ),
    Dial(
        "sentences", "Sentence shape", "How the prose is built, independent of the words in it.",
        SPEECH,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("clipped", "Clipped",
                       "fragments, often no verb; stops as soon as the point is made"),
            DialOption("short", "Short",
                       "short complete sentences, one idea in each"),
            DialOption("neutral", "Neutral", "ordinary sentence lengths, varied"),
            DialOption("flowing", "Flowing",
                       "longer sentences whose subordinate clauses connect ideas"),
            DialOption("rambling", "Rambling",
                       "runs on and digresses; arrives at the point late, or not at all"),
        ],
        "neutral",
    ),
    Dial(
        "register", "Register",
        "Politeness and profanity. This is about WORD CHOICE only — see Temperament for whether they escalate.",
        SPEECH,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("courteous", "Courteous",
                       "polite and careful; softens bad news"),
            DialOption("neutral", "Neutral",
                       "neither polite nor rough; says the thing"),
            DialOption("blunt", "Blunt",
                       "says the unwelcome thing without cushioning it; no profanity"),
            DialOption("coarse", "Coarse",
                       "crude turns of phrase and mild profanity. WORD CHOICE ONLY: this "
                       "does not make them hostile, impatient or uncooperative"),
            DialOption("profane", "Profane",
                       "swears freely and casually. WORD CHOICE ONLY: swearing is how they "
                       "talk to everyone, including people they like"),
        ],
        "neutral",
    ),
    Dial(
        "abstraction", "Abstraction",
        "Whether they argue from cases or from principles.",
        SPEECH,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("concrete", "Concrete",
                       "talks about specific things, people and events; examples rather than principles"),
            DialOption("neutral", "Neutral",
                       "moves between the specific and the general as the topic needs"),
            DialOption("theoretical", "Theoretical",
                       "reaches for principles, systems and generalisations"),
        ],
        "concrete",
    ),
    Dial(
        "temperament", "Temperament",
        "How easily they are provoked. This is the axis that decides whether a rough "
        "character is merely rough or actually belligerent.",
        ENGAGEMENT,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("unflappable", "Unflappable",
                       "nothing gets a rise out of them; rudeness and disagreement land without effect"),
            DialOption("steady", "Steady",
                       "slow to provoke; reacts to real provocation, not to tone"),
            DialOption("reactive", "Reactive",
                       "takes things personally and shows it quickly"),
            DialOption("volatile", "Volatile",
                       "escalates fast and out of proportion"),
        ],
        "steady",
    ),
    Dial(
        "certainty", "Certainty", "How much they qualify what they say.",
        ENGAGEMENT,
        [
            DialOption(UNSPECIFIED, "Let the draft decide", ""),
            DialOption("hedging", "Hedging",
                       "qualifies everything; says \"probably\", admits what they do not know"),
            DialOption("measured", "Measured",
                       "states what they are sure of and flags what they are not"),
            DialOption("confident", "Confident",
                       "asserts without qualifying; rarely says \"I think\""),
            DialOption("dogmatic", "Dogmatic",
                       "treats their own view as settled fact and will not entertain alternatives"),
        ],
        "measured",
    ),
    Dial(
        "stance", "Stance", "What they do with a turn — the single biggest differentiator.",
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
                       "interrupts to correct errors, including small ones"),
        ],
        "responds",
    ),
]

DIALS_BY_KEY: Dict[str, Dial] = {d.key: d for d in DIALS}

# Ordered groups for the form. Built here rather than grouped in the
# template so that the UI cannot drift from the prompt: adding a dial to
# DIALS puts it on screen and in the prompt in the same edit.
DIAL_GROUPS: List[Tuple[str, List[Dial]]] = [
    (group, [d for d in DIALS if d.group == group])
    for group in (SPEECH, ENGAGEMENT)
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
DETAILS: List[DetailField] = [
    DetailField("wants", "What they want",
                "to be proved right about the tide charts",
                "What they are after in a conversation. A character with a stake argues; "
                "one without a stake comments."),
    DetailField("never", "What they never do",
                "never speculates about cargo he has not seen logged",
                "Refusals and avoidances. The strongest single differentiator, because "
                "it cuts off the generic reply."),
    DetailField("wrong", "Where they are wrong",
                "still believes the new pilot rules are temporary",
                "A blind spot or an out-of-date belief. Characters with no flaws "
                "converge on the assistant voice."),
    DetailField("tic", "A verbal tic",
                "opens with \"Right.\" and asks who signed for it",
                "One repeatable thing you would recognise in a single line."),
    DetailField("background", "Background",
                "thirty years on the docks; took the job when his brother died",
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

    def is_usable(self) -> bool:
        """Enough to populate the form: a name and something to say."""
        return bool(self.name.strip() and self.system_prompt.strip())


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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
    lever_block = "\n".join(f"- {lv.title}: {lv.hint}" for lv in open_levers)
    anti_block = "\n".join(f"- Avoid {a}" for a in ANTI_PATTERNS)

    set_lines = [line for line in
                 (spec.instruction_for(d.key) for d in DIALS) if line]
    open_dials = [d.title for d in DIALS if spec.instruction_for(d.key) is None]

    dial_block = "\n".join(set_lines) or "- (nothing specified; choose all of it yourself)"
    if open_dials:
        dial_block += ("\n- Not specified, so decide for yourself and say what you chose: "
                       + ", ".join(open_dials))

    detail_lines = []
    for detail in DETAILS:
        given = spec.details.get(detail.key, "").strip()
        detail_lines.append(
            f"- {detail.label}: {given}" if given
            else f"- {detail.label}: NOT GIVEN — invent something specific"
        )

    system = f"""You write characters for a group chat where several of them talk to one human and to each other. You are given a specification. Write ONE character who follows it exactly.

HOW THIS CHARACTER SPEAKS AND ENGAGES
{dial_block}

These settings are independent of one another and must not bleed together. A coarse or profane register is about WORD CHOICE and nothing else: it does not make a character hostile, impatient, uncooperative or bad at conversation. Whether they escalate is Temperament, and nothing else. How this character feels about any *particular* person is NOT set here at all — that comes out of who they are and who they are talking to, so do not write them as uniformly warm or uniformly hostile toward everyone.

WHO THEY ARE
{spec.brief.strip()}

{chr(10).join(detail_lines)}

WHAT ELSE MAKES A CHARACTER BEHAVE DISTINCTLY (invent whatever the specification leaves open)
{lever_block}

WHAT DOES NOT WORK, AND MUST NOT APPEAR IN YOUR OUTPUT
{anti_block}

WRITING THE SYSTEM PROMPT
Second person, addressed to the character ("You interrupt when..."). Around {TARGET_PROMPT_WORDS} words. Every sentence should say something the character does or does not do; cut anything that is only description.

Write the prompt itself in the plainest language that will do the job. It is a set of instructions to an actor, not an essay about a character, and not a demonstration of the character's own vocabulary — an ornate character still gets a plainly-written prompt.

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


def parse_draft(text: str) -> PersonaDraft:
    """Turn a model reply into a draft, salvaging whatever is well formed."""
    blocks = _split_blocks(text)
    draft = PersonaDraft()

    draft.name = _clean_one_line(blocks.get("NAME", ""), MAX_NAME)
    # A slash makes the persona unreachable on /api/personas/{name}/...
    draft.name = draft.name.replace("/", " ").replace("\\", " ").strip()
    draft.description = _clean_one_line(blocks.get("DESCRIPTION", ""), MAX_DESCRIPTION)
    draft.router_hints = _clean_one_line(blocks.get("ROUTER_HINTS", ""), MAX_ROUTER_HINTS)
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
