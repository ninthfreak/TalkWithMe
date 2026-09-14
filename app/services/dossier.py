"""Interview notes — a larger, tagged store that is read a piece at a time.

``memories.txt`` is small on purpose: every line of it goes into every
reply, so its 16 KB ceiling is really a prompt-size ceiling. That is the
right shape for a persona who needs to remember the people they talk to,
and the wrong shape for an interview, where the whole point is to
accumulate more than a prompt can hold::

     Personas/
       Marion/
         memories.txt          what Marion carries everywhere (unchanged)
         notes/
           tony.txt            Marion's dossier on Tony
           kira.txt            Marion's dossier on Kira

One file per person being interviewed, named by the casefolded subject so
a hand-edit cannot split somebody into two files. The line format is the
memories format with two additions::

     [Tony] #work #vickers @1978 Started at Vickers straight from school.
     [Tony] (assumed) #work He resented the move to management.
     [Tony] (open) #work Why did he leave Vickers, really?
     [Tony] #health (sensitive) He would rather not discuss 2003.

``#topic`` is what makes a selective read possible; ``@when`` is the era
the fact is *about* (not when it was written down), which is what lets a
life story be summarised in order. Both are stripped before the text
reaches the model, and both are only recognised at the start of a line —
a ``#`` inside prose is prose.

**The two-level read is the whole design.** The topic index is tiny and
always goes into the prompt; the contents of two or three topics are
loaded to match what is being discussed. That is what lets an interviewer
say "we have barely touched your army years" — it can see the shape of
what it knows even when it cannot see the detail. It is also *cheaper*
per turn than a full memories file, however large the dossier grows.

Not a database, deliberately. At five thousand notes the file is ~300 KB
and re-reading it costs microseconds; SQLite would buy nothing at that
size and cost the thing the README promises — that you can open the file
and fix a wrong date.

Framework-agnostic like persona_store: files in, files out, no FastAPI,
no config cache, no knowledge of rooms.
"""

import difflib
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from app.config import MAX_MEMORY_LINE_CHARS, MAX_PERSONA_NAME

logger = logging.getLogger(__name__)


NOTES_DIRNAME = "notes"

# A note is one fact, so its text is capped exactly like a memory line.
MAX_NOTE_CHARS = MAX_MEMORY_LINE_CHARS

# Tags are an index, not a description. Four is enough to file a fact
# under a topic, a sub-topic and a person; more than that and the index
# stops narrowing anything, because every note matches every query.
MAX_TAGS_PER_NOTE = 4
MAX_TAG_CHARS = 24
MAX_WHEN_CHARS = 24

# How close two tag spellings have to be before the second is treated as
# the first. Generous, because the failure it exists to prevent — an
# index split across "school", "schools" and "schooling" — is worse than
# the occasional wrong merge, and every merge is logged.
#
# This only ever catches *spelling*. "job" and "work" are 0.0 similar by
# any string measure, so synonyms are the prompt's job: the note pass is
# shown the existing topics and asked to reuse one. This is the backstop
# under that, not a substitute for it.
SNAP_RATIO = 0.85

# Below this length a single character is most of the word, so ratios
# stop meaning anything ("war"/"bar" is 0.67).
_MIN_SNAP_CHARS = 4

# How relevant a topic has to be, against the best-matching one, before
# it is loaded at all. Asked about somebody's father, "work" shares the
# word "yard" with the question and scores one against the best topic's
# six — loading it anyway buried two useful lines under seventy about the
# shipyard. Half is enough to keep two topics that are genuinely both
# about the question, and enough to drop one that merely brushed it.
_RELEVANCE_FLOOR = 0.5

# No single topic may contribute more than this many notes, whatever the
# byte budget allows. Nobody opens a folder and reads four hundred pages;
# they read the few that answer the question. Without this, one big topic
# spends the whole budget on itself and the block stops being legible.
MAX_NOTES_PER_TOPIC = 8

# When a topic holds more than this, most of it can never be retrieved:
# only MAX_NOTES_PER_TOPIC of it is ever shown, so the rest is on disk
# and out of reach. Three times the cap is the point at which that stops
# being a rounding error and starts being the bulk of the topic. Reported
# rather than fixed, because splitting "work" into "the yard", "the union"
# and "management" is a judgement only the model can make.
CROWDED_TOPIC = MAX_NOTES_PER_TOPIC * 3


# ---------------------------------------------------------------------------
# The note format
# ---------------------------------------------------------------------------

_SUBJECT_RE = re.compile(r"^\[([^\]\n]{1,%d})\]\s*(.*)$" % MAX_PERSONA_NAME)
_TAG_TOKEN = re.compile(r"^#([A-Za-z0-9][A-Za-z0-9_-]*)$")
_WHEN_TOKEN = re.compile(r"^@([A-Za-z0-9][A-Za-z0-9_-]*)$")
_FLAG_TOKEN = re.compile(r"^\(\s*(assumed|assumption|guess|guessed|open|sensitive)\s*\)$", re.I)

_FLAG_NAMES = {
    "assumed": "assumed", "assumption": "assumed",
    "guess": "assumed", "guessed": "assumed",
    "open": "open",
    "sensitive": "sensitive",
}


class Note(NamedTuple):
    """One line of a dossier, pulled apart.

    *tags* is the index this note is filed under. *when* is the era the
    note is about, free text because a life is remembered in "the sixties"
    and "after the war" as often as in dates.

    The three flags are three different kinds of line, not three
    adjectives on one: a fact, something the interviewer worked out, and a
    question it still wants answered. *sensitive* is the exception — it
    rides along with any of them and marks a boundary the person set.
    """

    subject: str
    text: str
    tags: Tuple[str, ...] = ()
    when: str = ""
    assumed: bool = False
    open_question: bool = False
    sensitive: bool = False

    def stored(self) -> str:
        """The line as it is written to disk, in canonical order."""
        parts = [f"[{self.subject}]"] if self.subject else []
        if self.assumed:
            parts.append("(assumed)")
        if self.open_question:
            parts.append("(open)")
        if self.sensitive:
            parts.append("(sensitive)")
        parts.extend(f"#{t}" for t in self.tags)
        if self.when:
            parts.append(f"@{self.when}")
        parts.append(self.text)
        return " ".join(parts)


def parse_note_line(line: str) -> Note:
    """One stored line as a Note. Never raises.

    Markers are read only while they are still at the front of the line,
    so ``#blessed`` in the middle of a sentence stays in the sentence.
    Order among them is not enforced on input — these files are meant to
    be hand-edited, and insisting on an order would make a hand-written
    line silently lose its tag — but ``stored()`` always writes them back
    in one canonical order.
    """
    match = _SUBJECT_RE.match(line.strip())
    subject, rest = (match.group(1).strip(), match.group(2).strip()) if match else ("", line.strip())

    tags: List[str] = []
    when = ""
    flags = {"assumed": False, "open": False, "sensitive": False}

    tokens = rest.split()
    consumed = 0
    for token in tokens:
        flag = _FLAG_TOKEN.match(token)
        if flag:
            flags[_FLAG_NAMES[flag.group(1).lower()]] = True
            consumed += 1
            continue
        tag = _TAG_TOKEN.match(token)
        if tag:
            normalised = normalise_tag(tag.group(1))
            if normalised and normalised not in tags:
                tags.append(normalised)
            consumed += 1
            continue
        stamp = _WHEN_TOKEN.match(token)
        if stamp and not when:
            when = stamp.group(1)[:MAX_WHEN_CHARS]
            consumed += 1
            continue
        break

    text = " ".join(tokens[consumed:]).strip()
    return Note(
        subject=subject,
        text=text,
        tags=tuple(tags[:MAX_TAGS_PER_NOTE]),
        when=when,
        assumed=flags["assumed"],
        open_question=flags["open"],
        sensitive=flags["sensitive"],
    )


def normalise_tag(tag: str) -> str:
    """One tag in the spelling the index stores it in.

    Lowercase, dashes for spaces and underscores, nothing else but
    letters, digits and dashes. This is the cheap half of keeping the
    index from fragmenting; snap_tag() is the other half.
    """
    cleaned = re.sub(r"[\s_]+", "-", tag.strip().lstrip("#").lower())
    cleaned = re.sub(r"[^a-z0-9-]", "", cleaned).strip("-")
    return cleaned[:MAX_TAG_CHARS]


def _stem(tag: str) -> str:
    """A crude stem, for matching only — never for storage.

    Plurals and gerunds, and nothing cleverer. Both are how a model
    spells a topic it has already filed under another name: asked to tag
    a fact about a job it writes "work" one turn and "working" the next,
    and a real stemmer would be a dependency to fix a problem this size.

    The length guard is what keeps it from eating short words — "king"
    would otherwise stem to "k" and match nearly anything.
    """
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("es", ""), ("s", "")):
        if tag.endswith(suffix) and len(tag) - len(suffix) >= _MIN_SNAP_CHARS:
            return tag[:-len(suffix)] + replacement
    return tag


def snap_tag(tag: str, known: Iterable[str]) -> Tuple[str, Optional[str]]:
    """A proposed tag, resolved against the tags already in use.

    Returns ``(tag_to_use, what_it_was_snapped_from)`` — the second is
    None when the tag is new or already exact, and is what the caller
    logs. Nothing snaps silently: the whole risk of this mechanism is
    that it quietly files a genuinely new topic under an old one, and the
    only defence is that you can see it happen.

    Three steps, cheapest first: exact match, same word in another
    number ("schools" is "school"), then a close spelling. Anything that
    survives all three is a new topic.
    """
    proposed = normalise_tag(tag)
    if not proposed:
        return "", None

    existing = [normalise_tag(k) for k in known]
    existing = [k for k in existing if k]
    if proposed in existing:
        return proposed, None

    stem = _stem(proposed)
    for candidate in existing:
        if _stem(candidate) == stem:
            return candidate, proposed

    if len(proposed) >= _MIN_SNAP_CHARS:
        long_enough = [k for k in existing if len(k) >= _MIN_SNAP_CHARS]
        close = difflib.get_close_matches(proposed, long_enough, n=1, cutoff=SNAP_RATIO)
        if close:
            return close[0], proposed

    return proposed, None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def notes_dir(persona_dir: Path) -> Path:
    """Where this persona keeps its dossiers."""
    return persona_dir / NOTES_DIRNAME


def _subject_filename(subject: str) -> str:
    """The file a subject's notes live in.

    Casefolded, so a hand-edit that writes "[tony]" cannot start a second
    dossier on the same person. The display spelling survives inside the
    lines, which is where the model reads it from.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", subject.strip()).strip("._-")
    return f"{(safe or 'unknown').casefold()}.txt"


def notes_path(persona_dir: Path, subject: str) -> Path:
    """The dossier file for one subject. May not exist."""
    return notes_dir(persona_dir) / _subject_filename(subject)


def subjects(persona_dir: Path) -> List[str]:
    """Everyone this persona has a dossier on, by the name inside it.

    Read from the files rather than their names: the filename is
    casefolded for safety, and the name the model should see is the one
    written in the lines.
    """
    directory = notes_dir(persona_dir)
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.glob("*.txt")):
        notes = _read_path(path)
        name = next((n.subject for n in notes if n.subject), path.stem)
        found.append(name)
    return found


def _read_path(path: Path) -> List[Note]:
    if not path.is_file():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Unreadable dossier %s (%s)", path, exc)
        return []
    return [parse_note_line(line) for line in content.splitlines() if line.strip()]


def read_notes(persona_dir: Path, subject: str) -> List[Note]:
    """One subject's dossier, oldest first. Never raises.

    Best-effort like every other read path here: an unreadable file means
    the interviewer goes in without its notes, which is a worse interview
    and not a broken one.
    """
    return _read_path(notes_path(persona_dir, subject))


def write_notes(persona_dir: Path, subject: str, notes: Sequence[Note]) -> None:
    """Replace one subject's dossier. Raises OSError on failure.

    The only overwriting entry point, like persona_store.write_memories:
    everything else appends or replaces the open questions, and should
    keep doing so. Raises rather than degrading quietly, because every
    caller is somebody pressing a button and a silent no-op would leave
    them believing the file had changed.
    """
    directory = notes_dir(persona_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = notes_path(persona_dir, subject)
    kept = [n for n in notes if n.text.strip()]
    if not kept:
        # No file and an empty file mean the same thing to every reader,
        # and no file is the state the rest of the app produces.
        path.unlink(missing_ok=True)
        return
    body = "\n".join(n.stored() for n in kept) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


class AppendResult(NamedTuple):
    """What filing a batch of notes actually did.

    *merges* is ``(proposed, used)`` pairs — reported rather than
    swallowed, because tag snapping is the one step here that can quietly
    file a new topic under an old name.
    """

    filed: List[Note]
    duplicates: List[Note]
    merges: List[Tuple[str, str]]


def append_notes(persona_dir: Path, subject: str, notes: Iterable[Note]) -> AppendResult:
    """File new notes, snapping their tags to the ones already in use.

    Duplicates are refused on the way in, compared on text alone: the
    same fact arriving with different tags is the same fact, and the note
    pass re-reads the same conversation by design.

    Raises OSError on a failed write — this is called from a deliberate
    note-taking pass, and a note that was never written is worth knowing
    about.
    """
    existing = read_notes(persona_dir, subject)
    seen = {n.text.casefold() for n in existing}
    known_tags = list(dict.fromkeys(t for n in existing for t in n.tags))

    filed: List[Note] = []
    duplicates: List[Note] = []
    merges: List[Tuple[str, str]] = []

    for note in notes:
        text = note.text.strip()[:MAX_NOTE_CHARS]
        if not text:
            continue
        if text.casefold() in seen:
            duplicates.append(note)
            continue

        tags: List[str] = []
        for raw in note.tags[:MAX_TAGS_PER_NOTE]:
            tag, merged_from = snap_tag(raw, known_tags)
            if not tag or tag in tags:
                continue
            if merged_from:
                merges.append((merged_from, tag))
            tags.append(tag)
            if tag not in known_tags:
                known_tags.append(tag)

        settled = note._replace(
            subject=subject, text=text, tags=tuple(tags),
            when=note.when[:MAX_WHEN_CHARS],
        )
        filed.append(settled)
        seen.add(text.casefold())

    if filed:
        write_notes(persona_dir, subject, existing + filed)
    for proposed, used in merges:
        logger.info("Dossier %s/%s: filed '#%s' under existing '#%s'",
                    persona_dir.name, subject, proposed, used)
    return AppendResult(filed=filed, duplicates=duplicates, merges=merges)


def replace_open_questions(
    persona_dir: Path, subject: str, questions: Iterable[Note],
) -> int:
    """Swap the whole open-question list for a new one. Returns how many.

    Facts accumulate; questions do not. An open question is answered by
    the conversation moving on, and no amount of prompting reliably gets
    a model to *delete* a line — so it is never asked to. The note pass
    is shown the current questions and returns the ones that still stand,
    and what it leaves out disappears because it left it out.

    Raises OSError on a failed write.
    """
    kept = [n for n in read_notes(persona_dir, subject) if not n.open_question]
    fresh = [
        n._replace(subject=subject, open_question=True, text=n.text.strip()[:MAX_NOTE_CHARS])
        for n in questions if n.text.strip()
    ]
    write_notes(persona_dir, subject, kept + fresh)
    return len(fresh)


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

class Topic(NamedTuple):
    """One entry in the index: a tag, how much is filed under it, and
    whether anything under it is a boundary the person set."""

    tag: str
    count: int
    sensitive: bool = False


def topics(notes: Sequence[Note]) -> List[Topic]:
    """The index, biggest first. Open questions do not count as content.

    A question filed under "army" is a reason to *ask* about the army,
    not evidence that the army is covered — counting it would make a
    thin topic look answered, which is the one thing the index exists to
    prevent.
    """
    counts: Dict[str, int] = {}
    tender: Dict[str, bool] = {}
    for note in notes:
        if note.open_question:
            continue
        for tag in note.tags:
            counts[tag] = counts.get(tag, 0) + 1
            tender[tag] = tender.get(tag, False) or note.sensitive
    return [
        Topic(tag=tag, count=count, sensitive=tender.get(tag, False))
        for tag, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


# ---------------------------------------------------------------------------
# Keeping the index usable
# ---------------------------------------------------------------------------
#
# The memory system tidies memories.txt to save *bytes*, because every
# byte of it is injected. A dossier has bytes to spare — the whole point
# is that it outgrows a prompt — so nothing here is about size. What
# degrades instead is **retrieval**, in two ways:
#
#   * the index fragments. The write path snaps a proposed tag against
#     the topics already in the file, so the app's own writes arrive
#     tidy — but these files are meant to be opened and edited (the
#     README says so), and a hand-written "#schools" beside an
#     established "#school" splits a topic in two. The index is the one
#     part of a dossier that *is* injected every turn, so a split topic
#     costs something every turn.
#   * topics outgrow what can be shown. A topic of four hundred notes
#     surfaces eight, so the other three hundred and ninety-two are held
#     and unreachable.
#
# This pass fixes the first mechanically and reports the second, which
# needs a model. Same division as the memory system: dedupe_memories()
# runs itself because it can only ever remove an exact copy, and condense
# is behind a button because it rewrites sentences.

class ReindexReport(NamedTuple):
    """What a reindex did, and what it could not do without a model.

    *untagged* and *crowded* are the interesting half: both are notes the
    dossier holds and retrieval cannot reach, which is invisible from the
    file and from the conversation.
    """

    merges: List[Tuple[str, str]]
    crowded: List[Topic]
    duplicates_removed: int = 0
    untagged: int = 0
    changed: bool = False


def reindex(persona_dir: Path, subject: str) -> ReindexReport:
    """Tidy one dossier's index without a model. Never raises.

    Three mechanical steps, each of which can only ever collapse two
    spellings of one thing into one of them:

      * byte-identical notes lose their copies, first kept, like
        dedupe_memories — the note pass re-reads an overlapping window by
        design, so a file written only by this app can still hold them
        after a hand-edit;
      * every tag is re-snapped against the tags better established than
        it, biggest first, so a hand-written "#schools" folds into the
        "#school" that already holds forty notes and never the other way
        round;
      * nothing else. Merging two notes that say the same thing in
        different words is a judgement, and it belongs to the condense
        pass with a preview in front of it.

    Deterministic and idempotent: the surviving tags do not snap to each
    other, so a second run finds nothing. Writes only when something
    actually changed, since this runs on the read path.
    """
    notes = read_notes(persona_dir, subject)
    if not notes:
        return ReindexReport(merges=[], crowded=[])

    seen = set()
    deduped = []
    for note in notes:
        key = note.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(note)
    duplicates = len(notes) - len(deduped)

    # Established first, so a small topic folds into a big one and never
    # the other way round. First appearance breaks ties, which keeps two
    # equally-sized neighbours from swapping places on alternate runs.
    counts: Dict[str, int] = {}
    first: Dict[str, int] = {}
    for position, note in enumerate(deduped):
        for tag in note.tags:
            counts[tag] = counts.get(tag, 0) + 1
            first.setdefault(tag, position)
    ranked = sorted(counts, key=lambda t: (-counts[t], first[t], t))

    established: List[str] = []
    folded: Dict[str, str] = {}
    merges: List[Tuple[str, str]] = []
    for tag in ranked:
        settled, merged_from = snap_tag(tag, established)
        if merged_from and settled != tag:
            folded[tag] = settled
            merges.append((tag, settled))
            continue
        established.append(settled)

    rewritten = []
    for note in deduped:
        tags = tuple(dict.fromkeys(folded.get(t, t) for t in note.tags))
        rewritten.append(note if tags == note.tags else note._replace(tags=tags))

    changed = bool(duplicates or merges)
    if changed:
        try:
            write_notes(persona_dir, subject, rewritten)
        except OSError as exc:
            # The state it was called to improve is the state it leaves,
            # which is not a reason to fail a reply.
            logger.warning("Could not reindex %s/%s: %s", persona_dir.name, subject, exc)
            changed = False
        else:
            for was, now in merges:
                logger.info("Dossier %s/%s: folded '#%s' into '#%s'",
                            persona_dir.name, subject, was, now)
            if duplicates:
                logger.info("Dossier %s/%s: dropped %d duplicate note(s)",
                            persona_dir.name, subject, duplicates)

    index = topics(rewritten)
    untagged = sum(1 for n in rewritten if not n.tags and not n.open_question)
    crowded = [t for t in index if t.count > CROWDED_TOPIC]
    if untagged:
        # Only the recency fallback can ever show these. Said out loud
        # because nothing about the file looks wrong.
        logger.info("Dossier %s/%s: %d note(s) carry no topic and can only "
                    "be reached by recency", persona_dir.name, subject, untagged)
    for topic in crowded:
        logger.info("Dossier %s/%s: '#%s' holds %d notes and shows at most %d",
                    persona_dir.name, subject, topic.tag, topic.count, MAX_NOTES_PER_TOPIC)

    return ReindexReport(
        merges=merges, duplicates_removed=duplicates,
        untagged=untagged, crowded=crowded, changed=changed,
    )


_STOPWORDS = {
    "about", "after", "again", "all", "and", "any", "are", "because", "been",
    "before", "being", "but", "can", "did", "does", "doing", "done", "down",
    "for", "from", "had", "has", "have", "her", "here", "him", "his", "how",
    "into", "its", "just", "like", "more", "most", "much", "not", "now", "off",
    "once", "one", "only", "other", "our", "out", "over", "own", "said", "same",
    "she", "should", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "too", "under",
    "very", "was", "way", "were", "what", "when", "where", "which", "while",
    "who", "why", "will", "with", "would", "you", "your",
}


def _content_words(text: str) -> List[str]:
    """The words in a query worth matching on."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


class Selection(NamedTuple):
    """What one turn reads out of a dossier.

    *index* is every topic, always — it is thirty words and it is what
    lets the interviewer know the shape of what it has. *notes* is the
    contents of the topics that matched, and *open_questions* is the
    agenda, which is never selective: there are few of them and they are
    the reason to ask anything at all.
    """

    index: List[Topic]
    matched: List[str]
    notes: List[Note]
    open_questions: List[Note]


def select(
    notes: Sequence[Note],
    query: str,
    *,
    budget_chars: int = 2000,
    max_topics: int = 3,
    fallback_recent: int = 5,
) -> Selection:
    """Pick the notes worth putting in front of the model this turn.

    Topics are the unit, not lines: a person opens the folder marked
    "school", they do not retrieve four sentences from four folders. It
    also makes the selection legible — you can see which topics were
    loaded and say whether that was right.

    Scoring is keyword overlap, and deliberately so: no embeddings, no
    model call, no dependency, and a wrong answer that can be explained
    by looking at the words. A topic's name counts for more than its
    contents, since somebody asking about school says "school".

    **A topic scores on how many distinct query words it can answer, not
    on how many times it answers them.** Summing over notes looked
    reasonable and was wrong in the way that matters: after a long
    interview "work" holds four hundred notes, so a single stray word
    scored it four hundred and it won every turn — a question about
    somebody's father came back with the shipyard. Counting each word
    once puts a one-note "father" ahead of it, which is the answer a
    person would give.

    **The budget is enforced on the first topic too**, for the same
    reason. A topic large enough to swamp the prompt is usually the right
    topic, so it is trimmed rather than dropped: the notes in it that
    answer the question go in, newest first among equals, and the rest
    stay on disk where the index still counts them.

    With nothing matching — the first question of a sitting, or a subject
    change nothing is filed under — the most recent notes go in instead,
    so the block is never empty and the interviewer never sounds like it
    has just walked in.
    """
    index = topics(notes)
    open_questions = [n for n in notes if n.open_question]
    content = [n for n in notes if not n.open_question]

    words = set(_content_words(query))
    # Indices into *content*, not the notes themselves: two notes can be
    # equal (a repeated fact in a hand-edited file) and the selection has
    # to keep them apart to stay in file order.
    by_tag: Dict[str, List[int]] = {}
    for position, note in enumerate(content):
        for tag in note.tags:
            by_tag.setdefault(tag, []).append(position)

    scored = []
    for topic in index:
        tag_words = set(topic.tag.split("-"))
        answered = set()
        for position in by_tag.get(topic.tag, []):
            answered |= words & set(_content_words(content[position].text))
        score = 4 * len(words & tag_words) + len(answered)
        if score:
            # Ties go to the smaller topic: "father" says more about a
            # question than "family" does, and both are about the father.
            scored.append((-score, topic.count, topic.tag))
    scored.sort()

    # Only topics comparably relevant to the best one. A topic that
    # merely brushed the question contributes noise the model then has to
    # read past, and a big one contributes a lot of it.
    if scored:
        floor = -scored[0][0] * _RELEVANCE_FLOOR
        scored = [row for row in scored if -row[0] >= floor]

    matched: List[str] = []
    chosen: List[int] = []
    taken = set()
    spent = 0
    for _, _, tag in scored[:max_topics]:
        room = budget_chars - spent
        if room <= 0:
            break
        fitted = _fit(content, by_tag.get(tag, []), words, room)
        fresh = [p for p in fitted if p not in taken]
        if not fresh:
            continue
        matched.append(tag)
        taken.update(fresh)
        chosen.extend(fresh)
        spent += sum(_cost(content[p]) for p in fresh)

    if not chosen and content:
        recent = list(range(len(content)))[-fallback_recent:]
        chosen = _fit(content, recent, words, budget_chars)

    # File order, not score order: a dossier read out of sequence reads
    # as a shuffled life.
    return Selection(
        index=index,
        matched=matched,
        notes=[content[p] for p in sorted(chosen)],
        open_questions=open_questions,
    )


def _cost(note: Note) -> int:
    """Roughly what one note costs in the rendered block."""
    return len(note.text) + len(note.when) + 4


def _fit(
    content: Sequence[Note], positions: Sequence[int], words: set, budget: int,
) -> List[int]:
    """As much of one topic as the budget allows, most useful first.

    Whole topics are the normal case and this returns them untouched. A
    topic too large to fit — or simply too long to read — is where the
    choice happens: the notes that answer the question come first, ties
    broken by recency, because in a four-hundred-note topic the recent
    notes are the ones the conversation has been building on.

    Two limits, and the note count is the one that usually bites. A
    budget in characters lets one topic spend all of it on itself, which
    is legal and unreadable: seventy lines about a shipyard with the two
    that mattered somewhere in the middle.

    At least one note always comes back when there is one — a budget too
    small for a single line is a caller's problem, and silently returning
    nothing would read as "this topic is empty".
    """
    if not positions:
        return []
    if (len(positions) <= MAX_NOTES_PER_TOPIC
            and sum(_cost(content[p]) for p in positions) <= budget):
        return list(positions)

    ranked = sorted(
        positions,
        key=lambda p: (-len(words & set(_content_words(content[p].text))), -p),
    )
    kept: List[int] = []
    spent = 0
    for position in ranked[:MAX_NOTES_PER_TOPIC]:
        cost = _cost(content[position])
        if kept and spent + cost > budget:
            continue
        kept.append(position)
        spent += cost
    return kept


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _when(note: Note) -> str:
    return f" ({note.when.replace('-', ' ')})" if note.when else ""


def render_block(selection: Selection, subject: str) -> str:
    """The dossier as it reaches the model.

    Three parts, in the order they are useful: what there is, what is
    relevant now, and what is still missing. Everything is stated as a
    fact about the person or about what the interviewer holds — the
    prompt says what is, never what to avoid.

    The assumed wording is word-for-word the one the memory injection
    uses. A persona that meets the same idea in two spellings has to
    work out that they are the same idea, and that is a worse use of a
    small model than remembering the sentence.
    """
    if not selection.index and not selection.notes and not selection.open_questions:
        return ""

    parts = []

    if selection.index:
        entries = []
        for topic in selection.index:
            tender = ", he would rather not" if topic.sensitive else ""
            entries.append(f"{topic.tag.replace('-', ' ')} ({topic.count}{tender})")
        parts.append(
            f"What you have written down about {subject}:\n  " + " · ".join(entries)
        )

    if selection.notes:
        heading = (
            f"On what you are discussing now — {', '.join(t.replace('-', ' ') for t in selection.matched)}:"
            if selection.matched else "The last of what you wrote down:"
        )
        known = [n for n in selection.notes if not n.assumed]
        assumed = [n for n in selection.notes if n.assumed]
        body = [f"  {n.text}{_when(n)}" for n in known]
        if assumed:
            body.append("  You have assumed, though nobody said so: "
                        + " ".join(f"{n.text}{_when(n)}" for n in assumed))
        parts.append(heading + "\n" + "\n".join(body))

    if selection.open_questions:
        parts.append("You still want to know:\n"
                     + "\n".join(f"  {n.text}" for n in selection.open_questions))

    return "\n\n".join(parts)
