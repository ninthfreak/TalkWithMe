"""Keeps a persona's reply inside its own voice.

Two behaviours this filter exists to stop, both of which the room preamble
asks the model not to do and some models do anyway:

1. **Continuing someone else's turn.** History reaches the model as
   ``[Name]: text`` user turns (see ``SessionManager.build_llm_messages``),
   which reads like a transcript — and a transcript is a format models
   happily extend with another speaker's line.
2. **Inventing a character.** Same cause: a new ``[Marcus]:`` line is a
   perfectly natural continuation of a transcript.

It also strips the speaker's own name prefix, which models leak often
enough that the prompt rule alone is not enough.

**A tag is not always a bare ``Name:``.** For a long time that was the only
shape recognised, and it is the shape instruct-tuned models use least: they
reach for markdown. A live room produced one reply carrying three other
personas' turns, every tag written ``**Luna:**``, and the guard cut none of
them — while those three personas also answered for themselves, so each
appeared twice. So a tag is now: optional list or markdown decoration
(``**``, ``-``, ``>``, ``### ``, ``1. ``), optional brackets, the name,
optional emphasis, then a colon or a dash — or, for a name already known to
be a speaker, brackets or decoration alone (``### Luna``, ``[Luna]``).

**Known names are treated differently from invented ones**, and the
asymmetry is deliberate. A name in the room's roster is *known* to belong to
a speaker, so it cuts wherever it turns up, including mid-line after a
sentence end ("I agree. Luna: but…"). An unknown name is a judgement call,
so it only cuts at the start of a line, where a tag is unambiguous. The
guiding rule throughout is that a false positive silently truncates a
legitimate reply, which is worse than missing a cut — so every widening
here is bounded by something that makes the text a turn rather than prose.

The filter is a pure state machine over the streamed text — no config, no
I/O — so it can be unit-tested against token lists directly.

**Why it buffers at line starts.** ``static/chat.js`` appends every
``token`` SSE event straight into the bubble and never re-renders from the
``done`` event (whose ``text`` is consumed only by TTS). So a prefix has to
be resolved *before* the first token is emitted, not cleaned up at the end
— otherwise it stays on screen until reload and disagrees with what was
persisted. Buffering only ever happens just after a newline, which is a
natural pause in the stream.
"""

import re
from typing import Iterable, List, Optional

# Markdown and list decoration a model puts in front of a speaker tag.
# Matching only a bare "Name:" was the guard's biggest hole: instruct-tuned
# models write "**Luna:**", "- Luna:", "### Luna", "> Luna:" far more often
# than the plain form, and every one of those streamed through untouched.
_DECORATION = r"(?:[*_~`#>]+|[-\u2013\u2014\u2022]|\d{1,3}[.)])"
_LINE_LEAD_RE = re.compile(rf"^[ \t]*(?:{_DECORATION}[ \t]*)*")

# A speaker prefix: optional brackets around a short name, optional trailing
# emphasis ("**Luna**:"), then a separator. The name charset stays narrow on
# purpose — a wide one turns ordinary prose ("the answer is: yes") into a
# false positive, and a false positive silently truncates a legitimate reply.
#
# Groups: 1 open bracket, 2 name, 3 close bracket, 4 separator (empty when
# the tag is bracketed and carries no colon at all, as in "[Luna] hello").
_PREFIX_RE = re.compile(
    r"^([\[(])?[ \t]*([\w .'\-]{1,25}?)[ \t]*([\])])?[ \t]*[*_~`]*[ \t]*"
    r"(:|\u2014|\u2013|--)"
)

# The same shape with no separator at all, for a bracketed tag like
# "[Luna] hello" — the brackets alone say it is a speaker. The lookahead
# waits out a colon that has not arrived yet: deciding on "[Alex]" the
# moment the bracket closes stripped the tag but left the ": " behind.
_BRACKET_ONLY_RE = re.compile(
    r"^\[[ \t]*([\w .'\-]{1,25}?)[ \t]*\][ \t]*(?=[^:*_~`\s])"
)

# Characters a name may contain, for the "is this still a candidate?" check.
# Trailing emphasis and a close bracket are allowed because they sit between
# the name and the separator.
_NAME_CHARS_RE = re.compile(r"^[\w .'\-]*[\])]?[ \t]*[*_~`]*[ \t]*$")

# A list marker part-way through arriving ("1", then "1."). Without this the
# digit is released as prose and the marker leaks out ahead of the cut.
_PARTIAL_MARKER_RE = re.compile(r"^\d{1,3}[.)]?[ \t]*$")

# A sentence ending, so a tag can be caught mid-line too ("That settles it.
# Luna: I disagree."). Only known speakers are cut there — see _decide.
_SENTENCE_BREAK_RE = re.compile(r"[.!?\u2026][\"'\u201d\u2019)\]]*[ \t]$")

# Hard stop on buffering, so nothing can hold forever. Big enough for the
# longest decorated tag: "**" + a 25-character name + "**:".
_MAX_HOLD = 48

# Words that routinely open a line with a colon in ordinary prose. Without
# this, an unbracketed "Note:" or "Summary:" would be read as a new speaker
# and truncate the reply. Only consulted for the unbracketed, unknown-name
# case; a bracketed prefix or a known persona name never reaches it.
_PROSE_LEAD_INS = frozenset(
    {
        # Labels
        "note", "notes", "warning", "caution", "example", "examples",
        "answer", "question", "tip", "tips", "summary", "conclusion",
        "edit", "update", "ps", "aside", "caveat", "disclaimer", "hint",
        "key", "result", "results", "output", "input", "error", "todo",
        "step", "steps", "reason", "problem", "solution", "goal",
        "context", "source", "sources", "reference", "references",
        "translation", "pros", "cons", "takeaway", "recommendation",
        "verdict", "thought", "thoughts", "response", "reply", "plan",
        "approach",
        # Adverbs and deictics that open a line. None of them is a
        # plausible character name, and all of them were cutting replies:
        # a single capitalised word before a colon is exactly the shape of
        # an invented speaker.
        "first", "second", "third", "finally", "however", "instead",
        "here", "there", "now", "then", "next", "also", "besides",
        "meanwhile", "anyway", "overall", "again", "lastly", "otherwise",
        "therefore", "still",
    }
)

# The stoplist is consulted for the whole phrase *and* its last word, so a
# multi-word lead-in ending in one of the above ("Final Answer:",
# "Executive Summary:", "My Recommendation:") is covered by the tail alone.
# Only the unbracketed, unknown-name case reaches it; a bracketed prefix or
# a known persona name never does.


# Decisions the buffer can reach.
_WAIT, _FLUSH, _CUT, _STRIP = "wait", "flush", "cut", "strip"


def _capitalised_words(text: str, limit: int = 3) -> bool:
    """True while *text* could still be a name of at most *limit* words.

    Every word must start with a capital. This is what keeps buffering
    short: ordinary prose ("Hello there", "The answer is") fails on its
    second word and is released after a handful of characters, so replies
    still stream token by token instead of arriving in one lump.
    """
    words = text.split()
    if len(words) > limit:
        return False
    return all(w[:1].isupper() for w in words)


def _is_prose_lead_in(text: str) -> bool:
    """True for a phrase that opens a line in prose rather than naming a speaker.

    Checks the whole phrase *and* its last word, because the stoplist held
    single words while real lead-ins are often multi-word: "Final Answer:"
    and "Executive Summary:" both sailed past a bare-word lookup and cut
    correct replies to nothing.
    """
    flat = text.casefold().replace(".", "").strip()
    if flat in _PROSE_LEAD_INS:
        return True
    words = flat.split()
    return bool(words) and words[-1] in _PROSE_LEAD_INS


def _looks_like_a_name(text: str) -> bool:
    """True for 1-3 capitalised words — "Marcus", "Dr. Smith", "Mary Anne".

    Used only for unbracketed prefixes whose name is not a known persona,
    i.e. the invented-character case. Requiring capitalisation keeps
    ordinary prose ("the answer is:") out.
    """
    words = text.split()
    if not (1 <= len(words) <= 3):
        return False
    if _is_prose_lead_in(text):
        return False
    return all(w[:1].isupper() for w in words if w)


class ReplyGuard:
    """Filter one persona's streamed reply.

    Feed it token strings; it returns the text that is safe to emit. Once
    :attr:`stopped` is true the reply has been cut at a foreign speaker
    prefix and the caller should stop consuming the upstream LLM stream.
    """

    def __init__(self, persona_name: str, known_names: Iterable[str] = ()):
        self.persona_name = persona_name
        self._own = persona_name.casefold()
        # Other personas in the room. Their names cut even without brackets
        # and without looking name-like, because we know they are speakers.
        self._known = {n.casefold() for n in known_names} - {self._own}
        # Start buffering: position 0 is a line start, so a reply opening
        # with "Alex: " is caught just like one opening after a newline.
        self._holding = True
        self._hold = ""
        self._stopped = False
        self._cut_at: Optional[str] = None
        # Set after stripping a self-prefix: the space that followed the
        # colon belongs to the prefix, not to the reply.
        self._skip_ws = False
        # Set after stripping a self-prefix that opened with emphasis
        # ("**Alex:**"): the emphasis closing it arrives *after* the colon,
        # so it cannot be consumed by the match that spotted the tag.
        self._skip_marks = False
        # Inside a fenced code block nothing is a speaker turn, so a YAML
        # key like "Model:" or an HTTP header must not cut the reply.
        self._in_code = False
        # Text emitted on the current line, used to spot fence markers and
        # sentence ends.
        self._line = ""
        # Whether the current buffer was armed at a line start or mid-line
        # after a sentence end. Mid-line only ever cuts on a *known*
        # speaker: "I'd ask Luna: what now?" is a sentence, not a turn.
        self._at_line_start = True

    # -- Public API ---------------------------------------------------------

    @property
    def stopped(self) -> bool:
        """True once the reply was cut at another speaker's prefix."""
        return self._stopped

    @property
    def cut_at(self) -> Optional[str]:
        """The name that triggered the cut, for logging. None if no cut."""
        return self._cut_at

    def feed(self, token: str) -> str:
        """Consume one streamed token; return the text safe to emit now."""
        if self._stopped or not token:
            return ""
        out: List[str] = []
        for ch in token:
            if self._stopped:
                break
            if self._holding:
                self._feed_held(ch, out)
            else:
                if self._skip_marks:
                    if ch in "*_~`":
                        continue
                    self._skip_marks = False
                if self._skip_ws:
                    if ch in " \t":
                        continue
                    self._skip_ws = False
                self._emit(out, ch)
                if ch == "\n":
                    self._holding = True
                    self._hold = ""
                    self._at_line_start = True
                elif not self._in_code and _SENTENCE_BREAK_RE.search(self._line):
                    # A turn can start mid-line: "That settles it. Luna: ..."
                    # Only known speakers cut here, so this holds nothing
                    # unless the next characters spell one of their names.
                    self._holding = True
                    self._hold = ""
                    self._at_line_start = False
        return "".join(out)

    def flush(self) -> str:
        """Release anything still buffered at end of stream."""
        if self._stopped:
            return ""
        held, self._hold = self._hold, ""
        self._holding = False
        out: List[str] = []
        self._emit(out, held)
        return "".join(out)

    # -- Line-end handling --------------------------------------------------

    def _decorated_name_line(self) -> Optional[str]:
        """A known speaker's name alone on a decorated line, or None.

        Catches the heading and tag forms — "### Luna", "**Luna**",
        "[Luna]" — where the model opens someone else's turn with no colon
        at all. Decoration or brackets are required: a bare "Luna" alone on
        a line is a plausible one-word answer, and cutting that would
        delete a real reply.
        """
        if not self._at_line_start or self._in_code:
            return None
        lead = _LINE_LEAD_RE.match(self._hold).end()
        rest = self._hold[lead:].strip()
        bracketed = rest.startswith("[") and rest.endswith("]")
        if not lead and not bracketed:
            return None
        name = rest.strip("[]").strip().strip("*_~`").strip()
        if name.casefold() in self._known or name.casefold() == self._own:
            return name
        return None

    # -- Internals ----------------------------------------------------------

    def _emit(self, out: List[str], text: str) -> None:
        """Append emitted text, tracking ``` fences as lines complete.

        Fences are tracked on *output* rather than on the held buffer: a
        fence line contains a backtick, which is not a name character, so
        the line is released long before its newline is ever held. And it
        must happen as each character is emitted, not at the end of feed(),
        because the cut decision for the next line is taken inside the same
        call.
        """
        if not text:
            return
        out.append(text)
        for ch in text:
            if ch == "\n":
                if self._line.lstrip().startswith("```"):
                    self._in_code = not self._in_code
                self._line = ""
            else:
                self._line += ch

    def _feed_held(self, ch: str, out: List[str]) -> None:
        if ch == "\n":
            name = self._decorated_name_line()
            if name is not None:
                if name.casefold() == self._own:
                    # Our own name as a heading — the self-prefix leak in
                    # another shape. Drop the line, keep the reply.
                    self._hold = ""
                    self._skip_ws = True
                    return
                self._cut_at = name
                self._stopped = True
                self._hold = ""
                return
            # The line ended without a separator, so it was never a prefix.
            # Release it and stay at a line start for the next one.
            self._emit(out, self._hold + ch)
            self._hold = ""
            self._at_line_start = True
            return

        self._hold += ch
        decision, consumed = self._decide()
        if decision == _WAIT:
            return
        if decision == _CUT:
            self._stopped = True
            self._hold = ""
            return
        if decision == _STRIP:
            # Drop the persona's own prefix, keep whatever followed it.
            self._hold = self._hold[consumed:].lstrip()
            self._skip_ws = True
        self._emit(out, self._hold)
        self._hold = ""
        self._holding = False
        self._at_line_start = False

    def _decide(self) -> tuple:
        """Classify the buffer. Returns (decision, chars consumed by prefix)."""
        if self._in_code:
            # Code, not dialogue. Release it and stop inspecting this line.
            if len(self._hold) >= 3 or not "```".startswith(self._hold.lstrip()):
                return _FLUSH, 0
            return _WAIT, 0

        # Markdown and list decoration only counts at a line start; mid-line
        # it is ordinary punctuation.
        lead = _LINE_LEAD_RE.match(self._hold).end() if self._at_line_start else 0
        body = self._hold[lead:]

        match = _PREFIX_RE.match(body)
        bracket_only = None if match else _BRACKET_ONLY_RE.match(body)
        if match or bracket_only:
            consumed = lead + (match or bracket_only).end()
            name = (match or bracket_only).group(2 if match else 1).strip()
            low = name.casefold()

            if low == self._own:
                # "**Alex:**" closes its emphasis *after* the colon, so it
                # has not arrived yet when the tag is spotted — it is
                # skipped as it streams instead. Only when the tag opened
                # with emphasis, so a reply that genuinely starts
                # "Alex: **listen**" keeps its bold.
                if lead and set(self._hold[:lead].strip()) <= set("*_~`"):
                    self._skip_marks = True
                return _STRIP, consumed

            # A name we know belongs to a speaker is unambiguous, so it cuts
            # wherever it appears — decorated, bracketed, dash-separated, or
            # mid-line after a sentence end. This is the case that matters:
            # a persona writing another persona's turn.
            #
            # Mid-line the separator has to be a colon or brackets. A dash
            # there is punctuation far more often than a turn ("Ask her —
            # Luna — she was there"), and the cost of being wrong is a real
            # reply truncated where nobody can see why. A room with a
            # persona named after an ordinary word keeps some residual risk
            # ("Check the table. Data: 42 rows"), which is the price of
            # catching "I agree. Luna: but what about…".
            if low in self._known:
                if self._at_line_start or bracket_only or match.group(4) == ":":
                    self._cut_at = name
                    return _CUT, consumed
                return _FLUSH, 0

            # An unknown name is a judgement call, so it only cuts at a line
            # start. Mid-line, "I'd ask Marcus: what now?" is a sentence.
            if not self._at_line_start:
                return _FLUSH, 0

            # A bracketed tag is the format the model sees in history, so it
            # is the overwhelmingly common shape of an invented character.
            if bracket_only or (match.group(1) and match.group(3)):
                self._cut_at = name
                return _CUT, consumed

            # Unbracketed and unknown: only a colon after something that
            # reads like a character name. A dash is too weak on its own.
            if match.group(4) == ":" and _looks_like_a_name(name):
                self._cut_at = name
                return _CUT, consumed
            return _FLUSH, 0

        if len(self._hold) < _MAX_HOLD and self._still_a_candidate():
            return _WAIT, 0
        return _FLUSH, 0

    def _still_a_candidate(self) -> bool:
        """Could more characters still turn the buffer into a prefix?

        Answering "no" early is what keeps the held text short.
        """
        s = self._hold.lstrip(" \t")
        if self._at_line_start:
            # Decoration may still be arriving ("*", then "*", then the
            # name), so keep waiting while the buffer is only decoration.
            lead_match = _LINE_LEAD_RE.match(s)
            if lead_match.end() == len(s) and s:
                return True
            if _PARTIAL_MARKER_RE.match(s):
                return True
            s = s[lead_match.end():]

        if not self._at_line_start:
            # Mid-line only known speakers cut, so hold only while the
            # buffer could still spell one — brackets included, since
            # "[Luna]" is a tag wherever it appears. Anything else is prose
            # and must keep streaming, so this says "no" after a character
            # or two on any ordinary sentence.
            low = s.casefold()
            stem = low.lstrip("[").rstrip(" \t").rstrip("]").rstrip(" \t")
            return bool(low) and any(k.startswith(stem) for k in self._known)

        # A two-character separator may still be arriving: "Luna -" is one
        # keystroke short of "Luna --". Only for a name already known to be
        # a speaker, so prose that happens to end in a hyphen keeps
        # streaming.
        pending = re.fullmatch(r"(.*?)[ \t]*[*_~`]*[ \t]*-", s)
        if pending:
            stem = pending.group(1).strip().casefold()
            if stem and (stem in self._known or stem == self._own):
                return True

        if s.startswith("[") or s.startswith("("):
            # Bracketed prefixes cut whatever the name looks like, so only
            # the charset and length bound them.
            inner = s[1:]
            return len(inner) <= 27 and _NAME_CHARS_RE.match(inner) is not None
        if len(s) > 28 or _NAME_CHARS_RE.match(s) is None:
            return False
        # Still viable if it prefixes a real persona name in this room, or
        # the speaker's own name (a leaked self-prefix is stripped whatever
        # its casing, so it cannot rely on the capitalisation check below),
        # or if it could still become a name-shaped token.
        low = s.casefold()
        if self._own.startswith(low):
            return True
        if any(known.startswith(low) for known in self._known):
            return True
        return _capitalised_words(s.rstrip("*_~`"))


# The shapes a speaker tag actually takes at the start of a line. Bare
# "Name:" was the only one covered for a long time, and it is the one
# instruct-tuned models use least: they reach for markdown.
_TAG_FORMS = ("\n{name}:", "\n[{name}]:", "\n**{name}:", "\n**{name}**:")


def stop_sequences(persona_name: str, room_names: Iterable[str], limit: int = 24) -> List[str]:
    """Stop strings for the other personas' speaker prefixes.

    Exact names in exact shapes only. A looser pattern would be worse than
    nothing here: "\\n[" fires on any legitimate bracketed line, and
    "\\n**Name" (no colon) fires on a reply that merely opens a line by
    mentioning someone in bold — both truncate a real answer server-side,
    where the guard cannot see it happen.

    This is the cheap first layer: it stops the model before a token is
    generated, but only for personas that actually exist and only for tags
    at a line start. Invented names, mid-line tags and every other
    decoration are the guard's job.

    Capped at *limit* entries because some backends reject long stop lists.
    The default covers six other voices — a full room plus the human.
    """
    own = persona_name.casefold()
    out: List[str] = []
    for name in room_names:
        if name.casefold() == own:
            continue
        out.extend(form.format(name=name) for form in _TAG_FORMS)
        if len(out) >= limit:
            break
    return out[:limit]
