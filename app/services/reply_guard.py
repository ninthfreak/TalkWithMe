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

# A speaker prefix at the start of a line: optional brackets around a short
# name, then a colon. The name charset stays narrow on purpose — a wide one
# turns ordinary prose ("the answer is: yes") into a false positive, and a
# false positive silently truncates a legitimate reply.
_PREFIX_RE = re.compile(r"^[ \t]*(\[)?[ \t]*([\w .'\-]{1,25}?)[ \t]*(\])?[ \t]*:")

# Characters a name may contain, for the "is this still a candidate?" check.
_NAME_CHARS_RE = re.compile(r"^[\w .'\-]*\]?$")

# Hard stop on buffering, so nothing can hold forever.
_MAX_HOLD = 32

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
        # Inside a fenced code block nothing is a speaker turn, so a YAML
        # key like "Model:" or an HTTP header must not cut the reply.
        self._in_code = False
        # Text emitted on the current line, used only to spot fence markers.
        self._line = ""

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
                if self._skip_ws:
                    if ch in " \t":
                        continue
                    self._skip_ws = False
                self._emit(out, ch)
                if ch == "\n":
                    self._holding = True
                    self._hold = ""
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
            # The line ended without a colon, so it was never a prefix.
            # Release it and stay at a line start for the next one.
            self._emit(out, self._hold + ch)
            self._hold = ""
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

    def _decide(self) -> tuple:
        """Classify the buffer. Returns (decision, chars consumed by prefix)."""
        if self._in_code:
            # Code, not dialogue. Release it and stop inspecting this line.
            if len(self._hold) >= 3 or not "```".startswith(self._hold.lstrip()):
                return _FLUSH, 0
            return _WAIT, 0

        match = _PREFIX_RE.match(self._hold)
        if match:
            bracketed = bool(match.group(1) and match.group(3))
            name = match.group(2).strip()
            if name.casefold() == self._own:
                return _STRIP, match.end()
            # A bracketed prefix is the format the model sees in history, so
            # it is the overwhelmingly common shape of both failure modes.
            if bracketed:
                self._cut_at = name
                return _CUT, match.end()
            # Unbracketed: another persona in this room, or something that
            # reads like a character name rather than prose.
            if name.casefold() in self._known or _looks_like_a_name(name):
                self._cut_at = name
                return _CUT, match.end()
            return _FLUSH, 0

        if len(self._hold) < _MAX_HOLD and self._still_a_candidate():
            return _WAIT, 0
        return _FLUSH, 0

    def _still_a_candidate(self) -> bool:
        """Could more characters still turn the buffer into a prefix?

        Answering "no" early is what keeps the held text short.
        """
        s = self._hold.lstrip(" \t")
        if s.startswith("["):
            # Bracketed prefixes cut whatever the name looks like, so only
            # the charset and length bound them.
            inner = s[1:]
            return len(inner) <= 26 and _NAME_CHARS_RE.match(inner) is not None
        if len(s) > 25 or _NAME_CHARS_RE.match(s) is None:
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
        return _capitalised_words(s)


def stop_sequences(persona_name: str, room_names: Iterable[str], limit: int = 8) -> List[str]:
    """Stop strings for the other personas' speaker prefixes.

    Exact names only — a generic pattern like "\\n[" would fire on any
    legitimate bracketed line. This is the cheap first layer: it stops the
    model server-side before a token is generated, but only for personas
    that actually exist. Invented names are the guard's job.

    Capped at *limit* entries because some backends reject long stop lists.
    """
    own = persona_name.casefold()
    out: List[str] = []
    for name in room_names:
        if name.casefold() == own:
            continue
        out.append(f"\n{name}:")
        out.append(f"\n[{name}]:")
        if len(out) >= limit:
            break
    return out[:limit]
