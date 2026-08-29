# Response control: length and persona containment

This document describes an addition to the TalkWithMe app. It is **pass 1** of the
room-behaviour work; the wider roadmap is parked in
[`feature_room_dynamics.md`](feature_room_dynamics.md).

Two symptoms motivate it, and they turn out to be the same bug:

1. A persona **continues another persona's message**, especially when that message
   stopped mid-sentence.
2. A persona **invents a brand-new character** and writes dialogue as them.

## Root cause

`_base_payload()` in `app/services/llm.py` applies one global clamp to every call:

```python
"max_tokens": settings.llm.max_tokens,
```

Default responses are too long, so the natural fix is to turn `llm.max_tokens` down.
But `max_tokens` is a **guillotine, not a style control** — the model is still trying to
write a long answer and simply gets cut off mid-sentence. That truncated line then lands
in history, and `session.build_llm_messages()` hands it to the next persona as a `user`
turn:

```
[Alex]: I think the real problem with that argument is that it assumes
```

An unfinished sentence in the prompt is a completion cue. The next model dutifully
completes it — in Alex's voice, not its own. Lowering `max_tokens` to shorten replies is
what *causes* the continuation behaviour.

The invented-character symptom shares a cause: nothing in the prompt tells the model who
exists. It sees `[Name]: ` prefixes in the history, infers "this is a transcript", and
transcripts are a format models happily extend with new speakers.

So: **stop using `max_tokens` to shape length, and tell the model the rules of the room.**
Then add a mechanical backstop for when it ignores them anyway.

## Desired state

Three parts, in dependency order.

---

## 1. Typical response length

### Config

A tier, not a token count. Tokens are the wrong unit to expose — they are what caused
the problem — and a tier maps cleanly onto both a prompt phrasing and a safety cap.

```python
class TypicalLength(str, Enum):
    TERSE        = "terse"         # ~25 words  — one or two short sentences
    BRIEF        = "brief"         # ~60 words  — two to four sentences
    NORMAL       = "normal"        # ~120 words — a short paragraph
    DETAILED     = "detailed"      # ~250 words — a few paragraphs
    UNRESTRICTED = "unrestricted"  # no guidance, no derived cap
```

The word targets and their prompt phrasings live in one table in `app/config.py`, so
prompt text and token maths never drift apart.

Placement, per the three-level precedence **persona → room → global**:

- `Persona.typical_length: Optional[TypicalLength] = None` — optional, `None` inherits.
  A terse persona stays terse in a room of ramblers.
- `ChatRoom.typical_length: TypicalLength = NORMAL` — the room's house style.
- `GeneralConfig.typical_length: TypicalLength = NORMAL` — the fallback, and the only
  value the implicit `"default"` room can use, since it has no `chatrooms.yaml` entry.

### The soft control (this is the one that does the work)

The resolved tier becomes a line in the system message:

```
Aim for about {phrasing} (~{words} words). Go longer only when the question
genuinely needs it. Always finish your sentence — never stop mid-thought.
```

"Go longer when it genuinely needs it" is the explicit escape hatch: typical is a
target, not a ceiling. "Always finish your sentence" is cheap and measurably reduces
mid-thought stops.

For `UNRESTRICTED` the line is omitted entirely.

### The hard control becomes a safety net

`_base_payload()` gains an optional `max_tokens` override, and the chat flow passes a
value derived from the tier:

```python
derived = ceil(words * TOKENS_PER_WORD * HEADROOM)      # ~1.4 and 3.0
effective = min(settings.llm.max_tokens, max(derived, 256))
```

The headroom is deliberate and generous. At `brief` (60 words) the cap lands near 256
tokens — roughly **four times** the target — so a persona that genuinely needs more room
has it, and the cap only fires on a runaway. `UNRESTRICTED` uses the global value
unchanged.

`settings.llm.max_tokens` keeps its meaning as the absolute ceiling: it can only lower
the derived value, never raise it. Which means the migration advice for anyone currently
running a hand-lowered `max_tokens` is: **put it back up to 1024 and pick a tier
instead.** That needs to be said plainly in the README, because it inverts the tuning
people have already done.

---

## 2. Room rules in the prompt

The smallest slice of room-awareness that the two symptoms require — a roster (you
cannot say "don't invent people" without saying who exists) plus three prohibitions.
Appended to the persona's `system_prompt`, separated by a blank line:

```
You are {name}, in a group chat called "{room}".
The only people here are: Luna, Sam, and the user. There is nobody else.

Lines from other people appear as "[Name]: text". Lines with no prefix are the user.

- Write only as {name}. Never write a line, a reply, or a name prefix for anyone else.
- Never invent a new character or speak as one.
- Never continue, complete, or rewrite someone else's message, even if it looks
  cut off. If a message ends mid-sentence, just respond to it as it stands.
- Do not begin your reply with your own name.
{length line from §1}
```

`build_llm_messages()` in `app/session.py` gains one optional parameter,
`room_preamble: Optional[str] = None`, appended to the system message when present.
The preamble is *built* in `app/routers/chat.py`, not in `SessionManager` —
`SessionManager` currently imports no config and no router, and that separation is worth
keeping.

The `[Name]: ` remapping itself does not change. The preamble explains a convention the
code has always applied silently.

---

## 3. The containment guard

Prompt rules reduce these behaviours; they do not eliminate them. A mechanical backstop
runs on every reply, in three layers, cheapest first.

### Layer 1 — stop sequences

`_base_payload()` gains an optional `stop` list. For a persona reply it carries the
other room personas' names in speaker-prefix form:

```python
["\nLuna:", "\n[Luna]:", "\nSam:", "\n[Sam]:"]
```

Exact names only — never a generic pattern like `"\n["`, which would break legitimate
bracketed text at the start of a line. This catches the common case (continuing as an
*existing* persona) for free, server-side, before a token is generated. It cannot catch
an *invented* name, which is what layer 2 is for. Cap the list at 8 entries; some
backends limit how many stop strings they accept.

### Layer 2 — a streaming transcript guard

A small, pure, separately testable filter in a new `app/services/reply_guard.py`, sitting
between the LLM stream and the SSE emitter. It owns two jobs:

**Strip the speaker's own prefix.** Models leak `Alex:` or `[Alex]:` often enough that
the prompt rule alone is not sufficient.

**Cut at a foreign speaker prefix.** At the start of any line, match
`^\s*\[?([\w .'-]{1,25})\]?\s*:` — if the captured name is not the responding persona,
the reply ends there. Everything from that point on is discarded and the upstream LLM
stream is closed.

The mechanic that makes this work with streaming: **hold back output at line starts.**
After each newline the guard buffers until it has enough characters to decide — either a
`:` appears within the name-length budget, or a character that cannot appear in a name
rules the pattern out. Then it flushes. A prefix straddling two chunks is therefore still
caught, and the delay only ever occurs at a line break, which is a natural pause.

The same buffering handles the self-prefix case, which **must** be resolved before the
first token is emitted rather than at the end. `static/chat.js` appends each `token`
event straight into the bubble (`bubble.textContent += event.token`) and the `done`
event's `text` is consumed *only* by TTS — it never re-renders. Stripping late would
leave the prefix on screen until reload and disagree with what is on disk. Guarding at
the head means the streamed text, the `done` text, and the persisted text are identical
by construction, and no frontend change is needed.

### Layer 3 — truncation is marked, not hidden

`_iter_completion_chunks()` already surfaces `finish_reason`, but `stream_chat()`
discards it — today the code cannot tell a truncated reply from a complete one.

**Proposed change:** `stream_chat()` yields event dicts (`{"type": "token", ...}` and a
final `{"type": "finish", "reason": ...}`) instead of bare strings, matching the shape
`stream_chat_with_tools()` already uses. The two branches in `_chat_stream()` then
converge on one uniform stream that the guard wraps once. This is a small breaking change
to `stream_chat`'s contract; `tests/test_llm.py` needs its `_collect()` helper updated,
and `tests/test_chat_sse.py`'s fake streams need to yield dicts.

When `finish_reason == "length"`, `ChatMessage` records `truncated: bool = False` as
`True`. That flag changes only what **later personas** see — never what is displayed or
persisted as text:

- If the text contains a sentence terminator, the version shown to later personas is
  trimmed back to it, so no dangling fragment reaches the next model.
- If it does not (one long unfinished sentence), the text is kept and rendered as
  `[Alex]: {text} (message was cut off)`.

Combined with the §2 rule about cut-off messages, the continuation cue is removed at the
source rather than merely discouraged.

---

## Persistence

`personas.yaml` and `chatrooms.yaml` each gain one optional key:

```yaml
# personas.yaml
- name: Alex
  typical_length: terse        # omit or null to inherit the room

# chatrooms.yaml
chat_rooms:
  - name: debate-club
    persona_names: [Alex, Luna]
    typical_length: brief
```

`settings.yaml` gains `general.typical_length: normal`.

Backward compatibility is free: absent keys mean `None` (persona) or `normal` (room and
global), and `normal` is close to today's effective behaviour at the default
`max_tokens: 1024`. No migration, no version key.

`truncated` lives on the in-memory `ChatMessage` only. It is deliberately **not** added
to the on-disk history format — it is a property of one generation, not of the message,
and `app/persistence.py` and `chatrooms/<room>/history.json` stay untouched.

## UI

- **Persona editor** (`static/persona.js`): a "Typical response length" select, with
  `Use room default` as the first option, mapping to `null`.
- **Chat room editor** (`static/chatrooms.js`): the same select per room row, without the
  inherit option — a room always has a value.
- **General settings** (`static/gen-settings.js`): the global fallback select.
- **Servers modal** (`static/settings.js`): help text under Max Tokens explaining it is
  now a ceiling, and that length is controlled by the tier.

The `general` section of `PUT /api/settings` is a partial update — `typical_length` must
be `Optional` and default to `None` in `GeneralSettingsRequest`, or the Servers dialog
will reset it on every save. That is exactly how `show_tool_calls` broke before.

## API

No new endpoints. `typical_length` rides along on the existing persona and chat room
payloads (`PersonaCreateRequest`, `PersonaDetailResponse`, `ChatRoomResponse`,
`GeneralSettingsRequest`/`Response`), so `tests/test_docs.py` stays green with no
AGENTS.md table change.

## Testing

All offline, consistent with the existing suite:

- `tests/test_config.py` — tier defaults; persona `None` inherits; absent keys in both
  YAML files round-trip; the derived-cap maths at each tier, including the 256 floor and
  the `llm.max_tokens` ceiling.
- **`tests/test_reply_guard.py`** (new) — the guard as a pure unit, fed token lists:
  self-prefix stripped in both `Name:` and `[Name]:` forms; a foreign prefix cuts the
  reply; a prefix split across chunk boundaries is still caught; a colon in ordinary
  prose (`"the answer is: yes"`) does **not** trigger a cut; a name-like word
  mid-sentence does not; unmatched buffered text is always flushed at end of stream.
- `tests/test_llm.py` — `max_tokens` and `stop` overrides reach the payload; the global
  value is not overridden when no tier applies; `finish` events carry `finish_reason`.
- `tests/test_chat_sse.py` — the preamble reaches the system message and names the right
  roster; the length line matches the resolved tier; persona overrides room overrides
  global; a truncated reply is marked and trimmed for the next persona.
- `tests/test_session_manager.py` — `room_preamble` lands in the system message and is
  omitted when `None`; `truncated` affects only the other-persona rendering.

## Documentation

- **`README.md`** — a "Response length" section covering the tiers and the precedence,
  and the migration note that `llm.max_tokens` should go back up.
- **`AGENTS.md`** — the tier precedence and derived-cap formula; the guard's three layers
  and where `reply_guard.py` sits; `stream_chat()`'s new event-dict contract; the
  `room_preamble` parameter in the Architecture session bullet.

## Out of scope

Everything else from `feature_room_dynamics.md` — per-room reply counts, @mentions,
follow-up routing, personas talking to each other unprompted. Also out: per-persona
`stop` sequences beyond room persona names, and exposing `TOKENS_PER_WORD` / `HEADROOM`
as settings. They are constants until there is evidence they need to be knobs.
