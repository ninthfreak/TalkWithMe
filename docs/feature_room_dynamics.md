# Room dynamics

This document describes an addition to the TalkWithMe app.

Chat rooms today are a *filing system*: a named subset of personas plus a persistence
directory. This feature makes a room behave like an actual room — personas know who
else is in it, can be addressed directly, can decline to speak, and can carry on
talking to each other after answering you.

Nothing here changes the meaning of an existing `chatrooms.yaml`. Every new knob is
optional and defaults to today's behaviour.

## Current state

`POST /api/chat` (`app/routers/chat.py`) does the following for one user message:

1. `_resolve_room_personas()` returns the eligible persona names for the room.
2. `_pick_persona()` chooses the **first** speaker: `router` (an LLM call scored on
   each persona's `router_hints`), `random`, or an explicit name from the UI.
3. It then loops `general.max_persona_replies` times (1–4, global). Speakers 2..N are
   chosen by `random.choice()` over whoever has not spoken yet.
4. Each speaker streams a reply built by `session.build_llm_messages()`, which prepends
   that persona's `system_prompt` and remaps other personas' lines to `user` turns with
   a `[Name]: ` prefix.

The gaps this creates:

- **No addressing.** `detectMentionedPersona()` in `static/chat.js` does a bare-name
  scan and flips the "Selected persona" radio, but it is frontend-only, returns the
  *first* match, and has no `@` syntax. The server never sees a mention.
- **Speakers 2..N are noise.** A uniform random pick has no idea whether the persona
  has anything to say about the topic — the router's judgement is used once and then
  thrown away.
- **Nobody can stay quiet.** Every selected persona must produce a reply.
- **Personas don't know they're in a group.** A persona's system prompt says nothing
  about the room or its occupants; the `[Name]: ` prefixes arrive unexplained, so
  models frequently either ignore them or start prefixing their *own* output with
  `[Name]: `.
- **Conversation stops at the user.** Personas never take a turn unprompted.
- **All tuning is global.** `max_persona_replies` and `max_turns_for_context` live in
  `settings.yaml`, so a two-persona focused room and a six-persona debate room must
  share one setting.

## Desired state

Four changes, in dependency order. (3) and (4) both build on the speaker-selection
rewrite in (2), and all three read the per-room config from (1).

---

## 1. Per-room settings

### Config

`ChatRoom` in `app/config.py` gains an optional settings block. Fields that mirror a
global setting default to `None`, meaning *inherit the value from `settings.yaml`* —
the same sentinel-`None` merge contract `GeneralSettingsRequest` already uses.

```python
class RoomSettings(BaseModel):
    # Inheritable — None means "use the global value"
    max_persona_replies: Optional[int] = Field(default=None, ge=1, le=4)
    max_turns_for_context: Optional[int] = Field(default=None, ge=1, le=50)
    default_who_answers: Optional[str] = None      # "router" | "random" | persona name

    # Room-only — no global equivalent
    topic: Optional[str] = Field(default=None, max_length=512)
    roster_awareness: bool = True
    allow_silence: bool = True
    conversation_depth: int = Field(default=0, ge=0, le=3)


class ChatRoom(BaseModel):
    name: str
    persona_names: List[str] = Field(default_factory=list)
    echo_chamber: bool = False
    settings: RoomSettings = Field(default_factory=RoomSettings)
```

A new helper in `app/routers/chat.py` does the merge once per request:

```python
def _resolve_room_config(chat_room: str) -> ResolvedRoomConfig
```

It returns a plain object with every value already resolved (room override if set,
global otherwise), so the rest of the chat flow never reads `get_settings()` directly
for these four fields. One resolution point, not scattered `or` expressions.

### The "default" room

`"default"` is synthesized on the fly by `app/routers/chatrooms.py` and has no
`chatrooms.yaml` entry, so it has nowhere to store overrides. **The default room always
uses the global settings**, exactly as today. `PUT /api/chatrooms/default/settings`
returns 400, matching how the existing persona-assignment and echo-chamber endpoints
already reject it.

### API

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/api/chatrooms/{name}/settings` | Replace a room's settings block |

The existing `PUT /api/chatrooms/{name}/echo-chamber` stays as-is. `echo_chamber` is
deliberately *not* folded into `RoomSettings`: it is a mode switch, not a tuning knob,
and it already has an endpoint and a UI toggle.

Note `tests/test_docs.py` asserts the AGENTS.md API table matches the registered
routes — the new row must land in the same commit as the route.

### UI

The chat room editor (`static/chatrooms.js`, the `#chatrooms-overlay` modal) currently
shows one row per room with a delete button and an echo chamber checkbox. Each row gains
a disclosure triangle that expands a settings panel:

- **Replies per message** — select: `Use global (N)` / 1 / 2 / 3 / 4
- **Context turns** — select: `Use global (N)` / 2 / 4 / 6 / 10 / 20
- **Who answers by default** — select: `Use global` / LLM decides / Surprise me / *persona names in this room*
- **Room topic** — textarea, 512 chars, with the help text from §2
- **Personas know each other** — checkbox (`roster_awareness`)
- **Personas may stay quiet** — checkbox (`allow_silence`)
- **Keep talking for** — select: `0 (off)` / 1 / 2 / 3 rounds (`conversation_depth`)

When `echo_chamber` is checked, the whole panel is disabled and greyed with the note
"Echo chamber ignores these settings" — see the interaction rules in §4.

`default_who_answers` seeds the left panel's "Who should answer?" radio on room switch.
It is a default, not a lock: the user can still change the radio for the current turn.

---

## 2. Room awareness in prompts

### The preamble

When `roster_awareness` is on (default), the system message for each reply becomes the
persona's `system_prompt`, then a blank line, then an app-generated preamble:

```
You are {name}, in a group chat room called "{room}".
{topic, if set}
Also here: Luna (a philosophical poet), Sam (a grumpy sysadmin).
Lines from other people appear as "[Name]: text". Lines with no prefix are from the user.
Reply only as {name}. Never prefix your reply with your own name or with "[{name}]:".
Address the others by name when you are responding to something they said.
```

The roster is built from the room's eligible personas (excluding the speaker) using each
persona's existing `description` field, which is exactly what it is for and is already
capped at 30 chars. A persona with an empty description contributes just its name.

`session.build_llm_messages()` gains one optional parameter:

```python
def build_llm_messages(self, system_prompt, responding_persona,
                       max_turns_for_context=None, room_preamble=None)
```

Keeping the preamble a *parameter* rather than building it inside `SessionManager` keeps
the session ignorant of chat room config — it currently imports neither `config` nor any
router, and that separation is worth preserving. `app/routers/chat.py` owns
`_build_room_preamble()`.

The `[Name]: ` remapping itself does not change. The preamble simply explains to the
model a convention the code has always used silently.

### Self-prefix stripping

Models leak their own name prefix often enough that the instruction alone is not
sufficient. The reply must be stripped of a leading `{name}:` or `[{name}]:`
(case-insensitive, optional surrounding whitespace).

This cannot be done on the accumulated text at the end of the stream. The prefix
arrives split across several tokens, `chat.js` appends each `token` event straight into
the bubble (`bubble.textContent += event.token`), and the `done` event's `text` field is
consumed **only** by TTS — it never re-renders the bubble. Stripping late would leave
the prefix visible until the next page load, and the live bubble would disagree with
what is on disk.

So the strip happens **before the first token is emitted**, with a small prefix guard at
the head of each reply: hold tokens back until either the accumulated text exceeds the
longest possible prefix for this persona (`len("[{name}]: ")` plus a few characters of
slack) or a `:` is seen inside that budget. Then decide once — drop the prefix or not —
flush the buffer as a single `token` event, and stream normally from there. The guard is
per-reply and costs at most a few tokens of latency on the first chunk.

Because the guard runs on the server, the streamed text, the `done` text, and the
persisted text are identical by construction, and `static/chat.js` needs no change.

### Room topic

Free text, injected verbatim as the second line of the preamble. It is a *scene*, not a
system prompt override: "This is a code review channel. Be terse and cite line numbers."
Empty by default, in which case the line is omitted entirely.

---

## 3. Mentions and turn-taking

### Mention parsing moves to the server

A new `_detect_mentions(text, eligible) -> list[str]` in `app/routers/chat.py`:

- Matches `@Name` always, and bare `Name` only when `general.persona_name_mentions` is
  on — preserving what that flag means today for the frontend, and extending it to the
  server.
- Returns **every** mentioned persona in order of first appearance, deduplicated. The
  current frontend helper returns only the first match, which silently drops the second
  half of "Alex and Luna, what do you both think?".
- Candidate names are tried **longest first**, so a room containing both `Smith` and
  `Dr. Smith` resolves `@Dr. Smith` correctly. Word-boundary and flexible-whitespace
  matching carry over from the existing `detectMentionedPersona()`.
- `@all` and `@everyone` are reserved: they expand to every eligible persona in
  `persona_names` order. A persona actually named "all" shadows the keyword — an
  explicit name always beats a keyword.

`static/chat.js` keeps `detectMentionedPersona()` **only** to pick the avatar/name shown
in the placeholder bubble before the first `start` event arrives. The server's decision
is authoritative and the placeholder is corrected by `start` as it already is today.

### Speaker order

For one user message, the speaker list is resolved in this precedence:

1. **Mentioned personas**, in mention order. An @mentioned persona always speaks, and
   always speaks first; a mention overrides the "Who should answer?" radio.
2. If there were no mentions and `who_answers` is an explicit persona name, that persona.
3. Otherwise the selection mode (`router` / `random`) picks the first speaker, as today.
4. **Remaining slots** up to `max_persona_replies` are filled by the follow-up router
   (below) instead of `random.choice()`.

`@all` in a room of six personas with `max_persona_replies: 2` yields two speakers, not
six — the reply cap is a hard ceiling, and mentions fill it from the front. This is worth
surfacing in the UI help text next to **Replies per message**.

### The follow-up router

A second, smaller router prompt — distinct from `_build_router_prompt()`, which is
framed entirely around "who should answer *the user*":

> The conversation so far: {last N turns, including personas already in this reply}
> Who should speak next? Choose from: {personas who have not spoken this turn}.
> Reply with only a name, or NOBODY if the conversation is complete.

Same discipline as the existing router: non-streaming `chat_completion()`,
`max_tokens=16`, validate the returned name against the eligible set, and fall back to
`random.choice()` on an unparseable answer or an LLM failure. `NOBODY` is only honoured
when `allow_silence` is on; otherwise it is treated as an invalid answer and falls back
to random, which reproduces today's behaviour exactly.

**Cost.** This is one extra non-streaming LLM call per additional speaker. With
`max_persona_replies: 4` and `conversation_depth: 3` a single user message can cost up
to six router calls on top of seven streamed replies. The calls are small (16 tokens
out) but not free, and this is the main reason `conversation_depth` defaults to `0`.
An alternative — one call returning the whole speaker list up front — was rejected
because the right *second* speaker genuinely depends on what the first one said.

### Staying quiet

When `allow_silence` is on, a persona can decline in two ways:

- The follow-up router returns `NOBODY` → stop the loop, emit `complete`.
- A persona's reply comes back empty or is exactly `[pass]` (the preamble tells them
  this is available when `allow_silence` is on) → the reply is discarded.

A discarded reply is **not** persisted and does not count against `max_persona_replies`,
but it does count against a total-attempts ceiling so a room of reticent personas cannot
spin. A persona who was explicitly @mentioned may **not** pass — being addressed directly
is a commitment.

Because `start` is emitted before the first token, the frontend has already drawn a
bubble by the time we know a persona passed. A passing persona emits no tokens at all —
the prefix guard above swallows `[pass]` before anything reaches the client — so the row
is left empty, and `handleSSEEvent()`'s existing `existingRowSpent` check already reuses
an empty row for the next speaker. The only case that needs handling is a pass by the
*last* speaker of a turn, which would strand an empty bubble.

A new SSE event covers it explicitly rather than relying on that emergent behaviour:

```
{"type": "skipped", "persona": "Luna", "message_id": "...", "reason": "pass"}
```

`handleSSEEvent()` removes the row carrying that `message_id` (it is already stamped on
the row as `dataset.messageId` at `start`) and clears `currentAssistantRow` if it was the
one removed. Any TTS items enqueued under that ID are cancelled — `tts.js` stamps the
assistant message ID onto each item at enqueue time, so cancelling by ID is exact. In
practice the queue is empty for a passing persona; the cancel is a correctness guard,
not the common path.

---

## 4. Personas talking to each other

### Ambient rounds

After the reply cycle to the user message finishes, run `conversation_depth` additional
rounds. Each round picks exactly one speaker via the follow-up router (no repeat of the
immediately preceding speaker) and streams a reply whose prompt is appended with:

> Continue the conversation. React to what was just said, and address people by name.
> The user has not said anything new — do not answer them directly unless it is natural.
> Keep it to a couple of sentences.

Ambient replies are ordinary assistant messages: persisted, TTS-eligible, and visible in
history exactly like any other. They are not a separate class of message on disk, which
keeps `app/persistence.py` and the history format untouched.

`start` and `done` gain a `round` field (`0` for replies to the user, `1..N` for ambient
rounds) so the frontend can render ambient turns with a subtle left border and no
"replying to you" affordance. The field is additive; existing handlers that ignore it
keep working.

### Hard ceilings

The loop is bounded three ways, because a runaway one costs real tokens:

- Total assistant messages for one user turn never exceeds
  `max_persona_replies + conversation_depth`.
- `NOBODY` from the follow-up router ends ambient rounds immediately.
- Client disconnect ends the generator. FastAPI's `StreamingResponse` does not raise
  on disconnect on its own, so the loop must check `await request.is_disconnected()`
  between rounds — which means `_chat_stream()` needs the `Request` object passed in.

### Stopping it from the UI

With `conversation_depth > 0` a single turn can run for a while, and there is currently
**no way to interrupt a chat request** — `sendMessage()` awaits the reader to completion
and the send button is simply disabled. This feature makes that gap user-visible, so it
is in scope:

- `sendMessage()` creates an `AbortController` and passes its signal to `fetch()`.
- The send button becomes a stop button while `isStreaming` is true.
- Aborting closes the response body, which trips the `is_disconnected()` check above and
  ends the generator server-side.
- Messages already completed and persisted stay; the in-flight one is persisted as far
  as it streamed, matching what the user saw.

### Interaction with echo chamber

Echo chamber bypasses the LLM entirely and forces one speaker. When it is on for a room:
`conversation_depth` is forced to `0`, no room preamble is built, no router calls are
made, and mentions only choose *which* persona does the echoing. The editor panel is
disabled to make this visible rather than surprising.

### Interaction with an empty room

Unchanged. A room with no assigned personas still yields "No one is here." — the
frontend short-circuits before sending, and `_chat_stream()` emits the same error event
if a request arrives anyway.

---

## Persistence

`chatrooms.yaml` gains a nested `settings:` block per room:

```yaml
chat_rooms:
  - name: code-review
    persona_names: [Alex, Luna]
    echo_chamber: false
    settings:
      max_persona_replies: 2
      max_turns_for_context: null
      default_who_answers: router
      topic: A code review channel. Be terse and cite line numbers.
      roster_awareness: true
      allow_silence: true
      conversation_depth: 1
```

Backward compatibility is free: a room with no `settings:` key gets
`RoomSettings()`, whose inheritable fields are `None` and whose room-only fields default
to today's behaviour (`conversation_depth: 0`, no topic). Existing files load unchanged
and behave identically. `save_chatrooms()` writes the block for every room, so the first
save after upgrade materialises the defaults — harmless, and it makes the file
self-documenting.

No migration step, no version key.

## Settings that stay global

`persona_name_mentions` and `show_tool_calls` remain global. The first is a parsing rule
that should not vary per room now that the server honours it, and the second is a
display preference. `mcp:` stays global and yaml-only, per the existing contract.

## Testing

New coverage, alongside the existing suite (all offline — no LLM, TTS, or STT calls):

- `tests/test_config.py` — `RoomSettings` defaults; a `chatrooms.yaml` with no
  `settings:` block round-trips; save/load preserves the block.
- `tests/test_routers_chatrooms.py` — `PUT /settings` happy path, validation bounds,
  400 on `default`, 404 on unknown room, `default_who_answers` naming a persona not in
  the room.
- `tests/test_chat_sse.py` — mention parsing (`@Name`, bare name with the flag off,
  multiple mentions, longest-name-first, `@all`, `@all` capped by
  `max_persona_replies`); follow-up router picks and its random fallback; `NOBODY` with
  `allow_silence` on and off; the `skipped` event; a mentioned persona cannot pass;
  ambient rounds emit the right number of messages with correct `round` values; the
  ceiling holds; echo chamber forces depth 0.
- `tests/test_session_manager.py` — `room_preamble` lands in the system message and is
  omitted when `None`; the `[Name]: ` remapping is unaffected.
- `tests/test_docs.py` — passes with the new route documented.

`tests/factories.py` needs a `RoomSettings` factory and a `chat_room` factory that takes
overrides.

## Documentation

- **`README.md`** — a "Room dynamics" section after "Echo chamber", covering mentions,
  the per-room settings panel, and `conversation_depth`, plus a `chatrooms.yaml` example.
  The `max_persona_replies` paragraph gains a pointer to the per-room override.
- **`AGENTS.md`** — the new route row; the `RoomSettings` merge contract next to the
  existing settings-update contract; the speaker-selection precedence and the `skipped`
  and `round` additions in the **Chat flow** section; `room_preamble` in the
  **Architecture** session bullet.

## Out of scope

- **Renaming rooms.** Still delete-and-recreate, per `feature_chat_rooms.md`.
- **Timed / idle chatter.** Ambient rounds are bounded and always attached to a user
  turn. A room that talks to itself on a timer needs a scheduler and a push channel;
  that is a separate feature.
- **Pacing delays between ambient turns.** Tempting for realism, but it holds the SSE
  stream open doing nothing and complicates the disconnect path. Revisit if the rounds
  feel too fast in practice.
- **Per-persona room roles** (moderator, lurker). `router_hints` plus the room topic
  cover most of the intent without new per-room-per-persona state.
- **Multi-user rooms.** The single global `session` singleton is a deliberate design
  choice for this app and is not being revisited.
