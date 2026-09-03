# TalkWithMe — Agent Instructions

## Run the app

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Requires a locally running **llama.cpp** server with an OpenAI-compatible API. TTS and STT servers are optional.
Open `http://localhost:8000` in a browser.

## Testing — pytest, fully offline

```bash
pip install -r requirements-dev.txt
python3 -m pytest        # from the project root; config in pytest.ini
```

- **No servers needed.** The suite is hermetic: every external HTTP endpoint (LLM, TTS, STT, MCP) is faked via `tests/factories.py` (fake `httpx.AsyncClient`s, config factories, SSE helpers). It must run — and pass — with nothing but Python installed.
- **Isolation**: `tests/conftest.py` has an autouse fixture that points every module-level global at per-test `tmp_path` state: the config caches in `app/config.py`, `_PERSISTENCE_ROOT` (in both `app/persistence.py` and `app/routers/persistence.py` — the router imported it *by value*, so it needs its own patch), the `session` singleton, and the MCP tool registry. Real `settings.yaml` / `Personas/` / `chatrooms.yaml` / `chatrooms/` data is never read or written. **If you add a new module-level global to the app, add it to that fixture.**
- **Lifespan**: the `client` fixture uses `TestClient` *without* the startup lifespan, because the lifespan re-reads the real YAML files (clobbering test caches) and attempts MCP discovery. Tests that exercise the lifespan do so explicitly with a local `TestClient` in a `with` block and monkeypatched `load_*`/`load_tools` (see `tests/test_main.py`).
- **Coverage map**: `test_config.py` (config models + YAML load/save), `test_persona_store.py` (persona directory discovery: frontmatter, file ops, language/avatar helpers, memory-file append/purge, YAML→dir migration), `test_builtin.py` (the built-in `add_memory` tool: registry, availability gating, save/error paths), `test_models.py` (API request/response models), `test_persistence.py` (disk persistence + audio staging), `test_session_manager.py`, `test_llm.py` (SSE parsing + agentic tool loop), `test_mcp_client.py`, `test_tool_registry.py`, `test_tts_stt_clients.py`, `test_chat_sse.py` (the `/api/chat` SSE endpoints — LLM stubbed, selection/persistence/speak-as/tool events for real), `test_reply_guard.py` (the persona-containment state machine), `test_config_migrations.py` (old config files keep loading and keep their meaning), `test_main.py`, `test_docs.py` (the AGENTS.md API endpoints table must match the routes actually registered on the app — run it after adding/removing endpoints and update the table), one `test_routers_*.py` per API router, and `test_persona_form.js` (plain Node, **not** part of pytest — see below: the persona editor form logic in `static/persona.js`).
- **Frontend tests (Node)**: `tests/test_persona_form.js` runs with plain Node 20+ — no npm packages, no network: `node tests/test_persona_form.js`. It evaluates the real `static/utils.js` + `state.js` + `persona.js` in a fresh `vm.Context` per test against a minimal DOM stub, stubs `fetch` to capture the multipart `FormData`, and drives the real `openPersonaForm()`, Remove-button listeners, file-input change handlers, and `submitPersonaForm()`. It exists to lock in the invariants that `remove_avatar_image` / `remove_reference_audio` are sent **only** after an explicit "Remove" click — never derived from server-side file presence (that bug silently deleted a persona's avatar and reference audio on every plain text save) — and, by the same rule, that `clear_memories` is sent only after an explicit "Clear saved memories" click while `memory_size` is **always** sent (the update endpoint requires it; an omitted value must not silently reset the persona's memory budget). It is kept out of the pytest suite on purpose: pytest must stay runnable with nothing but Python installed.
- **Rules**:
  - Every code change must be followed by a clean run: `python3 -m pytest`, all green. No exceptions, no skipped tests. Changes to `static/persona.js` (or anything else covered by the Node tests) additionally require `node tests/test_persona_form.js` all green.
  - New functionality or API endpoints require new tests in the matching `test_*.py` file before the change is complete.
  - API tests use the `client` fixture + config caches (re-point `app.config._settings_cache` / `_personas_cache` / `_chatrooms_cache` via `monkeypatch`); router-level stubs are applied at the router's import site (e.g. `app.routers.chat.stream_chat`), since routers import service functions by name.
  - `httpx` version pin matters: with httpx 0.24, `response.url` / `raise_for_status()` raise `RuntimeError` if no `request` is attached to a `Response`, and the `.request` *getter* itself raises when unset (check `response._request` instead). All fake responses in `tests/factories.py` attach a request for this reason.

## Config — YAML files + a persona directory, cached at startup

| Source | Purpose |
|--------|---------|
| `config/settings.yaml` | LLM, TTS, STT endpoints and parameters, general chat parameters, MCP server list. `general.personas_directory` (default `Personas`) says where personas live |
| `Personas/` (one directory per persona) | Persona definitions — the single source of truth. See **Persona storage** below |
| `config/chatrooms.yaml` | Chat room groupings, and each room's settings |
| `config/player.yaml` | Which persona the human is playing |
| `config/personas.yaml` | **Legacy only.** Migrated into `Personas/` once at startup, then renamed `personas.yaml.bak` |

**Two locations, one of them inert.** The app reads and writes the `config/` copies (gitignored). The identically-named files at the **repo root are tracked shipped defaults and are never written to.** Do not "tidy this up" by untracking the root copies: `git rm --cached` on them makes the next `git pull` try to delete files users have edited locally, and git aborts the merge with *"Your local changes to the following files would be overwritten by merge"*. That is the bug this layout exists to prevent.

That invariant is also why the persona migration reads `config/personas.yaml` rather than the repo-root copy upstream uses: the `.bak` rename has to land on a gitignored file. `migrate_config_files()` skips `personas.yaml` entirely once `Personas/` holds anything, or it would copy the root default back into `config/` after every migration and warn about the conflict on every startup.

`config_path()` in `app/config.py` prefers `config/` and falls back to the root copy, so the very first run after an upgrade still sees the user's existing settings. `migrate_config_files()` (called from the `app/main.py` lifespan, before any load) **copies** root → `config/`, upgrading the schema on the way; it never moves or deletes, and never replaces the contents of an existing `config/` file. It then brings any already-present `config/` file whose `schema_version` is behind up to date **on disk** — loading migrates in memory, but nothing rewrote the file until the next save, so a stale key could sit there indefinitely while the app read something else. `save_*()` always writes to `config/`. All path helpers read `_PROJECT_ROOT` at call time so `tests/conftest.py` can repoint them at `tmp_path`.

**Schema versioning.** Every written file starts with `schema_version` (`CONFIG_SCHEMA_VERSION` in `app/config_migrations.py`); files predating it are version 1. `load_*()` runs the raw dict through `migrate_personas` / `migrate_chatrooms` / `migrate_settings` **before Pydantic sees it** — that ordering is load-bearing, since once a model has parsed or dropped a legacy key the information needed to migrate it is gone. Migrations return `(raw, notes)` and the notes are logged, so an upgrade is visible. A file from a *newer* schema loads with a warning rather than failing. To add a migration: bump the version, add a step to the relevant chain, and cover it in `tests/test_config_migrations.py` with a real old-format file.

Migrations translate rather than drop wherever intent can be preserved — a persona's old absolute `typical_length` becomes the equivalent relative `length_bias` (its offset from the old `normal`), so a laconic persona stays laconic instead of being silently flattened to the default.

## Persona storage

Personas live in `Personas/<Name>/` (directory = `sanitize_persona_dirname(name)`; name may differ, e.g. `O'Brien` → `OBrien`). All file I/O lives in `app/services/persona_store.py` (framework-agnostic; routers and config import from there).

Per-persona files:

| File | Content | Notes |
|------|---------|-------|
| `prompt.md` | YAML frontmatter (`description`, `router_hints`, `avatar_color`, `allow_tool_calls`, `memory_size`) + system prompt body | `name` is stored only when it differs from the directory name; parsed/serialized by `parse_frontmatter()` / `build_prompt_md()`. `memory_size` (0–16384 bytes, default 8192) is the persona's memory budget — `0` disables memories; an invalid/out-of-range value warns and falls back to the default. Malformed frontmatter degrades the whole file to the prompt body with a warning — never a crash |
| `memories.txt` | One memory per line (newlines inside a memory are flattened) | Persistent memory for the persona, written by the built-in `add_memory` tool and purged oldest-first when the file exceeds `memory_size`. Absent = no memories yet |
| `language.txt` | Single line: reference-audio language code | Absent → `en` (with warning) |
| `ref.wav` | TTS reference audio | Fixed filename — never user-chosen |
| `ref.txt` | Transcript of `ref.wav` | A persona is TTS-capable only when **both** are present (and non-blank) |
| `image.<ext>` | Avatar (png/jpg/jpeg/gif/webp) | One per persona; the editor's "replace image" uploads a new file and deletes the old one |

Startup decision matrix in `app/config.py::load_personas()` (lazy-imports `persona_store` to dodge a cycle):

| `personas.yaml` | `Personas/` dir | Behaviour |
|-----------------|-----------------|-----------|
| no | no | Create the (empty) dir, warn once; empty persona list |
| no | yes | Scan |
| yes | no | **Migrate** (see below) |
| yes | yes | Skip the YAML with a warning (dir wins); both sources kept on disk |

**Migration** (`persona_store.migrate_from_legacy_yaml()`): converts the old schema to per-persona directories. Each persona gets a `prompt.md` (frontmatter + system prompt) and `language.txt`, and the files referenced by its `avatar_image` / `reference_audio` / `reference_audio_transcript` **paths** are copied in as `image<ext>` / `ref.wav` / `ref.txt`. Success → the YAML is renamed to `personas.yaml.bak` so it never re-migrates. **Fatal** error (malformed YAML, unwritable directory, disk full) → raise `PersonaMigrationError` with the YAML left **untouched** and the partially created `Personas/` directory removed best-effort, so the next startup retries cleanly. **Minor** error (a referenced file missing, unreadable, or the wrong format — e.g. a non-wav `reference_audio`) → logged, that file skipped, migration continues. The YAML is the source of truth until the rename succeeds.

**Directory is never renamed.** A persona *name* change (editor) writes a new `prompt.md` `name:` field but keeps the directory; deleting a persona deletes the directory. `GET /api/personas` returns personas in raw directory/creation order — sorting is a frontend concern (see **Persona list ordering**).

## Architecture

- **Backend**: FastAPI. Entry point: `app/main.py`. Routers in `app/routers/`, external service clients in `app/services/`.
- **Session**: Single global `session` singleton in `app/session.py`. Intentional — this is a single-user app. No auth, no database — the `chatrooms/` directory is the only persistent storage. Tracks `current_room` and persists messages to disk automatically. `session.build_llm_messages()` constructs the per-call LLM payload: the responding persona's system prompt, then history remapped so **every voice except the responder's own is tagged** — the human becomes `user` with a `[user_label]: ` prefix, other personas become `user` with `[Name]: `, and only the responder's own turns stay `assistant` and untagged. Using `user` for other personas avoids consecutive `assistant` messages (which many LLMs reject with 400). Tagging the human is what stops a persona replying *as the user*: leaving them the one untagged speaker made "untagged text in the user role" the model's only example of how the human writes, and a persona answering third or fourth — which has no `assistant` turn of its own in context — copied it. Do not revert the human to an untagged turn. History is windowed by `recent_exchanges()` (also in `app/session.py`) to the last `general.max_turns_for_context` **exchanges**, default 6 — an exchange being one human message plus every persona reply it drew. It was a flat tail slice of that many *messages*, which is a different thing once more than one persona answers: with six personas an exchange is seven messages, so a setting of 6 could not hold even one, and asking a room to guess something then revealing the answer left every persona seeing the reveal with the question already cut away. Slice by exchange, never by message count — and `_build_router_prompt()` in `app/routers/chat.py` uses the same helper so the router cannot see answers whose question is gone. A `_CONTEXT_CHAR_BUDGET` safety valve (12000 chars, ~3k tokens) sheds whole exchanges oldest-first if a wide room overflows, and never drops the human message anchoring the last one; any trimming is logged. `build_llm_messages()` also takes an optional `room_preamble`, appended to the system message (built by the chat router — see **Chat flow**).
- **Persistence**: Per-room JSON + audio files under `chatrooms/<room>/`. Handled by `app/persistence.py` (framework-agnostic) and `app/routers/persistence.py` (audio upload/serving endpoints). Created lazily on first write.
- **MCP tools**: `app/services/mcp_client.py` speaks MCP (JSON-RPC 2.0 over the Streamable HTTP transport, protocol version 2025-03-26) with a *stateless, per-call* session — `initialize` runs on every discovery/call, no persistent sessions. `app/services/tool_registry.py` caches the discovered tools once at startup (called from `app/main.py` lifespan) and maps tool name → server; duplicate tool names across servers: first listed server wins. The cache is not refreshed by `reload_all()` — `mcp:` changes need a full restart. The `mcp:` section of `settings.yaml` is **yaml-only** (no UI, no API field) — `update_settings` in `app/routers/settings.py` copies it over from the current cache, otherwise a UI settings save would wipe it. Personas with `allow_tool_calls` run `stream_chat_with_tools()` in `app/services/llm.py` (agentic loop; the final round is sent without `tools` to force a text answer). Tool rounds are local to the loop — they are NOT persisted to the chat history. MCP server URLs are validated at config load (must start with `http://` or `https://`) so a scheme-less typo fails loudly at startup instead of surfacing as per-request connection timeouts.

- **Built-in tools**: `app/services/builtin.py` registers tools that run *locally* instead of over MCP (currently `add_memory`, which appends to the persona's `memories.txt` under its `memory_size` budget). They are offered on top of the MCP tools to tool-enabled personas, are dispatched before any MCP lookup in `stream_chat_with_tools()`, and never touch the network. Built-in names are **reserved**: `load_tools()` silently skips (with a warning) any MCP server advertising a built-in name. Availability is per persona (`memory_size > 0`) and gated by `general.enable_persona_memories` — see **Persona memories** in Chat flow.
- **Settings update contract**: `PUT /api/settings` treats the `general:` section as a *partial update* — omitted fields (or an absent `general` section) keep their current values; only fields explicitly sent override. `GeneralSettingsRequest` in `app/models.py` uses `Optional` fields defaulting to `None`, and `update_settings` in `app/routers/settings.py` merges via `model_dump(exclude_none=True)`. Dialogs that don't edit general settings (the Servers dialog) must NOT send a `general` section, and the router must NOT rebuild `GeneralConfig` from request-body defaults — doing so is exactly how `show_tool_calls` got reset to `true` on every Servers-dialog save. If you add a new field to `GeneralConfig`, the merge preserves it automatically; no per-field wiring needed. The LLM/TTS/STT sections are full replacements, normalized on save: blank base URLs → `None` (which deactivates the feature via `is_active`) and TTS `seed=0` → `None` (the frontend encodes "no seed" as 0). Changes take effect immediately — no restart.
- **Logging**: `app/main.py` configures logging via `logging.basicConfig(level=INFO, ...)` at import time (a no-op if the root logger already has handlers), because uvicorn's default config leaves the root logger at WARNING, which silently swallows every app `logger.info()` call (this is why per-server MCP discovery lines were invisible while failure `WARNING`s showed). The `httpx` logger is separately pinned to WARNING because it logs one line per HTTP request. Note: `uvicorn --log-level` only affects uvicorn's own loggers, not the app's.
- **Frontend**: Vanilla JS SPA, no bundler. Modules in `static/` communicate via shared globals in `state.js`. See the table below:

| File | Responsibility |
|------|---------------|
| `state.js` | Shared globals (personas, session state, chat room state, TTS/STT flags, audio queue, message IDs) |
| `app.js` | Bootstrap, health checks, event listener setup, session management |
| `chat.js` | Message rendering, SSE stream handling (incl. `tool_call` chips), sending messages, persisted history rendering, audio playback buttons |
| `persistence.js` | History loading, audio upload helpers, audio URL generation |
| `persona.js` | Persona sidebar + editor modal (CRUD, incl. memory size field + "Clear saved memories") |
| `chatrooms.js` | Chat room dropdown, room filtering, room editor, persona picker modal, “Playing as” picker, room switching with history load |
| `settings.js` | Servers modal (LLM/TTS/STT config) |
| `gen-settings.js` | General settings modal (max persona replies, name mentions, context turns, tool-call visibility, persona memories toggle) |
| `tts.js` | TTS synthesis, audio queues, Web Audio playback, audio persistence |
| `stt.js` | Microphone recording, STT proxy, transcript insertion, audio persistence |
| `theme.js` | Theme toggle |
| `utils.js` | Shared helpers (incl. `comparePersonasByName()`, the shared case-insensitive persona-name comparator) |

**Persona list ordering**: Anywhere a list of personas is shown to the user (the sidebar, the persona editor modal, the persona picker modal, and anything added in the future), it must be sorted alphabetically and case-insensitively using `comparePersonasByName()` from `utils.js`. The shared `personas` global is pre-sorted in `loadPersonas()` (`app.js`), but each render function sorts its own input list rather than trusting the caller's order. **Do not** rely on `GET /api/personas` returning any particular order — the backend intentionally returns personas in raw directory/creation order; sorting is a display concern and belongs in the frontend only.

## Chat flow

`POST /api/chat` streams SSE. Every event is a single `data: <JSON>\n\n` line; the `type` field is always present, and the frontend switches on it in `handleSSEEvent()` in `chat.js`. Event types: `start`, `token`, `tool_call`, `done`, `error`, `complete`.
Internally, both `stream_chat()` and `stream_chat_with_tools()` in `app/services/llm.py` yield event dicts (`{"type": "token"|"tool_call"|"finish", ...}`), ending with exactly one `finish` event carrying the backend's `finish_reason`. `stream_chat()` used to yield bare token strings; it was changed so both paths share one shape and so truncation is detectable at all. Both accept `max_tokens` and `stop` overrides.
The request body includes `chat_room` (which room to persist to) and `message_id` (frontend-generated UUID for audio association).

**Persona pool.** `_resolve_room_personas()` in `app/routers/chat.py` is the authoritative source of eligibility: the `"default"` room (or any room not present in `chatrooms.yaml`) includes all personas; a named room is limited to its assigned personas, filtered to those that still exist. It deliberately does NOT read the frontend-maintained `session.active_personas`.

**Persona selection modes:** `"router"` (LLM picks via a non-streaming `chat_completion()` at `temperature=0.1`, `max_tokens=16`; the prompt includes each eligible persona's `router_hints` plus the last `max_turns_for_context` turns of context; an unrecognized name or LLM failure falls back to random), `"random"` (uniform over eligible personas), or an explicit persona name (validated against the room; unknown values fall back to random).

**Multiple replies.** `general.max_persona_replies` (1–6, capped at the number of eligible personas) controls how many personas answer one user message: multiple `start`/`token`/`done` cycles, one per responding persona. The first responder comes from the selection mode; each subsequent one is a random non-repeating pick from the remaining eligible personas.

**Message IDs.** Both `start` and `done` carry `message_id` (server-generated UUID for that persona's assistant message); `start` also carries `user_message_id` (the frontend's ID, echoed back).
`start` carries the assistant ID first **on purpose**: the frontend stamps it onto streaming TTS items at enqueue time, so the ID must be generated *before* the `start` event is emitted — moving it back after the stream reintroduces cross-turn audio misattribution.

**Suggested player message.** `POST /api/chat/suggest` drafts the player's *own* next message — the deliberate inverse of everything the reply guard enforces, and fine because the player asked for it and the draft lands in their input box to edit; nothing is sent or persisted until they press send. `_build_suggestion_prompt()` in `app/routers/chat.py` keeps two jobs apart: the **character description** decides *what* to say (manner, what this character cares about, how they would react) and the **voice sample** — the player's own last 8 messages — decides *how* (vocabulary, sentence shape, length). Both get a labelled section and an explicit instruction; an earlier version gave the sample a section and the description a passing line, so drafts sounded right and behaved like nobody in particular. The whole prompt is second person — the model is the character, not a writer working on their behalf — and it deliberately does **not** reuse `_player_lines()`, which is addressed to personas talking *to* the player and ends "never write their lines for them". It also carries the windowed transcript and the room's persona list, calls `chat_completion()` with the configured sampling temperature rather than the router's 0.1, and runs the draft through `ReplyGuard` pointed the other way (speaker = the player) so a `"[Tony]: "` prefix is stripped and a draft running on into a persona's reply is cut. An empty result is a 503, not a silently blank box.

**Speaking as a persona.** Every persona card in the left panel carries a 🗣 button that opens a dialog; what the player types is posted as that persona's message, word for word, via `POST /api/chat/speak`. No LLM call is made — `_speak_stream()` in `app/routers/chat.py` emits the same `start`/`token`/`done`/`complete` SSE shape a real reply does, so the frontend renders, persists and speaks it with no special casing, and the next persona sees it in history as an ordinary `[Name]: ` turn. The persona has to be in the room, checked server-side against `_resolve_room_personas(..., exclude_adopted=False)`: a line from someone the room has never heard of contradicts the roster the next responder is given, and an unexplained voice in the transcript is exactly what makes a model invent a character. `max_persona_replies` does not apply: the player said one thing as one persona and nobody answers it. You *can* speak as the persona you have adopted — the exclusion below is about answering, and putting your own character's line in is the player writing, not the LLM.

This replaced the **echo chamber**, which got at the same idea sideways: a mode the whole room sat in, bouncing your own message back at you from whichever persona the selection rules happened to pick, and staying on until you remembered to switch it off. Its history is worth knowing because it moved twice: it was a `ChatRoom.echo_chamber` field (schema v4 drops the stored key), then a checkbox in the left panel riding on `ChatRequest.echo`, and is now gone entirely. The rule it left behind still holds: **the left panel holds controls you use *while in* a room, which is not the same as settings belonging *to* the room** — do not put room properties there.

**Response length.** `general.typical_length` and `ChatRoom.typical_length` hold a `TypicalLength` tier (`terse` | `brief` | `normal` | `detailed` | `verbose` | `unrestricted`). The scale is calibrated for **chat, not prose**: `normal` is ~20 words ("a sentence or two") and `verbose` — the longest named tier — is only ~110 words. An earlier calibration put `normal` at a paragraph, which made every room read like an essay thread; if you change `TYPICAL_LENGTH_SPECS`, keep it chat-shaped.

Personas do **not** carry an absolute tier. `Persona.length_bias` is a `LengthBias` (`much_shorter` … `much_longer`, default `match`) that moves them ±1 or ±2 steps along `LENGTH_SCALE` from the room's tier, clamped at both ends; `resolve_typical_length()` in `app/config.py` does this. Relative rather than absolute on purpose: a laconic persona should be laconic *for the room they are in*, not a fixed word count that fights the room. An `unrestricted` room ignores bias — with no target there is nothing to be relative to — and `LENGTH_SCALE` deliberately excludes `unrestricted` for the same reason. The implicit `"default"` room has no `chatrooms.yaml` entry, so its base tier is the global value.

**The tier shapes length through the prompt, not through `max_tokens`** — this is the whole point. `derive_max_tokens()` turns the tier's word target into a per-call cap (`words * 1.4 * 3.0`, floored at 256, never above `settings.llm.max_tokens`), which is a runaway guard only. At chat lengths every tier below `verbose` lands on the 256 floor, and that is intended: the prompt does the shaping. Clamping `llm.max_tokens` down to shorten replies is what *caused* personas to be cut off mid-sentence and then continued by the next persona — do not reintroduce it as a style control.

Legacy `typical_length` keys on a persona are dropped with an `INFO` log by `load_personas()` (they cannot be mapped to a relative bias without knowing the room).

**Room preamble.** `_build_room_preamble()` in `app/routers/chat.py` appends a generated block to the persona's `system_prompt`: the room name, the roster of other eligible personas (name + description), an explanation of the `[Name]: ` history convention, the length line, and four prohibitions (write only as yourself; never invent a character; never continue someone else's message; no self name prefix). The roster is what makes "don't invent a character" enforceable. It is passed to `session.build_llm_messages(room_preamble=...)`; `SessionManager` builds none of it, because it deliberately imports no config.

**Playing as a persona.** The player adopts one of the configured personas rather than writing a character of their own: the personas already carry a name, a description and a system prompt, so a second, parallel way to describe a character was redundant and could drift from them. `config/player.yaml` (`PlayerConfig`) holds a single `persona_name`; empty means "playing as themselves". **It is not room data.** A room can *require* that the player is someone (`ChatRoom.require_player_persona`, a property of the room, in the room editor); who they are playing belongs to the player and applies in whichever room they are in.

`PlayerConfig.adopted(known_names)` resolves the stored name against the **live** persona list on every read, and every caller goes through it. A persona can be deleted or renamed after it was adopted, and a dangling reference must degrade to "playing as themselves" rather than half-applying — the name reaches the preamble, the `[Name]: ` history tags, the `stop` strings and the reply guard, and one that exists in none of them is worse than none at all.

The adopted persona is **excluded from answering**: `_resolve_room_personas()` drops it, because a persona replying as itself while the player is also playing it means talking to yourself. `_adopted_persona()` feeds `_player_lines()`, which describes the player to the others using that persona's own `description` and `system_prompt` — one description of a character, not two.

`_chat_stream()` emits `PERSONA_REQUIRED_MESSAGE` and stops **before** recording the user message when a room requires a character and none is adopted, so nothing half-happens. `_user_label()` in `app/routers/chat.py` takes the name from here and is the single source for it: the preamble, the history tags, the stop strings and the reply guard must all agree.

**History.** This is the third shape of the idea. It was `ChatRoom.player_profile`, a free-text name/description/appearance stored per room (schema v5 drops the room key); then a written profile in `player.yaml`; schema v6 drops the written profile entirely and renames the room flag `require_player_profile` → `require_player_persona`. Nothing maps a written profile onto a persona, so the old value is discarded — but visibly, with the character's name in the log, because "my character vanished" with no explanation is worse than a line saying so.

**Room settings are one endpoint.** `PUT /api/chatrooms/{name}` takes a `ChatRoomUpdateRequest`: every field optional, omitted fields unchanged — the same partial-update contract as `general` settings. It replaced four single-purpose routes (echo-chamber, typical-length, player-profile, require-player-profile) that were already drifting; **add a new room attribute as a field here and a control in the room editor, not as another route**. `extra="forbid"` is set deliberately: on a partial-update endpoint a mistyped key would otherwise look exactly like a successful save. Membership keeps its own endpoints because assigning a persona validates against the persona list and can 422. `_editable_room_or_400()` and `_save_room_update()` back it; the latter uses `model_copy`, never a rebuilt `ChatRoom` — rebuilding field-by-field is how room settings used to get silently dropped (see **Persona CRUD cascades**).

**Frontend room settings** all write through `saveRoomSettings()` in `static/chatrooms.js`, which sends the partial update and merges the response into the cached `allChatRooms` entry. The room editor dialog (`#room-edit-overlay`, opened from Edit in the chat rooms list) is the canonical place to configure a room; the left panel holds no room settings at all — its "Playing as…" button opens the persona picker, which is the player's choice, not the room's.

**Speaking as the user.** `_user_label()` in `app/routers/chat.py` is the single source for what the human is called: the adopted persona's name if there is one, else `"User"`. It feeds the preamble, the history tags, the `stop` strings and `ReplyGuard`'s known names — they must agree, or a persona gets told not to speak as "Kira" while the transcript tags them "User". The human is passed to the guard and the stop list as just another speaker, because a persona answering as the user is the same failure as answering as another persona. The preamble states the tagging convention, forbids speaking or answering as the user by name, and ends by naming whose turn it is (a late responder has no assistant turn in context to anchor its own voice).

**Stream teardown.** `stream_chat()` and `stream_chat_with_tools()` hold a handle on `_iter_completion_chunks()` and `await`-close it in a `finally`. This is not tidiness: `async for` does **not** close an async generator when the loop exits, so a caller stopping early (the reply guard cutting at a speaker prefix) left the inner generator suspended inside `async with httpx.AsyncClient(...)` with the HTTP response open, its cleanup deferred to the event loop's asyncgen finalisation hook. llama.cpp serialises on its slots, so the abandoned request kept one and the *next* persona's call queued behind it — the chat stalled right after a "produced no reply of its own" log line. `tests/test_llm.py::TestEarlyCloseReleasesTheConnection` pins it.

**Reply slots.** `_chat_stream()` counts attempts per persona (`attempts`) separately from `replied_personas`. A persona whose reply the guard cuts to nothing has been tried but has not replied, so it does not consume one of `max_persona_replies`. Untried personas are preferred; a cut persona is re-rolled only once nobody fresh is left, which is what makes the requested count reachable in a room whose size *equals* `max_persona_replies` (a single cut would otherwise put it out of reach). A persona that actually replied is never asked again in the same turn, and the whole loop is bounded by `max_replies + MAX_CUT_RETRIES` attempts. If nobody produces a usable reply the turn emits `NO_USABLE_REPLY_MESSAGE` before `complete` — a turn ending with a `start` event and nothing after it is indistinguishable from a hang.

**`max_persona_replies` is capped by room size** (`min(setting, len(eligible))`) — the setting is global, the room may be smaller. That clamp is logged at INFO, because "I set 6 and got 4" is otherwise a mystery with no visible cause.

**Empty replies are dropped.** If the guard cuts a reply to nothing — the persona opened by writing as someone else and produced nothing of its own — `_chat_stream()` logs it and `continue`s without persisting or emitting `done`. An empty assistant turn would render as a blank bubble and feed the next persona a meaningless `[Name]: ` line. The frontend's `existingRowSpent` check reuses the untouched row for whoever speaks next.

**Reply guard.** `app/services/reply_guard.py` is a pure state machine (no config, no I/O) filtering each reply, in three layers. (1) `stop_sequences()` passes the other speakers' tags as `stop` strings, in the four shapes models actually write — `\nName:`, `\n[Name]:`, `\n**Name:`, `\n**Name**:`. Exact names in exact shapes only: a generic `"\n["` fires on any legitimate bracketed line, and `"\n**Name"` without the colon fires on a reply that merely opens a line by mentioning someone in bold — both truncate server-side where the guard cannot see it happen. The cap (24) covers five other personas plus the human. (2) `ReplyGuard` strips the speaker's own tag and cuts the reply at a *foreign* one. (3) Truncation is marked from `finish_reason == "length"`.

**What counts as a tag.** Matching only a bare `Name:` was the guard's biggest hole, and it is the shape instruct-tuned models use least — they reach for markdown. A live room produced one reply carrying three other personas' turns, every tag written `**Luna:**`, none of them cut, while those three personas also answered for themselves. A tag is now optional list/markdown decoration (`**`, `-`, `>`, `### `, `1. `), optional brackets, the name, optional emphasis, then a colon or dash — or, for a **known** name, brackets or decoration alone (`### Luna`, `[Luna]`). Known and invented names are treated asymmetrically on purpose: a name in the roster is *known* to be a speaker so it cuts anywhere, including mid-line after a sentence end ("I agree. Luna: but…"); an unknown name only cuts at a line start, where a tag is unambiguous. Mid-line needs a colon or brackets — a dash there is punctuation more often than a turn. Every widening is bounded by something that makes the text a turn rather than prose, because **a false positive silently truncates a legitimate reply, which is worse than missing a cut**: `_PROSE_LEAD_INS` (checked against the whole phrase *and* its last word, so `Final Answer:` is covered) and fenced-code tracking exist for exactly that reason.

The guard **buffers at line starts** while deciding — and briefly after a sentence end, but only while the buffer could still spell a known name, so ordinary prose releases after a character or two. This is why the head of a line can arrive coalesced into one `token` event; chunk boundaries are not part of the SSE contract. It must resolve tags before emitting, because `chat.js` appends each token straight into the bubble and never re-renders from `done`.

**Truncation.** `ChatMessage.truncated` is set when the LLM stopped at `max_tokens`. It changes only what *other* personas see: `_other_persona_text()` in `app/session.py` trims a truncated message back to its last complete sentence, or labels it `(message was cut off)` when it has none. It never changes displayed or persisted text, is excluded from `get_history_dicts()`, and is not written to disk — it is a property of one generation, not of the message.

**Tool calls.** `tool_call` is only emitted while a tool-enabled persona's agentic loop is running, and only when `general.show_tool_calls` is true (the server suppresses the event, not the frontend); payload: `{type, persona, tool_name, arguments, result, failed}`. `failed` is a server-computed boolean (tool error, unknown tool, or unparseable/truncated arguments) — the frontend styles the chip from it, not by sniffing the result string. Tool calls whose arguments are not valid JSON (typically truncated at `max_tokens`) are never executed; the LLM receives an `Error: ...` result and can retry. The agentic loop is capped at `mcp.max_tool_iterations` rounds (default 8) per persona reply.

**Persona memories.** Each persona has a `memory_size` budget (0–16384 bytes, default 8192, 0 = disabled) stored in its `prompt.md` frontmatter, plus a `memories.txt` with one flattened line per memory. Two global/per-persona gates control the feature: `general.enable_persona_memories` (default true) and the persona's own budget > 0. When both pass, the chat router appends the persona's current memories to its system prompt before every LLM call, and the built-in `add_memory` tool is offered to tool-enabled personas — its success/error strings are returned to the LLM verbatim (it never sees exceptions). `memories.txt` is purged oldest-first whenever it exceeds the budget: on save (in `add_memory`), best-effort on persona update when the budget is lowered, and on the read path before memory injection (via `persona_store.purge_memories_to_limit`, a cheap no-op when the file is within budget) — so a file inflated by an external editor is normalized before it ever reaches the LLM. The file contents themselves are never cached: `read_memories()` re-reads the disk on every chat request. The editor's "Clear saved memories" sends `clear_memories=true` **only** after an explicit click, and the update endpoint's `memory_size` form field is **required** — an omitted value must 422 rather than silently reset the budget.

`general.persona_name_mentions` (default true) controls whether the frontend prefixes assistant bubbles with the persona's name. It is frontend-only — it has no backend or LLM effect.

## Chat persistence

Every message is persisted to disk automatically — no configuration toggle needed.

- **Location**: `chatrooms/<room_name>/history.json` + audio files alongside it.
- **Format**: JSON with `datetime` (ISO-8601) and `messages` array. Each message has `id` (UUID), `sender` ("USER" or persona name), `text`, and `audio` (array of filenames).
**Audio files**: Named `<message_uuid>_<index>.<ext>` (e.g. `d4ee3044_1.wav`). Extension derived from MIME type, falls back to `.bin`. Audio that arrives *before* the message row exists (STT recordings upload before the chat request creates the user message; streaming TTS sentences upload while the reply is still streaming) is staged as `<message_uuid>_pending_<hex8>.<ext>` and automatically attached to the message's audio list when `persist_message()` runs — the staging registry lives in `app/persistence.py` (`_pending_audio`), in-memory only, so a process restart in that window leaves the (valid) file unreferenced.
- **Room switching**: `GET /api/session/load-room/<room_name>` loads persisted history and populates the in-memory session.
- **New Chat**: `POST /api/session/new` clears both in-memory history and deletes all files in the room's persistence directory.
- **Audio upload**: `POST /api/persist/audio?room=<room>` accepts base64 audio and appends it to the message's audio list.
- **Audio playback**: `GET /api/persist/audio/<room>/<filename>` serves persisted audio files.

The `SessionManager.add_user_message()` and `add_assistant_message()` methods take a `message_id` parameter and call `persistence.persist_message()` automatically.

## TTS and STT are independent

Each has its own `enabled` + `base_url` in `settings.yaml`. Use the `is_active` property (requires both enabled AND a non-empty base_url) instead of checking `enabled` alone.

**TTS server** (`app/routers/tts.py`, `app/services/tts_client.py`): The `/api/tts` and `/api/tts/health` routes live in `app/routers/tts.py`. The client functions `synthesize()` and `check_tts_health()` are in `app/services/tts_client.py`. A persona is TTS-capable only when both `reference_audio` and `reference_audio_transcript` are set (computed as the `Persona.tts_capable` property in `app/config.py`); `/api/tts` returns 404 for an unknown persona, 400 for a non-TTS-capable one, and 503 if the reference files can't be read. TTS supports a `streaming` mode (sentence-by-sentence) controlled by `settings.tts.streaming`, reported in `/api/tts/health` so the frontend can pick its queueing strategy.

**STT server** (`app/routers/stt.py`, `app/services/stt_client.py`): The `/api/stt` and `/api/stt/health` routes live in `app/routers/stt.py`. The client functions `transcribe_audio()` and `check_stt_health()` are in `app/services/stt_client.py`, forwarding the audio as multipart form data to an OpenAI-compatible `/v1/audio/transcriptions` endpoint. `/api/stt` returns 503 when STT is not `is_active`. STT requires no per-persona config.

**STT flow**: The microphone button in the input bar (or **Ctrl+Space**) uses `getUserMedia` + `MediaRecorder`. Click to start, click again to stop. The recorded blob is base64-encoded and POSTed to `/api/stt` with the actual `MediaRecorder` MIME type (browsers differ: webm, ogg, mp4, …). On success, the transcribed text is appended to the input box (never replaces), the recording is persisted to the current room via `persistence.js` with a playback button injected into the live user bubble, and `sendMessage()` is called automatically. Any non-2xx response (or a fetch failure) leaves the mic button disabled for the rest of the session.

Both TTS and STT capture audio and persist it to the current room via `persistence.js`. TTS audio is associated with the assistant message ID issued in the `start` event (stamped onto each TTS item at enqueue time — never looked up in shared state at fetch-resolution time). STT audio is associated with the user message ID generated before `sendMessage()`.

## Chat rooms

Chat rooms are stored in `chatrooms.yaml` and managed via `get_chatrooms()` / `save_chatrooms()` in `app/config.py`.

- The implicit **`"default"` room** always exists and always contains all configured personas. It cannot be created, modified, or deleted (the API rejects attempts with 400/409), and `GET /api/chatrooms/all` / `GET /api/chatrooms/default` synthesize it on the fly from the persona list.
- Room names match case-insensitively and may only contain letters, numbers, spaces, hyphens, and underscores; creating a duplicate (or the reserved name `default`) is rejected with 409.
- New rooms start with zero personas assigned; assigning a nonexistent persona returns 422.
- Switching rooms loads persisted history from disk into the session rather than clearing it.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/personas` | List all personas (summary) |
| `GET` | `/api/personas/{name}/detail` | Full persona detail |
| `POST` | `/api/personas` | Create a new persona (multipart form: text fields + optional avatar/reference audio files) |
| `PUT` | `/api/personas/{name}` | Update a persona (multipart form; rename cascades to chat rooms, directory is never renamed) |
| `DELETE` | `/api/personas/{name}` | Delete a persona and its directory (cascades to chat rooms) |
| `POST` | `/api/personas/{name}/clone` | Clone a persona with a numeric suffix (`Name_2`, `Name_3`, …) |
| `GET` | `/api/personas/{name}/avatar` | Serve a persona's avatar image file |
| `GET` | `/api/personas/{name}/reference-audio` | Serve a persona's reference audio file (`ref.wav`) |
| `GET` | `/api/chatrooms` | List all chat rooms (excluding implicit "default") |
| `GET` | `/api/chatrooms/all` | List all chat rooms including "default" (feeds the frontend dropdown) |
| `GET` | `/api/chatrooms/{name}` | Get a single chat room (including "default") |
| `POST` | `/api/chatrooms` | Create a chat room (starts with no personas) |
| `DELETE` | `/api/chatrooms/{name}` | Delete a chat room |
| `PUT` | `/api/chatrooms/{name}/personas` | Add personas to a room |
| `DELETE` | `/api/chatrooms/{name}/personas/{persona_name}` | Remove a persona from a room |
| `PUT` | `/api/chatrooms/{name}` | Update a room's settings (partial; omitted fields unchanged) |
| `GET` | `/api/player` | Get which persona the player is playing |
| `PUT` | `/api/player` | Adopt a persona (empty name = play as yourself) |
| `GET` | `/api/session` | Get current session state (history + active personas + current room) |
| `POST` | `/api/session/new` | Clear history and reset session (also clears persisted files) |
| `POST` | `/api/session/personas` | Update active personas for the session |
| `GET` | `/api/session/load-room/{room_name}` | Load persisted history for a room into the active session |
| `POST` | `/api/chat` | Send a message; returns an SSE stream |
| `POST` | `/api/chat/suggest` | Draft the player's next message in their own voice |
| `POST` | `/api/chat/speak` | Post a line as a persona, verbatim, with no LLM call |
| `POST` | `/api/persist/audio?room=<room>` | Upload base64 audio for a persisted message |
| `GET` | `/api/persist/audio/{room_name}/{filename}` | Serve a persisted audio file for playback |
| `GET` | `/api/tts/health` | TTS availability status |
| `POST` | `/api/tts` | Proxy text → TTS `/synthesize`; returns `{audio_base64, sample_rate}` |
| `GET` | `/api/stt/health` | STT availability status |
| `POST` | `/api/stt` | Proxy audio → STT `/v1/audio/transcriptions`; returns `{text, language}` |
| `GET` | `/api/settings` | Get current settings |
| `PUT` | `/api/settings` | Update and persist settings to `settings.yaml` |

## Persona CRUD cascades

Renaming or deleting a persona cascades to `chatrooms.yaml` via `_cascade_persona_rename()` / `_cascade_persona_delete()` in `app/routers/personas.py`. Keep this in sync if data models change. Persona create/update/delete/clone are **multipart** (`POST`/`PUT` with `Form` text fields + optional `UploadFile` avatar / reference audio); there is no JSON request model for them — the `PersonaResponse`/`PersonaDetailResponse` in `app/models.py` are the only persona request/response shapes, and they carry file *contents* and boolean capability flags rather than paths. `memory_size` is a `Form` field on both create (default 8192) and update (**required** — no server-side default, so an omitted value 422s instead of silently resetting the budget); `clear_memories` is an optional boolean on update only; cloning carries the source's `memory_size` over.

## Pydantic models

All request/response shapes and config models live in `app/models.py` and `app/config.py`. Add new fields there, not inline in routers.
