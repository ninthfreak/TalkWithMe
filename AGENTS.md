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
- **Isolation**: `tests/conftest.py` has an autouse fixture that points every module-level global at per-test `tmp_path` state: the config caches in `app/config.py`, `_PERSISTENCE_ROOT` (in both `app/persistence.py` and `app/routers/persistence.py` — the router imported it *by value*, so it needs its own patch), the `session` singleton, and the MCP tool registry. Real `settings.yaml` / `personas.yaml` / `chatrooms.yaml` / `chatrooms/` data is never read or written. **If you add a new module-level global to the app, add it to that fixture.**
- **Lifespan**: the `client` fixture uses `TestClient` *without* the startup lifespan, because the lifespan re-reads the real YAML files (clobbering test caches) and attempts MCP discovery. Tests that exercise the lifespan do so explicitly with a local `TestClient` in a `with` block and monkeypatched `load_*`/`load_tools` (see `tests/test_main.py`).
- **Coverage map**: `test_config.py` (config models + YAML load/save), `test_models.py` (API request/response models), `test_persistence.py` (disk persistence + audio staging), `test_session_manager.py`, `test_llm.py` (SSE parsing + agentic tool loop), `test_mcp_client.py`, `test_tool_registry.py`, `test_tts_stt_clients.py`, `test_chat_sse.py` (the `/api/chat` SSE endpoint — LLM stubbed, selection/persistence/echo/tool events for real), `test_main.py`, `test_docs.py` (the AGENTS.md API endpoints table must match the routes actually registered on the app — run it after adding/removing endpoints and update the table), and one `test_routers_*.py` per API router.
- **Rules**:
  - Every code change must be followed by a clean run: `python3 -m pytest`, all green. No exceptions, no skipped tests.
  - New functionality or API endpoints require new tests in the matching `test_*.py` file before the change is complete.
  - API tests use the `client` fixture + config caches (re-point `app.config._settings_cache` / `_personas_cache` / `_chatrooms_cache` via `monkeypatch`); router-level stubs are applied at the router's import site (e.g. `app.routers.chat.stream_chat`), since routers import service functions by name.
  - `httpx` version pin matters: with httpx 0.24, `response.url` / `raise_for_status()` raise `RuntimeError` if no `request` is attached to a `Response`, and the `.request` *getter* itself raises when unset (check `response._request` instead). All fake responses in `tests/factories.py` attach a request for this reason.

## Config — three YAML files, cached at startup

**Two locations, one of them inert.** The app reads and writes `config/settings.yaml`, `config/personas.yaml`, `config/chatrooms.yaml` (gitignored). The identically-named files at the **repo root are tracked shipped defaults and are never written to.** Do not "tidy this up" by untracking the root copies: `git rm --cached` on them makes the next `git pull` try to delete files users have edited locally, and git aborts the merge with *"Your local changes to the following files would be overwritten by merge"*. That is the bug this layout exists to prevent.

`config_path()` in `app/config.py` prefers `config/` and falls back to the root copy, so the very first run after an upgrade still sees the user's existing settings. `migrate_config_files()` (called from the `app/main.py` lifespan, before any load) **copies** root → `config/`, upgrading the schema on the way; it never moves or deletes, and never overwrites an existing `config/` file. `save_*()` always writes to `config/`. All path helpers read `_PROJECT_ROOT` at call time so `tests/conftest.py` can repoint them at `tmp_path`.

**Schema versioning.** Every written file starts with `schema_version` (`CONFIG_SCHEMA_VERSION` in `app/config_migrations.py`); files predating it are version 1. `load_*()` runs the raw dict through `migrate_personas` / `migrate_chatrooms` / `migrate_settings` **before Pydantic sees it** — that ordering is load-bearing, since once a model has parsed or dropped a legacy key the information needed to migrate it is gone. Migrations return `(raw, notes)` and the notes are logged, so an upgrade is visible. A file from a *newer* schema loads with a warning rather than failing. To add a migration: bump the version, add a step to the relevant chain, and cover it in `tests/test_config_migrations.py` with a real old-format file.

Migrations translate rather than drop wherever intent can be preserved — a persona's old absolute `typical_length` becomes the equivalent relative `length_bias` (its offset from the old `normal`), so a laconic persona stays laconic instead of being silently flattened to the default.

## Architecture

- **Backend**: FastAPI. Entry point: `app/main.py`. Routers in `app/routers/`, external service clients in `app/services/`.
- **Session**: Single global `session` singleton in `app/session.py`. Intentional — this is a single-user app. No auth, no database — the `chatrooms/` directory is the only persistent storage. Tracks `current_room` and persists messages to disk automatically. `session.build_llm_messages()` constructs the per-call LLM payload: the responding persona's system prompt, then history remapped so user messages stay `user`, the responder's own messages stay `assistant`, and *other personas'* messages are re-mapped to `user` with a `[Name]: <text>` prefix — this avoids consecutive `assistant` messages (which many LLMs reject with 400) and stops the model treating another persona's words as its own. History is optionally capped at the last `general.max_turns_for_context` entries (default 6). `build_llm_messages()` also takes an optional `room_preamble`, appended to the system message (built by the chat router — see **Chat flow**).
- **Persistence**: Per-room JSON + audio files under `chatrooms/<room>/`. Handled by `app/persistence.py` (framework-agnostic) and `app/routers/persistence.py` (audio upload/serving endpoints). Created lazily on first write.
- **MCP tools**: `app/services/mcp_client.py` speaks MCP (JSON-RPC 2.0 over the Streamable HTTP transport, protocol version 2025-03-26) with a *stateless, per-call* session — `initialize` runs on every discovery/call, no persistent sessions. `app/services/tool_registry.py` caches the discovered tools once at startup (called from `app/main.py` lifespan) and maps tool name → server; duplicate tool names across servers: first listed server wins. The cache is not refreshed by `reload_all()` — `mcp:` changes need a full restart. The `mcp:` section of `settings.yaml` is **yaml-only** (no UI, no API field) — `update_settings` in `app/routers/settings.py` copies it over from the current cache, otherwise a UI settings save would wipe it. Personas with `allow_tool_calls` run `stream_chat_with_tools()` in `app/services/llm.py` (agentic loop; the final round is sent without `tools` to force a text answer). Tool rounds are local to the loop — they are NOT persisted to the chat history. MCP server URLs are validated at config load (must start with `http://` or `https://`) so a scheme-less typo fails loudly at startup instead of surfacing as per-request connection timeouts.
- **Settings update contract**: `PUT /api/settings` treats the `general:` section as a *partial update* — omitted fields (or an absent `general` section) keep their current values; only fields explicitly sent override. `GeneralSettingsRequest` in `app/models.py` uses `Optional` fields defaulting to `None`, and `update_settings` in `app/routers/settings.py` merges via `model_dump(exclude_none=True)`. Dialogs that don't edit general settings (the Servers dialog) must NOT send a `general` section, and the router must NOT rebuild `GeneralConfig` from request-body defaults — doing so is exactly how `show_tool_calls` got reset to `true` on every Servers-dialog save. If you add a new field to `GeneralConfig`, the merge preserves it automatically; no per-field wiring needed. The LLM/TTS/STT sections are full replacements, normalized on save: blank base URLs → `None` (which deactivates the feature via `is_active`) and TTS `seed=0` → `None` (the frontend encodes "no seed" as 0). Changes take effect immediately — no restart.
- **Logging**: `app/main.py` configures logging via `logging.basicConfig(level=INFO, ...)` at import time (a no-op if the root logger already has handlers), because uvicorn's default config leaves the root logger at WARNING, which silently swallows every app `logger.info()` call (this is why per-server MCP discovery lines were invisible while failure `WARNING`s showed). The `httpx` logger is separately pinned to WARNING because it logs one line per HTTP request. Note: `uvicorn --log-level` only affects uvicorn's own loggers, not the app's.
- **Frontend**: Vanilla JS SPA, no bundler. Modules in `static/` communicate via shared globals in `state.js`. See the table below:

| File | Responsibility |
|------|---------------|
| `state.js` | Shared globals (personas, session state, chat room state, TTS/STT flags, audio queue, message IDs) |
| `app.js` | Bootstrap, health checks, event listener setup, session management |
| `chat.js` | Message rendering, SSE stream handling (incl. `tool_call` chips), sending messages, persisted history rendering, audio playback buttons |
| `persistence.js` | History loading, audio upload helpers, audio URL generation |
| `persona.js` | Persona sidebar + editor modal (CRUD) |
| `chatrooms.js` | Chat room dropdown, room filtering, room editor (incl. echo chamber toggle), persona picker modal, room switching with history load |
| `settings.js` | Servers modal (LLM/TTS/STT config) |
| `gen-settings.js` | General settings modal (max persona replies, name mentions, context turns, tool-call visibility) |
| `tts.js` | TTS synthesis, audio queues, Web Audio playback, audio persistence |
| `stt.js` | Microphone recording, STT proxy, transcript insertion, audio persistence |
| `theme.js` | Theme toggle |
| `utils.js` | Shared helpers |

## Chat flow

`POST /api/chat` streams SSE. Every event is a single `data: <JSON>\n\n` line; the `type` field is always present, and the frontend switches on it in `handleSSEEvent()` in `chat.js`. Event types: `start`, `token`, `tool_call`, `done`, `error`, `complete`.
Internally, both `stream_chat()` and `stream_chat_with_tools()` in `app/services/llm.py` yield event dicts (`{"type": "token"|"tool_call"|"finish", ...}`), ending with exactly one `finish` event carrying the backend's `finish_reason`. `stream_chat()` used to yield bare token strings; it was changed so both paths share one shape and so truncation is detectable at all. Both accept `max_tokens` and `stop` overrides.
The request body includes `chat_room` (which room to persist to) and `message_id` (frontend-generated UUID for audio association).

**Persona pool.** `_resolve_room_personas()` in `app/routers/chat.py` is the authoritative source of eligibility: the `"default"` room (or any room not present in `chatrooms.yaml`) includes all personas; a named room is limited to its assigned personas, filtered to those that still exist. It deliberately does NOT read the frontend-maintained `session.active_personas`.

**Persona selection modes:** `"router"` (LLM picks via a non-streaming `chat_completion()` at `temperature=0.1`, `max_tokens=16`; the prompt includes each eligible persona's `router_hints` plus the last `max_turns_for_context` turns of context; an unrecognized name or LLM failure falls back to random), `"random"` (uniform over eligible personas), or an explicit persona name (validated against the room; unknown values fall back to random).

**Multiple replies.** `general.max_persona_replies` (1–6, capped at the number of eligible personas) controls how many personas answer one user message: multiple `start`/`token`/`done` cycles, one per responding persona. The first responder comes from the selection mode; each subsequent one is a random non-repeating pick from the remaining eligible personas.

**Message IDs.** Both `start` and `done` carry `message_id` (server-generated UUID for that persona's assistant message); `start` also carries `user_message_id` (the frontend's ID, echoed back).
`start` carries the assistant ID first **on purpose**: the frontend stamps it onto streaming TTS items at enqueue time, so the ID must be generated *before* the `start` event is emitted — moving it back after the stream reintroduces cross-turn audio misattribution.

**Echo chamber.** Each room has an `echo_chamber` flag (set via `PUT /api/chatrooms/{name}/echo-chamber`, toggled in the room editor; the `default` room cannot be modified). When enabled for the active room, the LLM is bypassed entirely: exactly one persona (picked per the normal selection mode) echoes the user's message verbatim as a single `token` event, and `max_persona_replies` is forced to 1.

**Response length.** `general.typical_length` and `ChatRoom.typical_length` hold a `TypicalLength` tier (`terse` | `brief` | `normal` | `detailed` | `verbose` | `unrestricted`). The scale is calibrated for **chat, not prose**: `normal` is ~20 words ("a sentence or two") and `verbose` — the longest named tier — is only ~110 words. An earlier calibration put `normal` at a paragraph, which made every room read like an essay thread; if you change `TYPICAL_LENGTH_SPECS`, keep it chat-shaped.

Personas do **not** carry an absolute tier. `Persona.length_bias` is a `LengthBias` (`much_shorter` … `much_longer`, default `match`) that moves them ±1 or ±2 steps along `LENGTH_SCALE` from the room's tier, clamped at both ends; `resolve_typical_length()` in `app/config.py` does this. Relative rather than absolute on purpose: a laconic persona should be laconic *for the room they are in*, not a fixed word count that fights the room. An `unrestricted` room ignores bias — with no target there is nothing to be relative to — and `LENGTH_SCALE` deliberately excludes `unrestricted` for the same reason. The implicit `"default"` room has no `chatrooms.yaml` entry, so its base tier is the global value.

**The tier shapes length through the prompt, not through `max_tokens`** — this is the whole point. `derive_max_tokens()` turns the tier's word target into a per-call cap (`words * 1.4 * 3.0`, floored at 256, never above `settings.llm.max_tokens`), which is a runaway guard only. At chat lengths every tier below `verbose` lands on the 256 floor, and that is intended: the prompt does the shaping. Clamping `llm.max_tokens` down to shorten replies is what *caused* personas to be cut off mid-sentence and then continued by the next persona — do not reintroduce it as a style control.

Legacy `typical_length` keys on a persona are dropped with an `INFO` log by `load_personas()` (they cannot be mapped to a relative bias without knowing the room).

**Room preamble.** `_build_room_preamble()` in `app/routers/chat.py` appends a generated block to the persona's `system_prompt`: the room name, the roster of other eligible personas (name + description), an explanation of the `[Name]: ` history convention, the length line, and four prohibitions (write only as yourself; never invent a character; never continue someone else's message; no self name prefix). The roster is what makes "don't invent a character" enforceable. It is passed to `session.build_llm_messages(room_preamble=...)`; `SessionManager` builds none of it, because it deliberately imports no config. Echo chamber rooms get no preamble — the LLM is bypassed entirely.

**Player profile.** Each room carries a `PlayerProfile` (`name`, `description`, `appearance`) plus a `require_player_profile` flag, both in `chatrooms.yaml`. Per room on purpose: the player can be a different character in each room. `appearance` is the "picture" and is deliberately **text** — the LLM is its only audience and it reads descriptions, so an uploaded image would have to be captioned before it was any use. When a profile has content, `_player_lines()` adds it to the room preamble and the persona is told to address the player by name and react to who they are; the "lines with no prefix are from…" line names the character instead of "the user". `PlayerProfile.is_complete` (name **and** description; appearance stays optional) gates chat when `require_player_profile` is set: `_chat_stream()` emits `PROFILE_REQUIRED_MESSAGE` and stops **before** recording the user message, so nothing half-happens. The frontend blocks the send and opens the editor too, but the server is the authority — same rule as persona eligibility. The implicit `"default"` room supports neither field (no `chatrooms.yaml` entry); its controls are hidden in the UI and the endpoints return 400.

**Room mutation helpers.** `_editable_room_or_400()` and `_save_room_update()` in `app/routers/chatrooms.py` back all four room-setting endpoints (echo chamber, typical length, player profile, require-profile). `_save_room_update()` uses `model_copy`, never a rebuilt `ChatRoom` — rebuilding field-by-field is how room settings used to get silently dropped (see **Persona CRUD cascades**).

**Reply guard.** `app/services/reply_guard.py` is a pure state machine (no config, no I/O) filtering each reply, in three layers. (1) `stop_sequences()` passes the other room personas' speaker prefixes as `stop` strings — exact names only, never a generic `"\n["` which would fire on legitimate bracketed lines. (2) `ReplyGuard` strips the speaker's own `Name:` / `[Name]:` prefix and cuts the reply at a *foreign* speaker prefix: bracketed prefixes always cut; unbracketed ones cut only for a known room persona or a name-shaped token (1–3 capitalised words) not in `_PROSE_LEAD_INS` — that stoplist is what stops `Note:` / `Summary:` truncating a legitimate reply. (3) Truncation is marked from `finish_reason == "length"`. The guard **buffers at line starts** while deciding, which is why the head of a line can arrive coalesced into one `token` event; chunk boundaries are not part of the SSE contract. It must resolve prefixes before emitting, because `chat.js` appends each token straight into the bubble and never re-renders from `done`.

**Truncation.** `ChatMessage.truncated` is set when the LLM stopped at `max_tokens`. It changes only what *other* personas see: `_other_persona_text()` in `app/session.py` trims a truncated message back to its last complete sentence, or labels it `(message was cut off)` when it has none. It never changes displayed or persisted text, is excluded from `get_history_dicts()`, and is not written to disk — it is a property of one generation, not of the message.

**Tool calls.** `tool_call` is only emitted while a tool-enabled persona's agentic loop is running, and only when `general.show_tool_calls` is true (the server suppresses the event, not the frontend); payload: `{type, persona, tool_name, arguments, result, failed}`. `failed` is a server-computed boolean (tool error, unknown tool, or unparseable/truncated arguments) — the frontend styles the chip from it, not by sniffing the result string. Tool calls whose arguments are not valid JSON (typically truncated at `max_tokens`) are never executed; the LLM receives an `Error: ...` result and can retry. The agentic loop is capped at `mcp.max_tool_iterations` rounds (default 8) per persona reply.

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
| `POST` | `/api/personas` | Create a new persona |
| `PUT` | `/api/personas/{name}` | Update a persona (rename cascades to chat rooms) |
| `DELETE` | `/api/personas/{name}` | Delete a persona (cascades to chat rooms) |
| `POST` | `/api/personas/{name}/clone` | Clone a persona with a numeric suffix (`Name_2`, `Name_3`, …) |
| `GET` | `/api/personas/{name}/avatar` | Serve a persona's avatar image file |
| `GET` | `/api/chatrooms` | List all chat rooms (excluding implicit "default") |
| `GET` | `/api/chatrooms/all` | List all chat rooms including "default" (feeds the frontend dropdown) |
| `GET` | `/api/chatrooms/{name}` | Get a single chat room (including "default") |
| `POST` | `/api/chatrooms` | Create a chat room (starts with no personas) |
| `DELETE` | `/api/chatrooms/{name}` | Delete a chat room |
| `PUT` | `/api/chatrooms/{name}/personas` | Add personas to a room |
| `DELETE` | `/api/chatrooms/{name}/personas/{persona_name}` | Remove a persona from a room |
| `PUT` | `/api/chatrooms/{name}/echo-chamber` | Set/clear the room's echo chamber flag |
| `PUT` | `/api/chatrooms/{name}/typical-length` | Set the room's typical response length tier |
| `PUT` | `/api/chatrooms/{name}/player-profile` | Set the human user's character profile for the room |
| `PUT` | `/api/chatrooms/{name}/require-player-profile` | Set whether the room demands a profile before chatting |
| `GET` | `/api/session` | Get current session state (history + active personas + current room) |
| `POST` | `/api/session/new` | Clear history and reset session (also clears persisted files) |
| `POST` | `/api/session/personas` | Update active personas for the session |
| `GET` | `/api/session/load-room/{room_name}` | Load persisted history for a room into the active session |
| `POST` | `/api/chat` | Send a message; returns an SSE stream |
| `POST` | `/api/persist/audio?room=<room>` | Upload base64 audio for a persisted message |
| `GET` | `/api/persist/audio/{room_name}/{filename}` | Serve a persisted audio file for playback |
| `GET` | `/api/tts/health` | TTS availability status |
| `POST` | `/api/tts` | Proxy text → TTS `/synthesize`; returns `{audio_base64, sample_rate}` |
| `GET` | `/api/stt/health` | STT availability status |
| `POST` | `/api/stt` | Proxy audio → STT `/v1/audio/transcriptions`; returns `{text, language}` |
| `GET` | `/api/settings` | Get current settings |
| `PUT` | `/api/settings` | Update and persist settings to `settings.yaml` |

## Persona CRUD cascades

Renaming or deleting a persona cascades to `chatrooms.yaml` via `_cascade_persona_rename()` / `_cascade_persona_delete()` in `app/routers/personas.py`. Keep this in sync if data models change.

## Pydantic models

All request/response shapes and config models live in `app/models.py` and `app/config.py`. Add new fields there, not inline in routers.
