"""Configuration loading and validation.

Loads settings.yaml and the Personas directory from the project root.
Caches parsed config so we're not hitting disk on every request.

Personas are stored as per-persona subdirectories (see
app/services/persona_store.py). The legacy personas.yaml file is read
only for the one-time startup migration — never for anything else.
"""

import logging
import math
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from app.config_migrations import (
    CONFIG_SCHEMA_VERSION,
    migrate_chatrooms,
    migrate_personas,
    migrate_player,
    migrate_settings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typical response length
# ---------------------------------------------------------------------------
#
# Length is shaped by a *prompt instruction*, not by max_tokens. Clamping
# max_tokens down to shorten replies is what caused personas to be cut off
# mid-sentence, which in turn is what made the next persona continue the
# unfinished thought in the wrong voice. The tier below drives the prompt
# line; the derived token cap is only a runaway guard.


class TypicalLength(str, Enum):
    """How long replies in a room should typically be.

    Calibrated for *chat*, not prose. People in a chat room write a
    fragment, sometimes a full sentence, occasionally two when the thought
    needs it — so NORMAL is a sentence or two, and even VERBOSE is only a
    short paragraph. An earlier version put NORMAL at a paragraph, which
    made every room read like an essay thread.

    The members are ordered shortest to longest; LENGTH_SCALE below depends
    on that order, and UNRESTRICTED sits outside it.
    """

    TERSE = "terse"
    BRIEF = "brief"
    NORMAL = "normal"
    DETAILED = "detailed"
    VERBOSE = "verbose"
    UNRESTRICTED = "unrestricted"


class LengthBias(str, Enum):
    """How a persona's replies sit *relative to* their room.

    Relative, not absolute: a laconic persona should be laconic for the
    room they are in. In a clipped room they answer in a couple of words;
    in a talkative one they are still the short one, but not mute. An
    absolute per-persona length fights the room instead of sitting inside
    it.
    """

    MUCH_SHORTER = "much_shorter"
    SHORTER = "shorter"
    MATCH = "match"
    LONGER = "longer"
    MUCH_LONGER = "much_longer"


# How many steps along LENGTH_SCALE each bias moves a persona.
LENGTH_BIAS_STEPS: Dict[LengthBias, int] = {
    LengthBias.MUCH_SHORTER: -2,
    LengthBias.SHORTER: -1,
    LengthBias.MATCH: 0,
    LengthBias.LONGER: 1,
    LengthBias.MUCH_LONGER: 2,
}


class LengthSpec(NamedTuple):
    """A tier's word target and the phrasing used in the prompt.

    words == 0 means "no guidance": no prompt line and no derived cap.
    """

    words: int
    phrasing: str


TYPICAL_LENGTH_SPECS: Dict[TypicalLength, LengthSpec] = {
    TypicalLength.TERSE: LengthSpec(4, "a few words — often not even a full sentence"),
    TypicalLength.BRIEF: LengthSpec(10, "one short sentence"),
    TypicalLength.NORMAL: LengthSpec(20, "a sentence or two"),
    TypicalLength.DETAILED: LengthSpec(45, "two or three sentences"),
    TypicalLength.VERBOSE: LengthSpec(110, "a short paragraph"),
    TypicalLength.UNRESTRICTED: LengthSpec(0, ""),
}

# The ordinal scale a LengthBias moves along. UNRESTRICTED is deliberately
# absent: it is "no target at all", so there is nothing to be relative to.
LENGTH_SCALE: List[TypicalLength] = [
    TypicalLength.TERSE,
    TypicalLength.BRIEF,
    TypicalLength.NORMAL,
    TypicalLength.DETAILED,
    TypicalLength.VERBOSE,
]

# Rough English tokens-per-word for the derived cap. Deliberately not a
# setting — it is a constant of the encoding, not a preference.
_TOKENS_PER_WORD = 1.4
# The cap sits well above the target so "go longer when it genuinely needs
# it" stays possible. Only a runaway reply should ever hit it.
_LENGTH_HEADROOM = 3.0
# Below this, the cap starts shaping replies again instead of guarding them
# — which is the exact failure this feature exists to remove. At chat
# lengths every tier below VERBOSE lands on this floor, and that is the
# point: the prompt does the shaping, the cap only stops a runaway.
_MIN_DERIVED_MAX_TOKENS = 256


def derive_max_tokens(length: TypicalLength, ceiling: int) -> int:
    """Token cap for a reply of the given tier, never above *ceiling*.

    *ceiling* is settings.llm.max_tokens, which keeps its meaning as the
    absolute maximum: this can only lower it, never raise it.
    """
    spec = TYPICAL_LENGTH_SPECS[length]
    if spec.words == 0:
        return ceiling
    derived = math.ceil(spec.words * _TOKENS_PER_WORD * _LENGTH_HEADROOM)
    return min(ceiling, max(derived, _MIN_DERIVED_MAX_TOKENS))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class LLMSettings(BaseModel):
    base_url: str = "http://localhost:8080"
    model: str = "default"
    max_tokens: int = 1024
    temperature: float = 0.8


class TTSConfig(BaseModel):
    enabled: bool = True
    base_url: Optional[str] = None
    num_steps: int = 10
    guidance_scale: float = 1.5
    seed: Optional[int] = None
    timeout: float = 60.0
    streaming: bool = False

    @model_validator(mode="after")
    def _normalize_base_url(self) -> "TTSConfig":
        """Treat blank strings as None so a missing URL implicitly disables TTS."""
        if self.base_url is not None and not self.base_url.strip():
            self.base_url = None
        return self

    @property
    def is_active(self) -> bool:
        """Feature is active only when explicitly enabled AND a base_url is configured."""
        return self.enabled and bool(self.base_url)


class STTConfig(BaseModel):
    """Speech-to-text configuration, independent of TTS."""
    enabled: bool = True
    base_url: Optional[str] = None
    timeout: float = 30.0

    @model_validator(mode="after")
    def _normalize_base_url(self) -> "STTConfig":
        """Treat blank strings as None so a missing URL implicitly disables STT."""
        if self.base_url is not None and not self.base_url.strip():
            self.base_url = None
        return self

    @property
    def is_active(self) -> bool:
        """Feature is active only when explicitly enabled AND a base_url is configured."""
        return self.enabled and bool(self.base_url)


class GeneralConfig(BaseModel):
    """Application-wide feature flags and preferences."""
    persona_name_mentions: bool = True
    max_persona_replies: int = Field(default=1, ge=1, le=6)
    max_turns_for_context: int = Field(
        default=6, ge=1, le=50,
        description=(
            "How many recent exchanges to send to the LLM. One exchange is a "
            "human message plus every persona reply it drew, so this does not "
            "shrink as more personas answer."
        ),
    )
    show_tool_calls: bool = True
    # Fallback tier, and the only one the implicit "default" room can use
    # (it has no chatrooms.yaml entry to carry an override).
    typical_length: TypicalLength = TypicalLength.NORMAL
    # Where persona subdirectories live. Absolute, or relative to the
    # project root; None/empty falls back to <project root>/Personas.
    # yaml-only for now (no UI) — like the mcp: section, changes need a
    # restart, because the directory is resolved at startup and by the
    # persona router from this cache.
    personas_directory: Optional[str] = None


class MCPServerConfig(BaseModel):
    """A single MCP server endpoint (SSE/HTTP transport only — no stdio)."""
    name: str
    url: str
    # gt=0: with this httpx version a 0.0 timeout does NOT mean "no
    # timeout" — it fails every request instantly, so the typo would
    # silently kill the server. le=300: a hung tool call should not be
    # allowed to stall the SSE stream for unreasonably long.
    timeout: float = Field(default=10.0, gt=0, le=300)

    @model_validator(mode="after")
    def _validate_url_scheme(self) -> "MCPServerConfig":
        # Fail at config load, not per-call: a scheme-less typo
        # ("localhost:9000") used to surface as a ConnectTimeout warning
        # buried in the log on every request instead of a clear startup error.
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(
                f"MCP server '{self.name}': url must start with http:// or https://, got {self.url!r}"
            )
        return self


class MCPConfig(BaseModel):
    """MCP server configurations and the agentic tool-call loop cap."""
    servers: List[MCPServerConfig] = Field(default_factory=list)
    max_tool_iterations: int = Field(default=8, ge=1, le=50, description="Max tool-call rounds per persona reply")


class AppSettings(BaseModel):
    llm: LLMSettings = LLMSettings()
    tts: TTSConfig = TTSConfig()
    stt: STTConfig = STTConfig()
    general: GeneralConfig = GeneralConfig()
    mcp: MCPConfig = Field(default_factory=MCPConfig)


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------

class Persona(BaseModel):
    name: str
    description: str = ""
    system_prompt: str
    router_hints: str = ""
    avatar_color: str = "#888888"
    avatar_image: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_audio_transcript: Optional[str] = None
    reference_audio_language: str = "en"
    allow_tool_calls: bool = False
    # Relative to the room, never absolute: this nudges the persona a step
    # or two along the room's own scale. See LengthBias.
    length_bias: LengthBias = LengthBias.MATCH
    # Where this persona's files live on disk (set by the directory scan;
    # None for personas assembled outside of it, e.g. in tests).
    persona_dir: Optional[Path] = None

    @property
    def tts_capable(self) -> bool:
        """TTS requires both reference audio AND its transcript."""
        return bool(self.reference_audio and self.reference_audio_transcript)


class PersonasConfig(BaseModel):
    personas: List[Persona] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Chat Rooms
# ---------------------------------------------------------------------------

class PlayerConfig(BaseModel):
    """Which persona the human is playing, if any.

    The player adopts one of the configured personas rather than writing a
    character of their own: the personas already carry a name, a
    description and a system prompt, so a second, parallel way to describe
    a character was redundant and could drift from them.

    Stored in its own file beside personas.yaml. It is not room data — a
    room can *require* that the player has adopted someone, which is a
    property of the room, but who they are playing belongs to the player
    and applies in whichever room they are in.

    An empty string means "playing as themselves".
    """

    persona_name: str = ""

    def adopted(self, known_names) -> str:
        """The adopted persona's name, or "" if none/unknown.

        Resolved against the live persona list on every read rather than
        trusted from disk: a persona can be deleted or renamed after the
        player adopted it, and a dangling reference must degrade to
        "playing as themselves" rather than half-applying.
        """
        name = self.persona_name.strip()
        return name if name and name in known_names else ""


class ChatRoom(BaseModel):
    """A named grouping of personas."""
    name: str
    persona_names: List[str] = Field(default_factory=list)
    typical_length: TypicalLength = TypicalLength.NORMAL
    # A property of the room: whether it insists on knowing who you are
    # before you can chat. *Who* you are playing is not the room's —
    # that lives in player.yaml.
    require_player_persona: bool = False


class ChatRoomsConfig(BaseModel):
    chat_rooms: List[ChatRoom] = Field(default_factory=list)


def resolve_typical_length(
    persona: Optional[Persona],
    room: Optional[ChatRoom],
    general_default: TypicalLength,
) -> TypicalLength:
    """The tier a persona actually replies at: the room's, nudged by bias.

    The room sets the register; the persona only shifts a step or two
    within it, and the shift clamps at both ends of LENGTH_SCALE — a
    laconic persona in an already-terse room is simply terse, not silent.

    *room* is None for the implicit "default" room, which has no
    chatrooms.yaml entry and therefore falls back to the global value.

    An UNRESTRICTED room ignores bias entirely: with no target set there is
    nothing for a persona to be shorter or longer *than*.
    """
    base = room.typical_length if room is not None else general_default
    if base is TypicalLength.UNRESTRICTED or persona is None:
        return base

    steps = LENGTH_BIAS_STEPS[persona.length_bias]
    if steps == 0:
        return base

    index = LENGTH_SCALE.index(base) + steps
    return LENGTH_SCALE[max(0, min(index, len(LENGTH_SCALE) - 1))]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Live config lives in config/, which git never tracked and .gitignore
# excludes. The identically-named files at the repo root are the *shipped*
# starting points and stay tracked: untracking them made `git pull` try to
# delete files people had edited locally, which git refuses outright
# ("Your local changes to the following files would be overwritten by
# merge"). Nothing the app writes ever touches them again.
_CONFIG_DIR_NAME = "config"

SETTINGS_FILE = "settings.yaml"
PERSONAS_FILE = "personas.yaml"
CHATROOMS_FILE = "chatrooms.yaml"
PLAYER_FILE = "player.yaml"

_settings_cache: Optional[AppSettings] = None
_personas_cache: Optional[PersonasConfig] = None
_chatrooms_cache: Optional[ChatRoomsConfig] = None
_player_cache: Optional[PlayerConfig] = None


def config_dir() -> Path:
    """Directory holding the live config. Resolved per call, not at import,
    so tests can repoint _PROJECT_ROOT."""
    return _PROJECT_ROOT / _CONFIG_DIR_NAME


def legacy_path(filename: str) -> Path:
    """Where this file lived before config/ existed: the repo root."""
    return _PROJECT_ROOT / filename


def config_path(filename: str) -> Path:
    """The live path for *filename*, preferring config/.

    Falls back to the repo-root copy when config/ has nothing yet, so an
    existing install keeps working on the very first run after upgrading —
    ``migrate_config_files()`` then moves it across properly.
    """
    live = config_dir() / filename
    if live.exists():
        return live
    return legacy_path(filename)


def _read_raw(target: Path) -> dict:
    # encoding is explicit: the platform default is cp1252 on Windows, which
    # raises on a CJK character or emoji in a system prompt.
    with open(target, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _log_notes(filename: str, notes: list) -> None:
    for note in notes:
        logger.info("Migrated %s: %s", filename, note)


def _write_raw(target: Path, raw: dict) -> None:
    """Write *raw* to *target*, creating config/ if needed.

    schema_version leads the file so a human opening it can see which
    schema it is in without reading to the bottom.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": CONFIG_SCHEMA_VERSION}
    ordered.update({k: v for k, v in raw.items() if k != "schema_version"})

    # Write-then-rename: opening the real file "w" truncates it before the
    # dump runs, so an encoding error, a full disk or a kill mid-write left
    # a half-written config behind while the in-memory cache kept serving
    # the whole thing — the loss only surfaced on the next restart.
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(ordered, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_settings(path: Optional[Path] = None) -> AppSettings:
    """Parse settings.yaml. Falls back to defaults if the file is missing."""
    global _settings_cache
    target = path or config_path(SETTINGS_FILE)
    if not target.exists():
        return AppSettings()
    raw, notes = migrate_settings(_read_raw(target))
    _log_notes(SETTINGS_FILE, notes)
    _settings_cache = AppSettings(
        llm=LLMSettings(**raw.get("llm", {})),
        tts=TTSConfig(**raw.get("tts", {})),
        stt=STTConfig(**raw.get("stt", {})),
        general=GeneralConfig(**raw.get("general", {})),
        mcp=MCPConfig(**raw.get("mcp", {})),
    )
    return _settings_cache


def get_personas_directory() -> Path:
    """Resolve the configured Personas directory.

    ``general.personas_directory`` may be absolute or relative to the
    project root; missing/empty falls back to <project root>/Personas.
    """
    configured = (get_settings().general.personas_directory or "").strip()
    if not configured:
        return _PROJECT_ROOT / "Personas"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


def load_personas() -> PersonasConfig:
    """Load personas from the configured Personas directory and cache them.

    Startup decision matrix (docs/feature_persona_autodiscovery.md):

    * ``personas.yaml`` AND a populated Personas directory -> warn
      loudly; the directory wins and the YAML is left in place (it is
      IGNORED — not renamed, not deleted).
    * ``personas.yaml`` only -> one-time automatic migration into the
      directory; the YAML is renamed to personas.yaml.bak on success.
    * Neither -> log "No personas found!", create the (empty) directory,
      and start with zero personas.

    Raises on fatal errors (uncreatable directory, failed migration):
    the app must not run while unsure where its personas live.
    """
    global _personas_cache
    # Imported lazily: persona_store imports the Persona models from this
    # module, so a top-level import would be circular.
    from app.services import persona_store

    root = get_personas_directory()
    # config/, not the repo root. The root personas.yaml is tracked by
    # git and deliberately inert (see migrate_config_files); renaming it
    # to .bak is exactly the churn that makes `git pull` fail. The live
    # copy migrate_config_files() puts in config/ is the real one, and
    # it is the one that gets consumed and renamed.
    legacy_yaml = config_dir() / PERSONAS_FILE

    if legacy_yaml.is_file():
        if _personas_directory_populated(root):
            logger.warning(
                "Both personas.yaml and the Personas directory (%s) exist. "
                "The directory takes precedence and personas.yaml is IGNORED. "
                "Delete or rename personas.yaml to silence this warning.",
                root,
            )
        else:
            persona_store.migrate_from_legacy_yaml(legacy_yaml, root)
    elif not root.is_dir():
        logger.error("No personas found!")
        logger.error("Persona directory: %s", root)
        try:
            root.mkdir(parents=True)
        except OSError as exc:
            logger.error("Cannot create the Personas directory %s: %s — aborting startup.", root, exc)
            raise persona_store.PersonaStorageError(
                f"cannot create personas directory {root}: {exc}"
            ) from exc
        logger.info("Created empty Personas directory: %s", root)

    _personas_cache = PersonasConfig(personas=persona_store.scan_personas_directory(root))
    return _personas_cache


def _personas_directory_populated(root: Path) -> bool:
    """True when the directory exists and holds at least one persona subdirectory."""
    if not root.is_dir():
        return False
    return any(entry.is_dir() for entry in root.iterdir())


def set_personas_cache(config: PersonasConfig) -> None:
    """Replace the in-memory persona cache without touching the disk.

    The persona router calls this after every directory mutation; the
    directory on disk is the source of truth, so there is nothing to
    persist here. Skipping this step is how the UI ends up stale until
    the next restart — it has happened before.
    """
    global _personas_cache
    _personas_cache = config


def load_chatrooms(path: Optional[Path] = None) -> ChatRoomsConfig:
    """Parse chatrooms.yaml. Returns an empty config if the file is missing."""
    global _chatrooms_cache
    target = path or config_path(CHATROOMS_FILE)
    if not target.exists():
        return ChatRoomsConfig()
    raw, notes = migrate_chatrooms(_read_raw(target))
    _log_notes(CHATROOMS_FILE, notes)
    _chatrooms_cache = ChatRoomsConfig(
        chat_rooms=[ChatRoom(**cr) for cr in raw.get("chat_rooms", [])]
    )
    return _chatrooms_cache


def load_player(path: Optional[Path] = None) -> PlayerConfig:
    """Parse player.yaml. Playing as yourself if the file is missing."""
    global _player_cache
    target = path or config_path(PLAYER_FILE)
    if not target.exists():
        # The miss is cached too. get_player() is called several times per
        # persona per turn (eligibility, then the preamble), so on the
        # common "playing as yourself" install — no player.yaml at all —
        # every one of those calls was stat-ing the disk twice.
        _player_cache = PlayerConfig()
        return _player_cache
    raw, notes = migrate_player(_read_raw(target))
    _log_notes(PLAYER_FILE, notes)
    _player_cache = PlayerConfig(persona_name=str(raw.get("persona_name", "") or ""))
    return _player_cache


def get_player() -> PlayerConfig:
    """Return the cached player config, loading if necessary."""
    if _player_cache is None:
        return load_player()
    return _player_cache


def save_player(config: PlayerConfig, path: Optional[Path] = None) -> None:
    """Write player.yaml into config/ and update the in-memory cache."""
    global _player_cache
    target = path or config_dir() / PLAYER_FILE
    _write_raw(target, {"persona_name": config.persona_name})
    _player_cache = config


def get_settings() -> AppSettings:
    """Return cached settings, loading if necessary."""
    if _settings_cache is None:
        return load_settings()
    return _settings_cache


def get_personas() -> PersonasConfig:
    """Return cached personas, loading if necessary."""
    if _personas_cache is None:
        return load_personas()
    return _personas_cache


def get_chatrooms() -> ChatRoomsConfig:
    """Return cached chat rooms, loading if necessary."""
    if _chatrooms_cache is None:
        return load_chatrooms()
    return _chatrooms_cache


def save_settings(config: AppSettings, path: Optional[Path] = None) -> None:
    """Write settings.yaml into config/ and update the in-memory cache."""
    global _settings_cache
    target = path or config_dir() / SETTINGS_FILE
    _write_raw(target, {
        "llm": config.llm.model_dump(mode="json", exclude_none=False),
        "tts": config.tts.model_dump(mode="json", exclude_none=False),
        "stt": config.stt.model_dump(mode="json", exclude_none=False),
        "general": config.general.model_dump(mode="json", exclude_none=False),
        "mcp": config.mcp.model_dump(mode="json", exclude_none=False),
    })
    _settings_cache = config


def save_chatrooms(config: ChatRoomsConfig, path: Optional[Path] = None) -> None:
    """Write chatrooms.yaml into config/ and update the in-memory cache."""
    global _chatrooms_cache
    target = path or config_dir() / CHATROOMS_FILE
    _write_raw(target, {
        "chat_rooms": [cr.model_dump(mode="json", exclude_none=False) for cr in config.chat_rooms]
    })
    _chatrooms_cache = config


def migrate_config_files() -> list:
    """Copy any repo-root config into config/, upgrading the schema.

    Runs once at startup. It **copies**, never moves or deletes: the root
    files are tracked by git, and removing them is exactly what produced
    "Your local changes to the following files would be overwritten by
    merge" on the next pull. After this runs the root copies are inert —
    edits there are ignored, and `git restore` on them is safe.

    Returns the list of filenames that were migrated, for logging.
    """
    migrated = []

    # personas.yaml is a *source* for the one-time directory migration now,
    # not a live file. Once Personas/ holds anything, copying the tracked
    # root copy back into config/ would hand load_personas() a legacy file
    # to migrate all over again — and it would warn about the conflict on
    # every single startup.
    files = [
        (SETTINGS_FILE, migrate_settings),
        (PERSONAS_FILE, migrate_personas),
        (CHATROOMS_FILE, migrate_chatrooms),
        (PLAYER_FILE, migrate_player),
    ]
    if _personas_directory_populated(get_personas_directory()):
        files = [f for f in files if f[0] != PERSONAS_FILE]

    for filename, migrate in files:
        live = config_dir() / filename
        legacy = legacy_path(filename)
        if live.exists() or not legacy.exists():
            continue
        raw, notes = migrate(_read_raw(legacy))
        _log_notes(filename, notes)
        _write_raw(live, raw)
        migrated.append(filename)

    # Files already in config/ are migrated in memory on every load, but
    # nothing rewrites them until the next save, so a stale key can sit on
    # disk indefinitely. Bring any out-of-date file up to the current
    # schema here, once, so what is on disk matches what the app reads.
    for filename, migrate in files:
        live = config_dir() / filename
        if filename in migrated or not live.exists():
            continue
        raw = _read_raw(live)
        try:
            on_disk = int(raw.get("schema_version", 1))
        except (TypeError, ValueError):
            on_disk = 1
        if on_disk >= CONFIG_SCHEMA_VERSION:
            # Ahead of us means a newer release wrote it. _apply() refuses
            # to touch such a file, so rewriting it here only churned the
            # mtime and reported a migration that never happened.
            continue
        migrated_raw, notes = migrate(raw)
        _log_notes(filename, notes)
        _write_raw(live, migrated_raw)
        migrated.append(filename)

    return migrated


def reload_all():
    """Force-reload all config files. Useful for dev hot-reload."""
    global _settings_cache, _personas_cache, _chatrooms_cache, _player_cache
    _settings_cache = None
    _personas_cache = None
    _chatrooms_cache = None
    _player_cache = None
    load_settings()
    load_personas()
    load_chatrooms()
    load_player()
