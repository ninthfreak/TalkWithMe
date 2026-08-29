"""Configuration loading and validation.

Loads settings.yaml and personas.yaml from the project root.
Caches parsed config so we're not hitting disk on every request.
"""

import logging
import math
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

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
    """How long a persona's replies should typically be."""

    TERSE = "terse"
    BRIEF = "brief"
    NORMAL = "normal"
    DETAILED = "detailed"
    UNRESTRICTED = "unrestricted"


class LengthSpec(NamedTuple):
    """A tier's word target and the phrasing used in the prompt.

    words == 0 means "no guidance": no prompt line and no derived cap.
    """

    words: int
    phrasing: str


TYPICAL_LENGTH_SPECS: Dict[TypicalLength, LengthSpec] = {
    TypicalLength.TERSE: LengthSpec(25, "one or two short sentences"),
    TypicalLength.BRIEF: LengthSpec(60, "two to four sentences"),
    TypicalLength.NORMAL: LengthSpec(120, "a short paragraph"),
    TypicalLength.DETAILED: LengthSpec(250, "a few paragraphs"),
    TypicalLength.UNRESTRICTED: LengthSpec(0, ""),
}

# Rough English tokens-per-word for the derived cap. Deliberately not a
# setting — it is a constant of the encoding, not a preference.
_TOKENS_PER_WORD = 1.4
# The cap sits well above the target so "go longer when it genuinely needs
# it" stays possible. Only a runaway reply should ever hit it.
_LENGTH_HEADROOM = 3.0
# Below this, the cap starts shaping replies again instead of guarding them
# — which is the exact failure this feature exists to remove. TERSE and
# BRIEF therefore share this floor; the prompt line is what separates them.
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
    guidance_scale: float = 3.0
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
    max_turns_for_context: int = Field(default=6, ge=1, le=50, description="Max history turns sent to the LLM")
    show_tool_calls: bool = True
    # Fallback tier, and the only one the implicit "default" room can use
    # (it has no chatrooms.yaml entry to carry an override).
    typical_length: TypicalLength = TypicalLength.NORMAL


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
    # None inherits the room's tier — a terse persona can stay terse in a
    # room of ramblers, but most personas should just follow the room.
    typical_length: Optional[TypicalLength] = None

    @property
    def tts_capable(self) -> bool:
        """TTS requires both reference audio AND its transcript."""
        return bool(self.reference_audio and self.reference_audio_transcript)


class PersonasConfig(BaseModel):
    personas: List[Persona] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Chat Rooms
# ---------------------------------------------------------------------------

class PlayerProfile(BaseModel):
    """The human user's own character in a room.

    Personas are told who they are talking to, so the user can be a
    character in the room's fiction rather than an anonymous prompt.

    *appearance* is the "picture": deliberately text, not an image. The
    LLM is the audience for this field, and it reads a description; an
    uploaded image would have to be captioned before it was any use.
    """

    name: str = Field(default="", max_length=40)
    description: str = Field(default="", max_length=2000)
    appearance: str = Field(default="", max_length=1000)

    @property
    def is_complete(self) -> bool:
        """True once the profile says who the player is.

        Appearance stays optional — a character can be described without
        being pictured.
        """
        return bool(self.name.strip() and self.description.strip())


class ChatRoom(BaseModel):
    """A named grouping of personas."""
    name: str
    persona_names: List[str] = Field(default_factory=list)
    echo_chamber: bool = False
    typical_length: TypicalLength = TypicalLength.NORMAL
    # When true, the room refuses messages until player_profile is complete.
    require_player_profile: bool = False
    player_profile: PlayerProfile = Field(default_factory=PlayerProfile)


class ChatRoomsConfig(BaseModel):
    chat_rooms: List[ChatRoom] = Field(default_factory=list)


def resolve_typical_length(
    persona: Optional[Persona],
    room: Optional[ChatRoom],
    general_default: TypicalLength,
) -> TypicalLength:
    """Resolve the effective tier: persona override, else room, else global.

    *room* is None for the implicit "default" room, which has no
    chatrooms.yaml entry and therefore falls back to the global value.
    """
    if persona is not None and persona.typical_length is not None:
        return persona.typical_length
    if room is not None:
        return room.typical_length
    return general_default


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_settings_cache: Optional[AppSettings] = None
_personas_cache: Optional[PersonasConfig] = None
_chatrooms_cache: Optional[ChatRoomsConfig] = None


def load_settings(path: Optional[Path] = None) -> AppSettings:
    """Parse settings.yaml. Falls back to defaults if file is missing."""
    global _settings_cache
    target = path or _PROJECT_ROOT / "settings.yaml"
    if not target.exists():
        return AppSettings()
    with open(target) as f:
        raw = yaml.safe_load(f) or {}
    _settings_cache = AppSettings(
        llm=LLMSettings(**raw.get("llm", {})),
        tts=TTSConfig(**raw.get("tts", {})),
        stt=STTConfig(**raw.get("stt", {})),
        general=GeneralConfig(**raw.get("general", {})),
        mcp=MCPConfig(**raw.get("mcp", {})),
    )
    return _settings_cache


def load_personas(path: Optional[Path] = None) -> PersonasConfig:
    """Parse personas.yaml. Returns empty list if file is missing.

    Migrates the legacy 'language' key to 'reference_audio_language' on the
    fly, so existing personas.yaml files from before the rename still load
    without requiring manual edits.
    """
    global _personas_cache
    target = path or _PROJECT_ROOT / "personas.yaml"
    if not target.exists():
        return PersonasConfig()
    with open(target) as f:
        raw = yaml.safe_load(f) or {}
    migrated = []
    for p in raw.get("personas", []):
        if "language" in p and "reference_audio_language" not in p:
            name = p.get("name", "<unknown>")
            logger.info(
                "Persona '%s': migrating legacy 'language' key to 'reference_audio_language'",
                name,
            )
            p["reference_audio_language"] = p.pop("language")
        migrated.append(p)
    _personas_cache = PersonasConfig(personas=[Persona(**p) for p in migrated])
    return _personas_cache


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


def save_personas(config: PersonasConfig, path: Optional[Path] = None) -> None:
    """Serialize PersonasConfig back to personas.yaml and update the in-memory cache."""
    global _personas_cache
    target = path or _PROJECT_ROOT / "personas.yaml"
    # mode="json" keeps enums as plain strings; a bare model_dump() would
    # write a Python object tag into the YAML.
    raw = {
        "personas": [
            p.model_dump(mode="json", exclude_none=False)
            for p in config.personas
        ]
    }
    with open(target, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _personas_cache = config


def save_settings(config: AppSettings, path: Optional[Path] = None) -> None:
    """Serialize AppSettings back to settings.yaml and update the in-memory cache."""
    global _settings_cache
    target = path or _PROJECT_ROOT / "settings.yaml"
    raw = {
        "llm": config.llm.model_dump(mode="json", exclude_none=False),
        "tts": config.tts.model_dump(mode="json", exclude_none=False),
        "stt": config.stt.model_dump(mode="json", exclude_none=False),
        "general": config.general.model_dump(mode="json", exclude_none=False),
        "mcp": config.mcp.model_dump(mode="json", exclude_none=False),
    }
    with open(target, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _settings_cache = config


def get_chatrooms() -> ChatRoomsConfig:
    """Return cached chat rooms, loading if necessary."""
    if _chatrooms_cache is None:
        return load_chatrooms()
    return _chatrooms_cache


def load_chatrooms(path: Optional[Path] = None) -> ChatRoomsConfig:
    """Parse chatrooms.yaml. Returns empty config if file is missing."""
    global _chatrooms_cache
    target = path or _PROJECT_ROOT / "chatrooms.yaml"
    if not target.exists():
        return ChatRoomsConfig()
    with open(target) as f:
        raw = yaml.safe_load(f) or {}
    _chatrooms_cache = ChatRoomsConfig(
        chat_rooms=[ChatRoom(**cr) for cr in raw.get("chat_rooms", [])]
    )
    return _chatrooms_cache


def save_chatrooms(config: ChatRoomsConfig, path: Optional[Path] = None) -> None:
    """Serialize ChatRoomsConfig back to chatrooms.yaml and update the in-memory cache."""
    global _chatrooms_cache
    target = path or _PROJECT_ROOT / "chatrooms.yaml"
    raw = {
        "chat_rooms": [
            cr.model_dump(mode="json", exclude_none=False)
            for cr in config.chat_rooms
        ]
    }
    with open(target, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _chatrooms_cache = config


def reload_all():
    """Force-reload all config files. Useful for dev hot-reload."""
    global _settings_cache, _personas_cache, _chatrooms_cache
    _settings_cache = None
    _personas_cache = None
    _chatrooms_cache = None
    load_settings()
    load_personas()
    load_chatrooms()
