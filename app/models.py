"""Pydantic request / response models for the TalkWithMe API."""

from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import LengthBias, PromptFormat, TypicalLength

from app.config import DEFAULT_MEMORY_SIZE, MAX_MEMORY_SIZE

from app.services.persona_draft import MAX_REFINE_INSTRUCTION


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """User sends a message; optionally picks who answers."""
    message: str = Field(..., min_length=1, description="The user's message text")
    who_answers: str = Field(
        default="router",
        description='One of "router", "random", or a persona name',
    )
    chat_room: str = Field(
        default="default",
        description="The chat room this message belongs to (for persistence)",
    )
    message_id: Optional[str] = Field(
        default=None,
        description="Frontend-generated UUID for this message (for audio association)",
    )



class SpeakAsRequest(BaseModel):
    """Put words in a persona's mouth — the player writes their line."""
    persona: str = Field(..., min_length=1, description="Who says it")
    text: str = Field(..., min_length=1, max_length=8192)
    chat_room: str = Field(default="default")
    message_id: Optional[str] = Field(default=None)


class SuggestReplyRequest(BaseModel):
    """Ask the LLM to draft the player's next message."""
    chat_room: str = Field(default="default", description="Room to draft for")


class SuggestReplyResponse(BaseModel):
    """A drafted message for the player to review, edit and send."""
    text: str


# ---------------------------------------------------------------------------
# Persona drafting
# ---------------------------------------------------------------------------

class PersonaDraftRequest(BaseModel):
    """The specification for a new persona: a brief, dials, and details.

    Only ``brief`` is required. ``dials`` and ``details`` are free-form
    dicts here rather than a fixed model so that adding an axis is a
    one-line change in ``persona_draft`` — the service validates the keys
    and values, and silently drops anything it does not recognise.
    """
    brief: str = Field(..., min_length=1, max_length=2000)
    dials: Dict[str, str] = Field(default_factory=dict)
    details: Dict[str, str] = Field(default_factory=dict)


class PersonaDraftResponse(BaseModel):
    """A drafted persona, plus why it is shaped the way it is.

    Everything here lands in the editor form for review. Nothing is
    written to disk until the user saves, exactly like the suggested
    player message: the LLM drafts, the human decides.
    """
    name: str
    description: str
    system_prompt: str
    router_hints: str
    length_bias: LengthBias
    avatar_color: str
    # Which differentiation levers were used, and which the brief supplied.
    notes: List[str] = Field(default_factory=list)
    # Local checks the model is the wrong judge of (generic vocabulary, a
    # prompt too short to outweigh the room preamble, no negative space).
    warnings: List[str] = Field(default_factory=list)


class PersonaRefineRequest(BaseModel):
    """An existing persona, plus a sentence about what to change.

    The persona is sent whole rather than looked up by name: the user is
    looking at the editor form, and refining what is on disk would revise
    a version they cannot see and would throw away edits they have made
    but not yet saved.
    """
    name: str = Field(..., min_length=1, max_length=25)
    system_prompt: str = Field(..., min_length=1, max_length=8192)
    description: str = Field(default="", max_length=30)
    router_hints: str = Field(default="", max_length=256)
    length_bias: LengthBias = LengthBias.MATCH
    instruction: str = Field(..., min_length=1, max_length=MAX_REFINE_INSTRUCTION)


class PersonaRefineResponse(BaseModel):
    """A revised persona: only the fields a refinement is allowed to change.

    No name and no avatar colour on purpose. A refinement that renamed the
    character would be a different character, and one that repainted the
    avatar would be a surprise nobody asked for.
    """
    description: str
    system_prompt: str
    router_hints: str
    length_bias: LengthBias
    # What the model changed, and what it says it deliberately left alone.
    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PersonaPreviewRequest(BaseModel):
    """Try an unsaved draft against a question before committing to it."""
    name: str = Field(..., min_length=1, max_length=25)
    system_prompt: str = Field(..., min_length=1, max_length=8192)
    description: str = Field(default="", max_length=30)
    length_bias: LengthBias = LengthBias.MATCH
    question: str = Field(..., min_length=1, max_length=2000)
    # An existing persona to answer the same question, for comparison...
    compare_with: Optional[str] = Field(default=None, max_length=25)
    # ...or an unsaved prompt, which is how a refinement shows a before
    # and after of the same character. Exactly one of the two, or neither.
    compare_prompt: Optional[str] = Field(default=None, max_length=8192)
    compare_length_bias: LengthBias = LengthBias.MATCH
    # What to call each side. Both default to the persona's own name.
    label: str = Field(default="", max_length=40)
    compare_label: str = Field(default="", max_length=40)

    @model_validator(mode="after")
    def _one_comparison_at_most(self):
        if self.compare_with and self.compare_prompt:
            raise ValueError(
                "Compare against a saved persona or an unsaved prompt, not both"
            )
        return self


class PersonaPreviewReply(BaseModel):
    persona: str
    text: str


class PersonaPreviewResponse(BaseModel):
    """The draft's answer, and optionally an existing persona's, side by side."""
    draft: PersonaPreviewReply
    comparison: Optional[PersonaPreviewReply] = None


class SessionPersonasRequest(BaseModel):
    """Update which personas are active in the current session."""
    active_personas: List[str] = Field(
        ..., min_length=1, description="List of persona names to activate"
    )


# Persona create/update requests have no pydantic models on purpose: they
# are multipart/form-data (text fields + optional file uploads), so the
# shape is declared with FastAPI Form/File parameters on the router
# instead (see app/routers/personas.py).


class TTSRequest(BaseModel):
    """Proxy request to the TTS server."""
    text: str = Field(..., min_length=1, description="Text to synthesize")
    persona_name: str = Field(..., description="Which persona to synthesize for")


class STTRequest(BaseModel):
    """Proxy request to the STT server."""
    audio_base64: str = Field(..., min_length=1, description="Base64-encoded audio to transcribe")
    audio_mime_type: Optional[str] = Field(
        default="audio/webm",
        description="MIME type of the recorded audio (e.g. audio/webm, audio/ogg)",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PersonaResponse(BaseModel):
    """A single persona definition returned to the frontend."""
    name: str
    description: str
    avatar_color: str
    # Presence flag, not a path: the browser can't open filesystem paths,
    # and the file is owned by the persona directory (served via
    # GET /api/personas/{name}/avatar).
    avatar_image: bool = False
    tts_capable: bool = False


class PersonaDetailResponse(BaseModel):
    """Full persona detail including all editable fields.

    File-backed fields are reported as presence/contents, not paths:
    the avatar is a bool (file served via /avatar), the reference audio
    is a bool (file served via /reference-audio), and the transcript is
    the actual file contents (null if the file is absent).
    """
    name: str
    description: str
    system_prompt: str
    router_hints: str
    avatar_color: str
    avatar_image: bool = False
    reference_audio: bool = False
    reference_audio_transcript: Optional[str] = None
    reference_audio_language: str
    allow_tool_calls: bool = False
    length_bias: LengthBias = LengthBias.MATCH
    memory_size: int = Field(
        default=DEFAULT_MEMORY_SIZE,
        ge=0,
        le=MAX_MEMORY_SIZE,
        description="Size budget (bytes) for the persona's memories.txt; 0 disables memory saving",
    )
    tts_capable: bool = False


class SessionState(BaseModel):
    """Current session snapshot for the frontend."""
    history: List[dict] = Field(default_factory=list)
    active_personas: List[str] = Field(default_factory=list)
    current_room: str = Field(default="default", description="The currently active chat room")


class PersistedMessage(BaseModel):
    """A persisted chat message loaded from disk."""
    id: str
    sender: str
    text: str
    audio: List[str] = Field(default_factory=list)


class PersistedHistoryResponse(BaseModel):
    """Persisted chat history for a room."""
    room: str
    datetime: Optional[str] = None
    messages: List[PersistedMessage] = Field(default_factory=list)


class AudioUploadRequest(BaseModel):
    """Frontend uploads audio for a persisted message."""
    message_id: str
    audio_base64: str
    mime_type: Optional[str] = None


class TTSResponse(BaseModel):
    """Base64-encoded audio from the TTS server."""
    audio_base64: str
    sample_rate: int = 24000


class STTResponse(BaseModel):
    """Transcribed text from an OpenAI-compatible STT server."""
    text: str
    language: str = "en"
    language_probability: Optional[float] = None


class TTSHealthResponse(BaseModel):
    """TTS availability status."""
    enabled: bool
    available: bool = Field(
        default=False,
        description="True if the TTS server responded to /health",
    )
    streaming: bool = Field(
        default=False,
        description="True if streaming (sentence-by-sentence) TTS mode is configured",
    )
    server_type: Optional[str] = Field(
        default=None,
        description="Server type reported by the TTS server's /health endpoint (e.g. dots.tts)",
    )


class STTHealthResponse(BaseModel):
    """STT availability status."""
    enabled: bool
    available: bool = Field(
        default=False,
        description="True if the STT server responded to /health",
    )


# ---------------------------------------------------------------------------
# Settings models
# ---------------------------------------------------------------------------

class LLMSettingsRequest(BaseModel):
    """LLM configuration from the settings editor."""
    base_url: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    max_tokens: int = Field(..., ge=1)
    temperature: float = Field(..., ge=0.0, le=1.0)
    # Defaulted rather than required: an older frontend, or anything
    # posting the settings shape from before this existed, must not
    # silently reset how every persona is prompted.
    prompt_format: PromptFormat = PromptFormat.TRANSCRIPT


class TTSSettingsRequest(BaseModel):
    """TTS configuration from the settings editor."""
    enabled: bool = True
    base_url: str = Field(default="", min_length=0)
    num_steps: int = Field(..., ge=4, le=20)
    guidance_scale: float = Field(..., ge=1.0, le=2.0)
    seed: int = Field(default=0, description="0 means null (no seed)")
    timeout: float = Field(..., ge=5, le=300)
    streaming: bool = False


class STTSettingsRequest(BaseModel):
    """STT configuration from the settings editor."""
    enabled: bool = True
    base_url: str = Field(default="", min_length=0)
    timeout: float = Field(..., ge=5, le=300)


class GeneralSettingsRequest(BaseModel):
    """General configuration from the settings editor.

    A partial update: all fields are optional, and omitted fields (None)
    keep their current values (see update_settings in routers/settings.py).
    With required-with-default fields, any client that didn't manage a
    field silently reset it to its default — e.g. the Servers dialog used
    to wipe out show_tool_calls on every save.
    """
    persona_name_mentions: Optional[bool] = None
    max_persona_replies: Optional[int] = Field(default=None, ge=1, le=6)
    max_turns_for_context: Optional[int] = Field(default=None, ge=1, le=50)
    show_tool_calls: Optional[bool] = None
    typical_length: Optional[TypicalLength] = None
    enable_persona_memories: Optional[bool] = None


class SettingsUpdateRequest(BaseModel):
    """Full settings payload from the frontend settings editor."""
    llm: LLMSettingsRequest
    tts: TTSSettingsRequest
    stt: STTSettingsRequest
    general: GeneralSettingsRequest = GeneralSettingsRequest()


class LLMSettingsResponse(BaseModel):
    """LLM configuration for the frontend."""
    base_url: str
    model: str
    max_tokens: int
    temperature: float
    prompt_format: PromptFormat


class TTSSettingsResponse(BaseModel):
    """TTS configuration for the frontend."""
    enabled: bool
    base_url: Optional[str] = None
    num_steps: int
    guidance_scale: float
    seed: Optional[int] = None
    timeout: float
    streaming: bool


class STTSettingsResponse(BaseModel):
    """STT configuration for the frontend."""
    enabled: bool
    base_url: Optional[str] = None
    timeout: float


class GeneralSettingsResponse(BaseModel):
    """General configuration for the frontend."""
    persona_name_mentions: bool
    max_persona_replies: int
    max_turns_for_context: int
    show_tool_calls: bool
    typical_length: TypicalLength
    enable_persona_memories: bool


class SettingsResponse(BaseModel):
    """Full settings payload returned to the frontend."""
    llm: LLMSettingsResponse
    tts: TTSSettingsResponse
    stt: STTSettingsResponse
    general: GeneralSettingsResponse


# ---------------------------------------------------------------------------
# Internal models (not exposed over the API)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single turn in the conversation history."""
    role: Literal["user", "assistant"]
    content: str
    # Which persona produced this message (only set for assistant messages)
    persona: Optional[str] = None
    # True when the LLM stopped at max_tokens rather than finishing. Affects
    # only how the message is rendered to *other* personas (see
    # build_llm_messages) — never what is displayed or persisted. Kept out of
    # the on-disk format: it is a property of one generation, not the message.
    truncated: bool = False


# ---------------------------------------------------------------------------
# Chat Room models
# ---------------------------------------------------------------------------

class ChatRoomResponse(BaseModel):
    """A chat room returned to the frontend."""
    name: str
    persona_names: List[str] = Field(default_factory=list)
    typical_length: TypicalLength = TypicalLength.NORMAL
    require_player_persona: bool = False


class ChatRoomCreateRequest(BaseModel):
    """Create a new chat room."""
    name: str = Field(..., min_length=1, max_length=20)


class AssignPersonasRequest(BaseModel):
    """Assign personas to a chat room."""
    persona_names: List[str] = Field(..., min_length=1)


class AdoptPersonaRequest(BaseModel):
    """Which persona the player is playing. Empty means "themselves"."""
    persona_name: str = Field(default="", max_length=25)


class ChatRoomUpdateRequest(BaseModel):
    """Partial update of a chat room's settings.

    Every field is optional and omitted fields keep their current value —
    the same contract `GeneralSettingsRequest` uses. This is deliberately
    one endpoint rather than one per attribute: a new room attribute means
    adding a field here and a control in the room editor, not another
    route, another request model and another frontend fetch. The four
    single-purpose endpoints this replaces (echo-chamber, typical-length,
    player-profile, require-player-profile) were already drifting apart.

    `require_player_persona` was `require_player_profile` until schema 6;
    the flag is the same room property, but what it demands is now an
    adopted persona rather than a written profile.

    Membership is not here: personas have their own endpoints because
    assigning one validates against the persona list and can 422, which is
    a different operation from setting a flag.

    Unknown fields are rejected rather than ignored. On a partial-update
    endpoint a mistyped key would otherwise be indistinguishable from a
    successful save — the room comes back unchanged and nothing complains.
    """
    model_config = ConfigDict(extra="forbid")

    typical_length: Optional[TypicalLength] = None
    require_player_persona: Optional[bool] = None


class PlayerResponse(BaseModel):
    """Who the player is currently playing, as returned to the frontend."""
    persona_name: str = ""
