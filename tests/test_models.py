"""Tests for app/models.py — API request/response model validation."""

import pytest
from pydantic import ValidationError

from app.models import (
    AdoptPersonaRequest,
    AssignPersonasRequest,
    AudioUploadRequest,
    ChatRequest,
    ChatRoomCreateRequest,
    ChatRoomUpdateRequest,
    GeneralSettingsRequest,
    PersonaCreateRequest,
    SessionPersonasRequest,
    SettingsUpdateRequest,
    SpeakAsRequest,
    STTRequest,
    TTSRequest,
)


class TestChatRequest:
    def test_chat_request_defaults(self):
        req = ChatRequest(message="hello")
        assert req.who_answers == "router"
        assert req.chat_room == "default"
        assert req.message_id is None

    def test_chat_request_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_request_accepts_all_fields(self):
        req = ChatRequest(
            message="hi", who_answers="random", chat_room="TNG", message_id="abc-123"
        )
        assert req.message_id == "abc-123"


class TestSpeakAsRequest:
    def test_defaults(self):
        req = SpeakAsRequest(persona="Luna", text="I disagree.")
        assert req.chat_room == "default"
        assert req.message_id is None

    @pytest.mark.parametrize("field, value", [
        ("persona", ""),
        ("text", ""),
        ("text", "x" * 8193),
    ])
    def test_blank_or_oversized_values_are_rejected(self, field, value):
        payload = {"persona": "Luna", "text": "hi", field: value}
        with pytest.raises(ValidationError):
            SpeakAsRequest(**payload)


class TestAdoptPersonaRequest:
    def test_an_empty_name_means_playing_yourself(self):
        assert AdoptPersonaRequest().persona_name == ""

    def test_the_cap_matches_the_persona_name_cap(self):
        # No name that gets through here could ever fail to be a persona
        # name for length reasons alone.
        assert AdoptPersonaRequest(persona_name="K" * 25).persona_name == "K" * 25
        with pytest.raises(ValidationError):
            AdoptPersonaRequest(persona_name="K" * 26)


class TestSessionPersonasRequest:
    def test_session_personas_request_requires_at_least_one(self):
        with pytest.raises(ValidationError):
            SessionPersonasRequest(active_personas=[])

    def test_session_personas_request_accepts_names(self):
        req = SessionPersonasRequest(active_personas=["Alex", "Luna"])
        assert req.active_personas == ["Alex", "Luna"]


class TestPersonaCreateRequest:
    def _valid(self) -> dict:
        return {
            "name": "Zed",
            "system_prompt": "You are Zed.",
            "router_hints": "testing",
        }

    def test_persona_create_request_valid(self):
        req = PersonaCreateRequest(**self._valid())
        assert req.name == "Zed"
        assert req.reference_audio_language == "en"
        assert req.allow_tool_calls is False

    def test_persona_create_request_name_over_25_chars_rejected(self):
        fields = self._valid()
        fields["name"] = "x" * 26
        with pytest.raises(ValidationError):
            PersonaCreateRequest(**fields)

    def test_persona_create_request_blank_name_rejected(self):
        fields = self._valid()
        fields["name"] = ""
        with pytest.raises(ValidationError):
            PersonaCreateRequest(**fields)

    def test_persona_create_request_missing_router_hints_rejected(self):
        fields = self._valid()
        del fields["router_hints"]
        with pytest.raises(ValidationError):
            PersonaCreateRequest(**fields)

    def test_persona_create_request_bad_language_length_rejected(self):
        fields = self._valid()
        fields["reference_audio_language"] = "eng"
        with pytest.raises(ValidationError):
            PersonaCreateRequest(**fields)


class TestTTSRequest:
    def test_tts_request_missing_required_fields_rejected(self):
        with pytest.raises(ValidationError):
            TTSRequest()  # both text and persona_name are required
        with pytest.raises(ValidationError):
            TTSRequest(text="hello")  # persona_name missing

    def test_tts_request_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            TTSRequest(text="", persona_name="Luna")  # min_length=1

    def test_tts_request_valid(self):
        req = TTSRequest(text="hello", persona_name="Luna")
        assert req.persona_name == "Luna"


class TestSTTRequest:
    def test_stt_request_requires_audio(self):
        with pytest.raises(ValidationError):
            STTRequest(audio_base64="")

    def test_stt_request_default_mime_type_is_webm(self):
        req = STTRequest(audio_base64="QUJD")
        assert req.audio_mime_type == "audio/webm"


class TestChatRoomModels:
    def test_chat_room_create_request_name_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ChatRoomCreateRequest(name="x" * 21)

    def test_assign_personas_request_requires_at_least_one(self):
        with pytest.raises(ValidationError):
            AssignPersonasRequest(persona_names=[])

    def test_room_update_omits_what_it_does_not_set(self):
        # The partial-update contract: only what was sent is applied.
        req = ChatRoomUpdateRequest(require_player_persona=True)
        assert req.model_dump(exclude_none=True) == {"require_player_persona": True}

    def test_room_update_rejects_an_unknown_key(self):
        # extra="forbid" on purpose: on a partial-update endpoint a typo
        # would otherwise look exactly like a successful save.
        with pytest.raises(ValidationError):
            ChatRoomUpdateRequest(require_player_profile=True)


class TestAudioUploadRequest:
    def test_audio_upload_request_missing_required_fields_rejected(self):
        with pytest.raises(ValidationError):
            AudioUploadRequest()  # both message_id and audio_base64 are required
        with pytest.raises(ValidationError):
            AudioUploadRequest(message_id="m1")  # audio_base64 missing

    def test_audio_upload_request_mime_type_optional(self):
        req = AudioUploadRequest(message_id="m1", audio_base64="QUJD")
        assert req.mime_type is None


class TestGeneralSettingsRequestPartialUpdate:
    """The general section is a partial update: every field must be
    optional so omitted fields keep their current server-side values."""

    def test_general_settings_request_all_fields_omitted(self):
        req = GeneralSettingsRequest()
        assert req.model_dump() == {
            "persona_name_mentions": None,
            "max_persona_replies": None,
            "max_turns_for_context": None,
            "show_tool_calls": None,
            "typical_length": None,
        }

    def test_general_settings_request_exclude_none_drops_omitted_fields(self):
        req = GeneralSettingsRequest(show_tool_calls=False)
        dumped = req.model_dump(exclude_none=True)
        assert dumped == {"show_tool_calls": False}

    def test_general_settings_request_out_of_range_replies_rejected(self):
        with pytest.raises(ValidationError):
            GeneralSettingsRequest(max_persona_replies=7)

    def test_general_settings_request_accepts_the_top_of_the_range(self):
        assert GeneralSettingsRequest(max_persona_replies=6).max_persona_replies == 6

    def test_general_settings_request_out_of_range_context_rejected(self):
        with pytest.raises(ValidationError):
            GeneralSettingsRequest(max_turns_for_context=0)


class TestSettingsUpdateRequest:
    def test_settings_update_request_requires_llm_tts_stt(self):
        with pytest.raises(ValidationError):
            SettingsUpdateRequest(llm={"base_url": "http://x", "model": "m",
                                       "max_tokens": 1, "temperature": 0.5})

    def test_settings_update_request_general_defaults_to_empty_partial_update(self):
        req = SettingsUpdateRequest(
            llm={"base_url": "http://x", "model": "m", "max_tokens": 1, "temperature": 0.5},
            tts={"num_steps": 4, "guidance_scale": 1.0, "timeout": 5},
            stt={"timeout": 5},
        )
        # No general section sent: everything stays None -> nothing overrides.
        assert req.general.model_dump(exclude_none=True) == {}
