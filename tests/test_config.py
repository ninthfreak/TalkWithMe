"""Tests for app/config.py — model validation and YAML load/save behaviour."""

import yaml

import pytest
from pydantic import ValidationError

from app import config as app_config
from app.config import (
    AppSettings,
    ChatRoom,
    ChatRoomsConfig,
    GeneralConfig,
    LLMSettings,
    MCPServerConfig,
    Persona,
    PersonasConfig,
    LENGTH_SCALE,
    TYPICAL_LENGTH_SPECS,
    LengthBias,
    PlayerProfile,
    STTConfig,
    TTSConfig,
    TypicalLength,
    derive_max_tokens,
    load_chatrooms,
    load_personas,
    resolve_typical_length,
    save_chatrooms,
    save_personas,
)
from tests.factories import make_chatrooms, make_personas, make_settings


# ---------------------------------------------------------------------------
# TTS / STT is_active semantics
# ---------------------------------------------------------------------------

class TestTTSConfigIsActive:
    def test_tts_config_enabled_without_base_url_is_not_active(self):
        assert TTSConfig(enabled=True).is_active is False

    def test_tts_config_disabled_with_base_url_is_not_active(self):
        assert TTSConfig(enabled=False, base_url="http://tts:1").is_active is False

    def test_tts_config_enabled_with_base_url_is_active(self):
        assert TTSConfig(enabled=True, base_url="http://tts:1").is_active is True

    def test_tts_config_blank_base_url_is_normalized_to_none(self):
        cfg = TTSConfig(enabled=True, base_url="   ")
        assert cfg.base_url is None
        assert cfg.is_active is False


class TestSTTConfigIsActive:
    def test_stt_config_enabled_without_base_url_is_not_active(self):
        assert STTConfig(enabled=True).is_active is False

    def test_stt_config_disabled_with_base_url_is_not_active(self):
        assert STTConfig(enabled=False, base_url="http://stt:1").is_active is False

    def test_stt_config_enabled_with_base_url_is_active(self):
        assert STTConfig(enabled=True, base_url="http://stt:1").is_active is True

    def test_stt_config_blank_base_url_is_normalized_to_none(self):
        cfg = STTConfig(enabled=True, base_url="")
        assert cfg.base_url is None
        assert cfg.is_active is False


# ---------------------------------------------------------------------------
# GeneralConfig bounds
# ---------------------------------------------------------------------------

class TestGeneralConfigBounds:
    @pytest.mark.parametrize("value", [0, 7, -1])
    def test_general_config_max_persona_replies_out_of_range_rejected(self, value):
        with pytest.raises(ValidationError):
            GeneralConfig(max_persona_replies=value)

    @pytest.mark.parametrize("value", [1, 4, 6])
    def test_general_config_max_persona_replies_in_range_accepted(self, value):
        assert GeneralConfig(max_persona_replies=value).max_persona_replies == value

    @pytest.mark.parametrize("value", [0, 51])
    def test_general_config_max_turns_for_context_out_of_range_rejected(self, value):
        with pytest.raises(ValidationError):
            GeneralConfig(max_turns_for_context=value)


# ---------------------------------------------------------------------------
# MCP server config validation
# ---------------------------------------------------------------------------

class TestMCPServerConfig:
    def test_mcp_server_config_schemeless_url_rejected(self):
        with pytest.raises(ValidationError, match="must start with http"):
            MCPServerConfig(name="broken", url="localhost:9000")

    @pytest.mark.parametrize("url", ["http://mcp:9000", "https://mcp.example.com/rpc"])
    def test_mcp_server_config_valid_schemes_accepted(self, url):
        assert MCPServerConfig(name="ok", url=url).url == url

    @pytest.mark.parametrize("timeout", [0, -1, 301])
    def test_mcp_server_config_timeout_out_of_range_rejected(self, timeout):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="ok", url="http://mcp:9000", timeout=timeout)

    def test_mcp_server_config_default_timeout_is_ten_seconds(self):
        assert MCPServerConfig(name="ok", url="http://mcp:9000").timeout == 10.0


# ---------------------------------------------------------------------------
# Persona model
# ---------------------------------------------------------------------------

class TestPersona:
    def test_persona_tts_capable_requires_audio_and_transcript(self):
        assert Persona(name="A", system_prompt="p", reference_audio="a.wav").tts_capable is False
        assert (
            Persona(name="A", system_prompt="p", reference_audio="a.wav",
                    reference_audio_transcript="a.txt")
            .tts_capable
            is True
        )


# ---------------------------------------------------------------------------
# Loading: missing files fall back to defaults
# ---------------------------------------------------------------------------

class TestLoadingFallbacks:
    def test_load_settings_missing_file_returns_defaults(self, tmp_path):
        settings = app_config.load_settings(tmp_path / "nope.yaml")
        assert settings.llm.base_url == "http://localhost:8080"
        assert settings.mcp.servers == []

    def test_load_personas_missing_file_returns_empty(self, tmp_path):
        cfg = app_config.load_personas(tmp_path / "nope.yaml")
        assert cfg.personas == []

    def test_load_chatrooms_missing_file_returns_empty(self, tmp_path):
        cfg = app_config.load_chatrooms(tmp_path / "nope.yaml")
        assert cfg.chat_rooms == []

    def test_load_settings_empty_file_returns_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert app_config.load_settings(path) == AppSettings()


# ---------------------------------------------------------------------------
# Loading: content and migration
# ---------------------------------------------------------------------------

class TestLoadingContent:
    def test_load_settings_parses_all_sections(self, tmp_path):
        path = tmp_path / "settings.yaml"
        path.write_text(
            """
llm:
  base_url: http://custom:1234
  model: custom-model
tts:
  enabled: true
  base_url: http://tts:1
general:
  show_tool_calls: false
mcp:
  servers:
    - name: my-server
      url: http://mcp:9000
"""
        )
        settings = app_config.load_settings(path)
        assert settings.llm.base_url == "http://custom:1234"
        assert settings.tts.is_active is True
        assert settings.general.show_tool_calls is False
        assert settings.mcp.servers[0].name == "my-server"

    def test_load_personas_migrates_legacy_language_key(self, tmp_path):
        path = tmp_path / "personas.yaml"
        path.write_text(
            """
personas:
  - name: Alex
    system_prompt: You are Alex.
    language: de
"""
        )
        cfg = app_config.load_personas(path)
        assert cfg.personas[0].reference_audio_language == "de"

    def test_load_personas_keeps_explicit_reference_audio_language(self, tmp_path):
        # If both keys exist, the new key wins and the legacy one is ignored.
        path = tmp_path / "personas.yaml"
        path.write_text(
            """
personas:
  - name: Alex
    system_prompt: You are Alex.
    language: de
    reference_audio_language: es
"""
        )
        cfg = app_config.load_personas(path)
        assert cfg.personas[0].reference_audio_language == "es"

    def test_load_chatrooms_parses_rooms(self, tmp_path):
        path = tmp_path / "chatrooms.yaml"
        path.write_text(
            """
chat_rooms:
  - name: TNG
    persona_names: [Alex]
    echo_chamber: true
"""
        )
        cfg = app_config.load_chatrooms(path)
        assert cfg.chat_rooms[0] == ChatRoom(name="TNG", persona_names=["Alex"], echo_chamber=True)


# ---------------------------------------------------------------------------
# Save/load round-trips
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def test_save_settings_round_trip(self, tmp_path):
        path = tmp_path / "settings.yaml"
        settings = make_settings(
            general=GeneralConfig(max_persona_replies=3, show_tool_calls=False),
        )
        app_config.save_settings(settings, path)
        assert path.exists()
        reloaded = yaml.safe_load(path.read_text())
        assert reloaded["general"]["max_persona_replies"] == 3
        assert reloaded["general"]["show_tool_calls"] is False

    def test_save_personas_round_trip(self, tmp_path):
        path = tmp_path / "personas.yaml"
        cfg = make_personas()
        app_config.save_personas(cfg, path)
        reloaded = app_config.load_personas(path)
        assert [p.name for p in reloaded.personas] == ["Alex", "Luna"]
        assert reloaded.personas[1].reference_audio == "reference/luna.wav"

    def test_save_chatrooms_round_trip(self, tmp_path):
        path = tmp_path / "chatrooms.yaml"
        cfg = make_chatrooms()
        app_config.save_chatrooms(cfg, path)
        reloaded = app_config.load_chatrooms(path)
        assert [r.name for r in reloaded.chat_rooms] == ["TNG"]


# ---------------------------------------------------------------------------
# Cache semantics
# ---------------------------------------------------------------------------

class TestCaching:
    def test_get_settings_returns_cache_without_reloading(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.yaml"
        settings = make_settings()
        monkeypatch.setattr(app_config, "_settings_cache", settings)
        # Mutate the file after caching: get_settings must not see it.
        path.write_text("llm:\n  base_url: http://changed:1\n")
        assert app_config.get_settings() is settings

    def test_get_personas_returns_cache_without_reloading(self, tmp_path, monkeypatch):
        cfg = make_personas()
        monkeypatch.setattr(app_config, "_personas_cache", cfg)
        assert app_config.get_personas() is cfg

    def test_reload_all_replaces_caches(self, tmp_path, monkeypatch):
        # Point the module's project root at tmp so reload_all reads tmp files.
        monkeypatch.setattr(app_config, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "settings.yaml").write_text("llm:\n  base_url: http://reloaded:1\n")
        (tmp_path / "personas.yaml").write_text(
            "personas:\n  - name: Fresh\n    system_prompt: p\n"
        )
        (tmp_path / "chatrooms.yaml").write_text(
            "chat_rooms:\n  - name: NewRoom\n"
        )
        app_config.reload_all()
        assert app_config.get_settings().llm.base_url == "http://reloaded:1"
        assert [p.name for p in app_config.get_personas().personas] == ["Fresh"]
        assert [r.name for r in app_config.get_chatrooms().chat_rooms] == ["NewRoom"]


# ---------------------------------------------------------------------------
# Typical response length
# ---------------------------------------------------------------------------

class TestTypicalLengthDefaults:
    def test_persona_defaults_to_matching_the_room(self):
        assert Persona(name="A", system_prompt="p").length_bias is LengthBias.MATCH

    def test_room_and_global_default_to_normal(self):
        assert ChatRoom(name="R").typical_length is TypicalLength.NORMAL
        assert GeneralConfig().typical_length is TypicalLength.NORMAL


class TestChatCalibration:
    """The scale is for chat, not prose — a real chat line is a fragment,
    a sentence, or two when the thought needs it."""

    def test_normal_is_a_sentence_or_two(self):
        spec = TYPICAL_LENGTH_SPECS[TypicalLength.NORMAL]
        assert spec.words <= 25
        assert spec.phrasing == "a sentence or two"

    def test_even_the_longest_tier_is_only_a_paragraph(self):
        assert TYPICAL_LENGTH_SPECS[TypicalLength.VERBOSE].words <= 120

    def test_scale_increases_monotonically(self):
        words = [TYPICAL_LENGTH_SPECS[t].words for t in LENGTH_SCALE]
        assert words == sorted(words)
        assert len(set(words)) == len(words)

    def test_unrestricted_is_outside_the_ordinal_scale(self):
        # It means "no target", so there is nothing to be relative to.
        assert TypicalLength.UNRESTRICTED not in LENGTH_SCALE
        assert TYPICAL_LENGTH_SPECS[TypicalLength.UNRESTRICTED].words == 0


class TestResolveTypicalLength:
    @pytest.mark.parametrize("bias, expected", [
        (LengthBias.MUCH_SHORTER, TypicalLength.TERSE),
        (LengthBias.SHORTER, TypicalLength.BRIEF),
        (LengthBias.MATCH, TypicalLength.NORMAL),
        (LengthBias.LONGER, TypicalLength.DETAILED),
        (LengthBias.MUCH_LONGER, TypicalLength.VERBOSE),
    ])
    def test_bias_moves_along_the_rooms_scale(self, bias, expected):
        persona = Persona(name="A", system_prompt="p", length_bias=bias)
        room = ChatRoom(name="R", typical_length=TypicalLength.NORMAL)
        assert resolve_typical_length(persona, room, TypicalLength.NORMAL) is expected

    def test_the_same_persona_shifts_with_the_room(self):
        # The point of a relative bias: a laconic persona is laconic *for
        # the room*, not laconic in absolute terms.
        persona = Persona(name="A", system_prompt="p", length_bias=LengthBias.SHORTER)
        in_terse = resolve_typical_length(
            persona, ChatRoom(name="R", typical_length=TypicalLength.BRIEF), TypicalLength.NORMAL
        )
        in_wordy = resolve_typical_length(
            persona, ChatRoom(name="R", typical_length=TypicalLength.VERBOSE), TypicalLength.NORMAL
        )
        assert in_terse is TypicalLength.TERSE
        assert in_wordy is TypicalLength.DETAILED

    @pytest.mark.parametrize("base, bias, expected", [
        (TypicalLength.TERSE, LengthBias.MUCH_SHORTER, TypicalLength.TERSE),
        (TypicalLength.BRIEF, LengthBias.MUCH_SHORTER, TypicalLength.TERSE),
        (TypicalLength.VERBOSE, LengthBias.MUCH_LONGER, TypicalLength.VERBOSE),
        (TypicalLength.DETAILED, LengthBias.MUCH_LONGER, TypicalLength.VERBOSE),
    ])
    def test_bias_clamps_at_both_ends(self, base, bias, expected):
        # A laconic persona in an already-terse room is terse, not silent.
        persona = Persona(name="A", system_prompt="p", length_bias=bias)
        room = ChatRoom(name="R", typical_length=base)
        assert resolve_typical_length(persona, room, TypicalLength.NORMAL) is expected

    def test_unrestricted_room_ignores_bias(self):
        persona = Persona(name="A", system_prompt="p", length_bias=LengthBias.MUCH_SHORTER)
        room = ChatRoom(name="R", typical_length=TypicalLength.UNRESTRICTED)
        assert resolve_typical_length(persona, room, TypicalLength.NORMAL) is (
            TypicalLength.UNRESTRICTED
        )

    def test_global_used_when_there_is_no_room(self):
        # room=None is the implicit "default" room: no entry, no override.
        persona = Persona(name="A", system_prompt="p")
        assert resolve_typical_length(persona, None, TypicalLength.BRIEF) is TypicalLength.BRIEF

    def test_bias_applies_to_the_global_default_too(self):
        persona = Persona(name="A", system_prompt="p", length_bias=LengthBias.LONGER)
        assert resolve_typical_length(persona, None, TypicalLength.BRIEF) is (
            TypicalLength.NORMAL
        )


class TestDeriveMaxTokens:
    def test_chat_tiers_all_land_on_the_floor(self):
        # At chat lengths the derived cap would be tiny, so the floor takes
        # over. That is intended: the prompt shapes the reply, the cap only
        # stops a runaway. A lower floor would start truncating again.
        for tier in [TypicalLength.TERSE, TypicalLength.BRIEF,
                     TypicalLength.NORMAL, TypicalLength.DETAILED]:
            assert derive_max_tokens(tier, 1024) == 256

    def test_cap_sits_well_above_the_word_target(self):
        spec = TYPICAL_LENGTH_SPECS[TypicalLength.VERBOSE]
        cap = derive_max_tokens(TypicalLength.VERBOSE, 1024)
        assert cap > spec.words * 2

    def test_configured_ceiling_is_never_exceeded(self):
        assert derive_max_tokens(TypicalLength.VERBOSE, 300) == 300
        assert derive_max_tokens(TypicalLength.NORMAL, 100) == 100

    def test_unrestricted_uses_the_ceiling_unchanged(self):
        assert derive_max_tokens(TypicalLength.UNRESTRICTED, 777) == 777


class TestTypicalLengthPersistence:
    def test_absent_keys_load_as_defaults(self, tmp_path):
        # Files written before this feature must load unchanged.
        (tmp_path / "personas.yaml").write_text(
            "personas:\n- name: Alex\n  system_prompt: hi\n"
        )
        (tmp_path / "chatrooms.yaml").write_text(
            "chat_rooms:\n- name: TNG\n  persona_names: [Alex]\n"
        )
        personas = load_personas(tmp_path / "personas.yaml")
        rooms = load_chatrooms(tmp_path / "chatrooms.yaml")

        assert personas.personas[0].length_bias is LengthBias.MATCH
        assert rooms.chat_rooms[0].typical_length is TypicalLength.NORMAL

    def test_round_trips_as_a_plain_string(self, tmp_path):
        target = tmp_path / "chatrooms.yaml"
        save_chatrooms(
            ChatRoomsConfig(chat_rooms=[
                ChatRoom(name="TNG", typical_length=TypicalLength.TERSE)
            ]),
            target,
        )
        # A bare model_dump() would write a Python enum tag here.
        assert "typical_length: terse" in target.read_text()
        assert load_chatrooms(target).chat_rooms[0].typical_length is TypicalLength.TERSE

    def test_bias_round_trips_as_a_plain_string(self, tmp_path):
        target = tmp_path / "personas.yaml"
        save_personas(PersonasConfig(personas=[
            Persona(name="Alex", system_prompt="hi", length_bias=LengthBias.MUCH_SHORTER)
        ]), target)
        assert "length_bias: much_shorter" in target.read_text()
        assert load_personas(target).personas[0].length_bias is LengthBias.MUCH_SHORTER

    def test_legacy_persona_typical_length_becomes_a_bias(self, tmp_path, caplog):
        # The absolute per-persona tier is gone, but it encoded an intent
        # ("longer than usual"), so that intent carries over rather than
        # being dropped.
        target = tmp_path / "personas.yaml"
        target.write_text(
            "personas:\n- name: Alex\n  system_prompt: hi\n  typical_length: detailed\n"
        )
        with caplog.at_level("INFO"):
            persona = load_personas(target).personas[0]

        assert persona.length_bias is LengthBias.LONGER
        assert "typical_length: detailed" in caplog.text


class TestPlayerProfile:
    def test_room_defaults_to_no_profile_and_no_requirement(self):
        room = ChatRoom(name="R")
        assert room.require_player_profile is False
        assert room.player_profile.is_complete is False

    @pytest.mark.parametrize("fields, complete", [
        ({"name": "Kira", "description": "A thief."}, True),
        ({"name": "Kira"}, False),                      # nothing said about them
        ({"description": "A thief."}, False),           # nothing to call them
        ({"name": "  ", "description": "A thief."}, False),
        ({"name": "Kira", "description": "A thief.", "appearance": ""}, True),
    ])
    def test_is_complete_needs_a_name_and_a_description(self, fields, complete):
        # Appearance stays optional: a character can be described without
        # being pictured.
        assert PlayerProfile(**fields).is_complete is complete

    def test_absent_keys_load_as_an_empty_profile(self, tmp_path):
        target = tmp_path / "chatrooms.yaml"
        target.write_text("chat_rooms:\n- name: TNG\n  persona_names: [Alex]\n")
        room = load_chatrooms(target).chat_rooms[0]
        assert room.require_player_profile is False
        assert room.player_profile.name == ""

    def test_profile_round_trips_through_yaml(self, tmp_path):
        target = tmp_path / "chatrooms.yaml"
        save_chatrooms(ChatRoomsConfig(chat_rooms=[ChatRoom(
            name="TNG",
            require_player_profile=True,
            player_profile=PlayerProfile(
                name="Kira", description="A thief.", appearance="Green coat."
            ),
        )]), target)

        room = load_chatrooms(target).chat_rooms[0]
        assert room.require_player_profile is True
        assert room.player_profile.name == "Kira"
        assert room.player_profile.appearance == "Green coat."
