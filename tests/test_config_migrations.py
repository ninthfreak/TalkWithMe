"""Old config files must keep loading, and must keep their meaning.

Two failure modes matter here and neither is loud: a legacy file that
crashes on load, and one that loads while quietly losing what the user had
configured. Every case below feeds a *real* old-format file through the
public loaders.
"""

import pytest
import yaml

from app.config import (
    CONFIG_SCHEMA_VERSION,
    LengthBias,
    Persona,
    PersonasConfig,
    TypicalLength,
    config_dir,
    config_path,
    load_chatrooms,
    load_personas,
    load_settings,
    migrate_config_files,
    save_personas,
)
from app.config_migrations import migrate_chatrooms, migrate_personas, migrate_settings

# A personas.yaml exactly as it looked before any of this work: no schema
# version, the pre-rename 'language' key, and absolute per-persona lengths.
LEGACY_PERSONAS = """\
personas:
- name: Alex
  description: A friendly AI assistant
  system_prompt: You are Alex.
  router_hints: general questions
  avatar_color: '#4A90D9'
  language: fr
  typical_length: terse
- name: Luna
  description: A philosophical poet
  system_prompt: You are Luna.
  router_hints: philosophy
  typical_length: detailed
- name: Sam
  system_prompt: You are Sam.
"""

LEGACY_CHATROOMS = """\
chat_rooms:
- name: TNG
  persona_names: [Alex, Luna]
  echo_chamber: true
"""

LEGACY_SETTINGS = """\
llm:
  base_url: http://legacy:8080
  model: old-model
  max_tokens: 512
  temperature: 0.5
general:
  max_persona_replies: 2
  show_tool_calls: false
"""


class TestLegacyPersonas:
    def test_legacy_file_loads_without_error(self, tmp_path):
        target = tmp_path / "personas.yaml"
        target.write_text(LEGACY_PERSONAS)
        assert [p.name for p in load_personas(target).personas] == ["Alex", "Luna", "Sam"]

    def test_language_key_is_renamed(self, tmp_path):
        target = tmp_path / "personas.yaml"
        target.write_text(LEGACY_PERSONAS)
        assert load_personas(target).personas[0].reference_audio_language == "fr"

    @pytest.mark.parametrize("legacy, expected", [
        ("terse", LengthBias.MUCH_SHORTER),
        ("brief", LengthBias.SHORTER),
        ("normal", LengthBias.MATCH),
        ("detailed", LengthBias.LONGER),
        # Had no place on the scale, so there is no offset to carry over.
        ("unrestricted", LengthBias.MATCH),
        ("nonsense", LengthBias.MATCH),
    ])
    def test_absolute_length_becomes_the_equivalent_relative_bias(
        self, tmp_path, legacy, expected
    ):
        # The old tier said where the persona sat relative to a typical
        # reply. Keeping that offset keeps a laconic persona laconic;
        # dropping the field would have quietly flattened every persona.
        target = tmp_path / "personas.yaml"
        target.write_text(
            f"personas:\n- name: Alex\n  system_prompt: hi\n  typical_length: {legacy}\n"
        )
        assert load_personas(target).personas[0].length_bias is expected

    def test_personas_without_the_legacy_keys_are_untouched(self, tmp_path):
        target = tmp_path / "personas.yaml"
        target.write_text(LEGACY_PERSONAS)
        sam = load_personas(target).personas[2]
        assert sam.length_bias is LengthBias.MATCH
        assert sam.reference_audio_language == "en"

    def test_an_explicit_bias_is_not_overwritten_by_a_legacy_key(self):
        raw = {"personas": [
            {"name": "A", "typical_length": "terse", "length_bias": "much_longer"}
        ]}
        migrated, _ = migrate_personas(raw)
        assert migrated["personas"][0]["length_bias"] == "much_longer"

    def test_migration_notes_name_what_changed(self):
        _, notes = migrate_personas(yaml.safe_load(LEGACY_PERSONAS))
        joined = " | ".join(notes)
        assert "reference_audio_language" in joined
        assert "length_bias" in joined


class TestLegacyChatroomsAndSettings:
    def test_legacy_chatrooms_load_with_new_fields_defaulted(self, tmp_path):
        target = tmp_path / "chatrooms.yaml"
        target.write_text(LEGACY_CHATROOMS)
        room = load_chatrooms(target).chat_rooms[0]
        assert room.echo_chamber is True                       # preserved
        assert room.typical_length is TypicalLength.NORMAL     # defaulted
        assert room.require_player_profile is False
        assert room.player_profile.name == ""

    def test_legacy_settings_load_and_keep_their_values(self, tmp_path):
        target = tmp_path / "settings.yaml"
        target.write_text(LEGACY_SETTINGS)
        cfg = load_settings(target)
        assert cfg.llm.base_url == "http://legacy:8080"
        assert cfg.general.max_persona_replies == 2
        assert cfg.general.show_tool_calls is False
        assert cfg.general.typical_length is TypicalLength.NORMAL


class TestSchemaVersioning:
    @pytest.mark.parametrize("migrate", [migrate_personas, migrate_chatrooms, migrate_settings])
    def test_unversioned_files_are_stamped(self, migrate):
        raw, _ = migrate({})
        assert raw["schema_version"] == CONFIG_SCHEMA_VERSION

    def test_current_version_is_a_no_op(self):
        raw = {"schema_version": CONFIG_SCHEMA_VERSION,
               "personas": [{"name": "A", "typical_length": "terse"}]}
        migrated, notes = migrate_personas(raw)
        # Already current: the legacy key is left exactly as found rather
        # than being migrated a second time.
        assert notes == []
        assert migrated["personas"][0]["typical_length"] == "terse"

    def test_a_future_version_loads_rather_than_failing(self, caplog):
        raw = {"schema_version": CONFIG_SCHEMA_VERSION + 5, "personas": []}
        with caplog.at_level("WARNING"):
            migrated, _ = migrate_personas(raw)
        assert migrated["schema_version"] == CONFIG_SCHEMA_VERSION + 5
        assert "newer than this app understands" in caplog.text

    def test_a_junk_version_is_treated_as_the_oldest(self, caplog):
        with caplog.at_level("WARNING"):
            raw, _ = migrate_personas({"schema_version": "banana", "personas": []})
        assert raw["schema_version"] == CONFIG_SCHEMA_VERSION

    def test_saved_files_carry_the_version_first(self, tmp_path):
        target = tmp_path / "personas.yaml"
        save_personas(PersonasConfig(personas=[Persona(name="A", system_prompt="x")]), target)
        assert target.read_text().startswith(f"schema_version: {CONFIG_SCHEMA_VERSION}")


class TestLocationMigration:
    """Repo-root config moves into config/ — by copying, never deleting.

    Removing the tracked root files is what made `git pull` fail with
    "Your local changes to the following files would be overwritten by
    merge", so the root copies must survive untouched.
    """

    def _seed_root(self, tmp_path):
        (tmp_path / "personas.yaml").write_text(LEGACY_PERSONAS)
        (tmp_path / "chatrooms.yaml").write_text(LEGACY_CHATROOMS)
        (tmp_path / "settings.yaml").write_text(LEGACY_SETTINGS)

    def test_root_config_is_copied_into_the_config_dir(self, tmp_path):
        self._seed_root(tmp_path)
        assert sorted(migrate_config_files()) == [
            "chatrooms.yaml", "personas.yaml", "settings.yaml",
        ]
        for name in ("personas.yaml", "chatrooms.yaml", "settings.yaml"):
            assert (config_dir() / name).exists()

    def test_the_tracked_root_files_are_left_alone(self, tmp_path):
        self._seed_root(tmp_path)
        before = (tmp_path / "personas.yaml").read_text()
        migrate_config_files()
        assert (tmp_path / "personas.yaml").exists()
        assert (tmp_path / "personas.yaml").read_text() == before

    def test_the_copy_is_upgraded_on_the_way_across(self, tmp_path):
        self._seed_root(tmp_path)
        migrate_config_files()
        raw = yaml.safe_load((config_dir() / "personas.yaml").read_text())
        assert raw["schema_version"] == CONFIG_SCHEMA_VERSION
        assert raw["personas"][0]["length_bias"] == "much_shorter"
        assert "typical_length" not in raw["personas"][0]

    def test_settings_survive_the_move(self, tmp_path):
        self._seed_root(tmp_path)
        migrate_config_files()
        assert load_settings().llm.base_url == "http://legacy:8080"

    def test_an_existing_config_dir_is_never_overwritten(self, tmp_path):
        self._seed_root(tmp_path)
        (config_dir()).mkdir(parents=True)
        (config_dir() / "personas.yaml").write_text(
            "personas:\n- name: Mine\n  system_prompt: hi\n"
        )
        assert "personas.yaml" not in migrate_config_files()
        assert [p.name for p in load_personas().personas] == ["Mine"]

    def test_migration_is_idempotent(self, tmp_path):
        self._seed_root(tmp_path)
        migrate_config_files()
        assert migrate_config_files() == []

    def test_nothing_to_do_when_there_is_no_root_config(self, tmp_path):
        assert migrate_config_files() == []


class TestPathResolution:
    def test_config_dir_wins_when_present(self, tmp_path):
        (tmp_path / "personas.yaml").write_text(LEGACY_PERSONAS)
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "personas.yaml").write_text("personas: []\n")
        assert config_path("personas.yaml") == tmp_path / "config" / "personas.yaml"

    def test_falls_back_to_the_root_copy_before_migration_runs(self, tmp_path):
        # The very first run after upgrading, config/ does not exist yet;
        # the app must still come up with the user's existing settings.
        (tmp_path / "personas.yaml").write_text(LEGACY_PERSONAS)
        assert config_path("personas.yaml") == tmp_path / "personas.yaml"
        assert [p.name for p in load_personas().personas] == ["Alex", "Luna", "Sam"]


class TestEchoChamberSurvives:
    """echo_chamber is a room setting again after a brief removal.

    A version 3 dropped it; that was reverted. The version number stays at
    3 so files stamped by the short-lived version do not warn on every
    load, and no step rewrites rooms, so the flag is carried through.
    """

    @pytest.mark.parametrize("header", ["", "schema_version: 2\n"])
    def test_the_flag_is_preserved_however_the_file_is_stamped(self, tmp_path, header):
        target = tmp_path / "chatrooms.yaml"
        target.write_text(
            header + "chat_rooms:\n- name: TNG\n  persona_names: [Alex]\n"
            "  echo_chamber: true\n"
        )
        assert load_chatrooms(target).chat_rooms[0].echo_chamber is True

    def test_a_file_stamped_by_the_reverted_version_loads_without_warning(
        self, tmp_path, caplog
    ):
        target = tmp_path / "chatrooms.yaml"
        target.write_text(
            "schema_version: 3\nchat_rooms:\n- name: TNG\n  persona_names: [Alex]\n"
        )
        with caplog.at_level("WARNING"):
            room = load_chatrooms(target).chat_rooms[0]

        assert room.name == "TNG"
        assert room.echo_chamber is False   # the removal dropped it; defaults off
        assert "newer than this app understands" not in caplog.text

    def test_no_chatroom_migration_rewrites_rooms(self):
        raw, notes = migrate_chatrooms(
            {"chat_rooms": [{"name": "TNG", "echo_chamber": True}]}
        )
        assert raw["chat_rooms"][0]["echo_chamber"] is True
        assert notes == []
