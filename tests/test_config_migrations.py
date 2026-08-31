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
    load_player,
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
        assert room.persona_names == ["Alex", "Luna"]          # preserved
        assert room.typical_length is TypicalLength.NORMAL     # defaulted
        assert room.require_player_persona is False

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
        migrate_config_files()
        # It may be schema-stamped in place, but its *content* is never
        # replaced by the repo-root copy.
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


class TestEchoChamberIsNotRoomState:
    """The Echo chamber checkbox stays under the chat room selector, but it
    is UI state: the left panel holds controls you use *while in* a room,
    which is not the same as settings belonging *to* the room."""

    def test_a_room_with_echo_chamber_still_loads(self, tmp_path):
        target = tmp_path / "chatrooms.yaml"
        target.write_text(
            "chat_rooms:\n- name: TNG\n  persona_names: [Alex]\n  echo_chamber: true\n"
        )
        room = load_chatrooms(target).chat_rooms[0]
        assert room.persona_names == ["Alex"]
        assert not hasattr(room, "echo_chamber")

    @pytest.mark.parametrize("version", [None, 2, 3])
    def test_the_key_is_dropped_from_any_earlier_version(self, version):
        raw = {"chat_rooms": [{"name": "TNG", "echo_chamber": True}]}
        if version is not None:
            raw["schema_version"] = version

        migrated, notes = migrate_chatrooms(raw)

        assert "echo_chamber" not in migrated["chat_rooms"][0]
        assert migrated["schema_version"] == CONFIG_SCHEMA_VERSION
        assert any("echo_chamber" in n for n in notes)

    def test_other_room_settings_are_untouched(self):
        migrated, _ = migrate_chatrooms({"chat_rooms": [
            {"name": "TNG", "echo_chamber": True, "typical_length": "brief",
             "require_player_profile": True},
        ]})
        room = migrated["chat_rooms"][0]
        assert room["typical_length"] == "brief"
        # Renamed by the v5 -> v6 step, but the value survives.
        assert room["require_player_persona"] is True

    def test_rooms_without_the_key_produce_no_notes(self):
        _, notes = migrate_chatrooms({"chat_rooms": [{"name": "TNG"}]})
        assert notes == []


class TestWrittenProfilesBecomeAnAdoptedPersona:
    """Profiles were per room, then the player's, and now do not exist.

    The player adopts one of the configured personas instead, so there is
    one description of a character rather than two that can drift. Nothing
    maps a written profile onto a persona, so the old value is dropped —
    but visibly, in the log, because "my character vanished" with no
    explanation is worse than a line saying so.
    """

    ROOMS_WITH_PROFILES = """\
chat_rooms:
- name: Pub
  persona_names: [Alex]
  require_player_profile: true
  player_profile: {name: Gregory, description: An innkeeper., appearance: Greying.}
- name: Docks
  persona_names: [Luna]
  player_profile: {name: Sal, description: A dockhand., appearance: ''}
"""

    def _upgrade(self, tmp_path):
        (tmp_path / "chatrooms.yaml").write_text(self.ROOMS_WITH_PROFILES)
        return migrate_config_files()

    def test_the_rooms_keep_their_requirement_but_lose_the_profile(self, tmp_path):
        self._upgrade(tmp_path)
        rooms = {r.name: r for r in load_chatrooms().chat_rooms}
        assert rooms["Pub"].require_player_persona is True
        assert not hasattr(rooms["Pub"], "player_profile")

    def test_no_player_file_is_invented_from_the_rooms(self, tmp_path):
        # There is nothing in a written profile that names a persona, so
        # guessing one would put words in the player's mouth.
        assert "player.yaml" not in self._upgrade(tmp_path)
        assert load_player().persona_name == ""

    def test_a_room_key_alone_is_dropped_by_the_schema_step(self):
        migrated, notes = migrate_chatrooms({"chat_rooms": [
            {"name": "TNG", "require_player_profile": True,
             "player_profile": {"name": "Kira"}},
        ]})
        room = migrated["chat_rooms"][0]
        assert "player_profile" not in room
        assert room["require_player_persona"] is True
        assert any("player_profile" in n for n in notes)

    def test_the_room_flag_is_renamed_not_reset(self):
        migrated, notes = migrate_chatrooms({
            "schema_version": 5,
            "chat_rooms": [{"name": "TNG", "require_player_profile": True}],
        })
        room = migrated["chat_rooms"][0]
        assert "require_player_profile" not in room
        assert room["require_player_persona"] is True
        assert any("require_player_persona" in n for n in notes)

    def test_a_written_profile_in_player_yaml_is_dropped_and_reported(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "player.yaml").write_text(
            "schema_version: 5\nprofile:\n  name: Gregory\n  description: An innkeeper.\n"
        )

        assert "player.yaml" in migrate_config_files()

        raw = yaml.safe_load((tmp_path / "config" / "player.yaml").read_text())
        assert "profile" not in raw
        assert raw["persona_name"] == ""
        assert load_player().persona_name == ""

    def test_the_dropped_character_is_named_in_the_log(self, tmp_path, caplog):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "player.yaml").write_text(
            "schema_version: 5\nprofile:\n  name: Gregory\n"
        )
        with caplog.at_level("INFO"):
            migrate_config_files()
        assert "Gregory" in caplog.text

    def test_an_adopted_persona_survives_a_reload(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "player.yaml").write_text(
            f"schema_version: {CONFIG_SCHEMA_VERSION}\npersona_name: Luna\n"
        )
        assert migrate_config_files() == []
        assert load_player().persona_name == "Luna"


class TestOutOfDateFilesAreRewritten:
    """Loading migrates in memory; nothing rewrote the file until a save.

    A stale key could therefore sit on disk indefinitely, so what the app
    reads and what the file says drifted apart.
    """

    def test_an_existing_config_file_is_brought_up_to_date(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "chatrooms.yaml").write_text(
            "chat_rooms:\n- name: Pub\n  persona_names: [Alex]\n"
            "  player_profile: {name: Gregory, description: An innkeeper.}\n"
        )

        assert "chatrooms.yaml" in migrate_config_files()

        raw = yaml.safe_load((tmp_path / "config" / "chatrooms.yaml").read_text())
        assert raw["schema_version"] == CONFIG_SCHEMA_VERSION
        assert "player_profile" not in raw["chat_rooms"][0]

    def test_a_file_from_a_newer_release_is_left_alone(self, tmp_path):
        # _apply() refuses to touch it, so rewriting it here only churned
        # the mtime and reported a migration that never happened.
        (tmp_path / "config").mkdir()
        target = tmp_path / "config" / "chatrooms.yaml"
        target.write_text(
            f"schema_version: {CONFIG_SCHEMA_VERSION + 5}\nchat_rooms:\n- name: Pub\n"
        )
        before = target.read_text()

        assert migrate_config_files() == []
        assert target.read_text() == before

    def test_a_current_file_is_left_alone(self, tmp_path):
        (tmp_path / "config").mkdir()
        target = tmp_path / "config" / "chatrooms.yaml"
        target.write_text(
            f"schema_version: {CONFIG_SCHEMA_VERSION}\nchat_rooms:\n- name: Pub\n"
        )
        before = target.read_text()

        assert migrate_config_files() == []
        assert target.read_text() == before

    def test_it_is_idempotent(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "personas.yaml").write_text(
            "personas:\n- name: Alex\n  system_prompt: hi\n  typical_length: terse\n"
        )
        assert "personas.yaml" in migrate_config_files()
        assert migrate_config_files() == []
