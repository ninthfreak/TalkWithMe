"""Tests for app/services/persona_store.py — prompt.md frontmatter,
directory scanning, file helpers, and the legacy personas.yaml migration.

These are unit tests: they touch real files under pytest's tmp_path but
never the app's config cache (that is the router tests' job).
"""

import logging

import pytest
import yaml
from pydantic import ValidationError

from app.config import DEFAULT_MEMORY_SIZE, MAX_MEMORY_LINE_CHARS, MAX_MEMORY_SIZE
from app.services import persona_store
from app.services.persona_store import (
    PersonaMigrationError,
    PersonaStorageError,
    append_memory,
    build_prompt_md,
    count_name_mentions,
    find_avatar_file,
    forget_acquaintance,
    forget_subject,
    load_persona_from_dir,
    load_personas_yaml,
    migrate_from_legacy_yaml,
    parse_frontmatter,
    parse_memory_size,
    purge_memories_to_limit,
    read_language_file,
    read_memories,
    remove_memories_file,
    scan_personas_directory,
    sanitize_persona_dirname,
    unique_persona_dirname,
    write_avatar_file,
    write_language_file,
    write_prompt_md,
    write_transcript_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_persona_dir(root, name="Alex", **overrides):
    """Create one minimal valid persona directory and return its path."""
    fields = dict(
        description="A friendly assistant",
        router_hints="general questions",
        avatar_color="#888888",
        allow_tool_calls=False,
        system_prompt="You are Alex, a friendly assistant.",
    )
    fields.update(overrides)
    persona_dir = root / name
    persona_dir.mkdir(parents=True)
    write_prompt_md(persona_dir, name=name, **fields)
    return persona_dir


def write_legacy_yaml(path, personas) -> "path":
    """Write a legacy-format personas.yaml and return its path."""
    path.write_text(yaml.dump({"personas": personas}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# prompt.md frontmatter
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_wellformed_frontmatter_splits_fields_and_body(self):
        text = "---\nname: Alex\ndescription: A test\n---\n\nYou are Alex.\n"
        fields, body = parse_frontmatter(text)
        assert fields == {"name": "Alex", "description": "A test"}
        assert body == "You are Alex."

    def test_no_frontmatter_returns_whole_file_as_body(self):
        text = "Just a prompt, no frontmatter.\n"
        assert parse_frontmatter(text) == ({}, "Just a prompt, no frontmatter.")

    def test_unterminated_frontmatter_falls_back_to_whole_file(self):
        text = "---\nname: Alex\nYou are Alex.\n"
        assert parse_frontmatter(text) == ({}, text.strip("\n"))

    def test_non_mapping_frontmatter_falls_back_to_whole_file(self):
        text = "---\n- a\n- b\n---\nBody\n"
        # Malformed frontmatter never costs the prompt: the whole file
        # degrades to the system prompt body.
        assert parse_frontmatter(text) == ({}, text.strip("\n"))

    def test_malformed_yaml_frontmatter_falls_back_to_whole_file(self):
        text = "---\nname: [unclosed\n---\nBody\n"
        assert parse_frontmatter(text) == ({}, text.strip("\n"))


class TestBuildPromptMd:
    def test_round_trip_preserves_all_fields(self):
        built = build_prompt_md(
            "Alex",
            name="Alex",
            description="A friendly assistant",
            router_hints="general questions",
            avatar_color="#4A90D9",
            allow_tool_calls=True,
            system_prompt="You are Alex.",
        )
        fields, body = parse_frontmatter(built)
        assert fields == {
            "description": "A friendly assistant",
            "router_hints": "general questions",
            "avatar_color": "#4A90D9",
            "allow_tool_calls": True,
            # Ours: a persona's length sits relative to its room.
            "length_bias": "match",
        }
        assert body == "You are Alex."

    def test_name_field_omitted_when_same_as_directory(self):
        built = build_prompt_md(
            "Alex", name="Alex", description="", router_hints="h",
            avatar_color="#fff", allow_tool_calls=False, system_prompt="p",
        )
        # The directory name is the fallback identity, so no `name:` noise.
        assert "name:" not in built.split("---")[1]

    def test_name_field_present_when_different_from_directory(self):
        built = build_prompt_md(
            "OBrien", name="O'Brien", description="", router_hints="h",
            avatar_color="#fff", allow_tool_calls=False, system_prompt="p",
        )
        fields, _ = parse_frontmatter(built)
        assert fields["name"] == "O'Brien"

    def test_memory_size_omitted_when_none(self):
        # Legacy migration passes None so upgraded personas carry no new key.
        built = build_prompt_md(
            "Alex", name="Alex", description="", router_hints="h",
            avatar_color="#fff", allow_tool_calls=False, system_prompt="p",
        )
        fields, _ = parse_frontmatter(built)
        assert "memory_size" not in fields

    def test_memory_size_written_when_provided(self):
        built = build_prompt_md(
            "Alex", name="Alex", description="", router_hints="h",
            avatar_color="#fff", allow_tool_calls=False, system_prompt="p",
            memory_size=4096,
        )
        fields, _ = parse_frontmatter(built)
        assert fields["memory_size"] == 4096


# ---------------------------------------------------------------------------
# Directory names
# ---------------------------------------------------------------------------

class TestSanitizeDirname:
    def test_strips_directory_hostile_characters(self):
        assert sanitize_persona_dirname("O'Brien") == "OBrien"
        # Stripping deletes characters; it never inserts separators.
        assert sanitize_persona_dirname("a/b\\c.d") == "abcd"

    def test_allows_letters_numbers_spaces_hyphens_underscores(self):
        assert sanitize_persona_dirname("My Room-1_x2") == "My Room-1_x2"

    def test_all_hostile_name_returns_empty_string(self):
        assert sanitize_persona_dirname("???") == ""


class TestUniqueDirname:
    def test_free_name_used_as_is(self, tmp_path):
        assert unique_persona_dirname(tmp_path, "Alex") == "Alex"

    def test_collision_gets_incrementing_suffix(self, tmp_path):
        (tmp_path / "Alex").mkdir()
        assert unique_persona_dirname(tmp_path, "Alex") == "Alex_2"
        (tmp_path / "Alex_2").mkdir()
        assert unique_persona_dirname(tmp_path, "Alex") == "Alex_3"


# ---------------------------------------------------------------------------
# Loading (read-only)
# ---------------------------------------------------------------------------

class TestReadLanguageFile:
    def test_missing_file_defaults_to_en(self, tmp_path, caplog):
        assert read_language_file(tmp_path, "Alex") == "en"
        assert "defaulting to 'en'" in caplog.text

    def test_strips_whitespace(self, tmp_path):
        (tmp_path / "language.txt").write_text("  de  \n")
        assert read_language_file(tmp_path, "Alex") == "de"

    def test_invalid_length_defaults_to_en(self, tmp_path):
        (tmp_path / "language.txt").write_text("eng")
        assert read_language_file(tmp_path, "Alex") == "en"


class TestFindAvatarFile:
    def test_no_files_returns_none(self, tmp_path):
        assert find_avatar_file(tmp_path, "Alex") is None

    def test_returns_the_image_file(self, tmp_path):
        (tmp_path / "image.webp").write_bytes(b"x")
        assert find_avatar_file(tmp_path, "Alex").name == "image.webp"

    def test_multiple_images_warns_and_picks_first_alphabetically(self, tmp_path, caplog):
        (tmp_path / "image.webp").write_bytes(b"x")
        (tmp_path / "image.png").write_bytes(b"y")
        assert find_avatar_file(tmp_path, "Alex").name == "image.png"
        assert "multiple image files" in caplog.text

    def test_unsupported_extension_ignored(self, tmp_path):
        (tmp_path / "image.bmp").write_bytes(b"x")
        assert find_avatar_file(tmp_path, "Alex") is None


class TestLoadPersonaFromDir:
    def test_loads_all_fields_from_files(self, tmp_path):
        d = make_persona_dir(tmp_path, "Alex", description="A friend")
        (d / "language.txt").write_text("de")
        (d / "ref.wav").write_bytes(b"wav")
        (d / "ref.txt").write_text("transcript here")
        (d / "image.png").write_bytes(b"png")

        persona = load_persona_from_dir(d)
        assert persona.name == "Alex"
        assert persona.description == "A friend"
        assert persona.system_prompt == "You are Alex, a friendly assistant."
        assert persona.reference_audio_language == "de"
        assert persona.reference_audio == str(d / "ref.wav")
        assert persona.reference_audio_transcript == str(d / "ref.txt")
        assert persona.avatar_image == str(d / "image.png")
        assert persona.persona_dir == d
        assert persona.tts_capable is True

    def test_missing_prompt_md_raises(self, tmp_path):
        d = tmp_path / "Broken"
        d.mkdir()
        with pytest.raises(PersonaStorageError, match="prompt.md"):
            load_persona_from_dir(d)

    def test_frontmatter_name_overrides_directory_name(self, tmp_path):
        d = tmp_path / "OBrien"
        d.mkdir()
        write_prompt_md(
            d, name="O'Brien", description="", router_hints="h",
            avatar_color="#fff", allow_tool_calls=False, system_prompt="p",
        )
        assert load_persona_from_dir(d).name == "O'Brien"

    def test_empty_transcript_makes_persona_not_tts_capable(self, tmp_path, caplog):
        d = make_persona_dir(tmp_path)
        (d / "ref.wav").write_bytes(b"wav")
        (d / "ref.txt").write_text("   \n")

        persona = load_persona_from_dir(d)
        assert persona.reference_audio is not None
        assert persona.reference_audio_transcript is None
        assert persona.tts_capable is False
        assert "empty" in caplog.text


class TestScanPersonasDirectory:
    def test_missing_root_returns_empty_list(self, tmp_path):
        assert scan_personas_directory(tmp_path / "nope") == []

    def test_returns_personas_in_directory_order(self, tmp_path):
        make_persona_dir(tmp_path, "Zed")
        make_persona_dir(tmp_path, "Ann")
        names = [p.name for p in scan_personas_directory(tmp_path)]
        assert names == ["Ann", "Zed"]

    def test_ignores_stray_files(self, tmp_path):
        make_persona_dir(tmp_path, "Alex")
        (tmp_path / ".DS_Store").write_bytes(b"junk")
        (tmp_path / "notes.txt").write_text("not a persona")
        names = [p.name for p in scan_personas_directory(tmp_path)]
        assert names == ["Alex"]

    def test_skips_broken_directories_with_warning(self, tmp_path, caplog):
        make_persona_dir(tmp_path, "Alex")
        (tmp_path / "Broken").mkdir()  # no prompt.md
        names = [p.name for p in scan_personas_directory(tmp_path)]
        assert names == ["Alex"]
        assert "Skipping persona directory Broken" in caplog.text


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class TestWriteTranscriptFile:
    def test_writes_stripped_content(self, tmp_path):
        write_transcript_file(tmp_path, "  hello \n")
        assert (tmp_path / "ref.txt").read_text() == "hello"

    def test_blank_text_removes_existing_file(self, tmp_path):
        write_transcript_file(tmp_path, "hello")
        write_transcript_file(tmp_path, "   ")
        assert not (tmp_path / "ref.txt").exists()

    def test_blank_text_with_no_file_is_noop(self, tmp_path):
        write_transcript_file(tmp_path, "")
        assert not (tmp_path / "ref.txt").exists()


class TestAvatarFileOps:
    def test_write_replaces_different_extension(self, tmp_path):
        write_avatar_file(tmp_path, b"one", ".png")
        write_avatar_file(tmp_path, b"two", ".webp")
        assert not (tmp_path / "image.png").exists()
        assert (tmp_path / "image.webp").read_bytes() == b"two"

    def test_remove_returns_false_when_absent(self, tmp_path):
        assert persona_store.remove_avatar_file(tmp_path) is False

    def test_remove_returns_true_when_present(self, tmp_path):
        write_avatar_file(tmp_path, b"x", ".gif")
        assert persona_store.remove_avatar_file(tmp_path) is True
        assert not (tmp_path / "image.gif").exists()


# ---------------------------------------------------------------------------
# Legacy personas.yaml parsing
# ---------------------------------------------------------------------------

class TestLoadPersonasYaml:
    def test_parses_personas(self, tmp_path):
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Alex", "system_prompt": "p"},
        ])
        assert [p.name for p in load_personas_yaml(path)] == ["Alex"]

    def test_missing_personas_key_returns_empty_list(self, tmp_path):
        (tmp_path / "personas.yaml").write_text("{}")
        assert load_personas_yaml(tmp_path / "personas.yaml") == []

    def test_legacy_language_key_migrated(self, tmp_path):
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Alex", "system_prompt": "p", "language": "de"},
        ])
        assert load_personas_yaml(path)[0].reference_audio_language == "de"

    def test_explicit_reference_audio_language_wins_over_legacy_key(self, tmp_path):
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Alex", "system_prompt": "p",
             "language": "de", "reference_audio_language": "es"},
        ])
        assert load_personas_yaml(path)[0].reference_audio_language == "es"

    def test_top_level_not_mapping_raises(self, tmp_path):
        (tmp_path / "personas.yaml").write_text("- a\n- b\n")
        with pytest.raises(PersonaMigrationError, match="top-level mapping"):
            load_personas_yaml(tmp_path / "personas.yaml")

    def test_personas_not_a_list_raises(self, tmp_path):
        (tmp_path / "personas.yaml").write_text("personas: scalar\n")
        with pytest.raises(PersonaMigrationError, match="must be a list"):
            load_personas_yaml(tmp_path / "personas.yaml")

    def test_entry_not_a_mapping_raises(self, tmp_path):
        (tmp_path / "personas.yaml").write_text("personas:\n  - just-a-string\n")
        with pytest.raises(PersonaMigrationError, match="must be a mapping"):
            load_personas_yaml(tmp_path / "personas.yaml")

    def test_invalid_persona_raises_validation_error(self, tmp_path):
        (tmp_path / "personas.yaml").write_text("personas:\n  - name: Alex\n")
        with pytest.raises(ValidationError):
            load_personas_yaml(tmp_path / "personas.yaml")  # missing system_prompt


# ---------------------------------------------------------------------------
# One-time migration
# ---------------------------------------------------------------------------

class TestMigrateFromLegacyYaml:
    def test_success_renames_yaml_to_backup_and_creates_dirs(self, tmp_path):
        ref_wav = tmp_path / "luna.wav"
        ref_wav.write_bytes(b"wav-bytes")
        ref_txt = tmp_path / "luna.txt"
        ref_txt.write_text("a transcript")
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Luna", "system_prompt": "You are Luna.",
             "reference_audio": str(ref_wav),
             "reference_audio_transcript": str(ref_txt)},
        ])
        root = tmp_path / "Personas"
        migrate_from_legacy_yaml(path, root)

        backup = tmp_path / "personas.yaml.bak"
        assert not path.exists()
        assert backup.exists()
        assert yaml.safe_load(backup.read_text())["personas"][0]["name"] == "Luna"

        luna_dir = root / "Luna"
        assert (luna_dir / "prompt.md").exists()
        assert (luna_dir / "language.txt").read_text() == "en"
        assert (luna_dir / "ref.wav").read_bytes() == b"wav-bytes"
        assert (luna_dir / "ref.txt").read_text() == "a transcript"

    def test_missing_referenced_file_is_a_minor_error(self, tmp_path, caplog):
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Luna", "system_prompt": "p",
             "reference_audio": str(tmp_path / "gone.wav")},
        ])
        root = tmp_path / "Personas"
        migrate_from_legacy_yaml(path, root)  # must not raise

        assert (root / "Luna" / "prompt.md").exists()
        assert not (root / "Luna" / "ref.wav").exists()
        assert "could not be read" in caplog.text
        assert (tmp_path / "personas.yaml.bak").exists()

    def test_supported_image_is_copied(self, tmp_path):
        png = tmp_path / "avatar.png"
        png.write_bytes(b"png-bytes")
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Alex", "system_prompt": "p", "avatar_image": str(png)},
        ])
        migrate_from_legacy_yaml(path, tmp_path / "Personas")
        assert (tmp_path / "Personas" / "Alex" / "image.png").read_bytes() == b"png-bytes"

    def test_unsupported_image_extension_skipped_with_warning(self, tmp_path, caplog):
        bmp = tmp_path / "avatar.bmp"
        bmp.write_bytes(b"bmp")
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Alex", "system_prompt": "p", "avatar_image": str(bmp)},
        ])
        migrate_from_legacy_yaml(path, tmp_path / "Personas")
        assert "unsupported image file" in caplog.text
        assert not (tmp_path / "Personas" / "Alex" / "image.bmp").exists()

    def test_unsupported_audio_extension_skipped_with_warning(self, tmp_path, caplog):
        mp3 = tmp_path / "luna.mp3"
        mp3.write_bytes(b"mp3")
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Luna", "system_prompt": "p", "reference_audio": str(mp3)},
        ])
        migrate_from_legacy_yaml(path, tmp_path / "Personas")
        assert "unsupported audio file" in caplog.text
        assert not (tmp_path / "Personas" / "Luna" / "ref.wav").exists()

    def test_blank_transcript_not_written(self, tmp_path):
        ref_txt = tmp_path / "t.txt"
        ref_txt.write_text("   ")
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Luna", "system_prompt": "p",
             "reference_audio_transcript": str(ref_txt)},
        ])
        migrate_from_legacy_yaml(path, tmp_path / "Personas")
        assert not (tmp_path / "Personas" / "Luna" / "ref.txt").exists()

    def test_name_without_usable_directory_chars_falls_back_to_persona(self, tmp_path):
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "???", "system_prompt": "p"},
        ])
        migrate_from_legacy_yaml(path, tmp_path / "Personas")
        d = tmp_path / "Personas" / "persona"
        assert d.is_dir()
        # The frontmatter keeps the real name.
        assert load_persona_from_dir(d).name == "???"

    def test_migrated_directory_collision_gets_suffix(self, tmp_path):
        (tmp_path / "Personas" / "Alex").mkdir(parents=True)
        (tmp_path / "Personas" / "Alex" / "prompt.md").write_text("stale")
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Alex", "system_prompt": "p"},
        ])
        migrate_from_legacy_yaml(path, tmp_path / "Personas")
        assert (tmp_path / "Personas" / "Alex_2" / "prompt.md").exists()

    def test_malformed_yaml_raises_and_leaves_everything_untouched(self, tmp_path):
        path = tmp_path / "personas.yaml"
        path.write_text("personas: [\n")  # unclosed flow sequence
        root = tmp_path / "Personas"
        with pytest.raises(PersonaMigrationError):
            migrate_from_legacy_yaml(path, root)
        assert path.exists()          # the YAML is the only copy; never touched
        assert not root.exists()      # no partial directory left behind

    def test_uncreatable_root_raises_and_leaves_yaml_untouched(self, tmp_path):
        path = write_legacy_yaml(tmp_path / "personas.yaml", [
            {"name": "Alex", "system_prompt": "p"},
        ])
        blocker = tmp_path / "Personas"
        blocker.write_text("I am a file, not a directory")
        with pytest.raises(PersonaMigrationError):
            migrate_from_legacy_yaml(path, blocker)
        assert path.exists()
        assert blocker.is_file()  # rmtree cannot remove a file; left in place


# ---------------------------------------------------------------------------
# Persona memories (memories.txt) — docs/feature_persona_memory.md
# ---------------------------------------------------------------------------

def _dir(tmp_path, name="Alex"):
    persona_dir = tmp_path / name
    persona_dir.mkdir(parents=True)
    return persona_dir


class TestParseMemorySize:
    """parse_memory_size(): frontmatter values are hand-editable, so
    garbage must degrade to the default (with a warning) — never crash
    the directory scan at startup."""

    def test_missing_value_gets_default(self):
        assert parse_memory_size(None, "Alex") == DEFAULT_MEMORY_SIZE

    @pytest.mark.parametrize("value", [0, 1, 4096, 8192, 16384])
    def test_valid_values_pass_through(self, value):
        assert parse_memory_size(value, "Alex") == value

    @pytest.mark.parametrize("value", ["8192", "4k", 4096.0, "0", ["8192"]])
    def test_non_int_values_warn_and_get_default(self, value, caplog):
        # 4096.0 is not an int (a hand-typed "4096." parses as a float);
        # strings, even numeric ones, are not ints. All get the default.
        with caplog.at_level(logging.WARNING):
            assert parse_memory_size(value, "Alex") == DEFAULT_MEMORY_SIZE
        assert "invalid memory_size" in caplog.text

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_values_warn_and_get_default(self, value, caplog):
        # bool is an int subclass in Python — it must be explicitly
        # rejected, or true would silently mean 1 byte of memory.
        with caplog.at_level(logging.WARNING):
            assert parse_memory_size(value, "Alex") == DEFAULT_MEMORY_SIZE
        assert "invalid memory_size" in caplog.text

    @pytest.mark.parametrize("value", [-1, 16385])
    def test_out_of_range_values_warn_and_get_default(self, value, caplog):
        with caplog.at_level(logging.WARNING):
            assert parse_memory_size(value, "Alex") == DEFAULT_MEMORY_SIZE
        assert "out-of-range memory_size" in caplog.text

    def test_valid_value_logs_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_memory_size(4096, "Alex")
        assert "memory_size" not in caplog.text


class TestReadMemories:
    def test_absent_file_returns_empty_string(self, tmp_path):
        assert read_memories(_dir(tmp_path)) == ""

    def test_returns_raw_content(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("The user likes tea.\n")
        assert read_memories(d) == "The user likes tea.\n"

    def test_binary_garbage_does_not_raise(self, tmp_path):
        # errors="replace": a corrupted file must degrade to garbage
        # text, not kill the caller.
        d = _dir(tmp_path)
        (d / "memories.txt").write_bytes(b"\xff\xfe\x00\x01")
        assert isinstance(read_memories(d), str)


class TestRemoveMemoriesFile:
    def test_present_file_is_removed(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("stale\n")
        assert remove_memories_file(d) is True
        assert not (d / "memories.txt").exists()

    def test_absent_file_returns_false(self, tmp_path):
        assert remove_memories_file(_dir(tmp_path)) is False


class TestAppendMemory:
    """append_memory(): every failure mode is an LLM-facing 'Error:'
    string, never an exception — the tool loop feeds the result back to
    the model, and an exception would kill the reply stream."""

    def test_disabled_budget_deletes_stale_file(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("stale\n")
        result = append_memory(d, "Tony", "The user likes tea.", 0)
        assert result == "Error: Memory is not enabled for this persona."
        assert not (d / "memories.txt").exists()

    @pytest.mark.parametrize("memory", [None, 42, [], "   ", "\n\t\n", ""])
    def test_no_content_is_reported(self, tmp_path, memory):
        d = _dir(tmp_path)
        result = append_memory(d, "Tony", memory, DEFAULT_MEMORY_SIZE)
        assert result == "Error: The memory was not saved because it had no content."
        assert not (d / "memories.txt").exists()

    def test_too_long_memory_is_rejected_not_truncated(self, tmp_path):
        d = _dir(tmp_path)
        result = append_memory(d, "Tony", "x" * (MAX_MEMORY_LINE_CHARS + 1), DEFAULT_MEMORY_SIZE)
        assert result == (
            "Error: The memory was too large to save. "
            f"Max per-memory length is {MAX_MEMORY_LINE_CHARS} characters."
        )
        assert not (d / "memories.txt").exists()

    def test_memory_exactly_at_char_limit_is_accepted(self, tmp_path):
        d = _dir(tmp_path)
        result = append_memory(d, "Tony", "x" * MAX_MEMORY_LINE_CHARS, DEFAULT_MEMORY_SIZE)
        assert result == "The memory was saved successfully."

    def test_too_large_memory_reported_with_configured_limit(self, tmp_path):
        d = _dir(tmp_path)
        result = append_memory(d, "Tony", "ab" * 4, memory_size=7)  # 8 bytes > 7
        assert result == "Error: The memory was too large to save. Configured memory limit: 7 bytes"

    def test_memory_exactly_at_byte_limit_is_accepted(self, tmp_path):
        d = _dir(tmp_path)
        # "[Tony] abc" plus a newline. The tag counts against the budget.
        result = append_memory(d, "Tony", "abc", memory_size=11)
        assert result == "The memory was saved successfully."
        assert read_memories(d) == "[Tony] abc\n"

    def test_success_appends_one_line(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("old\n")
        assert append_memory(d, "Tony", "The user likes tea.", DEFAULT_MEMORY_SIZE) == (
            "The memory was saved successfully."
        )
        assert read_memories(d) == "old\n[Tony] The user likes tea.\n"

    def test_newlines_are_deleted_not_replaced(self, tmp_path):
        # The spec deletes newline characters ("a\nb" -> "ab"): a memory
        # must be a single line, and replacement would silently change
        # the memory's content.
        d = _dir(tmp_path)
        assert append_memory(d, "Tony", "a\nb\rc\rd", DEFAULT_MEMORY_SIZE) == (
            "The memory was saved successfully."
        )
        assert read_memories(d) == "[Tony] abcd\n"

    def test_edges_are_stripped(self, tmp_path):
        d = _dir(tmp_path)
        assert append_memory(d, "Tony", "  padded  ", DEFAULT_MEMORY_SIZE) == (
            "The memory was saved successfully."
        )
        assert read_memories(d) == "[Tony] padded\n"

    def test_new_blank_lines_are_dropped_from_existing_file(self, tmp_path):
        # The file is rewritten from non-blank lines, so hand-edited
        # blank lines do not survive an append.
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("first\n\n  \nsecond\n")
        append_memory(d, "Tony", "third", DEFAULT_MEMORY_SIZE)
        assert read_memories(d) == "first\nsecond\n[Tony] third\n"

    def test_oldest_memories_purged_when_over_limit(self, tmp_path):
        d = _dir(tmp_path)
        # The new line is "[Tony] a5", 10 bytes with its newline. A
        # budget of 19 leaves room for it plus the three newest old lines.
        (d / "memories.txt").write_text("a1\na2\na3\na4\n")
        assert append_memory(d, "Tony", "a5", memory_size=19) == "The memory was saved successfully."
        assert read_memories(d) == "a3\na4\n[Tony] a5\n"

    def test_purge_never_drops_the_new_memory(self, tmp_path):
        d = _dir(tmp_path)
        # The new memory fits the limit by itself, but combined with the
        # old line it does not: the OLD line must go, never the new one.
        (d / "memories.txt").write_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")  # 41 bytes
        # "[Tony] new" is 10 bytes; a budget of 11 fits it and nothing else.
        assert append_memory(d, "Tony", "new", memory_size=11) == "The memory was saved successfully."
        assert read_memories(d) == "[Tony] new\n"

    def test_write_failure_is_reported_and_leaves_no_temp_file(self, tmp_path, monkeypatch, caplog):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("old\n")

        def boom(persona_dir, lines):
            raise OSError("disk full")

        monkeypatch.setattr(persona_store, "_write_memories_file", boom)
        result = append_memory(d, "Tony", "new", DEFAULT_MEMORY_SIZE)
        assert result == "Error: The memory could not be saved."
        assert read_memories(d) == "old\n"  # untouched
        assert not list(d.glob("memories.txt.tmp*"))
        assert "failed to write memories.txt" in caplog.text

    def test_disabled_budget_stale_file_delete_failure_does_not_raise(self, tmp_path, monkeypatch, caplog):
        # A failed stale-file cleanup (e.g. read-only directory) must not
        # escape append_memory and kill the tool-call stream: the LLM gets
        # the intended "not enabled" message, the disk problem is logged.
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("stale\n")

        def boom(persona_dir):
            raise OSError("read-only directory")

        monkeypatch.setattr(persona_store, "remove_memories_file", boom)
        result = append_memory(d, "Tony", "The user likes tea.", 0)
        assert result == "Error: Memory is not enabled for this persona."
        assert (d / "memories.txt").read_text() == "stale\n"  # survives
        assert "could not delete memories.txt" in caplog.text


class TestAssumedMemories:
    """A memory records whether it was witnessed or worked out.

    Without it an inference is indistinguishable from a fact once both are
    prose in the same file, and the persona holding it has no way to find
    out which it was.
    """

    def test_the_marker_round_trips(self):
        line = "[Tony] (assumed) Tony is about forty."
        memory = persona_store.parse_memory_line(line)

        assert memory == persona_store.Memory("Tony", "Tony is about forty.", True)
        assert memory.stored() == line

    @pytest.mark.parametrize("written", [
        "[Tony] (assumed) He sails.",
        "[Tony] (Assumed) He sails.",
        "[Tony] (guess) He sails.",
        "[Tony] ( guessed ) He sails.",
    ])
    def test_the_ways_a_model_writes_it(self, written):
        # A small closed set: every extra spelling accepted is a phrase
        # that stops being part of the memory's text, and quietly losing
        # words out of a memory is worse than an unmarked one.
        memory = persona_store.parse_memory_line(written)
        assert memory.assumed is True
        assert memory.text == "He sails."

    def test_a_plain_bracket_is_left_in_the_text(self):
        memory = persona_store.parse_memory_line("[Tony] (probably) he sails.")
        assert memory.assumed is False
        assert memory.text == "(probably) he sails."

    def test_an_unmarked_memory_is_known(self):
        assert persona_store.parse_memory_line("[Tony] Tony is 43.").assumed is False

    def test_append_writes_the_marker(self, tmp_path):
        persona_store.append_memory(tmp_path, "Tony", "Tony is about forty.",
                                    1024, assumed=True)
        assert (tmp_path / "memories.txt").read_text() == (
            "[Tony] (assumed) Tony is about forty.\n"
        )

    def test_the_same_thing_assumed_and_known_are_two_memories(self, tmp_path):
        # Deduplication compares the stored line, and these are not the
        # same claim: one is what they said, the other what you decided.
        persona_store.append_memory(tmp_path, "Tony", "Tony sails.", 1024)
        persona_store.append_memory(tmp_path, "Tony", "Tony sails.", 1024, assumed=True)

        assert (tmp_path / "memories.txt").read_text() == (
            "[Tony] Tony sails.\n[Tony] (assumed) Tony sails.\n"
        )

    def test_grouping_keeps_the_marker(self, tmp_path):
        (tmp_path / "memories.txt").write_text(
            "[Tony] Tony is 43.\n[Tony] (assumed) Tony is unhappy.\n"
        )
        grouped = persona_store.memories_by_subject(tmp_path)

        assert [m.assumed for m in grouped["tony"]] == [False, True]


class TestDedupeMemories:
    """Byte-identical lines only, so it can never delete a real memory.

    Anything cleverer starts judging whether two sentences mean the same
    thing, and the failure mode there is losing something the persona
    knew. This function can only remove a line the file already contains
    character for character.
    """

    def _write(self, tmp_path, *lines):
        (tmp_path / "memories.txt").write_text("\n".join(lines) + "\n")

    def _read(self, tmp_path):
        return (tmp_path / "memories.txt").read_text().splitlines()

    def test_identical_lines_collapse_to_one(self, tmp_path):
        self._write(tmp_path, "[Brad] Brad is 43.", "[Brad] Brad is a banker.",
                    "[Brad] Brad is 43.")

        assert persona_store.dedupe_memories(tmp_path) == 2 - 1
        assert self._read(tmp_path) == ["[Brad] Brad is 43.", "[Brad] Brad is a banker."]

    def test_the_first_copy_is_the_one_kept(self, tmp_path):
        # Order is meaningful: the budget sheds from the oldest end, so a
        # memory must not quietly become younger by being repeated.
        self._write(tmp_path, "[Brad] aaa", "[Brad] bbb", "[Brad] aaa")

        persona_store.dedupe_memories(tmp_path)

        assert self._read(tmp_path) == ["[Brad] aaa", "[Brad] bbb"]

    def test_different_case_is_not_a_duplicate(self, tmp_path):
        # 100% identical, as asked. The write path's own dedup is
        # case-insensitive; this one is stricter on purpose.
        self._write(tmp_path, "[Brad] Brad is 43.", "[Brad] brad is 43.")

        assert persona_store.dedupe_memories(tmp_path) == 0
        assert len(self._read(tmp_path)) == 2

    def test_a_guess_is_not_a_duplicate_of_the_fact(self, tmp_path):
        # Same sentence, different claim: one is what they said, the other
        # what the persona decided.
        self._write(tmp_path, "[Brad] Brad is 43.", "[Brad] (assumed) Brad is 43.")

        assert persona_store.dedupe_memories(tmp_path) == 0

    def test_similar_lines_are_left_completely_alone(self, tmp_path):
        # The whole reason this is exact-match: these two overlap heavily
        # and are different facts.
        self._write(tmp_path, "[Brad] Brad likes tea.", "[Brad] Brad likes coffee.")

        assert persona_store.dedupe_memories(tmp_path) == 0
        assert len(self._read(tmp_path)) == 2

    def test_different_people_keep_their_own_copies(self, tmp_path):
        self._write(tmp_path, "[Brad] They sail.", "[Cora] They sail.")

        assert persona_store.dedupe_memories(tmp_path) == 0

    def test_a_clean_file_is_not_rewritten(self, tmp_path):
        self._write(tmp_path, "[Brad] Brad is 43.")
        before = (tmp_path / "memories.txt").stat().st_mtime_ns

        assert persona_store.dedupe_memories(tmp_path) == 0
        assert (tmp_path / "memories.txt").stat().st_mtime_ns == before

    def test_no_file_is_not_an_error(self, tmp_path):
        assert persona_store.dedupe_memories(tmp_path) == 0


class TestAssumptionsFadeFirst:
    """Under pressure, what you guessed goes before what you were told.

    A guess that never got confirmed is the cheapest thing in the file to
    be wrong about, and giving assumptions a shorter half-life is the
    closest this gets to a memory that settles.
    """

    @staticmethod
    def _lines(tmp_path):
        return (tmp_path / "memories.txt").read_text().splitlines()

    def test_the_oldest_assumption_goes_before_any_fact(self, tmp_path):
        # Budget fits three of these lines, not four.
        (tmp_path / "memories.txt").write_text(
            "[T] (assumed) aaaa\n[T] bbbb\n[T] (assumed) cccc\n"
        )
        budget = len("[T] (assumed) aaaa\n[T] bbbb\n[T] (assumed) cccc\n[T] dddd\n")

        persona_store.append_memory(tmp_path, "T", "dddd", budget - 1)

        assert self._lines(tmp_path) == [
            "[T] bbbb", "[T] (assumed) cccc", "[T] dddd",
        ]

    def test_facts_go_only_once_the_guesses_are_gone(self, tmp_path):
        (tmp_path / "memories.txt").write_text("[T] aaaa\n[T] bbbb\n")

        persona_store.append_memory(tmp_path, "T", "cccc", 22)

        assert self._lines(tmp_path) == ["[T] bbbb", "[T] cccc"]

    def test_the_memory_just_added_is_never_the_one_dropped(self, tmp_path):
        # Even when it is itself an assumption and the only other line is
        # a fact: the newest line always survives, and the budget fits one.
        (tmp_path / "memories.txt").write_text("[T] aaaa\n")

        persona_store.append_memory(tmp_path, "T", "zzzz", 25, assumed=True)

        assert self._lines(tmp_path) == ["[T] (assumed) zzzz"]


class TestPurgeMemoriesToLimit:
    """purge_memories_to_limit(): the editor-side cleanup run when a
    persona's memory_size drops on save. Never raises."""

    def test_zero_budget_deletes_file(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("stale\n")
        purge_memories_to_limit(d, 0)
        assert not (d / "memories.txt").exists()

    def test_zero_budget_with_no_file_is_a_noop(self, tmp_path):
        d = _dir(tmp_path)
        purge_memories_to_limit(d, 0)  # must not raise
        assert not (d / "memories.txt").exists()

    def test_missing_file_is_a_noop(self, tmp_path):
        d = _dir(tmp_path)
        purge_memories_to_limit(d, 100)  # must not raise

    def test_blank_file_is_a_noop(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("\n\n  \n")
        purge_memories_to_limit(d, 100)
        # No needless rewrite: the (pointless) file survives untouched.
        assert (d / "memories.txt").read_text() == "\n\n  \n"

    def test_file_within_limit_is_left_untouched(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("abc\n")
        mtime_before = (d / "memories.txt").stat().st_mtime_ns
        purge_memories_to_limit(d, 100)
        assert (d / "memories.txt").stat().st_mtime_ns == mtime_before

    def test_oldest_lines_purged_when_over_limit(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("a1\na2\na3\na4\n")  # 12 bytes
        # 10-byte budget: pops "a1\n" (9 bytes remain < 10). A 12-byte budget
        # would be a no-op (already within the limit) — don't get cute here.
        purge_memories_to_limit(d, 10)
        assert read_memories(d) == "a2\na3\na4\n"

    def test_single_memory_over_new_limit_deletes_whole_file(self, tmp_path):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
        purge_memories_to_limit(d, 10)
        assert not (d / "memories.txt").exists()

    def test_write_failure_is_swallowed(self, tmp_path, monkeypatch, caplog):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("a1\na2\na3\na4\n")

        def boom(persona_dir, lines):
            raise OSError("disk full")

        monkeypatch.setattr(persona_store, "_write_memories_file", boom)
        purge_memories_to_limit(d, 10)  # must not raise
        assert read_memories(d) == "a1\na2\na3\na4\n"  # survives until next attempt
        assert "failed to purge memories.txt" in caplog.text

    def test_zero_budget_delete_failure_does_not_raise(self, tmp_path, monkeypatch, caplog):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("stale\n")

        def boom(persona_dir):
            raise OSError("read-only directory")

        monkeypatch.setattr(persona_store, "remove_memories_file", boom)
        purge_memories_to_limit(d, 0)  # must not raise
        assert (d / "memories.txt").read_text() == "stale\n"  # survives
        assert "could not delete memories.txt" in caplog.text

    def test_single_memory_over_limit_delete_failure_does_not_raise(self, tmp_path, monkeypatch, caplog):
        d = _dir(tmp_path)
        (d / "memories.txt").write_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")

        def boom(persona_dir):
            raise OSError("read-only directory")

        monkeypatch.setattr(persona_store, "remove_memories_file", boom)
        purge_memories_to_limit(d, 10)  # must not raise
        assert read_memories(d).startswith("aaa")  # survives
        assert "could not delete memories.txt" in caplog.text


class TestLoadPersonaMemorySize:
    def test_missing_frontmatter_key_defaults(self, tmp_path):
        d = make_persona_dir(tmp_path)  # no memory_size in frontmatter
        assert load_persona_from_dir(d).memory_size == DEFAULT_MEMORY_SIZE

    def test_frontmatter_value_is_loaded(self, tmp_path):
        d = make_persona_dir(tmp_path, memory_size=4096)
        assert load_persona_from_dir(d).memory_size == 4096

    def test_zero_frontmatter_value_is_loaded(self, tmp_path):
        # 0 is a legal "memory disabled" value, not an error.
        d = make_persona_dir(tmp_path, memory_size=0)
        assert load_persona_from_dir(d).memory_size == 0

    def test_invalid_frontmatter_value_defaults(self, tmp_path, caplog):
        d = make_persona_dir(tmp_path)
        (d / "prompt.md").write_text(
            "---\ndescription: x\nrouter_hints: h\navatar_color: '#888888'\n"
            "allow_tool_calls: false\nmemory_size: 'big'\n---\n\nYou are Alex.\n"
        )
        with caplog.at_level(logging.WARNING):
            persona = load_persona_from_dir(d)
        assert persona.memory_size == DEFAULT_MEMORY_SIZE
        assert "invalid memory_size" in caplog.text


# ---------------------------------------------------------------------------
# Forgetting one person
# ---------------------------------------------------------------------------

class TestForgetSubject:
    """The small eraser: one relationship out, everything else untouched."""

    def _remember(self, persona_dir, *lines):
        (persona_dir / "memories.txt").write_text("\n".join(lines) + "\n")

    def test_removes_only_that_subjects_lines(self, tmp_path):
        d = _dir(tmp_path)
        self._remember(
            d,
            "[Brad] Brad is a banker.",
            "[Tony] Tony has two dogs.",
            "[Brad] Brad hates boats.",
        )
        assert forget_subject(d, "Brad") == 2
        assert read_memories(d).splitlines() == ["[Tony] Tony has two dogs."]

    def test_subject_match_is_case_insensitive(self, tmp_path):
        # Same rule a rename uses: the tag is structural, and a persona
        # filed under "brad" by a hand-edit is still Brad.
        d = _dir(tmp_path)
        self._remember(d, "[brad] Brad is a banker.")
        assert forget_subject(d, "Brad") == 1
        assert read_memories(d) == ""

    def test_a_mention_inside_someone_elses_memory_survives(self, tmp_path):
        # It is a memory of Tony that happens to name Brad. Deleting it
        # to be thorough would take a fact about Tony with it.
        d = _dir(tmp_path)
        self._remember(d, "[Tony] Tony met Brad at the bar.")
        assert forget_subject(d, "Brad") == 0
        assert "Brad" in read_memories(d)

    def test_untagged_legacy_lines_are_never_matched(self, tmp_path):
        d = _dir(tmp_path)
        self._remember(d, "Likes tea.", "[Brad] Brad is a banker.")
        assert forget_subject(d, "Brad") == 1
        assert read_memories(d).splitlines() == ["Likes tea."]

    def test_forgetting_the_last_subject_removes_the_file(self, tmp_path):
        # No file and an empty file mean the same thing to every reader,
        # and no file is the state the rest of the app produces.
        d = _dir(tmp_path)
        self._remember(d, "[Brad] Brad is a banker.")
        assert forget_subject(d, "Brad") == 1
        assert not (d / "memories.txt").exists()

    def test_unknown_subject_changes_nothing(self, tmp_path):
        d = _dir(tmp_path)
        self._remember(d, "[Brad] Brad is a banker.")
        assert forget_subject(d, "Nobody") == 0
        assert read_memories(d).splitlines() == ["[Brad] Brad is a banker."]

    def test_empty_subject_is_a_no_op(self, tmp_path):
        d = _dir(tmp_path)
        self._remember(d, "[Brad] Brad is a banker.")
        assert forget_subject(d, "   ") == 0
        assert read_memories(d).splitlines() == ["[Brad] Brad is a banker."]


class TestForgetAcquaintance:
    """The met-list is a memory too — the one that decides whether the
    next meeting is a first meeting."""

    def test_removes_one_name_and_keeps_the_rest(self, tmp_path):
        d = _dir(tmp_path)
        (d / "met.txt").write_text("Brad\nTony\n")
        assert forget_acquaintance(d, "Brad") is True
        assert persona_store.read_acquaintances(d) == {"Tony"}

    def test_match_is_case_insensitive(self, tmp_path):
        d = _dir(tmp_path)
        (d / "met.txt").write_text("brad\n")
        assert forget_acquaintance(d, "Brad") is True
        assert persona_store.read_acquaintances(d) == set()

    def test_forgetting_the_last_name_removes_the_file(self, tmp_path):
        d = _dir(tmp_path)
        (d / "met.txt").write_text("Brad\n")
        assert forget_acquaintance(d, "Brad") is True
        assert not (d / "met.txt").exists()

    def test_unknown_name_reports_false(self, tmp_path):
        d = _dir(tmp_path)
        (d / "met.txt").write_text("Tony\n")
        assert forget_acquaintance(d, "Brad") is False
        assert persona_store.read_acquaintances(d) == {"Tony"}

    def test_no_met_file_is_not_an_error(self, tmp_path):
        assert forget_acquaintance(_dir(tmp_path), "Brad") is False


class TestCountNameMentions:
    """What a forget is about to leave behind, counted before it runs."""

    def _remember(self, persona_dir, *lines):
        (persona_dir / "memories.txt").write_text("\n".join(lines) + "\n")

    def test_counts_lines_about_others_that_name_them(self, tmp_path):
        d = _dir(tmp_path)
        self._remember(
            d,
            "[Tony] Tony met Brad at the bar.",
            "[Luna] Luna thinks Brad is funny.",
            "[Brad] Brad is a banker.",
        )
        # The line filed under Brad is not a mention: it is going.
        assert count_name_mentions(d, "Brad") == 2

    def test_matches_whole_words_only(self, tmp_path):
        d = _dir(tmp_path)
        self._remember(d, "[Tony] Tony collects Bradbury first editions.")
        assert count_name_mentions(d, "Brad") == 0

    def test_no_memories_is_zero(self, tmp_path):
        assert count_name_mentions(_dir(tmp_path), "Brad") == 0
