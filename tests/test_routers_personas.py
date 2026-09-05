"""API tests for app/routers/personas.py — persona CRUD, cascades, clone, avatars.

Personas are backed by real directories (make_personas_in_dir), so every
test can assert both the API response and the files on disk. Create/update
are multipart/form-data; the TestClient sends form data as urlencoded when
no files are attached, which FastAPI's Form fields parse identically.
"""

import pytest

import app.config as app_config
from app.services import persona_store
from tests.factories import make_personas_in_dir, rescan_personas


@pytest.fixture
def personas_root(tmp_project_root):
    """Materialize the stock Alex/Luna set as real directories and point the
    in-memory cache at the scan result (persona_dir set, real on-disk paths)."""
    root = tmp_project_root / "Personas"
    app_config.set_personas_cache(make_personas_in_dir(root))
    return root


# ---------------------------------------------------------------------------
# GET /api/personas
# ---------------------------------------------------------------------------

class TestListPersonas:
    def test_returns_all_personas_with_tts_capability_flags(self, client, personas_root):
        resp = client.get("/api/personas")
        assert resp.status_code == 200
        body = resp.json()
        assert [p["name"] for p in body] == ["Alex", "Luna"]
        by_name = {p["name"]: p for p in body}
        assert by_name["Alex"]["tts_capable"] is False   # no reference audio
        assert by_name["Luna"]["tts_capable"] is True    # audio + transcript
        assert by_name["Luna"]["description"] == "A philosophical poet"
        # avatar_image is a presence flag, not a path.
        assert by_name["Alex"]["avatar_image"] is False
        assert by_name["Luna"]["avatar_image"] is False


# ---------------------------------------------------------------------------
# POST /api/personas
# ---------------------------------------------------------------------------

class TestCreatePersona:
    def _data(self, **overrides):
        data = {
            "name": "Data",
            "description": "A logic-driven captain",
            "system_prompt": "You are Data.",
            "router_hints": "logic, science",
            "avatar_color": "#4A90D9",
            "reference_audio_language": "en",
            "allow_tool_calls": "false",
            "reference_audio_transcript": "",
            "remove_avatar_image": "false",
            "remove_reference_audio": "false",
        }
        data.update(overrides)
        return data

    def test_create_writes_persona_directory_and_updates_cache(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data())
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "Data"
        assert body["description"] == "A logic-driven captain"
        assert body["system_prompt"] == "You are Data."
        assert body["avatar_image"] is False
        assert body["reference_audio"] is False
        assert body["reference_audio_transcript"] is None
        assert body["tts_capable"] is False

        # In-memory list now includes it...
        names = [p["name"] for p in client.get("/api/personas").json()]
        assert names == ["Alex", "Luna", "Data"]
        # ...and it landed in its own directory (not in personas.yaml).
        persona_dir = personas_root / "Data"
        assert (persona_dir / "prompt.md").exists()
        assert (persona_dir / "language.txt").read_text() == "en"
        # No transcript text was sent -> no ref.txt on disk.
        assert not (persona_dir / "ref.txt").exists()

    def test_create_with_file_uploads(self, client, personas_root):
        data = self._data(reference_audio_transcript="Beverage of choice: red or clear.")
        files = {
            "avatar_image": ("data.png", b"PNGDATA", "image/png"),
            "reference_audio": ("data.wav", b"WAVDATA", "audio/wav"),
        }
        resp = client.post("/api/personas", data=data, files=files)
        assert resp.status_code == 201
        body = resp.json()
        assert body["avatar_image"] is True
        assert body["reference_audio"] is True
        assert body["reference_audio_transcript"] == "Beverage of choice: red or clear."
        assert body["tts_capable"] is True

        persona_dir = personas_root / "Data"
        assert (persona_dir / "image.png").read_bytes() == b"PNGDATA"
        assert (persona_dir / "ref.wav").read_bytes() == b"WAVDATA"
        assert (persona_dir / "ref.txt").read_text() == "Beverage of choice: red or clear."

    def test_create_strips_whitespace_in_name(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data(name="  Worf  "))
        assert resp.status_code == 201
        assert resp.json()["name"] == "Worf"
        assert (personas_root / "Worf").is_dir()

    def test_create_blank_name_rejected(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data(name="   "))
        assert resp.status_code == 422

    def test_create_name_without_usable_directory_chars_rejected(self, client, personas_root):
        # '---' would actually be a legal directory name; '???' sanitizes
        # to nothing and must be rejected.
        resp = client.post("/api/personas", data=self._data(name="???"))
        assert resp.status_code == 422
        assert "letter, number, space, hyphen or underscore" in resp.json()["detail"]

    def test_create_reserved_name_user_rejected(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data(name="user"))
        assert resp.status_code == 422
        assert "reserved" in resp.json()["detail"]

    def test_create_reserved_name_case_insensitive(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data(name="USER"))
        assert resp.status_code == 422

    def test_create_duplicate_name_rejected_case_insensitively(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data(name="alex"))
        assert resp.status_code == 409

    def test_create_validation_rejects_missing_system_prompt(self, client, personas_root):
        data = self._data()
        del data["system_prompt"]
        resp = client.post("/api/personas", data=data)
        assert resp.status_code == 422

    def test_create_validation_rejects_long_name(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data(name="x" * 26))
        assert resp.status_code == 422

    def test_create_validation_rejects_bad_language_length(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data(reference_audio_language="eng"))
        assert resp.status_code == 422

    def test_create_directory_collision_gets_unique_dir_name(self, client, personas_root):
        # "O'Brien" sanitizes to the "OBrien" directory.
        resp = client.post("/api/personas", data=self._data(name="O'Brien"))
        assert resp.status_code == 201
        assert (personas_root / "OBrien").is_dir()

        # A different persona name sanitizing to the same directory gets a
        # suffixed directory instead of clobbering the first one.
        resp = client.post("/api/personas", data=self._data(name="OBrien"))
        assert resp.status_code == 201
        assert (personas_root / "OBrien_2").is_dir()

    def test_create_unsupported_image_extension_rejected(self, client, personas_root):
        resp = client.post(
            "/api/personas", data=self._data(),
            files={"avatar_image": ("data.bmp", b"BMP", "image/bmp")},
        )
        assert resp.status_code == 422
        assert "Unsupported avatar image" in resp.json()["detail"]

    def test_create_unsupported_audio_extension_rejected(self, client, personas_root):
        resp = client.post(
            "/api/personas", data=self._data(),
            files={"reference_audio": ("data.mp3", b"MP3", "audio/mpeg")},
        )
        assert resp.status_code == 422
        assert "wav" in resp.json()["detail"]

    def test_create_oversized_image_rejected(self, client, personas_root):
        huge = b"\x00" * (persona_store.MAX_IMAGE_BYTES + 1)
        resp = client.post(
            "/api/personas", data=self._data(),
            files={"avatar_image": ("data.png", huge, "image/png")},
        )
        assert resp.status_code == 422
        assert "limit" in resp.json()["detail"]

    def test_create_oversized_audio_rejected(self, client, personas_root):
        huge = b"\x00" * (persona_store.MAX_AUDIO_BYTES + 1)
        resp = client.post(
            "/api/personas", data=self._data(),
            files={"reference_audio": ("data.wav", huge, "audio/wav")},
        )
        assert resp.status_code == 422

    def test_create_memory_size_defaults_to_8192(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data())
        assert resp.status_code == 201
        assert resp.json()["memory_size"] == 8192
        fields, _ = persona_store.parse_frontmatter(
            (personas_root / "Data" / "prompt.md").read_text()
        )
        assert fields["memory_size"] == 8192

    def test_create_memory_size_custom_value_persisted(self, client, personas_root):
        resp = client.post("/api/personas", data=self._data(memory_size="4096"))
        assert resp.status_code == 201
        assert resp.json()["memory_size"] == 4096
        fields, _ = persona_store.parse_frontmatter(
            (personas_root / "Data" / "prompt.md").read_text()
        )
        assert fields["memory_size"] == 4096

    @pytest.mark.parametrize("value", ["-1", "16385", "not-a-number"])
    def test_create_memory_size_out_of_range_rejected(self, client, personas_root, value):
        resp = client.post("/api/personas", data=self._data(memory_size=value))
        assert resp.status_code == 422
        assert not (personas_root / "Data").exists()


# ---------------------------------------------------------------------------
# GET /api/personas/{name}/detail
# ---------------------------------------------------------------------------

class TestGetPersonaDetail:
    def test_returns_all_editable_fields(self, client, personas_root):
        resp = client.get("/api/personas/Luna/detail")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Luna"
        assert body["description"] == "A philosophical poet"
        assert body["system_prompt"] == "You are Luna, a philosophical poet."
        assert body["router_hints"] == "philosophy, feelings"
        assert body["avatar_image"] is False
        assert body["reference_audio"] is True
        # The transcript is file CONTENTS, not a path.
        assert body["reference_audio_transcript"] == "The stars are just pinpricks in the dark."
        assert body["reference_audio_language"] == "en"
        assert body["allow_tool_calls"] is False
        assert body["tts_capable"] is True
        # Stock personas predate the field: no key in frontmatter -> default.
        assert body["memory_size"] == 8192

    def test_detail_reports_custom_memory_size(self, client, personas_root):
        # Rewrite Luna's prompt.md with a custom budget, then refresh the
        # cache by re-scanning (rescan writes nothing — the custom file
        # must survive).
        persona_store.write_prompt_md(
            personas_root / "Luna",
            name="Luna",
            description="A philosophical poet",
            router_hints="philosophy, feelings",
            avatar_color="#888888",
            allow_tool_calls=False,
            system_prompt="You are Luna, a philosophical poet.",
            memory_size=2048,
        )
        app_config.set_personas_cache(rescan_personas(personas_root))

        body = client.get("/api/personas/Luna/detail").json()
        assert body["memory_size"] == 2048

    def test_detail_without_files_reports_absent(self, client, personas_root):
        body = client.get("/api/personas/Alex/detail").json()
        assert body["avatar_image"] is False
        assert body["reference_audio"] is False
        assert body["reference_audio_transcript"] is None
        assert body["tts_capable"] is False

    def test_unknown_persona_404(self, client, personas_root):
        resp = client.get("/api/personas/NoSuchOne/detail")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/personas/{name}
# ---------------------------------------------------------------------------

class TestUpdatePersona:
    def _data(self, **overrides):
        data = {
            "name": "Alex",
            "description": "Updated description",
            "system_prompt": "You are Alex, but updated.",
            "router_hints": "general questions",
            "avatar_color": "#4A90D9",
            "reference_audio_language": "en",
            "allow_tool_calls": "false",
            "reference_audio_transcript": "",
            # REQUIRED on update (no server-side default): an omitted value
            # must 422 rather than silently reset the memory budget.
            "memory_size": "8192",
            "remove_avatar_image": "false",
            "remove_reference_audio": "false",
        }
        data.update(overrides)
        return data

    def test_update_fields(self, client, personas_root):
        resp = client.put("/api/personas/Alex", data=self._data())
        assert resp.status_code == 200
        assert resp.json()["description"] == "Updated description"
        assert resp.json()["system_prompt"] == "You are Alex, but updated."
        # The changes are on disk, not just in memory.
        prompt = (personas_root / "Alex" / "prompt.md").read_text()
        assert "You are Alex, but updated." in prompt
        assert "Updated description" in prompt

    def test_rename_updates_frontmatter_cascades_and_keeps_directory(self, client, personas_root):
        resp = client.put("/api/personas/Alex", data=self._data(name="Alexander"))
        assert resp.status_code == 200
        assert resp.json()["name"] == "Alexander"

        # The directory keeps its original name; the frontmatter carries
        # the new one (renaming directories would break external paths).
        assert (personas_root / "Alex").is_dir()
        prompt = (personas_root / "Alex" / "prompt.md").read_text()
        assert "name: Alexander" in prompt

        rooms = client.get("/api/chatrooms").json()
        tng = next(r for r in rooms if r["name"] == "TNG")
        assert tng["persona_names"] == ["Alexander", "Luna"]

    def test_rename_to_existing_name_rejected(self, client, personas_root):
        resp = client.put("/api/personas/Alex", data=self._data(name="luna"))
        assert resp.status_code == 409

    def test_rename_to_reserved_user_rejected(self, client, personas_root):
        resp = client.put("/api/personas/Alex", data=self._data(name="User"))
        assert resp.status_code == 422

    def test_blank_name_rejected(self, client, personas_root):
        resp = client.put("/api/personas/Alex", data=self._data(name="  "))
        assert resp.status_code == 422

    def test_unknown_persona_404(self, client, personas_root):
        resp = client.put("/api/personas/NoSuchOne", data=self._data(name="NoSuchOne"))
        assert resp.status_code == 404

    def test_update_with_new_image_replaces_existing(self, client, personas_root):
        # Give Alex a png avatar, then update with a webp: the png must go.
        persona_store.write_avatar_file(personas_root / "Alex", b"OLDPNG", ".png")
        app_config.set_personas_cache(rescan_personas(personas_root))

        resp = client.put(
            "/api/personas/Alex", data=self._data(),
            files={"avatar_image": ("alex.webp", b"NEWWEBP", "image/webp")},
        )
        assert resp.status_code == 200
        assert resp.json()["avatar_image"] is True
        assert not (personas_root / "Alex" / "image.png").exists()
        assert (personas_root / "Alex" / "image.webp").read_bytes() == b"NEWWEBP"

    def test_update_with_new_audio_replaces_existing(self, client, personas_root):
        resp = client.put(
            "/api/personas/Luna",
            data=self._data(name="Luna"),
            files={"reference_audio": ("luna.wav", b"NEWWAV", "audio/wav")},
        )
        assert resp.status_code == 200
        assert (personas_root / "Luna" / "ref.wav").read_bytes() == b"NEWWAV"

    def test_update_remove_image_flag_removes_file(self, client, personas_root):
        persona_store.write_avatar_file(personas_root / "Alex", b"PNGDATA", ".png")
        app_config.set_personas_cache(rescan_personas(personas_root))

        resp = client.put("/api/personas/Alex", data=self._data(remove_avatar_image="true"))
        assert resp.status_code == 200
        assert resp.json()["avatar_image"] is False
        assert not (personas_root / "Alex" / "image.png").exists()

    def test_update_remove_audio_flag_removes_file(self, client, personas_root):
        resp = client.put(
            "/api/personas/Luna",
            data=self._data(name="Luna", remove_reference_audio="true"),
        )
        assert resp.status_code == 200
        assert resp.json()["reference_audio"] is False
        assert resp.json()["tts_capable"] is False  # no audio -> not TTS-capable
        assert not (personas_root / "Luna" / "ref.wav").exists()

    def test_update_blank_transcript_removes_ref_txt(self, client, personas_root):
        resp = client.put(
            "/api/personas/Luna",
            data=self._data(name="Luna", reference_audio_transcript="   "),
        )
        assert resp.status_code == 200
        assert resp.json()["reference_audio_transcript"] is None
        assert resp.json()["tts_capable"] is False
        assert not (personas_root / "Luna" / "ref.txt").exists()

    def test_update_new_transcript_written(self, client, personas_root):
        resp = client.put(
            "/api/personas/Luna",
            data=self._data(name="Luna", reference_audio_transcript="A new transcript."),
        )
        assert resp.status_code == 200
        assert resp.json()["reference_audio_transcript"] == "A new transcript."
        assert (personas_root / "Luna" / "ref.txt").read_text() == "A new transcript."

    def test_update_unsupported_image_extension_rejected(self, client, personas_root):
        resp = client.put(
            "/api/personas/Alex", data=self._data(),
            files={"avatar_image": ("alex.bmp", b"BMP", "image/bmp")},
        )
        assert resp.status_code == 422
        # The existing files are untouched by a failed update.
        assert not (personas_root / "Alex" / "image.bmp").exists()

    def test_update_requires_memory_size(self, client, personas_root):
        data = self._data()
        del data["memory_size"]
        resp = client.put("/api/personas/Alex", data=data)
        assert resp.status_code == 422

    def test_update_memory_size_persisted_to_frontmatter(self, client, personas_root):
        resp = client.put("/api/personas/Alex", data=self._data(memory_size="4096"))
        assert resp.status_code == 200
        assert resp.json()["memory_size"] == 4096
        fields, _ = persona_store.parse_frontmatter(
            (personas_root / "Alex" / "prompt.md").read_text()
        )
        assert fields["memory_size"] == 4096

    def test_update_out_of_range_memory_size_rejected(self, client, personas_root):
        resp = client.put("/api/personas/Alex", data=self._data(memory_size="16385"))
        assert resp.status_code == 422

    def test_update_preserves_memories_within_new_limit(self, client, personas_root):
        memories = personas_root / "Alex" / "memories.txt"
        memories.write_text("abc\n")  # 4 bytes, well within 8192
        resp = client.put("/api/personas/Alex", data=self._data(memory_size="8192"))
        assert resp.status_code == 200
        assert memories.read_text() == "abc\n"

    def test_update_lowered_memory_size_purges_oldest_first(self, client, personas_root):
        memories = personas_root / "Alex" / "memories.txt"
        memories.write_text("a1\na2\na3\na4\n")  # 12 bytes
        resp = client.put("/api/personas/Alex", data=self._data(memory_size="6"))
        assert resp.status_code == 200
        assert resp.json()["memory_size"] == 6
        # Oldest lines dropped until under the new budget; newest survives.
        assert memories.read_text() == "a4\n"

    def test_update_memory_size_zero_deletes_memories(self, client, personas_root):
        memories = personas_root / "Alex" / "memories.txt"
        memories.write_text("stale\n")
        resp = client.put("/api/personas/Alex", data=self._data(memory_size="0"))
        assert resp.status_code == 200
        assert resp.json()["memory_size"] == 0
        assert not memories.exists()

    def test_update_clear_memories_flag_deletes_file(self, client, personas_root):
        memories = personas_root / "Alex" / "memories.txt"
        memories.write_text("the user likes tea\n")
        resp = client.put(
            "/api/personas/Alex",
            data=self._data(memory_size="8192", clear_memories="true"),
        )
        assert resp.status_code == 200
        assert not memories.exists()
        # The persona itself is still fine — only the memories went.
        assert resp.json()["name"] == "Alex"


# ---------------------------------------------------------------------------
# DELETE /api/personas/{name}
# ---------------------------------------------------------------------------

class TestDeletePersona:
    def test_delete_removes_directory_cache_entry_and_cascades(self, client, personas_root):
        resp = client.delete("/api/personas/Luna")
        assert resp.status_code == 204

        names = [p["name"] for p in client.get("/api/personas").json()]
        assert names == ["Alex"]
        tng = next(r for r in client.get("/api/chatrooms").json() if r["name"] == "TNG")
        assert tng["persona_names"] == ["Alex"]
        # The whole directory is gone.
        assert not (personas_root / "Luna").exists()

    def test_delete_unknown_persona_404(self, client, personas_root):
        resp = client.delete("/api/personas/NoSuchOne")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/personas/{name}/clone
# ---------------------------------------------------------------------------

class TestClonePersona:
    def test_clone_copies_files_and_keeps_system_prompt(self, client, personas_root):
        resp = client.post("/api/personas/Luna/clone")
        assert resp.status_code == 201
        clone = resp.json()
        assert clone["name"] == "Luna_2"
        assert clone["system_prompt"] == "You are Luna, a philosophical poet."
        assert clone["description"] == "A philosophical poet"
        assert clone["reference_audio"] is True
        assert clone["tts_capable"] is True

        new_dir = personas_root / "Luna_2"
        # Files were copied, not merely referenced.
        assert (new_dir / "ref.wav").read_bytes() == b"RIFF-fake-wav"
        assert (new_dir / "ref.txt").read_text() == "The stars are just pinpricks in the dark."
        # The clone's prompt.md still carries the persona's prompt.
        assert "You are Luna, a philosophical poet." in (new_dir / "prompt.md").read_text()

    def test_clone_of_frontmatter_named_persona_rewrites_name_field(self, client, personas_root):
        # "O'Brien" lives in the "OBrien" directory, so its prompt.md carries
        # an explicit `name:` field. A raw copytree would leave the clone
        # claiming the source's name — the rewrite must fix that.
        create_data = {
            "name": "O'Brien",
            "description": "A gruff counselor",
            "system_prompt": "You are O'Brien.",
            "router_hints": "feelings",
            "avatar_color": "#FF0000",
            "reference_audio_language": "en",
            "allow_tool_calls": "false",
            "reference_audio_transcript": "",
            "remove_avatar_image": "false",
            "remove_reference_audio": "false",
        }
        assert client.post("/api/personas", data=create_data).status_code == 201
        assert (personas_root / "OBrien" / "prompt.md").exists()

        resp = client.post("/api/personas/O'Brien/clone")
        assert resp.status_code == 201
        assert resp.json()["name"] == "O'Brien_2"

        fields, _ = persona_store.parse_frontmatter(
            (personas_root / "OBrien_2" / "prompt.md").read_text()
        )
        assert fields["name"] == "O'Brien_2"

    def test_clone_carries_over_memory_size(self, client, personas_root):
        create_data = {
            "name": "Data",
            "description": "A logic-driven captain",
            "system_prompt": "You are Data.",
            "router_hints": "logic, science",
            "avatar_color": "#4A90D9",
            "reference_audio_language": "en",
            "allow_tool_calls": "false",
            "reference_audio_transcript": "",
            "memory_size": "4096",
            "remove_avatar_image": "false",
            "remove_reference_audio": "false",
        }
        assert client.post("/api/personas", data=create_data).status_code == 201

        resp = client.post("/api/personas/Data/clone")
        assert resp.status_code == 201
        clone = resp.json()
        assert clone["name"] == "Data_2"
        assert clone["memory_size"] == 4096
        # ...and it is on disk in the clone's own frontmatter, not just in
        # the response (a lost key would fall back to the default on reload).
        fields, _ = persona_store.parse_frontmatter(
            (personas_root / "Data_2" / "prompt.md").read_text()
        )
        assert fields["memory_size"] == 4096

    def test_clone_skips_taken_suffixes(self, client, personas_root):
        client.post("/api/personas/Alex/clone")  # creates Alex_2
        resp = client.post("/api/personas/Alex/clone")
        assert resp.status_code == 201
        assert resp.json()["name"] == "Alex_3"

    def test_clone_unknown_persona_404(self, client, personas_root):
        resp = client.post("/api/personas/NoSuchOne/clone")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/personas/{name}/avatar
# ---------------------------------------------------------------------------

class TestGetAvatar:
    def test_no_avatar_configured_404(self, client, personas_root):
        resp = client.get("/api/personas/Alex/avatar")
        assert resp.status_code == 404

    def test_avatar_file_missing_on_disk_404(self, client, personas_root):
        persona_store.write_avatar_file(personas_root / "Alex", b"PNGDATA", ".png")
        app_config.set_personas_cache(rescan_personas(personas_root))
        (personas_root / "Alex" / "image.png").unlink()  # cache still points at it

        resp = client.get("/api/personas/Alex/avatar")
        assert resp.status_code == 404

    def test_serves_avatar_bytes(self, client, personas_root):
        persona_store.write_avatar_file(personas_root / "Alex", b"\x89PNG fake bytes", ".png")
        app_config.set_personas_cache(rescan_personas(personas_root))

        resp = client.get("/api/personas/Alex/avatar")
        assert resp.status_code == 200
        assert resp.content == b"\x89PNG fake bytes"

    def test_unknown_persona_avatar_404(self, client, personas_root):
        resp = client.get("/api/personas/NoSuchOne/avatar")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/personas/{name}/reference-audio
# ---------------------------------------------------------------------------

class TestGetReferenceAudio:
    def test_serves_ref_wav(self, client, personas_root):
        resp = client.get("/api/personas/Luna/reference-audio")
        assert resp.status_code == 200
        assert resp.content == b"RIFF-fake-wav"
        assert resp.headers["content-type"] == "audio/wav"

    def test_no_reference_audio_404(self, client, personas_root):
        resp = client.get("/api/personas/Alex/reference-audio")
        assert resp.status_code == 404

    def test_reference_audio_missing_on_disk_404(self, client, personas_root):
        (personas_root / "Luna" / "ref.wav").unlink()  # cache still points at it
        resp = client.get("/api/personas/Luna/reference-audio")
        assert resp.status_code == 404

    def test_unknown_persona_404(self, client, personas_root):
        resp = client.get("/api/personas/NoSuchOne/reference-audio")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Behaviour carried over from before the directory rewrite
# ---------------------------------------------------------------------------

def _persona_form(**overrides):
    data = {
        "name": "Data",
        "description": "A logic-driven captain",
        "system_prompt": "You are Data.",
        "router_hints": "logic, science",
        "avatar_color": "#4A90D9",
        "reference_audio_language": "en",
        "allow_tool_calls": "false",
        "length_bias": "match",
        # Required on update (no default), so the shared helper carries it.
        "memory_size": "8192",
        "reference_audio_transcript": "",
        "remove_avatar_image": "false",
        "remove_reference_audio": "false",
    }
    data.update(overrides)
    return data


class TestCascadePreservesRoomSettings:
    """Renaming or deleting a persona must not reset the rooms it touches.

    The cascades rebuild each room's persona_names. Building a fresh
    ChatRoom instead of copying silently resets every field the call site
    forgets to list — which is how echo_chamber used to get wiped, and
    would now wipe typical_length and require_player_persona.
    """

    def _configure_room(self, client):
        client.put("/api/chatrooms/TNG", json={
            "typical_length": "terse", "require_player_persona": True,
        })

    def test_rename_preserves_room_settings(self, client, personas_root):
        self._configure_room(client)
        resp = client.put("/api/personas/Alex", data=_persona_form(name="Alexander"))
        assert resp.status_code == 200

        body = client.get("/api/chatrooms/TNG").json()
        assert body["persona_names"] == ["Alexander", "Luna"]
        assert body["typical_length"] == "terse"
        assert body["require_player_persona"] is True

    def test_delete_preserves_room_settings(self, client, personas_root):
        self._configure_room(client)
        assert client.delete("/api/personas/Alex").status_code == 204

        body = client.get("/api/chatrooms/TNG").json()
        assert body["persona_names"] == ["Luna"]
        assert body["typical_length"] == "terse"
        assert body["require_player_persona"] is True


class TestPersonaNamesStayAddressable:
    """sanitize_persona_dirname keeps the *directory* safe; the name is
    still a path segment on /api/personas/{name}/... A name containing a
    slash returned 201 on create and 404 on every edit, delete and clone."""

    @pytest.mark.parametrize("name", ["a/b", "a\\b", "line\nbreak", "tab\there"])
    def test_a_name_that_cannot_round_trip_is_refused(self, client, personas_root, name):
        assert client.post("/api/personas", data=_persona_form(name=name)).status_code == 422
        assert [p["name"] for p in client.get("/api/personas").json()] == ["Alex", "Luna"]

    @pytest.mark.parametrize("name", ["a/b", "a\\b"])
    def test_update_refuses_the_same_names(self, client, personas_root, name):
        assert client.put(
            "/api/personas/Alex", data=_persona_form(name=name)
        ).status_code == 422
        assert client.get("/api/personas/Alex/detail").status_code == 200

    def test_punctuation_that_survives_a_url_is_still_fine(self, client, personas_root):
        # The directory becomes "OBrien"; the frontmatter keeps the name.
        assert client.post(
            "/api/personas", data=_persona_form(name="Dr. Mary-Anne O'Neil")
        ).status_code == 201


class TestCloneNameFitsTheLimit:
    """A clone must be born editable: `{name}_{suffix}` on a name already
    at the cap produced a persona that PUT then rejected."""

    LONG = "K" * 25

    def _make_long(self, client):
        return client.post("/api/personas", data=_persona_form(name=self.LONG))

    def test_a_long_name_is_trimmed_to_fit(self, client, personas_root):
        assert self._make_long(client).status_code == 201

        clone = client.post(f"/api/personas/{self.LONG}/clone")
        assert clone.status_code == 201
        new_name = clone.json()["name"]
        assert len(new_name) <= 25
        assert new_name.endswith("_2")

        # The whole point: the clone can now be edited.
        assert client.put(
            f"/api/personas/{new_name}", data=_persona_form(name=new_name)
        ).status_code == 200

    def test_repeated_clones_stay_unique(self, client, personas_root):
        self._make_long(client)
        names = {
            client.post(f"/api/personas/{self.LONG}/clone").json()["name"]
            for _ in range(3)
        }
        assert len(names) == 3
        assert all(len(n) <= 25 for n in names)

    def test_a_short_name_is_untouched(self, client, personas_root):
        assert client.post("/api/personas/Alex/clone").json()["name"] == "Alex_2"


class TestLengthBiasRoundTrips:
    """Our per-persona length bias has to survive the move to prompt.md."""

    def test_it_is_saved_and_read_back(self, client, personas_root):
        resp = client.post("/api/personas", data=_persona_form(length_bias="much_shorter"))
        assert resp.status_code == 201
        assert resp.json()["length_bias"] == "much_shorter"
        assert client.get("/api/personas/Data/detail").json()["length_bias"] == "much_shorter"

    def test_it_lands_in_the_frontmatter(self, client, personas_root):
        client.post("/api/personas", data=_persona_form(length_bias="longer"))
        text = (personas_root / "Data" / "prompt.md").read_text(encoding="utf-8")
        assert "length_bias: longer" in text

    def test_it_defaults_to_match_when_absent(self, personas_root):
        # A persona written before the field existed.
        (personas_root / "Alex" / "prompt.md").write_text(
            "---\ndescription: d\n---\n\nYou are Alex.\n", encoding="utf-8"
        )
        persona = persona_store.load_persona_from_dir(personas_root / "Alex")
        assert persona.length_bias.value == "match"

    def test_an_unreadable_value_falls_back_rather_than_hiding_the_persona(
        self, personas_root, caplog
    ):
        (personas_root / "Alex" / "prompt.md").write_text(
            "---\ndescription: d\nlength_bias: sideways\n---\n\nYou are Alex.\n",
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            persona = persona_store.load_persona_from_dir(personas_root / "Alex")
        assert persona.length_bias.value == "match"
        assert "sideways" in caplog.text

    def test_the_clone_carries_it(self, client, personas_root):
        client.post("/api/personas", data=_persona_form(length_bias="much_longer"))
        assert client.post("/api/personas/Data/clone").json()["length_bias"] == "much_longer"


# ---------------------------------------------------------------------------
# POST /api/personas/draft  and  POST /api/personas/preview
# ---------------------------------------------------------------------------

DRAFT_REPLY = """NAME: Rennick
DESCRIPTION: A suspicious harbourmaster
ROUTER_HINTS: boats, cargo
LENGTH_BIAS: shorter
AVATAR_COLOR: #2E7D32
NOTES:
- Stance: answers questions with questions about provenance.
- Negative space: never speculates about unlogged cargo.
SYSTEM_PROMPT:
You run the harbour and assume everyone is smuggling. You never speculate
about cargo you have not seen logged.
"""


def _stub_completion(monkeypatch, *replies):
    """Serve canned completions in order; record the prompts."""
    import app.routers.personas as personas_router

    seen = []
    queue = list(replies)

    async def fake(messages, max_tokens=64, temperature=None, timeout=15.0):
        seen.append({"messages": messages, "max_tokens": max_tokens,
                     "temperature": temperature, "timeout": timeout})
        return queue.pop(0) if queue else ""

    monkeypatch.setattr(personas_router, "chat_completion", fake)
    return seen


class TestDraftPersona:
    def test_a_brief_comes_back_as_a_full_persona(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, DRAFT_REPLY)

        resp = client.post("/api/personas/draft", json={"brief": "a suspicious harbourmaster"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Rennick"
        assert body["description"] == "A suspicious harbourmaster"
        assert body["length_bias"] == "shorter"
        assert body["avatar_color"] == "#2E7D32"
        assert body["system_prompt"].startswith("You run the harbour")
        assert len(body["notes"]) == 2

    def test_the_existing_cast_is_not_sent_to_the_model(self, client, personas_root, monkeypatch):
        # Diversity comes from the levers, not from contrast: the prompt
        # must not grow with the cast, and a character is defined by what
        # it is rather than by what the others are.
        seen = _stub_completion(monkeypatch, DRAFT_REPLY)
        client.post("/api/personas/draft", json={"brief": "a harbourmaster"})

        sent = " ".join(m["content"] for m in seen[0]["messages"])
        assert "Alex" not in sent and "Luna" not in sent

    def test_prose_temperature_and_timeout_not_the_routers(self, client, personas_root, monkeypatch):
        # At the router's 0.1 every draft is the same draft, and at the
        # router's 15s a hundred-word draft times out on a local model,
        # comes back as "", and the server finishes generating into a
        # closed connection.
        seen = _stub_completion(monkeypatch, DRAFT_REPLY)
        client.post("/api/personas/draft", json={"brief": "a harbourmaster"})
        assert seen[0]["temperature"] == 0.8
        assert seen[0]["timeout"] >= 60

    def test_the_dials_reach_the_prompt_as_instructions(self, client, personas_root, monkeypatch):
        seen = _stub_completion(monkeypatch, DRAFT_REPLY)
        client.post("/api/personas/draft", json={
            "brief": "a harbourmaster",
            "dials": {"register": "coarse", "temperament": "unflappable"},
            "details": {"never": "never guesses at cargo"},
        })

        sent = seen[0]["messages"][0]["content"]
        assert "crude turns of phrase" in sent
        assert "nothing gets a rise out of them" in sent
        assert "never guesses at cargo" in sent

    def test_a_junk_dial_does_not_reach_the_model_or_500(self, client, personas_root, monkeypatch):
        # The dropdowns are rendered from the server's own constants, so
        # anything else is a stale page or a hand-rolled request.
        seen = _stub_completion(monkeypatch, DRAFT_REPLY)
        resp = client.post("/api/personas/draft", json={
            "brief": "x",
            "dials": {"register": "sassy", "nonsense": "yes"},
            "details": {"favourite_colour": "blue"},
        })

        assert resp.status_code == 200
        sent = seen[0]["messages"][0]["content"]
        assert "sassy" not in sent and "nonsense" not in sent and "blue" not in sent

    def test_a_brief_alone_still_drafts(self, client, personas_root, monkeypatch):
        # The dials and details are optional; the original one-box flow
        # must keep working for anyone posting without them.
        _stub_completion(monkeypatch, DRAFT_REPLY)
        assert client.post(
            "/api/personas/draft", json={"brief": "a harbourmaster"}
        ).status_code == 200

    def test_a_name_collision_is_resolved_against_the_cast(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, DRAFT_REPLY.replace("NAME: Rennick", "NAME: Luna"))

        body = client.post("/api/personas/draft", json={"brief": "x"}).json()

        assert body["name"] != "Luna"
        assert body["name"].startswith("Luna")

    def test_warnings_come_back_with_the_draft(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "NAME: R\nSYSTEM_PROMPT:\nYou are a friendly assistant.")

        body = client.post("/api/personas/draft", json={"brief": "x"}).json()

        joined = " ".join(body["warnings"])
        assert "generic assistant vocabulary" in joined
        assert "words" in joined            # too short to outweigh the preamble

    def test_an_unreadable_reply_is_a_503_not_a_blank_form(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "I'm sorry, I can't help with that.")
        resp = client.post("/api/personas/draft", json={"brief": "x"})
        assert resp.status_code == 503

    def test_an_empty_reply_is_a_503(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "")
        assert client.post("/api/personas/draft", json={"brief": "x"}).status_code == 503

    def test_a_blank_brief_is_rejected_before_the_llm(self, client, personas_root, monkeypatch):
        seen = _stub_completion(monkeypatch, DRAFT_REPLY)
        assert client.post("/api/personas/draft", json={"brief": ""}).status_code == 422
        assert seen == []

    def test_nothing_is_written_to_disk(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, DRAFT_REPLY)
        client.post("/api/personas/draft", json={"brief": "a harbourmaster"})

        # Drafting is not saving: the form is the review step.
        assert not (personas_root / "Rennick").exists()
        assert [p["name"] for p in client.get("/api/personas").json()] == ["Alex", "Luna"]

    def test_the_route_is_not_shadowed_by_the_name_routes(self, client, personas_root, monkeypatch):
        # /api/personas/draft must not be read as a persona called "draft".
        _stub_completion(monkeypatch, DRAFT_REPLY)
        assert client.post("/api/personas/draft", json={"brief": "x"}).status_code == 200


class TestPreviewPersona:
    def _req(self, **overrides):
        payload = {
            "name": "Rennick",
            "system_prompt": "You run the harbour and never speculate.",
            "description": "A harbourmaster",
            "length_bias": "match",
            "question": "Is the boat seaworthy?",
        }
        payload.update(overrides)
        return payload

    def test_a_draft_answers_without_being_saved(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "Depends whose boat.")

        resp = client.post("/api/personas/preview", json=self._req())

        assert resp.status_code == 200
        body = resp.json()
        assert body["draft"] == {"persona": "Rennick", "text": "Depends whose boat."}
        assert body["comparison"] is None
        assert not (personas_root / "Rennick").exists()

    def test_the_preview_gets_the_prose_timeout(self, client, personas_root, monkeypatch):
        seen = _stub_completion(monkeypatch, "Depends whose boat.")
        client.post("/api/personas/preview", json=self._req())
        assert seen[0]["timeout"] >= 60

    def test_the_preview_uses_the_real_room_preamble(self, client, personas_root, monkeypatch):
        # A preview built from a simpler prompt would preview something the
        # app never runs — including the voice restatement at the end.
        seen = _stub_completion(monkeypatch, "Depends whose boat.")
        client.post("/api/personas/preview", json=self._req())

        system = seen[0]["messages"][0]["content"]
        assert "There is nobody else." in system
        assert system.rstrip().endswith("You run the harbour and never speculate.")

    def test_comparison_runs_an_existing_persona_on_the_same_question(
        self, client, personas_root, monkeypatch
    ):
        seen = _stub_completion(monkeypatch, "Depends whose boat.", "The sea keeps its counsel.")

        body = client.post(
            "/api/personas/preview", json=self._req(compare_with="Luna")
        ).json()

        assert body["comparison"] == {"persona": "Luna", "text": "The sea keeps its counsel."}
        # Same question to both, or the comparison proves nothing.
        assert seen[0]["messages"][1] == seen[1]["messages"][1]

    def test_the_comparison_isolates_the_persona_prompt(self, client, personas_root, monkeypatch):
        # Both sides run in a room of one, so the only difference between
        # the two prompts is the persona being compared. Anything else and
        # the comparison stops being evidence.
        seen = _stub_completion(monkeypatch, "a", "b")
        client.post("/api/personas/preview", json=self._req(compare_with="Luna"))

        draft_sys, other_sys = (c["messages"][0]["content"] for c in seen)
        assert "You are the only one here" in draft_sys
        assert "You are the only one here" in other_sys
        assert seen[0]["max_tokens"] == seen[1]["max_tokens"]

    def test_an_unknown_comparison_persona_is_404(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "a", "b")
        resp = client.post("/api/personas/preview", json=self._req(compare_with="Nobody"))
        assert resp.status_code == 404

    def test_an_empty_draft_reply_is_a_503(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "   ")
        assert client.post("/api/personas/preview", json=self._req()).status_code == 503

    def test_an_empty_comparison_reply_is_dropped_not_fatal(self, client, personas_root, monkeypatch):
        # The draft is what the user asked about; a failed comparison
        # should not lose them the answer they wanted.
        _stub_completion(monkeypatch, "Depends whose boat.", "")
        body = client.post(
            "/api/personas/preview", json=self._req(compare_with="Luna")
        ).json()
        assert body["draft"]["text"] == "Depends whose boat."
        assert body["comparison"] is None

    def test_a_bad_draft_name_is_rejected(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "a")
        assert client.post(
            "/api/personas/preview", json=self._req(name="har/bour")
        ).status_code == 422

    def test_an_unsaved_prompt_can_be_the_comparison(self, client, personas_root, monkeypatch):
        # How a refinement shows a before and after of one character.
        seen = _stub_completion(monkeypatch, "Depends whose boat.", "Whose boat. That first.")

        body = client.post("/api/personas/preview", json=self._req(
            compare_prompt="You run the harbour.", label="After", compare_label="Before",
        )).json()

        assert body["draft"] == {"persona": "After", "text": "Depends whose boat."}
        assert body["comparison"] == {"persona": "Before", "text": "Whose boat. That first."}
        # Same question and same room to both; only the prompt differs.
        assert seen[0]["messages"][1] == seen[1]["messages"][1]
        after_sys, before_sys = (c["messages"][0]["content"] for c in seen)
        assert after_sys.startswith("You run the harbour and never speculate.")
        assert before_sys.startswith("You run the harbour.")

    def test_the_before_and_after_share_a_name(self, client, personas_root, monkeypatch):
        # The room preamble is built from the persona, so a different name
        # on the "before" side would make the two prompts differ twice and
        # the comparison would stop isolating the change.
        seen = _stub_completion(monkeypatch, "a", "b")
        client.post("/api/personas/preview",
                    json=self._req(compare_prompt="You run the harbour."))
        assert seen[0]["messages"][0]["content"].count("Rennick") == \
               seen[1]["messages"][0]["content"].count("Rennick")

    def test_the_unsaved_comparison_labels_itself_when_unlabelled(
        self, client, personas_root, monkeypatch
    ):
        _stub_completion(monkeypatch, "a", "b")
        body = client.post("/api/personas/preview",
                           json=self._req(compare_prompt="You run the harbour.")).json()
        assert body["comparison"]["persona"] == "Rennick (before)"

    def test_two_kinds_of_comparison_at_once_is_rejected(self, client, personas_root, monkeypatch):
        seen = _stub_completion(monkeypatch, "a", "b")
        resp = client.post("/api/personas/preview", json=self._req(
            compare_with="Luna", compare_prompt="You run the harbour."))
        assert resp.status_code == 422
        assert seen == []

    def test_a_blank_comparison_prompt_is_not_a_comparison(self, client, personas_root, monkeypatch):
        seen = _stub_completion(monkeypatch, "Depends whose boat.")
        body = client.post("/api/personas/preview", json=self._req(compare_prompt="  ")).json()
        assert body["comparison"] is None
        assert len(seen) == 1


REFINED_REPLY = """DESCRIPTION: Harbourmaster, coarse
NOTES:
- Changed: he swears now, in the same clipped way he already spoke.
- Kept: the questions about provenance, and never speculating.
SYSTEM_PROMPT:
You run the harbour and you swear about it. You never speculate about cargo
you have not seen logged.
"""


class TestRefinePersona:
    def _req(self, **overrides):
        payload = {
            "name": "Rennick",
            "system_prompt": "You run the harbour. You never speculate about cargo.",
            "description": "A harbourmaster",
            "router_hints": "boats, cargo",
            "length_bias": "shorter",
            "instruction": "make him coarser",
        }
        payload.update(overrides)
        return payload

    def test_an_instruction_comes_back_as_a_revision(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, REFINED_REPLY)

        resp = client.post("/api/personas/refine", json=self._req())

        assert resp.status_code == 200
        body = resp.json()
        assert body["system_prompt"].startswith("You run the harbour and you swear")
        assert body["description"] == "Harbourmaster, coarse"
        assert len(body["notes"]) == 2

    def test_fields_the_revision_left_out_keep_their_values(
        self, client, personas_root, monkeypatch
    ):
        # The prompt asks the model to omit what it did not change, so an
        # omission must not blank the field or reset the length bias.
        _stub_completion(monkeypatch, REFINED_REPLY)

        body = client.post("/api/personas/refine", json=self._req()).json()

        assert body["router_hints"] == "boats, cargo"
        assert body["length_bias"] == "shorter"

    def test_the_persona_and_the_instruction_both_reach_the_model(
        self, client, personas_root, monkeypatch
    ):
        seen = _stub_completion(monkeypatch, REFINED_REPLY)
        client.post("/api/personas/refine", json=self._req())

        sent = seen[0]["messages"][0]["content"]
        assert "You run the harbour. You never speculate about cargo." in sent
        assert "make him coarser" in sent
        assert seen[0]["timeout"] >= 60
        assert seen[0]["temperature"] == 0.8

    def test_the_form_is_refined_not_the_saved_copy(self, client, personas_root, monkeypatch):
        # The user is looking at the form; refining what is on disk would
        # revise a version they cannot see and lose their unsaved edits.
        seen = _stub_completion(monkeypatch, REFINED_REPLY)
        client.post("/api/personas/refine",
                    json=self._req(name="Luna", system_prompt="Edited but not saved."))

        assert "Edited but not saved." in seen[0]["messages"][0]["content"]

    def test_no_name_or_colour_comes_back(self, client, personas_root, monkeypatch):
        # A revision that renamed the character would be a different one.
        _stub_completion(monkeypatch, "NAME: Someone Else\nAVATAR_COLOR: #111111\n" + REFINED_REPLY)

        body = client.post("/api/personas/refine", json=self._req()).json()

        assert "name" not in body and "avatar_color" not in body

    def test_warnings_come_back_with_the_revision(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "NOTES:\n- Softened him.\nSYSTEM_PROMPT:\n"
                                      "You are a friendly and helpful assistant.")

        body = client.post("/api/personas/refine", json=self._req()).json()

        assert any("generic assistant vocabulary" in w for w in body["warnings"])

    def test_a_reply_that_changes_nothing_is_a_503(self, client, personas_root, monkeypatch):
        # Otherwise the user gets an unchanged form and no explanation.
        _stub_completion(monkeypatch, "I'm sorry, I can't help with that.")
        assert client.post("/api/personas/refine", json=self._req()).status_code == 503

    def test_an_empty_reply_is_a_503(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, "")
        assert client.post("/api/personas/refine", json=self._req()).status_code == 503

    def test_a_blank_instruction_is_rejected_before_the_llm(
        self, client, personas_root, monkeypatch
    ):
        seen = _stub_completion(monkeypatch, REFINED_REPLY)
        assert client.post(
            "/api/personas/refine", json=self._req(instruction="")
        ).status_code == 422
        assert seen == []

    def test_nothing_is_written_to_disk(self, client, personas_root, monkeypatch):
        _stub_completion(monkeypatch, REFINED_REPLY)
        before = (personas_root / "Luna" / "prompt.md").read_text()
        client.post("/api/personas/refine", json=self._req(name="Luna"))
        assert (personas_root / "Luna" / "prompt.md").read_text() == before

    def test_the_route_is_not_shadowed_by_the_name_routes(
        self, client, personas_root, monkeypatch
    ):
        _stub_completion(monkeypatch, REFINED_REPLY)
        assert client.post(
            "/api/personas/refine", json=self._req()
        ).status_code == 200
