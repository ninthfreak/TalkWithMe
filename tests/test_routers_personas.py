"""API tests for app/routers/personas.py — persona CRUD, cascades, clone, avatars."""

import pytest

from tests.factories import make_personas


# ---------------------------------------------------------------------------
# GET /api/personas
# ---------------------------------------------------------------------------

class TestListPersonas:
    def test_returns_all_personas_with_tts_capability_flags(self, client):
        resp = client.get("/api/personas")
        assert resp.status_code == 200
        body = resp.json()
        assert [p["name"] for p in body] == ["Alex", "Luna"]
        by_name = {p["name"]: p for p in body}
        assert by_name["Alex"]["tts_capable"] is False   # no reference audio
        assert by_name["Luna"]["tts_capable"] is True    # audio + transcript
        assert by_name["Luna"]["description"] == "A philosophical poet"


# ---------------------------------------------------------------------------
# POST /api/personas
# ---------------------------------------------------------------------------

class TestCreatePersona:
    def _payload(self, **overrides):
        payload = {
            "name": "Data",
            "description": "A logic-driven captain",
            "system_prompt": "You are Data.",
            "router_hints": "logic, science",
        }
        payload.update(overrides)
        return payload

    def test_create_appends_persona_and_persists_to_yaml(self, client, tmp_config_dir, tmp_project_root):
        resp = client.post("/api/personas", json=self._payload())
        assert resp.status_code == 201
        assert resp.json()["name"] == "Data"

        # In-memory list now includes it...
        names = [p["name"] for p in client.get("/api/personas").json()]
        assert names == ["Alex", "Luna", "Data"]
        # ...and it landed in config/personas.yaml, never the tracked
        # repo-root copy (writing there is what broke `git pull`).
        assert (tmp_config_dir / "personas.yaml").exists()
        assert not (tmp_project_root / "personas.yaml").exists()

    def test_create_strips_whitespace_in_name(self, client):
        resp = client.post("/api/personas", json=self._payload(name="  Worf  "))
        assert resp.status_code == 201
        assert resp.json()["name"] == "Worf"

    def test_create_blank_name_rejected(self, client):
        resp = client.post("/api/personas", json=self._payload(name="   "))
        assert resp.status_code == 422

    def test_create_reserved_name_user_rejected(self, client):
        resp = client.post("/api/personas", json=self._payload(name="user"))
        assert resp.status_code == 422
        assert "reserved" in resp.json()["detail"]

    def test_create_reserved_name_case_insensitive(self, client):
        resp = client.post("/api/personas", json=self._payload(name="USER"))
        assert resp.status_code == 422

    def test_create_duplicate_name_rejected_case_insensitively(self, client):
        resp = client.post("/api/personas", json=self._payload(name="alex"))
        assert resp.status_code == 409

    def test_create_validation_rejects_missing_system_prompt(self, client):
        payload = self._payload()
        del payload["system_prompt"]
        resp = client.post("/api/personas", json=payload)
        assert resp.status_code == 422

    def test_create_validation_rejects_long_name(self, client):
        resp = client.post("/api/personas", json=self._payload(name="x" * 26))
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/personas/{name}/detail
# ---------------------------------------------------------------------------

class TestGetPersonaDetail:
    def test_returns_all_editable_fields(self, client):
        resp = client.get("/api/personas/Luna/detail")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Luna"
        assert body["system_prompt"] == "You are Luna, a philosophical poet."
        assert body["reference_audio"] == "reference/luna.wav"
        assert body["reference_audio_language"] == "en"
        assert body["tts_capable"] is True

    def test_unknown_persona_404(self, client):
        resp = client.get("/api/personas/NoSuchOne/detail")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/personas/{name}
# ---------------------------------------------------------------------------

class TestUpdatePersona:
    def _payload(self, **overrides):
        payload = {
            "name": "Alex",
            "description": "Updated description",
            "system_prompt": "You are Alex, but updated.",
            "router_hints": "general questions",
        }
        payload.update(overrides)
        return payload

    def test_update_fields(self, client):
        resp = client.put("/api/personas/Alex", json=self._payload())
        assert resp.status_code == 200
        assert resp.json()["description"] == "Updated description"
        assert resp.json()["system_prompt"] == "You are Alex, but updated."

    def test_rename_cascades_to_chatrooms(self, client):
        resp = client.put("/api/personas/Alex", json=self._payload(name="Alexander"))
        assert resp.status_code == 200
        assert resp.json()["name"] == "Alexander"

        rooms = client.get("/api/chatrooms").json()
        tng = next(r for r in rooms if r["name"] == "TNG")
        assert "Alexander" in tng["persona_names"]
        assert "Alex" not in tng["persona_names"]
        assert tng["persona_names"] == ["Alexander", "Luna"]

    def test_rename_to_existing_name_rejected(self, client):
        resp = client.put("/api/personas/Alex", json=self._payload(name="luna"))
        assert resp.status_code == 409

    def test_rename_to_reserved_user_rejected(self, client):
        resp = client.put("/api/personas/Alex", json=self._payload(name="User"))
        assert resp.status_code == 422

    def test_blank_name_rejected(self, client):
        resp = client.put("/api/personas/Alex", json=self._payload(name="  "))
        assert resp.status_code == 422

    def test_unknown_persona_404(self, client):
        resp = client.put("/api/personas/NoSuchOne", json=self._payload(name="NoSuchOne"))
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/personas/{name}
# ---------------------------------------------------------------------------

class TestDeletePersona:
    def test_delete_removes_persona_and_cascades_to_chatrooms(self, client):
        resp = client.delete("/api/personas/Luna")
        assert resp.status_code == 204

        names = [p["name"] for p in client.get("/api/personas").json()]
        assert names == ["Alex"]
        tng = next(r for r in client.get("/api/chatrooms").json() if r["name"] == "TNG")
        assert tng["persona_names"] == ["Alex"]

    def test_delete_unknown_persona_404(self, client):
        resp = client.delete("/api/personas/NoSuchOne")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/personas/{name}/clone
# ---------------------------------------------------------------------------

class TestClonePersona:
    def test_clone_appends_numeric_suffix_and_copies_fields(self, client):
        resp = client.post("/api/personas/Alex/clone")
        assert resp.status_code == 201
        clone = resp.json()
        assert clone["name"] == "Alex_2"
        assert clone["system_prompt"] == "You are Alex, a friendly assistant."
        assert clone["description"] == "A friendly assistant"

    def test_clone_skips_taken_suffixes(self, client):
        client.post("/api/personas/Alex/clone")  # creates Alex_2
        resp = client.post("/api/personas/Alex/clone")
        assert resp.status_code == 201
        assert resp.json()["name"] == "Alex_3"

    def test_clone_of_tts_capable_keeps_reference_files(self, client):
        resp = client.post("/api/personas/Luna/clone")
        assert resp.status_code == 201
        clone = resp.json()
        assert clone["name"] == "Luna_2"
        assert clone["reference_audio"] == "reference/luna.wav"
        assert clone["tts_capable"] is True

    def test_clone_unknown_persona_404(self, client):
        resp = client.post("/api/personas/NoSuchOne/clone")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/personas/{name}/avatar
# ---------------------------------------------------------------------------

class TestGetAvatar:
    def test_no_avatar_configured_404(self, client):
        resp = client.get("/api/personas/Alex/avatar")
        assert resp.status_code == 404

    def test_avatar_file_missing_404(self, client, monkeypatch):
        import app.config as app_config

        personas = make_personas()
        personas.personas[0].avatar_image = "/nonexistent/avatar.png"
        monkeypatch.setattr(app_config, "_personas_cache", personas)

        resp = client.get("/api/personas/Alex/avatar")
        assert resp.status_code == 404

    def test_serves_avatar_bytes(self, client, monkeypatch, tmp_path):
        import app.config as app_config

        avatar = tmp_path / "alex.png"
        avatar.write_bytes(b"\x89PNG fake bytes")
        personas = make_personas()
        personas.personas[0].avatar_image = str(avatar)
        monkeypatch.setattr(app_config, "_personas_cache", personas)

        resp = client.get("/api/personas/Alex/avatar")
        assert resp.status_code == 200
        assert resp.content == b"\x89PNG fake bytes"

    def test_unknown_persona_avatar_404(self, client):
        resp = client.get("/api/personas/NoSuchOne/avatar")
        assert resp.status_code == 404


class TestCascadePreservesRoomSettings:
    """Renaming or deleting a persona must not reset the rooms it touches.

    The cascades used to rebuild each ChatRoom from just name +
    persona_names, which silently reset echo_chamber (and would have reset
    typical_length) on every rename and delete.
    """

    def _configure_room(self, client):
        client.put("/api/chatrooms/TNG", json={
            "typical_length": "terse", "require_player_persona": True,
        })

    def test_rename_preserves_room_settings(self, client):
        self._configure_room(client)
        detail = client.get("/api/personas/Alex/detail").json()
        detail["name"] = "Alexander"
        assert client.put("/api/personas/Alex", json=detail).status_code == 200

        body = client.get("/api/chatrooms/TNG").json()
        assert body["persona_names"] == ["Alexander", "Luna"]
        assert body["typical_length"] == "terse"

    def test_delete_preserves_room_settings(self, client):
        self._configure_room(client)
        assert client.delete("/api/personas/Alex").status_code == 204

        body = client.get("/api/chatrooms/TNG").json()
        assert body["persona_names"] == ["Luna"]
        assert body["typical_length"] == "terse"


# ---------------------------------------------------------------------------
# Persona names have to survive a round trip
# ---------------------------------------------------------------------------

class TestPersonaNameValidation:
    """A name goes into a URL path segment and into a "[Name]: " tag.

    A name containing "/" returned 201 on create and then 404 on every
    edit, delete and clone: permanently stuck in personas.yaml and still
    answering. Refusing it up front is the only place that can be fixed.
    """

    def _payload(self, name):
        return {
            "name": name,
            "description": "d",
            "system_prompt": "You are someone.",
            "router_hints": "things",
        }

    BAD_NAMES = ["a/b", "a\\b", "line\nbreak", "tab\there", "   ", "", "K" * 26]

    @pytest.mark.parametrize("name", BAD_NAMES)
    def test_create_refuses_a_name_that_cannot_round_trip(self, client, name):
        assert client.post("/api/personas", json=self._payload(name)).status_code == 422
        assert [p["name"] for p in client.get("/api/personas").json()] == ["Alex", "Luna"]

    @pytest.mark.parametrize("name", BAD_NAMES)
    def test_update_refuses_the_same_names(self, client, name):
        detail = client.get("/api/personas/Alex/detail").json()
        detail["name"] = name
        assert client.put("/api/personas/Alex", json=detail).status_code == 422
        assert client.get("/api/personas/Alex/detail").status_code == 200

    def test_a_name_at_the_limit_is_accepted(self, client):
        name = "K" * 25
        assert client.post("/api/personas", json=self._payload(name)).status_code == 201
        # And it is still reachable by every route that takes a name.
        assert client.get(f"/api/personas/{name}/detail").status_code == 200
        assert client.delete(f"/api/personas/{name}").status_code == 204

    def test_surrounding_whitespace_is_trimmed(self, client):
        assert client.post(
            "/api/personas", json=self._payload("  Data  ")
        ).json()["name"] == "Data"

    def test_spaces_and_punctuation_inside_a_name_are_fine(self, client):
        assert client.post(
            "/api/personas", json=self._payload("Dr. Mary-Anne O'Neil")
        ).status_code == 201


class TestCloneNameFitsTheLimit:
    """A clone must be born editable.

    `{name}_{suffix}` on a name already at the cap produced a persona that
    `PUT /api/personas/{name}` then rejected — saved under a name it would
    not accept back.
    """

    def _make(self, client, name):
        return client.post("/api/personas", json={
            "name": name, "description": "d",
            "system_prompt": "You are someone.", "router_hints": "things",
        })

    def test_a_long_name_is_trimmed_to_fit(self, client):
        long_name = "K" * 25
        self._make(client, long_name)

        clone = client.post(f"/api/personas/{long_name}/clone", json={})
        assert clone.status_code == 201
        new_name = clone.json()["name"]
        assert len(new_name) <= 25
        assert new_name.endswith("_2")

        # The whole point: the clone can now be edited.
        detail = client.get(f"/api/personas/{new_name}/detail").json()
        assert client.put(f"/api/personas/{new_name}", json=detail).status_code == 200

    def test_repeated_clones_stay_unique(self, client):
        long_name = "K" * 25
        self._make(client, long_name)
        names = {
            client.post(f"/api/personas/{long_name}/clone", json={}).json()["name"]
            for _ in range(3)
        }
        assert len(names) == 3
        assert all(len(n) <= 25 for n in names)

    def test_a_short_name_is_untouched(self, client):
        assert client.post(
            "/api/personas/Alex/clone", json={}
        ).json()["name"] == "Alex_2"
