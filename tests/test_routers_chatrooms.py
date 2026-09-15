"""API tests for app/routers/chatrooms.py — room CRUD, persona assignment, settings.

The fixture config has one room ("TNG" with Alex+Luna) and two personas.
The implicit "default" room is not in chatrooms.yaml.
"""


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

class TestListChatrooms:
    def test_list_excludes_implicit_default(self, client):
        resp = client.get("/api/chatrooms")
        assert resp.status_code == 200
        rooms = resp.json()
        assert [r["name"] for r in rooms] == ["TNG"]
        assert rooms[0]["persona_names"] == ["Alex", "Luna"]

    def test_list_all_includes_default_with_every_persona(self, client):
        resp = client.get("/api/chatrooms/all")
        assert resp.status_code == 200
        rooms = resp.json()
        assert rooms[0]["name"] == "default"
        assert rooms[0]["persona_names"] == ["Alex", "Luna"]
        assert [r["name"] for r in rooms[1:]] == ["TNG"]


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

class TestCreateChatroom:
    def test_create_empty_room(self, client):
        resp = client.post("/api/chatrooms", json={"name": "Enterprise"})
        assert resp.status_code == 201
        assert resp.json() == {
            "name": "Enterprise",
            "persona_names": [],
            "typical_length": "detailed",
            "require_player_persona": False,
            "interview": False,
            "interview_goal": "",
        }
        assert [r["name"] for r in client.get("/api/chatrooms").json()] == ["TNG", "Enterprise"]

    def test_create_reserved_default_rejected(self, client):
        assert client.post("/api/chatrooms", json={"name": "default"}).status_code == 409
        assert client.post("/api/chatrooms", json={"name": "Default"}).status_code == 409

    def test_create_blank_name_rejected(self, client):
        resp = client.post("/api/chatrooms", json={"name": "   "})
        assert resp.status_code == 422

    def test_create_invalid_characters_rejected(self, client):
        resp = client.post("/api/chatrooms", json={"name": "bad/name"})
        assert resp.status_code == 422
        assert "only contain" in resp.json()["detail"]

    def test_create_duplicate_rejected_case_insensitively(self, client):
        resp = client.post("/api/chatrooms", json={"name": "tng"})
        assert resp.status_code == 409

    def test_create_too_long_name_rejected(self, client):
        resp = client.post("/api/chatrooms", json={"name": "x" * 21})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Get one
# ---------------------------------------------------------------------------

class TestGetChatroom:
    def test_get_default_returns_all_personas(self, client):
        resp = client.get("/api/chatrooms/default")
        assert resp.status_code == 200
        assert resp.json()["persona_names"] == ["Alex", "Luna"]

    def test_get_configured_room(self, client):
        resp = client.get("/api/chatrooms/TNG")
        assert resp.status_code == 200
        assert resp.json()["name"] == "TNG"

    def test_get_is_case_insensitive(self, client):
        resp = client.get("/api/chatrooms/tng")
        assert resp.status_code == 200

    def test_get_unknown_room_404(self, client):
        assert client.get("/api/chatrooms/NoSuchRoom").status_code == 404


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDeleteChatroom:
    def test_delete_removes_room(self, client):
        client.post("/api/chatrooms", json={"name": "Enterprise"})
        resp = client.delete("/api/chatrooms/Enterprise")
        assert resp.status_code == 204
        assert [r["name"] for r in client.get("/api/chatrooms").json()] == ["TNG"]

    def test_delete_default_rejected(self, client):
        resp = client.delete("/api/chatrooms/default")
        assert resp.status_code == 400

    def test_delete_unknown_room_404(self, client):
        resp = client.delete("/api/chatrooms/NoSuchRoom")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Persona assignment
# ---------------------------------------------------------------------------

class TestAssignPersonas:
    def test_assign_adds_personas_without_duplicates(self, client):
        resp = client.put("/api/chatrooms/TNG/personas", json={"persona_names": ["Luna", "Alex"]})
        assert resp.status_code == 200
        # Already assigned — order preserved, no duplicates.
        assert resp.json()["persona_names"] == ["Alex", "Luna"]

    def test_assign_appends_new_persona(self, client):
        import app.config as app_config
        from app.config import Persona

        personas = app_config.get_personas()
        personas.personas.append(
            Persona(name="Data", system_prompt="You are Data.", router_hints="logic"))
        resp = client.put("/api/chatrooms/TNG/personas", json={"persona_names": ["Data"]})
        assert resp.status_code == 200
        assert resp.json()["persona_names"] == ["Alex", "Luna", "Data"]

    def test_assign_default_room_rejected(self, client):
        resp = client.put("/api/chatrooms/default/personas", json={"persona_names": ["Alex"]})
        assert resp.status_code == 400

    def test_assign_unknown_room_404(self, client):
        resp = client.put("/api/chatrooms/NoSuchRoom/personas", json={"persona_names": ["Alex"]})
        assert resp.status_code == 404

    def test_assign_nonexistent_persona_422(self, client):
        resp = client.put("/api/chatrooms/TNG/personas", json={"persona_names": ["Q"]})
        assert resp.status_code == 422
        assert "does not exist" in resp.json()["detail"]

    def test_assign_empty_list_rejected_by_model(self, client):
        resp = client.put("/api/chatrooms/TNG/personas", json={"persona_names": []})
        assert resp.status_code == 422


class TestRemovePersonaFromRoom:
    def test_remove_persona(self, client):
        resp = client.delete("/api/chatrooms/TNG/personas/Luna")
        assert resp.status_code == 200
        assert resp.json()["persona_names"] == ["Alex"]

    def test_remove_persona_not_in_room_is_noop(self, client):
        resp = client.delete("/api/chatrooms/TNG/personas/Q")
        assert resp.status_code == 200
        assert resp.json()["persona_names"] == ["Alex", "Luna"]

    def test_remove_from_default_room_rejected(self, client):
        resp = client.delete("/api/chatrooms/default/personas/Alex")
        assert resp.status_code == 400

    def test_remove_unknown_room_404(self, client):
        resp = client.delete("/api/chatrooms/NoSuchRoom/personas/Alex")
        assert resp.status_code == 404


class TestSetTypicalLength:
    def test_sets_and_persists_the_tier(self, client):
        resp = client.put("/api/chatrooms/TNG", json={"typical_length": "terse"})
        assert resp.status_code == 200
        assert resp.json()["typical_length"] == "terse"

        assert client.get("/api/chatrooms/TNG").json()["typical_length"] == "terse"

    def test_other_room_fields_survive_the_update(self, client):
        client.put("/api/chatrooms/TNG", json={"require_player_persona": True})
        client.put("/api/chatrooms/TNG", json={"typical_length": "brief"})

        body = client.get("/api/chatrooms/TNG").json()
        assert body["typical_length"] == "brief"
        assert body["require_player_persona"] is True
        assert body["persona_names"] == ["Alex", "Luna"]

    def test_default_room_cannot_be_modified(self, client):
        resp = client.put("/api/chatrooms/default", json={"typical_length": "terse"})
        assert resp.status_code == 400

    def test_default_room_reports_the_global_tier(self, client, monkeypatch):
        import app.config as app_config
        from app.config import TypicalLength
        from tests.factories import make_settings

        settings = make_settings()
        settings.general.typical_length = TypicalLength.BRIEF
        monkeypatch.setattr(app_config, "_settings_cache", settings)

        assert client.get("/api/chatrooms/default").json()["typical_length"] == "brief"

    def test_unknown_room_is_404(self, client):
        resp = client.put("/api/chatrooms/Nope", json={"typical_length": "terse"})
        assert resp.status_code == 404

    def test_invalid_tier_is_rejected(self, client):
        resp = client.put("/api/chatrooms/TNG", json={"typical_length": "medium-ish"})
        assert resp.status_code == 422


class TestRequirePlayerPersona:
    def test_toggles_and_persists(self, client):
        resp = client.put("/api/chatrooms/TNG", json={"require_player_persona": True})
        assert resp.status_code == 200
        assert resp.json()["require_player_persona"] is True
        assert client.get("/api/chatrooms/TNG").json()["require_player_persona"] is True

    def test_the_requirement_persists(self, client):
        # Turning the requirement on must not disturb an existing profile.
        client.put("/api/chatrooms/TNG", json={"typical_length": "terse"})
        client.put("/api/chatrooms/TNG", json={"require_player_persona": True})

        body = client.get("/api/chatrooms/TNG").json()
        assert body["require_player_persona"] is True
        assert body["typical_length"] == "terse"

    def test_default_room_cannot_require_one(self, client):
        resp = client.put("/api/chatrooms/default", json={"require_player_persona": True})
        assert resp.status_code == 400

    def test_default_room_reports_no_requirement(self, client):
        assert client.get("/api/chatrooms/default").json()["require_player_persona"] is False


# ---------------------------------------------------------------------------
# Interview rooms
# ---------------------------------------------------------------------------

class TestInterviewRoom:
    """An interview is a kind of conversation, so it is a property of the
    room. The implicit "default" room has no ChatRoom object, which is
    what makes "today's behaviour there is untouched" structural rather
    than a promise."""

    def test_a_new_room_is_not_an_interview(self, client):
        body = client.post("/api/chatrooms", json={"name": "Sitting"}).json()
        assert body["interview"] is False and body["interview_goal"] == ""

    def test_the_toggle_and_the_goal_persist(self, client):
        resp = client.put("/api/chatrooms/TNG", json={
            "interview": True,
            "interview_goal": "Wes's working life, 1978 to retirement.",
        })
        assert resp.status_code == 200

        body = client.get("/api/chatrooms/TNG").json()
        assert body["interview"] is True
        assert body["interview_goal"] == "Wes's working life, 1978 to retirement."

    def test_the_goal_survives_the_toggle_going_off(self, client):
        # Turning a room back into an interview should not mean retyping
        # what it was for.
        client.put("/api/chatrooms/TNG", json={
            "interview": True, "interview_goal": "His working life."})
        client.put("/api/chatrooms/TNG", json={"interview": False})

        body = client.get("/api/chatrooms/TNG").json()
        assert body["interview"] is False
        assert body["interview_goal"] == "His working life."

    def test_the_interview_fields_survive_an_unrelated_update(self, client):
        client.put("/api/chatrooms/TNG", json={
            "interview": True, "interview_goal": "His working life."})
        client.put("/api/chatrooms/TNG", json={"typical_length": "brief"})

        body = client.get("/api/chatrooms/TNG").json()
        assert body["interview"] is True
        assert body["interview_goal"] == "His working life."

    def test_an_overlong_goal_is_refused(self, client):
        from app.config import MAX_INTERVIEW_GOAL
        resp = client.put("/api/chatrooms/TNG", json={
            "interview_goal": "x" * (MAX_INTERVIEW_GOAL + 1)})
        assert resp.status_code == 422

    def test_the_default_room_is_never_an_interview(self, client):
        rooms = {r["name"]: r for r in client.get("/api/chatrooms/all").json()}
        assert rooms["default"]["interview"] is False
        assert client.put(
            "/api/chatrooms/default", json={"interview": True}).status_code == 400

    def test_an_old_chatrooms_file_loads_with_the_flag_off(self, tmp_project_root):
        # A new key with a default needs no migration; an existing
        # chatrooms.yaml written before this field must still load.
        import yaml
        from app.config import ChatRoomsConfig
        raw = yaml.safe_load(
            "chat_rooms:\n  - name: TNG\n    persona_names: [Alex]\n"
            "    typical_length: brief\n"
        )
        config = ChatRoomsConfig(**raw)
        assert config.chat_rooms[0].interview is False
        assert config.chat_rooms[0].interview_goal == ""
