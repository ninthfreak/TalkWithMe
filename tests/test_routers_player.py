"""API tests for app/routers/player.py — which persona the human is playing.

The player adopts one of the configured personas rather than writing a
character of their own. A room can *require* that they are playing someone,
which is a property of the room; who they play belongs to the player and
applies in whichever room they are in.
"""

import app.config as app_config


class TestGetAdoptedPersona:
    def test_empty_when_playing_yourself(self, client):
        assert client.get("/api/player").json() == {"persona_name": ""}

    def test_returns_what_was_saved(self, client):
        client.put("/api/player", json={"persona_name": "Luna"})
        assert client.get("/api/player").json() == {"persona_name": "Luna"}

    def test_a_deleted_persona_reads_back_as_yourself(self, client):
        # Resolved against the live list on every read: the name reaches the
        # prompt, the stop strings and the reply guard, so a persona that no
        # longer exists must degrade to "yourself" rather than half-apply.
        client.put("/api/player", json={"persona_name": "Luna"})
        assert client.delete("/api/personas/Luna").status_code == 204
        assert client.get("/api/player").json() == {"persona_name": ""}


class TestAdoptPersona:
    def test_saves_and_persists(self, client, tmp_config_dir):
        resp = client.put("/api/player", json={"persona_name": "Luna"})
        assert resp.status_code == 200
        assert resp.json() == {"persona_name": "Luna"}
        assert (tmp_config_dir / "player.yaml").exists()
        assert app_config.get_player().persona_name == "Luna"

    def test_the_name_is_trimmed(self, client):
        assert client.put(
            "/api/player", json={"persona_name": "  Luna  "}
        ).json() == {"persona_name": "Luna"}

    def test_an_empty_name_means_playing_yourself(self, client):
        client.put("/api/player", json={"persona_name": "Luna"})
        assert client.put(
            "/api/player", json={"persona_name": ""}
        ).json() == {"persona_name": ""}

    def test_an_unknown_persona_is_refused(self, client):
        resp = client.put("/api/player", json={"persona_name": "Nobody"})
        assert resp.status_code == 422
        assert app_config.get_player().persona_name == ""

    def test_an_over_long_name_is_rejected(self, client):
        # Matches the persona name cap, so no name that reaches this can
        # ever be one a persona could be created under.
        assert client.put(
            "/api/player", json={"persona_name": "K" * 26}
        ).status_code == 422


class TestOneChoiceAcrossRooms:
    def test_the_same_character_applies_in_every_room(self, client):
        client.put("/api/player", json={"persona_name": "Luna"})
        # Nothing about it is scoped to a room, so switching rooms cannot
        # change who you are.
        assert app_config.get_player().persona_name == "Luna"
        for room in ("default", "TNG"):
            body = client.get(f"/api/chatrooms/{room}").json()
            assert "player_profile" not in body
            assert "persona_name" not in body

    def test_the_requirement_is_still_per_room(self, client):
        client.put("/api/chatrooms/TNG", json={"require_player_persona": True})
        assert client.get("/api/chatrooms/TNG").json()["require_player_persona"] is True
        assert client.get("/api/chatrooms/default").json()["require_player_persona"] is False
