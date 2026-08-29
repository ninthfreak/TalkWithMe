"""API tests for app/routers/player.py — the human's own character.

One profile, not one per room. A room can *require* a character, which is
a property of the room; the character itself belongs to the player and
interacts with whichever room they are in.
"""

import app.config as app_config


class TestGetProfile:
    def test_empty_when_none_is_set(self, client):
        assert client.get("/api/player").json() == {
            "name": "", "description": "", "appearance": "",
        }

    def test_returns_what_was_saved(self, client):
        client.put("/api/player", json={"name": "Kira", "description": "A thief."})
        assert client.get("/api/player").json()["name"] == "Kira"


class TestSetProfile:
    def test_saves_and_persists(self, client, tmp_config_dir):
        resp = client.put("/api/player", json={
            "name": "Kira", "description": "A retired thief.",
            "appearance": "A patched green coat.",
        })
        assert resp.status_code == 200
        assert resp.json() == {
            "name": "Kira", "description": "A retired thief.",
            "appearance": "A patched green coat.",
        }
        assert (tmp_config_dir / "player.yaml").exists()

    def test_fields_are_trimmed(self, client):
        body = client.put("/api/player", json={
            "name": "  Kira  ", "description": " A thief. ", "appearance": "   ",
        }).json()
        assert body == {"name": "Kira", "description": "A thief.", "appearance": ""}

    def test_can_be_cleared(self, client):
        client.put("/api/player", json={"name": "Kira", "description": "A thief."})
        body = client.put("/api/player", json={"name": "", "description": ""}).json()
        assert body["name"] == ""

    def test_over_long_fields_are_rejected(self, client):
        resp = client.put("/api/player", json={"name": "K" * 41, "description": "x"})
        assert resp.status_code == 422


class TestOneProfileAcrossRooms:
    def test_the_same_character_applies_in_every_room(self, client):
        client.put("/api/player", json={"name": "Kira", "description": "A thief."})
        # Nothing about it is scoped to a room, so switching rooms cannot
        # change who you are.
        assert app_config.get_player().profile.name == "Kira"
        for room in ("default", "TNG"):
            assert "player_profile" not in client.get(f"/api/chatrooms/{room}").json()

    def test_the_requirement_is_still_per_room(self, client):
        client.put("/api/chatrooms/TNG", json={"require_player_profile": True})
        assert client.get("/api/chatrooms/TNG").json()["require_player_profile"] is True
        assert client.get("/api/chatrooms/default").json()["require_player_profile"] is False
