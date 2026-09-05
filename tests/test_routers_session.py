"""API tests for app/routers/session.py — session state, reset, personas, room loading."""

from pathlib import Path

import pytest

import app.config as app_config
from app.config import ChatRoomsConfig, PlayerConfig
from app.models import ChatMessage
from app.persistence import persist_message
from app.services import persona_store
from tests.factories import make_personas_in_dir


@pytest.fixture
def personas_root(tmp_project_root):
    """The stock Alex/Luna set as real directories, so memories have a home."""
    root = tmp_project_root / "Personas"
    app_config.set_personas_cache(make_personas_in_dir(root))
    return root


def _remember(personas_root, persona: str, *lines: str):
    (personas_root / persona / persona_store.MEMORIES_FILENAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _add_exchange(room: str, user_text: str, reply_text: str):
    """Add a user + assistant message straight to the global session (persisted)."""
    from app.session import session

    session.set_current_room(room)
    session.add_user_message(user_text, "id-u1")
    session.add_assistant_message(reply_text, "Alex", "id-a1")


class TestGetSession:
    def test_returns_fresh_state(self, client):
        resp = client.get("/api/session")
        assert resp.status_code == 200
        body = resp.json()
        assert body["history"] == []
        assert body["current_room"] == "default"

    def test_reflects_messages_in_history(self, client):
        _add_exchange("default", "hello", "hi there")
        body = client.get("/api/session").json()
        assert [m["role"] for m in body["history"]] == ["user", "assistant"]
        assert body["history"][0]["content"] == "hello"


class TestNewSession:
    def test_new_clears_history_and_persistence(self, client, persistence_root):
        _add_exchange("TNG", "hello", "hi there")
        history_file = persistence_root / "TNG" / "history.json"
        assert history_file.exists()

        resp = client.post("/api/session/new")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared"}

        assert client.get("/api/session").json()["history"] == []
        # The room directory survives, but its files are gone.
        assert (persistence_root / "TNG").is_dir()
        assert not history_file.exists()


class TestUpdateActivePersonas:
    def test_valid_names_applied(self, client):
        resp = client.post("/api/session/personas", json={"active_personas": ["Luna"]})
        assert resp.status_code == 200
        assert resp.json()["active_personas"] == ["Luna"]
        assert client.get("/api/session").json()["active_personas"] == ["Luna"]

    def test_unknown_names_silently_dropped(self, client):
        resp = client.post("/api/session/personas",
                           json={"active_personas": ["Alex", "Q"]})
        assert resp.status_code == 200
        assert resp.json()["active_personas"] == ["Alex"]

    def test_empty_list_rejected_by_model(self, client):
        resp = client.post("/api/session/personas", json={"active_personas": []})
        assert resp.status_code == 422


class TestLoadRoom:
    def _seed_room(self, room: str, persistence_root: Path):
        """Write a couple of messages straight to disk, as an earlier session would have."""
        persist_message(room, ChatMessage(role="user", content="earlier question"), "id-u1")
        persist_message(room, ChatMessage(role="assistant", content="earlier answer",
                                          persona="Alex"), "id-a1")

    def test_load_room_populates_session_and_returns_messages(self, client, persistence_root):
        self._seed_room("TNG", persistence_root)

        resp = client.get("/api/session/load-room/TNG")
        assert resp.status_code == 200
        body = resp.json()
        assert body["room"] == "TNG"
        assert [m["sender"] for m in body["messages"]] == ["USER", "Alex"]
        assert [m["text"] for m in body["messages"]] == ["earlier question", "earlier answer"]

        # The in-memory session now carries the loaded history too.
        session_state = client.get("/api/session").json()
        assert session_state["current_room"] == "TNG"
        assert [m["role"] for m in session_state["history"]] == ["user", "assistant"]

    def test_load_room_replaces_previous_history(self, client, persistence_root):
        self._seed_room("TNG", persistence_root)
        _add_exchange("Solo", "fresh question", "fresh answer")

        # Switching to TNG must not leak Solo's messages.
        client.get("/api/session/load-room/TNG")
        history = client.get("/api/session").json()["history"]
        assert [m["content"] for m in history] == ["earlier question", "earlier answer"]

    def test_load_room_with_no_history_returns_empty(self, client):
        resp = client.get("/api/session/load-room/EmptyRoom")
        assert resp.status_code == 200
        body = resp.json()
        assert body["messages"] == []
        assert body["datetime"] is None


class TestContextInventory:
    """What is stored, so a wipe can be checked rather than assumed."""

    def test_an_empty_install_has_nothing(self, client, personas_root):
        body = client.get("/api/session/context").json()
        assert body == {"rooms": [], "personas": [], "playing_as": ""}

    def test_rooms_and_their_message_counts(self, client, personas_root):
        _add_exchange("TNG", "hello", "hi there")
        body = client.get("/api/session/context").json()
        assert body["rooms"] == [{"room": "TNG", "messages": 2}]

    def test_memories_are_counted_per_persona(self, client, personas_root):
        _remember(personas_root, "Luna", "The user told me they sail.")
        body = client.get("/api/session/context").json()
        assert body["personas"] == [{"persona": "Luna", "memories": 1}]

    def test_who_the_player_is_playing_is_part_of_it(self, client, personas_root, monkeypatch):
        # Not accumulated state, but it reaches every persona in every
        # room, so a "why is this still happening" hunt has to see it.
        monkeypatch.setattr(app_config, "_player_cache", PlayerConfig(persona_name="Luna"))
        assert client.get("/api/session/context").json()["playing_as"] == "Luna"


class TestWipeContext:
    def test_nothing_is_deleted_without_being_asked(self, client, personas_root):
        _add_exchange("TNG", "hello", "hi there")
        _remember(personas_root, "Luna", "The user told me they sail.")

        body = client.post("/api/session/wipe", json={}).json()

        assert body["rooms_cleared"] == [] and body["memories_cleared"] == []
        assert body["remaining"]["rooms"] == [{"room": "TNG", "messages": 2}]

    def test_every_room_goes(self, client, personas_root, persistence_root):
        _add_exchange("TNG", "hello", "hi there")
        _add_exchange("Tavern", "anyone about?", "not really")

        body = client.post("/api/session/wipe", json={"rooms": "all"}).json()

        assert sorted(body["rooms_cleared"]) == ["TNG", "Tavern"]
        assert body["messages_deleted"] == 4
        assert body["remaining"]["rooms"] == []
        assert not (persistence_root / "TNG" / "history.json").exists()

    def test_the_room_in_use_is_emptied_in_memory_too(self, client, personas_root):
        # Clearing only the files would leave this turn's history alive,
        # and the next reply built on a conversation the user just watched
        # disappear.
        _add_exchange("TNG", "hello", "hi there")
        client.post("/api/session/wipe", json={"rooms": "all"})

        assert client.get("/api/session").json()["history"] == []

    def test_only_the_current_room_when_asked(self, client, personas_root):
        _add_exchange("Tavern", "anyone about?", "not really")
        _add_exchange("TNG", "hello", "hi there")   # leaves TNG current

        body = client.post("/api/session/wipe", json={"rooms": "current"}).json()

        assert body["rooms_cleared"] == ["TNG"]
        assert body["remaining"]["rooms"] == [{"room": "Tavern", "messages": 2}]

    def test_a_room_deleted_from_the_config_is_still_wiped(
        self, client, personas_root, persistence_root, monkeypatch
    ):
        # Transcripts outlive the rooms they belong to, and a wipe that
        # trusted chatrooms.yaml would leave exactly the conversations
        # nobody can see any more.
        _add_exchange("Ghost", "still here", "apparently")
        monkeypatch.setattr(app_config, "_chatrooms_cache",
                            ChatRoomsConfig(chat_rooms=[]))

        body = client.post("/api/session/wipe", json={"rooms": "all"}).json()

        assert body["rooms_cleared"] == ["Ghost"]

    def test_memories_go_when_asked(self, client, personas_root):
        _remember(personas_root, "Luna", "The user told me they sail.")
        _remember(personas_root, "Alex", "The user told me they cycle.")

        body = client.post("/api/session/wipe", json={"memories": True}).json()

        assert sorted(body["memories_cleared"]) == ["Alex", "Luna"]
        assert body["remaining"]["personas"] == []
        assert not (personas_root / "Luna" / persona_store.MEMORIES_FILENAME).exists()

    def test_a_persona_with_no_memories_is_not_reported_as_cleared(
        self, client, personas_root
    ):
        _remember(personas_root, "Luna", "The user told me they sail.")
        body = client.post("/api/session/wipe", json={"memories": True}).json()
        assert body["memories_cleared"] == ["Luna"]

    def test_memories_survive_a_rooms_only_wipe(self, client, personas_root):
        # The distinction that makes this feature necessary: memories are
        # the context that outlives every other clearing action.
        _remember(personas_root, "Luna", "The user told me they sail.")
        body = client.post("/api/session/wipe", json={"rooms": "all"}).json()
        assert body["remaining"]["personas"] == [{"persona": "Luna", "memories": 1}]

    def test_the_adopted_player_can_be_cleared(self, client, personas_root, monkeypatch):
        monkeypatch.setattr(app_config, "_player_cache", PlayerConfig(persona_name="Luna"))

        body = client.post("/api/session/wipe", json={"playing_as": True}).json()

        assert body["playing_as_cleared"] is True
        assert body["remaining"]["playing_as"] == ""

    def test_clearing_nobody_is_not_reported_as_a_change(self, client, personas_root):
        body = client.post("/api/session/wipe", json={"playing_as": True}).json()
        assert body["playing_as_cleared"] is False

    def test_the_cast_and_the_rooms_themselves_survive(
        self, client, personas_root, persistence_root
    ):
        # A wipe that deleted the cast would be a reset button; this is a
        # way to test the cast.
        _add_exchange("TNG", "hello", "hi there")
        _remember(personas_root, "Luna", "The user told me they sail.")

        client.post("/api/session/wipe",
                    json={"rooms": "all", "memories": True, "playing_as": True})

        assert [p["name"] for p in client.get("/api/personas").json()] == ["Alex", "Luna"]
        assert (personas_root / "Luna" / "prompt.md").exists()
        assert (persistence_root / "TNG").is_dir()

    def test_what_is_left_is_read_from_disk_not_predicted(
        self, client, personas_root, persistence_root
    ):
        # The whole point of the round trip: "I think I cleared it" is
        # what this exists to replace.
        _add_exchange("TNG", "hello", "hi there")
        (persistence_root / "Later").mkdir()

        body = client.post("/api/session/wipe", json={"rooms": "current"}).json()

        assert body["rooms_cleared"] == ["TNG"]
        # "Later" is an empty directory, so it is not context; TNG has
        # just been emptied, so it is not either.
        assert body["remaining"]["rooms"] == []
