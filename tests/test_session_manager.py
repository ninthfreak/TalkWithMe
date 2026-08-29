"""Tests for app/session.py — the SessionManager and its LLM message building."""

import json

import pytest

from app import persistence
from app.models import ChatMessage
from app.session import SessionManager


@pytest.fixture
def manager() -> SessionManager:
    """A fresh SessionManager (routers use the global singleton; unit tests
    don't need it). The persistence root is already patched to tmp."""
    return SessionManager()


class TestRoomTracking:
    def test_set_current_room_updates_tracker(self, manager):
        assert manager.current_room == "default"
        manager.set_current_room("TNG")
        assert manager.current_room == "TNG"

    def test_set_current_room_same_room_is_noop(self, manager):
        manager.set_current_room("default")
        assert manager.current_room == "default"


class TestActivePersonas:
    def test_set_active_personas_stores_a_copy(self, manager):
        names = ["Alex", "Luna"]
        manager.set_active_personas(names)
        names.append("Malfait")
        assert manager.active_personas == ["Alex", "Luna"]

    def test_active_personas_returns_a_copy(self, manager):
        manager.set_active_personas(["Alex"])
        manager.active_personas.append("Luna")
        assert manager.active_personas == ["Alex"]


class TestMessages:
    def test_add_user_message_persists_to_current_room(self, manager):
        manager.set_current_room("TNG")
        manager.add_user_message("hello", "uid-1")

        msgs = persistence.load_history("TNG")
        assert msgs == [{"id": "uid-1", "sender": "USER", "text": "hello", "audio": []}]
        assert manager.history == [ChatMessage(role="user", content="hello")]

    def test_add_assistant_message_persists_with_persona(self, manager):
        manager.add_assistant_message("why hello", "Luna", "aid-1")
        msgs = persistence.load_history("default")
        assert msgs[0]["sender"] == "Luna"
        assert msgs[0]["id"] == "aid-1"

    def test_history_returns_a_copy(self, manager):
        manager.add_user_message("hi", "uid-1")
        manager.history.clear()
        assert len(manager.history) == 1


class TestBuildLLMMessages:
    def test_build_llm_messages_empty_history_is_system_only(self, manager):
        messages = manager.build_llm_messages("You are Alex.", "Alex")
        assert messages == [{"role": "system", "content": "You are Alex."}]

    def test_build_llm_messages_reformats_roles(self, manager):
        manager.add_user_message_no_persist("what do you think?")
        manager.add_assistant_message_no_persist("I think, therefore I speak.", "Luna")
        manager.add_assistant_message_no_persist("I agree with Luna.", "Alex")
        manager.add_user_message_no_persist("thanks")

        messages = manager.build_llm_messages("You are Alex.", "Alex")

        assert messages[0] == {"role": "system", "content": "You are Alex."}
        # The human is tagged too. Leaving them the one untagged voice is
        # what taught personas to answer *as* the user.
        assert messages[1] == {"role": "user", "content": "[User]: what do you think?"}
        # Another persona's line becomes a user message, prefixed with the name.
        assert messages[2] == {
            "role": "user",
            "content": "[Luna]: I think, therefore I speak.",
        }
        # The responding persona keeps the assistant role, and is the only
        # untagged voice in the payload.
        assert messages[3] == {"role": "assistant", "content": "I agree with Luna."}
        assert messages[4] == {"role": "user", "content": "[User]: thanks"}

    def test_build_llm_messages_max_turns_keeps_only_last_entries(self, manager):
        for i in range(10):
            manager.add_user_message_no_persist(f"turn {i}")

        messages = manager.build_llm_messages(
            "sys", "Alex", max_turns_for_context=4
        )

        # 1 system + last 4 turns
        assert len(messages) == 5
        assert messages[1]["content"] == "[User]: turn 6"
        assert messages[-1]["content"] == "[User]: turn 9"

    def test_build_llm_messages_no_max_turns_keeps_everything(self, manager):
        for i in range(10):
            manager.add_user_message_no_persist(f"turn {i}")
        messages = manager.build_llm_messages("sys", "Alex")
        assert len(messages) == 11


class TestResetAndLoadRoom:
    def test_reset_clears_history_persistence_and_personas(self, manager):
        manager.set_current_room("TNG")
        manager.set_active_personas(["Alex"])
        manager.add_user_message("bye", "uid-1")
        assert persistence.load_history("TNG") != []

        manager.reset()

        assert manager.history == []
        assert manager.active_personas == []
        assert persistence.load_history("TNG") == []

    def test_load_room_populates_history_without_repersisting(self, manager):
        persistence.persist_message(
            "TNG", ChatMessage(role="user", content="old hello"), "uid-1"
        )
        persistence.persist_message(
            "TNG",
            ChatMessage(role="assistant", content="old reply", persona="Luna"),
            "aid-1",
        )

        manager.load_room("TNG")

        assert manager.current_room == "TNG"
        assert manager.history == [
            ChatMessage(role="user", content="old hello"),
            ChatMessage(role="assistant", content="old reply", persona="Luna"),
        ]

    def test_load_room_replaces_existing_history(self, manager):
        manager.add_user_message("fresh", "uid-fresh")
        persistence.persist_message("TNG", ChatMessage(role="user", content="old"), "uid-old")

        manager.load_room("TNG")

        assert [m.content for m in manager.history] == ["old"]

    def test_load_room_missing_room_yields_empty_history(self, manager):
        manager.load_room("never-existed")
        assert manager.history == []
        assert manager.current_room == "never-existed"


class TestGetHistoryDicts:
    def test_get_history_dicts_serializable(self, manager):
        manager.add_user_message("hi", "uid-1")
        manager.add_assistant_message("there", "Alex", "aid-1")

        dicts = manager.get_history_dicts()
        json.dumps(dicts)  # must be JSON-serializable
        assert dicts[0]["role"] == "user"
        assert dicts[1]["persona"] == "Alex"


# ---------------------------------------------------------------------------
# Room preamble and truncation
# ---------------------------------------------------------------------------

class TestRoomPreamble:
    def test_preamble_is_appended_to_the_system_message(self):
        session = SessionManager()
        messages = session.build_llm_messages(
            system_prompt="You are Alex.",
            responding_persona="Alex",
            room_preamble="You are in a room.",
        )
        assert messages[0] == {
            "role": "system",
            "content": "You are Alex.\n\nYou are in a room.",
        }

    def test_system_message_is_unchanged_without_a_preamble(self):
        session = SessionManager()
        messages = session.build_llm_messages(
            system_prompt="You are Alex.", responding_persona="Alex"
        )
        assert messages[0] == {"role": "system", "content": "You are Alex."}


class TestTruncatedRendering:
    def _history_for(self, content, truncated):
        session = SessionManager()
        session.add_assistant_message_no_persist(content, "Luna")
        session._history[-1].truncated = truncated
        return session.build_llm_messages(
            system_prompt="p", responding_persona="Alex"
        )[1]["content"]

    def test_truncated_reply_is_trimmed_to_its_last_full_sentence(self):
        # A dangling fragment is a completion cue — this is exactly how one
        # persona ends up finishing another's sentence in the wrong voice.
        assert self._history_for("One. Two. Three and it stops", True) == (
            "[Luna]: One. Two."
        )

    def test_truncated_reply_with_no_sentence_end_is_labelled(self):
        assert self._history_for("It just stops", True) == (
            "[Luna]: It just stops (message was cut off)"
        )

    def test_untruncated_reply_is_relayed_verbatim(self):
        assert self._history_for("One. Two. Three and it stops", False) == (
            "[Luna]: One. Two. Three and it stops"
        )

    def test_the_speakers_own_truncated_message_is_not_trimmed(self):
        # Trimming only protects *other* personas from a completion cue; a
        # persona seeing its own message needs it intact.
        session = SessionManager()
        session.add_assistant_message_no_persist("One. Two. Three and it stops", "Alex")
        session._history[-1].truncated = True
        messages = session.build_llm_messages(
            system_prompt="p", responding_persona="Alex"
        )
        assert messages[1] == {
            "role": "assistant",
            "content": "One. Two. Three and it stops",
        }

    def test_truncated_flag_is_not_exposed_in_the_session_shape(self):
        session = SessionManager()
        session.add_assistant_message_no_persist("hi", "Alex")
        assert "truncated" not in session.get_history_dicts()[0]


class TestUserLabel:
    """The human is a tagged speaker like everyone else."""

    def test_player_name_is_used_when_given(self, manager):
        manager.add_user_message_no_persist("hello")
        messages = manager.build_llm_messages("sys", "Alex", user_label="Kira")
        assert messages[1] == {"role": "user", "content": "[Kira]: hello"}

    def test_the_responding_persona_is_the_only_untagged_voice(self, manager):
        manager.add_user_message_no_persist("q")
        manager.add_assistant_message_no_persist("a1", "Alex")
        manager.add_assistant_message_no_persist("a2", "Luna")

        messages = manager.build_llm_messages("sys", "Alex", user_label="Kira")

        untagged = [
            m for m in messages[1:] if not m["content"].startswith("[")
        ]
        assert untagged == [{"role": "assistant", "content": "a1"}]

    def test_a_late_responder_still_sees_tagged_voices_only(self, manager):
        # The reported failure: the third persona to speak has no assistant
        # turn of its own in context, so an untagged human line was the only
        # model it had for "what a turn looks like".
        manager.add_user_message_no_persist("what do you all think?")
        manager.add_assistant_message_no_persist("North.", "Alex")
        manager.add_assistant_message_no_persist("Mistake.", "Sig")

        messages = manager.build_llm_messages("sys", "Luna", user_label="Kira")

        assert [m["content"] for m in messages[1:]] == [
            "[Kira]: what do you all think?",
            "[Alex]: North.",
            "[Sig]: Mistake.",
        ]


class TestExchangeWindowing:
    """Context is windowed by exchange, never by raw message count.

    Reported: a room was asked to guess something, six personas guessed,
    and the answer was then revealed — but the personas treated the reveal
    as a statement out of nowhere. A flat tail slice of six *messages* in a
    six-persona room cannot even hold one exchange, so the question that
    prompted the guesses had already fallen out of the window.
    """

    def _guessing_game(self, manager, personas=6):
        manager.add_user_message_no_persist("Guess what animal I'm thinking of.")
        for i in range(personas):
            manager.add_assistant_message_no_persist(f"A badger.", f"P{i}")
        manager.add_user_message_no_persist("It was an otter.")

    def test_the_question_survives_a_wide_room(self, manager):
        self._guessing_game(manager)
        messages = manager.build_llm_messages(
            "sys", "P0", max_turns_for_context=6, user_label="Tony"
        )
        contents = [m["content"] for m in messages]
        assert "[Tony]: Guess what animal I'm thinking of." in contents
        assert "[Tony]: It was an otter." in contents

    def test_one_exchange_of_context_still_holds_every_reply(self, manager):
        # Even at the minimum, an exchange is kept whole.
        self._guessing_game(manager)
        messages = manager.build_llm_messages(
            "sys", "P0", max_turns_for_context=1, user_label="Tony"
        )
        # The last exchange is the reveal, which has no replies yet.
        assert [m["content"] for m in messages[1:]] == ["[Tony]: It was an otter."]

    def test_two_exchanges_keeps_the_question_and_all_its_answers(self, manager):
        self._guessing_game(manager)
        messages = manager.build_llm_messages(
            "sys", "P9", max_turns_for_context=2, user_label="Tony"
        )
        contents = [m["content"] for m in messages[1:]]
        assert contents[0] == "[Tony]: Guess what animal I'm thinking of."
        assert contents[-1] == "[Tony]: It was an otter."
        assert len(contents) == 8   # question + 6 guesses + reveal

    def test_older_exchanges_are_dropped_whole(self, manager):
        for i in range(5):
            manager.add_user_message_no_persist(f"question {i}")
            manager.add_assistant_message_no_persist(f"answer {i}", "P0")

        messages = manager.build_llm_messages(
            "sys", "P1", max_turns_for_context=2, user_label="Tony"
        )
        contents = [m["content"] for m in messages[1:]]
        # Two complete exchanges, oldest first — never half of one.
        assert contents == [
            "[Tony]: question 3", "[P0]: answer 3",
            "[Tony]: question 4", "[P0]: answer 4",
        ]

    def test_window_does_not_shrink_as_personas_are_added(self, manager):
        # The bug in one line: the same setting used to buy less context in
        # a bigger room. It must not.
        for personas in (1, 6):
            m = SessionManager()
            m.add_user_message_no_persist("the question")
            for i in range(personas):
                m.add_assistant_message_no_persist("a reply", f"P{i}")
            m.add_user_message_no_persist("the reveal")
            contents = [
                x["content"]
                for x in m.build_llm_messages("sys", "Z", max_turns_for_context=2,
                                              user_label="Tony")
            ]
            assert "[Tony]: the question" in contents, f"lost with {personas} personas"

    def test_history_shorter_than_the_window_is_kept_entirely(self, manager):
        manager.add_user_message_no_persist("only question")
        manager.add_assistant_message_no_persist("only answer", "P0")
        messages = manager.build_llm_messages("sys", "P1", max_turns_for_context=10)
        assert len(messages) == 3

    def test_no_limit_keeps_everything(self, manager):
        self._guessing_game(manager)
        messages = manager.build_llm_messages("sys", "P0", max_turns_for_context=None)
        assert len(messages) == 9


class TestContextCharBudget:
    """A safety valve, not the main control — but it must shed whole
    exchanges and never the human message anchoring the last one."""

    def test_oldest_exchanges_go_first(self):
        from app.session import recent_exchanges
        from app.models import ChatMessage

        history = []
        for i in range(5):
            history.append(ChatMessage(role="user", content=f"q{i} " + "x" * 500))
            history.append(ChatMessage(role="assistant", content="a" * 500, persona="P"))

        window = recent_exchanges(history, max_exchanges=50, char_budget=2200)

        assert len(window) < len(history)
        assert window[0].role == "user"              # starts on a whole exchange
        assert window[-1].content.startswith("a")    # keeps the newest

    def test_the_last_human_message_is_never_dropped(self):
        from app.session import recent_exchanges
        from app.models import ChatMessage

        history = [ChatMessage(role="user", content="the question " + "x" * 200)]
        history += [
            ChatMessage(role="assistant", content="y" * 400, persona=f"P{i}")
            for i in range(6)
        ]

        window = recent_exchanges(history, max_exchanges=50, char_budget=500)

        assert window[0].role == "user"
        assert window[0].content.startswith("the question")

    def test_a_window_inside_budget_is_untouched(self):
        from app.session import recent_exchanges
        from app.models import ChatMessage

        history = [
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a", persona="P"),
        ]
        assert recent_exchanges(history, max_exchanges=10) == history
