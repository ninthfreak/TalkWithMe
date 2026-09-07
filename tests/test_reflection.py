"""Tests for app/services/reflection.py — writing memories without a tool.

The question this module answers: whether two characters have met is a
fact the app observes, so it writes met.txt itself; what is worth
remembering about somebody is a judgement, so it has to ask the model.
Asking does not require a *tool call*, though, and that distinction is
what every test here is about — the LLM is stubbed at chat_completion, so
nothing needs a backend and nothing needs allow_tool_calls.
"""

import asyncio

import pytest

from app.config import GeneralConfig, Persona
from app.models import ChatMessage
from app.services import persona_store, reflection
from app.services.persona_store import Memory
from tests.factories import make_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persona(tmp_path, name="Alex", **kwargs) -> Persona:
    persona_dir = tmp_path / name
    persona_dir.mkdir(parents=True, exist_ok=True)
    return Persona(
        name=name, system_prompt=f"You are {name}.", persona_dir=persona_dir, **kwargs,
    )


def _history():
    """A short conversation: the human speaks, two personas answer."""
    return [
        ChatMessage(role="user", content="I have never been on a boat."),
        ChatMessage(role="assistant", content="Not once?", persona="Alex"),
        ChatMessage(role="assistant", content="Hm.", persona="Marv"),
    ]


def _stub_llm(monkeypatch, answer, capture=None):
    """Replace the one LLM call reflection makes."""
    async def fake(messages, **kwargs):
        if capture is not None:
            capture.append((messages, kwargs))
        return answer
    monkeypatch.setattr(reflection, "chat_completion", fake)


def _memories(persona) -> str:
    return persona_store.read_memories(persona.persona_dir)


def _run(coro):
    """Drive one coroutine to completion.

    The project's convention (see tests/test_llm.py): no pytest-asyncio,
    because the suite has to run with nothing but Python installed.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _reflect(*args, **kwargs):
    return _run(reflection.reflect(*args, **kwargs))


def _reflect_all(*args, **kwargs):
    return _run(reflection.reflect_on_conversation(*args, **kwargs))


# ---------------------------------------------------------------------------
# Rendering the finished conversation
# ---------------------------------------------------------------------------

class TestRenderConversation:
    def test_every_line_carries_a_speaker(self):
        # Including the human's. An untagged line in a script has no
        # speaker, and the model is being asked who said what.
        rendered = reflection.render_conversation(_history(), "Tony")
        assert rendered == (
            "[Tony]: I have never been on a boat.\n"
            "[Alex]: Not once?\n"
            "[Marv]: Hm."
        )

    def test_the_human_is_whoever_they_are_playing(self):
        rendered = reflection.render_conversation(_history(), "Kira")
        assert rendered.startswith("[Kira]: I have never been on a boat.")

    def test_empty_messages_are_dropped(self):
        rendered = reflection.render_conversation(
            [ChatMessage(role="user", content="   "),
             ChatMessage(role="assistant", content="Hi.", persona="Alex")], "Tony",
        )
        assert rendered == "[Alex]: Hi."


class TestSpokeIn:
    def test_only_personas_with_a_turn(self):
        assert reflection.spoke_in(_history()) == ["Alex", "Marv"]

    def test_each_persona_once_in_first_seen_order(self):
        history = _history() + [
            ChatMessage(role="assistant", content="Again.", persona="Alex"),
        ]
        assert reflection.spoke_in(history) == ["Alex", "Marv"]

    def test_a_silent_room_produces_nobody(self):
        assert reflection.spoke_in(
            [ChatMessage(role="user", content="Anyone?")]
        ) == []


# ---------------------------------------------------------------------------
# Parsing what the model said
# ---------------------------------------------------------------------------

class TestParseReflection:
    def test_tagged_lines_about_people_present(self):
        saved, skipped = reflection.parse_reflection(
            "[Tony] Tony has never been on a boat.\n[Marv] Marv sighs at everything.",
            "Alex", ["Tony", "Marv"],
        )
        assert saved == [
            Memory("Tony", "Tony has never been on a boat."),
            Memory("Marv", "Marv sighs at everything."),
        ]
        assert skipped == []

    def test_somebody_who_was_not_there_is_dropped(self):
        # Injection only shows the people present, so a memory about
        # somebody absent is unreachable — it would sit in the file
        # forever, eating budget, and never be read.
        saved, skipped = reflection.parse_reflection(
            "[Ghost] Ghost was never here.", "Alex", ["Tony"],
        )
        assert saved == []
        assert skipped == ["[Ghost] Ghost was never here."]

    def test_a_memory_about_itself_is_dropped(self):
        # The original trait-bleed bug in slow motion: a note about what
        # somebody else is like, stored unattributed, comes back in every
        # later conversation as though it were true of the persona
        # holding it.
        saved, skipped = reflection.parse_reflection(
            "[Alex] Alex is a friendly assistant.", "Alex", ["Tony", "Alex"],
        )
        assert saved == []
        assert skipped == ["[Alex] Alex is a friendly assistant."]

    def test_the_subject_is_spelled_the_way_the_room_spells_it(self):
        # The subject is looked up in the memories file by name, so a
        # model answering "[tony]" must not create a second Tony.
        saved, _ = reflection.parse_reflection(
            "[tony] tony sails.", "Alex", ["Tony"],
        )
        assert saved == [Memory("Tony", "tony sails.")]

    @pytest.mark.parametrize("answer", [
        "nothing", "Nothing.", "none", "N/A", "  nothing at all", "no memories",
    ])
    def test_nothing_is_an_ordinary_answer(self, answer):
        # And has to be. Most conversations teach nobody anything, and a
        # model that believes it must produce a line will invent one.
        saved, skipped = reflection.parse_reflection(answer, "Alex", ["Tony"])
        assert saved == []
        assert skipped == []

    def test_an_assumption_is_kept_as_one(self):
        # The marker is understood in the answer exactly as it is on disk,
        # so a persona that worked something out can be told later that it
        # worked it out.
        saved, skipped = reflection.parse_reflection(
            "[Tony] (assumed) Tony is about forty.", "Alex", ["Tony"],
        )
        assert saved == [Memory("Tony", "Tony is about forty.", assumed=True)]
        assert skipped == []

    def test_an_unmarked_line_is_taken_as_known(self):
        saved, _ = reflection.parse_reflection(
            "[Tony] Tony is 43.", "Alex", ["Tony"],
        )
        assert saved[0].assumed is False

    def test_an_untagged_line_is_skipped_not_guessed_at(self):
        saved, skipped = reflection.parse_reflection(
            "Tony seems nice.", "Alex", ["Tony"],
        )
        assert saved == []
        assert skipped == ["Tony seems nice."]


# ---------------------------------------------------------------------------
# The pass itself
# ---------------------------------------------------------------------------

class TestReflect:
    def test_a_memory_is_filed_without_any_tool_call(self, tmp_path, monkeypatch):
        # The whole point: allow_tool_calls is False (the default), no
        # tools are offered anywhere, and memories.txt still gets written.
        persona = _persona(tmp_path)
        assert persona.allow_tool_calls is False
        _stub_llm(monkeypatch, "[Tony] Tony has never been on a boat.")

        result = _reflect(
            persona, ["Tony"], _history(), make_settings(), "Tony",
        )

        assert result.saved == ["[Tony] Tony has never been on a boat."]
        assert _memories(persona) == "[Tony] Tony has never been on a boat.\n"

    def test_it_asks_about_the_people_who_were_there(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, "nothing", capture=captured)

        _reflect(
            _persona(tmp_path), ["Tony", "Marv"], _history(), make_settings(), "Tony",
        )

        prompt = captured[0][0][-1]["content"]
        assert "[Tony]: I have never been on a boat." in prompt   # the conversation
        assert "Tony, Marv" in prompt                              # who to write about
        assert "nothing" in prompt                                 # the way out

    def test_the_persona_is_never_in_its_own_cast(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, "nothing", capture=captured)

        _reflect(
            _persona(tmp_path), ["Alex", "Tony"], _history(), make_settings(), "Tony",
        )

        assert "Write only about these people: Tony" in captured[0][0][-1]["content"]

    # -- knowing what it already knows ---------------------------------------

    def test_it_is_shown_what_it_already_knows(self, tmp_path, monkeypatch):
        # The cause of a memories file filling with the same fact: the
        # question used to be asked in ignorance every single time, so the
        # persona re-derived what it had already recorded and filed it
        # again in slightly different words — which exact-match dedup
        # cannot catch.
        persona = _persona(tmp_path)
        (persona.persona_dir / "memories.txt").write_text(
            "[Tony] Tony is 43.\n[Tony] (assumed) Tony dislikes his job.\n"
        )
        captured = []
        _stub_llm(monkeypatch, "nothing", capture=captured)

        _reflect(persona, ["Tony"], _history(), make_settings(), "Tony")

        prompt = captured[0][0][-1]["content"]
        assert "[Tony] Tony is 43." in prompt
        assert "[Tony] (assumed) Tony dislikes his job." in prompt
        assert "only what is NEW" in prompt

    def test_memories_about_people_who_are_not_here_are_not_listed(
        self, tmp_path, monkeypatch,
    ):
        # Every line spent listing somebody absent is prompt paid for
        # nothing: this conversation cannot restate it.
        persona = _persona(tmp_path)
        (persona.persona_dir / "memories.txt").write_text(
            "[Tony] Tony is 43.\n[Ghost] Ghost is elsewhere.\n"
        )
        captured = []
        _stub_llm(monkeypatch, "nothing", capture=captured)

        _reflect(persona, ["Tony"], _history(), make_settings(), "Tony")

        prompt = captured[0][0][-1]["content"]
        assert "Tony is 43." in prompt
        assert "Ghost" not in prompt

    def test_with_nothing_saved_it_is_told_so(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, "nothing", capture=captured)

        _reflect(_persona(tmp_path), ["Tony"], _history(), make_settings(), "Tony")

        assert "nothing saved about them yet" in captured[0][0][-1]["content"]

    # -- assumptions ----------------------------------------------------------

    def test_an_assumption_is_stored_marked(self, tmp_path, monkeypatch):
        persona = _persona(tmp_path)
        _stub_llm(monkeypatch, "[Tony] (assumed) Tony is about forty.")

        result = _reflect(persona, ["Tony"], _history(), make_settings(), "Tony")

        assert _memories(persona) == "[Tony] (assumed) Tony is about forty.\n"
        assert result.saved == ["[Tony] (assumed) Tony is about forty."]

    def test_the_prompt_shows_how_to_mark_one(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, "nothing", capture=captured)

        _reflect(_persona(tmp_path), ["Tony"], _history(), make_settings(), "Tony")

        prompt = captured[0][0][-1]["content"]
        assert "worked something out rather than being told" in prompt
        assert "[Tony] (assumed) Tony is about forty." in prompt

    def test_no_more_than_the_cap_is_filed(self, tmp_path, monkeypatch):
        # A model answering a "what did you learn" question with a dozen
        # lines has started narrating the conversation back.
        persona = _persona(tmp_path)
        _stub_llm(monkeypatch, "\n".join(
            f"[Tony] Fact number {i}." for i in range(10)
        ))

        result = _reflect(
            persona, ["Tony"], _history(), make_settings(), "Tony",
        )

        assert len(result.saved) == reflection.MAX_MEMORIES_PER_REFLECTION
        assert len(_memories(persona).splitlines()) == (
            reflection.MAX_MEMORIES_PER_REFLECTION
        )

    def test_reflecting_twice_does_not_duplicate(self, tmp_path, monkeypatch):
        # Reflection re-reads the same conversation, so it re-derives the
        # same facts by design. Without dedup a persona's whole budget
        # fills with one thing it knows.
        persona = _persona(tmp_path)
        _stub_llm(monkeypatch, "[Tony] Tony has never been on a boat.")

        first = _reflect(
            persona, ["Tony"], _history(), make_settings(), "Tony")
        second = _reflect(
            persona, ["Tony"], _history(), make_settings(), "Tony")

        assert first.saved and second.saved == []
        assert _memories(persona) == "[Tony] Tony has never been on a boat.\n"

    def test_a_failing_llm_costs_memories_and_nothing_else(
        self, tmp_path, monkeypatch,
    ):
        # This runs from "New Chat" and from changing rooms. Losing a
        # conversation's memories is a disappointment; taking out the
        # action that triggered it is a bug.
        async def boom(*a, **k):
            raise RuntimeError("backend down")
        monkeypatch.setattr(reflection, "chat_completion", boom)

        result = _reflect(
            _persona(tmp_path), ["Tony"], _history(), make_settings(), "Tony",
        )

        assert result.saved == []

    def test_an_empty_answer_files_nothing(self, tmp_path, monkeypatch):
        persona = _persona(tmp_path)
        _stub_llm(monkeypatch, "")

        assert (_reflect(
            persona, ["Tony"], _history(), make_settings(), "Tony")).saved == []
        assert not (persona.persona_dir / "memories.txt").exists()

    def test_a_zero_budget_persona_is_not_even_asked(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, "[Tony] Tony sails.", capture=captured)

        _reflect(
            _persona(tmp_path, memory_size=0), ["Tony"], _history(),
            make_settings(), "Tony",
        )

        assert captured == []   # no completion spent on a persona that cannot save

    def test_the_global_switch_stops_it(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, "[Tony] Tony sails.", capture=captured)
        settings = make_settings(
            general=GeneralConfig(enable_persona_memories=False))

        _reflect(
            _persona(tmp_path), ["Tony"], _history(), settings, "Tony")

        assert captured == []

    def test_an_empty_conversation_is_not_reflected_on(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, "nothing", capture=captured)

        _reflect(
            _persona(tmp_path), ["Tony"], [], make_settings(), "Tony")

        assert captured == []


# ---------------------------------------------------------------------------
# The whole room
# ---------------------------------------------------------------------------

class TestReflectOnConversation:
    def test_everyone_who_spoke_looks_back(self, tmp_path, monkeypatch):
        alex = _persona(tmp_path, "Alex")
        marv = _persona(tmp_path, "Marv")
        _stub_llm(monkeypatch, "[Tony] Tony has never been on a boat.")

        results = _reflect_all(
            _history(), [alex, marv], make_settings(), "Tony", room="TNG",
        )

        assert [r.persona for r in results] == ["Alex", "Marv"]
        assert _memories(alex) == _memories(marv) == (
            "[Tony] Tony has never been on a boat.\n"
        )

    def test_a_persona_who_said_nothing_is_not_asked(self, tmp_path, monkeypatch):
        # It has nothing to look back on, and asking costs a whole
        # completion to be told "nothing" — or to be told something
        # invented instead.
        alex = _persona(tmp_path, "Alex")
        luna = _persona(tmp_path, "Luna")   # in the room, never spoke
        _stub_llm(monkeypatch, "nothing")

        results = _reflect_all(
            _history(), [alex, luna], make_settings(), "Tony",
        )

        assert [r.persona for r in results] == ["Alex"]

    def test_personas_learn_about_each_other_and_the_human(
        self, tmp_path, monkeypatch,
    ):
        captured = []
        alex = _persona(tmp_path, "Alex")
        marv = _persona(tmp_path, "Marv")
        _stub_llm(monkeypatch, "nothing", capture=captured)

        _reflect_all(
            _history(), [alex, marv], make_settings(), "Tony",
        )

        # Alex is asked about Marv and Tony; Marv about Alex and Tony.
        assert "Write only about these people: Marv, Tony" in captured[0][0][-1]["content"]
        assert "Write only about these people: Alex, Tony" in captured[1][0][-1]["content"]

    def test_a_speaker_who_no_longer_exists_is_skipped(self, tmp_path, monkeypatch):
        # Deleted or renamed between speaking and the conversation ending.
        _stub_llm(monkeypatch, "nothing")

        results = _reflect_all(
            _history(), [_persona(tmp_path, "Alex")], make_settings(), "Tony",
        )

        assert [r.persona for r in results] == ["Alex"]
