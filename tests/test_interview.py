"""Tests for app/services/interview.py — notes taken during a sitting.

The reflection pass asks "what did you learn" once, at the end, and files
three lines. An interview needs the opposite of all three: facts rather
than impressions, every few exchanges rather than at the end, and into a
store that can outgrow a prompt. These tests are mostly about the seams
where those differ, and about the note pass never being the thing that
breaks a reply.
"""

import asyncio
import logging

import pytest

from app.config import Persona
from app.models import ChatMessage
from app.services import dossier, interview
from app.services.dossier import Note
from tests.factories import make_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persona(tmp_path, name="Marion") -> Persona:
    persona_dir = tmp_path / name
    persona_dir.mkdir(parents=True, exist_ok=True)
    return Persona(
        name=name, system_prompt=f"You are {name}, a biographer.",
        persona_dir=persona_dir,
    )


def _history(*turns):
    """Turns as ("user", text) / ("Marion", text) pairs."""
    out = []
    for who, text in turns:
        if who == "user":
            out.append(ChatMessage(role="user", content=text))
        else:
            out.append(ChatMessage(role="assistant", content=text, persona=who))
    return out


def _sitting(exchanges=2):
    turns = []
    for i in range(exchanges):
        turns.append(("user", f"Answer number {i} about the yard."))
        turns.append(("Marion", f"Question number {i}?"))
    return _history(*turns)


def _stub(monkeypatch, answer, capture=None):
    async def fake(messages, **kwargs):
        if capture is not None:
            capture.append((messages, kwargs))
        return answer
    monkeypatch.setattr(interview, "chat_completion", fake)


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


def _notes(persona, subject="Wes"):
    return dossier.read_notes(persona.persona_dir, subject)


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------

class TestCadence:
    """Notes are taken during the conversation, which is the whole point:
    the reply window is six exchanges, so anything written down only at
    the end was already out of the prompt when it mattered."""

    def test_the_unit_is_the_humans_turn_not_the_message_count(self):
        # A room where three personas answer every message would
        # otherwise take notes three times as often.
        history = _history(
            ("user", "Tell me about the yard."),
            ("Marion", "What year?"), ("Cora", "Go on."), ("Alex", "Mm."),
        )
        assert interview.exchanges_so_far(history) == 1

    def test_a_pass_is_due_on_the_cadence(self):
        assert interview.due_for_notes(_sitting(3), every=3) is True
        assert interview.due_for_notes(_sitting(6), every=3) is True

    def test_no_pass_between_cadences(self):
        assert interview.due_for_notes(_sitting(2), every=3) is False
        assert interview.due_for_notes(_sitting(4), every=3) is False

    def test_an_empty_conversation_is_never_due(self):
        assert interview.due_for_notes([], every=3) is False

    def test_the_window_overlaps_the_last_one(self):
        # A pass that fails is covered by the next, which is why the
        # window is one exchange longer than the cadence.
        assert interview.NOTES_WINDOW_EXCHANGES > interview.NOTES_EVERY_EXCHANGES

    def test_the_window_is_cut_on_the_humans_turns(self):
        # Cutting mid-exchange would hand the pass an answer with no
        # question above it.
        history = _history(
            ("user", "First question."), ("Marion", "First answer."),
            ("user", "Second question."), ("Marion", "Second answer."),
            ("user", "Third question."), ("Marion", "Third answer."),
        )
        window = interview.recent_conversation(history, "Wes", exchanges=2)
        assert window.startswith("[Wes]: Second question.")
        assert "First question" not in window

    def test_a_short_conversation_is_the_whole_window(self):
        window = interview.recent_conversation(_sitting(1), "Wes", exchanges=4)
        assert "Answer number 0" in window


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

class TestNotesPrompt:
    def test_the_goal_is_what_the_pass_is_told_to_look_for(self):
        messages = interview.build_notes_prompt(
            Persona(name="Marion", system_prompt="You are Marion."),
            "Wes", "His working life, 1978 to retirement.", "[Wes]: I started in 78.",
        )
        assert "His working life, 1978 to retirement." in messages[-1]["content"]

    def test_the_existing_topics_are_shown_so_the_model_reuses_them(self):
        # The half of tag consistency no string comparison can do:
        # snap_tag will never merge "job" into "work".
        messages = interview.build_notes_prompt(
            Persona(name="Marion", system_prompt="You are Marion."),
            "Wes", "", "[Wes]: I started in 78.",
            index=[dossier.Topic("work", 12), dossier.Topic("family", 3)],
        )
        content = messages[-1]["content"]
        assert "work, family" in content
        assert "stays together" in content

    def test_the_standing_questions_are_shown_to_be_written_back(self):
        # Nothing here asks a model to delete a line.
        messages = interview.build_notes_prompt(
            Persona(name="Marion", system_prompt="You are Marion."),
            "Wes", "", "[Wes]: I started in 78.",
            open_questions=[Note(subject="Wes", text="Why did he leave?",
                                 open_question=True)],
        )
        content = messages[-1]["content"]
        assert "Why did he leave?" in content
        assert "still waiting on" in content

    def test_nothing_is_offered_as_an_ordinary_answer(self):
        # In an interview, a model that believes it must produce a line
        # invents a fact about somebody's life.
        messages = interview.build_notes_prompt(
            Persona(name="Marion", system_prompt="You are Marion."),
            "Wes", "", "[Wes]: Morning.",
        )
        assert "answer with the single word: nothing" in messages[-1]["content"]

    def test_the_format_asked_for_is_the_format_the_file_stores(self):
        # One representation, not two: what the model writes, the dossier
        # parser reads.
        messages = interview.build_notes_prompt(
            Persona(name="Marion", system_prompt="You are Marion."),
            "Wes", "", "[Wes]: I started in 78.",
        )
        shown = [
            line.strip() for line in messages[-1]["content"].splitlines()
            if line.strip().startswith("[Wes]")
        ]
        assert shown
        for line in shown:
            assert dossier.parse_note_line(line).text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParseNotes:
    def test_facts_and_questions_come_back_apart(self):
        facts, questions, dropped = interview.parse_notes(
            "[Wes] #work @1978 Started at the yard.\n"
            "[Wes] (open) #work Why did he leave?\n",
            "Wes",
        )
        assert [n.text for n in facts] == ["Started at the yard."]
        assert [n.text for n in questions] == ["Why did he leave?"]
        assert dropped == []

    def test_tags_and_eras_survive(self):
        facts, _, _ = interview.parse_notes(
            "[Wes] #work #vickers @1978 Started at the yard.", "Wes")
        assert facts[0].tags == ("work", "vickers") and facts[0].when == "1978"

    def test_a_line_about_somebody_else_is_dropped(self):
        # A dossier is one person's. A fact about their mother is a fact
        # about them, filed under their name.
        facts, _, dropped = interview.parse_notes(
            "[Wes] #family Mother taught at the grammar.\n"
            "[Mother] She taught at the grammar.\n",
            "Wes",
        )
        assert len(facts) == 1 and len(dropped) == 1

    def test_nothing_is_read_as_nothing(self):
        for answer in ("nothing", "Nothing.", "none", "nothing new"):
            facts, questions, _ = interview.parse_notes(answer, "Wes")
            assert facts == [] and questions == []

    def test_list_bullets_are_tolerated(self):
        facts, _, _ = interview.parse_notes(
            "- [Wes] #work Started at the yard.", "Wes")
        assert len(facts) == 1

    def test_a_line_with_only_markers_is_dropped(self):
        facts, _, dropped = interview.parse_notes("[Wes] #work @1978", "Wes")
        assert facts == [] and len(dropped) == 1

    def test_an_untagged_line_is_still_filed_under_the_subject(self):
        facts, _, _ = interview.parse_notes("He hated the cold.", "Wes")
        assert facts[0].subject == "Wes" and facts[0].text == "He hated the cold."


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

class TestTakeNotes:
    def test_notes_are_written_to_the_dossier(self, tmp_path, monkeypatch):
        marion = _persona(tmp_path)
        _stub(monkeypatch, "[Wes] #work @1978 Started at the yard.")

        result = _run(interview.take_notes(
            marion, "Wes", "His working life", _sitting(), make_settings(), "Wes"))

        assert result.filed == ["[Wes] #work @1978 Started at the yard."]
        assert [n.text for n in _notes(marion)] == ["Started at the yard."]

    def test_nothing_is_written_to_memories(self, tmp_path, monkeypatch):
        # The two stores are exclusive: filing into both would put the
        # same fact in a 16 KB file and a file that can hold a life.
        from app.services import persona_store
        marion = _persona(tmp_path)
        _stub(monkeypatch, "[Wes] #work Started at the yard.")

        _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes"))

        assert persona_store.read_memories(marion.persona_dir) == ""

    def test_open_questions_replace_rather_than_accumulate(self, tmp_path, monkeypatch):
        marion = _persona(tmp_path)
        _stub(monkeypatch, "[Wes] (open) Why did he leave?\n"
                           "[Wes] (open) What about 1986?")
        _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes"))

        _stub(monkeypatch, "[Wes] (open) What about 1986?")
        _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes"))

        standing = [n.text for n in _notes(marion) if n.open_question]
        assert standing == ["What about 1986?"]

    def test_a_pass_is_shown_the_topics_already_in_use(self, tmp_path, monkeypatch):
        marion = _persona(tmp_path)
        dossier.append_notes(marion.persona_dir, "Wes", [
            Note(subject="Wes", text="Started at the yard.", tags=("work",))])
        captured = []
        _stub(monkeypatch, "nothing", capture=captured)

        _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes"))

        assert "work" in captured[0][0][-1]["content"]

    def test_the_cap_holds(self, tmp_path, monkeypatch):
        marion = _persona(tmp_path)
        _stub(monkeypatch, "\n".join(
            f"[Wes] #work Fact number {i}." for i in range(40)))

        _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes"))

        assert len(_notes(marion)) == interview.MAX_NOTES_PER_PASS

    def test_the_open_question_cap_holds(self, tmp_path, monkeypatch):
        marion = _persona(tmp_path)
        _stub(monkeypatch, "\n".join(
            f"[Wes] (open) Question {i}?" for i in range(30)))

        _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes"))

        standing = [n for n in _notes(marion) if n.open_question]
        assert len(standing) == interview.MAX_OPEN_QUESTIONS

    def test_a_repeated_fact_is_filed_once(self, tmp_path, monkeypatch):
        # The window overlaps by design, so the same stretch is read twice.
        marion = _persona(tmp_path)
        _stub(monkeypatch, "[Wes] #work Started at the yard.")
        _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes"))
        _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes"))

        assert len(_notes(marion)) == 1

    def test_a_failing_backend_costs_a_stretch_and_nothing_else(
        self, tmp_path, monkeypatch,
    ):
        marion = _persona(tmp_path)

        async def boom(messages, **kwargs):
            raise RuntimeError("backend down")
        monkeypatch.setattr(interview, "chat_completion", boom)

        assert _run(interview.take_notes(
            marion, "Wes", "", _sitting(), make_settings(), "Wes")) is None
        assert _notes(marion) == []

    def test_an_empty_answer_is_reported_at_info(self, tmp_path, monkeypatch, caplog):
        # chat_completion swallows a backend error and returns "", so the
        # log is the only place the difference shows.
        marion = _persona(tmp_path)
        _stub(monkeypatch, "")
        with caplog.at_level(logging.INFO):
            _run(interview.take_notes(
                marion, "Wes", "", _sitting(), make_settings(), "Wes"))
        assert "returned nothing" in caplog.text

    def test_memories_switched_off_stops_the_pass(self, tmp_path, monkeypatch):
        from app.config import GeneralConfig
        marion = _persona(tmp_path)
        _stub(monkeypatch, "[Wes] #work Started at the yard.")
        settings = make_settings()
        settings.general = GeneralConfig(enable_persona_memories=False)

        assert _run(interview.take_notes(
            marion, "Wes", "", _sitting(), settings, "Wes")) is None
        assert _notes(marion) == []

    def test_the_final_pass_reads_the_whole_conversation(self, tmp_path, monkeypatch):
        # The one pass with no successor to recover what it misses.
        marion = _persona(tmp_path)
        captured = []
        _stub(monkeypatch, "nothing", capture=captured)

        _run(interview.take_notes(
            marion, "Wes", "", _sitting(exchanges=9), make_settings(), "Wes",
            whole_conversation=True))

        content = captured[0][0][-1]["content"]
        assert "Answer number 0" in content and "Answer number 8" in content


# ---------------------------------------------------------------------------
# What reaches the reply prompt
# ---------------------------------------------------------------------------

class TestWhatTheInterviewerReads:
    def test_the_preamble_says_what_the_sitting_is_for(self):
        lines = interview.preamble_lines(
            Persona(name="Marion", system_prompt="x"), "Wes",
            "His working life, 1978 to retirement.")
        assert any("This is an interview" in line for line in lines)
        assert any("1978 to retirement" in line for line in lines)

    def test_the_subject_is_capitalised_for_the_sentence_start(self):
        # With nobody adopted the subject is the literal string "the
        # user", which read as "This is an interview. the user is..."
        lines = interview.preamble_lines(
            Persona(name="Marion", system_prompt="x"), "the user", "")
        assert "The user is talking about their life" in lines[0]

    def test_an_empty_goal_adds_no_second_line(self):
        lines = interview.preamble_lines(
            Persona(name="Marion", system_prompt="x"), "Wes", "")
        assert len(lines) == 1

    def test_the_notes_block_is_the_index_plus_what_matches(self, tmp_path):
        marion = _persona(tmp_path)
        dossier.append_notes(marion.persona_dir, "Wes", [
            Note(subject="Wes", text="Started at the yard.", tags=("work",)),
            Note(subject="Wes", text="Father was a fitter.", tags=("family",)),
        ])

        block = interview.notes_block(marion, "Wes", "tell me about your father")

        assert "What you have written down about Wes:" in block
        assert "Father was a fitter." in block
        assert "Started at the yard." not in block

    def test_an_empty_dossier_adds_nothing_to_the_prompt(self, tmp_path):
        assert interview.notes_block(_persona(tmp_path), "Wes", "anything") == ""

    def test_reading_the_notes_tidies_the_index_on_the_way_past(self, tmp_path):
        # reindex runs on the read path, like dedupe_memories.
        marion = _persona(tmp_path)
        path = dossier.notes_path(marion.persona_dir, "Wes")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(f"[Wes] #school Fact {i}." for i in range(5))
            + "\n[Wes] #schools One more.\n"
        )

        interview.notes_block(marion, "Wes", "school")

        assert {t.tag for t in dossier.topics(_notes(marion))} == {"school"}
