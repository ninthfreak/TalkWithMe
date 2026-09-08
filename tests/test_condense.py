"""Tests for app/services/condense.py — the hand-run memory tidy-up.

This is the one part of the memory system that *rewrites* rather than
appending, so most of what is tested here is what it refuses to do: the
guards that make a suspect rewrite fall back to what is already on disk.
The LLM is stubbed, so nothing here needs a backend.
"""

import asyncio

import pytest

from app.config import Persona
from app.services import condense, persona_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persona(tmp_path, *lines, name="Alex") -> Persona:
    persona_dir = tmp_path / name
    persona_dir.mkdir(parents=True, exist_ok=True)
    if lines:
        (persona_dir / "memories.txt").write_text("\n".join(lines) + "\n")
    return Persona(name=name, system_prompt=f"You are {name}.", persona_dir=persona_dir)


def _stub_llm(monkeypatch, answer, capture=None):
    async def fake(messages, **kwargs):
        if capture is not None:
            capture.append(messages)
        return answer
    monkeypatch.setattr(condense, "chat_completion", fake)


def _plan(persona):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(condense.plan(persona))
    finally:
        loop.close()


def _memories(persona):
    return persona_store.read_memories(persona.persona_dir).splitlines()


THREE_ABOUT_BRAD = (
    "[Brad] Brad is 43 years old.",
    "[Brad] Brad is a banker.",
    "[Brad] Brad is a tall, handsome man.",
)
MERGED = "[Brad] Brad is a tall, handsome, 43 year old man. He is a banker."


# ---------------------------------------------------------------------------
# Proposing
# ---------------------------------------------------------------------------

class TestPlan:
    def test_three_notes_become_one(self, tmp_path, monkeypatch):
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)
        _stub_llm(monkeypatch, MERGED)

        plan = _plan(persona)

        assert plan.before == list(THREE_ABOUT_BRAD)
        assert plan.after == [MERGED]
        assert plan.saved_bytes > 0
        assert plan.worth_applying

    def test_it_writes_nothing(self, tmp_path, monkeypatch):
        # The whole point of the two-step: this rewrites sentences the
        # persona will act on, and there is no undo.
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)
        _stub_llm(monkeypatch, MERGED)

        _plan(persona)

        assert _memories(persona) == list(THREE_ABOUT_BRAD)

    def test_exact_duplicates_go_before_the_model_sees_them(
        self, tmp_path, monkeypatch,
    ):
        # A plain function can delete those for free; spending the model's
        # attention on them would be waste on top of waste.
        persona = _persona(tmp_path, THREE_ABOUT_BRAD[0], THREE_ABOUT_BRAD[0],
                           THREE_ABOUT_BRAD[1])
        captured = []
        _stub_llm(monkeypatch, MERGED, capture=captured)

        plan = _plan(persona)

        assert plan.duplicates_removed == 1
        # plan.before is the listing the model was handed. (Counting
        # occurrences in the whole prompt would also match the worked
        # example it contains.)
        assert plan.before == [THREE_ABOUT_BRAD[0], THREE_ABOUT_BRAD[1]]
        assert captured, "the model should still have been asked"

    def test_a_single_note_has_nothing_to_merge(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, MERGED, capture=captured)

        plan = _plan(_persona(tmp_path, "[Brad] Brad is 43 years old."))

        assert captured == []          # no completion spent
        assert plan.note == "Nothing to merge."
        assert not plan.worth_applying


class TestPlanRefuses:
    """A suspect rewrite loses information silently, so each of these
    keeps what is already on disk instead."""

    def test_a_person_who_was_not_in_the_notes(self, tmp_path, monkeypatch):
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)
        _stub_llm(monkeypatch, f"{MERGED}\n[Cora] Cora is Brad's wife.")

        plan = _plan(persona)

        assert plan.after == list(THREE_ABOUT_BRAD)
        assert "unusable" in plan.note

    def test_a_line_with_no_owner(self, tmp_path, monkeypatch):
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)
        _stub_llm(monkeypatch, "Brad is a tall banker of 43.")

        plan = _plan(persona)

        assert plan.after == list(THREE_ABOUT_BRAD)

    def test_a_rewrite_that_is_longer(self, tmp_path, monkeypatch):
        # The shape a model takes when it starts embroidering rather than
        # merging: more words, no more information.
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)
        _stub_llm(monkeypatch, "\n".join(
            f"[Brad] Brad, who is quite a character, {n}" for n in "abcd"
        ))

        plan = _plan(persona)

        assert plan.after == list(THREE_ABOUT_BRAD)
        assert "longer" in plan.note

    def test_an_empty_answer(self, tmp_path, monkeypatch):
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)
        _stub_llm(monkeypatch, "")

        assert _plan(persona).after == list(THREE_ABOUT_BRAD)

    def test_an_unreachable_model(self, tmp_path, monkeypatch):
        async def boom(*a, **k):
            raise RuntimeError("backend down")
        monkeypatch.setattr(condense, "chat_completion", boom)
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)

        plan = _plan(persona)

        assert plan.after == list(THREE_ABOUT_BRAD)
        assert "could not be reached" in plan.note


# ---------------------------------------------------------------------------
# Guesses
# ---------------------------------------------------------------------------

class TestAssumptions:
    def test_the_marking_survives_a_rewrite(self, tmp_path, monkeypatch):
        persona = _persona(
            tmp_path,
            "[Brad] (assumed) Brad is unhappy at work.",
            "[Brad] (assumed) Brad dislikes his boss.",
        )
        _stub_llm(monkeypatch, "[Brad] (assumed) Brad is unhappy at work "
                               "and dislikes his boss.")

        plan = _plan(persona)

        assert persona_store.parse_memory_line(plan.after[0]).assumed is True

    def test_a_guess_a_fact_has_settled_can_be_dropped(self, tmp_path, monkeypatch):
        # Carrying both wastes the space this pass exists to reclaim, and
        # leaves the persona holding a belief it has been corrected on.
        # This is the only place that sees guesses and facts side by side.
        persona = _persona(
            tmp_path,
            "[Brad] (assumed) Brad is about fifty.",
            "[Brad] Brad is 43 years old.",
        )
        _stub_llm(monkeypatch, "[Brad] Brad is 43 years old.")

        plan = _plan(persona)

        assert plan.after == ["[Brad] Brad is 43 years old."]
        assert plan.worth_applying

    def test_the_prompt_says_which_way_it_goes(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, MERGED, capture=captured)

        _plan(_persona(tmp_path, *THREE_ABOUT_BRAD))

        asked = captured[0][-1]["content"]
        assert "never merge a guess into a line of facts" in asked
        assert "where a fact settles a guess" in asked


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

class TestApply:
    def test_it_writes_exactly_what_it_was_given(self, tmp_path):
        # Not a second generation: what gets saved is what was displayed.
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)

        assert condense.apply(persona, [MERGED]) == 1
        assert _memories(persona) == [MERGED]

    def test_blank_lines_are_dropped(self, tmp_path):
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)

        condense.apply(persona, [MERGED, "", "   "])

        assert _memories(persona) == [MERGED]

    def test_a_persona_with_no_directory_raises(self, tmp_path):
        # Explicitly asked for, so a silent no-op would leave the user
        # believing the file had changed.
        persona = Persona(name="Alex", system_prompt="You are Alex.")

        with pytest.raises(OSError):
            condense.apply(persona, [MERGED])


# ---------------------------------------------------------------------------
# Never trading a fact for a paraphrase of itself
# ---------------------------------------------------------------------------

class TestSpecifics:
    """The hard details in a memory: every number, and every proper noun.

    The part of "keep every fact" that can be checked without a model.
    """

    def test_numbers_and_names(self):
        assert condense.specifics(
            "Brad is 43 years old. He works at Barclays in Leeds since 1998."
        ) == {"43", "barclays", "leeds", "1998"}

    def test_sentence_starts_are_not_names(self):
        # "Volvo" opening a sentence is ambiguous; a word capitalised
        # mid-sentence is not.
        assert condense.specifics("Volvo driver. Drives a Volvo.") == {"volvo"}
        assert condense.specifics("Volvo driver.") == set()

    def test_grammatical_capitals_are_not_names(self):
        assert condense.specifics("Brad sails, and I think He does too.") == set()

    def test_comparison_is_case_insensitive(self):
        assert condense.lost_details(
            ["Brad works at Barclays."], ["Brad works at BARCLAYS."]
        ) == set()

    def test_a_repeated_word_does_not_confuse_sentence_detection(self):
        # text.find() would have located the first "Brad" and mis-read
        # the second as a sentence start.
        assert "leeds" in condense.specifics("Brad sails. Cora and Brad met in Leeds.")

    def test_the_reported_loss(self):
        # The complaint: a specific age gone in the rewrite.
        assert condense.lost_details(
            ["Brad is 43 years old.", "Brad is a banker."],
            ["Brad is a middle-aged banker."],
        ) == {"43"}

    def test_a_faithful_merge_loses_nothing(self):
        assert condense.lost_details(
            ["Brad is 43 years old.", "Brad is a banker."],
            ["Brad is a tall, handsome, 43 year old man. He is a banker."],
        ) == set()


class TestFactsAreProtected:
    """A subject whose rewrite drops a hard detail keeps their notes as
    written, and the preview says which detail would have gone."""

    def test_a_lost_age_keeps_the_original_lines(self, tmp_path, monkeypatch):
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD)
        _stub_llm(monkeypatch, "[Brad] Brad is a tall, handsome, middle-aged banker.")

        plan = _plan(persona)

        assert plan.after == list(THREE_ABOUT_BRAD)
        assert plan.protected == [
            "Brad: kept as written — the rewrite would have lost 43"
        ]
        assert not plan.worth_applying

    def test_one_bad_merge_does_not_block_a_good_one(self, tmp_path, monkeypatch):
        # Per subject: Brad's rewrite drops his age, Cora's is fine.
        persona = _persona(
            tmp_path, *THREE_ABOUT_BRAD,
            "[Cora] Cora paints.", "[Cora] Cora has a dog called Pip.",
        )
        _stub_llm(monkeypatch,
                  "[Brad] Brad is a middle-aged banker.\n"
                  "[Cora] Cora paints, and has a dog called Pip.")

        plan = _plan(persona)

        assert plan.after == [
            *THREE_ABOUT_BRAD,
            "[Cora] Cora paints, and has a dog called Pip.",
        ]
        assert len(plan.protected) == 1 and plan.protected[0].startswith("Brad:")
        assert plan.worth_applying

    def test_a_lost_name_is_caught_too(self, tmp_path, monkeypatch):
        persona = _persona(tmp_path, "[Cora] Cora paints.", "[Cora] Cora has a dog called Pip.")
        _stub_llm(monkeypatch, "[Cora] Cora paints and has a dog.")

        plan = _plan(persona)

        assert plan.protected == ["Cora: kept as written — the rewrite would have lost pip"]

    def test_somebody_the_rewrite_forgot_keeps_their_notes(self, tmp_path, monkeypatch):
        # Silence is not a merge.
        persona = _persona(tmp_path, *THREE_ABOUT_BRAD, "[Cora] Cora paints.")
        _stub_llm(monkeypatch, MERGED)     # says nothing about Cora at all

        plan = _plan(persona)

        assert plan.after == [MERGED, "[Cora] Cora paints."]
        assert plan.protected == []

    def test_a_culled_guess_may_take_its_number_with_it(self, tmp_path, monkeypatch):
        # Guesses are the one permitted loss: the standard is that a FACT
        # is never traded for a paraphrase. "(assumed) about fifty" going
        # when "43 years old" is known is the cull working, not a leak.
        persona = _persona(
            tmp_path,
            "[Brad] (assumed) Brad is about 50.",
            "[Brad] Brad is 43 years old.",
        )
        _stub_llm(monkeypatch, "[Brad] Brad is 43 years old.")

        plan = _plan(persona)

        assert plan.after == ["[Brad] Brad is 43 years old."]
        assert plan.protected == []

    def test_the_prompt_says_numbers_stay_as_written(self, tmp_path, monkeypatch):
        captured = []
        _stub_llm(monkeypatch, MERGED, capture=captured)

        _plan(_persona(tmp_path, *THREE_ABOUT_BRAD))

        assert "43 stays 43" in captured[0][-1]["content"]
