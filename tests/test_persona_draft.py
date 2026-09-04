"""Unit tests for app/services/persona_draft.py.

The parser is the risky half: it consumes whatever a local model felt like
emitting. Every "sloppy output" case below is a shape models actually
produce — a code fence, a chatty preamble, a missing label, a quoted
value, an invented enum member.
"""

import pytest

from app.config import LengthBias, Persona
from app.services import persona_draft
from app.services.persona_draft import (
    PersonaDraft,
    build_draft_prompt,
    critique,
    parse_draft,
)

WELL_FORMED = """NAME: Rennick
DESCRIPTION: A suspicious harbourmaster
ROUTER_HINTS: boats, cargo, the harbour
LENGTH_BIAS: shorter
AVATAR_COLOR: #2E7D32
CONTRAST: Where Luna reaches for metaphor, Rennick asks for paperwork.
NOTES:
- Stance: he answers questions with questions about provenance.
- Negative space: never speculates about anything he has not seen logged.
SYSTEM_PROMPT:
You run the harbour and you assume everyone is smuggling something. You
answer a question with a question about where the goods came from.
"""


class TestParseWellFormed:
    def test_every_field_is_read(self):
        draft = parse_draft(WELL_FORMED)
        assert draft.name == "Rennick"
        assert draft.description == "A suspicious harbourmaster"
        assert draft.router_hints == "boats, cargo, the harbour"
        assert draft.length_bias is LengthBias.SHORTER
        assert draft.avatar_color == "#2E7D32"
        assert draft.contrast.startswith("Where Luna reaches for metaphor")
        assert len(draft.notes) == 2
        assert draft.system_prompt.startswith("You run the harbour")
        assert draft.is_usable()

    def test_the_prompt_keeps_its_line_breaks(self):
        # It is prose destined for a textarea, not a one-liner.
        assert "\n" in parse_draft(WELL_FORMED).system_prompt


class TestParseSloppyOutput:
    """Shapes local models actually emit."""

    def test_a_code_fence_is_ignored(self):
        draft = parse_draft("```\n" + WELL_FORMED + "```\n")
        assert draft.name == "Rennick"
        assert draft.system_prompt.startswith("You run the harbour")

    def test_a_chatty_preamble_is_skipped(self):
        draft = parse_draft("Sure! Here is your character:\n\n" + WELL_FORMED)
        assert draft.name == "Rennick"

    def test_lowercase_labels_are_accepted(self):
        draft = parse_draft("name: Rennick\nsystem_prompt:\nYou run the harbour.")
        assert draft.name == "Rennick"
        assert draft.system_prompt == "You run the harbour."

    def test_quoted_values_are_unquoted(self):
        draft = parse_draft('NAME: "Rennick"\nSYSTEM_PROMPT:\nYou run the harbour.')
        assert draft.name == "Rennick"

    def test_a_missing_label_costs_only_that_field(self):
        draft = parse_draft(
            "NAME: Rennick\nROUTER_HINTS: boats\nSYSTEM_PROMPT:\nYou run the harbour."
        )
        assert draft.name == "Rennick"
        assert draft.description == ""
        assert draft.is_usable()

    def test_an_invented_length_bias_falls_back_to_match(self, caplog):
        with caplog.at_level("INFO"):
            draft = parse_draft(
                "NAME: R\nLENGTH_BIAS: extremely terse\nSYSTEM_PROMPT:\nYou run it."
            )
        assert draft.length_bias is LengthBias.MATCH
        assert "extremely terse" in caplog.text

    @pytest.mark.parametrize("colour", ["blue", "rgb(1,2,3)", "#GGG", "2E7D32"])
    def test_an_unusable_colour_keeps_the_default(self, colour):
        draft = parse_draft(f"NAME: R\nAVATAR_COLOR: {colour}\nSYSTEM_PROMPT:\nYou run it.")
        assert draft.avatar_color == "#4A90D9"

    def test_nothing_usable_is_reported_as_unusable(self):
        assert not parse_draft("I'm sorry, I can't help with that.").is_usable()
        assert not parse_draft("").is_usable()

    def test_notes_lose_their_bullets(self):
        draft = parse_draft("NAME: R\nSYSTEM_PROMPT:\np\nNOTES:\n- one\n* two\n• three")
        # NOTES after SYSTEM_PROMPT still parses; bullets are stripped.
        assert draft.notes == ["one", "two", "three"]


class TestParseEnforcesFieldLimits:
    """The form rejects over-long values; the draft must not produce them."""

    def test_name_is_capped_and_slash_free(self):
        draft = parse_draft("NAME: " + "K" * 40 + "\nSYSTEM_PROMPT:\np")
        assert len(draft.name) <= persona_draft.MAX_NAME

    @pytest.mark.parametrize("bad", ["Har/bour", "Har\\bour"])
    def test_a_slash_in_the_name_is_removed(self, bad):
        # A slash makes the persona unreachable on /api/personas/{name}/...
        draft = parse_draft(f"NAME: {bad}\nSYSTEM_PROMPT:\np")
        assert "/" not in draft.name and "\\" not in draft.name

    def test_description_is_capped(self):
        draft = parse_draft("NAME: R\nDESCRIPTION: " + "d" * 80 + "\nSYSTEM_PROMPT:\np")
        assert len(draft.description) <= persona_draft.MAX_DESCRIPTION

    def test_router_hints_are_capped(self):
        draft = parse_draft("NAME: R\nROUTER_HINTS: " + "h, " * 200 + "\nSYSTEM_PROMPT:\np")
        assert len(draft.router_hints) <= persona_draft.MAX_ROUTER_HINTS


class TestDraftPrompt:
    def test_the_existing_cast_is_included_to_write_against(self):
        cast = [
            Persona(name="Luna", description="A poet", router_hints="x",
                    system_prompt="You speak in metaphor and never answer directly."),
        ]
        system = build_draft_prompt("a harbourmaster", cast)[0]["content"]
        assert "Luna" in system
        assert "You speak in metaphor" in system
        assert "unmistakably NOT one of the others" in system

    def test_an_empty_cast_says_so_rather_than_leaving_a_gap(self):
        system = build_draft_prompt("a harbourmaster", [])[0]["content"]
        assert "no other personas yet" in system

    def test_every_lever_reaches_the_prompt(self):
        system = build_draft_prompt("x", [])[0]["content"]
        for lever in persona_draft.LEVERS:
            assert lever.title in system

    def test_the_anti_patterns_are_named_explicitly(self):
        # The model produces exactly these unless told not to.
        system = build_draft_prompt("x", [])[0]["content"]
        assert "topic lists" in system
        assert "adjective piles" in system

    def test_the_brief_is_the_user_turn(self):
        messages = build_draft_prompt("a suspicious harbourmaster", [])
        assert messages[1]["role"] == "user"
        assert "a suspicious harbourmaster" in messages[1]["content"]

    def test_a_long_cast_prompt_is_trimmed_not_dumped(self):
        # Six personas with 8KB prompts would bury the instructions.
        cast = [Persona(name=f"P{i}", description="d", router_hints="x",
                        system_prompt="word " * 2000) for i in range(6)]
        system = build_draft_prompt("x", cast)[0]["content"]
        assert len(system) < 8000


class TestCritique:
    """Local checks on the failures the model is the wrong judge of."""

    def test_a_short_prompt_is_flagged(self):
        warnings = critique(PersonaDraft(name="R", system_prompt="You run the harbour."))
        assert any("only 4 words" in w for w in warnings)

    def test_generic_assistant_vocabulary_is_flagged(self):
        draft = PersonaDraft(
            name="R",
            system_prompt="You are a helpful and friendly assistant who is curious. " * 8,
        )
        warnings = critique(draft)
        assert any("generic assistant vocabulary" in w for w in warnings)

    def test_missing_negative_space_is_flagged(self):
        draft = PersonaDraft(name="R", system_prompt="You ask about cargo. " * 20)
        assert any("will not do" in w for w in critique(draft))

    def test_negative_space_satisfies_the_check(self):
        draft = PersonaDraft(
            name="R",
            system_prompt="You ask about cargo. " * 20 + "You never speculate.",
        )
        assert not any("will not do" in w for w in critique(draft))

    def test_a_third_person_prompt_is_flagged(self):
        draft = PersonaDraft(
            name="R",
            system_prompt="Rennick runs the harbour and never speculates. " * 8,
        )
        assert any("second person" in w for w in critique(draft))

    def test_a_good_draft_draws_no_warnings(self):
        draft = PersonaDraft(
            name="Rennick",
            system_prompt=(
                "You run the harbour and you assume everyone is smuggling. You answer "
                "a question with a question about provenance. You never speculate "
                "about cargo you have not seen logged, and you say so plainly when "
                "asked. You speak in short declaratives and you do not soften them. "
                "You think Luna is wasting everyone's time with her metaphors, and "
                "you have said as much to her face more than once already."
            ),
        )
        assert critique(draft) == []
