"""Unit tests for app/services/persona_draft.py.

The parser is the risky half: it consumes whatever a local model felt like
emitting. Every "sloppy output" case below is a shape models actually
produce — a code fence, a chatty preamble, a missing label, a quoted
value, an invented enum member.
"""

import pytest

from dataclasses import replace

from app.config import LengthBias
from app.services import persona_draft
from app.services.persona_draft import (
    PersonaDraft,
    PersonaSpec,
    build_draft_prompt,
    build_refine_prompt,
    critique,
    parse_draft,
)


def spec(brief="x", **kwargs):
    """A spec built the way the route builds one, so tests exercise the
    sanitising path rather than reaching around it."""
    return PersonaSpec.from_request(
        brief, kwargs.pop("dials", {}), kwargs.pop("details", {})
    )


def system_of(spec_):
    return build_draft_prompt(spec_)[0]["content"]

WELL_FORMED = """NAME: Rennick
DESCRIPTION: A suspicious harbourmaster
ROUTER_HINTS: boats, cargo, the harbour
LENGTH_BIAS: shorter
AVATAR_COLOR: #2E7D32
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
    def test_only_the_levers_the_form_does_not_set_reach_the_prompt(self):
        # A lever the dials or details already set is an instruction, not
        # advice; restating it as advice invites the model to overrule it.
        system = system_of(spec())
        # Matched on the hint, not the title: several lever titles are also
        # detail-field labels, which legitimately appear elsewhere.
        for lever in persona_draft.LEVERS:
            assert (lever.hint in system) is not bool(lever.superseded_by)

    def test_every_superseded_lever_names_a_real_field(self):
        for lever in persona_draft.LEVERS:
            if lever.superseded_by:
                assert (lever.superseded_by in persona_draft.DIALS_BY_KEY
                        or lever.superseded_by in persona_draft.DETAILS_BY_KEY)

    def test_something_is_still_left_for_the_model_to_invent(self):
        # If the form ever covered every lever the block would be empty and
        # the "invent whatever is left open" framing would be a lie.
        assert any(not lv.superseded_by for lv in persona_draft.LEVERS)

    def test_the_anti_patterns_are_named_explicitly(self):
        # The model produces exactly these unless told not to.
        system = system_of(spec())
        assert "topic lists" in system
        assert "adjective piles" in system

    def test_the_brief_says_who_they_are(self):
        system = system_of(spec("a suspicious harbourmaster"))
        assert "a suspicious harbourmaster" in system

    def test_the_prompt_does_not_grow_with_the_cast(self):
        # Distinctness comes from the specification, not from contrast with
        # the existing personas — so the prompt is the same size whether
        # there are none or fifty, and a draft costs the same either way.
        system = system_of(spec())
        assert "cast" not in system.lower()
        assert len(system) < 6000

    def test_a_chosen_option_sends_its_instruction_not_its_label(self):
        # "Coarse" on its own is exactly as vague as the brief was; the
        # instruction behind it is what does the work.
        system = system_of(spec(dials={"register": "coarse"}))
        assert "crude turns of phrase" in system

    def test_the_defaults_push_against_the_house_style(self):
        # Left alone, the model writes every character as an essayist.
        system = system_of(spec())
        assert "clear and unshowy" in system          # vocabulary
        assert "examples rather than principles" in system   # abstraction

    def test_an_unspecified_dial_is_handed_back_to_the_model(self):
        system = system_of(spec(dials={"stance": ""}))
        assert "decide for yourself" in system
        assert "Stance" in system

    def test_the_dials_are_declared_independent(self):
        # The failure this whole feature exists for: one word bleeding
        # across word choice, temper and cooperativeness at once.
        system = system_of(spec(dials={"register": "coarse"}))
        assert "must not bleed together" in system
        assert "does not make a character hostile" in system

    def test_a_given_detail_is_quoted_and_a_blank_one_is_left_open(self):
        system = system_of(spec(details={"never": "never guesses at cargo"}))
        assert "never guesses at cargo" in system
        assert "NOT GIVEN" in system

    def test_the_user_turn_is_a_constant(self):
        # Everything the user typed is in the system turn, so the user turn
        # carries no instruction a model could mistake for the brief.
        messages = build_draft_prompt(spec("a harbourmaster"))
        assert messages[1] == {"role": "user", "content": "Write the character."}


class TestPersonaSpec:
    def test_unknown_dial_keys_are_dropped(self):
        assert "colour" not in spec(dials={"colour": "blue"}).dials

    def test_an_unknown_dial_value_becomes_unspecified(self):
        # Passing the bare word through would send an option the prompt has
        # no instruction for — the vagueness the dials exist to remove.
        s = spec(dials={"register": "sassy"})
        assert s.dials["register"] == persona_draft.UNSPECIFIED
        assert s.instruction_for("register") is None

    def test_an_unset_dial_falls_back_to_its_default(self):
        assert "unshowy" in spec().instruction_for("vocabulary")

    def test_every_dial_offers_an_opt_out(self):
        for dial in persona_draft.DIALS:
            assert dial.options[0].value == persona_draft.UNSPECIFIED
            assert not dial.options[0].instruction

    def test_every_dial_default_is_a_real_option(self):
        for dial in persona_draft.DIALS:
            assert dial.option(dial.default) is not None

    def test_every_dial_is_in_a_rendered_group(self):
        grouped = [d.key for _, dials in persona_draft.DIAL_GROUPS for d in dials]
        assert grouped == [d.key for d in persona_draft.DIALS]

    def test_unknown_detail_keys_are_dropped(self):
        assert spec(details={"favourite_colour": "blue"}).details == {}

    def test_blank_details_are_dropped_rather_than_sent_empty(self):
        assert spec(details={"wants": "   "}).details == {}

    def test_details_are_capped_and_flattened(self):
        s = spec(details={"wants": "a\nb " + "x" * 1000})
        assert len(s.details["wants"]) <= persona_draft.MAX_DETAIL_CHARS
        assert "\n" not in s.details["wants"]


class TestSeededParsing:
    """Refining parses over the persona as it stands, not over nothing."""

    CURRENT = PersonaDraft(
        name="Rennick",
        description="Harbourmaster",
        system_prompt="You run the harbour.",
        router_hints="boats, cargo",
        length_bias=LengthBias.SHORTER,
        avatar_color="#2E7D32",
    )

    def test_an_omitted_field_keeps_its_current_value(self):
        # The refine prompt asks the model to omit unchanged fields, which
        # is only safe if omitting one changes nothing.
        out = parse_draft("SYSTEM_PROMPT:\nYou run the harbour and you swear.",
                          base=self.CURRENT)
        assert out.description == "Harbourmaster"
        assert out.router_hints == "boats, cargo"
        assert out.length_bias is LengthBias.SHORTER
        assert out.name == "Rennick"
        assert out.system_prompt.endswith("you swear.")

    def test_an_empty_block_is_not_a_deletion(self):
        # Models emit "DESCRIPTION:" with nothing after it; that is a
        # non-answer, not an instruction to blank the field.
        out = parse_draft("DESCRIPTION:\nSYSTEM_PROMPT:\nYou run it.", base=self.CURRENT)
        assert out.description == "Harbourmaster"

    def test_an_unusable_length_bias_keeps_the_current_one(self):
        out = parse_draft("LENGTH_BIAS: terse\nSYSTEM_PROMPT:\nYou run it.",
                          base=self.CURRENT)
        assert out.length_bias is LengthBias.SHORTER

    def test_a_changed_field_is_taken(self):
        out = parse_draft("DESCRIPTION: Harbourmaster, coarse\nLENGTH_BIAS: longer\n"
                          "SYSTEM_PROMPT:\nYou run it.", base=self.CURRENT)
        assert out.description == "Harbourmaster, coarse"
        assert out.length_bias is LengthBias.LONGER

    def test_notes_are_never_inherited(self):
        # The notes describe the reply that produced them.
        base = replace(self.CURRENT, notes=["from a previous round"])
        out = parse_draft("SYSTEM_PROMPT:\nYou run it.", base=base)
        assert out.notes == []

    def test_the_base_is_not_mutated(self):
        before = replace(self.CURRENT)
        parse_draft("NAME: Someone\nSYSTEM_PROMPT:\np", base=self.CURRENT)
        assert self.CURRENT == before


class TestRefinePrompt:
    CURRENT = TestSeededParsing.CURRENT

    def system(self, instruction="make him coarser"):
        return build_refine_prompt(self.CURRENT, instruction)[0]["content"]

    def test_the_whole_persona_is_sent(self):
        system = self.system()
        assert "You run the harbour." in system
        assert "Rennick" in system
        assert "Harbourmaster" in system
        assert "boats, cargo" in system
        assert "shorter" in system

    def test_the_instruction_is_sent(self):
        assert "make him coarser" in self.system()

    def test_conservation_is_stated_as_loudly_as_the_change(self):
        # The failure mode: a model handed a prompt and one instruction
        # rewrites the whole thing in its own register.
        system = self.system()
        assert "nothing else" in system
        assert "revision, not a rewrite" in system
        assert "same character" in system

    def test_a_vague_instruction_is_read_narrowly(self):
        assert "smallest part" in self.system("make him better")

    def test_the_independence_note_is_repeated_here(self):
        # A free-text instruction is the same global-dial trap the dials
        # exist to remove: "make him crude" must not make him hostile.
        system = self.system("make him crude")
        assert "does not make a character hostile" in system
        assert "word choice and nothing else" in system.lower()

    def test_the_name_is_not_up_for_revision(self):
        system = self.system()
        assert "Do not change the name" in system
        assert "NAME:" not in system

    def test_unchanged_fields_may_be_omitted(self):
        assert "Omit a label entirely if that field is unchanged" in self.system()

    def test_the_shared_writing_rules_are_used(self):
        assert persona_draft.WRITING_RULES in self.system()


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
