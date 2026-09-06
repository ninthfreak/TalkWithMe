"""Unit tests for app/services/persona_draft.py.

The parser is the risky half: it consumes whatever a local model felt like
emitting. Every "sloppy output" case below is a shape models actually
produce — a code fence, a chatty preamble, a missing label, a quoted
value, an invented enum member.
"""

import re

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
    """The draft does paperwork, not authorship.

    Built forwards from a theory of what makes a character distinct, this
    file spent a long time asking a model to *write* one. Every failure
    was the theory wrong somewhere new, and every fix was another
    paragraph of counter-instruction. Checked at last against a
    description known to work, the brief used as-is beat everything the
    generator wrote from it — so it stopped writing them.
    """

    def test_the_brief_is_what_the_prompt_is_about(self):
        system = system_of(spec("a bookbinder who repairs family bibles"))
        assert "a bookbinder who repairs family bibles" in system
        assert "The character is theirs and is not yours to improve" in system

    def test_it_asks_for_the_words_back_in_the_second_person(self):
        system = system_of(spec())
        assert "Give their words back" in system
        assert "Put them in the second person" in system

    def test_it_forbids_the_embellishment_that_used_to_be_the_job(self):
        system = system_of(spec())
        assert "Do not add traits, habits, opinions, mannerisms or backstory" in system
        assert "do not make it longer" in system
        assert "A short description stays a short prompt" in system

    def test_details_are_carried_as_written(self):
        system = system_of(spec(details={"never": "never lets anyone leave empty-handed"}))
        assert "never lets anyone leave empty-handed" in system

    def test_a_blank_detail_is_not_something_to_invent(self):
        # It used to say "invent only what earns its place", which is
        # still an invitation to invent.
        system = system_of(spec())
        assert "invent" not in system.lower()

    def test_a_set_dial_is_an_instruction_and_an_unset_one_is_silence(self):
        system = system_of(spec(dials={"vocabulary": "crude"}))
        assert "crude turns of phrase" in system
        assert "Sentence shape" not in system

    def test_the_theory_blocks_are_gone_from_the_prompt(self):
        # Levers, anti-patterns and the writing rules were all
        # instructions for writing a character. Nobody is writing one.
        system = system_of(spec())
        for lever in persona_draft.LEVERS:
            assert (lever.prompt_hint or lever.hint) not in system
        for anti in persona_draft.ANTI_PATTERNS:
            assert anti not in system

    def test_the_prompt_is_a_fraction_of_what_it_was(self):
        # 534 words of instruction for writing a character, against a
        # twelve-word brief. Asking for paperwork needs far less.
        assert len(system_of(spec("a bookbinder")).split()) < 300

    def test_the_user_turn_is_a_constant(self):
        messages = build_draft_prompt(spec("a harbourmaster"))
        assert messages[1] == {"role": "user", "content": "Write the character."}


class TestKeepingTheUsersWords:
    """The floor, enforced in code rather than asked for in a prompt."""

    BRIEF = "A bookbinder who repairs family bibles and remembers every story"

    def test_the_brief_alone_makes_a_usable_prompt(self):
        assert persona_draft.prompt_from_brief(spec(self.BRIEF)) == self.BRIEF

    def test_details_are_appended_in_the_users_words(self):
        built = persona_draft.prompt_from_brief(
            spec(self.BRIEF, details={"never": "never lets anyone leave empty-handed"})
        )
        assert built.startswith(self.BRIEF)
        assert "never lets anyone leave empty-handed" in built

    def test_a_second_person_rewrite_counts_as_kept(self):
        # The one change the draft is asked to make must not read as a
        # rewrite, or the fallback would fire on every good draft.
        rewritten = ("You are a bookbinder. You repair family bibles and you "
                     "remember every story behind them.")
        assert persona_draft.kept_fraction(self.BRIEF, rewritten) >= \
            persona_draft._KEPT_WORDS_FLOOR

    def test_a_replacement_character_does_not(self):
        invented = ("You run a harbour and you assume everyone is smuggling. You "
                    "answer a question with a question about who signed for it.")
        assert persona_draft.kept_fraction(self.BRIEF, invented) < \
            persona_draft._KEPT_WORDS_FLOOR

    def test_an_empty_brief_cannot_fail_the_check(self):
        assert persona_draft.kept_fraction("", "anything at all") == 1.0

    def test_filler_words_do_not_prop_up_the_score(self):
        # "the a and of" overlapping is not the brief surviving.
        assert persona_draft.kept_fraction(self.BRIEF, "the and of a to in it") == 0.0


class TestPersonaSpec:
    def test_unknown_dial_keys_are_dropped(self):
        assert "colour" not in spec(dials={"colour": "blue"}).dials

    def test_an_unknown_dial_value_becomes_unspecified(self):
        # Passing the bare word through would send an option the prompt has
        # no instruction for — the vagueness the dials exist to remove.
        s = spec(dials={"vocabulary": "sassy"})
        assert s.dials["vocabulary"] == persona_draft.UNSPECIFIED
        assert s.instruction_for("vocabulary") is None

    def test_a_dial_dropped_from_the_form_is_ignored_not_an_error(self):
        # Register, Temperament and Certainty were removed; a stale page
        # still posting them must draft, not 500.
        s = spec(dials={"register": "coarse", "temperament": "volatile"})
        assert s.dials == {}

    def test_an_unset_dial_says_nothing(self):
        assert spec().instruction_for("vocabulary") is None

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

    def test_a_word_about_speech_changes_only_speech(self):
        # A free-text instruction is the same trap a disposition dial is:
        # "make him crude" must not make him coarser in temper as well as
        # in vocabulary.
        system = self.system("make him crude")
        assert "changes their word choice and nothing else" in system
        assert "hostile" not in system.lower()

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

    def test_assistant_vocabulary_is_flagged(self):
        draft = PersonaDraft(
            name="R",
            system_prompt="You are a helpful and friendly assistant who is curious. " * 8,
        )
        assert any("assistant vocabulary" in w for w in critique(draft))

    def test_one_warm_word_is_not_a_fault(self):
        # Flagging a single "kind" taught the opposite of the lesson: it
        # read as "warmth is a mistake", which is how a cast ends up
        # uniformly unpleasant.
        draft = PersonaDraft(
            name="Bess",
            system_prompt=(
                "You run the bakery and you are delighted to see whoever walks in. "
                "You ask after their family by name before you talk about bread. "
                "You never let anyone leave without something in their hand, and "
                "you are kind about people the others have written off."
            ),
        )
        assert critique(draft) == []

    def test_a_pile_of_adjectives_is_a_fault(self):
        draft = PersonaDraft(
            name="R",
            system_prompt="You are friendly, curious and thoughtful, and you never stop. " * 6,
        )
        assert any("pile of adjectives" in w for w in critique(draft))

    def test_a_prompt_with_no_refusal_in_it_is_not_a_fault(self):
        # This used to warn that "nothing here says what this character
        # will not do", which put a refusal into every persona the app
        # ever drafted — heavy-handed, and a steady source of characters
        # announcing what they do not do.
        draft = PersonaDraft(
            name="Bess",
            system_prompt=(
                "You have bound books for thirty years and you can tell how someone "
                "was loved by how their bible was handled. You keep the good glue "
                "for jobs nobody is paying for."
            ),
        )
        assert critique(draft) == []

    def test_a_long_prompt_is_flagged_as_a_rulebook(self):
        draft = PersonaDraft(
            name="R",
            system_prompt="You always ask who signed for it. You never speculate. " * 12,
        )
        assert any("rulebook" in w for w in critique(draft))

    def test_a_sketch_length_prompt_draws_no_length_warning(self):
        draft = PersonaDraft(name="R", system_prompt="You ask about cargo. " * 15)
        assert not any("rulebook" in w or "too little" in w for w in critique(draft))

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
