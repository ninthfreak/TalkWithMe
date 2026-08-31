"""Unit tests for app/services/reply_guard.py.

The guard is a pure state machine, so these feed it token lists directly —
no LLM, no router, no app. Two properties matter and pull against each
other: it must cut the reply when another speaker's turn starts, and it
must never mangle ordinary prose. The false-positive cases below are the
ones that would silently truncate a legitimate reply, which is worse than
missing a cut.
"""

import pytest

from app.services.reply_guard import ReplyGuard, stop_sequences


def run(tokens, persona="Alex", known=("Alex", "Luna")):
    """Feed tokens; return (emitted text, stopped)."""
    guard = ReplyGuard(persona, known)
    out = []
    for token in tokens:
        out.append(guard.feed(token))
        if guard.stopped:
            break
    out.append(guard.flush())
    return "".join(out), guard.stopped


def chunks(tokens, persona="Alex", known=("Alex", "Luna")):
    """Feed tokens; return the non-empty emitted chunks."""
    guard = ReplyGuard(persona, known)
    out = []
    for token in tokens:
        piece = guard.feed(token)
        if piece:
            out.append(piece)
        if guard.stopped:
            break
    tail = guard.flush()
    if tail:
        out.append(tail)
    return out


class TestPassThrough:
    def test_ordinary_reply_is_untouched(self):
        text = "Sure, I can help with that. It is mostly a matter of scale."
        assert run([text]) == (text, False)

    def test_text_is_identical_however_it_is_chunked(self):
        text = "Hello there, how are you doing today?"
        one_shot, _ = run([text])
        split, _ = run(list(text))  # one character per token
        assert one_shot == split == text

    def test_multi_line_reply_survives(self):
        text = "First point.\nSecond point.\nThird point."
        assert run([text]) == (text, False)


class TestSelfPrefix:
    @pytest.mark.parametrize("prefix", ["Alex: ", "[Alex]: ", "alex: ", "Alex:  "])
    def test_own_prefix_is_stripped(self, prefix):
        assert run([prefix, "Hello there"]) == ("Hello there", False)

    def test_own_prefix_stripped_when_split_across_tokens(self):
        assert run(["Al", "ex", ":", " Hi"]) == ("Hi", False)

    def test_own_prefix_stripped_mid_reply_without_cutting(self):
        # Leaking your own name later in the reply is noise, not a new
        # speaker — strip it, but keep the rest of the reply.
        text, stopped = run(["Fine.\n", "Alex: still me"])
        assert text == "Fine.\nstill me"
        assert stopped is False


class TestForeignSpeakerCut:
    def test_bracketed_known_persona_cuts(self):
        text, stopped = run(["Hi there.\n", "[Luna]: ", "and I disagree"])
        assert text == "Hi there.\n"
        assert stopped is True

    def test_unbracketed_known_persona_cuts(self):
        text, stopped = run(["Hi there.\n", "Luna: continuing"])
        assert text == "Hi there.\n"
        assert stopped is True

    def test_invented_character_cuts(self):
        # The behaviour this exists for: a name nobody configured.
        text, stopped = run(["Hi.\n", "Marcus: I am new here"])
        assert text == "Hi.\n"
        assert stopped is True

    def test_bracketed_invented_character_cuts(self):
        text, stopped = run(["Hi.\n", "[Marcus]: hello everyone"])
        assert text == "Hi.\n"
        assert stopped is True

    def test_prefix_split_across_token_boundaries_still_cuts(self):
        text, stopped = run(["Hi.\n", "[Mar", "cus", "]", ": hello"])
        assert text == "Hi.\n"
        assert stopped is True

    def test_cut_at_the_very_start_yields_nothing(self):
        text, stopped = run(["Luna: I will answer instead"])
        assert text == ""
        assert stopped is True

    def test_cut_at_records_the_name(self):
        guard = ReplyGuard("Alex", ["Luna"])
        guard.feed("Hi.\n[Marcus]: hello")
        assert guard.stopped is True
        assert guard.cut_at == "Marcus"

    def test_feed_after_stop_emits_nothing(self):
        guard = ReplyGuard("Alex", ["Luna"])
        guard.feed("Hi.\nLuna: x")
        assert guard.feed(" more text") == ""
        assert guard.flush() == ""


class TestDecoratedSpeakerTags:
    """The shapes a model actually writes, all of which used to stream through.

    Reported from a live room: one persona wrote three others' turns in a
    single reply, and each of those personas then answered again for
    themselves. Every tag was markdown — "**Luna:**" — and the guard only
    knew the bare "Luna:" form, so it cut nothing.
    """

    # Chunked several ways because the decision is taken mid-stream: a tag
    # arriving one character at a time takes a different path through the
    # buffer than one arriving whole.
    CHUNKS = (1, 2, 3, 7, 500)

    def _cut(self, text, persona="Alex", known=("Alex", "Luna", "Marcus")):
        results = []
        for size in self.CHUNKS:
            guard = ReplyGuard(persona, known)
            out = []
            for i in range(0, len(text), size):
                out.append(guard.feed(text[i:i + size]))
                if guard.stopped:
                    break
            out.append(guard.flush())
            results.append(("".join(out), guard.stopped))
        first = results[0]
        assert all(r == first for r in results), f"chunking changed the result: {results}"
        return first

    @pytest.mark.parametrize("tag", [
        "Luna: ",           # the bare form, the only one ever caught
        "**Luna:** ",       # emphasis around the whole tag
        "**Luna**: ",       # emphasis around the name only
        "*Luna:* ",
        "__Luna:__ ",
        "- Luna: ",         # a list item
        "* Luna: ",
        "1. Luna: ",
        "> Luna: ",         # a blockquote
        "[Luna]: ",
        "[Luna] ",          # brackets alone are enough
        "(Luna): ",
        "Luna — ",          # a dash instead of a colon
        "Luna -- ",
    ])
    def test_a_known_speakers_tag_cuts_whatever_dresses_it(self, tag):
        text, stopped = self._cut(f"That is my view.\n{tag}I disagree.")
        assert stopped is True
        assert "Luna" not in text
        assert "I disagree" not in text
        assert text.startswith("That is my view.")

    @pytest.mark.parametrize("tag", ["### Luna", "**Luna**", "[Luna]"])
    def test_a_name_alone_on_a_line_is_a_turn_header(self, tag):
        text, stopped = self._cut(f"That is my view.\n{tag}\nI disagree.")
        assert stopped is True
        assert "Luna" not in text

    def test_a_bare_name_alone_on_a_line_is_left_alone(self):
        # Undecorated, it is a plausible one-word answer ("Who said so?"
        # "Luna"), and cutting would delete the whole reply.
        text, stopped = self._cut("Who told you?\nLuna")
        assert stopped is False
        assert text == "Who told you?\nLuna"

    def test_a_tag_mid_line_after_a_sentence_end_cuts(self):
        text, stopped = self._cut("That settles it. Luna: I disagree.")
        assert stopped is True
        assert text == "That settles it. "

    def test_a_name_with_a_colon_inside_a_sentence_does_not_cut(self):
        # The other half of the mid-line rule: only a sentence boundary
        # arms it, so this is prose, not a turn.
        text, stopped = self._cut("I would ask Luna: what do you think?")
        assert stopped is False
        assert text == "I would ask Luna: what do you think?"

    def test_an_unknown_name_never_cuts_mid_line(self):
        # Mid-line is only safe for names we know belong to speakers.
        text, stopped = self._cut("It is done. Whatever: we move on.")
        assert stopped is False

    def test_an_invented_character_still_cuts_when_decorated(self):
        text, stopped = self._cut("Fine.\n**Silas:** And who am I?")
        assert stopped is True
        assert "Silas" not in text

    def test_the_whole_run_of_impostor_turns_goes(self):
        text, stopped = self._cut(
            "Fine by me.\n"
            "**Luna:** Not by me.\n"
            "**Marcus:** Nor me.\n"
            "**Harold:** Agreed."
        )
        assert stopped is True
        assert text == "Fine by me.\n"

    @pytest.mark.parametrize("tag", ["Alex: ", "**Alex:** ", "**Alex**: ", "[Alex]: ", "alex: "])
    def test_the_speakers_own_tag_is_stripped_not_cut(self, tag):
        text, stopped = self._cut(f"{tag}Hello there")
        assert stopped is False
        assert text == "Hello there"

    def test_the_speakers_own_name_as_a_heading_is_dropped(self):
        text, stopped = self._cut("### Alex\nHello there")
        assert stopped is False
        assert text == "Hello there"

    def test_emphasis_in_the_body_survives_a_stripped_tag(self):
        # Only the emphasis that dressed the tag is skipped.
        assert self._cut("Alex: **listen** to me.") == ("**listen** to me.", False)
        assert self._cut("**Alex:** **listen** to me.") == ("**listen** to me.", False)


class TestMarkdownThatIsNotASpeaker:
    """The other side of the same change: decoration is everywhere in
    ordinary replies, and none of it may cut."""

    @pytest.mark.parametrize("text", [
        "**Bold opening** and then some prose.",
        "Done. **Bold** start of a sentence.",
        "1. First point\n2. Second point",
        "- a plain bullet\n- another bullet",
        "> A quoted line of prose.",
        "### A heading of my own\nBody text.",
        "Two options:\n- keep it\n- drop it",
        "Luna and Marcus both agree with me.",
        "Ask her. Marcus and I already did.",
        "I said no. She said yes.",
        "Wait. What?",
        "**Note:** this is worth remembering.",
        "- Final Answer: 42.",
    ])
    def test_ordinary_markdown_streams_untouched(self, text):
        guard = ReplyGuard("Alex", ("Alex", "Luna", "Marcus"))
        out = "".join(guard.feed(ch) for ch in text) + guard.flush()
        assert guard.stopped is False
        assert out == text


class TestNoFalsePositives:
    """Cases that must NOT cut — a wrong cut truncates a real reply."""

    @pytest.mark.parametrize(
        "line",
        [
            "The answer is: yes, definitely.",
            "Note: this is worth remembering.",
            "Summary: three things went wrong.",
            "Warning: that will not scale.",
            "Example: consider a queue of ten items.",
            "However: I would still push back on that.",
        ],
    )
    def test_prose_lead_ins_are_not_speakers(self, line):
        text, stopped = run(["Sure.\n", line])
        assert text == f"Sure.\n{line}"
        assert stopped is False

    def test_lowercase_name_like_word_is_not_a_speaker(self):
        text, stopped = run(["Look at this.\n", "config: the value is wrong"])
        assert stopped is False
        assert text.endswith("config: the value is wrong")

    def test_colon_mid_sentence_is_not_a_speaker(self):
        text = "It comes down to one thing: whether the cache is warm."
        assert run([text]) == (text, False)

    def test_long_capitalised_sentence_is_not_a_speaker(self):
        # More than three words rules out a name before any colon appears.
        text = "Alpha Beta Gamma Delta Epsilon are the five stages here."
        assert run([text]) == (text, False)


class TestMultiWordProseLeadIns:
    """A stoplist of single words missed the phrases that actually appear.

    "Final Answer:" and "Executive Summary:" both look exactly like a
    two-word name followed by a colon, so they sailed past a bare-word
    lookup and cut correct replies to nothing.
    """

    @pytest.mark.parametrize(
        "line",
        [
            "Final Answer: 42.",
            "Executive Summary: it went badly.",
            "Short Answer: no.",
            "One Caveat: the cache has to be warm.",
            "My Recommendation: ship it Monday.",
        ],
    )
    def test_a_phrase_ending_in_a_stoplist_word_is_not_a_speaker(self, line):
        text, stopped = run(["Sure.\n", line])
        assert stopped is False
        assert text == f"Sure.\n{line}"

    def test_a_real_two_word_name_still_cuts(self):
        # The last-word check must not swallow the case it exists beside.
        text, stopped = run(["Sure.\n", "Mary Anne: I disagree."])
        assert stopped is True
        assert text == "Sure.\n"


class TestFencedCode:
    """Inside a fenced block nothing is dialogue.

    A YAML key or an HTTP header sits at the start of a line and ends in a
    colon, which is exactly the shape of a speaker prefix.
    """

    def test_a_yaml_key_in_a_fence_does_not_cut(self):
        text, stopped = run(["Here:\n", "```\n", "Model: llama-3\n", "```\n", "Done."])
        assert stopped is False
        assert "Model: llama-3" in text
        assert text.endswith("Done.")

    def test_a_speaker_after_the_fence_closes_still_cuts(self):
        # The fence must be tracked, not merely used as an escape hatch for
        # the rest of the reply.
        text, stopped = run(["```\n", "Model: llama-3\n", "```\n", "Luna: I disagree."])
        assert stopped is True
        assert "Model: llama-3" in text
        assert "I disagree" not in text

    def test_fences_arriving_split_across_tokens_still_open(self):
        text, stopped = run(["``", "`\n", "Note: x\n", "Header: y\n"])
        assert stopped is False
        assert "Header: y" in text


class TestBuffering:
    def test_buffered_text_is_always_released(self):
        # A reply that ends while the guard is still deciding must not lose
        # its tail — flush() is what guarantees that.
        assert run(["Marcus"]) == ("Marcus", False)
        assert run(["["]) == ("[", False)
        assert run(["Alex"]) == ("Alex", False)

    def test_buffering_does_not_stall_a_long_line(self):
        # Nothing may be held indefinitely: a long unpunctuated line must
        # still come out in full.
        text = "x" * 200
        assert run([text]) == (text, False)

    def test_only_the_head_of_a_line_is_coalesced(self):
        # The guard buffers while deciding, so the head of a line may arrive
        # in one chunk; everything after it streams as fed.
        pieces = chunks(["Hello ", "there ", "friend ", "of ", "mine"])
        assert "".join(pieces) == "Hello there friend of mine"
        assert len(pieces) > 1  # not collapsed into a single lump


class TestStopSequences:
    def test_covers_the_shapes_a_tag_actually_takes(self):
        # Bare "Name:" is the form instruct-tuned models use least — they
        # reach for markdown — so the markdown forms have to be here too.
        assert stop_sequences("Alex", ["Alex", "Luna"]) == [
            "\nLuna:", "\n[Luna]:", "\n**Luna:", "\n**Luna**:",
        ]

    def test_every_entry_ends_at_a_colon(self):
        # A looser stop like "\n**Luna" would fire on a reply that merely
        # opens a line by mentioning her in bold, truncating it server-side
        # where the guard cannot see it happen.
        assert all(s.endswith(":") for s in stop_sequences("Alex", ["Luna", "Marcus"]))

    def test_excludes_the_speaker(self):
        assert all("Alex" not in s for s in stop_sequences("Alex", ["Alex", "Luna"]))

    def test_a_full_room_plus_the_human_fits_under_the_cap(self):
        # Five other personas and the user — the widest ordinary room.
        voices = ["Alex", "Luna", "Marcus", "Harold", "Gregory", "Ada", "User"]
        stops = stop_sequences("Alex", voices)
        assert len(stops) == 24
        assert all(any(n in s for s in stops) for n in voices if n != "Alex")

    def test_capped_for_backends_that_reject_long_lists(self):
        names = [f"P{i}" for i in range(20)]
        assert len(stop_sequences("Alex", names)) == 24

    def test_empty_when_alone_in_the_room(self):
        assert stop_sequences("Alex", ["Alex"]) == []
