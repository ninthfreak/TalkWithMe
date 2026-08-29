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
    def test_covers_other_personas_in_both_forms(self):
        assert stop_sequences("Alex", ["Alex", "Luna"]) == ["\nLuna:", "\n[Luna]:"]

    def test_excludes_the_speaker(self):
        assert all("Alex" not in s for s in stop_sequences("Alex", ["Alex", "Luna"]))

    def test_capped_for_backends_that_reject_long_lists(self):
        names = [f"P{i}" for i in range(20)]
        assert len(stop_sequences("Alex", names)) == 8

    def test_empty_when_alone_in_the_room(self):
        assert stop_sequences("Alex", ["Alex"]) == []
