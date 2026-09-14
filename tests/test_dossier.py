"""Tests for app/services/dossier.py — the interview note store.

The question this module answers: how does an interviewer accumulate more
than a prompt can hold, and still walk into each turn knowing what it has?
The answer is a tagged file read two levels at a time — the index always,
the contents selectively — so most of what is tested here is about which
notes come back, and about the tag snapping that keeps the index from
fragmenting into "school", "schools" and "schooling".
"""

import pytest

from app.services import dossier
from app.services.dossier import Note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir(tmp_path, name="Marion"):
    persona_dir = tmp_path / name
    persona_dir.mkdir(parents=True, exist_ok=True)
    return persona_dir


def _note(text, *tags, when="", subject="Tony", **flags):
    return Note(subject=subject, text=text, tags=tuple(tags), when=when, **flags)


def _file(persona_dir, subject="Tony"):
    return dossier.notes_path(persona_dir, subject)


def _write(persona_dir, *lines, subject="Tony"):
    path = _file(persona_dir, subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The line format
# ---------------------------------------------------------------------------

class TestParseNoteLine:
    def test_a_plain_tagged_note(self):
        note = dossier.parse_note_line(
            "[Tony] #work #vickers @1978 Started at Vickers straight from school."
        )
        assert note.subject == "Tony"
        assert note.tags == ("work", "vickers")
        assert note.when == "1978"
        assert note.text == "Started at Vickers straight from school."
        assert not note.assumed and not note.open_question

    def test_flags_are_read(self):
        assert dossier.parse_note_line("[Tony] (assumed) #work He resented it.").assumed
        assert dossier.parse_note_line("[Tony] (open) #work Why did he leave?").open_question
        assert dossier.parse_note_line("[Tony] (sensitive) #health Not 2003.").sensitive

    def test_the_memory_files_assumed_spellings_are_accepted(self):
        # Same tolerance as parse_memory_line: these files are hand-edited
        # and a guess written as "(guess)" is still a guess.
        for spelling in ("(assumed)", "(assumption)", "(guess)", "(guessed)"):
            assert dossier.parse_note_line(f"[Tony] {spelling} He resented it.").assumed

    def test_marker_order_is_not_enforced_on_input(self):
        # Hand-edited files. Insisting on an order would silently lose a tag.
        note = dossier.parse_note_line("[Tony] #work (assumed) @1986 #vickers He was bitter.")
        assert note.assumed and note.tags == ("work", "vickers") and note.when == "1986"

    def test_a_hash_inside_prose_stays_in_the_prose(self):
        note = dossier.parse_note_line("[Tony] #work He joined the #2 machine shop.")
        assert note.tags == ("work",)
        assert note.text == "He joined the #2 machine shop."

    def test_an_untagged_line_is_still_a_note(self):
        note = dossier.parse_note_line("[Tony] He hated the cold.")
        assert note.tags == () and note.text == "He hated the cold."

    def test_a_line_with_no_subject_parses_rather_than_raising(self):
        assert dossier.parse_note_line("loose text").text == "loose text"

    def test_tags_are_normalised_on_the_way_in(self):
        note = dossier.parse_note_line("[Tony] #Work #WORK #war-years Fought.")
        assert note.tags == ("work", "war-years")   # deduped, lowercased

    def test_round_trips_through_stored(self):
        line = "[Tony] (assumed) #work #vickers @1978 He resented it."
        assert dossier.parse_note_line(line).stored() == line

    def test_stored_writes_markers_in_one_canonical_order(self):
        note = _note("He was bitter.", "work", when="1986", assumed=True)
        assert note.stored() == "[Tony] (assumed) #work @1986 He was bitter."


class TestNormaliseTag:
    @pytest.mark.parametrize("raw,expected", [
        ("Work", "work"),
        ("#work", "work"),
        ("war years", "war-years"),
        ("war_years", "war-years"),
        ("Vickers!", "vickers"),
        ("  school  ", "school"),
        ("", ""),
        ("!!!", ""),
    ])
    def test_spellings_collapse(self, raw, expected):
        assert dossier.normalise_tag(raw) == expected

    def test_a_very_long_tag_is_cut(self):
        assert len(dossier.normalise_tag("a" * 200)) == dossier.MAX_TAG_CHARS


# ---------------------------------------------------------------------------
# Tag snapping
# ---------------------------------------------------------------------------

class TestSnapTag:
    """The chosen design: the model proposes, the app snaps to a tag
    already in use unless nothing is near."""

    def test_an_exact_tag_is_not_a_merge(self):
        assert dossier.snap_tag("work", ["work", "family"]) == ("work", None)

    def test_a_plural_snaps_to_the_singular_already_in_use(self):
        assert dossier.snap_tag("schools", ["school"]) == ("school", "schools")

    def test_a_singular_snaps_to_the_plural_already_in_use(self):
        assert dossier.snap_tag("school", ["schools"]) == ("schools", "school")

    def test_an_ies_plural_snaps(self):
        assert dossier.snap_tag("factories", ["factory"]) == ("factory", "factories")

    def test_a_gerund_snaps_to_the_plain_word(self):
        # The spelling a model reaches for constantly when asked to tag a
        # fact about a job it has already filed under "work".
        assert dossier.snap_tag("working", ["work"]) == ("work", "working")
        assert dossier.snap_tag("schooling", ["school"]) == ("school", "schooling")

    def test_stemming_does_not_eat_short_words(self):
        # "king" would otherwise stem to "k" and match nearly anything.
        assert dossier.snap_tag("king", ["k"]) == ("king", None)
        assert dossier.snap_tag("doing", ["do"]) == ("doing", None)

    def test_a_near_spelling_snaps(self):
        assert dossier.snap_tag("vickers", ["vicker"]) == ("vicker", "vickers")

    def test_a_similar_word_for_a_different_topic_stays_apart(self):
        # "arms" and "army" are 0.75 similar and are not the same subject.
        assert dossier.snap_tag("arms", ["army"]) == ("arms", None)

    def test_a_genuinely_new_topic_is_new(self):
        tag, merged = dossier.snap_tag("army", ["work", "family", "school"])
        assert (tag, merged) == ("army", None)

    def test_synonyms_do_not_snap_and_that_is_the_prompts_job(self):
        # "job" and "work" share no letters in order; no string measure
        # merges them. The note prompt shows the model the existing topics
        # and asks it to reuse one — this is only the backstop under that.
        assert dossier.snap_tag("job", ["work"]) == ("job", None)

    def test_short_tags_do_not_snap_on_spelling(self):
        # "war"/"bar" is 0.67 similar and they are different topics.
        assert dossier.snap_tag("war", ["bar"]) == ("war", None)

    def test_the_first_tag_of_an_empty_index_is_kept(self):
        assert dossier.snap_tag("work", []) == ("work", None)

    def test_an_unusable_tag_comes_back_empty(self):
        assert dossier.snap_tag("!!!", ["work"]) == ("", None)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class TestStorage:
    def test_notes_live_in_a_per_subject_file_under_notes(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("Born in Leeds.", "family")])
        path = persona_dir / "notes" / "tony.txt"
        assert path.is_file()
        assert path.read_text() == "[Tony] #family Born in Leeds.\n"

    def test_the_filename_is_casefolded_so_a_hand_edit_cannot_split_it(self, tmp_path):
        persona_dir = _dir(tmp_path)
        assert dossier.notes_path(persona_dir, "TONY") == dossier.notes_path(persona_dir, "tony")

    def test_the_display_spelling_survives_inside_the_file(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("Born in Leeds.")])
        assert dossier.read_notes(persona_dir, "tony")[0].subject == "Tony"

    def test_a_missing_dossier_reads_as_empty(self, tmp_path):
        assert dossier.read_notes(_dir(tmp_path), "Nobody") == []

    def test_subjects_lists_everyone_with_a_dossier(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("Born in Leeds.")])
        dossier.append_notes(persona_dir, "Kira", [_note("Sails.", subject="Kira")])
        assert sorted(dossier.subjects(persona_dir)) == ["Kira", "Tony"]

    def test_writing_an_empty_dossier_removes_the_file(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("Born in Leeds.")])
        dossier.write_notes(persona_dir, "Tony", [])
        assert not _file(persona_dir).exists()

    def test_the_file_grows_past_the_memory_ceiling(self, tmp_path):
        # The whole reason this store exists: memories.txt caps at 16 KB
        # because every byte is injected. Nothing here caps.
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [
            _note(f"Fact number {i} about a long working life.", "work")
            for i in range(800)
        ])
        assert len(_file(persona_dir).read_bytes()) > 32_000
        assert len(dossier.read_notes(persona_dir, "Tony")) == 800


class TestAppendNotes:
    def test_tags_snap_to_what_is_already_filed(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("Left school at 15.", "school")])

        result = dossier.append_notes(
            persona_dir, "Tony", [_note("Hated the grammar.", "schools")])

        assert result.filed[0].tags == ("school",)
        assert result.merges == [("schools", "school")]

    def test_a_merge_is_reported_so_it_can_be_seen(self, tmp_path, caplog):
        import logging
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("A.", "school")])
        with caplog.at_level(logging.INFO):
            dossier.append_notes(persona_dir, "Tony", [_note("B.", "schools")])
        assert "filed '#schools' under existing '#school'" in caplog.text

    def test_the_same_fact_twice_is_filed_once(self, tmp_path):
        # The note pass re-reads the same conversation by design.
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("Born in Leeds.", "family")])

        result = dossier.append_notes(
            persona_dir, "Tony", [_note("born in leeds.", "childhood")])

        assert result.filed == [] and len(result.duplicates) == 1
        assert len(dossier.read_notes(persona_dir, "Tony")) == 1

    def test_notes_append_in_order(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("First.")])
        dossier.append_notes(persona_dir, "Tony", [_note("Second.")])
        assert [n.text for n in dossier.read_notes(persona_dir, "Tony")] == ["First.", "Second."]

    def test_the_subject_is_forced_to_the_file_it_is_filed_in(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("Sails.", subject="Somebody Else")])
        assert dossier.read_notes(persona_dir, "Tony")[0].subject == "Tony"

    def test_an_empty_note_is_not_filed(self, tmp_path):
        persona_dir = _dir(tmp_path)
        result = dossier.append_notes(persona_dir, "Tony", [_note("   ")])
        assert result.filed == [] and not _file(persona_dir).exists()

    def test_too_many_tags_are_cut_to_the_cap(self, tmp_path):
        persona_dir = _dir(tmp_path)
        result = dossier.append_notes(
            persona_dir, "Tony", [_note("A.", "a1", "b2", "c3", "d4", "e5", "f6")])
        assert len(result.filed[0].tags) == dossier.MAX_TAGS_PER_NOTE


class TestOpenQuestions:
    """Facts accumulate; questions do not. Replace-on-write, so nothing
    has to ask a model to delete a line."""

    def test_questions_replace_rather_than_append(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.replace_open_questions(persona_dir, "Tony", [
            _note("Why did he leave Vickers?"), _note("What about 1986-89?"),
        ])
        dossier.replace_open_questions(persona_dir, "Tony", [_note("What about 1986-89?")])

        questions = [n.text for n in dossier.read_notes(persona_dir, "Tony") if n.open_question]
        assert questions == ["What about 1986-89?"]

    def test_replacing_questions_leaves_the_facts_alone(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.append_notes(persona_dir, "Tony", [_note("Born in Leeds.", "family")])
        dossier.replace_open_questions(persona_dir, "Tony", [_note("Which hospital?")])
        dossier.replace_open_questions(persona_dir, "Tony", [])

        remaining = dossier.read_notes(persona_dir, "Tony")
        assert [n.text for n in remaining] == ["Born in Leeds."]

    def test_questions_are_stored_with_the_open_marker(self, tmp_path):
        persona_dir = _dir(tmp_path)
        dossier.replace_open_questions(persona_dir, "Tony", [_note("Why Leeds?")])
        assert _file(persona_dir).read_text() == "[Tony] (open) Why Leeds?\n"


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

class TestTopics:
    def test_topics_come_back_biggest_first(self):
        notes = [_note("a", "work"), _note("b", "work"), _note("c", "family")]
        assert [t.tag for t in dossier.topics(notes)] == ["work", "family"]

    def test_an_open_question_is_a_reason_to_ask_not_evidence_of_coverage(self):
        # Counting it would make a thin topic look answered.
        notes = [_note("a", "work"), _note("Why?", "army", open_question=True)]
        counts = {t.tag: t.count for t in dossier.topics(notes)}
        assert counts == {"work": 1}

    def test_a_topic_holding_a_boundary_is_marked(self):
        notes = [_note("Not 2003.", "health", sensitive=True), _note("Fit.", "health")]
        assert dossier.topics(notes)[0].sensitive is True

    def test_untagged_notes_are_in_no_topic(self):
        assert dossier.topics([_note("loose")]) == []


# ---------------------------------------------------------------------------
# Selection — the two-level read
# ---------------------------------------------------------------------------

class TestSelect:
    def _dossier(self):
        return [
            _note("Started at Vickers straight from school.", "work", "vickers", when="1978"),
            _note("Left after the second round of layoffs.", "work", "vickers", when="1986"),
            _note("Father was a fitter at the same yard.", "family", "father", when="1960s"),
            _note("Mother taught at the grammar.", "family", "mother"),
            _note("Two years in Cyprus.", "army", when="1972"),
            _note("Why did he leave Vickers, really?", "work", open_question=True),
        ]

    def test_the_index_is_always_complete(self):
        chosen = dossier.select(self._dossier(), "tell me about the army")
        assert {t.tag for t in chosen.index} == {"work", "vickers", "family", "father", "mother", "army"}

    def test_only_the_matching_topic_is_loaded(self):
        chosen = dossier.select(self._dossier(), "what about your father?")
        assert "family" in chosen.matched or "father" in chosen.matched
        assert any("fitter" in n.text for n in chosen.notes)
        assert not any("Vickers" in n.text for n in chosen.notes)

    def test_a_topic_name_in_the_question_wins(self):
        chosen = dossier.select(self._dossier(), "and the army?")
        assert chosen.matched[0] == "army"

    def test_a_word_in_a_notes_text_also_matches(self):
        chosen = dossier.select(self._dossier(), "did you ever go to Cyprus?")
        assert any("Cyprus" in n.text for n in chosen.notes)

    def test_open_questions_are_never_selective(self):
        # There are few and they are the reason to ask anything.
        chosen = dossier.select(self._dossier(), "something unrelated entirely")
        assert [n.text for n in chosen.open_questions] == ["Why did he leave Vickers, really?"]

    def test_nothing_matching_falls_back_to_the_most_recent(self):
        chosen = dossier.select(self._dossier(), "hello", fallback_recent=2)
        assert chosen.matched == []
        assert [n.text for n in chosen.notes] == [
            "Mother taught at the grammar.", "Two years in Cyprus."]

    def test_an_empty_dossier_selects_nothing(self):
        chosen = dossier.select([], "anything")
        assert chosen.notes == [] and chosen.index == []

    def test_selection_stays_in_file_order(self):
        # A life read out of sequence reads as a shuffled life.
        chosen = dossier.select(self._dossier(), "vickers work")
        assert [n.when for n in chosen.notes] == ["1978", "1986"]

    def test_the_budget_stops_a_third_topic_from_piling_on(self):
        notes = [_note("x" * 300, f"t{i}") for i in range(3)]
        chosen = dossier.select(notes, "t0 t1 t2", budget_chars=400)
        assert len(chosen.matched) < 3

    def test_a_dossier_far_larger_than_a_prompt_still_selects_small(self):
        notes = [_note(f"Fact {i} about the yard.", "work") for i in range(500)]
        notes += [_note("Two years in Cyprus.", "army", when="1972")]
        chosen = dossier.select(notes, "tell me about Cyprus and the army")
        assert [n.text for n in chosen.notes] == ["Two years in Cyprus."]


class TestABigTopicCannotSwampTheRest:
    """After a long interview one topic holds hundreds of notes. Every
    rule here exists because that topic otherwise wins every turn and
    spends the whole prompt on itself."""

    def _lopsided(self):
        # 400 notes about the yard, two about his father. Both mention
        # "yard", which is what makes this the hard case.
        notes = [_note(f"Yard detail number {i}.", "work", "vickers") for i in range(400)]
        notes += [
            _note("Father was a fitter at the same yard.", "family", "father", when="1960s"),
            _note("Father died the winter after the wedding.", "family", "father", when="1989"),
        ]
        return notes

    def test_a_topic_scores_on_distinct_words_not_on_how_often(self):
        # Summing over notes scored "vickers" 400 for one stray word and
        # it won a question about somebody's father.
        chosen = dossier.select(self._lopsided(), "tell me about your father, was he at the yard?")
        assert chosen.matched == ["father"]
        assert all("Father" in n.text for n in chosen.notes)

    def test_the_budget_is_enforced_on_the_first_topic_too(self):
        # It used to be applied from the second topic onward, so one huge
        # topic went in whole — 12,000 characters of shipyard.
        chosen = dossier.select(self._lopsided(), "yard", budget_chars=500)
        assert sum(len(n.text) for n in chosen.notes) <= 500

    def test_a_topic_that_merely_brushed_the_question_is_not_loaded(self):
        chosen = dossier.select(self._lopsided(), "tell me about your father")
        assert "vickers" not in chosen.matched and "work" not in chosen.matched

    def test_no_topic_contributes_more_than_the_note_cap(self):
        # Per topic, not per block: a turn that legitimately loads two
        # folders may show a handful from each.
        chosen = dossier.select(self._lopsided(), "yard", budget_chars=100_000)
        for tag in chosen.matched:
            from_topic = [n for n in chosen.notes if tag in n.tags]
            assert len(from_topic) <= dossier.MAX_NOTES_PER_TOPIC
        assert len(chosen.notes) < 400

    def test_trimming_keeps_the_notes_that_answer_the_question(self):
        notes = [_note(f"Routine entry {i}.", "work") for i in range(60)]
        notes.append(_note("The Cyprus posting came through in March.", "work"))
        chosen = dossier.select(notes, "what about Cyprus?")
        assert any("Cyprus" in n.text for n in chosen.notes)

    def test_a_topic_whose_contents_were_not_loaded_is_still_in_the_index(self):
        # The point of an always-complete index: the interviewer can see
        # that there are army notes even on a turn that did not load them,
        # which is what lets it say "we have barely touched that".
        notes = self._lopsided() + [_note("Two years in Cyprus.", "army")]
        chosen = dossier.select(notes, "tell me about your father")
        assert "army" in {t.tag for t in chosen.index}
        assert "army" not in chosen.matched


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRenderBlock:
    def test_the_block_names_what_there_is_and_what_is_relevant(self):
        notes = [
            _note("Started at Vickers straight from school.", "work", when="1978"),
            _note("Mother taught at the grammar.", "family"),
            _note("Why did he leave, really?", "work", open_question=True),
        ]
        block = dossier.render_block(dossier.select(notes, "about your work"), "Tony")

        assert "What you have written down about Tony:" in block
        assert "work (1)" in block and "family (1)" in block
        assert "Started at Vickers straight from school. (1978)" in block
        assert "You still want to know:" in block
        assert "Why did he leave, really?" in block

    def test_the_assumed_wording_matches_the_memory_injection(self):
        # Two spellings of one idea is a worse use of a small model than
        # remembering the sentence.
        notes = [_note("He resented management.", "work", assumed=True)]
        block = dossier.render_block(dossier.select(notes, "work"), "Tony")
        assert "You have assumed, though nobody said so:" in block

    def test_a_boundary_is_marked_in_the_index(self):
        notes = [_note("He would rather not discuss 2003.", "health", sensitive=True)]
        block = dossier.render_block(dossier.select(notes, "hello"), "Tony")
        assert "health (1, he would rather not)" in block

    def test_the_fallback_says_it_is_the_last_of_what_there_is(self):
        notes = [_note("Born in Leeds.", "family")]
        block = dossier.render_block(dossier.select(notes, "unrelated"), "Tony")
        assert "The last of what you wrote down:" in block

    def test_an_empty_dossier_renders_nothing(self):
        assert dossier.render_block(dossier.select([], "anything"), "Tony") == ""

    def test_dashes_in_tags_and_eras_read_as_words(self):
        notes = [_note("Fought.", "war-years", when="after-the-war")]
        block = dossier.render_block(dossier.select(notes, "war years"), "Tony")
        assert "war years" in block and "(after the war)" in block
