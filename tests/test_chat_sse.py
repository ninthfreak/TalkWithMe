"""API tests for app/routers/chat.py — the SSE streaming endpoint.

The LLM layer (stream_chat / stream_chat_with_tools / chat_completion) is
replaced with stubs; the session, persistence, and persona-selection logic
are exercised for real.
"""

import uuid

import app.config as app_config
import app.routers.chat as chat_router
from app.session import session
from app.config import (
    ChatRoom,
    ChatRoomsConfig,
    GeneralConfig,
    Persona,
    PersonasConfig,
    LengthBias,
    PlayerProfile,
    TypicalLength,
)
from tests.factories import (
    make_chatrooms,
    make_personas,
    make_settings,
    parse_sse_events,
    sse_events_by_type,
)


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

def _stub_stream(monkeypatch, tokens, finish_reason="stop", capture=None):
    """Replace the plain streaming path with a canned token sequence.

    stream_chat yields event dicts (not bare strings) so both LLM paths
    share one shape; the trailing "finish" event is how the router learns a
    reply was truncated. *capture*, if given, is appended the call kwargs.
    """

    async def fake_stream(messages, max_tokens=None, stop=None):
        if capture is not None:
            capture.append({"messages": messages, "max_tokens": max_tokens, "stop": stop})
        for token in tokens:
            yield {"type": "token", "token": token}
        yield {"type": "finish", "reason": finish_reason}

    monkeypatch.setattr(chat_router, "stream_chat", fake_stream)


def _stub_stream_error_after(monkeypatch, tokens_before_error):
    async def fake_stream(messages, max_tokens=None, stop=None):
        for token in tokens_before_error:
            yield {"type": "token", "token": token}
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_router, "stream_chat", fake_stream)


def _stub_tools(monkeypatch, events, finish_reason="stop"):
    """Replace the agentic path with canned tool-loop events."""

    async def fake_tools(messages, tools, max_tokens=None, stop=None):
        for event in events:
            yield event
        yield {"type": "finish", "reason": finish_reason}

    monkeypatch.setattr(chat_router, "stream_chat_with_tools", fake_tools)


def _stub_completion(monkeypatch, result: str):
    async def fake_completion(prompt, max_tokens=16):
        return result

    monkeypatch.setattr(chat_router, "chat_completion", fake_completion)


def _stub_completion_error(monkeypatch):
    async def fake_completion(prompt, max_tokens=16):
        raise RuntimeError("llm down")

    monkeypatch.setattr(chat_router, "chat_completion", fake_completion)


def _tool_call_event(**overrides):
    event = {
        "type": "tool_call",
        "tool_name": "get_time",
        "arguments": {"zone": "utc"},
        "result": "It is noon.",
        "failed": False,
    }
    event.update(overrides)
    return event


def _patch_chatrooms(monkeypatch, extra_rooms):
    """Extend the fixture chatrooms cache with the given rooms."""
    config = make_chatrooms()
    config.chat_rooms.extend(extra_rooms)
    monkeypatch.setattr(app_config, "_chatrooms_cache", config)
    return config


def _patch_personas(monkeypatch, personas_config: PersonasConfig):
    monkeypatch.setattr(app_config, "_personas_cache", personas_config)


def _patch_general(monkeypatch, **overrides):
    settings = make_settings()
    settings.general = GeneralConfig(**overrides)
    monkeypatch.setattr(app_config, "_settings_cache", settings)
    return settings


def _chat(client, monkeypatch=None, **overrides) -> list:
    """POST /api/chat with sensible defaults and return the parsed SSE events."""
    payload = {"message": "hello there", "who_answers": "Alex", "chat_room": "default"}
    payload.update(overrides)
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    return parse_sse_events(resp.text)


# ---------------------------------------------------------------------------
# Basic single-reply flow
# ---------------------------------------------------------------------------

class TestSingleReply:
    def test_event_sequence_and_payloads(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["Hel", "lo"])
        events = _chat(client, message_id="my-user-uuid")

        types = [e["type"] for e in events]
        # The reply guard buffers the first few characters of a line while
        # it decides whether a speaker prefix is forming, so the head of a
        # reply may arrive coalesced. The sequence and the text are what
        # matter; exact chunk boundaries are not part of the contract.
        assert types[0] == "start"
        assert types[-2:] == ["done", "complete"]
        assert set(types[1:-2]) == {"token"}

        start = events[0]
        assert start["persona"] == "Alex"
        assert start["user_message_id"] == "my-user-uuid"
        assert uuid.UUID(start["message_id"])  # server-generated assistant id

        tokens = sse_events_by_type(events, "token")
        assert "".join(t["token"] for t in tokens) == "Hello"
        assert all(t["persona"] == "Alex" for t in tokens)

        done = sse_events_by_type(events, "done")[0]
        assert done["text"] == "Hello"
        assert done["message_id"] == start["message_id"]  # stable across the reply

    def test_session_history_updated(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["hi"])
        _chat(client)

        history = client.get("/api/session").json()["history"]
        assert history == [
            {"role": "user", "content": "hello there", "persona": None},
            {"role": "assistant", "content": "hi", "persona": "Alex"},
        ]

    def test_generated_user_message_id_when_absent(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["hi"])
        events = _chat(client)  # no message_id in the request

        start = events[0]
        assert uuid.UUID(start["user_message_id"])  # valid UUID, server-generated

    def test_user_message_persisted_to_requested_room(self, client, monkeypatch, persistence_root):
        _stub_stream(monkeypatch, ["hi"])
        _chat(client, chat_room="TNG")

        from app.persistence import load_history

        messages = load_history("TNG")
        assert [m["sender"] for m in messages] == ["USER", "Alex"]
        assert messages[0]["text"] == "hello there"
        # Session room tracker follows the request.
        assert client.get("/api/session").json()["current_room"] == "TNG"


# ---------------------------------------------------------------------------
# Persona selection
# ---------------------------------------------------------------------------

class TestPersonaSelection:
    def test_explicit_persona_used_directly(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["hi"])
        events = _chat(client, who_answers="Luna")
        assert events[0]["persona"] == "Luna"

    def test_random_mode_picks_from_room(self, client, monkeypatch):
        _patch_chatrooms(monkeypatch, [ChatRoom(name="Solo", persona_names=["Luna"])])
        _stub_stream(monkeypatch, ["hi"])

        events = _chat(client, who_answers="random", chat_room="Solo")

        # "Solo" only contains Luna, so "random" is deterministic here.
        assert events[0]["persona"] == "Luna"

    def test_explicit_persona_not_in_room_falls_back_to_random(self, client, monkeypatch):
        _patch_chatrooms(monkeypatch, [ChatRoom(name="Solo", persona_names=["Luna"])])
        _stub_stream(monkeypatch, ["hi"])

        events = _chat(client, who_answers="Alex", chat_room="Solo")

        assert events[0]["persona"] == "Luna"

    def test_unknown_room_falls_back_to_all_personas(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["hi"])
        events = _chat(client, chat_room="Nowhere")  # not in chatrooms.yaml
        assert events[0]["persona"] == "Alex"  # explicit name still honored

    def test_room_with_no_personas_emits_error_not_reply(self, client, monkeypatch):
        _patch_chatrooms(monkeypatch, [ChatRoom(name="Empty", persona_names=[])])

        def fail(*a, **kw):
            raise AssertionError("LLM must not be called when no persona is eligible")

        monkeypatch.setattr(chat_router, "stream_chat", fail)

        events = _chat(client, chat_room="Empty")

        types = [e["type"] for e in events]
        assert types == ["error", "complete"]
        assert "No eligible personas" in events[0]["message"]
        # The user message must not have been recorded either.
        assert client.get("/api/session").json()["history"] == []

    # -- router mode ---------------------------------------------------------

    def test_router_mode_uses_llm_choice(self, client, monkeypatch):
        _stub_completion(monkeypatch, "Luna")
        _stub_stream(monkeypatch, ["hi"])

        events = _chat(client, who_answers="router")

        assert events[0]["persona"] == "Luna"

    def test_router_mode_strips_quotes_and_whitespace(self, client, monkeypatch):
        _stub_completion(monkeypatch, '  "Alex"  ')
        _stub_stream(monkeypatch, ["hi"])

        events = _chat(client, who_answers="router")

        assert events[0]["persona"] == "Alex"

    def test_router_mode_invalid_choice_falls_back_to_random(self, client, monkeypatch):
        _patch_chatrooms(monkeypatch, [ChatRoom(name="Solo", persona_names=["Luna"])])
        _stub_completion(monkeypatch, "Q")  # not an eligible persona
        _stub_stream(monkeypatch, ["hi"])

        events = _chat(client, who_answers="router", chat_room="Solo")

        assert events[0]["persona"] == "Luna"

    def test_router_mode_llm_failure_falls_back_to_random(self, client, monkeypatch):
        _patch_chatrooms(monkeypatch, [ChatRoom(name="Solo", persona_names=["Luna"])])
        _stub_completion_error(monkeypatch)
        _stub_stream(monkeypatch, ["hi"])

        events = _chat(client, who_answers="router", chat_room="Solo")

        assert events[0]["persona"] == "Luna"


# ---------------------------------------------------------------------------
# Multi-persona replies
# ---------------------------------------------------------------------------

class TestMultiPersonaReplies:
    def test_two_replies_from_two_personas(self, client, monkeypatch):
        _patch_general(monkeypatch, max_persona_replies=2)
        _stub_stream(monkeypatch, ["hi"])

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        starts = sse_events_by_type(events, "start")
        dones = sse_events_by_type(events, "done")
        assert [e["persona"] for e in starts] == ["Alex", "Luna"]
        assert [e["persona"] for e in dones] == ["Alex", "Luna"]
        assert [e["type"] for e in events][-1] == "complete"
        # Each reply gets its own assistant message id.
        assert len({e["message_id"] for e in starts}) == 2

    def test_replies_capped_at_eligible_count(self, client, monkeypatch):
        _patch_general(monkeypatch, max_persona_replies=4)
        _patch_chatrooms(monkeypatch, [ChatRoom(name="Solo", persona_names=["Luna"])])
        _stub_stream(monkeypatch, ["hi"])

        events = _chat(client, who_answers="random", chat_room="Solo")

        starts = sse_events_by_type(events, "start")
        assert [e["persona"] for e in starts] == ["Luna"]  # only one persona available

    def test_second_reply_sees_first_reply_in_history(self, client, monkeypatch):
        _patch_general(monkeypatch, max_persona_replies=2)
        seen_contexts = []

        async def capturing_stream(messages, max_tokens=None, stop=None):
            seen_contexts.append(
                [(m["role"], m["content"]) for m in messages if m["role"] != "system"])
            yield {"type": "token", "token": "hi"}
            yield {"type": "finish", "reason": "stop"}

        monkeypatch.setattr(chat_router, "stream_chat", capturing_stream)

        _chat(client, who_answers="Alex", chat_room="TNG")

        assert len(seen_contexts) == 2
        # The first reply only saw the user's message...
        assert seen_contexts[0] == [("user", "[User]: hello there")]
        # ...the second also saw Alex's answer, reformatted as a prefixed
        # "user" turn (another persona's words must not look like its own).
        assert seen_contexts[1] == [
            ("user", "[User]: hello there"),
            ("user", "[Alex]: hi"),
        ]


# ---------------------------------------------------------------------------
# Echo chamber
# ---------------------------------------------------------------------------

class TestEchoChamber:
    """Driven by the left-panel checkbox, which is UI state — the flag
    arrives on the request, and no room stores it."""

    def _no_llm(self, monkeypatch):
        def fail(*a, **kw):
            raise AssertionError("echo chamber must bypass the LLM entirely")
        monkeypatch.setattr(chat_router, "stream_chat", fail)

    def test_echoes_user_message_verbatim_without_llm(self, client, monkeypatch):
        _patch_general(monkeypatch, max_persona_replies=4)  # must be overridden to 1
        self._no_llm(monkeypatch)

        events = _chat(client, who_answers="Alex", chat_room="TNG", echo=True)

        assert [t["token"] for t in sse_events_by_type(events, "token")] == ["hello there"]
        assert sse_events_by_type(events, "done")[0]["text"] == "hello there"
        # Exactly one persona responds, even though max_persona_replies is 4.
        assert [e["persona"] for e in sse_events_by_type(events, "start")] == ["Alex"]

    def test_it_works_in_the_default_room_too(self, client, monkeypatch):
        # It is not room config, so "default" — which cannot hold settings —
        # is no longer a special case.
        self._no_llm(monkeypatch)
        events = _chat(client, who_answers="Alex", chat_room="default", echo=True)
        assert sse_events_by_type(events, "done")[0]["text"] == "hello there"

    def test_unticked_is_the_default(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["A real answer."])
        events = _chat(client, who_answers="Alex", chat_room="TNG")
        assert sse_events_by_type(events, "done")[0]["text"] == "A real answer."


# ---------------------------------------------------------------------------
# Tool calls (agentic persona)
# ---------------------------------------------------------------------------

class TestToolCalls:
    def _tool_persona_cache(self, monkeypatch):
        config = make_personas()
        config.personas.append(
            Persona(name="ToolUser", system_prompt="You use tools.",
                    router_hints="tools", allow_tool_calls=True))
        _patch_personas(monkeypatch, config)

    def test_tool_call_event_emitted_when_enabled(self, client, monkeypatch):
        self._tool_persona_cache(monkeypatch)
        _stub_tools(monkeypatch, [
            _tool_call_event(),
            {"type": "token", "token": "It is "},
            {"type": "token", "token": "noon."},
        ])

        events = _chat(client, who_answers="ToolUser")

        tool_events = sse_events_by_type(events, "tool_call")
        assert len(tool_events) == 1
        tool_event = tool_events[0]
        assert tool_event["persona"] == "ToolUser"
        assert tool_event["tool_name"] == "get_time"
        assert tool_event["arguments"] == {"zone": "utc"}
        assert tool_event["result"] == "It is noon."
        assert tool_event["failed"] is False

        done = sse_events_by_type(events, "done")[0]
        assert done["text"] == "It is noon."

    def test_failed_tool_call_flag_survives_to_sse(self, client, monkeypatch):
        self._tool_persona_cache(monkeypatch)
        _stub_tools(monkeypatch, [
            _tool_call_event(result="Error: connection refused", failed=True),
            {"type": "token", "token": "sorry"},
        ])

        events = _chat(client, who_answers="ToolUser")

        tool_event = sse_events_by_type(events, "tool_call")[0]
        assert tool_event["failed"] is True
        assert tool_event["result"] == "Error: connection refused"

    def test_tool_events_suppressed_when_show_tool_calls_false(self, client, monkeypatch):
        self._tool_persona_cache(monkeypatch)
        _patch_general(monkeypatch, show_tool_calls=False)
        _stub_tools(monkeypatch, [
            _tool_call_event(),
            {"type": "token", "token": "noon"},
        ])

        events = _chat(client, who_answers="ToolUser")

        assert sse_events_by_type(events, "tool_call") == []
        # The reply itself still streams and completes.
        assert sse_events_by_type(events, "done")[0]["text"] == "noon"


# ---------------------------------------------------------------------------
# LLM failures mid-stream
# ---------------------------------------------------------------------------

class TestStreamErrors:
    def test_error_after_partial_tokens_terminates_stream(self, client, monkeypatch):
        _stub_stream_error_after(monkeypatch, ["par"])

        events = _chat(client)

        types = [e["type"] for e in events]
        assert types == ["start", "token", "error"]
        assert events[-1]["message"] == "boom"
        # No done/complete after a mid-stream failure.
        assert "done" not in types
        assert "complete" not in types

    def test_partial_reply_is_not_persisted(self, client, monkeypatch):
        _stub_stream_error_after(monkeypatch, ["par"])
        _chat(client)

        from app.persistence import load_history

        messages = load_history("default")
        # Only the user message landed; the assistant row never did.
        assert [m["sender"] for m in messages] == ["USER"]


# ---------------------------------------------------------------------------
# Room preamble, response length, and the reply guard
# ---------------------------------------------------------------------------

def _capture(monkeypatch, tokens=("ok",), finish_reason="stop"):
    """Stub the stream and capture the call it was made with."""
    calls = []
    _stub_stream(monkeypatch, list(tokens), finish_reason=finish_reason, capture=calls)
    return calls


def _system_prompt(call):
    return next(m["content"] for m in call["messages"] if m["role"] == "system")


class TestRoomPreamble:
    def test_preamble_follows_the_persona_system_prompt(self, client, monkeypatch):
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="TNG")

        system = _system_prompt(calls[0])
        # The persona's own prompt still leads; the preamble is appended.
        assert system.startswith("You are Alex, a friendly assistant.")
        assert 'in a group chat called "TNG"' in system

    def test_preamble_lists_the_other_personas_in_the_room(self, client, monkeypatch):
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="TNG")

        system = _system_prompt(calls[0])
        # The roster is what makes "never invent a character" enforceable.
        assert "Luna (A philosophical poet)" in system
        assert "There is nobody else." in system
        # The speaker is not listed among the others.
        assert "Alex (A friendly assistant)" not in system

    def test_preamble_states_the_three_prohibitions(self, client, monkeypatch):
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="TNG")

        system = _system_prompt(calls[0])
        assert "Never invent a new character" in system
        assert "Never continue, complete, or rewrite someone else's message" in system
        assert "Write only as Alex" in system

    def test_solo_room_says_so_rather_than_listing_nobody(self, client, monkeypatch):
        _patch_chatrooms(monkeypatch, [ChatRoom(name="Solo", persona_names=["Luna"])])
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Luna", chat_room="Solo")

        assert "You are the only one here, besides the user." in _system_prompt(calls[0])

    def test_an_echoed_message_gets_no_preamble_or_llm_call(self, client, monkeypatch):
        calls = _capture(monkeypatch)
        events = _chat(client, who_answers="Alex", chat_room="TNG", echo=True)

        assert calls == []  # the LLM is bypassed entirely
        assert sse_events_by_type(events, "done")[0]["text"] == "hello there"


class TestTypicalLength:
    def test_length_line_reflects_the_room_tier(self, client, monkeypatch):
        _patch_chatrooms(
            monkeypatch,
            [ChatRoom(name="Terse", persona_names=["Alex"],
                      typical_length=TypicalLength.TERSE)],
        )
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Terse")

        system = _system_prompt(calls[0])
        assert "a few words" in system
        assert "~4 words" in system
        # The register is stated outright — this is chat, not prose.
        assert "This is a chat room, not an essay" in system
        # The escape hatch must survive: typical is a target, not a ceiling.
        assert "Go longer only when the thought genuinely needs it" in system

    def test_normal_room_asks_for_a_sentence_or_two(self, client, monkeypatch):
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="TNG")

        system = _system_prompt(calls[0])
        assert "a sentence or two" in system
        assert "~20 words" in system

    def test_persona_bias_shifts_within_the_rooms_scale(self, client, monkeypatch):
        # Relative, not absolute: "shorter" in a DETAILED room means NORMAL,
        # not the shortest tier there is.
        _patch_personas(monkeypatch, PersonasConfig(personas=[
            Persona(name="Alex", description="", system_prompt="You are Alex.",
                    router_hints="x", length_bias=LengthBias.SHORTER),
        ]))
        _patch_chatrooms(
            monkeypatch,
            [ChatRoom(name="Long", persona_names=["Alex"],
                      typical_length=TypicalLength.DETAILED)],
        )
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Long")

        assert "a sentence or two" in _system_prompt(calls[0])

    def test_the_same_bias_lands_differently_in_a_shorter_room(self, client, monkeypatch):
        _patch_personas(monkeypatch, PersonasConfig(personas=[
            Persona(name="Alex", description="", system_prompt="You are Alex.",
                    router_hints="x", length_bias=LengthBias.SHORTER),
        ]))
        _patch_chatrooms(
            monkeypatch,
            [ChatRoom(name="Short", persona_names=["Alex"],
                      typical_length=TypicalLength.BRIEF)],
        )
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Short")

        assert "a few words" in _system_prompt(calls[0])

    def test_default_room_uses_the_global_tier(self, client, monkeypatch):
        # "default" has no chatrooms.yaml entry, so it has nowhere to store
        # an override and must fall back to general.typical_length.
        _patch_general(monkeypatch, typical_length=TypicalLength.BRIEF)
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="default")

        assert "one short sentence" in _system_prompt(calls[0])

    def test_unrestricted_omits_the_length_line_entirely(self, client, monkeypatch):
        _patch_general(monkeypatch, typical_length=TypicalLength.UNRESTRICTED)
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="default")

        assert "Aim for about" not in _system_prompt(calls[0])

    def test_derived_cap_is_passed_and_stays_under_the_configured_ceiling(
        self, client, monkeypatch
    ):
        _patch_general(monkeypatch, typical_length=TypicalLength.NORMAL)
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="default")

        # Far above the ~20 word target, and below llm.max_tokens (1024):
        # a runaway guard, not a style control.
        assert calls[0]["max_tokens"] == 256

    def test_stop_sequences_cover_the_other_personas_and_the_human(self, client, monkeypatch):
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="TNG")

        # The human is a speaker too: a persona answering *as the user* is
        # the same failure as answering as another persona.
        assert calls[0]["stop"] == ["\nLuna:", "\n[Luna]:", "\nUser:", "\n[User]:"]


class TestReplyGuardIntegration:
    def test_reply_is_cut_at_another_speakers_prefix(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["My view.\n", "[Luna]: ", "actually, no"])

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        done = sse_events_by_type(events, "done")[0]
        assert done["text"] == "My view.\n"
        assert "actually, no" not in "".join(
            e["token"] for e in sse_events_by_type(events, "token")
        )

    def test_cut_text_is_what_gets_persisted(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["Mine.\n", "[Marcus]: invented"])
        _chat(client, who_answers="Alex", chat_room="TNG")

        assert [m.content for m in session.history if m.role == "assistant"] == ["Mine.\n"]

    def test_own_name_prefix_is_stripped_before_it_reaches_the_client(
        self, client, monkeypatch
    ):
        _stub_stream(monkeypatch, ["Alex: ", "here is my answer"])

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        tokens = "".join(e["token"] for e in sse_events_by_type(events, "token"))
        # Streamed text, done text, and history must all agree — the frontend
        # never re-renders from done, so a late strip would leave it on screen.
        assert tokens == "here is my answer"
        assert sse_events_by_type(events, "done")[0]["text"] == "here is my answer"


class TestTruncation:
    def test_next_persona_sees_a_truncated_reply_trimmed_to_a_full_sentence(
        self, client, monkeypatch
    ):
        _patch_general(monkeypatch, max_persona_replies=2)
        seen = []

        async def fake_stream(messages, max_tokens=None, stop=None):
            seen.append([(m["role"], m["content"]) for m in messages])
            yield {"type": "token", "token": "One. Two. Three and it stops mid"}
            yield {"type": "finish", "reason": "length"}

        monkeypatch.setattr(chat_router, "stream_chat", fake_stream)
        _chat(client, who_answers="Alex", chat_room="TNG")

        second = seen[1]
        relayed = [c for r, c in second if c.startswith("[Alex]:")]
        # The dangling fragment is what the next persona would otherwise
        # have been invited to complete.
        assert relayed == ["[Alex]: One. Two."]

    def test_untruncated_reply_is_relayed_verbatim(self, client, monkeypatch):
        _patch_general(monkeypatch, max_persona_replies=2)
        seen = []

        async def fake_stream(messages, max_tokens=None, stop=None):
            seen.append([(m["role"], m["content"]) for m in messages])
            yield {"type": "token", "token": "One. Two. Three and it stops mid"}
            yield {"type": "finish", "reason": "stop"}

        monkeypatch.setattr(chat_router, "stream_chat", fake_stream)
        _chat(client, who_answers="Alex", chat_room="TNG")

        relayed = [c for r, c in seen[1] if c.startswith("[Alex]:")]
        assert relayed == ["[Alex]: One. Two. Three and it stops mid"]

    def test_truncation_does_not_change_what_the_user_saw(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["A full thought. And a partial"], finish_reason="length")

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        assert sse_events_by_type(events, "done")[0]["text"] == (
            "A full thought. And a partial"
        )


# ---------------------------------------------------------------------------
# Player profile — the human's own character
# ---------------------------------------------------------------------------

def _profiled_room(monkeypatch, *, require=False, **profile_fields):
    profile = PlayerProfile(**profile_fields)
    _patch_chatrooms(monkeypatch, [
        ChatRoom(
            name="Tavern",
            persona_names=["Alex", "Luna"],
            require_player_profile=require,
            player_profile=profile,
        )
    ])


class TestPlayerProfileInPrompt:
    def test_profile_is_described_to_the_persona(self, client, monkeypatch):
        _profiled_room(
            monkeypatch,
            name="Kira",
            description="A retired thief who owes everyone money.",
            appearance="Short, scarred hands, a patched green coat.",
        )
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Tavern")

        system = _system_prompt(calls[0])
        assert "You are talking with Kira." in system
        assert "Who they are: A retired thief who owes everyone money." in system
        # The "picture", as text.
        assert "What they look like: Short, scarred hands, a patched green coat." in system
        assert "Treat Kira as that character" in system

    def test_named_player_replaces_the_user_in_the_preamble(self, client, monkeypatch):
        _profiled_room(monkeypatch, name="Kira", description="A thief.")
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Tavern")

        system = _system_prompt(calls[0])
        assert "Kira's included" in system
        assert "and Kira. There is nobody else." in system
        assert "You are not Kira. Never speak or write as Kira" in system
        assert "the user" not in system

    def test_appearance_is_optional(self, client, monkeypatch):
        _profiled_room(monkeypatch, name="Kira", description="A thief.")
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Tavern")

        system = _system_prompt(calls[0])
        assert "Who they are: A thief." in system
        assert "What they look like" not in system

    def test_empty_profile_leaves_the_preamble_as_it_was(self, client, monkeypatch):
        _profiled_room(monkeypatch)
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Tavern")

        system = _system_prompt(calls[0])
        assert "You are talking with" not in system
        assert "the user's included" in system
        assert "You are not the user. Never speak or write as the user" in system

    def test_default_room_has_no_profile(self, client, monkeypatch):
        # "default" has no chatrooms.yaml entry to carry one.
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="default")

        assert "You are talking with" not in _system_prompt(calls[0])


class TestPlayerProfileGate:
    def test_required_but_missing_refuses_the_message(self, client, monkeypatch):
        _profiled_room(monkeypatch, require=True)
        calls = _capture(monkeypatch)

        events = _chat(client, who_answers="Alex", chat_room="Tavern")

        types = [e["type"] for e in events]
        assert types == ["error", "complete"]
        assert "character profile" in events[0]["message"]
        assert calls == []  # the LLM is never reached

    def test_refused_message_is_not_recorded(self, client, monkeypatch):
        # Bailing before the user message is added keeps the room's history
        # clean — a refused turn should leave no trace.
        _profiled_room(monkeypatch, require=True)
        _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Tavern")

        assert session.history == []
        from app.persistence import load_history
        assert load_history("Tavern") == []

    def test_half_filled_profile_still_counts_as_missing(self, client, monkeypatch):
        # A name with no description says nothing useful about the character.
        _profiled_room(monkeypatch, require=True, name="Kira")
        events = _chat(client, who_answers="Alex", chat_room="Tavern")

        assert [e["type"] for e in events] == ["error", "complete"]

    def test_complete_profile_lets_the_message_through(self, client, monkeypatch):
        _profiled_room(monkeypatch, require=True, name="Kira", description="A thief.")
        _capture(monkeypatch)

        events = _chat(client, who_answers="Alex", chat_room="Tavern")

        assert [e["type"] for e in events][-1] == "complete"
        assert sse_events_by_type(events, "done")

    def test_profile_without_the_requirement_never_blocks(self, client, monkeypatch):
        _profiled_room(monkeypatch, require=False)
        _capture(monkeypatch)

        events = _chat(client, who_answers="Alex", chat_room="Tavern")

        assert sse_events_by_type(events, "done")


# ---------------------------------------------------------------------------
# Personas must always be themselves
# ---------------------------------------------------------------------------

class TestNeverSpeakAsTheUser:
    """One question, several replies, and one of them answers *as the user*.

    Reported in a six-persona room: persona 2 replied as the human, persona
    3 replied as itself, persona 4 replied as the human to persona 3. The
    cause was structural — the human was the only untagged voice in the
    transcript, so "untagged text in the user role" was the sole example a
    late responder had of what a turn looks like.
    """

    def test_every_other_voice_is_tagged_for_a_late_responder(self, client, monkeypatch):
        _patch_general(monkeypatch, max_persona_replies=2)
        seen = []

        async def capturing(messages, max_tokens=None, stop=None):
            seen.append([m["content"] for m in messages if m["role"] != "system"])
            yield {"type": "token", "token": "ok"}
            yield {"type": "finish", "reason": "stop"}

        monkeypatch.setattr(chat_router, "stream_chat", capturing)
        _chat(client, who_answers="Alex", chat_room="TNG")

        # Nothing the second responder sees is untagged: the only untagged
        # voice in its payload would be its own assistant turns, and it has
        # none yet.
        assert all(c.startswith("[") for c in seen[1])

    def test_the_human_is_named_in_the_stop_sequences(self, client, monkeypatch):
        _profiled_room(monkeypatch, name="Kira", description="A thief.")
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Tavern")

        assert "\nKira:" in calls[0]["stop"]
        assert "\n[Kira]:" in calls[0]["stop"]

    def test_a_reply_opening_as_the_user_is_cut(self, client, monkeypatch):
        _profiled_room(monkeypatch, name="Kira", description="A thief.")
        _stub_stream(monkeypatch, ["My view.\n", "[Kira]: ", "and here is what I say back"])

        events = _chat(client, who_answers="Alex", chat_room="Tavern")

        assert sse_events_by_type(events, "done")[0]["text"] == "My view.\n"

    def test_an_unprofiled_room_still_guards_the_generic_user(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["Sure.\n", "User: what about tomorrow?"])

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        assert sse_events_by_type(events, "done")[0]["text"] == "Sure.\n"

    def test_a_reply_that_is_only_the_users_voice_is_dropped_entirely(
        self, client, monkeypatch
    ):
        # Nothing of the persona's own survives the cut, so there is no turn
        # to record: an empty bubble and an empty "[Alex]: " line in history
        # would both be worse than nothing.
        _stub_stream(monkeypatch, ["User: so what should we do?"])

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        assert sse_events_by_type(events, "done") == []
        assert [m for m in session.history if m.role == "assistant"] == []
        assert [e["type"] for e in events][-1] == "complete"

    def test_a_dropped_reply_does_not_use_up_a_reply_slot(self, client, monkeypatch):
        # One reply was asked for, the first persona was cut, so the next
        # persona is tried rather than the turn ending empty-handed.
        _patch_general(monkeypatch, max_persona_replies=1)
        calls = {"n": 0}

        async def alternating(messages, max_tokens=None, stop=None):
            calls["n"] += 1
            token = "User: hmm" if calls["n"] == 1 else "A real answer."
            yield {"type": "token", "token": token}
            yield {"type": "finish", "reason": "stop"}

        monkeypatch.setattr(chat_router, "stream_chat", alternating)
        events = _chat(client, who_answers="Alex", chat_room="TNG")

        assert [d["text"] for d in sse_events_by_type(events, "done")] == ["A real answer."]

    def test_retries_after_a_cut_are_bounded(self, client, monkeypatch):
        # Every persona is cut, so the turn re-rolls until the attempt
        # budget runs out rather than looping forever.
        _patch_general(monkeypatch, max_persona_replies=2)
        _stub_stream(monkeypatch, ["User: nope"])

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        starts = sse_events_by_type(events, "start")
        assert len(starts) == 2 + chat_router.MAX_CUT_RETRIES
        assert set(e["persona"] for e in starts) <= {"Alex", "Luna"}

    def test_a_persona_that_replied_is_never_asked_again(self, client, monkeypatch):
        _patch_general(monkeypatch, max_persona_replies=2)
        _stub_stream(monkeypatch, ["A real answer."])

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        who = [e["persona"] for e in sse_events_by_type(events, "done")]
        assert sorted(who) == ["Alex", "Luna"]
        assert len(who) == len(set(who))

    def test_everyone_cut_reports_it_instead_of_going_silent(self, client, monkeypatch):
        # A turn that ends with a start event and nothing after it is
        # indistinguishable from the app hanging.
        _stub_stream(monkeypatch, ["User: nope"])

        events = _chat(client, who_answers="Alex", chat_room="TNG")

        assert sse_events_by_type(events, "done") == []
        errors = sse_events_by_type(events, "error")
        assert len(errors) == 1
        assert "in their own voice" in errors[0]["message"]
        assert [e["type"] for e in events][-1] == "complete"

    def test_a_normal_turn_reports_no_error(self, client, monkeypatch):
        _stub_stream(monkeypatch, ["A real answer."])
        events = _chat(client, who_answers="Alex", chat_room="TNG")
        assert sse_events_by_type(events, "error") == []

    def test_the_turn_is_stated_at_the_end_of_the_preamble(self, client, monkeypatch):
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="TNG")

        # A late responder has no assistant turn in context to anchor its
        # own voice, so the system message says whose turn it is outright.
        assert _system_prompt(calls[0]).rstrip().endswith(
            "It is Alex's turn. Reply as Alex, and no one else."
        )


class TestReplyCountIsReached:
    """max_persona_replies must actually be reachable.

    Two separate reasons it silently was not: a cut reply used up the only
    remaining persona, and the global setting is capped by how many personas
    the room has.
    """

    def _room_of(self, monkeypatch, size, max_replies):
        names = [f"P{i}" for i in range(size)]
        _patch_personas(monkeypatch, PersonasConfig(personas=[
            Persona(name=n, description=n, system_prompt=f"You are {n}.", router_hints="x")
            for n in names
        ]))
        _patch_chatrooms(monkeypatch, [ChatRoom(name="Big", persona_names=names)])
        settings = make_settings()
        settings.general = GeneralConfig(max_persona_replies=max_replies)
        monkeypatch.setattr(app_config, "_settings_cache", settings)
        return names

    def test_six_replies_when_the_room_is_big_enough(self, client, monkeypatch):
        self._room_of(monkeypatch, size=8, max_replies=6)
        _stub_stream(monkeypatch, ["fine"])

        events = _chat(client, who_answers="P0", chat_room="Big")

        assert len(sse_events_by_type(events, "done")) == 6

    def test_a_cut_reply_is_replaced_even_at_full_room_size(self, client, monkeypatch):
        # The room has exactly as many personas as replies requested, so
        # without a retry budget one cut would make 6 unreachable.
        self._room_of(monkeypatch, size=6, max_replies=6)
        calls = {"n": 0}

        async def one_cut(messages, max_tokens=None, stop=None):
            calls["n"] += 1
            token = "User: nope" if calls["n"] == 2 else "fine"
            yield {"type": "token", "token": token}
            yield {"type": "finish", "reason": "stop"}

        monkeypatch.setattr(chat_router, "stream_chat", one_cut)
        events = _chat(client, who_answers="P0", chat_room="Big")

        assert len(sse_events_by_type(events, "done")) == 6

    def test_room_smaller_than_the_setting_caps_and_says_so(self, client, monkeypatch, caplog):
        # Not a bug, but the commonest reason the setting looks ignored —
        # so it is logged rather than left to guesswork.
        self._room_of(monkeypatch, size=4, max_replies=6)
        _stub_stream(monkeypatch, ["fine"])

        with caplog.at_level("INFO"):
            events = _chat(client, who_answers="P0", chat_room="Big")

        assert len(sse_events_by_type(events, "done")) == 4
        assert "has 4 persona(s), so at most 4 can reply" in caplog.text

    def test_no_cap_log_when_the_room_is_large_enough(self, client, monkeypatch, caplog):
        self._room_of(monkeypatch, size=6, max_replies=6)
        _stub_stream(monkeypatch, ["fine"])

        with caplog.at_level("INFO"):
            _chat(client, who_answers="P0", chat_room="Big")

        assert "can reply" not in caplog.text


# ---------------------------------------------------------------------------
# Suggested player message
# ---------------------------------------------------------------------------

class TestSuggestReply:
    """Drafting the player's own next message.

    The deliberate inverse of the reply guard's job: here the LLM is asked
    to write as the player, because the player asked it to and the result
    lands in their input box to edit.
    """

    def _stub(self, monkeypatch, result="Aye, that'll be tuppence."):
        seen = []

        async def fake(messages, max_tokens=64, temperature=None):
            seen.append({"messages": messages, "max_tokens": max_tokens,
                         "temperature": temperature})
            return result

        monkeypatch.setattr(chat_router, "chat_completion", fake)
        return seen

    def _suggest(self, client, room="TNG"):
        resp = client.post("/api/chat/suggest", json={"chat_room": room})
        assert resp.status_code == 200, resp.text
        return resp.json()["text"]

    def test_returns_a_draft(self, client, monkeypatch):
        self._stub(monkeypatch)
        assert self._suggest(client) == "Aye, that'll be tuppence."

    def test_the_players_own_messages_are_the_voice_sample(self, client, monkeypatch):
        seen = self._stub(monkeypatch)
        session.add_user_message_no_persist("aye, what'll it be then")
        session.add_assistant_message_no_persist("A stout.", "Alex")
        session.add_user_message_no_persist("comin right up")

        self._suggest(client)

        prompt = seen[0]["messages"][0]["content"]
        assert "How you write" in prompt
        assert "- aye, what'll it be then" in prompt
        assert "- comin right up" in prompt
        # A persona's line is context, not a voice sample.
        assert "- A stout." not in prompt

    def test_the_conversation_is_included_for_context(self, client, monkeypatch):
        seen = self._stub(monkeypatch)
        session.add_user_message_no_persist("what's the news?")
        session.add_assistant_message_no_persist("Rain, mostly.", "Alex")

        self._suggest(client)

        prompt = seen[0]["messages"][0]["content"]
        assert "[Alex]: Rain, mostly." in prompt

    def test_the_character_description_is_a_section_of_its_own(self, client, monkeypatch):
        # It carries *what* to say, so it gets the same structural weight as
        # the voice sample rather than a passing mention.
        _profiled_room(monkeypatch, name="Kira", description="A retired thief.",
                       appearance="A patched green coat.")
        seen = self._stub(monkeypatch)

        self._suggest(client, room="Tavern")

        prompt = seen[0]["messages"][0]["content"]
        assert "Who you are:\nA retired thief." in prompt
        assert "What you look like:\nA patched green coat." in prompt

    def test_the_description_carries_an_instruction_to_act_on_it(self, client, monkeypatch):
        # Without this the description was decoration: the draft sounded
        # right and behaved like nobody in particular.
        _profiled_room(monkeypatch, name="Kira", description="A retired thief.")
        seen = self._stub(monkeypatch)

        self._suggest(client, room="Tavern")

        prompt = seen[0]["messages"][0]["content"]
        assert "Stay in character" in prompt
        assert "should follow from who you are" in prompt

    def test_no_stay_in_character_line_without_a_description(self, client, monkeypatch):
        seen = self._stub(monkeypatch)
        self._suggest(client)
        assert "Stay in character" not in seen[0]["messages"][0]["content"]

    def test_the_persona_facing_profile_block_is_not_reused(self, client, monkeypatch):
        # _player_lines() ends "never write their lines for them" — the exact
        # opposite of what is being asked here.
        _profiled_room(monkeypatch, name="Kira", description="A thief.",
                       appearance="A green coat.")
        seen = self._stub(monkeypatch)

        self._suggest(client, room="Tavern")

        prompt = seen[0]["messages"][0]["content"]
        assert "Never write their lines for them" not in prompt
        assert "You are talking with" not in prompt

    def test_it_writes_in_the_second_person_as_the_character(self, client, monkeypatch):
        _profiled_room(monkeypatch, name="Kira", description="A thief.")
        seen = self._stub(monkeypatch)

        self._suggest(client, room="Tavern")

        prompt = seen[0]["messages"][0]["content"]
        assert prompt.startswith('You are Kira, in a group chat called "Tavern".')
        # No third-person framing left over — the model is the character,
        # not a writer working on their behalf.
        assert "next message for Kira" not in prompt

    def test_an_unnamed_player_is_described_rather_than_called_user(self, client, monkeypatch):
        seen = self._stub(monkeypatch)
        self._suggest(client)
        prompt = seen[0]["messages"][0]["content"]
        assert "You are the human in a group chat" in prompt
        assert 'shown in the transcript as "User"' in prompt

    def test_prose_uses_the_configured_temperature_not_the_routers(self, client, monkeypatch):
        # chat_completion defaults to 0.1, which is right for picking a name
        # and wrong for writing a line.
        seen = self._stub(monkeypatch)
        self._suggest(client)
        assert seen[0]["temperature"] == 0.8
        assert seen[0]["max_tokens"] == 256

    def test_a_name_prefix_is_stripped_from_the_draft(self, client, monkeypatch):
        self._stub(monkeypatch, result="User: aye, right you are")
        assert self._suggest(client) == "aye, right you are"

    def test_a_draft_that_runs_into_a_persona_is_cut(self, client, monkeypatch):
        self._stub(monkeypatch, result="Right you are.\n[Luna]: And I would add...")
        assert self._suggest(client) == "Right you are."

    def test_works_with_no_history_at_all(self, client, monkeypatch):
        seen = self._stub(monkeypatch)
        assert self._suggest(client) == "Aye, that'll be tuppence."
        assert "Match their voice" not in seen[0]["messages"][0]["content"]

    def test_an_llm_failure_is_reported_not_swallowed(self, client, monkeypatch):
        self._stub(monkeypatch, result="")
        resp = client.post("/api/chat/suggest", json={"chat_room": "TNG"})
        assert resp.status_code == 503
        assert "did not return a suggestion" in resp.json()["detail"]

    def test_nothing_is_sent_or_persisted(self, client, monkeypatch):
        self._stub(monkeypatch)
        self._suggest(client)
        assert session.history == []
        from app.persistence import load_history
        assert load_history("TNG") == []
