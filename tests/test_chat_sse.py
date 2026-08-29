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
        assert seen_contexts[0] == [("user", "hello there")]
        # ...the second also saw Alex's answer, reformatted as a prefixed
        # "user" turn (another persona's words must not look like its own).
        assert seen_contexts[1] == [("user", "hello there"), ("user", "[Alex]: hi")]


# ---------------------------------------------------------------------------
# Echo chamber
# ---------------------------------------------------------------------------

class TestEchoChamber:
    def test_echoes_user_message_verbatim_without_llm(self, client, monkeypatch):
        _patch_chatrooms(monkeypatch,
                         [ChatRoom(name="Echo", persona_names=["Alex"], echo_chamber=True)])
        _patch_general(monkeypatch, max_persona_replies=4)  # must be overridden to 1

        def fail(*a, **kw):
            raise AssertionError("echo chamber must bypass the LLM entirely")

        monkeypatch.setattr(chat_router, "stream_chat", fail)

        events = _chat(client, who_answers="Alex", chat_room="Echo")

        tokens = sse_events_by_type(events, "token")
        assert [t["token"] for t in tokens] == ["hello there"]
        done = sse_events_by_type(events, "done")[0]
        assert done["text"] == "hello there"
        # Exactly one persona responds, even though max_persona_replies is 4.
        assert [e["persona"] for e in sse_events_by_type(events, "start")] == ["Alex"]


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

    def test_echo_chamber_room_gets_no_preamble_or_llm_call(self, client, monkeypatch):
        _patch_chatrooms(
            monkeypatch,
            [ChatRoom(name="Echo", persona_names=["Alex"], echo_chamber=True)],
        )
        calls = _capture(monkeypatch)
        events = _chat(client, who_answers="Alex", chat_room="Echo")

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
        assert "one or two short sentences" in system
        assert "~25 words" in system
        # The escape hatch must survive: typical is a target, not a ceiling.
        assert "Go longer only when the question genuinely needs it" in system

    def test_persona_tier_overrides_the_room(self, client, monkeypatch):
        _patch_personas(monkeypatch, PersonasConfig(personas=[
            Persona(name="Alex", description="", system_prompt="You are Alex.",
                    router_hints="x", typical_length=TypicalLength.TERSE),
        ]))
        _patch_chatrooms(
            monkeypatch,
            [ChatRoom(name="Long", persona_names=["Alex"],
                      typical_length=TypicalLength.DETAILED)],
        )
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="Long")

        assert "~25 words" in _system_prompt(calls[0])  # persona wins

    def test_default_room_uses_the_global_tier(self, client, monkeypatch):
        # "default" has no chatrooms.yaml entry, so it has nowhere to store
        # an override and must fall back to general.typical_length.
        _patch_general(monkeypatch, typical_length=TypicalLength.BRIEF)
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="default")

        assert "two to four sentences" in _system_prompt(calls[0])

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

        # Well above the ~120 word target, and below llm.max_tokens (1024):
        # a runaway guard, not a style control.
        assert calls[0]["max_tokens"] == 504

    def test_stop_sequences_cover_the_other_personas(self, client, monkeypatch):
        calls = _capture(monkeypatch)
        _chat(client, who_answers="Alex", chat_room="TNG")

        assert calls[0]["stop"] == ["\nLuna:", "\n[Luna]:"]


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
