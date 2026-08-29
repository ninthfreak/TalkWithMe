"""In-memory session state with disk persistence.

Single-user app means one global session object.
History is a simple list of dicts: {"role": "user"|"assistant", "content": str, "persona": str|None}.

Messages are automatically persisted to disk per chat room as they arrive.
"""

import logging
from typing import Dict, List, Optional

from app.models import ChatMessage
from app import persistence

logger = logging.getLogger(__name__)


# Sentence terminators used to trim a truncated reply back to a clean stop.
_SENTENCE_END = ".!?\u2026"

# Safety valve for the context window, in characters (~3k tokens). The
# exchange count is the real control; this only stops a very wide room
# (six personas answering at length) from overflowing a small local model.
# Trimming here is logged, because context vanishing silently is the whole
# bug this module's windowing exists to avoid.
_CONTEXT_CHAR_BUDGET = 12000


def _exchange_starts(messages: List[ChatMessage]) -> List[int]:
    """Indices where the human speaks — each one opens a new exchange."""
    return [i for i, m in enumerate(messages) if m.role == "user"]


def _window_chars(messages: List[ChatMessage]) -> int:
    return sum(len(m.content) for m in messages)


def recent_exchanges(
    history: List[ChatMessage],
    max_exchanges: Optional[int],
    char_budget: int = _CONTEXT_CHAR_BUDGET,
) -> List[ChatMessage]:
    """The last *max_exchanges* exchanges, newest last.

    An **exchange** is one human message plus every persona reply that
    followed it, so the question and the answers it prompted are kept or
    dropped together.

    This used to be a flat tail slice of the last N *messages*, which is a
    different thing entirely once more than one persona replies: with six
    personas an exchange is seven messages, so a setting of 6 could not
    hold even one. Asking a room to guess something and then revealing the
    answer left every persona seeing a handful of guesses and a bare "it
    was an otter", with the question that prompted them already cut off —
    the reveal read as a statement out of nowhere.
    """
    if max_exchanges is None:
        window = list(history)
    else:
        starts = _exchange_starts(history)
        # Fewer exchanges than asked for: keep everything, including any
        # replies that precede the first human message.
        window = (
            list(history[starts[-max_exchanges]:])
            if len(starts) > max_exchanges
            else list(history)
        )

    if _window_chars(window) <= char_budget:
        return window

    # Over budget: shed whole exchanges from the oldest end first, so what
    # survives is always complete.
    dropped_exchanges = 0
    while _window_chars(window) > char_budget:
        starts = _exchange_starts(window)
        if len(starts) <= 1:
            break
        window = window[starts[1]:]
        dropped_exchanges += 1

    # A single exchange still over budget — a wide room answering at length.
    # Shed its oldest replies, never the human message anchoring them.
    dropped_replies = 0
    while _window_chars(window) > char_budget and len(window) > 2:
        del window[1]
        dropped_replies += 1

    logger.info(
        "Context over %d chars: dropped %d older exchange(s) and %d reply/replies",
        char_budget, dropped_exchanges, dropped_replies,
    )
    return window


def _other_persona_text(msg: ChatMessage) -> str:
    """Render another persona's message for the responding persona's prompt.

    A reply cut off at max_tokens ends mid-sentence, and an unfinished
    sentence in a prompt is a completion cue — this is exactly how one
    persona ends up finishing another's thought in the wrong voice. So a
    truncated message is trimmed back to its last complete sentence, or, if
    it has none, labelled so the model can see it is not a prompt to finish.
    """
    if not msg.truncated:
        return msg.content

    trimmed = msg.content.rstrip()
    cut = max(trimmed.rfind(c) for c in _SENTENCE_END)
    if cut != -1:
        return trimmed[: cut + 1]
    return f"{trimmed} (message was cut off)"


class SessionManager:
    """Manages the single active chat session."""

    def __init__(self):
        self._history: List[ChatMessage] = []
        self._active_personas: List[str] = []
        self._current_room: str = "default"

    # -- Public API ----------------------------------------------------------

    @property
    def current_room(self) -> str:
        return self._current_room

    def set_current_room(self, room_name: str):
        """Switch the active chat room.

        Messages are persisted individually as they arrive, so no bulk
        flush is needed here. Just updates the room tracker.
        """
        if room_name == self._current_room:
            return
        self._current_room = room_name

    def reset(self):
        """Wipe history, clear persistence for current room, and reset personas.
        Called on 'New Chat'.
        """
        persistence.clear_room(self._current_room)
        self._history.clear()
        self._active_personas.clear()

    def load_room(self, room_name: str):
        """Load persisted history for a room into the active session.

        Clears any existing in-memory history first, then populates from disk.
        Uses no-persist variants since messages are already on disk.
        """
        self._history.clear()
        self._current_room = room_name
        persisted = persistence.load_history(room_name)
        for msg in persisted:
            if msg["sender"] == "USER":
                self.add_user_message_no_persist(msg["text"])
            else:
                self.add_assistant_message_no_persist(msg["text"], msg["sender"])

    def set_active_personas(self, names: List[str]):
        """Replace the active persona list."""
        self._active_personas = list(names)

    @property
    def active_personas(self) -> List[str]:
        return list(self._active_personas)

    @property
    def history(self) -> List[ChatMessage]:
        return list(self._history)

    def add_user_message(self, content: str, message_id: str):
        """Append a user message to history and persist it."""
        self._history.append(ChatMessage(role="user", content=content))
        persistence.persist_message(self._current_room, self._history[-1], message_id)

    def add_assistant_message(
        self, content: str, persona: str, message_id: str, truncated: bool = False
    ):
        """Append an assistant message to history and persist it.

        *truncated* marks a reply the LLM cut off at max_tokens. It changes
        only what later personas see (see build_llm_messages); the persisted
        text is exactly what the user saw.
        """
        self._history.append(
            ChatMessage(
                role="assistant", content=content, persona=persona, truncated=truncated
            )
        )
        persistence.persist_message(self._current_room, self._history[-1], message_id)

    def add_user_message_no_persist(self, content: str):
        """Append a user message to history without persisting.

        Used when loading from disk (messages are already persisted).
        """
        self._history.append(ChatMessage(role="user", content=content))

    def add_assistant_message_no_persist(self, content: str, persona: str):
        """Append an assistant message to history without persisting.

        Used when loading from disk (messages are already persisted).
        """
        self._history.append(ChatMessage(role="assistant", content=content, persona=persona))

    def build_llm_messages(
        self,
        system_prompt: str,
        responding_persona: str,
        max_turns_for_context: Optional[int] = None,
        room_preamble: Optional[str] = None,
        user_label: str = "User",
    ) -> List[Dict[str, str]]:
        """Build the messages list for an LLM call.

        - System message with the responding persona's system prompt,
          followed by *room_preamble* when given (who is in the room and the
          rules of the room — built by the chat router, which owns the chat
          room config; this module deliberately imports none of it).
        - Conversation history, reformatted so:
            * The human's messages become "user" with prefix "[user_label]: ".
            * This persona's messages keep role "assistant", untagged.
            * Other personas' messages become "user" with prefix "[Name]: ".
        - Limited to the last *max_turns_for_context* **exchanges** (a human
          message plus the replies it drew), never a flat slice of messages.

        **Every voice but this persona's own is tagged.** The human used to
        be the one untagged speaker in the transcript, which made "untagged
        text in the user role" the model's only example of how the human
        writes — so a persona answering third or fourth, having never seen
        an assistant turn in the conversation, would produce a line in the
        user's voice as a perfectly consistent continuation. Tagging the
        human too leaves exactly one untagged voice: this persona's own.
        """
        system_content = system_prompt
        if room_preamble:
            system_content = f"{system_prompt}\n\n{room_preamble}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        history_slice = recent_exchanges(self._history, max_turns_for_context)

        for msg in history_slice:
            if msg.role == "user":
                messages.append(
                    {"role": "user", "content": f"[{user_label}]: {msg.content}"}
                )
            elif msg.role == "assistant":
                if msg.persona == responding_persona:
                    messages.append({"role": "assistant", "content": msg.content})
                else:
                    # Another persona spoke — use role "user" to avoid consecutive
                    # assistant messages (which many LLMs reject with 400) and to
                    # prevent the model from treating another persona's words as its own.
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[{msg.persona}]: {_other_persona_text(msg)}",
                        }
                    )

        return messages

    def get_history_dicts(self) -> List[dict]:
        """Return history as plain dicts for JSON serialization.

        `truncated` is excluded: it is internal bookkeeping for prompt
        construction, not part of the session shape the frontend consumes.
        """
        return [m.model_dump(exclude={"truncated"}) for m in self._history]


# Singleton instance — single-user app, one session to rule them all.
session = SessionManager()
