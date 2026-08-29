"""In-memory session state with disk persistence.

Single-user app means one global session object.
History is a simple list of dicts: {"role": "user"|"assistant", "content": str, "persona": str|None}.

Messages are automatically persisted to disk per chat room as they arrive.
"""

from typing import Dict, List, Optional

from app.models import ChatMessage
from app import persistence


# Sentence terminators used to trim a truncated reply back to a clean stop.
_SENTENCE_END = ".!?\u2026"


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
    ) -> List[Dict[str, str]]:
        """Build the messages list for an LLM call.

        - System message with the responding persona's system prompt,
          followed by *room_preamble* when given (who is in the room and the
          rules of the room — built by the chat router, which owns the chat
          room config; this module deliberately imports none of it).
        - Conversation history, reformatted so:
            * User messages keep role "user".
            * This persona's messages keep role "assistant".
            * Other personas' messages become "user" with prefix "[Name]: <text>".
        - Optionally limited to the last *max_turns_for_context* history entries.
        """
        system_content = system_prompt
        if room_preamble:
            system_content = f"{system_prompt}\n\n{room_preamble}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        history_slice = self._history
        if max_turns_for_context is not None:
            history_slice = self._history[-max_turns_for_context:]

        for msg in history_slice:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
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
