"""In-memory session state.

Single-user app means one global session object.
History is a simple list of dicts: {"role": "user"|"assistant", "content": str, "persona": str|None}.
"""

from typing import Dict, List, Optional

from app.models import ChatMessage


class SessionManager:
    """Manages the single active chat session."""

    def __init__(self):
        self._history: List[ChatMessage] = []
        self._active_personas: List[str] = []

    # -- Public API ----------------------------------------------------------

    def reset(self):
        """Wipe history and active personas. Called on 'New Chat'."""
        self._history.clear()
        self._active_personas.clear()

    def set_active_personas(self, names: List[str]):
        """Replace the active persona list."""
        self._active_personas = list(names)

    @property
    def active_personas(self) -> List[str]:
        return list(self._active_personas)

    @property
    def history(self) -> List[ChatMessage]:
        return list(self._history)

    def add_user_message(self, content: str):
        """Append a user message to history."""
        self._history.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str, persona: str):
        """Append an assistant message to history."""
        self._history.append(ChatMessage(role="assistant", content=content, persona=persona))

    def build_llm_messages(
        self,
        system_prompt: str,
        responding_persona: str,
        max_turns: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages list for an LLM call.

        - System message with the responding persona's system prompt.
        - Conversation history, reformatted so:
            * User messages keep role "user".
            * This persona's messages keep role "assistant".
            * Other personas' messages become "assistant" with prefix "[Name]: <text>".
        - Optionally limited to the last *max_turns* history entries.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        history_slice = self._history
        if max_turns is not None:
            history_slice = self._history[-max_turns:]

        for msg in history_slice:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                if msg.persona == responding_persona:
                    messages.append({"role": "assistant", "content": msg.content})
                else:
                    # Another persona spoke — prefix it so the model knows
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"[{msg.persona}]: {msg.content}",
                        }
                    )

        return messages

    def get_history_dicts(self) -> List[dict]:
        """Return history as plain dicts for JSON serialization."""
        return [m.model_dump() for m in self._history]


# Singleton instance — single-user app, one session to rule them all.
session = SessionManager()
