"""Built-in tools — tools owned by this application, not by any MCP server.

Unlike MCP tools (app/services/tool_registry.py), built-in tools are:

* always available, even when no MCP server is configured;
* registered statically at import time (no discovery, no network, no
  mutable module-level state to reset in tests);
* name-protected: an MCP server that advertises a tool with a built-in
  name loses the conflict (tool_registry skips it with a warning).

Handlers are plain synchronous functions ``(persona, arguments) -> str``
and must return the LLM-facing result string. call_builtin_tool() wraps
them so a raising handler degrades to an "Error:" result instead of
killing the persona's reply stream.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from app.config import DEFAULT_USER_LABEL, AppSettings, Persona, user_label
from app.services import persona_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BuiltinTool:
    """A registered built-in tool and how to run it."""

    spec: dict
    handler: Callable[[Persona, dict], str]
    # Optional per-request gate (e.g. the memory feature's global
    # kill-switch). None means "always available".
    is_available: Optional[Callable[[Persona, AppSettings], bool]] = None


# name -> tool. Populated by register_builtin_tool() at import time;
# intentionally never mutated afterwards.
_BUILTIN_TOOLS: Dict[str, _BuiltinTool] = {}


def register_builtin_tool(
    name: str,
    spec: dict,
    handler: Callable[[Persona, dict], str],
    is_available: Optional[Callable[[Persona, AppSettings], bool]] = None,
) -> None:
    """Register a built-in tool. Called once per tool, at module import."""
    if name in _BUILTIN_TOOLS:
        raise ValueError(f"built-in tool '{name}' is already registered")
    _BUILTIN_TOOLS[name] = _BuiltinTool(spec=spec, handler=handler, is_available=is_available)


def is_builtin_tool(name: str) -> bool:
    """True when an MCP server must NOT claim this tool name."""
    return name in _BUILTIN_TOOLS


def get_builtin_tools() -> List[dict]:
    """Every built-in tool spec, in registration order (OpenAI format)."""
    return [tool.spec for tool in _BUILTIN_TOOLS.values()]


def get_builtin_tools_for(persona: Persona, settings: AppSettings) -> List[dict]:
    """Built-in tool specs available to this persona in this request.

    The caller (chat router) separately gates the ENTIRE tool list on
    persona.allow_tool_calls — no tools of any kind are offered to a
    persona that may not call tools.
    """
    available: List[dict] = []
    for tool in _BUILTIN_TOOLS.values():
        if tool.is_available is None or tool.is_available(persona, settings):
            available.append(tool.spec)
        else:
            # The gate inputs (enable_persona_memories, memory_size) are
            # logged by the chat router for the same request, so a generic
            # "gate failed" line here keeps this module tool-agnostic.
            logger.debug(
                "Persona memory: built-in tool '%s' NOT offered to persona '%s' "
                "(per-request availability gate failed)",
                tool.spec["function"]["name"], persona.name,
            )
    return available


def call_builtin_tool(persona: Persona, tool_name: str, arguments: dict) -> str:
    """Execute a built-in tool; ALWAYS returns a result string.

    The agentic loop (app/services/llm.py) feeds the return value straight
    back to the LLM, so this must never raise: a bug in a built-in handler
    surfaces as an "Error:" result the model can react to, not a dead
    stream. (MCP tools get the same treatment via mcp_client.call_tool.)
    """
    tool = _BUILTIN_TOOLS.get(tool_name)
    if tool is None:
        available = ", ".join(sorted(_BUILTIN_TOOLS)) or "none"
        return f"Error: unknown tool '{tool_name}'. Available built-in tools: {available}"
    logger.debug(
        "Persona memory: built-in tool '%s' invoked for persona '%s' with arguments: %r",
        tool_name, persona.name, arguments,
    )
    try:
        result = tool.handler(persona, arguments)
    except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the stream
        logger.exception("Built-in tool '%s' raised an exception", tool_name)
        return f"Error: the built-in tool '{tool_name}' failed unexpectedly ({exc})"
    if not isinstance(result, str):
        logger.error("Built-in tool '%s' returned a non-string result: %r", tool_name, result)
        return "Error: the built-in tool returned an invalid result"
    logger.debug(
        "Persona memory: built-in tool '%s' result for persona '%s': %s",
        tool_name, persona.name, result,
    )
    return result


# ---------------------------------------------------------------------------
# add_memory (docs/feature_persona_memory.md)
# ---------------------------------------------------------------------------

ADD_MEMORY_NAME = "add_memory"

# The description IS the prompt: the LLM has nothing else telling it when
# or how to save memories, so every behavioural rule from the spec lives
# in here. Keep it in sync with docs/feature_persona_memory.md.
ADD_MEMORY_SPEC = {
    "type": "function",
    "function": {
        "name": ADD_MEMORY_NAME,
        "description": (
            "Save something you have learned about one of the people in this room, "
            "so you still know it next time you meet them. "
            "Submit at most ONE memory per turn, and only when somebody has revealed "
            "something worth remembering about themselves — what they want, what they "
            "fear, something that happened to them, a strong opinion, or something "
            "they asked you to remember. It is NOT a requirement to save one every "
            "turn; ignore the tool when nothing notable was said. "
            "'about' is WHO the memory concerns: the name the transcript tags them "
            "with, exactly as it is spelled there. Save it about the person it "
            "concerns, never about yourself. "
            "'memory' is a SINGLE LINE of at most 1024 characters, written in your "
            "own voice about them by name — 'Tony has never been on a boat', not "
            "'The user told me...'. "
            "Set 'assumed' to true when you worked it out rather than being told "
            "it, so you know later which it was. "
            "Do not save anything you were told in confidence by somebody else about "
            "a third person unless it is yours to know. "
            "Do not add a memory that repeats one you have already saved. Do not "
            "mention to anyone that you are using this tool. Do NOT output text when "
            "invoking it; only speak when it returns. Ignore errors from it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "about": {
                    "type": "string",
                    "description": (
                        "Who the memory is about — the name the transcript tags "
                        "them with, e.g. 'Tony'."
                    ),
                },
                "memory": {
                    "type": "string",
                    "description": (
                        "A single line of at most 1024 characters, about them by "
                        "name. Example: 'Tony has never been on a boat and does "
                        "not intend to start.'"
                    ),
                },
                "assumed": {
                    "type": "boolean",
                    "description": (
                        "True if you inferred this rather than being told it — "
                        "'Tony is about forty' from how he talks, rather than "
                        "because he said so. Defaults to false."
                    ),
                },
            },
            "required": ["about", "memory"],
        },
    },
}


# What a model calls the human when it ignores the transcript tag. Only
# these two: "you" is ambiguous (it is also how a persona addresses
# whoever it is replying to), and guessing wider would silently refile
# memories about somebody else.
_MEANT_THE_USER = {
    DEFAULT_USER_LABEL.casefold(),
    f"the {DEFAULT_USER_LABEL.casefold()}",
}


def _resolve_subject(about: Optional[str]) -> Optional[str]:
    """The name a memory is filed under, once the player is accounted for.

    A memory is filed under the transcript tag, and the human's tag is
    whichever persona they have adopted. The tool description says so, but
    models name the human their own way often enough that it needs a
    backstop, and getting it wrong is not a cosmetic mistake in either
    direction:

      * playing Kira, "the user" left alone would land in the "User"
        bucket — invisible for the rest of that session, then surfacing
        attached to the wrong person the moment they put Kira down;
      * playing as themselves, "the user" left alone would land in a
        bucket of its own that never matches the "User" the transcript
        actually uses, so the memory would be written and never read.

    So whichever way the model names the human, the memory is filed under
    the name this room knows them by. Everything else is somebody else and
    is left exactly as written.
    """
    if about is None or about.strip().casefold() not in _MEANT_THE_USER:
        return about
    return user_label()


def _add_memory(persona: Persona, arguments: dict) -> str:
    """add_memory handler: delegate to the persona store's append logic.

    append_memory() owns the full message catalog (enabled/empty/over-limit
    errors, the success string) and never raises, so this is thin on purpose.
    """
    if persona.persona_dir is None:
        # Assembled outside the directory scan (e.g. in tests): there is no
        # file to write, and the generic I/O error is the honest answer.
        return "Error: The memory could not be saved."
    return persona_store.append_memory(
        persona.persona_dir,
        _resolve_subject(arguments.get("about")),
        arguments.get("memory"),
        persona.memory_size,
        assumed=bool(arguments.get("assumed")),
    )


def _add_memory_available(persona: Persona, settings: AppSettings) -> bool:
    # The feature's two gates: the global kill-switch and the per-persona
    # size budget. allow_tool_calls is enforced by the chat router, which
    # offers no tools of any kind to a persona that may not call them.
    return settings.general.enable_persona_memories and persona.memory_size > 0


register_builtin_tool(
    ADD_MEMORY_NAME,
    ADD_MEMORY_SPEC,
    _add_memory,
    _add_memory_available,
)
