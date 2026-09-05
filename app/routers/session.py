"""Session router — inspect, reset, and configure the active session."""

import logging
import uuid

from fastapi import APIRouter, HTTPException

from app import persistence
from app.config import PlayerConfig, get_personas, get_player, save_player
from app.models import (
    ContextInventory,
    PersistedHistoryResponse,
    PersistedMessage,
    PersonaMemoryContext,
    RoomContext,
    SessionPersonasRequest,
    SessionState,
    WipeRequest,
    WipeResult,
)
from app.persistence import load_history_with_metadata
from app.services import persona_store
from app.session import session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/session", tags=["session"])


@router.get("", response_model=SessionState)
def get_session():
    """Return the current session state (history + active personas)."""
    return SessionState(
        history=session.get_history_dicts(),
        active_personas=session.active_personas,
        current_room=session.current_room,
    )


@router.post("/new")
def new_session():
    """Clear history and reset the session. Returns the fresh state."""
    session.reset()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Wiping context
# ---------------------------------------------------------------------------
#
# "New Chat" clears the room you are in, which is the right size for
# starting a fresh conversation and the wrong size for the question this
# answers: is anything at all still carrying over? Memories in particular
# survive every other clearing action in the app, live outside any room,
# and reach a persona in all of them — so proving a persona's behaviour is
# its own and not a leftover means clearing them too, and being able to
# see that they are gone.


def _inventory() -> ContextInventory:
    """Everything on disk that can reach a future turn. Read, never cached."""
    # Only rooms that hold something. An emptied room keeps its directory
    # (so it still exists for future messages), and listing those would
    # mean "nothing stored" never appeared however much was deleted —
    # which is the one sentence this whole feature exists to be able to
    # say. The wipe still sweeps every directory, messages or not.
    rooms = []
    for name in persistence.persisted_rooms():
        count = persistence.message_count(name)
        if count:
            rooms.append(RoomContext(room=name, messages=count))
    personas = []
    for persona in get_personas().personas:
        if persona.persona_dir is None:
            continue
        lines = [
            line for line in
            persona_store.read_memories(persona.persona_dir).splitlines()
            if line.strip()
        ]
        if lines:
            personas.append(
                PersonaMemoryContext(persona=persona.name, memories=len(lines))
            )
    return ContextInventory(
        rooms=rooms,
        personas=personas,
        playing_as=get_player().adopted({p.name for p in get_personas().personas}) or "",
    )


@router.get("/context", response_model=ContextInventory)
def get_context():
    """What is currently stored that could carry into a future turn."""
    return _inventory()


@router.post("/wipe", response_model=WipeResult)
def wipe_context(req: WipeRequest):
    """Delete stored context, and report what went and what is left.

    Deliberately does not touch personas, rooms or settings — only the
    conversational residue. A wipe that also deleted the cast would be a
    reset button, and this is a way to test the cast.
    """
    result = WipeResult(remaining=ContextInventory())

    if req.rooms != "none":
        targets = (
            persistence.persisted_rooms() if req.rooms == "all"
            else [session.current_room]
        )
        for room in targets:
            count = persistence.message_count(room)
            persistence.clear_room(room)
            result.rooms_cleared.append(room)
            result.messages_deleted += count
        # The room in use is also held in memory; clearing only the files
        # would leave this turn's history alive and the next reply built
        # on a conversation the user just watched disappear.
        if req.rooms == "all" or session.current_room in targets:
            session.load_room(session.current_room)

    if req.memories:
        for persona in get_personas().personas:
            if persona.persona_dir is None:
                continue
            try:
                if persona_store.remove_memories_file(persona.persona_dir):
                    result.memories_cleared.append(persona.name)
            except OSError as exc:
                # Surfaced rather than swallowed: a wipe that quietly
                # failed on one persona is worse than no wipe at all,
                # because the user goes on to trust it.
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not clear {persona.name}'s memories: {exc}",
                ) from exc

    if req.playing_as and get_player().persona_name:
        save_player(PlayerConfig(persona_name=""))
        result.playing_as_cleared = True

    result.remaining = _inventory()
    logger.info(
        "Context wipe: %d room(s), %d message(s), %d memory file(s), playing_as=%s",
        len(result.rooms_cleared), result.messages_deleted,
        len(result.memories_cleared), result.playing_as_cleared,
    )
    return result


@router.post("/personas")
def update_active_personas(req: SessionPersonasRequest):
    """Update which personas are active in the current session.

    Validates that all requested persona names exist in the config.
    """
    config = get_personas()
    valid_names = {p.name for p in config.personas}
    requested = set(req.active_personas)

    # Silently drop unknown names — they'll just be ignored
    unknown = requested - valid_names
    if unknown:
        logger.warning("Unknown persona names requested: %s", unknown)

    session.set_active_personas(list(requested & valid_names))
    return {"status": "updated", "active_personas": session.active_personas}


@router.get("/load-room/{room_name}")
def load_room(room_name: str):
    """Load persisted chat history for a room into the active session.

    Used when switching chat rooms. Clears any existing in-memory history
    and populates from the room's persisted data.
    """
    session.load_room(room_name)
    metadata = load_history_with_metadata(room_name)
    return PersistedHistoryResponse(
        room=room_name,
        datetime=metadata["datetime"],
        messages=[PersistedMessage(**m) for m in metadata["messages"]],
    )
