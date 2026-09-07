"""Session router — inspect, reset, and configure the active session."""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app import persistence
from app.config import (
    PlayerConfig,
    get_personas,
    get_player,
    get_settings,
    save_player,
    user_label,
)
from app.models import (
    ContextInventory,
    PersonaReflection,
    PersistedHistoryResponse,
    PersistedMessage,
    PersonaMemoryContext,
    RoomContext,
    SessionPersonasRequest,
    ReflectionResult,
    SessionState,
    WipeRequest,
    WipeResult,
)
from app.persistence import load_history_with_metadata
from app.services import persona_store, reflection
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


# ---------------------------------------------------------------------------
# Looking back on a finished conversation (app/services/reflection.py)
# ---------------------------------------------------------------------------
#
# A conversation has no "end" event — you stop typing — so the app takes
# the two moments where you visibly leave one: starting a new chat, and
# switching to another room. Both clear the history, so the snapshot has
# to be taken BEFORE they do.
#
# It runs in the background because it is a completion per persona who
# spoke, against a backend that serves one request at a time: awaited, a
# room of four would make "New Chat" sit there. Backgrounded, it queues
# behind nothing the user is waiting on — they have just started a fresh
# conversation and are typing the first line of it.


def _conversation_snapshot():
    """The live conversation, detached from the session that holds it.

    A copy, deliberately: the caller is about to clear the history, and a
    background task reading the session's own list would find it empty by
    the time it ran.
    """
    return list(session.history), session.current_room, user_label()


async def _reflect(history, room: str, label: str) -> list:
    """Run the pass, swallowing anything it throws.

    This is triggered by starting a new chat and by changing rooms. Losing
    the memories of one conversation is a disappointment; taking out the
    action that triggered it is a bug.
    """
    try:
        return await reflection.reflect_on_conversation(
            history, get_personas().personas, get_settings(), label, room=room,
        )
    except Exception:  # noqa: BLE001 — see the docstring
        logger.exception("Reflection on room '%s' failed", room)
        return []


def _schedule_reflection(background: BackgroundTasks) -> None:
    """Queue a look back at the conversation that is about to be cleared."""
    settings = get_settings()
    if not (settings.general.enable_persona_memories
            and settings.general.reflect_after_conversation):
        return
    history, room, label = _conversation_snapshot()
    if not history:
        return
    background.add_task(_reflect, history, room, label)


@router.post("/new")
def new_session(background: BackgroundTasks):
    """Clear history and reset the session. Returns the fresh state."""
    _schedule_reflection(background)
    session.reset()
    return {"status": "cleared"}


@router.post("/reflect", response_model=ReflectionResult)
async def reflect_now():
    """Look back at the current conversation now, and say what was learned.

    Awaited rather than backgrounded, unlike the automatic passes: this
    one is asked for, so its answer is the point. The history is left
    alone — reflecting is not the same as ending the conversation, and you
    may well want to carry on afterwards.
    """
    settings = get_settings()
    if not settings.general.enable_persona_memories:
        raise HTTPException(
            status_code=409,
            detail="Persona memories are switched off in settings.",
        )
    history, room, label = _conversation_snapshot()
    results = await _reflect(history, room, label)
    return ReflectionResult(
        room=room,
        personas=[
            PersonaReflection(persona=r.persona, saved=r.saved, skipped=len(r.skipped))
            for r in results
        ],
    )


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
        met = persona_store.read_acquaintances(persona.persona_dir)
        if lines or met:
            personas.append(
                PersonaMemoryContext(
                    persona=persona.name, memories=len(lines), met=len(met),
                )
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
                # The met-list goes with the memories: leaving it behind
                # would mean a wiped persona still greets everyone as an
                # old acquaintance, which is the one thing this is for.
                forgot = persona_store.forget_acquaintances(persona.persona_dir)
                if persona_store.remove_memories_file(persona.persona_dir) or forgot:
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
def load_room(room_name: str, background: BackgroundTasks):
    """Load persisted chat history for a room into the active session.

    Used when switching chat rooms. Clears any existing in-memory history
    and populates from the room's persisted data.

    Leaving a room ends the conversation you were having in it, so the
    outgoing one is reflected on first — snapshotted before load_room()
    replaces the history. Re-loading the room you are already in is a
    refresh, not a departure, and reflects on nothing.
    """
    if room_name != session.current_room:
        _schedule_reflection(background)
    session.load_room(room_name)
    metadata = load_history_with_metadata(room_name)
    return PersistedHistoryResponse(
        room=room_name,
        datetime=metadata["datetime"],
        messages=[PersistedMessage(**m) for m in metadata["messages"]],
    )
