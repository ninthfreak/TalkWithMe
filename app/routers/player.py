"""Player router — which persona the human is playing.

The player adopts one of the configured personas rather than writing a
character of their own. A room can *require* that they have adopted
someone (`require_player_persona`, a property of the room), but who they
are playing belongs to the player and applies in whichever room they are
in.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.config import PlayerConfig, get_personas, get_player, save_player
from app.models import AdoptPersonaRequest, PlayerResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/player", tags=["player"])


@router.get("", response_model=PlayerResponse)
def get_adopted():
    """Who the player is playing. Empty when they are playing themselves.

    Resolved against the live persona list, so a persona deleted or renamed
    since it was adopted reads back as "themselves" rather than as a name
    that no longer exists.
    """
    known = {p.name for p in get_personas().personas}
    return PlayerResponse(persona_name=get_player().adopted(known))


@router.put("", response_model=PlayerResponse)
def adopt_persona(req: AdoptPersonaRequest):
    """Adopt a persona, or pass an empty name to play as yourself."""
    name = req.persona_name.strip()
    if name and name not in {p.name for p in get_personas().personas}:
        raise HTTPException(status_code=422, detail=f"Persona '{name}' does not exist.")

    save_player(PlayerConfig(persona_name=name))
    logger.info("Player is now playing as %s", name or "themselves")
    return PlayerResponse(persona_name=name)
