"""Player router — the human's own character.

One profile, not one per room. A room can *require* a profile
(`require_player_profile`, a property of the room), but the profile itself
belongs to the player and interacts with whichever room they are in.
"""

import logging

from fastapi import APIRouter

from app.config import PlayerConfig, PlayerProfile, get_player, save_player
from app.models import PlayerProfileRequest, PlayerProfileResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/player", tags=["player"])


def _to_response(profile: PlayerProfile) -> PlayerProfileResponse:
    return PlayerProfileResponse(
        name=profile.name,
        description=profile.description,
        appearance=profile.appearance,
    )


@router.get("", response_model=PlayerProfileResponse)
def get_profile():
    """The player's character. Empty fields when none has been set."""
    return _to_response(get_player().profile)


@router.put("", response_model=PlayerProfileResponse)
def set_profile(req: PlayerProfileRequest):
    """Replace the player's character.

    Whitespace-only fields are stored as empty, so a profile is never
    counted as complete by accident — `PlayerProfile.is_complete` gates
    chat in rooms that require one.
    """
    profile = PlayerProfile(
        name=req.name.strip(),
        description=req.description.strip(),
        appearance=req.appearance.strip(),
    )
    save_player(PlayerConfig(profile=profile))
    logger.info("Player character set to %r", profile.name or "(none)")
    return _to_response(profile)
