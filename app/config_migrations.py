"""Bring old config files up to the current schema.

Every YAML file the app owns carries a ``schema_version``. Files written
before versioning existed have none and are treated as version 1. Loading
always runs the raw dict through the migrations below first, so a file from
any earlier release still loads — the app never fails on an old file, and
never silently misreads one either.

Migrations work on the **raw dict**, before Pydantic sees it. That is the
point: once a model has parsed (or rejected, or quietly dropped) a legacy
key, the information needed to migrate it is already gone.

Each function returns ``(migrated_raw, notes)``. The notes are logged by
the caller, so an upgrade is visible rather than mysterious.

Adding a migration
------------------
1. Bump ``CONFIG_SCHEMA_VERSION``.
2. Add a ``_personas_v2_to_v3``-style step and list it in the chain.
3. Cover it in ``tests/test_config_migrations.py`` with a real old file.
"""

import logging
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# 1 = unversioned (everything before schema_version existed)
# 2 = personas use length_bias (relative) instead of typical_length
#     (absolute); rooms gained typical_length / player_profile; the length
#     scale was recalibrated for chat.
# 3 = no structural change (a removal that was reverted; the number is
#     kept so files stamped by it do not warn on every load).
# 4 = echo_chamber is not a room attribute. The Echo chamber checkbox under
#     the chat room selector is client-side UI state, sent with each
#     message — the room never knew about it.
# 5 = the player's character profile is not a room attribute either. A room
#     can *require* one (require_player_profile stays), but the profile
#     itself belongs to the player and now lives in player.yaml.
CONFIG_SCHEMA_VERSION = 5

Notes = List[str]
Step = Callable[[dict], Tuple[dict, Notes]]


def _version_of(raw: dict) -> int:
    """The file's schema version, defaulting to 1 for unversioned files."""
    try:
        return int(raw.get("schema_version", 1))
    except (TypeError, ValueError):
        logger.warning(
            "Config has a non-numeric schema_version %r; treating it as 1",
            raw.get("schema_version"),
        )
        return 1


def _apply(raw: dict, chain: Dict[int, Step]) -> Tuple[dict, Notes]:
    """Run every step from the file's version up to the current one."""
    version = _version_of(raw)
    notes: Notes = []

    if version > CONFIG_SCHEMA_VERSION:
        # A file written by a newer version of the app. Load it anyway —
        # unknown keys are ignored by the models — but say so, because
        # settings silently reverting is worse than a warning.
        logger.warning(
            "Config schema version %d is newer than this app understands (%d). "
            "Loading it anyway; unrecognised settings will be ignored.",
            version, CONFIG_SCHEMA_VERSION,
        )
        return raw, notes

    while version < CONFIG_SCHEMA_VERSION:
        step = chain.get(version)
        if step is not None:
            raw, step_notes = step(raw)
            notes.extend(step_notes)
        version += 1

    raw["schema_version"] = CONFIG_SCHEMA_VERSION
    return raw, notes


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------

# Old absolute tiers, in order, with the old "normal" as the anchor. A
# persona's tier said where it sat relative to a typical reply, so that
# offset is exactly what carries over to the new relative bias.
_LEGACY_LENGTH_ORDER = ["terse", "brief", "normal", "detailed"]
_LEGACY_NORMAL_INDEX = 2
_OFFSET_TO_BIAS = {
    -2: "much_shorter",
    -1: "shorter",
    0: "match",
    1: "longer",
    2: "much_longer",
}


def _legacy_length_to_bias(value: str) -> str:
    """Map an old absolute persona tier onto the new relative bias.

    The old tiers were absolute word targets, the new bias is a step along
    the room's scale — but an old tier still encoded an *intent*: "shorter
    than usual", "longer than usual". Preserving that offset keeps a
    laconic persona laconic, which dropping the field would not.

    "unrestricted" had no position on the scale, so it becomes "match".
    """
    try:
        offset = _LEGACY_LENGTH_ORDER.index(str(value).lower()) - _LEGACY_NORMAL_INDEX
    except ValueError:
        return "match"
    return _OFFSET_TO_BIAS.get(offset, "match")


def _personas_v1_to_v2(raw: dict) -> Tuple[dict, Notes]:
    notes: Notes = []
    for persona in raw.get("personas") or []:
        if not isinstance(persona, dict):
            continue
        name = persona.get("name", "<unknown>")

        # Predates the reference_audio_* rename.
        if "language" in persona and "reference_audio_language" not in persona:
            persona["reference_audio_language"] = persona.pop("language")
            notes.append(f"{name}: 'language' -> 'reference_audio_language'")

        # Absolute per-persona length replaced by a bias relative to the room.
        if "typical_length" in persona:
            old = persona.pop("typical_length")
            if old is not None and "length_bias" not in persona:
                bias = _legacy_length_to_bias(old)
                persona["length_bias"] = bias
                notes.append(f"{name}: 'typical_length: {old}' -> 'length_bias: {bias}'")
    return raw, notes


_PERSONA_STEPS: Dict[int, Step] = {1: _personas_v1_to_v2}


def _chatrooms_v3_to_v4(raw: dict) -> Tuple[dict, Notes]:
    """Drop echo_chamber from stored rooms.

    The checkbox stays exactly where it was, under the chat room selector,
    but it is UI state rather than something the room owns: the left panel
    holds controls you use *while in* a room, which is not the same as
    settings that belong *to* the room.
    """
    notes: Notes = []
    for room in raw.get("chat_rooms") or []:
        if isinstance(room, dict) and "echo_chamber" in room:
            was = room.pop("echo_chamber")
            notes.append(
                f"{room.get('name', '<unknown>')}: dropped 'echo_chamber: {was}' "
                "— the Echo chamber checkbox is UI state now, not a room setting"
            )
    return raw, notes


def migrate_personas(raw: dict) -> Tuple[dict, Notes]:
    """Bring a raw personas.yaml dict up to the current schema."""
    return _apply(raw, _PERSONA_STEPS)


# ---------------------------------------------------------------------------
# Chat rooms and settings
# ---------------------------------------------------------------------------
#
# Both gained only new keys, and every new key has a default, so there is
# nothing to rewrite — the version stamp is the whole migration. They still
# go through _apply so the stamp is added and a future step has somewhere
# to live.

def _chatrooms_v4_to_v5(raw: dict) -> Tuple[dict, Notes]:
    """Drop player_profile from stored rooms.

    Whether a room requires a profile is a property of the room; the
    profile itself is the player's, and interacts with whichever room they
    are in. `lift_player_profile()` moves the value into player.yaml before
    this drops it, so nothing is lost.
    """
    notes: Notes = []
    for room in raw.get("chat_rooms") or []:
        if isinstance(room, dict) and "player_profile" in room:
            room.pop("player_profile")
            notes.append(
                f"{room.get('name', '<unknown>')}: dropped 'player_profile' "
                "— your character lives in player.yaml now, not in a room"
            )
    return raw, notes


_CHATROOM_STEPS: Dict[int, Step] = {
    3: _chatrooms_v3_to_v4,
    4: _chatrooms_v4_to_v5,
}


def migrate_player(raw: dict) -> Tuple[dict, Notes]:
    """Bring a raw player.yaml dict up to the current schema."""
    return _apply(raw, {})


def lift_player_profile(chatrooms_raw: dict) -> dict:
    """The first non-empty player_profile found in a rooms file, or {}.

    Used once, when player.yaml does not exist yet, to carry a profile
    across from the rooms it used to be stored in. Profiles were per room,
    so several may exist; the first named one wins and the rest are
    reported, because silently picking one of several characters would be
    worse than saying so.
    """
    found = []
    for room in chatrooms_raw.get("chat_rooms") or []:
        if not isinstance(room, dict):
            continue
        profile = room.get("player_profile") or {}
        if isinstance(profile, dict) and str(profile.get("name", "")).strip():
            found.append((room.get("name", "<unknown>"), profile))
    if not found:
        return {}
    if len(found) > 1:
        logger.info(
            "Several rooms had a character; keeping %s's (from '%s') and "
            "discarding %s. Edit it under \"Your character\" if that is the wrong one.",
            found[0][1].get("name"), found[0][0],
            ", ".join(f"{p.get('name')} (from '{r}')" for r, p in found[1:]),
        )
    return found[0][1]


def migrate_chatrooms(raw: dict) -> Tuple[dict, Notes]:
    """Bring a raw chatrooms.yaml dict up to the current schema."""
    return _apply(raw, _CHATROOM_STEPS)


def migrate_settings(raw: dict) -> Tuple[dict, Notes]:
    """Bring a raw settings.yaml dict up to the current schema."""
    return _apply(raw, {})
