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
CONFIG_SCHEMA_VERSION = 2

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

def migrate_chatrooms(raw: dict) -> Tuple[dict, Notes]:
    """Bring a raw chatrooms.yaml dict up to the current schema."""
    return _apply(raw, {})


def migrate_settings(raw: dict) -> Tuple[dict, Notes]:
    """Bring a raw settings.yaml dict up to the current schema."""
    return _apply(raw, {})
