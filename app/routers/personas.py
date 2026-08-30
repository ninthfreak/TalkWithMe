"""Personas router — list, create, update, delete, clone personas and serve avatar images."""

import logging
import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.config import (
    ChatRoomsConfig,
    get_chatrooms,
    get_personas,
    save_chatrooms,
    save_personas,
)
from app.config import Persona, PersonasConfig
from app.models import PersonaResponse, PersonaDetailResponse, PersonaCreateRequest, PersonaUpdateRequest

logger = logging.getLogger(__name__)

# Persona names appear in URL path segments (/api/personas/{name}/detail) and
# in the "[Name]: " transcript tags. A name containing "/" produced a persona
# that returned 201 on create and then 404 on every edit, delete and clone —
# permanently stuck in personas.yaml and still answering.
_SAFE_PERSONA_NAME = re.compile(r"^[^/\\\n\r\t]+$")
MAX_PERSONA_NAME = 25


def _validate_persona_name(name: str) -> str:
    """Trim and reject a name that cannot survive a round trip."""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Persona name cannot be blank.")
    if not _SAFE_PERSONA_NAME.match(name):
        raise HTTPException(
            status_code=422,
            detail="Persona name may not contain slashes, backslashes or line breaks.",
        )
    if len(name) > MAX_PERSONA_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"Persona name may be at most {MAX_PERSONA_NAME} characters.",
        )
    return name


router = APIRouter(prefix="/api/personas", tags=["personas"])


def _cascade_persona_rename(old_name: str, new_name: str) -> None:
    """Update persona name references in all chat rooms.

    When a persona is renamed, every chat room that references it
    must be updated to use the new name so assignments stay valid.
    """
    config = get_chatrooms()
    if not config.chat_rooms:
        return
    updated = []
    for room in config.chat_rooms:
        new_names = [new_name if p == old_name else p for p in room.persona_names]
        # model_copy, not a fresh ChatRoom: rebuilding field-by-field drops
        # every room setting the call site forgets to list (echo_chamber was
        # being reset to False by every rename and delete).
        updated.append(room.model_copy(update={"persona_names": new_names}))
    save_chatrooms(ChatRoomsConfig(chat_rooms=updated))
    logger.info("Cascaded persona rename '%s' -> '%s' to chat rooms", old_name, new_name)


def _cascade_persona_delete(persona_name: str) -> None:
    """Remove a persona from all chat rooms.

    When a persona is deleted, it must be removed from every chat room
    that had it assigned to avoid dangling references.
    """
    config = get_chatrooms()
    if not config.chat_rooms:
        return
    updated = []
    for room in config.chat_rooms:
        new_names = [p for p in room.persona_names if p != persona_name]
        updated.append(room.model_copy(update={"persona_names": new_names}))
    save_chatrooms(ChatRoomsConfig(chat_rooms=updated))
    logger.info("Cascaded persona delete '%s' from chat rooms", persona_name)


def _to_response(p: Persona) -> PersonaResponse:
    return PersonaResponse(
        name=p.name,
        description=p.description,
        avatar_color=p.avatar_color,
        avatar_image=p.avatar_image,
        tts_capable=p.tts_capable,
    )


def _to_detail(p: Persona) -> PersonaDetailResponse:
    return PersonaDetailResponse(
        name=p.name,
        description=p.description,
        system_prompt=p.system_prompt,
        router_hints=p.router_hints,
        avatar_color=p.avatar_color,
        avatar_image=p.avatar_image,
        reference_audio=p.reference_audio,
        reference_audio_transcript=p.reference_audio_transcript,
        reference_audio_language=p.reference_audio_language,
        allow_tool_calls=p.allow_tool_calls,
        length_bias=p.length_bias,
        tts_capable=p.tts_capable,
    )


@router.get("", response_model=List[PersonaResponse])
def list_personas():
    """Return all configured personas with TTS capability flags."""
    return [_to_response(p) for p in get_personas().personas]


@router.post("", response_model=PersonaDetailResponse, status_code=201)
def create_persona(req: PersonaCreateRequest):
    """Add a new persona to personas.yaml."""
    config = get_personas()
    name = _validate_persona_name(req.name)
    lower_name = name.lower()
    if lower_name == "user":
        raise HTTPException(
            status_code=422,
            detail="'user' is a reserved persona name and cannot be used.",
        )
    if any(p.name.lower() == lower_name for p in config.personas):
        raise HTTPException(status_code=409, detail=f"A persona named '{req.name}' already exists")

    new_persona = Persona(
        name=name,
        description=req.description or "",
        system_prompt=req.system_prompt,
        router_hints=req.router_hints,
        avatar_color=req.avatar_color,
        avatar_image=req.avatar_image or None,
        reference_audio=req.reference_audio or None,
        reference_audio_transcript=req.reference_audio_transcript or None,
        reference_audio_language=req.reference_audio_language,
        allow_tool_calls=req.allow_tool_calls,
        length_bias=req.length_bias,
    )
    save_personas(PersonasConfig(personas=config.personas + [new_persona]))
    return _to_detail(new_persona)


@router.get("/{name}/detail", response_model=PersonaDetailResponse)
def get_persona_detail(name: str):
    """Return full detail for a single persona (all editable fields)."""
    config = get_personas()
    persona = next((p for p in config.personas if p.name == name), None)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    return _to_detail(persona)


@router.put("/{name}", response_model=PersonaDetailResponse)
def update_persona(name: str, req: PersonaUpdateRequest):
    """Update an existing persona in personas.yaml."""
    config = get_personas()
    existing = next((p for p in config.personas if p.name == name), None)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")

    new_name = _validate_persona_name(req.name)
    if new_name.lower() == "user":
        raise HTTPException(
            status_code=422,
            detail="'user' is a reserved persona name and cannot be used.",
        )

    if new_name.lower() != name.lower():
        if any(p.name.lower() == new_name.lower() for p in config.personas if p.name != name):
            raise HTTPException(status_code=409, detail=f"A persona named '{new_name}' already exists")

    updated_persona = Persona(
        name=new_name,
        description=req.description or "",
        system_prompt=req.system_prompt,
        router_hints=req.router_hints,
        avatar_color=req.avatar_color,
        avatar_image=req.avatar_image or None,
        reference_audio=req.reference_audio or None,
        reference_audio_transcript=req.reference_audio_transcript or None,
        reference_audio_language=req.reference_audio_language,
        allow_tool_calls=req.allow_tool_calls,
        length_bias=req.length_bias,
    )
    new_list = [updated_persona if p.name == name else p for p in config.personas]
    save_personas(PersonasConfig(personas=new_list))

    # If the persona was renamed, update all chat rooms referencing it
    if new_name != name:
        _cascade_persona_rename(name, new_name)

    return _to_detail(updated_persona)


@router.delete("/{name}", status_code=204)
def delete_persona(name: str):
    """Remove a persona from personas.yaml and all chat rooms."""
    config = get_personas()
    if not any(p.name == name for p in config.personas):
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    save_personas(PersonasConfig(personas=[p for p in config.personas if p.name != name]))
    # Remove this persona from every chat room that had it assigned
    _cascade_persona_delete(name)


@router.post("/{name}/clone", response_model=PersonaDetailResponse, status_code=201)
def clone_persona(name: str):
    """Clone an existing persona, appending a numeric suffix to ensure uniqueness."""
    config = get_personas()
    source = next((p for p in config.personas if p.name == name), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")

    # The suffix has to fit inside MAX_PERSONA_NAME or the clone is born
    # un-editable: PUT /api/personas/{name} would reject the very name the
    # clone was saved under. Trim the base, not the suffix, so the result
    # stays unique.
    existing_names = {p.name.lower() for p in config.personas}
    suffix = 2
    while True:
        tail = f"_{suffix}"
        base = name[: MAX_PERSONA_NAME - len(tail)].rstrip()
        new_name = f"{base}{tail}"
        if new_name.lower() not in existing_names:
            break
        suffix += 1

    clone = Persona(
        name=new_name,
        description=source.description,
        system_prompt=source.system_prompt,
        router_hints=source.router_hints,
        avatar_color=source.avatar_color,
        avatar_image=source.avatar_image,
        reference_audio=source.reference_audio,
        reference_audio_transcript=source.reference_audio_transcript,
        reference_audio_language=source.reference_audio_language,
        allow_tool_calls=source.allow_tool_calls,
        length_bias=source.length_bias,
    )
    save_personas(PersonasConfig(personas=config.personas + [clone]))
    return _to_detail(clone)


@router.get("/{name}/avatar")
async def get_avatar(name: str):
    """Serve a persona's avatar image file.

    Returns 404 if the persona has no avatar_image configured or the file
    doesn't exist on disk.
    """
    config = get_personas()
    persona = next((p for p in config.personas if p.name == name), None)
    if not persona or not persona.avatar_image:
        return Response(status_code=404, content="No avatar configured")

    path = Path(persona.avatar_image)
    if not path.exists():
        logger.warning("Avatar file not found for %s: %s", name, persona.avatar_image)
        return Response(status_code=404, content="Avatar file not found")

    return FileResponse(str(path))
