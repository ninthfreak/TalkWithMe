"""Personas router — list personas and serve avatar images."""

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

from app.config import get_personas
from app.models import PersonaResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/personas", tags=["personas"])


@router.get("", response_model=List[PersonaResponse])
def list_personas():
    """Return all configured personas with TTS capability flags."""
    config = get_personas()
    result = []
    for p in config.personas:
        result.append(
            PersonaResponse(
                name=p.name,
                description=p.description,
                avatar_color=p.avatar_color,
                avatar_image=p.avatar_image,
                tts_capable=p.tts_capable,
            )
        )
    return result


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
