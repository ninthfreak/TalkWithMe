"""Personas router — list, create, update, delete, clone personas and serve
avatar / reference audio files.

Personas live in per-persona subdirectories of the configured Personas
directory (see app/services/persona_store.py); the directory on disk is
the source of truth. Create/update are multipart/form-data so the editor
can submit text fields and file uploads in a single request. Every
mutation refreshes the in-memory persona cache via set_personas_cache()
— skipping that step is how the UI ends up stale until the next restart.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.config import (
    LengthBias,
    DEFAULT_MEMORY_SIZE,
    DEFAULT_USER_LABEL,
    MAX_MEMORY_SIZE,
    MAX_PERSONA_NAME,
    ChatRoom,
    ChatRoomsConfig,
    Persona,
    PersonasConfig,
    derive_max_tokens,
    get_chatrooms,
    get_personas,
    get_personas_directory,
    PlayerConfig,
    get_player,
    get_settings,
    resolve_typical_length,
    save_chatrooms,
    save_player,
    set_personas_cache,
)
from app.models import (
    CondenseRequest,
    CondenseResponse,
    PersonaDetailResponse,
    PersonaDraftRequest,
    PersonaDraftResponse,
    PersonaPreviewReply,
    PersonaPreviewRequest,
    PersonaPreviewResponse,
    PersonaRefineRequest,
    PersonaRefineResponse,
    PersonaRenameRequest,
    PersonaRenameResponse,
    PersonaResponse,
)
from app import persistence
from app.services import condense, persona_draft, persona_store
from app.services.llm import PROSE_TIMEOUT, chat_completion
from app.services.reply_guard import ReplyGuard, stop_sequences

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/personas", tags=["personas"])


# ---------------------------------------------------------------------------
# Chat-room cascades (kept in sync with the data model — see AGENTS.md)
# ---------------------------------------------------------------------------

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
        # every room setting the call site forgets to list (typical_length
        # and require_player_persona were both being reset by every rename
        # and delete).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_persona_dir(persona_dir: Path) -> None:
    """Best-effort removal of a persona directory (create-failure cleanup)."""
    try:
        if persona_dir.is_dir():
            shutil.rmtree(persona_dir)
    except OSError as exc:
        logger.warning("Could not remove persona directory %s: %s", persona_dir, exc)


def _validate_image_upload(upload: UploadFile) -> Tuple[str, bytes]:
    """Validate an avatar image upload; return (extension, content) or raise 422."""
    filename = upload.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in persona_store.IMAGE_EXTENSIONS:
        allowed = ", ".join(ext.lstrip(".") for ext in persona_store.IMAGE_EXTENSIONS)
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported avatar image file '{filename}'. Allowed: {allowed}.",
        )
    content = upload.file.read()
    if len(content) > persona_store.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"Avatar image exceeds the {persona_store.MAX_IMAGE_BYTES // (1024 * 1024)}MB limit.",
        )
    return extension, content


def _validate_audio_upload(upload: UploadFile) -> bytes:
    """Validate a reference audio upload; return content or raise 422."""
    filename = upload.filename or ""
    if Path(filename).suffix.lower() != persona_store.AUDIO_EXTENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported reference audio file '{filename}'. Only wav audio is supported.",
        )
    content = upload.file.read()
    if len(content) > persona_store.MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"Reference audio exceeds the {persona_store.MAX_AUDIO_BYTES // (1024 * 1024)}MB limit.",
        )
    return content


def _apply_persona_fields(
    persona_dir: Path,
    *,
    name: str,
    description: str,
    system_prompt: str,
    router_hints: str,
    avatar_color: str,
    reference_audio_language: str,
    allow_tool_calls: bool,
    length_bias: str = LengthBias.MATCH.value,
    reference_audio_transcript: str,
    image: Optional[Tuple[str, bytes]],
    audio: Optional[bytes],
    remove_image: bool,
    remove_audio: bool,
    memory_size: int,
) -> None:
    """Write one create/update payload into a persona directory.

    Uploads are validated (422) BEFORE this is called, so this function
    only raises OSError. A failure mid-way can leave the persona
    half-updated; with local disk and small files that is acceptable
    (the previous values are recoverable from a backup).
    """
    persona_store.write_prompt_md(
        persona_dir,
        name=name,
        description=description,
        router_hints=router_hints,
        avatar_color=avatar_color,
        allow_tool_calls=allow_tool_calls,
        length_bias=length_bias,
        system_prompt=system_prompt,
        memory_size=memory_size,
    )
    persona_store.write_language_file(persona_dir, reference_audio_language)
    persona_store.write_transcript_file(persona_dir, reference_audio_transcript)
    if image is not None:
        persona_store.write_avatar_file(persona_dir, image[1], image[0])
    elif remove_image:
        persona_store.remove_avatar_file(persona_dir)
    if audio is not None:
        persona_store.write_reference_audio_file(persona_dir, audio)
    elif remove_audio:
        persona_store.remove_reference_audio_file(persona_dir)


def _read_uploads(
    avatar_image: Optional[UploadFile],
    reference_audio: Optional[UploadFile],
) -> Tuple[Optional[Tuple[str, bytes]], Optional[bytes]]:
    """Validate both optional file uploads (or treat them as absent)."""
    image = _validate_image_upload(avatar_image) if avatar_image and avatar_image.filename else None
    audio = _validate_audio_upload(reference_audio) if reference_audio and reference_audio.filename else None
    return image, audio


_SAFE_PERSONA_NAME = re.compile(r"^[^/\\\n\r\t]+$")

# Matches the Form(max_length=...) on create and update. Defined in
# app/config.py because it is also the cap on a memory's [Subject] tag —
# a persona is filed in other personas' memories under this name.


def _validate_name(name: str, reserved_check: bool = True) -> str:
    """Shared name rules for create/update; returns the stripped name."""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name may not be blank")
    if reserved_check and name.lower() == "user":
        raise HTTPException(
            status_code=422,
            detail="'user' is a reserved persona name and cannot be used.",
        )
    # sanitize_persona_dirname keeps the *directory* safe, but the name is
    # still a path segment on /api/personas/{name}/... — one containing a
    # slash produced a persona that returned 201 on create and 404 on every
    # edit, delete and clone.
    if not _SAFE_PERSONA_NAME.match(name):
        raise HTTPException(
            status_code=422,
            detail="Persona name may not contain slashes, backslashes or line breaks.",
        )
    return name


def _to_response(p: Persona) -> PersonaResponse:
    return PersonaResponse(
        name=p.name,
        description=p.description,
        avatar_color=p.avatar_color,
        avatar_image=bool(p.avatar_image),
        tts_capable=p.tts_capable,
    )


def _to_detail(p: Persona) -> PersonaDetailResponse:
    transcript: Optional[str] = None
    if p.reference_audio_transcript:
        path = Path(p.reference_audio_transcript)
        if path.is_file():
            transcript = path.read_text(encoding="utf-8", errors="replace")
    return PersonaDetailResponse(
        name=p.name,
        description=p.description,
        system_prompt=p.system_prompt,
        router_hints=p.router_hints,
        avatar_color=p.avatar_color,
        avatar_image=bool(p.avatar_image),
        reference_audio=bool(p.reference_audio),
        reference_audio_transcript=transcript,
        reference_audio_language=p.reference_audio_language,
        allow_tool_calls=p.allow_tool_calls,
        length_bias=p.length_bias,
        memory_size=p.memory_size,
        tts_capable=p.tts_capable,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=List[PersonaResponse])
def list_personas():
    """Return all configured personas with TTS capability flags."""
    return [_to_response(p) for p in get_personas().personas]


@router.post("", response_model=PersonaDetailResponse, status_code=201)
def create_persona(
    name: str = Form(..., max_length=25),
    description: str = Form("", max_length=30),
    system_prompt: str = Form(..., min_length=1, max_length=8192),
    router_hints: str = Form(..., min_length=1, max_length=256),
    avatar_color: str = Form("#FF0000"),
    reference_audio_language: str = Form("en", min_length=2, max_length=2),
    allow_tool_calls: bool = Form(False),
    # Relative to the room, never absolute — see LengthBias.
    length_bias: LengthBias = Form(LengthBias.MATCH),
    reference_audio_transcript: str = Form("", max_length=16384),
    memory_size: int = Form(DEFAULT_MEMORY_SIZE, ge=0, le=MAX_MEMORY_SIZE),
    avatar_image: Optional[UploadFile] = File(None),
    remove_avatar_image: bool = Form(False),
    reference_audio: Optional[UploadFile] = File(None),
    remove_reference_audio: bool = Form(False),
    clear_memories: bool = Form(False),
):
    """Create a new persona in the Personas directory (multipart/form-data).

    The remove_* / clear_memories flags are accepted for API symmetry with
    update but are ignored here: a brand-new directory has nothing to
    remove.
    """
    name = _validate_name(name)
    if any(p.name.lower() == name.lower() for p in get_personas().personas):
        raise HTTPException(status_code=409, detail=f"A persona named '{name}' already exists")

    dir_base = persona_store.sanitize_persona_dirname(name)
    if not dir_base:
        raise HTTPException(
            status_code=422,
            detail="Name must contain at least one letter, number, space, hyphen or underscore",
        )

    image, audio = _read_uploads(avatar_image, reference_audio)

    root = get_personas_directory()
    persona_dir = root / persona_store.unique_persona_dirname(root, dir_base)

    try:
        persona_dir.mkdir(parents=True)
        _apply_persona_fields(
            persona_dir,
            name=name,
            description=description or "",
            system_prompt=system_prompt,
            router_hints=router_hints,
            avatar_color=avatar_color,
            reference_audio_language=reference_audio_language,
            allow_tool_calls=allow_tool_calls,
            length_bias=length_bias.value,
            reference_audio_transcript=reference_audio_transcript,
            image=image,
            audio=audio,
            remove_image=False,
            remove_audio=False,
            memory_size=memory_size,
        )
        persona = persona_store.load_persona_from_dir(persona_dir)
    except OSError as exc:
        _remove_persona_dir(persona_dir)
        logger.error("Failed to create persona '%s' in %s: %s", name, persona_dir, exc)
        raise HTTPException(status_code=500, detail=f"Failed to create persona: {exc}") from exc

    set_personas_cache(PersonasConfig(personas=get_personas().personas + [persona]))
    return _to_detail(persona)


@router.get("/{name}/detail", response_model=PersonaDetailResponse)
def get_persona_detail(name: str):
    """Return full detail for a single persona (all editable fields)."""
    config = get_personas()
    persona = next((p for p in config.personas if p.name == name), None)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    return _to_detail(persona)


@router.put("/{name}", response_model=PersonaDetailResponse)
def update_persona(
    name: str,
    new_name: str = Form(..., max_length=25, alias="name"),
    description: str = Form("", max_length=30),
    system_prompt: str = Form(..., min_length=1, max_length=8192),
    router_hints: str = Form(..., min_length=1, max_length=256),
    avatar_color: str = Form("#FF0000"),
    reference_audio_language: str = Form("en", min_length=2, max_length=2),
    allow_tool_calls: bool = Form(False),
    # Relative to the room, never absolute — see LengthBias.
    length_bias: LengthBias = Form(LengthBias.MATCH),
    reference_audio_transcript: str = Form("", max_length=16384),
    # REQUIRED on update (no default): an omitted value must 422, not
    # silently reset the persona's memory budget to the default.
    memory_size: int = Form(..., ge=0, le=MAX_MEMORY_SIZE),
    avatar_image: Optional[UploadFile] = File(None),
    remove_avatar_image: bool = Form(False),
    reference_audio: Optional[UploadFile] = File(None),
    remove_reference_audio: bool = Form(False),
    clear_memories: bool = Form(False),
):
    """Update an existing persona in its directory (multipart/form-data).

    Every field except the name. A changed name is refused with a 409 and
    sent to POST /api/personas/{name}/rename, which cascades it to the
    places a save cannot reach — other personas' memories and met-lists,
    room membership, the adopted player, stored transcripts and the
    folder itself. This endpoint used to accept a rename and cascade it
    to chat rooms only, which left everything else pointing at somebody
    who no longer existed.

    Memory handling: clear_memories=True deletes memories.txt outright
    (an explicit user action — a failure here DOES fail the save, so the
    user knows the clear did not happen). Otherwise, dropping memory_size
    below the current file size triggers an oldest-first purge; that
    path is best-effort and never fails the save.
    """
    config = get_personas()
    existing = next((p for p in config.personas if p.name == name), None)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    if existing.persona_dir is None or not existing.persona_dir.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Persona '{name}' has no directory on disk; cannot update it",
        )

    new_name = _validate_name(new_name)
    # A name change is not an edit, and this refusal is what makes that
    # true. Names are identifiers here: the same string is a memory's
    # subject tag, a met-list entry, a room member, the adopted player and
    # the tag on every line this persona has ever spoken. Changing it here
    # would rewrite prompt.md and orphan all of it — which is exactly what
    # this endpoint used to do. The rename endpoint cascades; a read-only
    # field in the browser only reminds.
    if new_name != name:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Use Rename to change a persona's name. Saving cannot rename "
                f"'{name}' to '{new_name}', because the old name is also how "
                f"other personas remember them."
            ),
        )

    image, audio = _read_uploads(avatar_image, reference_audio)
    persona_dir = existing.persona_dir

    try:
        _apply_persona_fields(
            persona_dir,
            name=new_name,
            description=description or "",
            system_prompt=system_prompt,
            router_hints=router_hints,
            avatar_color=avatar_color,
            reference_audio_language=reference_audio_language,
            allow_tool_calls=allow_tool_calls,
            length_bias=length_bias.value,
            reference_audio_transcript=reference_audio_transcript,
            image=image,
            audio=audio,
            remove_image=remove_avatar_image,
            remove_audio=remove_reference_audio,
            memory_size=memory_size,
        )
        if clear_memories:
            # Explicit user action: propagate I/O failures as 500 (the
            # persona fields ARE saved; a silent no-op clear is worse).
            persona_store.remove_memories_file(persona_dir)
        else:
            # Best-effort: shrink an over-limit memories file to the new
            # budget. Never raises (see purge_memories_to_limit).
            persona_store.purge_memories_to_limit(persona_dir, memory_size)
        updated = persona_store.load_persona_from_dir(persona_dir)
    except OSError as exc:
        logger.error("Failed to update persona '%s' in %s: %s", name, persona_dir, exc)
        raise HTTPException(status_code=500, detail=f"Failed to update persona: {exc}") from exc

    new_list = [updated if p.name == name else p for p in config.personas]
    set_personas_cache(PersonasConfig(personas=new_list))

    return _to_detail(updated)


@router.post("/{name}/rename", response_model=PersonaRenameResponse)
def rename_persona(name: str, req: PersonaRenameRequest):
    """Rename a persona everywhere it is referred to.

    This exists because the app uses **names as identifiers**, and that is
    a deliberate choice rather than an oversight: the name is not only a
    key, it is the string the model reads and writes. A memory's subject
    tag, a met-list entry, a chat-room member, the adopted player and the
    "[Name]: " tag on every stored line are all the same text — and they
    have to be, or the mechanical containment layers (stop_sequences,
    ReplyGuard) would be guarding a name the prompt never uses.

    The price of that is that a rename cannot be an edit to one field, so
    it is not offered as one: ``PUT /api/personas/{name}`` refuses a
    changed name and sends the caller here. That refusal is the actual
    guarantee — a read-only input in the browser is only a reminder.

    Ordering is chosen for its failure modes. The persona's own
    prompt.md goes first, because until it is written nothing has
    happened and the error is clean. The references follow, each
    best-effort and counted, so one unwritable file leaves a reported
    warning rather than an abandoned half-rename. The directory moves
    **last** and is the most expendable step: the name in frontmatter is
    the identity, so a failure there leaves a correctly renamed character
    in a stale folder.
    """
    config = get_personas()
    existing = next((p for p in config.personas if p.name == name), None)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    if existing.persona_dir is None or not existing.persona_dir.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Persona '{name}' has no directory on disk; cannot rename it",
        )

    new_name = _validate_name(req.new_name)
    if new_name == name:
        raise HTTPException(
            status_code=422, detail=f"'{name}' is already this persona's name.",
        )
    if any(p.name.lower() == new_name.lower() for p in config.personas if p.name != name):
        raise HTTPException(
            status_code=409, detail=f"A persona named '{new_name}' already exists",
        )

    persona_dir = existing.persona_dir
    warnings: list[str] = []

    # 1. The persona's own identity, including their own words about
    #    themselves. A prompt reading "You are Alex, a friendly assistant"
    #    on a persona called Alexander is not a cosmetic leftover: the
    #    room preamble opens "You are Alexander" and the two are
    #    concatenated into one system message, so the model is handed a
    #    contradiction about who it is playing. That is worse than any
    #    stale memory, which is why the sweep defaults to on.
    #
    #    Everything downstream is a reference to this, so a failure here
    #    must change nothing else.
    description = existing.description
    system_prompt = existing.system_prompt
    if req.sweep_old_name:
        description = persona_store.replace_name_in_text(description, name, new_name)
        system_prompt = persona_store.replace_name_in_text(system_prompt, name, new_name)
    own_prose_updated = (description, system_prompt) != (
        existing.description, existing.system_prompt
    )

    try:
        persona_store.write_prompt_md(
            persona_dir,
            name=new_name,
            description=description,
            router_hints=existing.router_hints,
            avatar_color=existing.avatar_color,
            allow_tool_calls=existing.allow_tool_calls,
            length_bias=existing.length_bias.value,
            system_prompt=system_prompt,
            memory_size=existing.memory_size,
        )
    except OSError as exc:
        logger.error("Failed to rename persona '%s' in %s: %s", name, persona_dir, exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to rename persona: {exc}",
        ) from exc

    # 2. What everyone else knows about them. Counted per persona so the
    #    response can say what it reached; a persona that has never met
    #    the renamed one contributes nothing and is not reported.
    memories_updated = personas_touched = acquaintances_updated = 0
    for other in config.personas:
        if other.persona_dir is None or not other.persona_dir.is_dir():
            continue
        try:
            changed = persona_store.rename_memory_subjects(
                other.persona_dir, name, new_name, sweep_text=req.sweep_old_name,
            )
        except OSError as exc:
            warnings.append(f"Could not update {other.name}'s memories: {exc}")
            changed = 0
        try:
            met = persona_store.rename_acquaintance(other.persona_dir, name, new_name)
        except OSError as exc:
            warnings.append(f"Could not update who {other.name} has met: {exc}")
            met = False
        memories_updated += changed
        acquaintances_updated += int(met)
        if changed or met:
            personas_touched += 1

    # 3. Room membership.
    rooms_updated = sum(
        1 for room in get_chatrooms().chat_rooms if name in room.persona_names
    )
    _cascade_persona_rename(name, new_name)

    # 4. Who the human is playing. Skipping this would silently drop them
    #    back to playing as themselves, since adopted() resolves against
    #    the live persona list and the old name is no longer in it.
    player_updated = False
    if get_player().persona_name.strip() == name:
        save_player(PlayerConfig(persona_name=new_name))
        player_updated = True

    # 5. Stored transcripts. Attribution only — see the note on
    #    rename_persona_in_history for why the prose is left alone.
    try:
        messages_reattributed = persistence.rename_persona_in_history(name, new_name)
    except OSError as exc:
        warnings.append(f"Could not re-attribute stored messages: {exc}")
        messages_reattributed = 0

    # 6. The folder, last and least.
    moved = persona_store.rename_persona_directory(persona_dir, new_name)
    if moved == persona_dir and persona_dir.name != new_name:
        warnings.append(
            f"The persona was renamed, but its folder is still "
            f"'{persona_dir.name}'."
        )

    # Rebuild from disk rather than patching the cached list: the
    # directory may have moved, and every Persona carries its own path.
    set_personas_cache(PersonasConfig(
        personas=persona_store.scan_personas_directory(get_personas_directory())
    ))

    logger.info(
        "Renamed persona '%s' -> '%s' (%d memory line(s) across %d persona(s), "
        "%d met-list(s), %d room(s), %d message(s), player=%s)",
        name, new_name, memories_updated, personas_touched,
        acquaintances_updated, rooms_updated, messages_reattributed, player_updated,
    )

    return PersonaRenameResponse(
        name=new_name,
        previous_name=name,
        memories_updated=memories_updated,
        personas_touched=personas_touched,
        acquaintances_updated=acquaintances_updated,
        rooms_updated=rooms_updated,
        messages_reattributed=messages_reattributed,
        player_updated=player_updated,
        own_prose_updated=own_prose_updated,
        directory_renamed=moved != persona_dir,
        warnings=warnings,
    )


@router.post("/{name}/condense", response_model=CondenseResponse)
async def condense_memories(name: str, req: CondenseRequest):
    """Rewrite a persona's memories as fewer lines saying the same thing.

    Two steps on purpose. Without *apply* it proposes and writes nothing;
    with it, the lines in the request are written verbatim. Everything
    else in the memory system only adds a line or drops an exact
    duplicate — this rewrites sentences the persona will act on, so it
    does not happen without somebody having read the result.
    """
    config = get_personas()
    persona = next((p for p in config.personas if p.name == name), None)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    if persona.persona_dir is None or not persona.persona_dir.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Persona '{name}' has no directory on disk",
        )

    if req.apply:
        if not req.memories:
            raise HTTPException(
                status_code=422,
                detail="Nothing to apply. Preview the condense first.",
            )
        before = [
            line for line in
            persona_store.read_memories(persona.persona_dir).splitlines()
            if line.strip()
        ]
        try:
            condense.apply(persona, req.memories)
        except OSError as exc:
            logger.error("Failed to condense persona '%s': %s", name, exc)
            raise HTTPException(
                status_code=500, detail=f"Could not write the memories: {exc}",
            ) from exc
        after = [line.strip() for line in req.memories if line.strip()]
        return CondenseResponse(
            persona=name, before=before, after=after,
            before_bytes=condense._bytes(before),
            after_bytes=condense._bytes(after),
            saved_bytes=max(0, condense._bytes(before) - condense._bytes(after)),
            applied=True,
        )

    plan = await condense.plan(persona)
    return CondenseResponse(
        persona=name,
        before=plan.before,
        after=plan.after,
        before_bytes=plan.before_bytes,
        after_bytes=plan.after_bytes,
        saved_bytes=plan.saved_bytes,
        duplicates_removed=plan.duplicates_removed,
        applied=False,
        note=plan.note,
    )


@router.delete("/{name}", status_code=204)
def delete_persona(name: str):
    """Remove a persona's directory and drop it from all chat rooms."""
    config = get_personas()
    persona = next((p for p in config.personas if p.name == name), None)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    if persona.persona_dir is not None and persona.persona_dir.is_dir():
        try:
            shutil.rmtree(persona.persona_dir)
        except OSError as exc:
            logger.error("Failed to remove persona directory %s: %s", persona.persona_dir, exc)
            raise HTTPException(status_code=500, detail=f"Failed to delete persona: {exc}") from exc
    else:
        logger.warning("Deleting persona '%s' with no directory on disk", name)
    set_personas_cache(PersonasConfig(personas=[p for p in config.personas if p.name != name]))
    _cascade_persona_delete(name)


@router.post("/{name}/clone", response_model=PersonaDetailResponse, status_code=201)
def clone_persona(name: str):
    """Clone an existing persona, appending a numeric suffix to ensure uniqueness.

    The clone gets its own directory (a copy of the source's files), so
    editing one never affects the other.
    """
    config = get_personas()
    source = next((p for p in config.personas if p.name == name), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    if source.persona_dir is None or not source.persona_dir.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Persona '{name}' has no directory on disk; cannot clone it",
        )

    # The suffix has to fit inside the 25-character cap the create/update
    # forms enforce, or the clone is born un-editable: PUT would reject the
    # very name it was saved under. Trim the base, not the suffix, so the
    # result stays unique.
    existing_names = {p.name.lower() for p in config.personas}
    suffix = 2
    while True:
        tail = f"_{suffix}"
        new_name = f"{name[: MAX_PERSONA_NAME - len(tail)].rstrip()}{tail}"
        if new_name.lower() not in existing_names:
            break
        suffix += 1

    root = get_personas_directory()
    new_dir = root / persona_store.unique_persona_dirname(
        root, persona_store.sanitize_persona_dirname(new_name)
    )
    try:
        shutil.copytree(source.persona_dir, new_dir)
        # The copy still carries the source's frontmatter; rewrite it with
        # the clone's name (build_prompt_md adds the `name` field when it
        # differs from the directory name).
        persona_store.write_prompt_md(
            new_dir,
            name=new_name,
            description=source.description,
            router_hints=source.router_hints,
            avatar_color=source.avatar_color,
            allow_tool_calls=source.allow_tool_calls,
            length_bias=source.length_bias.value,
            system_prompt=source.system_prompt,
            # Must be carried over explicitly: build_prompt_md omits the
            # key when memory_size is None, and a clone that lost its
            # budget would quietly fall back to the default on reload.
            memory_size=source.memory_size,
        )
        clone = persona_store.load_persona_from_dir(new_dir)
    except OSError as exc:
        _remove_persona_dir(new_dir)
        logger.error("Failed to clone persona '%s': %s", name, exc)
        raise HTTPException(status_code=500, detail=f"Failed to clone persona: {exc}") from exc

    set_personas_cache(PersonasConfig(personas=config.personas + [clone]))
    return _to_detail(clone)


@router.get("/{name}/avatar")
async def get_avatar(name: str):
    """Serve a persona's avatar image file.

    Returns 404 if the persona has no avatar configured or the file
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


@router.get("/{name}/reference-audio")
async def get_reference_audio(name: str):
    """Serve a persona's reference audio file (ref.wav).

    Returns 404 if the persona has no reference audio configured or the
    file doesn't exist on disk.
    """
    config = get_personas()
    persona = next((p for p in config.personas if p.name == name), None)
    if not persona or not persona.reference_audio:
        return Response(status_code=404, content="No reference audio configured")

    path = Path(persona.reference_audio)
    if not path.exists():
        logger.warning("Reference audio file not found for %s: %s", name, persona.reference_audio)
        return Response(status_code=404, content="Reference audio file not found")

    return FileResponse(str(path), media_type="audio/wav")


# ---------------------------------------------------------------------------
# Drafting a persona with the LLM
# ---------------------------------------------------------------------------
#
# Same contract as the suggested player message: the model drafts, the
# result lands in the form, and nothing touches disk until the human
# presses Save.

# A full draft is ~120 words of prompt plus notes and the labelled fields.
_DRAFT_MAX_TOKENS = 1400
# One sample reply. Generous enough that a "much_longer" persona is not
# cut off mid-demonstration, which would misrepresent it.
_PREVIEW_MAX_TOKENS = 400

# What the human is called in a preview. One constant, because the same
# name has to reach the transcript tag, the stop strings and the guard —
# a persona told not to speak as "User" while the tag says something else
# is the bug this names away. A preview is not a room, so it is always the
# neutral label: whoever the player has adopted is not in this scene.
_PREVIEW_USER = DEFAULT_USER_LABEL


@router.post("/draft", response_model=PersonaDraftResponse)
async def draft_persona(req: PersonaDraftRequest):
    """Draft a persona from a brief plus whatever dials and details were set.

    The existing cast is read only to keep the name unique; none of it
    goes to the model. Distinctness comes from the specification, not from
    contrast with whoever already exists — so the cost of a draft does not
    grow with the number of personas.
    """
    settings = get_settings()
    existing = list(get_personas().personas)
    spec = persona_draft.PersonaSpec.from_request(req.brief, req.dials, req.details)

    text = await chat_completion(
        # The existing names go in so the model does not land on one that
        # is taken; the rename below is the backstop, not the plan.
        persona_draft.build_draft_prompt(spec, [p.name for p in existing]),
        max_tokens=_DRAFT_MAX_TOKENS,
        # Prose, not routing: the router's 0.1 produces four drafts that
        # are the same draft, and the router's timeout is sized for
        # sixteen tokens, not a hundred-odd words.
        temperature=settings.llm.temperature,
        timeout=PROSE_TIMEOUT,
    )
    if not text.strip():
        raise HTTPException(
            status_code=503,
            detail="The LLM returned nothing. Is the server running?",
        )

    draft = persona_draft.parse_draft(text)

    # The floor, enforced rather than asked for. The brief used as-is beat
    # everything the generator ever wrote from it, so a reply that has
    # written its own character over the top is not an improvement to
    # weigh up — it is a regression, and the user's words go back in.
    kept = persona_draft.kept_fraction(req.brief, draft.system_prompt)
    if draft.system_prompt.strip() and kept < persona_draft._KEPT_WORDS_FLOOR:
        logger.info(
            "Draft kept only %.0f%% of the brief; using the brief itself", kept * 100
        )
        draft.system_prompt = persona_draft.prompt_from_brief(spec)
        draft.notes.append(
            "The draft rewrote your description rather than putting it in the "
            "second person, so your own words were kept instead."
        )

    if not draft.is_usable():
        logger.info("Unusable persona draft returned: %.400s", text)
        raise HTTPException(
            status_code=503,
            detail=(
                "The draft came back in a shape this could not read. Try again, "
                "or give the brief a bit more to work with."
            ),
        )

    # Names must be unique. The model is told which are taken, so reaching
    # here means it ignored that — and "Alex_2" is a poor name to hand
    # somebody without saying why they got it.
    taken = {p.name.lower() for p in existing}
    if draft.name.lower() in taken:
        base = draft.name[: persona_draft.MAX_NAME - 2].rstrip()
        suffix = 2
        while f"{base}_{suffix}".lower() in taken:
            suffix += 1
        draft.notes.append(
            f"The draft chose the name {draft.name}, which is already in use, "
            f"so it became {base}_{suffix}. Worth renaming by hand."
        )
        draft.name = f"{base}_{suffix}"

    return PersonaDraftResponse(
        name=draft.name,
        description=draft.description,
        system_prompt=draft.system_prompt,
        router_hints=draft.router_hints or "general conversation",
        length_bias=draft.length_bias,
        avatar_color=draft.avatar_color,
        notes=draft.notes,
        warnings=persona_draft.critique(draft),
    )


@router.post("/refine", response_model=PersonaRefineResponse)
async def refine_persona(req: PersonaRefineRequest):
    """Revise an existing persona from one instruction about what to change.

    The inverse of drafting: there the risk is a character with no shape,
    here it is losing a shape that already works. So the model is given
    the persona whole, told to change only what was asked and to keep the
    rest word for word, and its reply is parsed *over* the current values
    — an omitted block leaves that field exactly as it was rather than
    blanking it.
    """
    settings = get_settings()
    current = persona_draft.PersonaDraft(
        name=req.name,
        description=req.description,
        system_prompt=req.system_prompt,
        router_hints=req.router_hints,
        length_bias=req.length_bias,
    )

    text = await chat_completion(
        persona_draft.build_refine_prompt(current, req.instruction),
        max_tokens=_DRAFT_MAX_TOKENS,
        temperature=settings.llm.temperature,
        timeout=PROSE_TIMEOUT,
    )
    if not text.strip():
        raise HTTPException(
            status_code=503,
            detail="The LLM returned nothing. Is the server running?",
        )

    revised = persona_draft.parse_draft(text, base=current)
    # A reply that changed nothing at all is a failure worth naming: the
    # user asked for something and would otherwise see an unchanged form
    # and no explanation.
    if revised.system_prompt.strip() == current.system_prompt.strip() and not revised.notes:
        logger.info("Refinement returned nothing usable: %.400s", text)
        raise HTTPException(
            status_code=503,
            detail=(
                "The revision came back in a shape this could not read. Try again, "
                "or say more plainly what should change."
            ),
        )

    return PersonaRefineResponse(
        description=revised.description,
        system_prompt=revised.system_prompt,
        router_hints=revised.router_hints,
        length_bias=revised.length_bias,
        notes=revised.notes,
        warnings=persona_draft.critique(revised),
    )


async def _preview_reply(persona: Persona, question: str) -> str:
    """One reply from *persona*, built exactly as a real turn would be.

    Deliberately reuses the chat router's own preamble builder. A preview
    assembled from a simpler prompt would be a preview of something the
    app never runs — including the voice restatement at the end, which is
    a large part of why a persona sounds like itself at all.

    Both sides of a comparison run in a room of one, so the only thing
    that differs between them is the persona's own prompt. Putting each in
    the other's roster would have been more lifelike and less useful: an
    unsaved draft cannot appear in a saved persona's roster anyway, so the
    two prompts would have differed in a second way and the comparison
    would no longer isolate the thing being compared.
    """
    # Local import: this is the only place personas reaches into chat, and
    # a module-level import would be a cycle waiting to happen.
    from app.routers.chat import _build_room_preamble

    settings = get_settings()
    length = resolve_typical_length(persona, None, settings.general.typical_length)
    preamble = _build_room_preamble(persona, "default", [persona.name], length)
    messages = [
        {"role": "system", "content": f"{persona.system_prompt}\n\n{preamble}"},
        {"role": "user", "content": f"[{_PREVIEW_USER}]: {question}"},
    ]
    # Both layers the room uses, for the same reason it uses them: without
    # the stop strings a transcript-mode prompt ends at "[Leo]:" and the
    # model writes the whole scene — the user's next line, its own reply to
    # that, and any character it needs to invent to fill the room. The
    # guard then catches what the stop strings cannot, which is exactly the
    # invented ones, since a stop string can only name a speaker we know.
    text = await chat_completion(
        messages,
        max_tokens=derive_max_tokens(length, min(settings.llm.max_tokens, _PREVIEW_MAX_TOKENS)),
        temperature=settings.llm.temperature,
        timeout=PROSE_TIMEOUT,
        # Auditioned the same way they will be played: a persona previewed
        # through the instruct template and then run as a transcript is a
        # preview of a different character.
        persona_name=persona.name,
        stop=stop_sequences(persona.name, [_PREVIEW_USER]),
    )
    guard = ReplyGuard(persona.name, [_PREVIEW_USER])
    return guard.feed(text) + guard.flush()


@router.post("/preview", response_model=PersonaPreviewResponse)
async def preview_persona(req: PersonaPreviewRequest):
    """Answer one question as an unsaved draft, and optionally as a real persona.

    The comparison is the point. A draft read on its own always sounds
    plausible; read beside an existing persona answering the same
    question, "these two are the same character" is obvious at a glance.
    The comparison side is either a saved persona (drafting) or an unsaved
    prompt (refining, where the pair is one character before and after).
    """
    draft = Persona(
        name=_validate_name(req.name),
        description=req.description,
        system_prompt=req.system_prompt,
        router_hints="preview",
        length_bias=req.length_bias,
    )

    other: Optional[Persona] = None
    other_label = ""
    if req.compare_with:
        other = next(
            (p for p in get_personas().personas if p.name == req.compare_with), None
        )
        if other is None:
            raise HTTPException(
                status_code=404, detail=f"Persona '{req.compare_with}' not found"
            )
        other_label = req.compare_label.strip() or other.name
    elif req.compare_prompt and req.compare_prompt.strip():
        # A before-and-after of one character. Same name as the other
        # side, deliberately: the room preamble is built from the persona,
        # so a different name would make the two prompts differ in a
        # second way and the comparison would stop isolating the change.
        other = draft.model_copy(update={
            "system_prompt": req.compare_prompt,
            "length_bias": req.compare_length_bias,
        })
        other_label = req.compare_label.strip() or f"{draft.name} (before)"

    draft_text = await _preview_reply(draft, req.question)
    if not draft_text.strip():
        raise HTTPException(
            status_code=503,
            detail="The LLM returned nothing for the draft. Is the server running?",
        )

    comparison = None
    if other is not None:
        other_text = await _preview_reply(other, req.question)
        if other_text.strip():
            comparison = PersonaPreviewReply(persona=other_label, text=other_text.strip())

    return PersonaPreviewResponse(
        draft=PersonaPreviewReply(
            persona=req.label.strip() or draft.name, text=draft_text.strip()
        ),
        comparison=comparison,
    )
