"""File-based persona storage — the "Personas directory".

Each persona lives in its own subdirectory of the configured Personas
directory (``general.personas_directory`` in settings.yaml, default
``<project root>/Personas``)::

     Personas/
       Alex/
         prompt.md       YAML frontmatter (name, description, router_hints,
                         avatar_color, allow_tool_calls, memory_size) +
                         system prompt body
         language.txt    2-letter code for the reference audio (optional)
         ref.wav         Reference audio for TTS voice cloning (optional)
         ref.txt         Transcript of ref.wav (optional)
         memories.txt    Saved persona memories, one per line (optional)
         image.png       Avatar image (optional; png/jpg/jpeg/gif/webp)

This module is framework-agnostic on purpose: it reads and writes files
and knows nothing about FastAPI, the config cache, or the frontend. The
router (app/routers/personas.py) translates HTTP into these operations
and refreshes the in-memory persona cache afterwards. The directory on
disk is the source of truth.
"""

import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import NamedTuple, Dict, Iterable, List, Optional, Set, Tuple

import yaml
from pydantic import ValidationError

from app.config import (
    DEFAULT_MEMORY_SIZE,
    MAX_MEMORY_LINE_CHARS,
    MAX_MEMORY_SIZE,
    MAX_PERSONA_NAME,
    LengthBias,
    Persona,
)
from app.config_migrations import migrate_personas

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File layout constants
# ---------------------------------------------------------------------------

PROMPT_FILENAME = "prompt.md"
LANGUAGE_FILENAME = "language.txt"
REFERENCE_AUDIO_FILENAME = "ref.wav"
TRANSCRIPT_FILENAME = "ref.txt"
MEMORIES_FILENAME = "memories.txt"
ACQUAINTANCES_FILENAME = "met.txt"
IMAGE_BASENAME = "image"
DEFAULT_LANGUAGE = "en"

# The browser can decode all of these natively; anything else is rejected
# at upload time (422) and skipped at migration time (warning).
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
# Reference audio is wav-only: the TTS servers we support resample from
# wav, and a single extension keeps the on-disk layout unambiguous.
AUDIO_EXTENSION = ".wav"

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 20 * 1024 * 1024


class PersonaStorageError(RuntimeError):
    """Fatal problem with the Personas directory itself (not one persona)."""


class PersonaMigrationError(PersonaStorageError):
    """The one-time personas.yaml -> directory migration failed fatally."""


# ---------------------------------------------------------------------------
# prompt.md frontmatter
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> Tuple[dict, str]:
    """Split a prompt.md into (frontmatter dict, system prompt body).

    Frontmatter is optional: the file must start with a lone ``---`` line
    and close with another ``---`` line. Anything malformed degrades to
    ``({}, whole file)`` — a broken prompt.md should not take the app down
    at startup; it just loses its structured fields.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip("\n")
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() != "---":
            continue
        try:
            data = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError as exc:
            logger.warning("Malformed prompt.md frontmatter; ignoring it: %s", exc)
            return {}, text.strip("\n")
        if not isinstance(data, dict):
            logger.warning("prompt.md frontmatter is not a mapping; ignoring it")
            return {}, text.strip("\n")
        return data, "\n".join(lines[end + 1:]).strip("\n")
    # Opening delimiter but no closing one: treat the whole file as body.
    return {}, text.strip("\n")


def build_prompt_md(
    dir_name: str,
    *,
    name: str,
    description: str,
    router_hints: str,
    avatar_color: str,
    allow_tool_calls: bool,
    length_bias: str = LengthBias.MATCH.value,
    system_prompt: str,
    memory_size: Optional[int] = None,
) -> str:
    """Serialize a persona's frontmatter + system prompt for prompt.md.

    The ``name`` field is written only when it differs from the directory
    name — the directory name is the fallback identity, so duplicating it
    would be noise. (This is also how names with directory-hostile
    characters keep working: the dir is "OBrien", the frontmatter says
    "O'Brien".)

    ``memory_size`` is written only when explicitly provided: the legacy
    migration passes None so upgraded personas carry no new key (missing
    means "default" at load time), while the editor always passes an int.
    """
    frontmatter: dict = {}
    if name != dir_name:
        frontmatter["name"] = name
    frontmatter["description"] = description
    frontmatter["router_hints"] = router_hints
    frontmatter["avatar_color"] = avatar_color
    frontmatter["allow_tool_calls"] = bool(allow_tool_calls)
    frontmatter["length_bias"] = str(length_bias)
    if memory_size is not None:
        frontmatter["memory_size"] = memory_size
    dumped = yaml.dump(
        frontmatter, default_flow_style=False, sort_keys=False, allow_unicode=True
    ).strip()
    return f"---\n{dumped}\n---\n\n{system_prompt}\n"


# ---------------------------------------------------------------------------
# Directory names
# ---------------------------------------------------------------------------

_DIRNAME_STRIP_PATTERN = re.compile(r"[^a-zA-Z0-9 _-]")


def sanitize_persona_dirname(name: str) -> str:
    """Reduce a persona name to a safe directory name.

    Same allowed character set as chat room names (letters, numbers,
    spaces, hyphens, underscores); everything else is stripped. May
    return "" — the caller decides what that means (422 for a new
    persona, a fallback for migrated legacy data).
    """
    return _DIRNAME_STRIP_PATTERN.sub("", name)


def unique_persona_dirname(root: Path, base: str) -> str:
    """Pick ``base``, or the first free ``base_2``, ``base_3``, ... in root.

    Suffixes start at 2 and use the same ``Name_2`` convention as the
    clone endpoint, so a collision already reads as a duplicate.
    """
    candidate = base
    suffix = 2
    while (root / candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


# ---------------------------------------------------------------------------
# Loading (read-only; never mutates the directory)
# ---------------------------------------------------------------------------

def read_language_file(persona_dir: Path, persona_name: str) -> str:
    """Read the 2-letter language code, defaulting (with a warning) to 'en'."""
    path = persona_dir / LANGUAGE_FILENAME
    if not path.is_file():
        logger.warning(
            "Persona %s has no %s; defaulting to '%s'",
            persona_name, LANGUAGE_FILENAME, DEFAULT_LANGUAGE,
        )
        return DEFAULT_LANGUAGE
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        logger.warning(
            "Persona %s: unreadable %s (%s); defaulting to '%s'",
            persona_name, LANGUAGE_FILENAME, exc, DEFAULT_LANGUAGE,
        )
        return DEFAULT_LANGUAGE
    if len(value) != 2:
        logger.warning(
            "Persona %s has invalid language code %r in %s; defaulting to '%s'",
            persona_name, value, LANGUAGE_FILENAME, DEFAULT_LANGUAGE,
        )
        return DEFAULT_LANGUAGE
    return value


def find_avatar_file(persona_dir: Path, persona_name: str) -> Optional[Path]:
    """Locate the persona's avatar image (image.png, image.webp, ...)."""
    candidates = sorted(
        path
        for path in persona_dir.glob(f"{IMAGE_BASENAME}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            "Persona %s has multiple image files; using %s",
            persona_name, candidates[0].name,
        )
    return candidates[0]


def _read_length_bias(frontmatter: dict, persona_name: str) -> LengthBias:
    """The persona's length bias from frontmatter, defaulting to MATCH.

    A missing key is normal (a persona written before the field existed).
    An unrecognised one is a hand-edit typo: warn and fall back, rather
    than raising and making the whole persona disappear from the list.
    """
    raw = frontmatter.get("length_bias")
    if raw is None:
        return LengthBias.MATCH
    try:
        return LengthBias(str(raw).strip().lower())
    except ValueError:
        logger.warning(
            "Persona %s has an unrecognised length_bias %r; using 'match'. "
            "Valid values: %s",
            persona_name, raw, ", ".join(b.value for b in LengthBias),
        )
        return LengthBias.MATCH


def load_persona_from_dir(persona_dir: Path) -> Persona:
    """Build a Persona from one subdirectory of the Personas directory.

    Raises PersonaStorageError when the directory has no readable
    prompt.md — the scanner (scan_personas_directory) logs and skips such
    directories rather than failing the whole load.
    """
    prompt_path = persona_dir / PROMPT_FILENAME
    if not prompt_path.is_file():
        raise PersonaStorageError(f"missing {PROMPT_FILENAME}")
    try:
        text = prompt_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PersonaStorageError(f"unreadable {PROMPT_FILENAME}: {exc}") from exc

    frontmatter, system_prompt = parse_frontmatter(text)
    name = str(frontmatter.get("name") or "").strip() or persona_dir.name

    avatar_path = find_avatar_file(persona_dir, name)

    audio_path = persona_dir / REFERENCE_AUDIO_FILENAME
    reference_audio = str(audio_path) if audio_path.is_file() else None

    transcript_path = persona_dir / TRANSCRIPT_FILENAME
    reference_transcript: Optional[str] = None
    if transcript_path.is_file():
        try:
            transcript = transcript_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Persona %s: unreadable %s: %s", name, TRANSCRIPT_FILENAME, exc)
            transcript = ""
        if transcript.strip():
            reference_transcript = str(transcript_path)
        else:
            # An empty transcript makes TTS voice cloning meaningless, so
            # the persona reports as not-TTS-capable instead of silently
            # synthesizing with an empty prompt.
            logger.warning(
                "Persona %s has an empty %s; persona will be treated as not TTS-capable",
                name, TRANSCRIPT_FILENAME,
            )

    allow_tool_calls = bool(frontmatter.get("allow_tool_calls") or False)
    memory_size = parse_memory_size(frontmatter.get("memory_size"), name)
    # Diagnostic trail (DEBUG): shows exactly what the runtime parsed out
    # of the frontmatter (raw value AND sanitized value), so a stale/
    # mis-parsed prompt.md is diagnosable instead of surfacing as a silent
    # "no tools".
    logger.debug(
        "Persona memory: loaded '%s' from %s: allow_tool_calls=%s, memory_size=%d "
        "(raw frontmatter: allow_tool_calls=%r, memory_size=%r)",
        name, persona_dir, allow_tool_calls, memory_size,
        frontmatter.get("allow_tool_calls"), frontmatter.get("memory_size"),
    )
    return Persona(
        name=name,
        description=str(frontmatter.get("description") or ""),
        system_prompt=system_prompt,
        router_hints=str(frontmatter.get("router_hints") or ""),
        avatar_color=str(frontmatter.get("avatar_color") or "#888888"),
        avatar_image=str(avatar_path) if avatar_path else None,
        reference_audio=reference_audio,
        reference_audio_transcript=reference_transcript,
        reference_audio_language=read_language_file(persona_dir, name),
        allow_tool_calls=allow_tool_calls,
        length_bias=_read_length_bias(frontmatter, name),
        memory_size=memory_size,
        persona_dir=persona_dir,
    )


def scan_personas_directory(root: Path) -> List[Persona]:
    """Load every persona subdirectory under root, in directory-name order.

    Unrecognized top-level files are ignored (they may be OS junk like
    .DS_Store or editor swap files). Subdirectories that fail to load
    are skipped with a warning — one broken persona should not blind the
    rest.
    """
    if not root.is_dir():
        return []
    personas: List[Persona] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        try:
            personas.append(load_persona_from_dir(entry))
        except PersonaStorageError as exc:
            logger.warning("Skipping persona directory %s: %s", entry.name, exc)
    return personas


# ---------------------------------------------------------------------------
# Writing (used by the router for create/update and by the migration)
# ---------------------------------------------------------------------------

def write_prompt_md(
    persona_dir: Path,
    *,
    name: str,
    description: str,
    router_hints: str,
    avatar_color: str,
    allow_tool_calls: bool,
    length_bias: str = LengthBias.MATCH.value,
    system_prompt: str,
    memory_size: Optional[int] = None,
) -> None:
    (persona_dir / PROMPT_FILENAME).write_text(
        build_prompt_md(
            persona_dir.name,
            name=name,
            description=description,
            router_hints=router_hints,
            avatar_color=avatar_color,
            allow_tool_calls=allow_tool_calls,
            length_bias=length_bias,
            system_prompt=system_prompt,
            memory_size=memory_size,
        ),
        encoding="utf-8",
    )


def write_language_file(persona_dir: Path, language: str) -> None:
    (persona_dir / LANGUAGE_FILENAME).write_text(language, encoding="utf-8")


def write_transcript_file(persona_dir: Path, text: str) -> None:
    """Write the transcript, or delete ref.txt when the text is blank.

    The file is stored stripped: a transcript that is only whitespace is
    the same as no transcript, and keeping the file would hide that from
    the UI (the detail endpoint reports file contents).
    """
    path = persona_dir / TRANSCRIPT_FILENAME
    stripped = text.strip()
    if not stripped:
        if path.exists():
            path.unlink()
        return
    path.write_text(stripped, encoding="utf-8")


def write_avatar_file(persona_dir: Path, data: bytes, extension: str) -> Path:
    """Store a new avatar as image.<ext>, replacing any existing avatar."""
    remove_avatar_file(persona_dir)
    target = persona_dir / f"{IMAGE_BASENAME}{extension.lower()}"
    target.write_bytes(data)
    return target


def write_reference_audio_file(persona_dir: Path, data: bytes) -> Path:
    target = persona_dir / REFERENCE_AUDIO_FILENAME
    target.write_bytes(data)
    return target


def remove_avatar_file(persona_dir: Path) -> bool:
    """Delete the persona's avatar image if present. Returns True if removed."""
    removed = False
    for path in persona_dir.glob(f"{IMAGE_BASENAME}.*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            path.unlink()
            removed = True
    return removed


def remove_reference_audio_file(persona_dir: Path) -> bool:
    """Delete the persona's ref.wav if present. Returns True if removed."""
    path = persona_dir / REFERENCE_AUDIO_FILENAME
    if path.is_file():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Persona memories (memories.txt) — docs/feature_persona_memory.md
# ---------------------------------------------------------------------------

def parse_memory_size(raw: object, persona_name: str) -> int:
    """Sanitize a memory_size value read from prompt.md frontmatter.

    Missing values get the default (legacy personas predate the field).
    Invalid values — non-int (including bool, which *is* an int subclass),
    negative, or above MAX_MEMORY_SIZE — get the default too, with a
    warning. A bad value in one persona's frontmatter must never take the
    whole app down at startup.
    """
    if raw is None:
        return DEFAULT_MEMORY_SIZE
    if isinstance(raw, bool) or not isinstance(raw, int):
        logger.warning(
            "Persona %s has an invalid memory_size %r in prompt.md frontmatter; "
            "assuming the default (%d)", persona_name, raw, DEFAULT_MEMORY_SIZE,
        )
        return DEFAULT_MEMORY_SIZE
    if raw < 0 or raw > MAX_MEMORY_SIZE:
        logger.warning(
            "Persona %s has an out-of-range memory_size %r (allowed 0..%d); "
            "assuming the default (%d)",
            persona_name, raw, MAX_MEMORY_SIZE, DEFAULT_MEMORY_SIZE,
        )
        return DEFAULT_MEMORY_SIZE
    return raw


# ---------------------------------------------------------------------------
# Who a persona has met
# ---------------------------------------------------------------------------
#
# Written by the app, not by the model, and that is the whole point. An
# empty memories file cannot tell you whether two characters have never
# met or simply had nothing worth writing down, so "you have not met them
# before" was not a claim the app could honestly make. A turn taken in a
# room is an encounter whether or not anything memorable came of it, so
# the app records it directly — which also means a first meeting reads as
# a first meeting without the memory feature being switched on at all.

def read_acquaintances(persona_dir: Path) -> Set[str]:
    """Everyone this persona has shared a room with. Empty when unknown."""
    path = persona_dir / ACQUAINTANCES_FILENAME
    if not path.is_file():
        return set()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Persona %s: unreadable %s (%s)",
                       persona_dir.name, ACQUAINTANCES_FILENAME, exc)
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


def record_acquaintances(persona_dir: Path, names: Iterable[str]) -> Set[str]:
    """Note that this persona has now met *names*. Returns the full set.

    Best-effort: failing to record an encounter must never break a reply,
    it only means the next conversation starts as another first meeting.
    """
    known = read_acquaintances(persona_dir)
    fresh = {n.strip() for n in names if n and n.strip()} - known
    if not fresh:
        return known
    known |= fresh
    try:
        (persona_dir / ACQUAINTANCES_FILENAME).write_text(
            "\n".join(sorted(known)) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Persona %s: could not record %s: %s",
                       persona_dir.name, ACQUAINTANCES_FILENAME, exc)
        return known - fresh
    return known


def forget_acquaintances(persona_dir: Path) -> bool:
    """Delete the met-list. True if there was one."""
    path = persona_dir / ACQUAINTANCES_FILENAME
    if path.is_file():
        path.unlink()
        return True
    return False


def read_memories(persona_dir: Path) -> str:
    """Read the persona's memories file, or "" when absent/unreadable.

    Callers must treat the result as best-effort: memory injection simply
    does not happen when there is nothing readable.
    """
    path = persona_dir / MEMORIES_FILENAME
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Persona %s: unreadable %s (%s)", persona_dir.name, MEMORIES_FILENAME, exc)
        return ""


def remove_memories_file(persona_dir: Path) -> bool:
    """Delete the persona's memories.txt if present. Returns True if removed.

    Can raise OSError when the file exists but cannot be deleted (e.g. a
    read-only directory). Callers on a never-raises path (append_memory,
    purge_memories_to_limit) must use _remove_memories_file_best_effort.
    """
    path = persona_dir / MEMORIES_FILENAME
    if path.is_file():
        path.unlink()
        return True
    return False


def _remove_memories_file_best_effort(persona_dir: Path) -> bool:
    """remove_memories_file() that never raises (best-effort cleanup).

    For the never-raises paths (append_memory's disabled-memory cleanup,
    purge_memories_to_limit), where a disk error must not break a tool
    call stream or a chat request. The personas router's explicit-clear
    path deliberately uses the raising variant instead: there a failed
    clear should surface as a 500, not a silent no-op.
    Returns False when the file could not be deleted.
    """
    try:
        return remove_memories_file(persona_dir)
    except OSError as exc:
        logger.warning(
            "Persona %s: could not delete %s: %s",
            persona_dir.name, MEMORIES_FILENAME, exc,
        )
        return False


def _memory_lines(content: str) -> List[str]:
    """Split memories-file content into its non-blank lines, oldest first."""
    return [line.strip() for line in content.splitlines() if line.strip()]


def _memories_content(lines: List[str]) -> str:
    """Serialize memory lines back to file content (one memory per line)."""
    return "".join(line + "\n" for line in lines)


def _memories_bytes(lines: List[str]) -> int:
    """UTF-8 byte size of the file the given lines would produce."""
    return len(_memories_content(lines).encode("utf-8"))


def _write_memories_file(persona_dir: Path, lines: List[str]) -> None:
    """Atomically rewrite memories.txt (temp file + os.replace).

    The rename is atomic on POSIX, so a crash mid-write leaves either the
    old file or the new one — never a half-written file.
    """
    target = persona_dir / MEMORIES_FILENAME
    tmp = persona_dir / f"{MEMORIES_FILENAME}.tmp{uuid.uuid4().hex[:8]}"
    try:
        tmp.write_text(_memories_content(lines), encoding="utf-8")
        os.replace(tmp, target)
    except BaseException:
        # Never leave a temp file behind; the error itself propagates.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# A memory is about somebody, and the line says who: "[Tony] ...". The
# subject is whatever the transcript calls them, so the human is filed
# under the persona they are playing rather than under "the user" — play
# somebody else tomorrow and you are somebody else to the room.
#
# Untagged lines are legacy, from when every memory was about the human by
# definition. They are read as belonging to no one in particular and shown
# whoever is present, which is what they used to do.
# The subject of a memory is somebody's name, so the cap IS the name cap:
# a longer subject could not be written and read back.
MAX_SUBJECT_CHARS = MAX_PERSONA_NAME
_SUBJECT_RE = re.compile(r"^\[([^\]\n]{1,%d})\]\s*(.+)$" % MAX_SUBJECT_CHARS)


# How a memory records that it was worked out rather than witnessed.
#
# A persona that decides somebody is about forty, and files "Tony is
# forty", meets them next week believing it the way it believes anything
# else it was told. That is not what happened, and the persona has no way
# to find out — an inference is indistinguishable from a fact once both
# are prose in the same file.
#
# So the line carries which it is, and the prompt says it back (see
# _who_is_here_block). It is a word rather than a symbol because
# memories.txt is meant to be opened and edited by hand, and "(assumed)"
# explains itself where "~" would need a legend.
ASSUMED_PREFIX = "(assumed)"

# What a model writes when asked to mark an assumption. Only these: every
# extra spelling is a phrase that stops being part of the memory's text,
# and quietly losing words out of a memory is worse than an unmarked one.
_ASSUMED_RE = re.compile(r"^\(\s*(?:assumed|assumption|guess|guessed)\s*\)\s*", re.I)


class Memory(NamedTuple):
    """One stored line, pulled apart.

    *subject* is "" for the untagged legacy lines that predate memories
    being about anybody. *assumed* is True when the persona worked this
    out rather than being told it.
    """

    subject: str
    text: str
    assumed: bool = False

    def stored(self) -> str:
        """The line as it is written to disk."""
        body = f"{ASSUMED_PREFIX} {self.text}" if self.assumed else self.text
        return f"[{self.subject}] {body}" if self.subject else body


def parse_memory_line(line: str) -> Memory:
    """A stored line as a Memory. Never raises; anything unparseable is
    treated as an untagged memory, which is what it looks like."""
    match = _SUBJECT_RE.match(line.strip())
    subject, body = (
        (match.group(1).strip(), match.group(2).strip()) if match else ("", line.strip())
    )
    stripped = _ASSUMED_RE.sub("", body, count=1)
    return Memory(subject=subject, text=stripped.strip(), assumed=stripped != body)


def split_memory_line(line: str) -> Tuple[str, str]:
    """A stored line as (subject, text), dropping any assumed marker."""
    memory = parse_memory_line(line)
    return memory.subject, memory.text


def memories_by_subject(persona_dir: Path) -> Dict[str, List[Memory]]:
    """Stored memories grouped by who they are about, casefolded keys.

    Untagged legacy lines land under "". Callers get whole Memory rows
    rather than bare text because what a persona *assumed* has to be
    presented differently from what it knows.
    """
    grouped: Dict[str, List[Memory]] = {}
    for line in _memory_lines(read_memories(persona_dir)):
        memory = parse_memory_line(line)
        grouped.setdefault(memory.subject.casefold(), []).append(memory)
    return grouped


# ---------------------------------------------------------------------------
# Renaming a persona
# ---------------------------------------------------------------------------
#
# Names are the identifiers here, and deliberately so: the name is not
# just a key, it is what the model reads and writes. A memory's subject
# tag, a met-list entry and a transcript tag are all the same string, and
# they have to be, or the mechanical containment layers (stop sequences,
# ReplyGuard) would be guarding a different name from the one in the
# prompt.
#
# The price of that choice is that a rename has to be a real operation
# rather than an edit to one field, which is what these helpers are for.


def replace_name_in_text(text: str, old_name: str, new_name: str) -> str:
    """Swap *old_name* for *new_name* where it stands as a whole word.

    Whole-word and case-sensitive: a name is a proper noun, so matching
    loosely would rewrite "alex" inside "alexandrite". It still cannot be
    made safe in general — a character called Will, May or Mark shares a
    spelling with an ordinary word — which is why every caller of this
    puts it behind a choice rather than doing it silently.
    """
    old, new = old_name.strip(), new_name.strip()
    if not old or not new or old == new or not text:
        return text
    return re.sub(r"\b%s\b" % re.escape(old), new, text)


def rename_memory_subjects(
    persona_dir: Path, old_name: str, new_name: str, sweep_text: bool = True,
) -> int:
    """Repoint one persona's memories from *old_name* to *new_name*.

    Two different jobs, and only the first is unambiguous:

      * the ``[Subject]`` tag is structural, matched whole and
        case-insensitively — this is the one that decides whose memory it
        is, and getting it wrong orphans the line;
      * the name *inside* the memory is prose the model wrote ("Alex has
        never been on a boat"). Left alone it turns a renamed character
        into a memory about somebody who no longer exists, which reads
        worse than no memory at all — so it is swept too, whole-word and
        case-sensitively.

    That second sweep is optional because it cannot be made safe in
    general: a persona called Will, May or Mark shares a spelling with an
    ordinary word, and no amount of care distinguishes "Will you pass the
    salt" from the character. The caller decides; the count comes back so
    the decision is visible rather than silent.

    Returns the number of lines changed. Raises OSError on a failed write.
    """
    old, new = old_name.strip(), new_name.strip()
    if not old or not new or old == new:
        return 0

    lines = _memory_lines(read_memories(persona_dir))
    if not lines:
        return 0

    changed, rewritten = 0, []
    for line in lines:
        subject, text = split_memory_line(line)
        was = (subject, text)
        if subject.casefold() == old.casefold():
            subject = new
        if sweep_text:
            text = replace_name_in_text(text, old, new)
        if (subject, text) != was:
            changed += 1
        rewritten.append(f"[{subject}] {text}" if subject else text)

    if changed:
        _write_memories_file(persona_dir, rewritten)
    return changed


def rename_acquaintance(persona_dir: Path, old_name: str, new_name: str) -> bool:
    """Repoint one persona's met-list entry. True if it knew *old_name*.

    Losing this would be quieter than losing a memory and worse: the
    persona would meet a character it has known for weeks and be told
    they have never met.

    Raises OSError on a failed write.
    """
    old, new = old_name.strip(), new_name.strip()
    if not old or not new or old == new:
        return False

    known = read_acquaintances(persona_dir)
    matches = {n for n in known if n.casefold() == old.casefold()}
    if not matches:
        return False

    known = (known - matches) | {new}
    (persona_dir / ACQUAINTANCES_FILENAME).write_text(
        "\n".join(sorted(known)) + "\n", encoding="utf-8"
    )
    return True


def rename_persona_directory(persona_dir: Path, new_name: str) -> Path:
    """Move a persona's directory to match its new name.

    Cosmetic, and done last for that reason: the directory name is not the
    persona's identity (the name in prompt.md frontmatter is), so a
    failure here leaves a correctly renamed character in a stale folder
    rather than anything broken. But the folder is meant to be opened and
    hand-edited, and Alexander living in ``Personas/Alex/`` is a trap laid
    for the person who does.

    Returns the directory the persona now lives in — the new path, or the
    original one when the move was impossible.
    """
    root = persona_dir.parent
    base = sanitize_persona_dirname(new_name).strip()
    if not base:
        # Nothing usable as a directory name (a name of pure punctuation).
        # The frontmatter still carries the real name, so leave the folder.
        return persona_dir
    # Only bump for a collision with somebody ELSE. Renaming "Alex" to
    # "alex" collides with itself on a case-insensitive filesystem, and
    # "alex_2" would be a strange answer to a change of capitalisation.
    if base.casefold() == persona_dir.name.casefold():
        target = root / base
    else:
        target = root / unique_persona_dirname(root, base)
    if target == persona_dir:
        return persona_dir
    try:
        os.replace(persona_dir, target)
    except OSError as exc:
        logger.warning(
            "Renamed persona to '%s' but could not move %s to %s: %s",
            new_name, persona_dir, target, exc,
        )
        return persona_dir
    return target


def _shed_to_fit(lines: List[str], memory_size: int) -> None:
    """Drop memories in place until *lines* fits the budget.

    Assumptions go first, oldest among them first, and only then the
    things the persona was actually told. A guess that never got confirmed
    is the cheapest thing in the file to be wrong about, and giving them a
    shorter half-life than facts is the closest this gets to a memory that
    settles: what you worked out fades, what you witnessed stays.

    Never drops the last line — the caller has just added it, and a memory
    that alone exceeds the budget was refused before reaching here.
    """
    def over() -> bool:
        return len(lines) > 1 and _memories_bytes(lines) >= memory_size

    if not over():
        return

    # Oldest assumption first each time round, keeping the newest line
    # whatever it is. Recomputed rather than iterated, because deleting
    # shifts every index after it.
    while over():
        stale = next(
            (i for i, line in enumerate(lines[:-1]) if parse_memory_line(line).assumed),
            None,
        )
        if stale is None:
            break
        del lines[stale]

    while over():
        lines.pop(0)


def append_memory(
    persona_dir: Path, about: object, memory: object, memory_size: int,
    assumed: bool = False,
) -> str:
    """Append one memory to the persona's memories.txt, enforcing all limits.

    Returns the LLM-facing result string (see docs/feature_persona_memory.md
    for the message catalog). Never raises: every failure mode is reported
    to the LLM as an "Error:" message it can react to, because an exception
    here would kill the persona's whole reply stream.

    Check order: enabled -> has content -> per-memory char limit ->
    configured byte limit -> append (with oldest-first purge as needed).
    """
    if memory_size <= 0:
        # Memory is disabled: also delete a stale file so re-enabling the
        # persona starts from a clean slate rather than resurrecting
        # memories that outlived their limit. Best-effort: a failed
        # cleanup must not break the tool-call stream — the LLM-facing
        # answer is still "not enabled", the disk problem is only logged.
        _remove_memories_file_best_effort(persona_dir)
        return "Error: Memory is not enabled for this persona."

    if not isinstance(memory, str):
        return "Error: The memory was not saved because it had no content."
    # Normalize LLM garbage: strip the edges and remove embedded newlines
    # (a memory must be a single line). Deletion, not replacement:
    # "a\nb" -> "ab", not "a b".
    cleaned = memory.strip()
    for newline in ("\r\n", "\n", "\r"):
        cleaned = cleaned.replace(newline, "")
    cleaned = cleaned.strip()
    if not cleaned:
        return "Error: The memory was not saved because it had no content."
    # Reject, never truncate: the LLM is instructed to keep memories short
    # and can reformulate on an error.
    if len(cleaned) > MAX_MEMORY_LINE_CHARS:
        return (
            f"Error: The memory was too large to save. "
            f"Max per-memory length is {MAX_MEMORY_LINE_CHARS} characters."
        )
    subject = " ".join(str(about or "").split())[:MAX_SUBJECT_CHARS].strip("[]").strip()
    if not subject:
        return (
            "Error: The memory was not saved because it did not say who it is "
            "about. Use the name the transcript tags them with."
        )
    cleaned = Memory(subject=subject, text=cleaned, assumed=bool(assumed)).stored()

    if len(cleaned.encode("utf-8")) > memory_size:
        return (
            "Error: The memory was too large to save. "
            f"Configured memory limit: {memory_size} bytes"
        )

    lines = _memory_lines(read_memories(persona_dir))

    # Enforced, not merely asked for. "Do not add a memory that repeats one
    # you have already saved" was a sentence in the tool description, which
    # is a request rather than a rule — and the reflection pass re-reads the
    # same conversation, so it re-derives the same facts by design. Without
    # this, a persona's whole budget fills with one thing it knows.
    #
    # Compared on the stored form, so the subject counts: the same sentence
    # about two different people is two memories.
    if any(line.casefold() == cleaned.casefold() for line in lines):
        return "The memory was already saved."

    lines.append(cleaned)
    # Purge until the file is under the limit, but never drop the memory
    # just added (the newest line). A memory that alone exceeds the limit
    # was rejected above, so this always terminates with the new memory
    # surviving.
    _shed_to_fit(lines, memory_size)
    try:
        _write_memories_file(persona_dir, lines)
    except OSError as exc:
        logger.warning("Persona %s: failed to write %s: %s", persona_dir.name, MEMORIES_FILENAME, exc)
        return "Error: The memory could not be saved."
    return "The memory was saved successfully."


def purge_memories_to_limit(persona_dir: Path, memory_size: int) -> None:
    """Shrink (or delete) memories.txt to the given limit.

    Called from the persona update route when memory_size drops, and from
    the chat read path before memory injection (an external process may
    have left the file over the limit). 0 deletes the file outright. A
    file that already fits is left untouched (no needless rewrite) — a
    cheap no-op, which is what makes the per-read call affordable. If
    even the newest memory exceeds the limit the whole file is deleted —
    keeping an over-limit single memory would just be purged by the next
    add_memory anyway.

    Never raises: a disk error here must not fail the persona save (the
    frontmatter was already written; the memories file simply survives
    until the next attempt) nor the chat request (the memories just go
    un-injected for that reply).
    """
    if memory_size <= 0:
        if _remove_memories_file_best_effort(persona_dir):
            logger.info("Deleted %s for %s (memory disabled)", MEMORIES_FILENAME, persona_dir.name)
        return
    lines = _memory_lines(read_memories(persona_dir))
    if not lines:
        return  # no file, or blank file: nothing to purge
    if _memories_bytes(lines) <= memory_size:
        return  # already within the new limit
    # Same rule as the write path: assumptions before facts.
    _shed_to_fit(lines, memory_size)
    if len(lines) == 1 and _memories_bytes(lines) > memory_size:
        # Best-effort: the file simply survives until the next attempt when
        # the delete fails (logged inside the helper), matching the
        # never-raises contract above.
        if _remove_memories_file_best_effort(persona_dir):
            logger.info(
                "Deleted %s for %s: newest memory exceeds the new limit (%d bytes)",
                MEMORIES_FILENAME, persona_dir.name, memory_size,
            )
        return
    try:
        _write_memories_file(persona_dir, lines)
    except OSError as exc:
        logger.warning(
            "Persona %s: failed to purge %s to %d bytes: %s",
            persona_dir.name, MEMORIES_FILENAME, memory_size, exc,
        )


# ---------------------------------------------------------------------------
# Legacy personas.yaml: parsing + one-time migration
# ---------------------------------------------------------------------------

def load_personas_yaml(path: Path) -> List[Persona]:
    """Parse a legacy personas.yaml into Persona objects.

    Exists for the one-time startup migration (and its tests) — the live
    app reads personas from the directory and never from this file again.
    Raises on malformed content; the migration turns that into a fatal error.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise PersonaMigrationError(
            f"{path.name} is malformed: expected a top-level mapping with a 'personas' list"
        )
    # The same migration chain load_personas() ran before this file
    # existed. Without it a personas.yaml from an older release loses its
    # per-persona length settings on the way into the directory — this
    # one-time migration is the last chance anything reads those keys.
    raw, notes = migrate_personas(raw)
    for note in notes:
        logger.info("Migrated %s: %s", path.name, note)

    entries = raw.get("personas", [])
    if not isinstance(entries, list):
        raise PersonaMigrationError(
            f"{path.name} is malformed: 'personas' must be a list"
        )
    personas: List[Persona] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise PersonaMigrationError(
                f"{path.name} is malformed: each persona must be a mapping"
            )
        personas.append(Persona(**entry))
    return personas


def migrate_from_legacy_yaml(yaml_path: Path, root: Path) -> None:
    """One-time migration: personas.yaml -> per-persona subdirectories.

    On success ``yaml_path`` is renamed to ``personas.yaml.bak`` (never
    deleted) so the migration can never run twice.

    Error policy (docs/feature_persona_autodiscovery.md):
      * fatal (malformed YAML, unreadable file, unwritable directory,
        disk full) -> raise PersonaMigrationError with the YAML left
        untouched and any partially created directory removed
        best-effort, so the next startup retries cleanly;
      * minor (missing referenced files, unsupported image/audio
        formats) -> log a warning and continue without the file.
    """
    logger.info("Persona migration in progress: %s -> %s", yaml_path, root)
    try:
        personas = load_personas_yaml(yaml_path)
    except PersonaMigrationError as exc:
        raise _abort_migration(yaml_path, root, str(exc)) from exc
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise _abort_migration(yaml_path, root, f"cannot parse {yaml_path.name}: {exc}") from exc

    try:
        root.mkdir(parents=True, exist_ok=True)
        for persona in personas:
            _migrate_one_persona(persona, root)
    except PersonaMigrationError:
        raise
    except OSError as exc:
        raise _abort_migration(yaml_path, root, f"disk error while writing personas: {exc}") from exc

    backup_path = yaml_path.with_name(yaml_path.name + ".bak")
    try:
        yaml_path.rename(backup_path)
    except OSError as exc:
        # The directory is complete and the YAML is intact, so the next
        # startup will take the (noisy but safe) "both exist" path. Do not
        # rmtree the finished directory here.
        raise _abort_migration(
            yaml_path, root,
            f"cannot rename {yaml_path.name} to {backup_path.name}: {exc}",
            remove_partial_dir=False,
        ) from exc
    logger.info(
        "Persona migration complete: %s -> %s (%d persona%s)",
        yaml_path.name, root, len(personas), "" if len(personas) == 1 else "s",
    )


def _migrate_one_persona(persona: Persona, root: Path) -> None:
    """Migrate one legacy persona into its own subdirectory of root."""
    base_name = sanitize_persona_dirname(persona.name)
    if not base_name:
        # A name with no directory-usable characters still needs a home;
        # the frontmatter `name` field keeps the real one.
        logger.warning(
            "Migration: persona name '%s' has no usable directory characters; using 'persona'",
            persona.name,
        )
        base_name = "persona"
    dir_name = unique_persona_dirname(root, base_name)
    persona_dir = root / dir_name

    try:
        persona_dir.mkdir()
        write_prompt_md(
            persona_dir,
            name=persona.name,
            description=persona.description,
            router_hints=persona.router_hints,
            avatar_color=persona.avatar_color,
            allow_tool_calls=persona.allow_tool_calls,
            length_bias=persona.length_bias.value,
            system_prompt=persona.system_prompt,
        )
        write_language_file(persona_dir, persona.reference_audio_language)
    except OSError as exc:
        raise PersonaMigrationError(
            f"persona '{persona.name}': cannot write {persona_dir}: {exc}"
        ) from exc

    if persona.avatar_image:
        source = Path(persona.avatar_image)
        extension = source.suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            logger.warning(
                "Migration: ignoring unsupported image file '%s' for persona %s. "
                "Only png, jpg, jpeg, gif, webp images are supported.",
                source.name, persona.name,
            )
        else:
            _migrate_file(persona, source, persona_dir / f"image{extension}", "avatar image")

    if persona.reference_audio:
        source = Path(persona.reference_audio)
        if source.suffix.lower() != AUDIO_EXTENSION:
            logger.warning(
                "Migration: ignoring unsupported audio file '%s' for persona %s. "
                "Only wav audio is supported.",
                source.name, persona.name,
            )
        else:
            _migrate_file(
                persona, source, persona_dir / REFERENCE_AUDIO_FILENAME, "reference audio"
            )

    if persona.reference_audio_transcript:
        source = Path(persona.reference_audio_transcript)
        try:
            transcript = source.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            logger.warning(
                "Migration: transcript file '%s' for persona %s could not be read (%s); skipped.",
                source.name, persona.name, exc,
            )
        else:
            if transcript:
                try:
                    (persona_dir / TRANSCRIPT_FILENAME).write_text(transcript, encoding="utf-8")
                except OSError as exc:
                    raise PersonaMigrationError(
                        f"persona '{persona.name}': cannot write {TRANSCRIPT_FILENAME}: {exc}"
                    ) from exc


def _migrate_file(persona: Persona, source: Path, target: Path, kind: str) -> None:
    """Copy one referenced legacy file into the persona directory.

    A missing or unreadable source is a *minor* error (warn + skip); a
    write failure is *fatal* (raises PersonaMigrationError).
    """
    try:
        data = source.read_bytes()
    except OSError as exc:
        logger.warning(
            "Migration: %s file '%s' for persona %s could not be read (%s); skipped.",
            kind, source.name, persona.name, exc,
        )
        return
    try:
        target.write_bytes(data)
    except OSError as exc:
        raise PersonaMigrationError(
            f"persona '{persona.name}': cannot write {target.name}: {exc}"
        ) from exc


def _abort_migration(
    yaml_path: Path, root: Path, reason: str, *, remove_partial_dir: bool = True
) -> PersonaMigrationError:
    """Log a fatal migration failure and roll the directory state back.

    The YAML file is ALWAYS left untouched — it is the only guaranteed
    copy of the data until the rename succeeds.
    """
    logger.error("Persona migration failed: %s", reason)
    logger.error(
        "%s was left untouched. Fix the problem and restart to retry the migration.",
        yaml_path.name,
    )
    if remove_partial_dir and root.exists():
        try:
            shutil.rmtree(root)
            logger.info("Removed partially created personas directory: %s", root)
        except OSError as cleanup_exc:
            logger.warning(
                "Could not remove partially created personas directory %s: %s (left in place)",
                root, cleanup_exc,
            )
    return PersonaMigrationError(f"persona migration aborted: {reason}")
