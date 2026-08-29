"""Static consistency checks that would otherwise only fail at runtime.

Two kinds live here: AGENTS.md's API table against the registered routes,
and the frontend's getElementById calls against the template's ids.

The endpoint table is hand-maintained next to the code, so it drifts
silently (it has, historically). This test extracts the real /api/ routes
from the FastAPI app and compares them against the markdown table, failing
loudly if either set diverges. Route metadata (methods, paths) is static
at import time, so no lifespan, config, or network access is needed.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

from fastapi.routing import APIRoute

from app.main import app

AGENTS_MD = Path(__file__).resolve().parent.parent / "AGENTS.md"

# Starlette auto-registers HEAD (for GET routes) and OPTIONS; they are not
# part of the documented API surface.
_IGNORED_METHODS = {"HEAD", "OPTIONS"}


def _actual_api_routes() -> set[tuple[str, str]]:
    """(METHOD, path) pairs registered on the app, scoped to /api/ routes.

    Scoped to /api/ on purpose: the table documents the API, not the UI
    (GET /) or the /static mount (a Mount, not an APIRoute).
    """
    routes = set()
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/api/"):
            continue
        for method in route.methods - _IGNORED_METHODS:
            routes.add((method, route.path))
    return routes


def _documented_api_routes() -> list[tuple[str, str]]:
    """(METHOD, path) pairs from the '## API endpoints' table in AGENTS.md.

    Returns a list (not a set) so the test can also detect duplicate rows.
    """
    text = AGENTS_MD.read_text(encoding="utf-8")
    match = re.search(
        r"^## API endpoints\s*$(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "AGENTS.md is missing the '## API endpoints' section"

    rows = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        method, path = cells[0], cells[1]
        # Skip the header row and the |---|---|---| separator row.
        if method.upper() == "METHOD" or set(method) <= {"-"}:
            continue
        method = method.strip("`").upper()
        # The table annotates one endpoint with an illustrative query string
        # (?room=<room>); route paths carry no query part.
        path = path.strip("`").split("?", 1)[0]
        rows.append((method, path))

    assert rows, "The AGENTS.md '## API endpoints' table contains no data rows"
    return rows


def test_agents_md_api_endpoints_table_matches_registered_routes():
    actual = _actual_api_routes()
    documented_rows = _documented_api_routes()
    documented = set(documented_rows)

    duplicates = sorted(r for r, count in Counter(documented_rows).items() if count > 1)
    assert not duplicates, f"Duplicate rows in the AGENTS.md API table: {duplicates}"

    missing_from_docs = sorted(actual - documented)
    assert not missing_from_docs, (
        "Routes registered on the app but missing from the AGENTS.md API table: "
        + ", ".join(f"{method} {path}" for method, path in missing_from_docs)
    )

    ghost_rows = sorted(documented - actual)
    assert not ghost_rows, (
        "Rows in the AGENTS.md API table that match no registered route: "
        + ", ".join(f"{method} {path}" for method, path in ghost_rows)
    )


# ---------------------------------------------------------------------------
# Frontend ids: getElementById targets must exist in the template
# ---------------------------------------------------------------------------
#
# The JS modules resolve their elements at load time and never null-check,
# so an id that does not exist in index.html is a TypeError the first time
# that dialog opens — invisible to every other test here, since none of
# them load the page. This has already shipped once: a select was added to
# the JS but the matching HTML edit silently failed to apply, leaving the
# general settings modal broken.

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_HTML = Path(__file__).resolve().parent.parent / "templates" / "index.html"


def _template_ids() -> set[str]:
    return set(re.findall(r'id="([^"]+)"', INDEX_HTML.read_text(encoding="utf-8")))


def test_getelementbyid_targets_exist_in_the_template():
    ids = _template_ids()
    missing = []
    for js in sorted(STATIC_DIR.glob("*.js")):
        for target in re.findall(r'getElementById\("([^"]+)"\)', js.read_text(encoding="utf-8")):
            if target not in ids:
                missing.append(f"{js.name} -> #{target}")
    assert not missing, (
        "static/*.js resolves ids that templates/index.html does not define: "
        + ", ".join(sorted(set(missing)))
    )


def test_labels_point_at_elements_that_exist():
    ids = _template_ids()
    html = INDEX_HTML.read_text(encoding="utf-8")
    dangling = sorted({f for f in re.findall(r'<label for="([^"]+)"', html) if f not in ids})
    assert not dangling, f"<label for=...> targets with no matching id: {dangling}"


# ---------------------------------------------------------------------------
# Enum options offered by the UI must match the enums the API accepts
# ---------------------------------------------------------------------------
#
# The selects are hand-written HTML/JS, so a renamed or added enum member
# drifts silently: the UI keeps offering a value the API now rejects, or
# quietly hides one it accepts. Both fail only when a user clicks.

def _select_options(html: str, select_id: str) -> set[str]:
    block = re.search(
        rf'<select id="{re.escape(select_id)}">(.*?)</select>', html, re.DOTALL
    )
    assert block, f"no <select id={select_id}> in index.html"
    return set(re.findall(r'<option value="([^"]*)"', block.group(1)))


@pytest.mark.parametrize("select_id", ["gsf-typical-length", "re-typical-length"])
def test_room_length_selects_offer_every_tier(select_id):
    from app.config import TypicalLength

    html = INDEX_HTML.read_text(encoding="utf-8")
    assert _select_options(html, select_id) == {t.value for t in TypicalLength}


def test_persona_length_bias_select_offers_every_bias():
    from app.config import LengthBias

    html = INDEX_HTML.read_text(encoding="utf-8")
    assert _select_options(html, "pf-length-bias") == {b.value for b in LengthBias}


# ---------------------------------------------------------------------------
# Numeric input bounds must match what the API actually accepts
# ---------------------------------------------------------------------------
#
# A settings field's range lives in three places: the Pydantic constraint,
# the input's min/max, and (until it was made to read the element) the JS
# validator. Raising Max Persona Replies from 4 to 6 updated the first two
# and the label, but a hardcoded `> 4` in gen-settings.js kept rejecting 5
# and 6 with a stale message — the UI advertised a range it refused to
# accept. This pins the remaining pair together.

# input id -> (model, field). Only fields with a bound worth checking.
_BOUNDED_INPUTS = {
    "gsf-max-persona-replies": ("GeneralSettingsRequest", "max_persona_replies"),
    "gsf-max-turns-for-context": ("GeneralSettingsRequest", "max_turns_for_context"),
    "sf-llm-temperature": ("LLMSettingsRequest", "temperature"),
    "sf-llm-max-tokens": ("LLMSettingsRequest", "max_tokens"),
    "sf-tts-num-steps": ("TTSSettingsRequest", "num_steps"),
    "sf-tts-guidance-scale": ("TTSSettingsRequest", "guidance_scale"),
    "sf-tts-timeout": ("TTSSettingsRequest", "timeout"),
    "sf-stt-timeout": ("STTSettingsRequest", "timeout"),
}


def _model_bounds(model_name: str, field_name: str):
    """(ge, le) declared on a Pydantic field, either as None."""
    from app import models

    field = getattr(models, model_name).model_fields[field_name]
    ge = le = None
    for meta in field.metadata:
        ge = getattr(meta, "ge", None) if ge is None else ge
        le = getattr(meta, "le", None) if le is None else le
    return ge, le


def _input_attrs(html: str, input_id: str):
    tag = re.search(rf'<input id="{re.escape(input_id)}"[^>]*>', html)
    assert tag, f"no <input id={input_id}> in index.html"
    def attr(name):
        m = re.search(rf'{name}="([^"]+)"', tag.group(0))
        return float(m.group(1)) if m else None
    return attr("min"), attr("max")


@pytest.mark.parametrize("input_id", sorted(_BOUNDED_INPUTS))
def test_number_input_bounds_match_the_api(input_id):
    model_name, field_name = _BOUNDED_INPUTS[input_id]
    ge, le = _model_bounds(model_name, field_name)
    html_min, html_max = _input_attrs(INDEX_HTML.read_text(encoding="utf-8"), input_id)

    assert html_min == (float(ge) if ge is not None else None), (
        f"#{input_id} min={html_min} but {model_name}.{field_name} requires ge={ge}"
    )
    assert html_max == (float(le) if le is not None else None), (
        f"#{input_id} max={html_max} but {model_name}.{field_name} requires le={le}"
    )


def test_general_settings_js_does_not_hardcode_bounds():
    """The validator must read min/max off the element, not repeat them."""
    js = (STATIC_DIR / "gen-settings.js").read_text(encoding="utf-8")
    stale = re.findall(r"must be between \d+ and \d+", js)
    assert not stale, f"gen-settings.js hardcodes a range: {stale}"


# ---------------------------------------------------------------------------
# Modal dialogs must use a class that exists
# ---------------------------------------------------------------------------
#
# Two dialogs shipped using `class="modal"`, which no stylesheet defines,
# so they rendered transparent on top of whatever was behind them. The
# markup was valid and every behavioural test passed; only opening them in
# a browser showed it.

def test_modal_overlays_use_a_styled_box():
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")

    box_classes = {
        c for c in re.findall(r'<div class="(modal[^"]*)"', html)
        if not c.startswith("modal-overlay") and c != "modal-header"
    }
    assert box_classes, "no modal boxes found in index.html"

    for cls in sorted(box_classes):
        for token in cls.split():
            # (?![\w-]) rather than \b: a hyphen is a word boundary, so
            # "^\.modal\b" matches ".modal-overlay" and the check passes
            # for a class nothing actually defines.
            assert re.search(rf"^\.{re.escape(token)}(?![\w-])", css, re.MULTILINE), (
                f'index.html uses class "{token}" which style.css does not define'
            )
