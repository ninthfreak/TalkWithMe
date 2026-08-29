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

CHATROOMS_JS = STATIC_DIR / "chatrooms.js"


def _select_options(html: str, select_id: str) -> set[str]:
    block = re.search(
        rf'<select id="{re.escape(select_id)}">(.*?)</select>', html, re.DOTALL
    )
    assert block, f"no <select id={select_id}> in index.html"
    return set(re.findall(r'<option value="([^"]*)"', block.group(1)))


def test_room_length_selects_offer_every_tier():
    from app.config import TypicalLength

    expected = {t.value for t in TypicalLength}
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert _select_options(html, "gsf-typical-length") == expected

    # The per-room select in the chat room editor is built in JS.
    js = CHATROOMS_JS.read_text(encoding="utf-8")
    listed = re.search(r"const TYPICAL_LENGTH_OPTIONS = \[(.*?)\];", js, re.DOTALL)
    assert listed, "TYPICAL_LENGTH_OPTIONS not found in chatrooms.js"
    assert set(re.findall(r'\["([^"]+)"', listed.group(1))) == expected


def test_persona_length_bias_select_offers_every_bias():
    from app.config import LengthBias

    html = INDEX_HTML.read_text(encoding="utf-8")
    assert _select_options(html, "pf-length-bias") == {b.value for b in LengthBias}
