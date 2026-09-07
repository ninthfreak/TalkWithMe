/**
 * utils.js — Pure utility functions shared across modules.
 */

/** Scroll the message panel to the bottom, deferred to next frame. */
function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });
}

/**
 * Extract a human-readable message from a FastAPI error response body.
 *
 * `detail` is a plain string for HTTPException, but a 422 validation
 * error returns an *array* of {loc, msg, type} objects — passing that
 * array to textContent renders as "[object Object]".
 *
 * @param {object} errBody - Parsed JSON error body (may be empty).
 * @param {number} status - HTTP status code, used for the fallback message.
 * @returns {string} Human-readable error message.
 */
function extractApiErrorMessage(errBody, status) {
    const detail = errBody ? errBody.detail : undefined;
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail) && detail.length > 0) {
        return detail
            .map((e) => {
                // loc looks like ["body", "description"] or ["path", "name"] —
                // drop the transport-level segment and keep the field name.
                const field = Array.isArray(e.loc) ? e.loc.slice(1).join(".") : "";
                return e.msg ? (field ? `${field}: ${e.msg}` : e.msg) : "unknown error";
            })
            .join("; ");
    }
    return `Error ${status}`;
}

/**
 * Case-insensitive alphabetical comparator for personas (anything with a
 * `name` property). Pair with Array.prototype.sort, e.g.
 * `[...personas].sort(comparePersonasByName)`.
 *
 * `sensitivity: "base"` makes the comparison case-insensitive ("alice"
 * before "Bob"), matching how chat room names are sorted in chatrooms.js.
 */
function comparePersonasByName(a, b) {
    return a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
}

/** Escape HTML special characters to prevent XSS in dynamically rendered text. */
function escapeHtml(str) {
    if (typeof str !== 'string') return str;

    return str.replace(/[&<>"']/g, match => {
        return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }[match];
    });
}


/**
 * Bind a listener, tolerating an element that is not there.
 *
 * The reason this exists: startup is a chain. init() runs the setup
 * functions one after another, and a single `null.addEventListener`
 * aborts the rest of it — so one missing element takes out every
 * binding after it AND the history load at the end. That failure is
 * silent and does not look like a JavaScript error to anyone using the
 * app; it looks like "picking a room does nothing" and "the character
 * picker won't open", which is exactly how it was reported.
 *
 * A missing element is still a bug, so it is logged loudly by name.
 * What it is not any more is fatal to everything downstream of it.
 *
 * Returns true when the listener was attached.
 */
function bind(el, event, handler, label) {
    if (!el) {
        console.error(
            `UI element missing: cannot bind "${event}" for ${label || "an unnamed control"}. ` +
            `If the app was just updated, reload with Ctrl+Shift+R — a cached page ` +
            `with newer scripts is the usual cause.`
        );
        return false;
    }
    el.addEventListener(event, handler);
    return true;
}
