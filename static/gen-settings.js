/**
 * gen-settings.js — General Settings modal: load and persist app-level config.
 *
 * Handles max_persona_replies and persona_name_mentions through the general
 * settings overlay. Reads/writes via the existing /api/settings endpoint.
 */

/* ==========================================================================
   Event listeners
   ========================================================================== */

document.getElementById("btn-gen-settings").addEventListener("click", openGenSettings);
document.getElementById("gen-settings-btn-close").addEventListener("click", closeGenSettings);
document.getElementById("gen-settings-btn-cancel").addEventListener("click", closeGenSettings);
genSettingsForm.addEventListener("submit", submitGenSettings);

gsfWipeBtn.addEventListener("click", openWipeConfirm);
document.getElementById("gsf-wipe-cancel").addEventListener("click", closeWipeConfirm);
document.getElementById("gsf-wipe-confirm").addEventListener("click", wipeContext);
gsfWipeOverlay.addEventListener("click", (e) => {
    if (e.target === gsfWipeOverlay) closeWipeConfirm();
});

genSettingsOverlay.addEventListener("click", (e) => {
    if (e.target === genSettingsOverlay) closeGenSettings();
});

/* ==========================================================================
   Modal lifecycle
   ========================================================================== */

async function openGenSettings() {
    genSettingsOverlay.classList.remove("hidden");
    genSettingsError.classList.add("hidden");

    const saveBtn = document.getElementById("gen-settings-btn-save");
    saveBtn.disabled = true;

    gsfWipeResult.classList.add("hidden");
    const ok = await loadGenSettingsIntoForm();
    saveBtn.disabled = !ok;
    // Deliberately after the form: the wipe section is informational, and
    // a slow or failed inventory must not hold up the settings.
    refreshContextSummary();
}

function closeGenSettings() {
    genSettingsOverlay.classList.add("hidden");
}

/* ==========================================================================
   Form population
   ========================================================================== */

async function loadGenSettingsIntoForm() {
    try {
        const resp = await fetch("/api/settings");
        if (!resp.ok) {
            showGenSettingsError(`Failed to load settings (HTTP ${resp.status}).`);
            return false;
        }
        const data = await resp.json();
        gsfMaxPersonaReplies.value = data.general.max_persona_replies ?? 1;
        gsfPersonaNameMentions.checked = data.general.persona_name_mentions ?? true;
        gsfMaxTurnsForContext.value = data.general.max_turns_for_context ?? 6;
        gsfShowToolCalls.checked = data.general.show_tool_calls ?? true;
        gsfTypicalLength.value = data.general.typical_length || "normal";
        gsfEnablePersonaMemories.checked = data.general.enable_persona_memories ?? true;
        return true;
    } catch (err) {
        console.error("Failed to load settings:", err);
        showGenSettingsError("Failed to load settings. Is the server running?");
        return false;
    }
}

function showGenSettingsError(msg) {
    genSettingsError.textContent = msg;
    genSettingsError.classList.remove("hidden");
}

/* ==========================================================================
   Form submission
   ========================================================================== */

/**
 * Validate a number input against its own min/max attributes.
 *
 * Reading the bounds off the element rather than hardcoding them keeps
 * this in step with the HTML and, through it, with the API. Duplicating
 * the numbers here is how raising Max Persona Replies to 6 got silently
 * rejected at 4 by this function long after everything else allowed it.
 *
 * Returns the parsed value, or null after showing an error.
 */
function checkRange(el, label) {
    const value = parseInt(el.value, 10);
    const min = parseInt(el.min, 10);
    const max = parseInt(el.max, 10);
    if (isNaN(value) || value < min || value > max) {
        showGenSettingsError(`${label} must be between ${min} and ${max}.`);
        return null;
    }
    return value;
}


async function submitGenSettings(e) {
    e.preventDefault();
    genSettingsError.classList.add("hidden");

    const maxReplies = checkRange(gsfMaxPersonaReplies, "Max Persona Replies");
    if (maxReplies === null) return;

    const maxTurns = checkRange(gsfMaxTurnsForContext, "Exchanges of Context");
    if (maxTurns === null) return;

    // Fetch current full settings so we can patch only the general section
    let current;
    try {
        const resp = await fetch("/api/settings");
        if (!resp.ok) return showGenSettingsError(`Failed to load current settings (HTTP ${resp.status}).`);
        current = await resp.json();
    } catch (err) {
        return showGenSettingsError("Failed to load current settings.");
    }

    const payload = {
        ...current,
        // Restore null seed as 0 (API contract: 0 means no seed)
        tts: { ...current.tts, base_url: current.tts.base_url ?? "", seed: current.tts.seed ?? 0 },
        stt: { ...current.stt, base_url: current.stt.base_url ?? "" },
        general: {
            persona_name_mentions: gsfPersonaNameMentions.checked,
            max_persona_replies: maxReplies,
            max_turns_for_context: maxTurns,
            show_tool_calls: gsfShowToolCalls.checked,
            typical_length: gsfTypicalLength.value,
            enable_persona_memories: gsfEnablePersonaMemories.checked,
        },
    };

    try {
        const resp = await fetch("/api/settings", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            return showGenSettingsError(extractApiErrorMessage(err, resp.status));
        }

        // Sync in-memory state (forgetting one of these lets the next
        // settings save from another dialog persist a stale value)
        personaNameMentionsEnabled = gsfPersonaNameMentions.checked;
        maxPersonaReplies = maxReplies;
        maxTurnsForContext = maxTurns;

        closeGenSettings();
    } catch (err) {
        console.error("Failed to save settings:", err);
        showGenSettingsError("Request failed. Is the server running?");
    }
}


/* ==========================================================================
   Wiping stored context

   The point of this section is not the deletion — "New Chat" has always
   deleted a room. It is being able to answer "is anything still carrying
   over?" without guessing: what exists is listed before, and what is left
   is read back from disk after, so the answer is checked rather than
   assumed.
   ========================================================================== */

/** The inventory last fetched, so the confirmation can describe the damage. */
let gsfContext = null;

function describeContext(ctx) {
    if (!ctx) return "Could not read what is stored.";
    const bits = [];
    const messages = ctx.rooms.reduce((sum, r) => sum + r.messages, 0);
    if (messages) {
        bits.push(`${messages} message${messages === 1 ? "" : "s"} across ` +
                  `${ctx.rooms.length} room${ctx.rooms.length === 1 ? "" : "s"}`);
    }
    const memories = ctx.personas.reduce((sum, p) => sum + p.memories, 0);
    if (memories) {
        bits.push(`${memories} saved memor${memories === 1 ? "y" : "ies"} ` +
                  `(${ctx.personas.map(p => p.persona).join(", ")})`);
    }
    if (ctx.playing_as) bits.push(`playing as ${ctx.playing_as}`);
    return bits.length ? `Stored now: ${bits.join("; ")}.` : "Nothing stored.";
}

async function refreshContextSummary() {
    gsfContextSummary.textContent = "Checking…";
    try {
        const resp = await fetch("/api/session/context");
        gsfContext = resp.ok ? await resp.json() : null;
    } catch (err) {
        console.error("Failed to read stored context:", err);
        gsfContext = null;
    }
    gsfContextSummary.textContent = describeContext(gsfContext);
}

function wipeSelection() {
    return {
        rooms: gsfWipeRooms.checked ? "all" : "none",
        memories: gsfWipeMemories.checked,
        playing_as: gsfWipePlayingAs.checked,
    };
}

function openWipeConfirm() {
    const wanted = wipeSelection();
    if (wanted.rooms === "none" && !wanted.memories && !wanted.playing_as) {
        return showGenSettingsError("Tick at least one thing to wipe.");
    }
    genSettingsError.classList.add("hidden");

    const lines = [];
    if (wanted.rooms !== "none" && gsfContext) {
        const messages = gsfContext.rooms.reduce((sum, r) => sum + r.messages, 0);
        lines.push(`Every room's conversation — ${messages} message${messages === 1 ? "" : "s"}, and any audio with them.`);
    }
    if (wanted.memories && gsfContext) {
        const memories = gsfContext.personas.reduce((sum, p) => sum + p.memories, 0);
        lines.push(`Saved memories — ${memories} line${memories === 1 ? "" : "s"}` +
                   (gsfContext.personas.length ? ` from ${gsfContext.personas.map(p => p.persona).join(", ")}.` : "."));
    }
    if (wanted.playing_as) {
        lines.push(gsfContext && gsfContext.playing_as
            ? `You stop playing as ${gsfContext.playing_as}.`
            : "Who you are playing as (nobody is set).");
    }
    renderList(gsfWipePlan, lines);
    gsfWipeOverlay.classList.remove("hidden");
}

function closeWipeConfirm() {
    gsfWipeOverlay.classList.add("hidden");
}

async function wipeContext() {
    const confirmBtn = document.getElementById("gsf-wipe-confirm");
    confirmBtn.disabled = true;
    try {
        const resp = await fetch("/api/session/wipe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(wipeSelection()),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            closeWipeConfirm();
            return showGenSettingsError(extractApiErrorMessage(err, resp.status));
        }
        const result = await resp.json();
        gsfContext = result.remaining;

        const done = [];
        if (result.messages_deleted || result.rooms_cleared.length) {
            done.push(`${result.messages_deleted} message${result.messages_deleted === 1 ? "" : "s"} ` +
                      `from ${result.rooms_cleared.length} room${result.rooms_cleared.length === 1 ? "" : "s"}`);
        }
        if (result.memories_cleared.length) {
            done.push(`memories for ${result.memories_cleared.join(", ")}`);
        }
        if (result.playing_as_cleared) done.push("who you are playing as");

        gsfWipeResult.textContent =
            (done.length ? `Deleted ${done.join(", ")}. ` : "Nothing to delete. ") +
            describeContext(result.remaining);
        gsfWipeResult.classList.remove("hidden");
        gsfContextSummary.textContent = describeContext(result.remaining);

        // The room on screen is one of the ones just emptied, and leaving
        // its messages in the transcript would be the app showing context
        // that no longer exists — the exact doubt this feature removes.
        if (result.rooms_cleared.length) {
            messagesEl.innerHTML = "";
            showEmptyState();
        }
        if (result.playing_as_cleared) {
            player.persona_name = "";
            applyPlayingAsControls();
        }
    } catch (err) {
        console.error("Failed to wipe context:", err);
        showGenSettingsError("Request failed. Is the server running?");
    } finally {
        confirmBtn.disabled = false;
        closeWipeConfirm();
    }
}
