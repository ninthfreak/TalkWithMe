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

    const ok = await loadGenSettingsIntoForm();
    saveBtn.disabled = !ok;
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
