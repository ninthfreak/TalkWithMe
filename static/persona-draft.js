/**
 * persona-draft.js — draft a persona with the LLM, then try it before keeping it.
 *
 * The dialog does three things, in order of how much they matter:
 *
 *  1. Specifies. A brief on its own makes every word a global intensity
 *     dial — "crude" comes back as hostile, uncooperative and rude at
 *     once. The dropdowns and detail fields split that into independent
 *     axes, which is the whole reason they exist.
 *  2. Drafts. The specification comes back as every field.
 *  3. Proves it. The same question answered by the draft and by a persona
 *     you already have, side by side — because a draft read on its own
 *     always sounds distinctive, and read beside its neighbour often
 *     doesn't.
 *
 * The dials and details are rendered server-side from the constants the
 * drafting prompt is built from, so this file never names one: it reads
 * whatever is on the page and posts it back by key.
 *
 * Nothing reaches disk here. "Use this draft" fills the editor form and
 * the normal Save is still the only thing that writes a persona.
 */

/** The draft currently on screen, or null before the first Draft it. */
let pdDraft = null;

function openPersonaDraft() {
    pdDraft = null;
    pdBrief.value = "";
    // The dials keep whatever they were left on — they are settings, and
    // re-picking seven of them for every draft is the tedium the dialog
    // exists to remove. The details are about one specific character, so
    // carrying them into the next one would be wrong.
    for (const input of pdDetails) input.value = "";
    pdResult.classList.add("hidden");
    pdUseBtn.classList.add("hidden");
    pdRepliesEl.classList.add("hidden");
    pdRepliesEl.innerHTML = "";
    hidePersonaDraftError();
    populateDraftComparison();
    pdOverlay.classList.remove("hidden");
    pdBrief.focus();
}

function closePersonaDraft() {
    pdOverlay.classList.add("hidden");
}

function showPersonaDraftError(msg) {
    pdErrorEl.textContent = msg;
    pdErrorEl.classList.remove("hidden");
}

function hidePersonaDraftError() {
    pdErrorEl.classList.add("hidden");
}

/**
 * Fill the "compare against" list from the saved personas.
 *
 * Rebuilt on every open rather than cached: a persona created or deleted
 * since the last open should be right, and the list is two lines long.
 */
function populateDraftComparison() {
    pdCompareEl.innerHTML = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "Nobody — just show the draft";
    pdCompareEl.appendChild(none);
    for (const p of [...personas].sort(comparePersonasByName)) {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.name;
        pdCompareEl.appendChild(opt);
    }
    // Comparing is the point, so default to it when there is anyone to
    // compare against.
    if (personas.length) pdCompareEl.selectedIndex = 1;
}

function renderList(el, items) {
    el.innerHTML = "";
    for (const item of items) {
        const li = document.createElement("li");
        li.textContent = item;
        el.appendChild(li);
    }
}

async function requestPersonaDraft() {
    const brief = pdBrief.value.trim();
    if (!brief) return showPersonaDraftError("Say a line or two about the character first.");

    hidePersonaDraftError();
    pdDraftBtn.disabled = true;
    pdDraftBtn.textContent = "Drafting…";

    const dials = {};
    for (const select of pdDials) dials[select.dataset.dial] = select.value;
    const details = {};
    for (const input of pdDetails) {
        const value = input.value.trim();
        if (value) details[input.dataset.detail] = value;
    }

    try {
        const resp = await fetch("/api/personas/draft", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ brief, dials, details }),
        });
        if (!resp.ok) {
            const detail = await resp.json().catch(() => null);
            return showPersonaDraftError(
                (detail && detail.detail) || `Could not draft (HTTP ${resp.status}).`
            );
        }
        pdDraft = await resp.json();
    } catch (err) {
        console.error("Persona draft failed:", err);
        return showPersonaDraftError("Could not reach the server.");
    } finally {
        pdDraftBtn.disabled = false;
        pdDraftBtn.textContent = "Draft again";
    }

    renderList(pdNotesEl, pdDraft.notes || []);
    renderList(pdWarningsEl, pdDraft.warnings || []);
    pdWarningsRow.classList.toggle("hidden", !(pdDraft.warnings || []).length);
    pdRepliesEl.classList.add("hidden");
    pdRepliesEl.innerHTML = "";
    pdResult.classList.remove("hidden");
    pdUseBtn.classList.remove("hidden");
    // The dialog scrolls, and everything worth reading has just appeared
    // below the fold — with the levers block open it is well below it.
    pdResult.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function tryPersonaDraft() {
    if (!pdDraft) return;
    const question = pdQuestionEl.value.trim();
    if (!question) return showPersonaDraftError("Give them a question to answer.");

    hidePersonaDraftError();
    pdTryBtn.disabled = true;
    pdTryBtn.textContent = "Asking…";
    try {
        const resp = await fetch("/api/personas/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: pdDraft.name,
                system_prompt: pdDraft.system_prompt,
                description: pdDraft.description,
                length_bias: pdDraft.length_bias,
                question,
                compare_with: pdCompareEl.value || null,
            }),
        });
        if (!resp.ok) {
            const detail = await resp.json().catch(() => null);
            return showPersonaDraftError(
                (detail && detail.detail) || `Could not run it (HTTP ${resp.status}).`
            );
        }
        const body = await resp.json();
        pdRepliesEl.innerHTML = "";
        for (const reply of [body.draft, body.comparison].filter(Boolean)) {
            const box = document.createElement("div");
            box.className = "pd-reply";
            const who = document.createElement("div");
            who.className = "pd-reply-who";
            who.textContent = reply.persona;
            const text = document.createElement("div");
            text.className = "pd-reply-text";
            text.textContent = reply.text;
            box.appendChild(who);
            box.appendChild(text);
            pdRepliesEl.appendChild(box);
        }
        pdRepliesEl.classList.remove("hidden");
        pdRepliesEl.scrollIntoView({ behavior: "smooth", block: "end" });
    } catch (err) {
        console.error("Persona preview failed:", err);
        showPersonaDraftError("Could not reach the server.");
    } finally {
        pdTryBtn.disabled = false;
        pdTryBtn.textContent = "Run it";
    }
}

/**
 * Copy the draft into the editor form and close.
 *
 * Deliberately not a save. The draft is a starting point — the fields are
 * there to be argued with, and the persona only exists once the user
 * presses Save like any other.
 */
function usePersonaDraft() {
    if (!pdDraft) return;
    pfName.value = pdDraft.name;
    pfDescription.value = pdDraft.description;
    pfSystemPrompt.value = pdDraft.system_prompt;
    pfRouterHints.value = pdDraft.router_hints;
    pfAvatarColor.value = pdDraft.avatar_color;
    pfLengthBias.value = pdDraft.length_bias || "match";
    closePersonaDraft();
    pfName.focus();
}

function setupPersonaDraftEventListeners() {
    document.getElementById("pf-btn-draft").addEventListener("click", openPersonaDraft);
    pdDraftBtn.addEventListener("click", requestPersonaDraft);
    pdTryBtn.addEventListener("click", tryPersonaDraft);
    pdUseBtn.addEventListener("click", usePersonaDraft);
    document.getElementById("pd-btn-close").addEventListener("click", closePersonaDraft);
    document.getElementById("pd-btn-cancel").addEventListener("click", closePersonaDraft);
    pdOverlay.addEventListener("click", (e) => {
        if (e.target === pdOverlay) closePersonaDraft();
    });
    // Ctrl/Cmd+Enter drafts, matching the other text boxes in the app.
    pdBrief.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            requestPersonaDraft();
        }
    });
}
