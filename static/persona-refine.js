/**
 * persona-refine.js — change one thing about a persona you already have.
 *
 * The inverse of drafting, and the risk is the opposite one. A draft can
 * fail by having no shape; a refinement fails by losing a shape that
 * already worked — you ask for a coarser harbourmaster and get back a
 * stranger who happens to swear. So the dialog is built around showing
 * what survived: the notes say what was left alone, and "Before and
 * after" runs the same question through the persona as it stands and the
 * revision, side by side.
 *
 * It works on what is in the editor form, not on what is on disk. The
 * form is what the user is looking at, and refining anything else would
 * revise a version they cannot see.
 */

/** The revision currently on screen, or null before the first Refine it. */
let prRevision = null;
/** The persona the revision was made from, kept for the before/after. */
let prBefore = null;

function currentPersonaFromForm() {
    return {
        name: pfName.value.trim(),
        description: pfDescription.value.trim(),
        system_prompt: pfSystemPrompt.value,
        router_hints: pfRouterHints.value.trim(),
        length_bias: pfLengthBias.value || "match",
    };
}

function openPersonaRefine() {
    prBefore = currentPersonaFromForm();
    if (!prBefore.system_prompt.trim()) {
        return showPersonaFormError("There is no system prompt here to refine yet.");
    }
    prRevision = null;
    prTitle.textContent = `Refine ${prBefore.name || "this persona"}`;
    prCurrentPrompt.textContent = prBefore.system_prompt;
    prInstruction.value = "";
    prResult.classList.add("hidden");
    prUseBtn.classList.add("hidden");
    prRepliesEl.classList.add("hidden");
    prRepliesEl.innerHTML = "";
    prRefineBtn.textContent = "Refine it";
    hidePersonaRefineError();
    prOverlay.classList.remove("hidden");
    prInstruction.focus();
}

function closePersonaRefine() {
    prOverlay.classList.add("hidden");
}

function showPersonaRefineError(msg) {
    prErrorEl.textContent = msg;
    prErrorEl.classList.remove("hidden");
}

function hidePersonaRefineError() {
    prErrorEl.classList.add("hidden");
}

async function requestPersonaRefinement() {
    const instruction = prInstruction.value.trim();
    if (!instruction) return showPersonaRefineError("Say what should change first.");

    hidePersonaRefineError();
    prRefineBtn.disabled = true;
    prRefineBtn.textContent = "Refining…";

    try {
        const resp = await fetch("/api/personas/refine", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...prBefore, instruction }),
        });
        if (!resp.ok) {
            const detail = await resp.json().catch(() => null);
            return showPersonaRefineError(
                (detail && detail.detail) || `Could not refine (HTTP ${resp.status}).`
            );
        }
        prRevision = await resp.json();
    } catch (err) {
        console.error("Persona refine failed:", err);
        return showPersonaRefineError("Could not reach the server.");
    } finally {
        prRefineBtn.disabled = false;
        prRefineBtn.textContent = "Refine again";
    }

    renderList(prNotesEl, prRevision.notes || []);
    renderList(prWarningsEl, prRevision.warnings || []);
    prWarningsRow.classList.toggle("hidden", !(prRevision.warnings || []).length);
    // Editable: a revision is usually nearly right, and the one word worth
    // changing is easier to change here than after another round trip.
    prPromptEl.value = prRevision.system_prompt;
    prRepliesEl.classList.add("hidden");
    prRepliesEl.innerHTML = "";
    prResult.classList.remove("hidden");
    prUseBtn.classList.remove("hidden");
    prResult.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function tryPersonaRefinement() {
    if (!prRevision) return;
    const question = prQuestionEl.value.trim();
    if (!question) return showPersonaRefineError("Give them a question to answer.");

    hidePersonaRefineError();
    prTryBtn.disabled = true;
    prTryBtn.textContent = "Asking…";
    try {
        const resp = await fetch("/api/personas/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: prBefore.name,
                // Whatever is in the box, which may have been edited since
                // the revision came back.
                system_prompt: prPromptEl.value,
                description: prRevision.description,
                length_bias: prRevision.length_bias,
                label: "After",
                question,
                compare_prompt: prBefore.system_prompt,
                compare_length_bias: prBefore.length_bias,
                compare_label: "Before",
            }),
        });
        if (!resp.ok) {
            const detail = await resp.json().catch(() => null);
            return showPersonaRefineError(
                (detail && detail.detail) || `Could not run it (HTTP ${resp.status}).`
            );
        }
        const body = await resp.json();
        prRepliesEl.innerHTML = "";
        // Before first: it is the thing being changed, and reading the
        // revision first makes it hard to see what actually moved.
        for (const reply of [body.comparison, body.draft].filter(Boolean)) {
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
            prRepliesEl.appendChild(box);
        }
        prRepliesEl.classList.remove("hidden");
        prRepliesEl.scrollIntoView({ behavior: "smooth", block: "end" });
    } catch (err) {
        console.error("Persona preview failed:", err);
        showPersonaRefineError("Could not reach the server.");
    } finally {
        prTryBtn.disabled = false;
        prTryBtn.textContent = "Run both";
    }
}

/**
 * Copy the revision into the editor form and close.
 *
 * Only the fields a refinement is allowed to touch — the name and the
 * avatar are the user's, and a revision that quietly repainted them
 * would be a surprise. Still not a save: the form is the review step.
 */
function usePersonaRefinement() {
    if (!prRevision) return;
    pfSystemPrompt.value = prPromptEl.value;
    pfDescription.value = prRevision.description;
    pfRouterHints.value = prRevision.router_hints;
    pfLengthBias.value = prRevision.length_bias || "match";
    closePersonaRefine();
    pfSystemPrompt.focus();
}

function setupPersonaRefineEventListeners() {
    document.getElementById("pf-btn-refine").addEventListener("click", openPersonaRefine);
    prRefineBtn.addEventListener("click", requestPersonaRefinement);
    prTryBtn.addEventListener("click", tryPersonaRefinement);
    prUseBtn.addEventListener("click", usePersonaRefinement);
    document.getElementById("pr-btn-close").addEventListener("click", closePersonaRefine);
    document.getElementById("pr-btn-cancel").addEventListener("click", closePersonaRefine);
    prOverlay.addEventListener("click", (e) => {
        if (e.target === prOverlay) closePersonaRefine();
    });
    prInstruction.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            requestPersonaRefinement();
        }
    });
}
