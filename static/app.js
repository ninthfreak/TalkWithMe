/**
 * app.js — TalkWithMe frontend orchestrator.
 *
 * This file is intentionally slim. It coordinates initialization across
 * feature modules (state, chat, tts, stt, persona, chatrooms, settings, theme).
 *
 * Cross-cutting concerns that multiple modules depend on live here:
 *  - loadPersonas() — fetched by persona.js after CRUD, triggers loadChatRooms()
 *  - Health checks — called by init() and settings.js after save
 *  - Event listener setup — wires up topbar buttons shared across modules
 */

/* ==========================================================================
   Initialization
   ========================================================================== */

/**
 * Run one startup step without letting it take the others down.
 *
 * Startup used to be a straight sequence, which meant the first failure
 * silently cancelled everything after it: one missing element in
 * setupEventListeners() left the room dropdown and the character picker
 * unbound and the current room's history unrendered, with nothing on
 * screen to say why. Each step is independent, so each gets to fail
 * alone — loudly in the console, and visibly in the chat panel.
 */
async function step(label, fn) {
    try {
        return await fn();
    } catch (err) {
        console.error(`Startup step failed: ${label}`, err);
        startupFailures.push(label);
        return undefined;
    }
}

/** Startup steps that threw, reported once at the end rather than per failure. */
const startupFailures = [];

async function init() {
    await step("theme", () => initTheme());
    await step("personas and chat rooms", () => loadPersonas());
    await step("text-to-speech health", () => checkTTSHealth());
    await step("speech-to-text health", () => checkSTTHealth());
    await step("general settings", () => loadGeneralSettings());
    await step("chat controls", () => setupEventListeners());
    await step("chat room controls", () => setupChatRoomEventListeners());
    await step("persona drafting controls", () => setupPersonaDraftEventListeners());
    await step("persona refining controls", () => setupPersonaRefineEventListeners());

    // Load persisted history for the current room
    await step("stored conversation", async () => {
        const history = await loadPersistedHistory(currentChatRoom);
        renderPersistedHistory(history.messages, currentChatRoom);
    });

    if (startupFailures.length) {
        // Said in the chat panel, not only the console: the symptom of a
        // half-initialised page is controls quietly doing nothing, which
        // reads as the app being broken rather than as an error.
        appendErrorBubble(
            `Some of the page did not start up (${startupFailures.join(", ")}). ` +
            `If the app was just updated, reload with Ctrl+Shift+R. ` +
            `The browser console has the details.`
        );
    }
}

/**
 * Fetch general settings from the server. Currently only used to gate
 * the persona-name-mention detection feature.
 */
async function loadGeneralSettings() {
    try {
        const resp = await fetch("/api/settings");
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.general != null) {
            personaNameMentionsEnabled = data.general.persona_name_mentions;
            maxPersonaReplies = data.general.max_persona_replies ?? 1;
            maxTurnsForContext = data.general.max_turns_for_context ?? 6;
        }
    } catch (err) {
        console.warn("Failed to load general settings, using defaults:", err);
    }
}

/**
 * Load personas from server, then refresh chat rooms so persona lists
 * in each room are up to date (handles rename/delete cascades).
 * Called by init() on startup and by persona.js after CRUD operations.
 */
async function loadPersonas() {
    try {
        const resp = await fetch("/api/personas");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        // Keep the shared global sorted: consumers like applyChatRoomFilter()
        // auto-select `filtered[0]`, which should match the first rendered
        // (alphabetical) card in the sidebar.
        const list = await resp.json();
        personas = list.sort(comparePersonasByName);
        await loadChatRooms();
    } catch (err) {
        console.error("Failed to load personas:", err);
    }
}

/* ==========================================================================
   Health checks
   ========================================================================== */

async function checkTTSHealth() {
    try {
        const resp = await fetch("/api/tts/health");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        ttsAvailable = data.available;
        ttsStreaming = data.streaming || false;
        ttsEnabled = data.available; // Default on if server is available
        ttsServerType = data.server_type || "";
        updateTTSToggleUI();
    } catch (err) {
        console.warn("TTS health check failed:", err);
        ttsAvailable = false;
        ttsStreaming = false;
        ttsEnabled = false;
        ttsServerType = "";
        updateTTSToggleUI();
    }
}

async function checkSTTHealth() {
    try {
        const resp = await fetch("/api/stt/health");
        const data = await resp.json();
        sttAvailable = data.available;
        updateMicButtonUI();
    } catch (err) {
        console.warn("STT health check failed:", err);
        sttAvailable = false;
        updateMicButtonUI();
    }
}

/* ==========================================================================
   Top-level event listeners (shared UI controls)
   ========================================================================== */

function setupEventListeners() {
    // bind() rather than addEventListener() throughout: this function runs
    // in the middle of init(), so a single missing element used to abort
    // everything after it — the chat-room dropdown, the character picker
    // and the history load all being bound or run later. See bind().
    bind(sendBtn, "click", sendMessage, "Send");
    bind(suggestBtn, "click", suggestMessage, "Suggest a message");
    bind(continueBtn, "click", continueConversation, "Continue");
    inputEl.addEventListener("keydown", (e) => {
        // Enter sends; Shift+Enter for newline
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    inputEl.addEventListener("input", () => {
        inputEl.style.height = "auto";
        inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
    });

    bind(newChatBtn, "click", newChat, "New Chat");
    bind(ttsToggleBtn, "click", toggleTTS, "the speech toggle");
    bind(micBtn, "click", toggleMicrophone, "the microphone");
    bind(themeSelectEl, "change", () => applyTheme(themeSelectEl.value, true), "the theme picker");

    document.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.code === "Space" && !micBtn.disabled) {
            e.preventDefault();
            toggleMicrophone();
        }
    });
}

/* ==========================================================================
   Session management
   ========================================================================== */

async function newChat() {
    try {
        // POST /api/session/new clears both the in-memory session AND
        // the persisted files for the current room.
        await fetch("/api/session/new", { method: "POST" });
        messagesEl.innerHTML = "";
        showEmptyState();
    } catch (err) {
        console.error("Failed to reset session:", err);
    }
}

/* ==========================================================================
   Boot
   ========================================================================== */

init();
