/**
 * chatrooms.js — Chat room management: CRUD, persona picker, room switching.
 *
 * Handles:
 *  - Loading and rendering the chat room dropdown
 *  - Filtering personas by room
 *  - Chat room editor modal (create, delete rooms)
 *  - Persona picker modal (add personas to a room)
 *  - Removing personas from rooms
 */

/* ==========================================================================
   Chat room loading and filtering
   ========================================================================== */

/**
 * Load all chat rooms from the server and initialize the room state.
 * After loading, applies the current room filter and renders the persona list.
 */
async function loadChatRooms() {
    try {
        const resp = await fetch("/api/chatrooms/all");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        allChatRooms = await resp.json();

        // Build the persona map
        roomPersonas = {};
        for (const room of allChatRooms) {
            roomPersonas[room.name] = room.persona_names;
        }

        // Populate the dropdown
        renderChatRoomDropdown();

        // If previously selected room no longer exists, revert to default
        const roomExists = allChatRooms.some(r => r.name === currentChatRoom);
        if (!roomExists && allChatRooms.length > 0) {
            currentChatRoom = "default";
            chatRoomDropdown.value = "default";
        }

        // Apply the current room filter and render
        applyChatRoomFilter();
    } catch (err) {
        console.error("Failed to load chat rooms:", err);
        // Fallback: show all personas in "default" room
        currentChatRoom = "default";
        renderPersonaList();
    }
}

/**
 * Apply the current chat room filter: update persona list, active session,
 * and UI controls (add/remove buttons).
 */
function applyChatRoomFilter() {
    const isActiveRoom = currentChatRoom !== "default";
    const roomPersonaNames = roomPersonas[currentChatRoom] || [];

    // Filter the persona list to only those in this room
    const filtered = isActiveRoom
        ? personas.filter(p => roomPersonaNames.includes(p.name))
        : [...personas];

    // Update the persona list rendering
    renderPersonaList(filtered, isActiveRoom);

    // Select first persona if none selected or selected one not in room
    if (filtered.length > 0) {
        if (!selectedPersona || !filtered.some(p => p.name === selectedPersona)) {
            selectedPersona = filtered[0].name;
        }
        highlightSelectedPersona();
    } else {
        selectedPersona = null;
    }

    // Activate only the room's personas in the session
    const activeNames = filtered.map(p => p.name);
    if (activeNames.length > 0) {
        fetch("/api/session/personas", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ active_personas: activeNames }),
        }).catch(err => console.error("Failed to update session personas:", err));
    }

    // Show/hide the "Add persona" button
    if (isActiveRoom) {
        btnAddPersona.classList.remove("hidden");
    } else {
        btnAddPersona.classList.add("hidden");
    }

    // Update dropdown selection
    chatRoomDropdown.value = currentChatRoom;

    // Who the player is playing is deliberately untouched here: it is the
    // player's, not the room's, so a room switch neither loads nor resets
    // it. Only the "is this room's requirement met?" styling is refreshed.
    applyPlayingAsControls();
}

/**
 * Populate the chat room dropdown from the server's full list.
 */
function renderChatRoomDropdown() {
    chatRoomDropdown.innerHTML = "";
    // "default" (All Personas) always first; remainder sorted alphabetically, case-insensitive
    const sorted = [...allChatRooms].sort((a, b) => {
        if (a.name === "default") return -1;
        if (b.name === "default") return 1;
        return a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
    });
    for (const room of sorted) {
        const opt = document.createElement("option");
        opt.value = room.name;
        opt.textContent = room.name === "default" ? "All Personas" : room.name;
        chatRoomDropdown.appendChild(opt);
    }
}

/* ==========================================================================
   Event listeners
   ========================================================================== */

function setupChatRoomEventListeners() {
    // Dropdown change: switch rooms
    chatRoomDropdown.addEventListener("change", () => {
        switchChatRoom(chatRoomDropdown.value);
    });

    // Echo chamber toggle
    // "Add persona" button in sidebar
    btnAddPersona.addEventListener("click", openPersonaPicker);

    setupPlayingAsEventListeners();
    setupSpeakAsEventListeners();
    setupRoomEditorEventListeners();
    loadPlayer();

    // Chat rooms editor button in topbar
    document.getElementById("btn-chat-rooms").addEventListener("click", openChatRoomsEditor);
    document.getElementById("cr-btn-close").addEventListener("click", closeChatRoomsEditor);

    // New room form
    document.getElementById("cr-btn-new").addEventListener("click", showNewRoomForm);
    document.getElementById("cr-new-cancel").addEventListener("click", hideNewRoomForm);
    document.getElementById("cr-new-save").addEventListener("click", createChatRoom);

    // Delete confirmation
    document.getElementById("cr-confirm-cancel").addEventListener("click", () => {
        crConfirmOverlay.classList.add("hidden");
    });

    // Backdrop click to close
    chatroomsOverlay.addEventListener("click", (e) => {
        if (e.target === chatroomsOverlay) closeChatRoomsEditor();
    });
    crConfirmOverlay.addEventListener("click", (e) => {
        if (e.target === crConfirmOverlay) {
            crConfirmOverlay.classList.add("hidden");
        }
    });

    // Persona picker
    document.getElementById("pp-btn-close").addEventListener("click", closePersonaPicker);
    document.getElementById("pp-btn-cancel").addEventListener("click", closePersonaPicker);
    document.getElementById("pp-btn-add").addEventListener("click", addSelectedPersonasToRoom);
    personaPickerOverlay.addEventListener("click", (e) => {
        if (e.target === personaPickerOverlay) closePersonaPicker();
    });
}

/* ==========================================================================
   Room switching
   ========================================================================== */

/**
 * Switch to a different chat room. Clears the chat display, loads the
 * persisted history for the new room, and updates the persona list.
 */
async function switchChatRoom(roomName) {
    currentChatRoom = roomName;

    // Clear chat panel momentarily
    messagesEl.innerHTML = "";
    showEmptyState();

    // Load persisted history for this room (also resets the backend session)
    const history = await loadPersistedHistory(roomName);
    renderPersistedHistory(history.messages, roomName);

    // Re-apply filter
    applyChatRoomFilter();
}

/**
 * Remove a persona from the current chat room.
 */
async function removePersonaFromRoom(personaName) {
    if (currentChatRoom === "default") return; // Shouldn't happen, but guard anyway

    try {
        const resp = await fetch(
            `/api/chatrooms/${encodeURIComponent(currentChatRoom)}/personas/${encodeURIComponent(personaName)}`,
            { method: "DELETE" }
        );
        if (!resp.ok) {
            console.error("Failed to remove persona from room:", resp.status);
            return;
        }
        // Update local state
        if (roomPersonas[currentChatRoom]) {
            roomPersonas[currentChatRoom] = roomPersonas[currentChatRoom].filter(
                p => p !== personaName
            );
        }
        // If the removed persona was selected, clear selection
        if (selectedPersona === personaName) {
            const roomNames = roomPersonas[currentChatRoom] || [];
            selectedPersona = roomNames.length > 0 ? roomNames[0] : null;
        }
        applyChatRoomFilter();
    } catch (err) {
        console.error("Remove persona from room error:", err);
    }
}

/* ==========================================================================
   Chat Rooms Editor Modal
   ========================================================================== */

function openChatRoomsEditor() {
    hideNewRoomForm();
    chatroomsOverlay.classList.remove("hidden");
    crFormError.classList.add("hidden");
    renderChatRoomList();
}

function closeChatRoomsEditor() {
    chatroomsOverlay.classList.add("hidden");
    // Refresh the dropdown and re-apply room filter (in case rooms were deleted)
    loadChatRooms();
}

function renderChatRoomList() {
    crListEl.innerHTML = "";

    // Only show non-default rooms, sorted alphabetically (case-insensitive)
    const rooms = allChatRooms
        .filter(r => r.name !== "default")
        .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));

    if (rooms.length === 0) {
        crListEl.innerHTML = '<p class="cr-empty">No chat rooms yet. Click &ldquo;+ New Room&rdquo; to create one.</p>';
        return;
    }

    for (const room of rooms) {
        const item = document.createElement("div");
        item.className = "cr-list-item";

        const nameEl = document.createElement("span");
        nameEl.className = "cr-list-item-name";
        nameEl.textContent = room.name;

        const countEl = document.createElement("span");
        countEl.className = "cr-list-item-count";
        countEl.textContent = `${room.persona_names.length} persona${room.persona_names.length !== 1 ? 's' : ''}`;

        const editBtn = document.createElement("button");
        editBtn.className = "cr-list-item-edit";
        editBtn.textContent = "Edit";
        editBtn.title = `Edit "${room.name}" settings`;
        editBtn.addEventListener("click", () => openRoomEditor(room.name));

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "cr-list-item-delete";
        deleteBtn.textContent = "Delete";
        deleteBtn.title = `Delete "${room.name}"`;
        deleteBtn.addEventListener("click", () => confirmDeleteChatRoom(room.name));

        item.appendChild(nameEl);
        item.appendChild(countEl);
        item.appendChild(editBtn);
        item.appendChild(deleteBtn);
        crListEl.appendChild(item);
    }
}

/**
 * Save a partial room update. Every room-settings control goes through
 * here, so there is one request shape and one place to refresh cached
 * state — and a future room attribute needs a control and a field, not
 * another fetch helper.
 *
 * Returns the updated room; throws on failure for the caller to report.
 */
async function saveRoomSettings(roomName, patch) {
    const resp = await fetch(`/api/chatrooms/${encodeURIComponent(roomName)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const updated = await resp.json();
    // Keep the cached list in step so reopening any editor shows the saved
    // values rather than stale ones.
    const cached = allChatRooms.find(r => r.name.toLowerCase() === roomName.toLowerCase());
    if (cached) Object.assign(cached, updated);
    return updated;
}


function showNewRoomForm() {
    crNewForm.classList.remove("hidden");
    crListEl.classList.add("hidden");
    crNameInput.value = "";
    crFormError.classList.add("hidden");
    crNameInput.focus();
    // Allow Enter key to create the room
    crNameInput.onkeydown = (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            createChatRoom();
        }
    };
}

function hideNewRoomForm() {
    crNewForm.classList.add("hidden");
    crListEl.classList.remove("hidden");
}

async function createChatRoom() {
    const name = crNameInput.value.trim();

    if (!name) {
        crFormError.textContent = "Room name is required.";
        crFormError.classList.remove("hidden");
        return;
    }
    if (name.length > 20) {
        crFormError.textContent = "Room name must be 20 characters or fewer.";
        crFormError.classList.remove("hidden");
        return;
    }
    if (name.toLowerCase() === "default") {
        crFormError.textContent = "'default' is a reserved name and cannot be used.";
        crFormError.classList.remove("hidden");
        return;
    }
    if (!/^[a-zA-Z0-9 _-]+$/.test(name)) {
        crFormError.textContent = "Name may only contain letters, numbers, spaces, hyphens, and underscores.";
        crFormError.classList.remove("hidden");
        return;
    }

    try {
        const resp = await fetch("/api/chatrooms", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            crFormError.textContent = extractApiErrorMessage(err, resp.status);
            crFormError.classList.remove("hidden");
            return;
        }

        hideNewRoomForm();
        // Reload rooms to refresh the list
        await loadChatRooms();
        renderChatRoomList();
    } catch (err) {
        crFormError.textContent = "Request failed. Is the server running?";
        crFormError.classList.remove("hidden");
    }
}

function confirmDeleteChatRoom(name) {
    crConfirmMsg.textContent = `Delete chat room "${name}"? Personas will not be deleted, only unassigned from this room.`;
    crConfirmOverlay.classList.remove("hidden");

    const deleteBtn = document.getElementById("cr-confirm-delete");
    const newBtn = deleteBtn.cloneNode(true);
    deleteBtn.parentNode.replaceChild(newBtn, deleteBtn);
    newBtn.addEventListener("click", () => deleteChatRoom(name));
}

async function deleteChatRoom(name) {
    crConfirmOverlay.classList.add("hidden");
    try {
        const resp = await fetch(`/api/chatrooms/${encodeURIComponent(name)}`, { method: "DELETE" });
        if (!resp.ok) {
            console.error("Delete chat room failed:", resp.status);
            return;
        }
        // If we deleted the currently selected room, switch back to default
        if (currentChatRoom === name) {
            currentChatRoom = "default";
        }
        await loadChatRooms();
        renderChatRoomList();
    } catch (err) {
        console.error("Delete chat room error:", err);
    }
}

/* ==========================================================================
   Persona Picker Modal (for adding personas to a chat room)
   ========================================================================== */

function openPersonaPicker() {
    if (currentChatRoom === "default") return;

    ppSelectedNames = [];
    personaPickerOverlay.classList.remove("hidden");
    renderPersonaPickerList();
}

function closePersonaPicker() {
    personaPickerOverlay.classList.add("hidden");
}

function renderPersonaPickerList() {
    ppListEl.innerHTML = "";

    // Get personas already in this room
    const alreadyInRoom = new Set(roomPersonas[currentChatRoom] || []);

    if (personas.length === 0) {
        ppListEl.innerHTML = '<p class="pp-empty">No personas configured.</p>';
        return;
    }

    for (const p of personas) {
        const item = document.createElement("div");
        item.className = "pp-list-item";
        item.dataset.name = p.name;

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "pp-checkbox";
        checkbox.checked = false;
        checkbox.addEventListener("click", (e) => {
            e.stopPropagation();
            togglePickerSelection(p.name, checkbox.checked, item);
        });

        const avatar = document.createElement("div");
        avatar.className = "pp-list-item-avatar";
        avatar.style.backgroundColor = p.avatar_color;
        avatar.textContent = p.name.charAt(0).toUpperCase();

        const info = document.createElement("div");
        info.className = "pp-list-item-info";

        const nameEl = document.createElement("div");
        nameEl.className = "pp-list-item-name";
        nameEl.textContent = p.name;

        const descEl = document.createElement("div");
        descEl.className = "pp-list-item-desc";
        descEl.textContent = alreadyInRoom.has(p.name) ? (p.description || "") + " (already in room)" : (p.description || "");

        info.appendChild(nameEl);
        info.appendChild(descEl);

        item.appendChild(checkbox);
        item.appendChild(avatar);
        item.appendChild(info);

        // Clicking the row toggles the checkbox
        item.addEventListener("click", () => {
            const isChecked = !checkbox.checked;
            checkbox.checked = isChecked;
            togglePickerSelection(p.name, isChecked, item);
        });

        ppListEl.appendChild(item);
    }
}

function togglePickerSelection(name, isSelected, itemEl) {
    if (isSelected) {
        ppSelectedNames.push(name);
        if (itemEl) itemEl.classList.add("selected");
    } else {
        ppSelectedNames = ppSelectedNames.filter(n => n !== name);
        if (itemEl) itemEl.classList.remove("selected");
    }
}

async function addSelectedPersonasToRoom() {
    if (ppSelectedNames.length === 0 || currentChatRoom === "default") return;

    try {
        const resp = await fetch(
            `/api/chatrooms/${encodeURIComponent(currentChatRoom)}/personas`,
            {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ persona_names: ppSelectedNames }),
            }
        );

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            console.error("Failed to add personas to room:", extractApiErrorMessage(err, resp.status));
            return;
        }

        closePersonaPicker();
        // Reload to refresh the persona list
        await loadChatRooms();
    } catch (err) {
        console.error("Add personas to room error:", err);
    }
}

/* ==========================================================================
   Playing as — which persona the human has adopted

   Not a written profile and not room data: the personas already carry a
   name, a description and a system prompt, so the player picks one of them
   rather than describing a second, parallel character that could drift
   from the list. It follows the player between rooms; a room may *require*
   that one is picked (a property of the room, in the room editor), but it
   never owns the choice.
   ========================================================================== */

function currentRoomInfo() {
    return roomByName(currentChatRoom);
}

/** Who the player is playing, cached from /api/player. */
let player = { persona_name: "" };

async function loadPlayer() {
    try {
        const resp = await fetch("/api/player");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        player = await resp.json();
    } catch (err) {
        console.error("Failed to load who you are playing:", err);
    }
    applyPlayingAsControls();
}

/**
 * The name to show on the human's own messages, or "" for none.
 *
 * Resolved against the loaded persona list, matching what the server does:
 * the adopted persona can be deleted or renamed, and a dangling name must
 * degrade to an unlabelled bubble rather than showing a character who no
 * longer exists.
 */
function playerDisplayName() {
    const name = (player.persona_name || "").trim();
    if (!name) return "";
    return personas.some(p => p.name === name) ? name : "";
}

/** True when the current room demands a character and none is adopted. */
function personaRequiredButMissing() {
    const room = currentRoomInfo();
    if (!room || !room.require_player_persona) return false;
    return !playerDisplayName();
}

/** Reflect who the player is playing in the left panel. */
function applyPlayingAsControls() {
    const name = playerDisplayName();
    btnPlayingAs.textContent = name ? `Playing as: ${name}` : "Playing as\u2026";
    btnPlayingAs.classList.toggle("needs-character", personaRequiredButMissing());
    refreshRoomEditPlayingAsSummary();
    // The adopted persona is the player, so their card is marked and their
    // "speak as" button withdrawn — you speak as them by typing.
    highlightAdoptedPersona();
}

function roomByName(name) {
    return allChatRooms.find(
        r => r.name.toLowerCase() === String(name).toLowerCase());
}

function openPlayingAs() {
    renderPlayingAsList();
    hidePlayingAsError();
    playingAsOverlay.classList.remove("hidden");
}

/**
 * One radio per persona, plus "myself". Rendered from the live persona
 * list on every open rather than cached, so a persona added or deleted
 * since the last open is right.
 */
function renderPlayingAsList() {
    const chosen = (player.persona_name || "").trim();
    paListEl.innerHTML = "";

    const options = [{ name: "", label: "Myself \u2014 no character", desc: "" }].concat(
        personas.map(p => ({ name: p.name, label: p.name, desc: p.description || "" }))
    );

    for (const opt of options) {
        const row = document.createElement("label");
        row.className = "pa-option";

        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "playing_as";
        radio.value = opt.name;
        radio.checked = opt.name === chosen;

        const text = document.createElement("span");
        text.className = "pa-option-text";

        const nameEl = document.createElement("span");
        nameEl.className = "pa-option-name";
        nameEl.textContent = opt.label;
        text.appendChild(nameEl);

        if (opt.desc) {
            const descEl = document.createElement("span");
            descEl.className = "pa-option-desc";
            descEl.textContent = opt.desc;
            text.appendChild(descEl);
        }

        row.appendChild(radio);
        row.appendChild(text);
        paListEl.appendChild(row);
    }

    // Nothing matched — the stored name refers to a persona that is gone.
    // Fall back to "myself" so the dialog always shows a live selection.
    if (!paListEl.querySelector("input:checked")) {
        const first = paListEl.querySelector("input");
        if (first) first.checked = true;
    }
}

function closePlayingAs() {
    playingAsOverlay.classList.add("hidden");
}

function showPlayingAsError(msg) {
    const el = document.getElementById("pa-error");
    el.textContent = msg;
    el.classList.remove("hidden");
}

function hidePlayingAsError() {
    document.getElementById("pa-error").classList.add("hidden");
}

async function savePlayingAs(e) {
    if (e) e.preventDefault();
    const checked = paListEl.querySelector('input[name="playing_as"]:checked');
    const personaName = checked ? checked.value : "";

    try {
        const resp = await fetch("/api/player", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ persona_name: personaName }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        player = await resp.json();
    } catch (err) {
        console.error("Failed to save who you are playing:", err);
        return showPlayingAsError("Could not save. Is the server running?");
    }

    closePlayingAs();
    applyPlayingAsControls();
}

function setupPlayingAsEventListeners() {
    btnPlayingAs.addEventListener("click", openPlayingAs);
    document.getElementById("pa-form").addEventListener("submit", savePlayingAs);
    document.getElementById("pa-btn-close").addEventListener("click", closePlayingAs);
    document.getElementById("pa-btn-cancel").addEventListener("click", closePlayingAs);
    playingAsOverlay.addEventListener("click", (e) => {
        if (e.target === playingAsOverlay) closePlayingAs();
    });
}

/* ==========================================================================
   Room editor — every room attribute in one dialog

   Deliberately the canonical place to configure a room. The left-panel
   toggles are shortcuts for the room you are already in; both write through
   saveRoomSettings() and read from the same cached list, so they cannot
   drift apart.
   ========================================================================== */

let editingRoomName = null;

function openRoomEditor(roomName) {
    const room = allChatRooms.find(r => r.name.toLowerCase() === roomName.toLowerCase());
    if (!room) return;
    editingRoomName = room.name;

    document.getElementById("re-title").textContent = `Edit “${room.name}”`;
    reTypicalLength.value = room.typical_length || "normal";
    reRequirePersona.checked = !!room.require_player_persona;
    document.getElementById("re-personas").textContent =
        room.persona_names.length
            ? `${room.persona_names.length} assigned: ${room.persona_names.join(", ")}`
            : "No personas assigned yet.";
    refreshRoomEditPlayingAsSummary();
    hideRoomEditError();
    roomEditOverlay.classList.remove("hidden");
}

function refreshRoomEditPlayingAsSummary() {
    // Read-only: whether a character is adopted decides whether this
    // room's requirement can be met. Choosing one happens under
    // "Playing as", because the choice is the player's, not the room's.
    const el = document.getElementById("re-playing-as-summary");
    if (!el) return;
    const name = playerDisplayName();
    el.textContent = name
        ? `You are playing as ${name}.`
        : "You are playing as yourself \u2014 pick a character under \u201cPlaying as\u201d.";
}

function closeRoomEditor() {
    roomEditOverlay.classList.add("hidden");
    editingRoomName = null;
}

function showRoomEditError(msg) {
    const el = document.getElementById("re-error");
    el.textContent = msg;
    el.classList.remove("hidden");
}

function hideRoomEditError() {
    document.getElementById("re-error").classList.add("hidden");
}

async function submitRoomEditor(e) {
    if (e) e.preventDefault();
    if (!editingRoomName) return;

    // Send the whole form. The endpoint is a partial update, so adding a
    // future attribute here needs no change on either side beyond the
    // control and the field.
    const patch = {
        typical_length: reTypicalLength.value,
        require_player_persona: reRequirePersona.checked,
    };

    try {
        await saveRoomSettings(editingRoomName, patch);
    } catch (err) {
        console.error("Failed to save room settings:", err);
        return showRoomEditError("Could not save. Is the server running?");
    }

    closeRoomEditor();
    // The edited room may be the one on screen: refresh the left panel and
    // the room list so the new settings take effect immediately.
    await loadChatRooms();
    renderChatRoomList();
    // "Require a character" may have just turned on or off, which changes
    // whether the Playing as button reads as unmet.
    applyPlayingAsControls();
}

function setupRoomEditorEventListeners() {
    document.getElementById("re-form").addEventListener("submit", submitRoomEditor);
    document.getElementById("re-btn-close").addEventListener("click", closeRoomEditor);
    document.getElementById("re-btn-cancel").addEventListener("click", closeRoomEditor);
    roomEditOverlay.addEventListener("click", (e) => {
        if (e.target === roomEditOverlay) closeRoomEditor();
    });
}
