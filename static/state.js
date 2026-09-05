/**
 * state.js — Shared application state and DOM references.
 *
 * All modules read from and write to these globals. This avoids circular
 * dependencies between feature modules while keeping a single source of truth.
 */

/* ==========================================================================
   Application State
   ========================================================================== */

let personas = [];
let selectedPersona = null;
let personaNameMentionsEnabled = true;
let maxPersonaReplies = 1;
let maxTurnsForContext = 6;
let ttsEnabled = false;
let ttsAvailable = false;
let ttsStreaming = false;
let ttsServerType = "";
let sttAvailable = false;
let isStreaming = false;

// Chat room state
let currentChatRoom = "default";
let allChatRooms = [];
let roomPersonas = {};

// Microphone / STT state
let mediaRecorder = null;
let recordedChunks = [];

// Non-streaming: FIFO audio queue
const audioQueue = [];
let audioCtx = null;
let isPlayingAudio = false;

// Streaming TTS state
let sentenceBuffer = "";
let currentStreamingPersona = null;

// Streaming: decoupled fetch queue and decoded-buffer playback queue
const ttsRequestQueue = [];
const audioBufferQueue = [];
let isFetchingTTS = false;
let isPlayingAudioBuffer = false;

// Chat persistence — track message IDs for audio association
let pendingUserMessageId = null; // UUID generated before sending, used for STT audio
// UUID issued by the server in the "start" event; stamped onto TTS items at enqueue time
let currentAssistantMessageId = null;
let currentAssistantRow = null; // The active assistant bubble row (updated on each "start" event)

const THEME_STORAGE_KEY = "talkwithme_theme";

/* ==========================================================================
   DOM References — grouped by the module that uses them
   ========================================================================== */

// Main UI (used by chat.js, stt.js, app.js)
const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("message-input");
const sendBtn = document.getElementById("btn-send");
const suggestBtn = document.getElementById("btn-suggest");
const micBtn = document.getElementById("btn-mic");
const newChatBtn = document.getElementById("btn-new-chat");
const ttsToggleBtn = document.getElementById("btn-tts-toggle");
const ttsIcon = document.getElementById("tts-icon");
const personaListEl = document.getElementById("persona-list");
const themeSelectEl = document.getElementById("theme-select");

// Chat room selector (used by chatrooms.js)
const chatRoomDropdown = document.getElementById("chat-room-dropdown");
const btnAddPersona = document.getElementById("btn-add-persona");

// Persona Editor (used by persona.js)
const personaEditorOverlay = document.getElementById("persona-editor-overlay");
const peListView = document.getElementById("pe-list-view");
const peFormView = document.getElementById("pe-form-view");
const peListEl = document.getElementById("pe-list");
const peFormTitle = document.getElementById("pe-form-title");
const peFormError = document.getElementById("pe-form-error");
const peForm = document.getElementById("pe-form");
const peConfirmOverlay = document.getElementById("pe-confirm-overlay");
const peConfirmMsg = document.getElementById("pe-confirm-msg");

// Persona Editor form fields
const pfName = document.getElementById("pf-name");
const pfDescription = document.getElementById("pf-description");
const pfSystemPrompt = document.getElementById("pf-system-prompt");
const pfRouterHints = document.getElementById("pf-router-hints");
const pfAvatarColor = document.getElementById("pf-avatar-color");
const pfReferenceAudioLanguage = document.getElementById("pf-reference-audio-language");
const pfAvatarImage = document.getElementById("pf-avatar-image");      // <input type=file>
const pfAvatarPreview = document.getElementById("pf-avatar-preview");
const pfAvatarRemoveBtn = document.getElementById("pf-avatar-remove");
const pfReferenceAudio = document.getElementById("pf-reference-audio"); // <input type=file>
const pfAudioStatus = document.getElementById("pf-audio-status");
const pfAudioPlayBtn = document.getElementById("pf-audio-play");
const pfAudioRemoveBtn = document.getElementById("pf-audio-remove");
const pfReferenceAudioTx = document.getElementById("pf-reference-audio-transcript");
const pfAllowToolCalls = document.getElementById("pf-allow-tool-calls");
const pfLengthBias = document.getElementById("pf-length-bias");
const gsfTypicalLength = document.getElementById("gsf-typical-length");

// "Start fresh" — wiping stored context (used by gen-settings.js)
const gsfContextSummary = document.getElementById("gsf-context-summary");
const gsfWipeRooms = document.getElementById("gsf-wipe-rooms");
const gsfWipeMemories = document.getElementById("gsf-wipe-memories");
const gsfWipePlayingAs = document.getElementById("gsf-wipe-playing-as");
const gsfWipeBtn = document.getElementById("gsf-btn-wipe");
const gsfWipeResult = document.getElementById("gsf-wipe-result");
const gsfWipeOverlay = document.getElementById("gsf-wipe-overlay");
const gsfWipePlan = document.getElementById("gsf-wipe-plan");

// Who the player is playing — one of the personas, or nobody
const btnPlayingAs = document.getElementById("btn-playing-as");
const playingAsOverlay = document.getElementById("playing-as-overlay");
const paListEl = document.getElementById("pa-list");

// Room editor (used by chatrooms.js)
const roomEditOverlay = document.getElementById("room-edit-overlay");
const reTypicalLength = document.getElementById("re-typical-length");
const reRequirePersona = document.getElementById("re-require-persona");

// "Speak as" — the player writes a line and a persona says it verbatim
const speakAsOverlay = document.getElementById("speak-as-overlay");
const saTextEl = document.getElementById("sa-text");
const pfMemorySize = document.getElementById("pf-memory-size");
const pfMemoriesClearBtn = document.getElementById("pf-memories-clear");
const pfDraftBtn = document.getElementById("pf-btn-draft");
const pfDraftHint = document.getElementById("pf-draft-hint");
const pfRefineBtn = document.getElementById("pf-btn-refine");
const pfRefineHint = document.getElementById("pf-refine-hint");

// Persona drafting (used by persona-draft.js)
const pdOverlay = document.getElementById("pd-overlay");
const pdBrief = document.getElementById("pd-brief");
// Queried as lists rather than by id: the dials and detail fields are
// rendered from the server's constants, so the JS must not hold a
// hand-written copy of which ones exist.
const pdDials = document.querySelectorAll("select.pd-dial");
const pdDetails = document.querySelectorAll("input.pd-detail");
const pdErrorEl = document.getElementById("pd-error");
const pdResult = document.getElementById("pd-result");
const pdNotesEl = document.getElementById("pd-notes");
const pdWarningsEl = document.getElementById("pd-warnings");
const pdWarningsRow = document.getElementById("pd-warnings-row");
const pdQuestionEl = document.getElementById("pd-question");
const pdCompareEl = document.getElementById("pd-compare");
const pdRepliesEl = document.getElementById("pd-replies");
const pdDraftBtn = document.getElementById("pd-btn-draft");
const pdTryBtn = document.getElementById("pd-btn-try");
const pdUseBtn = document.getElementById("pd-btn-use");

// Persona refining (used by persona-refine.js)
const prOverlay = document.getElementById("pr-overlay");
const prTitle = document.getElementById("pr-title");
const prCurrentPrompt = document.getElementById("pr-current-prompt");
const prInstruction = document.getElementById("pr-instruction");
const prErrorEl = document.getElementById("pr-error");
const prResult = document.getElementById("pr-result");
const prNotesEl = document.getElementById("pr-notes");
const prWarningsEl = document.getElementById("pr-warnings");
const prWarningsRow = document.getElementById("pr-warnings-row");
const prPromptEl = document.getElementById("pr-prompt");
const prQuestionEl = document.getElementById("pr-question");
const prRepliesEl = document.getElementById("pr-replies");
const prRefineBtn = document.getElementById("pr-btn-refine");
const prTryBtn = document.getElementById("pr-btn-try");
const prUseBtn = document.getElementById("pr-btn-use");

// Persona editor editing state
let peEditingName = null;

// Chat Rooms Editor (used by chatrooms.js)
const chatroomsOverlay = document.getElementById("chatrooms-overlay");
const crListEl = document.getElementById("cr-list");
const crFormError = document.getElementById("cr-form-error");
const crNewForm = document.getElementById("cr-new-form");
const crNameInput = document.getElementById("cr-name-input");
const crConfirmOverlay = document.getElementById("cr-confirm-overlay");
const crConfirmMsg = document.getElementById("cr-confirm-msg");

// Persona Picker (used by chatrooms.js)
const personaPickerOverlay = document.getElementById("persona-picker-overlay");
const ppListEl = document.getElementById("pp-list");
let ppSelectedNames = [];

// Settings Modal (used by settings.js)
const settingsOverlay = document.getElementById("settings-overlay");
const settingsForm = document.getElementById("settings-form");
const settingsError = document.getElementById("settings-error");

// Settings form fields — LLM
const sfLlmBaseUrl = document.getElementById("sf-llm-base-url");
const sfLlmModel = document.getElementById("sf-llm-model");
const sfLlmMaxTokens = document.getElementById("sf-llm-max-tokens");
const sfLlmTemperature = document.getElementById("sf-llm-temperature");
const sfLlmPromptFormat = document.getElementById("sf-llm-prompt-format");

// Settings form fields — TTS
const sfTtsEnabled = document.getElementById("sf-tts-enabled");
const sfTtsFields = document.getElementById("sf-tts-fields");
const sfTtsBaseUrl = document.getElementById("sf-tts-base-url");
const sfTtsNumSteps = document.getElementById("sf-tts-num-steps");
const sfTtsGuidanceScale = document.getElementById("sf-tts-guidance-scale");
const sfTtsSeed = document.getElementById("sf-tts-seed");
const sfTtsTimeout = document.getElementById("sf-tts-timeout");
const sfTtsStreaming = document.getElementById("sf-tts-streaming");
const sfTtsServerType = document.getElementById("sf-tts-server-type");

// Settings form fields — STT
const sfSttEnabled = document.getElementById("sf-stt-enabled");
const sfSttFields = document.getElementById("sf-stt-fields");
const sfSttBaseUrl = document.getElementById("sf-stt-base-url");
const sfSttTimeout = document.getElementById("sf-stt-timeout");

// General Settings Modal (used by gen-settings.js)
const genSettingsOverlay = document.getElementById("gen-settings-overlay");
const genSettingsForm = document.getElementById("gen-settings-form");
const genSettingsError = document.getElementById("gen-settings-error");
const gsfMaxPersonaReplies = document.getElementById("gsf-max-persona-replies");
const gsfPersonaNameMentions = document.getElementById("gsf-persona-name-mentions");
const gsfMaxTurnsForContext = document.getElementById("gsf-max-turns-for-context");
const gsfShowToolCalls = document.getElementById("gsf-show-tool-calls");
const gsfEnablePersonaMemories = document.getElementById("gsf-enable-persona-memories");
