/* ============================================================
   PDF Chatbot — Phase 3 frontend
   Handles: upload, document list, session management,
            streaming chat (SSE), source attribution panel.
   ============================================================ */

// ── State ─────────────────────────────────────────────────────
let sessionId = null;          // active chat session UUID
let isStreaming = false;       // block concurrent sends
let currentSources = [];       // sources for latest bot turn

// ── Element refs ──────────────────────────────────────────────
const dropzone        = document.getElementById("dropzone");
const fileInput       = document.getElementById("file-input");
const uploadStatus    = document.getElementById("upload-status");
const docList         = document.getElementById("document-list");
const sessionInfo     = document.getElementById("session-info");
const newChatBtn      = document.getElementById("new-chat-btn");
const providerBadge   = document.getElementById("provider-badge");
const chatMessages    = document.getElementById("chat-messages");
const welcomeScreen   = document.getElementById("welcome-screen");
const welcomeStartBtn = document.getElementById("welcome-start-btn");
const chatInput       = document.getElementById("chat-input");
const sendBtn         = document.getElementById("send-btn");
const sourcesPanel    = document.getElementById("sources-panel");
const sourcesList     = document.getElementById("sources-list");
const closeSourcesBtn = document.getElementById("close-sources-btn");

// ── Init ──────────────────────────────────────────────────────
(async function init() {
  await loadDocuments();
  await loadProviderInfo();
})();

// ── Provider badge ────────────────────────────────────────────
async function loadProviderInfo() {
  try {
    const res  = await fetch("/api/health");
    const data = await res.json();
    providerBadge.textContent = `LLM: ${data.llm_provider ?? "?"} · ${data.vector_store_chunks ?? 0} chunks`;
  } catch {
    providerBadge.textContent = "server unreachable";
  }
}

// ── Upload ────────────────────────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);

dropzone.addEventListener("drop", e => handleFiles(Array.from(e.dataTransfer.files || [])));
fileInput.addEventListener("change", () => {
  handleFiles(Array.from(fileInput.files || []));
  fileInput.value = "";
});

function setUploadStatus(msg, type) {
  uploadStatus.textContent = msg;
  uploadStatus.className   = "upload-status " + (type || "");
}

async function handleFiles(files) {
  const pdfs = files.filter(f => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
  if (!pdfs.length) { setUploadStatus("Please select PDF files.", "error"); return; }
  for (const f of pdfs) await uploadFile(f);
  await loadProviderInfo();
}

async function uploadFile(file) {
  setUploadStatus(`Uploading "${file.name}"…`, "");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res  = await fetch("/api/documents/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { setUploadStatus(`Error: ${data.detail || "Upload failed."}`, "error"); return; }

    const ok = data.status === "indexed";
    setUploadStatus(
      ok ? `✓ "${file.name}" — ${data.num_pages} pages, ${data.num_chunks} chunks indexed`
         : `⚠ "${file.name}" saved but not indexed: ${data.message}`,
      ok ? "success" : "error"
    );
    if (ok) addDocToList({ document_id: data.document_id, filename: data.filename,
                           num_pages: data.num_pages, num_chunks: data.num_chunks, indexed: true });
  } catch {
    setUploadStatus(`Network error uploading "${file.name}".`, "error");
  }
}

// ── Document list ─────────────────────────────────────────────
async function loadDocuments() {
  try {
    const res  = await fetch("/api/documents");
    const data = await res.json();
    docList.innerHTML = "";
    if (!data.documents?.length) { renderEmptyDocList(); return; }
    data.documents.forEach(addDocToList);
  } catch {
    docList.innerHTML = "";
    renderEmptyDocList();
  }
}

function renderEmptyDocList() {
  const li = document.createElement("li");
  li.className = "empty-note";
  li.textContent = "No documents yet.";
  docList.appendChild(li);
}

function addDocToList(doc) {
  // Remove empty-note if present
  docList.querySelector(".empty-note")?.remove();

  // Don't duplicate
  if (docList.querySelector(`[data-id="${doc.document_id}"]`)) return;

  const li = document.createElement("li");
  li.dataset.id = doc.document_id;
  const chunks = doc.num_chunks || doc.num_chunks === 0 ? `${doc.num_chunks} chunks` : "";
  li.innerHTML = `
    <div class="doc-row">
      <div>
        <span class="doc-name" title="${esc(doc.filename)}">📄 ${esc(doc.filename)}</span>
        <span class="doc-meta">${doc.num_pages} pages${chunks ? " · " + chunks : ""}</span>
      </div>
      <button class="btn-delete-doc" data-id="${doc.document_id}" title="Delete">✕</button>
    </div>
  `;
  li.querySelector(".btn-delete-doc").addEventListener("click", e => {
    e.stopPropagation();
    deleteDocument(doc.document_id, li);
  });
  docList.prepend(li);
}

async function deleteDocument(docId, li) {
  if (!confirm("Delete this document and remove it from the index?")) return;
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
    if (res.ok) {
      li.remove();
      if (!docList.querySelector("li")) renderEmptyDocList();
      await loadProviderInfo();
    } else {
      const d = await res.json();
      alert(d.detail || "Delete failed.");
    }
  } catch { alert("Network error."); }
}

// ── Session ───────────────────────────────────────────────────
welcomeStartBtn.addEventListener("click", startNewSession);
newChatBtn.addEventListener("click", startNewSession);

async function startNewSession() {
  try {
    const res  = await fetch("/api/chat/sessions", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_ids: [] }),   // search all docs
    });
    const data = await res.json();
    sessionId = data.session_id;

    // Reset UI
    chatMessages.innerHTML = "";
    closeSources();
    enableInput();
    sessionInfo.textContent = `Session: ${sessionId.slice(0, 8)}…`;

    appendSystemMsg("Session started. Ask me anything about your documents.");
    chatInput.focus();
  } catch {
    alert("Could not create session. Is the server running?");
  }
}

function enableInput() {
  chatInput.disabled = false;
  sendBtn.disabled   = false;
}

function disableInput() {
  chatInput.disabled = true;
  sendBtn.disabled   = true;
}

// ── Input handling ────────────────────────────────────────────
chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    trySend();
  }
});
sendBtn.addEventListener("click", trySend);

// Auto-resize textarea
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px";
});

function trySend() {
  const q = chatInput.value.trim();
  if (!q || isStreaming || !sessionId) return;
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendMessage(q);
}

// ── Streaming chat ────────────────────────────────────────────
async function sendMessage(question) {
  if (!sessionId) { alert("Start a new chat session first."); return; }
  isStreaming = true;
  disableInput();
  closeSources();

  // Render user bubble
  appendUserMsg(question);

  // Thinking indicator
  const thinkingId = appendThinking();

  try {
    const res = await fetch("/api/chat/stream", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ session_id: sessionId, question }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      removeThinking(thinkingId);
      appendBotMsg(`⚠ Error: ${err.detail || res.statusText}`, []);
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";
    let   sources = [];
    let   botBubble = null;   // created on first token
    let   fullText  = "";

    removeThinking(thinkingId);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch { continue; }

        if (event.type === "sources") {
          sources = event.payload || [];

        } else if (event.type === "token") {
          fullText += event.payload;
          if (!botBubble) {
            botBubble = createBotBubble();
          }
          botBubble.textContent = fullText;
          botBubble.classList.add("streaming-cursor");
          scrollToBottom();

        } else if (event.type === "done") {
          if (botBubble) botBubble.classList.remove("streaming-cursor");
          attachSourcesButton(botBubble?.closest(".msg-content"), sources);
          currentSources = sources;

        } else if (event.type === "error") {
          if (botBubble) {
            botBubble.classList.remove("streaming-cursor");
          } else {
            botBubble = createBotBubble();
          }
          botBubble.textContent = `⚠ ${event.payload}`;
        }
      }
    }

    // Edge case: if no tokens arrived at all
    if (!botBubble) appendBotMsg("(No response received.)", sources);

  } catch (err) {
    removeThinking(thinkingId);
    appendBotMsg(`⚠ Network error: ${err.message}`, []);
  } finally {
    isStreaming = false;
    enableInput();
    chatInput.focus();
    await loadProviderInfo();
  }
}

// ── Message rendering ─────────────────────────────────────────
function appendUserMsg(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="msg-avatar">🧑</div>
    <div class="msg-content">
      <div class="msg-bubble">${esc(text)}</div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();
}

function createBotBubble() {
  // Create a bot row with an empty bubble and return the bubble element
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-content">
      <div class="msg-bubble"></div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();
  return row.querySelector(".msg-bubble");
}

function appendBotMsg(text, sources) {
  const bubble = createBotBubble();
  bubble.textContent = text;
  attachSourcesButton(bubble.closest(".msg-content"), sources);
  scrollToBottom();
}

function appendSystemMsg(text) {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-content">
      <div class="msg-bubble" style="color:var(--muted);font-style:italic">${esc(text)}</div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();
}

function appendThinking() {
  const id  = "thinking-" + Date.now();
  const row = document.createElement("div");
  row.className = "msg-row bot thinking-row";
  row.id = id;
  row.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-content">
      <div class="msg-bubble"><span class="thinking-dots"></span> Thinking</div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();
  return id;
}

function removeThinking(id) {
  document.getElementById(id)?.remove();
}

function attachSourcesButton(contentEl, sources) {
  if (!contentEl || !sources?.length) return;
  const btn = document.createElement("button");
  btn.className = "msg-sources-btn";
  btn.textContent = `📎 ${sources.length} source${sources.length > 1 ? "s" : ""}`;
  btn.addEventListener("click", () => openSources(sources));
  contentEl.appendChild(btn);
}

// ── Sources panel ─────────────────────────────────────────────
closeSourcesBtn.addEventListener("click", closeSources);

function openSources(sources) {
  sourcesList.innerHTML = "";
  sources.forEach((s, i) => {
    const pages = s.page_numbers.join(", ");
    const card  = document.createElement("div");
    card.className = "source-card";
    card.innerHTML = `
      <div class="source-card-header">
        <span>📄 ${esc(s.filename)}</span>
        <span class="source-card-page">p. ${esc(pages)}</span>
      </div>
      <div class="source-card-excerpt">${esc(s.excerpt)}${s.excerpt.length >= 300 ? "…" : ""}</div>
      <div class="source-score">similarity ${s.similarity.toFixed(3)}</div>
    `;
    sourcesList.appendChild(card);
  });
  sourcesPanel.classList.add("open");
}

function closeSources() {
  sourcesPanel.classList.remove("open");
}

// ── Helpers ───────────────────────────────────────────────────
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
