const api = typeof browser !== "undefined" ? browser : chrome;
const TOOLTIP_ID = "vocab-tooltip-host";
let tooltip = null;

document.addEventListener("mousedown", onDocMouseDown, true);
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") hideTooltip();
});

api.runtime.onMessage.addListener((msg) => {
    if (msg && msg.type === "show-translation") {
        showTooltip(msg);
    }
});

function onDocMouseDown(event) {
    if (tooltip && !tooltip.contains(event.target)) {
        hideTooltip();
    }
}

function showTooltip({ word, sentence, translation, alternatives, ipa, error }) {
    hideTooltip();

    const sel = window.getSelection();
    const range = sel && sel.rangeCount ? sel.getRangeAt(0) : null;
    const rect = range ? range.getBoundingClientRect() : null;

    tooltip = document.createElement("div");
    tooltip.id = TOOLTIP_ID;
    Object.assign(tooltip.style, {
        position: "fixed",
        top: rect ? `${Math.min(window.innerHeight - 160, rect.bottom + 6)}px` : "20px",
        left: rect ? `${Math.min(window.innerWidth - 320, Math.max(8, rect.left))}px` : "20px",
        background: "white",
        border: "1px solid #d4d4d8",
        borderRadius: "6px",
        padding: "10px 12px",
        maxWidth: "320px",
        minWidth: "200px",
        fontFamily: "system-ui, sans-serif",
        fontSize: "14px",
        lineHeight: "1.35",
        color: "#171717",
        boxShadow: "0 6px 16px rgba(0,0,0,0.18)",
        zIndex: "2147483647",
    });

    const wordEl = el("div", { fontWeight: "600", marginBottom: "4px" }, word);
    tooltip.appendChild(wordEl);

    if (error) {
        const errEl = el("div", { color: "#b91c1c", fontSize: "13px" }, `Fehler: ${error}`);
        tooltip.appendChild(errEl);
        document.body.appendChild(tooltip);
        return;
    }

    tooltip.appendChild(el("div", { color: "#525252" }, translation || "—"));
    if (alternatives) {
        tooltip.appendChild(
            el("div", { fontSize: "12px", color: "#737373", marginTop: "2px" }, alternatives)
        );
    }
    if (ipa) {
        tooltip.appendChild(
            el("code", { fontSize: "12px", color: "#737373", display: "block" }, ipa)
        );
    }

    const errEl = el("div", { color: "#b91c1c", fontSize: "12px", marginTop: "4px" }, "");
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "+ vocab";
    Object.assign(saveBtn.style, {
        marginTop: "8px",
        padding: "4px 10px",
        background: "#2563eb",
        color: "white",
        border: "0",
        borderRadius: "4px",
        cursor: "pointer",
        fontSize: "13px",
    });

    saveBtn.addEventListener("click", async () => {
        saveBtn.disabled = true;
        saveBtn.textContent = "…";
        try {
            await sendMessage({ type: "save", word, sentence, source: location.href });
            saveBtn.textContent = "✓ gespeichert";
            setTimeout(hideTooltip, 800);
        } catch (err) {
            errEl.textContent = `Fehler: ${err.message}`;
            saveBtn.disabled = false;
            saveBtn.textContent = "+ vocab";
        }
    });

    tooltip.append(errEl, saveBtn);
    document.body.appendChild(tooltip);
}

function hideTooltip() {
    if (tooltip) {
        tooltip.remove();
        tooltip = null;
    }
}

function el(tag, style, text) {
    const node = document.createElement(tag);
    Object.assign(node.style, style);
    if (text != null) node.textContent = text;
    return node;
}

function sendMessage(msg) {
    return new Promise((resolve, reject) => {
        api.runtime.sendMessage(msg, (response) => {
            const err = api.runtime.lastError;
            if (err) { reject(new Error(err.message)); return; }
            if (!response) { reject(new Error("no response")); return; }
            if (!response.ok) { reject(new Error(response.error || "request failed")); return; }
            resolve(response.data);
        });
    });
}
