// Cross-browser API alias (Firefox exposes `browser`, Chromium uses `chrome`).
const api = typeof browser !== "undefined" ? browser : chrome;

const DEFAULTS = {
    apiBaseUrl: "https://vocab.example.com",
    oidcClientId: "",
};

const MENU_SAVE_ID = "vocab-save-selection";
const MENU_TRANSLATE_ID = "vocab-translate-selection";

async function getConfig() {
    const stored = await api.storage.sync.get(DEFAULTS);
    return { ...DEFAULTS, ...stored };
}

// ── Token management ──────────────────────────────────────────────────────────

async function getAccessToken() {
    const { accessToken, refreshToken, expiresAt, oidcTokenEndpoint, oidcClientId } =
        await api.storage.sync.get([
            "accessToken", "refreshToken", "expiresAt", "oidcTokenEndpoint", "oidcClientId"
        ]);

    if (!accessToken) return null;

    // Refresh if within 60 s of expiry.
    if (expiresAt && Date.now() < expiresAt - 60_000) return accessToken;
    if (!refreshToken || !oidcTokenEndpoint) return null;

    try {
        const res = await fetch(oidcTokenEndpoint, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({
                grant_type: "refresh_token",
                client_id: oidcClientId,
                refresh_token: refreshToken,
            }),
        });
        if (!res.ok) throw new Error(`refresh ${res.status}`);
        const tokens = await res.json();
        const newExpiresAt = Date.now() + tokens.expires_in * 1000;
        await api.storage.sync.set({
            accessToken: tokens.access_token,
            refreshToken: tokens.refresh_token || refreshToken,
            expiresAt: newExpiresAt,
        });
        return tokens.access_token;
    } catch {
        // Refresh failed — clear tokens so user knows they need to re-login.
        await api.storage.sync.remove(["accessToken", "refreshToken", "expiresAt"]);
        return null;
    }
}

function authHeaders(accessToken) {
    const headers = { "Content-Type": "application/json" };
    if (accessToken) headers["Authorization"] = `Bearer ${accessToken}`;
    return headers;
}

// ── Context menus ─────────────────────────────────────────────────────────────

api.runtime.onInstalled.addListener(() => {
    api.contextMenus.create({
        id: MENU_SAVE_ID,
        title: "vocab: Wort speichern",
        contexts: ["selection"]
    });
    api.contextMenus.create({
        id: MENU_TRANSLATE_ID,
        title: "vocab: Übersetzung anzeigen",
        contexts: ["selection"]
    });
});

api.contextMenus.onClicked.addListener(async (info, tab) => {
    const word = (info.selectionText || "").trim();
    if (!word) {
        await notify("Keine Auswahl");
        return;
    }
    const sentence = await extractSentence(tab, word);

    if (info.menuItemId === MENU_SAVE_ID) {
        const source = info.pageUrl || (tab && tab.url) || "";
        try {
            await postEntry({ word, sentence, source });
            await notify(`Gespeichert: ${word}`);
        } catch (err) {
            await notify(`Fehler: ${err.message}`);
        }
        return;
    }

    if (info.menuItemId === MENU_TRANSLATE_ID) {
        if (tab && tab.id != null) {
            try {
                const data = await postTranslate({ word, sentence });
                await api.tabs.sendMessage(tab.id, {
                    type: "show-translation",
                    word, sentence,
                    translation: data.translation,
                    alternatives: data.alternatives,
                    ipa: data.ipa
                });
            } catch (err) {
                await api.tabs.sendMessage(tab.id, {
                    type: "show-translation",
                    word, sentence,
                    error: err.message
                });
            }
        }
    }
});

// ── Sentence extraction ───────────────────────────────────────────────────────

async function extractSentence(tab, word) {
    if (!tab || tab.id == null) return "";
    try {
        const [{ result } = {}] = await api.scripting.executeScript({
            target: { tabId: tab.id },
            func: extractSentenceInPage,
            args: [word]
        });
        return result || "";
    } catch (e) {
        return "";
    }
}

function extractSentenceInPage(word) {
    const sel = window.getSelection && window.getSelection();
    if (!sel || sel.rangeCount === 0) return "";
    const node = sel.anchorNode;
    const container = node ? (node.parentElement || node) : document.body;
    const text = (container.textContent || "").replace(/\s+/g, " ").trim();
    const idx = text.indexOf(word);
    if (idx < 0) return "";
    const start = Math.max(0, idx - 80);
    const end = Math.min(text.length, idx + word.length + 80);
    return text.slice(start, end).trim();
}

// ── API calls ─────────────────────────────────────────────────────────────────

async function postEntry({ word, sentence, source }) {
    const { apiBaseUrl } = await getConfig();
    const token = await getAccessToken();
    const url = `${apiBaseUrl.replace(/\/$/, "")}/vocab`;
    const res = await fetch(url, {
        method: "POST",
        headers: authHeaders(token),
        body: JSON.stringify({ word, sentence: sentence || null, source: source || null })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

async function postTranslate({ word, sentence }) {
    const { apiBaseUrl } = await getConfig();
    const token = await getAccessToken();
    const url = `${apiBaseUrl.replace(/\/$/, "")}/translate`;
    const res = await fetch(url, {
        method: "POST",
        headers: authHeaders(token),
        body: JSON.stringify({ word, sentence: sentence || null })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

api.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type !== "save") return false;
    postEntry(msg)
        .then((data) => sendResponse({ ok: true, data }))
        .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
});

// ── Notifications ─────────────────────────────────────────────────────────────

async function notify(message) {
    if (!api.notifications) return;
    try {
        await api.notifications.create({
            type: "basic",
            iconUrl: api.runtime.getURL("icon-128.png"),
            title: "vocab",
            message
        });
    } catch (e) {
        // Notifications can fail (e.g. permissions revoked); swallow.
    }
}
