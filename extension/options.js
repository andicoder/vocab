const api = typeof browser !== "undefined" ? browser : chrome;
const DEFAULT_URL = "https://vocab.example.com";

async function load() {
    const { apiBaseUrl, apiToken } = await api.storage.sync.get({
        apiBaseUrl: DEFAULT_URL,
        apiToken: ""
    });
    document.getElementById("apiBaseUrl").value = apiBaseUrl;
    document.getElementById("apiToken").value = apiToken;
}

async function save() {
    const apiBaseUrl = document.getElementById("apiBaseUrl").value.trim() || DEFAULT_URL;
    const apiToken = document.getElementById("apiToken").value.trim();
    await api.storage.sync.set({ apiBaseUrl, apiToken });
    const status = document.getElementById("status");
    status.textContent = "Gespeichert.";
    setTimeout(() => { status.textContent = ""; }, 1500);
}

document.addEventListener("DOMContentLoaded", load);
document.getElementById("save").addEventListener("click", save);
