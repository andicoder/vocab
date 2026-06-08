const api = typeof browser !== "undefined" ? browser : chrome;
const DEFAULT_URL = "https://vocab.example.com";

async function load() {
    const { apiBaseUrl } = await api.storage.sync.get({ apiBaseUrl: DEFAULT_URL });
    document.getElementById("apiBaseUrl").value = apiBaseUrl;
}

async function save() {
    const apiBaseUrl = document.getElementById("apiBaseUrl").value.trim() || DEFAULT_URL;
    await api.storage.sync.set({ apiBaseUrl });
    const status = document.getElementById("status");
    status.textContent = "Gespeichert.";
    setTimeout(() => { status.textContent = ""; }, 1500);
}

document.addEventListener("DOMContentLoaded", load);
document.getElementById("save").addEventListener("click", save);
