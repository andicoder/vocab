const api = typeof browser !== "undefined" ? browser : chrome;
const DEFAULT_URL = "https://vocab.example.com";

// ── PKCE helpers ──────────────────────────────────────────────────────────────

function generateCodeVerifier() {
    const bytes = new Uint8Array(32);
    crypto.getRandomValues(bytes);
    return base64urlEncode(bytes);
}

async function generateCodeChallenge(verifier) {
    const data = new TextEncoder().encode(verifier);
    const digest = await crypto.subtle.digest("SHA-256", data);
    return base64urlEncode(new Uint8Array(digest));
}

function base64urlEncode(bytes) {
    return btoa(String.fromCharCode(...bytes))
        .replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
}

// ── OIDC discovery ────────────────────────────────────────────────────────────

async function fetchOidcEndpoints(issuer) {
    const discovery = issuer.replace(/\/$/, "") + "/.well-known/openid-configuration";
    const res = await fetch(discovery);
    if (!res.ok) throw new Error(`OIDC discovery failed: ${res.status}`);
    return res.json();
}

// ── Login flow ────────────────────────────────────────────────────────────────

async function login() {
    const oidcIssuer = document.getElementById("oidcIssuer").value.trim();
    const oidcClientId = document.getElementById("oidcClientId").value.trim();
    if (!oidcIssuer || !oidcClientId) {
        showAuthStatus("Enter Issuer URL and Client ID first", false);
        return;
    }

    try {
        const endpoints = await fetchOidcEndpoints(oidcIssuer);
        const redirectUrl = api.identity.getRedirectURL();
        const verifier = generateCodeVerifier();
        const challenge = await generateCodeChallenge(verifier);

        const params = new URLSearchParams({
            response_type: "code",
            client_id: oidcClientId,
            redirect_uri: redirectUrl,
            scope: "openid profile",
            code_challenge: challenge,
            code_challenge_method: "S256",
            state: crypto.randomUUID(),
        });

        const authUrl = endpoints.authorization_endpoint + "?" + params.toString();
        const resultUrl = await api.identity.launchWebAuthFlow({ url: authUrl, interactive: true });
        const code = new URL(resultUrl).searchParams.get("code");
        if (!code) throw new Error("No code in redirect");

        const tokenRes = await fetch(endpoints.token_endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({
                grant_type: "authorization_code",
                client_id: oidcClientId,
                code,
                redirect_uri: redirectUrl,
                code_verifier: verifier,
            }),
        });
        if (!tokenRes.ok) throw new Error(`Token exchange failed: ${tokenRes.status}`);

        const tokens = await tokenRes.json();
        const expiresAt = Date.now() + tokens.expires_in * 1000;
        const username = parseJwtPayload(tokens.access_token)?.preferred_username ?? "unknown";

        await api.storage.sync.set({
            accessToken: tokens.access_token,
            refreshToken: tokens.refresh_token ?? "",
            expiresAt,
            oidcTokenEndpoint: endpoints.token_endpoint,
            oidcLoggedInUser: username,
        });

        showAuthStatus(`Logged in as ${username}`, true);
    } catch (err) {
        showAuthStatus(`Login failed: ${err.message}`, false);
    }
}

async function logout() {
    await api.storage.sync.remove(["accessToken", "refreshToken", "expiresAt",
        "oidcTokenEndpoint", "oidcLoggedInUser"]);
    showAuthStatus("Not logged in", false);
}

// ── JWT payload decode (no verification — Authentik validates on the server) ──

function parseJwtPayload(token) {
    try {
        const payload = token.split(".")[1];
        return JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
    } catch {
        return null;
    }
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function showAuthStatus(message, loggedIn) {
    const el = document.getElementById("authStatus");
    el.textContent = message;
    el.className = "auth-status " + (loggedIn ? "logged-in" : "logged-out");
}

// ── Settings persistence ──────────────────────────────────────────────────────

async function load() {
    const stored = await api.storage.sync.get({
        apiBaseUrl: DEFAULT_URL,
        oidcIssuer: "",
        oidcClientId: "",
        oidcLoggedInUser: "",
        accessToken: "",
        expiresAt: 0,
    });
    document.getElementById("apiBaseUrl").value = stored.apiBaseUrl;
    document.getElementById("oidcIssuer").value = stored.oidcIssuer;
    document.getElementById("oidcClientId").value = stored.oidcClientId;

    const redirectUrl = api.identity.getRedirectURL();
    document.getElementById("redirectUrl").textContent = redirectUrl;

    if (stored.accessToken && stored.expiresAt > Date.now()) {
        showAuthStatus(`Logged in as ${stored.oidcLoggedInUser || "unknown"}`, true);
    } else {
        showAuthStatus("Not logged in", false);
    }
}

async function save() {
    const apiBaseUrl = document.getElementById("apiBaseUrl").value.trim() || DEFAULT_URL;
    const oidcIssuer = document.getElementById("oidcIssuer").value.trim();
    const oidcClientId = document.getElementById("oidcClientId").value.trim();
    await api.storage.sync.set({ apiBaseUrl, oidcIssuer, oidcClientId });
    const status = document.getElementById("status");
    status.textContent = "Saved.";
    setTimeout(() => { status.textContent = ""; }, 1500);
}

document.addEventListener("DOMContentLoaded", load);
document.getElementById("save").addEventListener("click", save);
document.getElementById("loginBtn").addEventListener("click", login);
document.getElementById("logoutBtn").addEventListener("click", logout);
