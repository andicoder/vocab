const CACHE = "vocab-shell-v1";
const SHELL = ["/", "/queue", "/static/manifest.webmanifest", "/static/icon.svg"];

self.addEventListener("install", (event) => {
    event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
    self.skipWaiting();
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
        )
    );
    self.clients.claim();
});

self.addEventListener("fetch", (event) => {
    const req = event.request;
    if (req.method !== "GET") return;
    const url = new URL(req.url);
    if (url.origin !== self.location.origin) return;

    // Network-first for HTML (so the queue stays fresh); cache fallback when offline.
    if (req.headers.get("accept")?.includes("text/html")) {
        event.respondWith(
            fetch(req)
                .then((res) => {
                    const copy = res.clone();
                    caches.open(CACHE).then((c) => c.put(req, copy));
                    return res;
                })
                .catch(() => caches.match(req).then((m) => m || caches.match("/")))
        );
        return;
    }

    // Cache-first for static assets.
    if (url.pathname.startsWith("/static/")) {
        event.respondWith(
            caches.match(req).then((m) => m || fetch(req).then((res) => {
                const copy = res.clone();
                caches.open(CACHE).then((c) => c.put(req, copy));
                return res;
            }))
        );
    }
});
