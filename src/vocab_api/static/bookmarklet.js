// Loaded into the bookmarklet href as a single-line javascript: URI by the
// /bookmarklet page. The BASE_URL placeholder is filled in server-side so the
// bookmarklet works from any origin without CORS gymnastics.
(function () {
    var sel = (window.getSelection && window.getSelection().toString()) || "";
    sel = sel.trim();
    if (!sel) {
        sel = window.prompt("Wort?") || "";
        sel = sel.trim();
    }
    if (!sel) return;
    var sentence = "";
    try {
        var node = window.getSelection().anchorNode;
        var container = node ? (node.parentElement || node) : document.body;
        var text = (container.textContent || "").replace(/\s+/g, " ").trim();
        var idx = text.indexOf(sel);
        if (idx >= 0) {
            var start = Math.max(0, idx - 80);
            var end = Math.min(text.length, idx + sel.length + 80);
            sentence = text.slice(start, end).trim();
        }
    } catch (e) { /* ignore */ }
    var params = new URLSearchParams({
        word: sel,
        sentence: sentence,
        source: location.href
    });
    window.open("__BASE_URL__/?" + params.toString(), "_blank", "noopener");
})();
