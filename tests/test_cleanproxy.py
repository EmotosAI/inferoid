import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inferoid.cleanproxy import CleanProxy, clean


def test_clean_rewrites_only_what_the_spec_says():
    body = {"thinking": {"type": "adaptive"}, "max_tokens": 8192,
            "tools": [{"name": "Bash"}, {"name": "DeferredToolPlaceholder"}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>Codebase and user instructions are shown below ..."},
                {"type": "text", "text": "<system-reminder>Attribution for git commits ..."},
                {"type": "text", "text": "plan the transition"}]}]}
    counts = {}
    clean(body, counts)
    assert body["thinking"] == {"type": "disabled"} and body["max_tokens"] == 8192
    assert [t["name"] for t in body["tools"]] == ["Bash"]
    assert body["messages"][0]["content"] == [{"type": "text", "text": "plan the transition"}]
    assert counts == {"thinking_disabled": 1, "placeholder_removed": 1, "reminders_removed": 2}


def test_proxy_forwards_cleaned_body():
    seen = {}

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    proxy = CleanProxy(f"http://127.0.0.1:{up.server_address[1]}").start()
    try:
        req = urllib.request.Request(proxy.url + "/v1/messages", json.dumps({"messages": []}).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.loads(r.read()) == {"ok": True}
        assert seen["body"]["thinking"] == {"type": "disabled"}
        assert proxy.counts["thinking_disabled"] == 1
    finally:
        proxy.stop()
        up.shutdown()
