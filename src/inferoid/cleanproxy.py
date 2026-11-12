"""Clean mode: a small local proxy between Claude Code and Ollama that removes what confuses local models.

Measured on a 24 GB M4 (djmix bench, 2026-11-12; spec from the bench's PROXY-SPEC.md):
- thinking -> {"type": "disabled"}. Claude Code sends adaptive thinking (or no field), and gemma4 then never closes its
  thought channel (a full max_tokens of invisible output). Replays: 3/3 runaway before, 3/3 clean after.
- the DeferredToolPlaceholder tool is removed (local models call it; LFM2.5 made 8 invalid calls).
- <system-reminder> blocks carrying the user's global CLAUDE.md or git-attribution text are removed. They are the user's
  instructions for Claude, not for the task, and they carry private context into a local session.
Everything except POST /v1/messages passes through unchanged. Only counts of changes are kept, never request bodies.
"""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request
from typing import Any

DROP_MARKERS = ("Codebase and user instructions are shown below", "Attribution for git commits")


def clean(body: dict[str, Any], counts: dict[str, int]) -> None:
    if body.get("thinking") != {"type": "disabled"}:
        body["thinking"] = {"type": "disabled"}
        counts["thinking_disabled"] = counts.get("thinking_disabled", 0) + 1
    tools = body.get("tools") or []
    kept = [t for t in tools if not (isinstance(t, dict) and t.get("name") == "DeferredToolPlaceholder")]
    if len(kept) != len(tools):
        body["tools"] = kept
        counts["placeholder_removed"] = counts.get("placeholder_removed", 0) + 1
    # Claude Code sends its "# Environment" note as a role "system" message after the first user turn. Some chat
    # templates (Qwen3.5/3.6) raise "System message must be at the beginning" and every request fails with HTTP 500.
    # Move such messages into the top-level system prompt.
    messages = body.get("messages") or []
    moved = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    if moved:
        texts = []
        for m in moved:
            content = m.get("content")
            texts.append(content if isinstance(content, str) else "\n".join(
                c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text"))
        system = body.get("system")
        extra = [{"type": "text", "text": t} for t in texts if t]
        if isinstance(system, list):
            body["system"] = system + extra
        elif isinstance(system, str) and system:
            body["system"] = [{"type": "text", "text": system}] + extra
        else:
            body["system"] = extra
        body["messages"] = [m for m in messages if m not in moved]
        counts["system_moved"] = counts.get("system_moved", 0) + len(moved)
    for message in body.get("messages") or []:
        if not (isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), list)):
            continue
        before = len(message["content"])
        message["content"] = [c for c in message["content"] if not (
            isinstance(c, dict) and c.get("type") == "text" and str(c.get("text", "")).startswith("<system-reminder>")
            and any(marker in c["text"] for marker in DROP_MARKERS))]
        removed = before - len(message["content"])
        if removed:
            counts["reminders_removed"] = counts.get("reminders_removed", 0) + removed
        if not message["content"]:
            message["content"] = [{"type": "text", "text": "(no content)"}]


class CleanProxy:
    """Start with .start(); point ANTHROPIC_BASE_URL at .url; .counts holds what was changed."""

    def __init__(self, upstream: str):
        self.upstream = upstream.rstrip("/")
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _forward(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                if body and method == "POST" and self.path.startswith("/v1/messages"):
                    try:
                        data = json.loads(body)
                        with proxy.lock:
                            clean(data, proxy.counts)
                        body = json.dumps(data).encode()
                    except ValueError:
                        pass
                headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
                request = urllib.request.Request(proxy.upstream + self.path, body, method=method, headers=headers)
                try:
                    response = urllib.request.urlopen(request, timeout=3600)
                except urllib.error.HTTPError as exc:
                    response = exc
                except OSError as exc:
                    message = json.dumps({"error": f"Ollama is not reachable: {exc}"}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(message)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(message)
                    return
                self.send_response(getattr(response, "status", None) or response.code)
                for key, value in response.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                self.close_connection = True

            def do_POST(self) -> None:  # noqa: N802
                self._forward("POST")

            def do_GET(self) -> None:  # noqa: N802
                self._forward("GET")

            def do_HEAD(self) -> None:  # noqa: N802
                self._forward("HEAD")

            def do_DELETE(self) -> None:  # noqa: N802
                self._forward("DELETE")

            def log_message(self, *args: Any) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread: threading.Thread | None = None

    def start(self) -> "CleanProxy":
        self.thread = threading.Thread(target=self.server.serve_forever, name="inferoid-clean-proxy", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
