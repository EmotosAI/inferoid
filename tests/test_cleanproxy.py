import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inferoid.cleanproxy import CleanProxy, clean


def _sse(*chunks):
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"


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


def test_late_system_message_moves_to_top():
    body = {"system": [{"type": "text", "text": "You are Claude Code."}],
            "messages": [{"role": "user", "content": "plan it"}, {"role": "system", "content": "# Environment\ncwd: /x"}]}
    counts = {}
    clean(body, counts)
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["system"][-1]["text"].startswith("# Environment") and counts["system_moved"] == 1


def test_tool_name_case_is_fixed_in_replies():
    from inferoid.cleanproxy import fix_tool_names
    counts = {}
    chunk = b'event: content_block_start\ndata: {"type":"content_block_start","content_block":{"type":"tool_use","id":"t1","name":"bash","input":{}}}\n'
    out = fix_tool_names(chunk, {"bash": "Bash", "read": "Read"}, counts)
    assert b'"name":"Bash"' in out and counts["tool_name_case_fixed"] == 1
    assert fix_tool_names(b'"name":"Bash"', {"bash": "Bash"}, counts) == b'"name":"Bash"'


def test_openai_request_translation_and_non_stream_response():
    seen = {}

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = {"id": "chat-1", "choices": [{"message": {"content": "done", "tool_calls": []},
                                                    "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
            raw = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    proxy = CleanProxy(f"http://127.0.0.1:{up.server_address[1]}", mode="openai", model="engine-name").start()
    body = {"system": [{"type": "text", "text": "system"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "question"},
                                                               {"type": "image", "source": {}}]},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "call-1",
                                                               "name": "Bash", "input": {"command": "pwd"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1",
                                                          "is_error": True, "content": [{"type": "text", "text": "no"}]}]}],
            "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}},
                      {"name": "DeferredToolPlaceholder", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "Bash"}, "max_tokens": 20,
            "temperature": 0.2, "top_p": 0.9, "stop_sequences": ["END"], "thinking": {"type": "adaptive"},
            "metadata": {"private": True}, "stream": False}
    try:
        req = urllib.request.Request(proxy.url + "/v1/messages?beta=tools", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read())
        assert seen["path"] == "/v1/chat/completions"
        sent = seen["body"]
        assert sent["model"] == "engine-name" and sent["stream"] is False
        assert sent["messages"] == [
            {"role": "system", "content": "system"}, {"role": "user", "content": "question"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                                                                        "function": {"name": "Bash", "arguments": '{"command":"pwd"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "Error: no"},
        ]
        assert sent["tools"] == [{"type": "function", "function": {"name": "Bash", "description": "run",
                                                                          "parameters": {"type": "object"}}}]
        assert sent["tool_choice"] == {"type": "function", "function": {"name": "Bash"}}
        assert result["type"] == "message" and result["usage"] == {"input_tokens": 11, "output_tokens": 3}
    finally:
        proxy.stop()
        up.shutdown()


def test_openai_streaming_text_has_anthropic_event_sequence():
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = _sse({"id": "chat-2", "choices": [{"delta": {"role": "assistant", "content": "Hi"},
                                                         "finish_reason": None}]},
                       {"choices": [{"delta": {"content": " there"}, "finish_reason": None}]},
                       {"choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    proxy = CleanProxy(f"http://127.0.0.1:{up.server_address[1]}", mode="openai", model="m").start()
    try:
        request = urllib.request.Request(proxy.url + "/v1/messages", json.dumps({"messages": [], "stream": True}).encode(),
                                          {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            events = [json.loads(line[5:]) for line in response.read().decode().splitlines() if line.startswith("data:")]
        assert [event["type"] for event in events] == ["message_start", "content_block_start", "content_block_delta",
                                                         "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
        assert "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta") == "Hi there"
        assert events[-2]["delta"]["stop_reason"] == "end_turn"
        assert events[-2]["usage"] == {"output_tokens": 2, "input_tokens": 4}
    finally:
        proxy.stop()
        up.shutdown()


def test_openai_streaming_interleaved_tool_calls_and_stop_mapping():
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = _sse({"id": "chat-3", "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                       {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "type": "function",
                                                                    "function": {"name": "bash", "arguments": '{"x":'}}]},
                                          "finish_reason": None}]},
                       {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "b", "type": "function",
                                                                    "function": {"name": "Read", "arguments": '{"p":'}}]},
                                          "finish_reason": None}]},
                       {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}},
                                                               {"index": 1, "function": {"arguments": '"f"}'}}]},
                                          "finish_reason": "tool_calls"}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 5}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    proxy = CleanProxy(f"http://127.0.0.1:{up.server_address[1]}", mode="openai", model="m").start()
    try:
        request = urllib.request.Request(proxy.url + "/v1/messages", json.dumps({"messages": [], "stream": True,
                                                                                    "tools": [{"name": "Bash"}, {"name": "Read"}]}).encode(),
                                          {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            events = [json.loads(line[5:]) for line in response.read().decode().splitlines() if line.startswith("data:")]
        assert [event["type"] for event in events] == ["message_start", "content_block_start", "content_block_delta",
                                                         "content_block_start", "content_block_delta", "content_block_delta",
                                                         "content_block_delta",
                                                         "content_block_stop", "content_block_stop", "message_delta", "message_stop"]
        blocks = {0: [], 1: []}
        names = {}
        for event in events:
            if event["type"] == "content_block_start":
                block = event["content_block"]
                names[event["index"]] = block.get("name")
            elif event["type"] == "content_block_delta":
                blocks[event["index"]].append(event["delta"]["partial_json"])
        assert names == {0: "Bash", 1: "Read"}
        assert json.loads("".join(blocks[0])) == {"x": 1}
        assert json.loads("".join(blocks[1])) == {"p": "f"}
        assert events[-2]["delta"]["stop_reason"] == "tool_use"
    finally:
        proxy.stop()
        up.shutdown()


def test_openai_upstream_error_keeps_status_and_uses_anthropic_error_shape():
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = b'{"error":{"type":"invalid_request_error","message":"bad input"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    proxy = CleanProxy(f"http://127.0.0.1:{up.server_address[1]}", mode="openai", model="m").start()
    try:
        request = urllib.request.Request(proxy.url + "/v1/messages", b'{"messages":[]}', {"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            assert error.code == 400
            assert json.loads(error.read()) == {"type": "error", "error": {"type": "invalid_request_error", "message": "bad input"}}
        else:
            raise AssertionError("expected HTTPError")
    finally:
        proxy.stop()
        up.shutdown()
