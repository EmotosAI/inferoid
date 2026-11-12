"""Minimal newline-delimited JSON-RPC MCP server for supervisor agents."""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from . import dispatch

SERVER_INFO = {"name": "inferoid", "version": "0.2.0"}
TOOLS = [
    {"name": "local_agent",
     "description": ("Delegate a multi-step local coding chore. Give one concrete goal, the exact files to focus on, "
                      "a shell check that proves success, and review the diff afterwards. Never delegate secrets."),
     "inputSchema": {"type": "object", "properties": {
         "task": {"type": "string", "description": "One concrete goal for the local model."},
         "check": {"type": "string", "description": "Exact shell command that proves the goal is complete."},
         "files": {"type": "array", "items": {"type": "string"}, "description": "Exact files to focus on."},
         "timeout_s": {"type": "integer", "default": 900},
         "model": {"type": "string"}}, "required": ["task"], "additionalProperties": False}},
    {"name": "local_status", "description": "Show the local model, safe context, and engine reachability.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def _response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _text_result(value: Any, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": json.dumps(value, sort_keys=True)}]}
    if is_error:
        result["isError"] = True
    return result


def handle(request: dict[str, Any], cwd: Path | None = None,
           runner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any] | None:
    if request.get("jsonrpc") != "2.0" or "method" not in request:
        return _error(request.get("id"), -32600, "Invalid JSON-RPC request")
    request_id = request.get("id")
    method = request["method"]
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return _response(request_id, {})
    if method == "initialize":
        return _response(request_id, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                                      "serverInfo": SERVER_INFO})
    if method == "tools/list":
        return _response(request_id, {"tools": TOOLS})
    if method != "tools/call":
        return _error(request_id, -32601, f"Unknown method: {method}")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        return _error(request_id, -32602, "tools/call params must be an object")
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return _error(request_id, -32602, "tools/call arguments must be an object")
    if name == "local_status":
        return _response(request_id, _text_result(dispatch.local_status()))
    if name != "local_agent":
        return _error(request_id, -32602, f"Unknown tool: {name}")
    task = arguments.get("task")
    if not isinstance(task, str) or not task.strip():
        return _error(request_id, -32602, "local_agent requires a non-empty task")
    files = arguments.get("files") or []
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        return _error(request_id, -32602, "files must be a list of paths")
    timeout_s = arguments.get("timeout_s", 900)
    if not isinstance(timeout_s, int) or isinstance(timeout_s, bool) or timeout_s <= 0:
        return _error(request_id, -32602, "timeout_s must be a positive integer")
    check = arguments.get("check")
    if check is not None and not isinstance(check, str):
        return _error(request_id, -32602, "check must be a shell command string")
    model = arguments.get("model")
    if model is not None and not isinstance(model, str):
        return _error(request_id, -32602, "model must be a string")
    runner = runner or dispatch.run_task
    try:
        value = runner(project=cwd or Path.cwd(), task=task, check=check, files=files,
                       timeout=timeout_s, model=model)
    except Exception as exc:  # noqa: BLE001  # MCP must return a protocol response even when setup fails.
        value = {"status": "error", "applied": False, "notes": [str(exc)]}
    return _response(request_id, _text_result(value, value.get("status") != "done"))


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout,
          cwd: Path | None = None, runner: Callable[..., dict[str, Any]] | None = None) -> None:
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = handle(request, cwd=cwd, runner=runner)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            response = _error(None, -32700, f"Invalid JSON: {exc}")
        if response is not None:
            stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            stdout.flush()
