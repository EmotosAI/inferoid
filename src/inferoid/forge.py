"""Metrics-only Forge transport and local Forge configuration."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import __version__, claude

DEFAULT_URL = "https://app.inferoid.com/forge"


def config_path() -> Path:
    return claude.state_dir().parent / "config.json"


def read_config() -> dict[str, Any]:
    try:
        value = json.loads(config_path().read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_config(config: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    tmp.replace(path)
    path.chmod(0o600)


def validate_url(url: str) -> str:
    url = url.rstrip("/")
    parsed = urllib.parse.urlparse(url)
    local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
    if parsed.scheme != "https" and not local_http:
        raise ValueError("Forge URL must use HTTPS (HTTP is allowed only for localhost testing).")
    if not parsed.netloc:
        raise ValueError("Forge URL must include a host.")
    return url


def login(token: str, url: str | None, share: bool | None = None) -> str:
    if not token.strip():
        raise ValueError("Forge token cannot be empty.")
    target = validate_url(url or DEFAULT_URL)
    if share is None:
        try:
            answer = input("Share metrics-only bench reports with Forge? (y/N) ").strip().lower()
        except EOFError:
            answer = ""
        share = answer.startswith("y")
    config = read_config()
    config.update({"forge_token": token, "forge_url": target, "share_reports": bool(share)})
    write_config(config)
    return f"Forge login saved for {target}. Metrics sharing is {'on' if share else 'off'}."


def logout() -> str:
    config = read_config()
    config.pop("forge_token", None)
    write_config(config)
    return "Forge token removed."


def _health(url: str) -> str:
    try:
        url = validate_url(url)
        with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=10) as response:
            response.read()
        return "ok"
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        return f"unavailable ({type(exc).__name__})"


def status() -> list[str]:
    config = read_config()
    url = str(config.get("forge_url") or DEFAULT_URL)
    token_set = bool(config.get("forge_token"))
    sharing = bool(config.get("share_reports", False))
    return [f"Forge URL: {url}", f"Forge token: {'set' if token_set else 'not set'}",
            f"Sharing: {'on' if sharing else 'off'}", f"Health: {_health(url)}"]


def allowlisted_report(data: dict[str, Any]) -> dict[str, Any]:
    """Copy only schema fields into a transport object.

    This is deliberately verbose. A report can be read from a file that was edited by
    another program, so passing its dictionary through would weaken the privacy boundary.
    """
    machine = data.get("machine") if isinstance(data.get("machine"), dict) else {}
    engine = data.get("engine") if isinstance(data.get("engine"), dict) else {}
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    task = data.get("task") if isinstance(data.get("task"), dict) else {}
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    result = data.get("result") if isinstance(data.get("result"), dict) else {}

    def text_value(obj: dict[str, Any], key: str, default: str = "") -> str:
        value = obj.get(key, default)
        return value if isinstance(value, str) else default

    def number_value(obj: dict[str, Any], key: str, default: float = 0) -> int | float:
        value = obj.get(key, default)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default

    clean: dict[str, Any] = {
        "schema": 1,
        "inferoid_version": text_value(data, "inferoid_version", __version__),
        "submitted_at": text_value(data, "submitted_at"),
        "machine": {
            "chip": text_value(machine, "chip", "unknown"),
            "ram_gb": number_value(machine, "ram_gb"),
            "gpu_budget_gb": number_value(machine, "gpu_budget_gb"),
            "os": text_value(machine, "os", "unknown"),
        },
        "engine": {"name": text_value(engine, "name", "ollama"),
                   "version": text_value(engine, "version", "unknown")},
        "model": {"name": text_value(model, "name"), "size_gb": number_value(model, "size_gb"),
                  "family": text_value(model, "family"), "quant": text_value(model, "quant")},
        "task": {"pack": text_value(task, "pack"), "id": text_value(task, "id"),
                 "kind": text_value(task, "kind", "coding")},
        "settings": {"context_cap": int(number_value(settings, "context_cap")),
                     "lean": bool(settings.get("lean", True)),
                     "effort": text_value(settings, "effort", "low"),
                     "timeout_s": int(number_value(settings, "timeout_s"))},
        "result": {
            "status": text_value(result, "status", "error"),
            "check_passed": bool(result.get("check_passed", False)),
            "failure_kind": text_value(result, "failure_kind", "engine_error"),
            "wall_s": number_value(result, "wall_s"),
            "first_reply_s": number_value(result, "first_reply_s"),
            "turns": int(number_value(result, "turns")),
            "tool_calls": int(number_value(result, "tool_calls")),
            "input_tokens": int(number_value(result, "input_tokens")),
            "output_tokens": int(number_value(result, "output_tokens")),
            "peak_model_mem_gb": number_value(result, "peak_model_mem_gb"),
            "peak_swap_gb": number_value(result, "peak_swap_gb"),
            "memory_pressure": text_value(result, "memory_pressure", "normal"),
        },
    }
    reference = data.get("reference")
    if isinstance(reference, dict):
        clean["reference"] = {"model": text_value(reference, "model"),
                               "status": text_value(reference, "status"),
                               "wall_s": number_value(reference, "wall_s"),
                               "tool_calls": int(number_value(reference, "tool_calls"))}
    verdict = data.get("verdict")
    if isinstance(verdict, dict):
        clean["verdict"] = {key: verdict[key] for key in ("kind", "suggested_next")
                             if isinstance(verdict.get(key), str)}
    return clean


def submit(data: dict[str, Any], token: str | None = None, url: str | None = None) -> tuple[bool, str, dict[str, Any] | None]:
    config = read_config()
    token = token or str(config.get("forge_token") or "")
    try:
        target = validate_url(url or str(config.get("forge_url") or DEFAULT_URL))
    except ValueError as exc:
        return False, f"Forge submission failed ({exc}).", None
    if not token:
        return False, "Forge submission skipped: no token is set.", None
    body = json.dumps(allowlisted_report(data), ensure_ascii=False).encode()
    request = urllib.request.Request(target + "/api/v1/reports", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            if response.status != 201:
                return False, f"Forge returned HTTP {response.status}.", None
            payload = json.loads(raw) if raw else {}
            verdict = payload.get("verdict") if isinstance(payload, dict) else None
            return True, "Forge accepted the report.", verdict if isinstance(verdict, dict) else None
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "Forge rejected the token (HTTP 401).", None
        return False, f"Forge returned HTTP {exc.code}.", None
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return False, f"Forge submission failed ({type(exc).__name__}).", None
