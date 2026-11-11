from __future__ import annotations

import collections
import json
import platform
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .storage import (
    CLASSES,
    EDIT_TOOLS,
    MEMORY_PATH,
    READ_ONLY_SHELL_WORDS,
    READ_SEARCH_TOOLS,
    SHELL_TOOLS,
    STEP_LOCAL_CLASSES,
    connect,
    load_json,
    safe_json,
    tier_for_ram,
    tier_spec,
    turn_calls,
)

DOCS_INTENT = re.compile(
    r"\b(?:documentation|readme|changelog|release\s+notes?|"
    r"(?:update|write|add|fix|improve|review|draft|generate|document)\s+(?:the\s+)?docs?(?!\s+folder\b))\b"
    r"|\bdocs?\s+(?:page|section|content|example|api|reference)\b"
)


def json_facts(call) -> list[dict[str, Any]]:
    try:
        facts = json.loads(call["tool_facts"] or "[]")
        if isinstance(facts, list) and facts:
            return facts
    except (TypeError, json.JSONDecodeError):
        pass
    try:
        return [{"name": name} for name in json.loads(call["tool_names"] or "[]")]
    except (TypeError, json.JSONDecodeError):
        return []


def classify_step(call) -> str:
    """Classify one model output using only normalized tool facts and output length."""
    facts = json_facts(call)
    names = [str(fact.get("name", "")).lower().replace("-", "_") for fact in facts]
    if any(name not in READ_SEARCH_TOOLS and (name.startswith("web") or name in {"browser", "fetch", "imageview", "image_view"} or "search" in name or "fetch" in name) for name in names):
        return "web/search tool"
    if any(name.startswith("task") or name in {"agent", "subagent", "delegate", "spawn"} or "subagent" in name for name in names):
        return "subagent/task dispatch"
    if facts:
        if all(name in READ_SEARCH_TOOLS for name in names):
            return "read/search dispatch"
        shell_facts = [
            fact for fact, name in zip(facts, names)
            if (name in SHELL_TOOLS or "command" in name or "shell" in name)
            and not (name == "exec" and not fact.get("command_first_words"))
        ]
        edit_facts = [fact for fact, name in zip(facts, names) if name in EDIT_TOOLS or "edit" in name or "patch" in name or "filechange" in name]
        if shell_facts and not edit_facts:
            words = [
                word.lower()
                for fact in shell_facts
                for word in fact.get("command_first_words", [fact.get("command_first_word", "")])
            ]
            if words and all(
                word in READ_ONLY_SHELL_WORDS and not fact.get("shell_redirect")
                for fact in shell_facts
                for word in fact.get("command_first_words", [fact.get("command_first_word", "")])
            ):
                return "read-only shell dispatch"
            if words:
                return "mutating shell dispatch"
            return "mutating shell dispatch"
        if edit_facts:
            changed = sum(int(fact.get("added_lines", 0) or 0) + int(fact.get("removed_lines", 0) or 0) for fact in edit_facts)
            file_count = sum(int(fact.get("file_count", 1) or 1) for fact in edit_facts)
            if len(edit_facts) == 1 and file_count <= 1 and changed <= 30:
                return "small edit"
            return "large edit"
        return "unknown tool"
    text_length = int(call["final_text_chars"] or len(call["response_text"] or ""))
    if text_length == 0:
        return "reasoning only"
    legacy_single_call = not call["final_text_chars"] and not json_facts(call) and call["request_index"] == 0
    if (call["is_last"] or legacy_single_call) and text_length <= 1_500:
        return "short text"
    if text_length <= 1_500:
        return "intermediate text"
    if text_length > 1_500:
        return "long text / planning"
    return "other / unknown"


def update_step_classes(conn) -> collections.Counter:
    counts = collections.Counter()
    for call in conn.execute("SELECT * FROM model_calls"):
        step_class = classify_step(call)
        conn.execute("UPDATE model_calls SET step_class=? WHERE id=?", (step_class, call["id"]))
        counts[step_class] += 1
    conn.commit()
    return counts


def step_is_local_capable(call, tier: str) -> bool:
    return call["step_class"] in STEP_LOCAL_CLASSES and int(call["context_tokens"] or 0) <= int(tier_spec(tier)["max_context_tokens"])


def is_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def validate_local_endpoint(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if not is_local_endpoint(normalized):
        raise ValueError("endpoint must use localhost, 127.0.0.1, or ::1")
    return normalized


def classify_heuristic(turn, calls) -> tuple[str, str, str]:
    text = (turn["text"] or "").strip().lower()
    tools = [name.lower() for call in calls for name in json.loads(call["tool_names"] or "[]")]
    tool_count = sum(call["tool_call_count"] for call in calls)
    tokens = sum(call["input_tokens"] + call["output_tokens"] for call in calls)
    if not text:
        category = "unclassified"
    elif DOCS_INTENT.search(text) or re.search(r"\bcommit(?:\s+message)?\b", text):
        category = "docs/commit message"
    elif re.search(r"\b(traceback|error|failing|failed|failure|bug|debug|broken|regression)\b", text) and (len(calls) > 1 or tool_count > 1):
        category = "debugging loop"
    elif re.search(r"\b(test|tests|pytest|jest|spec|coverage)\b", text) or any("test" in tool for tool in tools):
        category = "tests"
    elif re.search(r"\b(docker|kubernetes|k8s|terraform|deploy|shell|bash|terminal|infra|ci/cd)\b", text) or any("command" in tool or "shell" in tool for tool in tools):
        category = "shell/infra"
    elif re.search(r"\b(refactor|rename|move .*files?|multi[- ]file|architecture|migrate)\b", text) or tool_count > 5:
        category = "refactor/multi-file"
    elif re.search(r"\b(plan|planning|design|approach|steps|roadmap)\b", text):
        category = "planning"
    elif re.search(r"\b(explain|what is|how does|why does|question|difference between|help me understand)\b", text) and (tool_count == 0 or set(tools) <= {"read", "grep", "glob", "search"}):
        category = "explain/question"
    elif re.search(r"\b(search|find|read|inspect|look at|where is|locate)\b", text) or any(name in {"read", "grep", "glob", "search", "websearch"} for name in tools):
        category = "search/read code"
    else:
        category = "small edit"
    size = "small" if tool_count <= 1 and len(calls) <= 1 and tokens < 4_000 else "medium" if tool_count <= 5 and len(calls) <= 4 and tokens < 20_000 else "large"
    return category, size, "heuristic"


def _chat_request(model: str, messages: list[dict[str, Any]], endpoint: str | None = None) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    payload = {"model": model, "stream": False, "messages": messages}
    if endpoint:
        base = validate_local_endpoint(endpoint)
        url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    else:
        url = "http://127.0.0.1:11434/api/chat"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not reach Ollama on localhost: {exc}") from exc
    choices = result.get("choices") or []
    openai_message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    text = str(result.get("message", {}).get("content") or openai_message.get("content") or result.get("response") or "")
    duration_ns = sum(
        float(result.get(key, 0) or 0) for key in ("prompt_eval_duration", "eval_duration")
    )
    tool_calls = openai_message.get("tool_calls") or result.get("message", {}).get("tool_calls") or []
    return text, {
        "wall_time_ms": duration_ns / 1_000_000 if duration_ns else (time.perf_counter() - started) * 1000,
        "tool_calls": tool_calls,
    }


def local_chat(model: str, prompt: str | list[dict[str, Any]], system: str = "", endpoint: str | None = None) -> tuple[str, dict[str, Any]]:
    messages = ([{"role": "system", "content": system}] if system else []) + (prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}])
    return _chat_request(model, messages, endpoint)


def local_chat_messages(model: str, messages: list[dict[str, Any]], endpoint: str | None = None) -> tuple[str, dict[str, Any]]:
    return _chat_request(model, messages, endpoint)


def ollama_chat(model: str, prompt: str, system: str = "", endpoint: str | None = None) -> str:
    return local_chat(model, prompt, system, endpoint)[0]


def classify(local_model: str | None = None, endpoint: str | None = None) -> dict[str, int]:
    if endpoint:
        validate_local_endpoint(endpoint)
    conn = connect()
    counts = collections.Counter()
    for turn in conn.execute("SELECT * FROM user_turns ORDER BY timestamp,id"):
        calls = turn_calls(conn, turn["id"])
        category, size, method = classify_heuristic(turn, calls)
        if local_model and category != "unclassified":
            prompt = safe_json({"user_turn": turn["text"], "tool_names": [name for call in calls for name in json.loads(call["tool_names"] or "[]")], "call_count": len(calls), "token_count": sum(call["input_tokens"] + call["output_tokens"] for call in calls)})
            raw = ollama_chat(local_model, prompt, "Return JSON only with keys task_class and size. task_class must be one of: " + ", ".join(CLASSES), endpoint)
            try:
                parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                if parsed.get("task_class") in CLASSES:
                    category = parsed["task_class"]
                if parsed.get("size") in {"small", "medium", "large"}:
                    size = parsed["size"]
                method = "local:" + local_model
            except (ValueError, TypeError, json.JSONDecodeError):
                method = "heuristic (local response invalid)"
        conn.execute("UPDATE user_turns SET task_class=?,class_method=?,size=?,tool_calls=?,total_tokens=? WHERE id=?", (category, method, size, sum(call["tool_call_count"] for call in calls), sum(call["input_tokens"] + call["output_tokens"] for call in calls), turn["id"]))
        counts[category] += 1
    update_step_classes(conn)
    conn.commit()
    conn.close()
    return dict(counts)


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_machine() -> dict[str, Any]:
    if platform.system() == "Darwin":
        chip = command("sysctl", "-n", "machdep.cpu.brand_string") or command("sysctl", "-n", "hw.model") or "unknown"
        raw_ram = command("sysctl", "-n", "hw.memsize")
        ram_gb = round(int(raw_ram) / 1024**3) if raw_ram.isdigit() else 0
        version = command("sw_vers", "-productVersion") or platform.mac_ver()[0]
    else:
        chip = platform.processor() or command("uname", "-m") or "unknown"
        ram_gb = 0
        try:
            with open("/proc/meminfo", encoding="utf-8") as meminfo:
                lines = meminfo.readlines()
            for line in lines:
                if line.startswith("MemTotal:"):
                    ram_gb = round(int(line.split()[1]) * 1024 / 1024**3)
                    break
        except (OSError, ValueError):
            pass
        version = platform.release()
    ollama_models: list[str] = []
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            ollama_models = [str(item.get("name")) for item in json.loads(response.read().decode()).get("models", []) if item.get("name")]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass
    lm_models: list[str] = []
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2) as response:
            lm_models = [str(item.get("id")) for item in json.loads(response.read().decode()).get("data", []) if item.get("id")]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass
    table = load_json(MEMORY_PATH, {"classes": []})
    tier = tier_for_ram(ram_gb)
    tier_info = table.get("tiers", {}).get(str(tier), tier_spec(tier))
    fit = [row for row in table.get("classes", []) if ram_gb >= int(row.get("ram_gb", 999))]
    recommended = fit[-1]["class"] if fit else (table.get("classes", [{}])[0].get("class", "7-8B") if table.get("classes") else "7-8B")
    suitable = fit[-1].get("suitable_task_classes", []) if fit else []
    return {"os": platform.system(), "os_version": version or "unknown", "chip": chip, "ram_gb": ram_gb, "tier_name": tier_info.get("ram_label", "unknown"), "max_context_tokens": int(tier_info.get("max_context_tokens", 0) or 0), "ollama_models": ollama_models, "lm_studio_models": lm_models, "mlx_detected": bool(shutil.which("mlx_lm.generate") or shutil.which("mlx_lm")), "recommended_class": recommended, "suitable_task_classes": suitable, "memory_table_date": table.get("as_of")}


def machine() -> dict[str, Any]:
    result = detect_machine()
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
