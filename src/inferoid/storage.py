from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any

DATA_ROOT = resources.files("inferoid.data")
PRICES_PATH = DATA_ROOT.joinpath("prices.json")
MEMORY_PATH = DATA_ROOT.joinpath("model-memory.json")
CLASSES = (
    "explain/question", "search/read code", "small edit", "tests",
    "refactor/multi-file", "debugging loop", "shell/infra",
    "docs/commit message", "planning", "unclassified",
)
LOCAL_HEURISTIC_CLASSES = {
    "explain/question", "search/read code", "small edit", "tests",
    "docs/commit message", "planning",
}
LOCAL_CAPABLE_SIZES = {"small", "medium"}
STEP_CLASSES = (
    "read/search dispatch", "read-only shell dispatch", "mutating shell dispatch",
    "small edit", "large edit", "short text", "long text / planning", "reasoning only",
    "intermediate text", "web/search tool", "subagent/task dispatch", "unknown tool", "other / unknown",
)
STEP_LOCAL_CLASSES = {
    "read/search dispatch", "read-only shell dispatch", "small edit", "short text",
}
READ_ONLY_SHELL_WORDS = {
    "cat", "cut", "date", "df", "du", "echo", "file", "find", "git branch", "git diff",
    "git log", "git show", "git status", "grep", "head", "jq", "ls", "pwd", "rg", "sed -n",
    "stat", "tail", "tree", "wc", "which",
}
READ_SEARCH_TOOLS = {
    "read", "grep", "glob", "ls", "search", "find", "listfiles",
}
SHELL_TOOLS = {
    "bash", "shell", "exec", "execute", "command", "commandexecution", "local_shell",
}
EDIT_TOOLS = {
    "edit", "write", "multiedit", "apply_patch", "applypatch", "filechange", "file_change",
}
LOCAL_MODEL_PRICE = {
    "label": "local model ($0)",
    "local": True,
    "input_per_million": 0.0,
    "output_per_million": 0.0,
    "cache_read_per_million": 0.0,
    "cache_write_per_million": 0.0,
}


def tier_spec(tier: str) -> dict[str, Any]:
    table = load_json(MEMORY_PATH, {})
    tiers = table.get("tiers", {})
    spec = tiers.get(str(tier))
    if isinstance(spec, dict) and spec.get("max_context_tokens"):
        return spec
    defaults = {
        "16": {"ram_label": "16-32 GB", "max_context_tokens": 16_000},
        "48": {"ram_label": "48-64 GB", "max_context_tokens": 32_000},
        "128": {"ram_label": "128 GB+", "max_context_tokens": 64_000},
    }
    return defaults[str(tier)]


def tier_for_ram(ram_gb: int) -> str:
    if ram_gb >= 128:
        return "128"
    if ram_gb >= 48:
        return "48"
    return "16"


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def parse_timestamp(value: Any) -> dt.datetime | None:
    if isinstance(value, (int, float)):
        try:
            return dt.datetime.fromtimestamp(value / 1000 if value > 10_000_000_000 else value, dt.UTC)
        except (ValueError, OverflowError, OSError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.UTC) if parsed.tzinfo is None else parsed.astimezone(dt.UTC)


def in_window(timestamp: Any, cutoff: dt.datetime) -> bool:
    parsed = parse_timestamp(timestamp)
    return parsed is not None and parsed >= cutoff


def text_from(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (text_from(item) for item in value)))
    if isinstance(value, dict):
        for key in ("text", "input_text", "output_text", "content"):
            if key in value:
                found = text_from(value[key])
                if found:
                    return found
    return ""


def content_text(content: Any, include_tool_results: bool = False) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return text_from(content)
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if not include_tool_results and item.get("type") in {"tool_result", "tool_output"}:
            continue
        if item.get("type") in {"text", "input_text", "output_text"} or "text" in item:
            found = text_from(item)
            if found:
                parts.append(found)
    return "\n".join(parts)


def project_hash(path: Any) -> str:
    return hashlib.sha256(str(path or "unknown").encode("utf-8", "replace")).hexdigest()[:20]


def coarse_key(value: Any) -> str:
    """Stable comparison key for a path, pattern, or other tool argument."""
    return hashlib.sha256(str(value or "").encode("utf-8", "replace")).hexdigest()[:20]


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def value_int(mapping: Any, *keys: str) -> int:
    if not isinstance(mapping, dict):
        return 0
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return 0


def db_home() -> Path:
    return Path(os.environ.get("INFEROID_HOME", Path.home() / ".inferoid")).expanduser()


def db_path() -> Path:
    return db_home() / "inferoid.db"


def connect() -> sqlite3.Connection:
    db_home().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS user_turns (
          id TEXT PRIMARY KEY, user TEXT NOT NULL, source TEXT NOT NULL, session_id TEXT NOT NULL,
          project_hash TEXT NOT NULL, timestamp TEXT, text TEXT NOT NULL, turn_index INTEGER NOT NULL,
          task_class TEXT, class_method TEXT, size TEXT, tool_calls INTEGER NOT NULL DEFAULT 0,
          total_tokens INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS model_calls (
          id TEXT PRIMARY KEY, user_turn_id TEXT NOT NULL REFERENCES user_turns(id), source TEXT NOT NULL,
          session_id TEXT NOT NULL, project_hash TEXT NOT NULL, timestamp TEXT, model TEXT NOT NULL,
          input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
          cache_read_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0,
          context_tokens INTEGER NOT NULL DEFAULT 0, tool_call_count INTEGER NOT NULL DEFAULT 0,
          tool_names TEXT NOT NULL DEFAULT '[]', tool_facts TEXT NOT NULL DEFAULT '[]',
          final_text_chars INTEGER NOT NULL DEFAULT 0, request_index INTEGER NOT NULL DEFAULT 0,
          is_first INTEGER NOT NULL DEFAULT 0, is_last INTEGER NOT NULL DEFAULT 0,
          is_subagent INTEGER NOT NULL DEFAULT 0, response_text TEXT NOT NULL DEFAULT '',
          step_class TEXT
        );
        CREATE TABLE IF NOT EXISTS replay_results (
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_turn_id TEXT NOT NULL, call_id TEXT,
          model TEXT NOT NULL, step_class TEXT, tier TEXT, context_tokens INTEGER NOT NULL DEFAULT 0,
          outcome TEXT NOT NULL, agreement INTEGER NOT NULL DEFAULT 0, score REAL, reason TEXT NOT NULL,
          created_at TEXT NOT NULL, wall_time_ms REAL, agreement_basis TEXT,
          FOREIGN KEY(user_turn_id) REFERENCES user_turns(id)
        );
        CREATE INDEX IF NOT EXISTS calls_turn_idx ON model_calls(user_turn_id);
        CREATE INDEX IF NOT EXISTS turns_class_idx ON user_turns(task_class);
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(replay_results)")}
    if "wall_time_ms" not in columns:
        conn.execute("ALTER TABLE replay_results ADD COLUMN wall_time_ms REAL")
    model_columns = {row[1] for row in conn.execute("PRAGMA table_info(model_calls)")}
    additions = {
        "context_tokens": "INTEGER NOT NULL DEFAULT 0",
        "tool_facts": "TEXT NOT NULL DEFAULT '[]'",
        "final_text_chars": "INTEGER NOT NULL DEFAULT 0",
        "request_index": "INTEGER NOT NULL DEFAULT 0",
        "is_first": "INTEGER NOT NULL DEFAULT 0",
        "is_last": "INTEGER NOT NULL DEFAULT 0",
        "step_class": "TEXT",
    }
    for name, definition in additions.items():
        if name not in model_columns:
            conn.execute(f"ALTER TABLE model_calls ADD COLUMN {name} {definition}")
    replay_columns = {row[1] for row in conn.execute("PRAGMA table_info(replay_results)")}
    replay_additions = {
        "call_id": "TEXT", "step_class": "TEXT", "tier": "TEXT",
        "context_tokens": "INTEGER NOT NULL DEFAULT 0", "agreement": "INTEGER NOT NULL DEFAULT 0",
        "agreement_basis": "TEXT",
    }
    for name, definition in replay_additions.items():
        if name not in replay_columns:
            conn.execute(f"ALTER TABLE replay_results ADD COLUMN {name} {definition}")
    return conn


def reset_data(conn: sqlite3.Connection) -> None:
    conn.executescript("DELETE FROM replay_results; DELETE FROM model_calls; DELETE FROM user_turns;")


def add_turn(conn: sqlite3.Connection, turn: dict[str, Any]) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO user_turns
        (id,user,source,session_id,project_hash,timestamp,text,turn_index)
        VALUES (:id,:user,:source,:session_id,:project_hash,:timestamp,:text,:turn_index)""",
        turn,
    )


def add_call(conn: sqlite3.Connection, call: dict[str, Any]) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO model_calls
        (id,user_turn_id,source,session_id,project_hash,timestamp,model,input_tokens,output_tokens,
         cache_read_tokens,cache_write_tokens,context_tokens,tool_call_count,tool_names,tool_facts,
         final_text_chars,request_index,is_first,is_last,is_subagent,response_text,step_class)
        VALUES (:id,:user_turn_id,:source,:session_id,:project_hash,:timestamp,:model,:input_tokens,
                :output_tokens,:cache_read_tokens,:cache_write_tokens,:context_tokens,:tool_call_count,
                :tool_names,:tool_facts,:final_text_chars,:request_index,:is_first,:is_last,
                :is_subagent,:response_text,:step_class)""",
        call,
    )


def base_turn(turn_id: str, source: str, session_id: str, project: Any, timestamp: Any, text: str, index: int) -> dict[str, Any]:
    return {
        "id": turn_id, "user": os.environ.get("INFEROID_USER", "local"), "source": source,
        "session_id": session_id, "project_hash": project_hash(project),
        "timestamp": timestamp, "text": text, "turn_index": index,
    }


def empty_call(
    call_id: str,
    turn: dict[str, Any],
    timestamp: Any,
    model: str,
    tools: list[str],
    response: str,
    subagent: bool,
    tool_facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": call_id, "user_turn_id": turn["id"], "source": turn["source"],
        "session_id": turn["session_id"], "project_hash": turn["project_hash"],
        "timestamp": timestamp, "model": model or "unknown", "input_tokens": 0,
        "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
        "context_tokens": 0, "tool_call_count": len(tools), "tool_names": safe_json(tools),
        "tool_facts": safe_json(tool_facts or []), "final_text_chars": len(response),
        "request_index": 0, "is_first": 0, "is_last": 0,
        "is_subagent": int(subagent), "response_text": response, "step_class": None,
    }


def context_tokens_for(source: str, input_tokens: int, cache_read_tokens: int, cache_write_tokens: int) -> int:
    if source == "codex":
        return max(0, input_tokens) + max(0, cache_write_tokens)
    return max(0, input_tokens) + max(0, cache_read_tokens) + max(0, cache_write_tokens)


def load_json(path: Any, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def price_for(model: str) -> dict[str, Any]:
    table = load_json(PRICES_PATH, {"models": []}).get("models", [])
    lowered = model.lower()
    matches = [
        row for row in table
        if row.get("match") != "default" and str(row.get("match", "")).lower() in lowered
    ]
    if matches:
        return max(matches, key=lambda row: len(str(row.get("match", ""))))
    if is_local_model(model):
        return dict(LOCAL_MODEL_PRICE)
    return {}


def is_local_model(model: str) -> bool:
    """Recognize local model IDs without treating them as unpriced API models."""
    lowered = str(model or "").strip().lower()
    return ":" in lowered or lowered.startswith(("ollama/", "local/"))


def estimated_cost(call: sqlite3.Row) -> float:
    price = price_for(call["model"])
    if not price:
        return 0.0
    output = (call["output_tokens"] or 0) * float(price.get("output_per_million", 0))
    cached = (call["cache_read_tokens"] or 0) * float(price.get("cache_read_per_million", 0))
    if call["source"] == "codex":
        uncached_input = max((call["input_tokens"] or 0) - (call["cache_read_tokens"] or 0), 0)
        input_cost = uncached_input * float(price.get("input_per_million", 0))
        return (input_cost + cached + output) / 1_000_000
    input_cost = (call["input_tokens"] or 0) * float(price.get("input_per_million", 0))
    cache_write = (call["cache_write_tokens"] or 0) * float(price.get("cache_write_per_million", 0))
    return (input_cost + cached + cache_write + output) / 1_000_000


def turn_calls(conn: sqlite3.Connection, turn_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM model_calls WHERE user_turn_id=? ORDER BY timestamp,id", (turn_id,)).fetchall()


def is_local_capable(task_class: str | None, size: str | None) -> bool:
    return task_class in LOCAL_HEURISTIC_CLASSES and size in LOCAL_CAPABLE_SIZES
