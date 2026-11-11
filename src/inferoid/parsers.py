from __future__ import annotations

import collections
import datetime as dt
import difflib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from .storage import (
    add_call,
    add_turn,
    base_turn,
    coarse_key,
    connect,
    content_text,
    context_tokens_for,
    empty_call,
    in_window,
    now_iso,
    reset_data,
    safe_json,
    value_int,
)

KNOWN_CLAUDE_TYPES = {
    "assistant", "user", "system", "attachment", "file-history-snapshot", "file-history-delta",
    "api-request", "api-request-blob", "api-request-shape", "queue-operation", "atis-latch",
    "last-prompt", "ai-title", "tools-used", "pr-link", "frame-link", "mode", "cost-state",
    "permission-mode", "bridge-session", "session-owner", "artifact-comment-monitor",
    "artifact-autoreact-ledger",
}
KNOWN_CODEX_TYPES = {"session_meta", "event_msg", "response_item", "world_state", "turn_context", "token_usage_record", "compacted"}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _line_count(value: Any) -> int:
    if not isinstance(value, str) or not value:
        return 0
    return len(value.splitlines())


def _patch_line_counts(value: Any) -> tuple[int, int]:
    if not isinstance(value, str):
        return 0, 0
    value = value.replace("\\r\\n", "\n").replace("\\n", "\n")
    lines = value.splitlines()
    return (
        sum(1 for line in lines if line.startswith("+") and not line.startswith("+++")),
        sum(1 for line in lines if line.startswith("-") and not line.startswith("---")),
    )


def _patch_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("patch", "unified_diff", "diff"):
            if isinstance(value.get(key), str):
                return value[key]
        return ""
    if not isinstance(value, str):
        return ""
    begin = value.find("*** Begin Patch")
    end = value.find("*** End Patch", begin + 1) if begin >= 0 else -1
    if begin >= 0 and end >= 0:
        return value[begin:end]
    return value if "apply_patch" in value.lower() and ("+" in value or "-" in value) else ""


def _file_change_counts(changes: Any) -> tuple[int, int, int]:
    if not isinstance(changes, dict):
        return 0, 0, 0
    added = removed = 0
    for change in changes.values():
        if not isinstance(change, dict):
            continue
        patch = _patch_text(change)
        if patch:
            item_added, item_removed = _patch_line_counts(patch)
        else:
            item_added, item_removed = _line_count(change.get("content")), 0
        added += item_added
        removed += item_removed
    return added, removed, len(changes)


def _edit_counts(arguments: dict[str, Any], name: str) -> tuple[int, int, int]:
    if "changes" in arguments:
        return _file_change_counts(arguments.get("changes"))
    edits = arguments.get("edits")
    if isinstance(edits, list):
        added = removed = files = 0
        for edit in edits:
            item = _mapping(edit)
            item_added, item_removed, item_files = _edit_counts(item, name)
            added += item_added
            removed += item_removed
            files += item_files
        return added, removed, max(files, len(edits))
    old = arguments.get("old_string", arguments.get("oldText", arguments.get("old", "")))
    new = arguments.get("new_string", arguments.get("newText", arguments.get("new", "")))
    if "content" in arguments and not old and not new:
        return _line_count(arguments.get("content")), 0, 1
    if "patch" in arguments and isinstance(arguments["patch"], str):
        added, removed = _patch_line_counts(arguments["patch"])
        return added, removed, 1
    if old or new:
        old_lines = old.splitlines() if isinstance(old, str) else []
        new_lines = new.splitlines() if isinstance(new, str) else []
        added = removed = 0
        for tag, old_start, old_end, new_start, new_end in difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes():
            if tag in {"replace", "insert"}:
                added += new_end - new_start
            if tag in {"replace", "delete"}:
                removed += old_end - old_start
        return added, removed, 1
    return 0, 0, 1


SHELL_COMMAND_KEYS = ("command", "cmd", "shell_command", "argv")
SHELL_OPERATORS = {"&&", "||", ";", "|"}
SHELL_LAUNCHERS = {"bash", "sh", "zsh", "dash", "/bin/bash", "/bin/sh", "/bin/zsh", "/bin/dash"}
SHELL_FIELD_RE = re.compile(r"[\"']?(?:command|cmd|shell_command|argv)[\"']?\s*:\s*([\"'`])")
ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _quoted_value(value: str, start: int, quote: str) -> str:
    escaped = False
    chars: list[str] = []
    for char in value[start:]:
        if escaped:
            chars.append(char)
            escaped = False
        elif char == "\\" and quote != "`":
            escaped = True
        elif char == quote:
            break
        else:
            chars.append(char)
    result = "".join(chars)
    if quote == '"':
        try:
            return json.loads(quote + result + quote)
        except json.JSONDecodeError:
            pass
    return result


def _shell_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in SHELL_COMMAND_KEYS:
            if value.get(key) is not None:
                return value[key]
        return None
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        found = _shell_value(parsed)
        if found is not None:
            return found
    match = SHELL_FIELD_RE.search(value)
    if match:
        return _quoted_value(value, match.end(), match.group(1))
    return value


def _wrapper_shell_values(value: str) -> list[str]:
    values: list[str] = []
    for match in SHELL_FIELD_RE.finditer(value):
        values.append(_quoted_value(value, match.end(), match.group(1)))
    array_match = re.search(r"\b(?:cmds|commands|specs)\s*=\s*\[", value)
    if array_match:
        groups: list[list[str]] = [[]]
        stack = [0]
        index = array_match.end()
        quote: str | None = None
        escaped = False
        while index < len(value):
            char = value[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\" and quote != "`":
                    escaped = True
                elif char == quote:
                    quote = None
                index += 1
                continue
            if char in "\"'`":
                quote = char
                start = index + 1
                end = start
                escaped = False
                while end < len(value):
                    if escaped:
                        escaped = False
                    elif value[end] == "\\" and char != "`":
                        escaped = True
                    elif value[end] == char:
                        break
                    end += 1
                groups[stack[-1]].append(_quoted_value(value, start, char))
                quote = None
                index = end + 1
                continue
            if char == "[":
                groups.append([])
                stack.append(len(groups) - 1)
            elif char == "]":
                if len(stack) == 1:
                    break
                stack.pop()
            index += 1
        nested = [group[-1] for group in groups[1:] if group]
        values.extend(nested or groups[0])
    command_match = re.search(r"\b\w*(?:cmd|command)\s*=\s*([\"'`])", value, re.IGNORECASE)
    if command_match:
        values.append(_quoted_value(value, command_match.end(), command_match.group(1)))
    return values


def _shell_tokens(value: Any) -> list[str]:
    value = _shell_value(value)
    if isinstance(value, list):
        tokens = [str(item) for item in value if isinstance(item, (str, int, float))]
    elif isinstance(value, str):
        try:
            lexer = shlex.shlex(value, posix=True, punctuation_chars=";&|<>")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            tokens = value.split()
    else:
        return []
    if tokens and tokens[0] in SHELL_LAUNCHERS:
        for index, token in enumerate(tokens[1:], start=1):
            if token in {"-c", "-lc", "-cl"} and index + 1 < len(tokens):
                return _shell_tokens(tokens[index + 1])
    return tokens


def _shell_segments(tokens: list[str]) -> tuple[list[list[str]], bool]:
    segments: list[list[str]] = []
    current: list[str] = []
    redirected = False
    for token in tokens:
        if token in SHELL_OPERATORS:
            if current:
                segments.append(current)
                current = []
        elif token in {">", ">>"}:
            redirected = True
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments, redirected


def _segment_command(segment: list[str]) -> str:
    words = list(segment)
    while words and (ENV_ASSIGNMENT_RE.match(words[0]) or words[0] in {"cd", "time"}):
        if words[0] == "cd":
            words = words[2:]
        else:
            words.pop(0)
    if words and words[0] == "timeout":
        words.pop(0)
        while words and words[0].startswith("-"):
            words.pop(0)
        if words and re.fullmatch(r"(?:\d+(?:\.\d+)?)(?:ms|s|m|h)?", words[0]):
            words.pop(0)
    if not words:
        return ""
    command = words[0].rsplit("/", 1)[-1]
    if command == "git":
        subcommand = next((word for word in words[1:] if not word.startswith("-")), "")
        return f"git {subcommand}" if subcommand else "git"
    if command == "sed":
        options = [word for word in words[1:] if word.startswith("-")]
        if any("i" in option[1:] for option in options):
            return "sed"
        if any(word == "-n" or word.startswith("-n") for word in options):
            return "sed -n"
    return command


def shell_command_first_words(value: Any) -> tuple[list[str], bool]:
    if isinstance(value, str) and value.lstrip().startswith("const"):
        wrapper_values = _wrapper_shell_values(value)
        if not wrapper_values:
            return [], False
        words: list[str] = []
        redirected = False
        for wrapper_value in wrapper_values:
            segment_words, segment_redirected = shell_command_first_words(wrapper_value)
            words.extend(segment_words)
            redirected = redirected or segment_redirected
        return words, redirected
    segments, redirected = _shell_segments(_shell_tokens(value))
    return [_segment_command(segment) for segment in segments if _segment_command(segment)], redirected


def tool_fact(name: Any, arguments: Any = None, item_type: str = "") -> dict[str, Any]:
    """Keep only facts needed for deterministic step classes and replay comparison."""
    tool_name = str(name or item_type or "unknown")
    lowered = tool_name.lower().replace("-", "_")
    args = _mapping(arguments)
    fact: dict[str, Any] = {"name": tool_name}
    patch = _patch_text(args) or _patch_text(arguments)
    if not patch and isinstance(arguments, dict):
        patch = _patch_text(arguments.get("command"))
    if patch and ("apply_patch" in str(arguments).lower() or "patch" in lowered or lowered in {"exec", "command"}):
        added, removed = _patch_line_counts(patch)
        fact.update({"name": "apply_patch", "added_lines": added, "removed_lines": removed, "file_count": 1})
    path = next((args.get(key) for key in ("file_path", "path", "filename", "file") if args.get(key)), None)
    pattern = next((args.get(key) for key in ("pattern", "query", "glob") if args.get(key)), None)
    if path is not None:
        fact["key_arg_hash"] = coarse_key(path)
        fact["key_arg_kind"] = "path"
    elif pattern is not None:
        fact["key_arg_hash"] = coarse_key(pattern)
        fact["key_arg_kind"] = "pattern"
    shell_value = next((args.get(key) for key in SHELL_COMMAND_KEYS if args.get(key)), None)
    if not shell_value and isinstance(arguments, str) and lowered in {"bash", "shell", "exec", "command"}:
        shell_value = _shell_value(arguments)
        if (
            shell_value == arguments
            and arguments.lstrip().startswith("const")
            and "exec_command" not in arguments
            and not re.search(r"\b(?:cmds|commands|specs)\s*=\s*\[", arguments)
            and not re.search(r"\b\w*(?:cmd|command)\s*=\s*[\"'`]", arguments, re.IGNORECASE)
        ):
            shell_value = None
    if shell_value is not None:
        first_words, redirected = shell_command_first_words(shell_value)
        if first_words:
            fact["command_first_words"] = first_words
            fact["command_first_word"] = first_words[0]
        if redirected:
            fact["shell_redirect"] = True
    if not patch and (lowered in {"edit", "write", "multiedit", "apply_patch", "applypatch", "filechange", "file_change"} or "edit" in lowered or "patch" in lowered):
        added, removed, files = _edit_counts(args, tool_name)
        fact.update({"added_lines": added, "removed_lines": removed, "file_count": files})
    return fact


def tool_facts_from_claude(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [
        tool_fact(item.get("name"), item.get("input"), str(item.get("type", "")))
        for item in content if isinstance(item, dict) and item.get("type") == "tool_use"
    ]


def tool_fact_from_codex(item: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    item_type = str(item.get("type") or payload.get("type") or "unknown")
    name = item.get("name") or payload.get("name") or item_type
    if item_type == "CommandExecution":
        args = {"command": item.get("command", item.get("parsed_cmd", []))}
    elif item_type == "FileChange":
        args = {"changes": item.get("changes", {})}
    else:
        args = item.get("arguments", item.get("input", item.get("command", item.get("changes", {}))))
    if isinstance(args, str) and item_type in {"CommandExecution", "command_execution"}:
        args = {"command": args}
    return tool_fact(name, args, item_type)


def source_stats() -> dict[str, Any]:
    return {
        "lines": 0,
        "parsed": 0,
        "ignored": 0,
        "unknown": 0,
        "malformed": 0,
        "unreadable": 0,
        "turns": 0,
        "calls": 0,
        "cache_context_warning": False,
        "cache_context_warning_count": 0,
        "unknown_types": collections.Counter(),
    }


def parse_claude(path: Path, cutoff: dt.datetime, conn, stats: dict[str, Any]) -> None:
    session_id = path.stem
    subagent = path.parent.name == "subagents" or "subagents" in path.parts
    latest_turn: dict[str, Any] | None = None
    turn_index = call_index = 0
    cache_usage_by_request: dict[str, list[tuple[int, bool]]] = {}
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        stats["unreadable"] += 1
        return
    with handle:
        for line in handle:
            stats["lines"] += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                stats["malformed"] += 1
                continue
            if not isinstance(record, dict):
                stats["unknown"] += 1
                continue
            typ = record.get("type")
            if typ not in KNOWN_CLAUDE_TYPES:
                stats["unknown"] += 1
                stats["unknown_types"][str(typ)] += 1
                continue
            message = record.get("message")
            timestamp = record.get("timestamp")
            if (
                typ == "user"
                and isinstance(message, dict)
                and message.get("role") == "user"
                and not record.get("isMeta")
            ):
                content = message.get("content")
                text = content_text(content)
                item_types = {item.get("type") for item in content if isinstance(item, dict)} if isinstance(content, list) else set()
                if text and (not item_types or not item_types.issubset({"tool_result", "tool_output"})):
                    turn_index += 1
                    latest_turn = base_turn(f"claude:{session_id}:{turn_index}", "claude", session_id, record.get("cwd", path.parent.name), timestamp, text, turn_index)
                    if in_window(timestamp, cutoff):
                        add_turn(conn, latest_turn)
                        stats["turns"] += 1
                stats["parsed"] += 1
                continue
            if typ != "assistant" or not isinstance(message, dict) or message.get("role") != "assistant":
                stats["ignored"] += 1
                continue
            timestamp = timestamp or message.get("timestamp")
            if not in_window(timestamp, cutoff):
                stats["ignored"] += 1
                continue
            if latest_turn is None or conn.execute("SELECT 1 FROM user_turns WHERE id=?", (latest_turn["id"],)).fetchone() is None:
                turn_index += 1
                latest_turn = base_turn(f"claude:{session_id}:{turn_index}", "claude", session_id, record.get("cwd", path.parent.name), timestamp, "", turn_index)
                add_turn(conn, latest_turn)
                stats["turns"] += 1
            usage = message.get("usage", {})
            request_ref = record.get("requestRef") or message.get("requestRef")
            if request_ref is not None:
                cache_read = value_int(usage, "cache_read_input_tokens", "cached_input_tokens")
                has_cache_read_field = any(
                    key in usage for key in ("cache_read_input_tokens", "cached_input_tokens")
                )
                observed = cache_usage_by_request.setdefault(str(request_ref), [])
                if (
                    observed
                    and max(cache for cache, _ in observed + [(cache_read, has_cache_read_field)]) > 0
                    and any(cache == 0 or not present for cache, present in observed)
                    != (cache_read == 0 or not has_cache_read_field)
                ):
                    stats["cache_context_warning"] = True
                    stats["cache_context_warning_count"] += 1
                observed.append((cache_read, has_cache_read_field))
            content = message.get("content", [])
            facts = tool_facts_from_claude(content)
            tools = [fact["name"] for fact in facts]
            response = content_text([item for item in content if isinstance(item, dict) and item.get("type") == "text"])
            model = str(message.get("model") or record.get("requestedModel") or "unknown")
            call_index += 1
            call = empty_call(
                f"claude:{session_id}:call:{call_index}", latest_turn, timestamp, model, tools,
                response, subagent or bool(record.get("isSidechain")), facts,
            )
            call.update(
                input_tokens=value_int(usage, "input_tokens"), output_tokens=value_int(usage, "output_tokens"),
                cache_read_tokens=value_int(usage, "cache_read_input_tokens", "cached_input_tokens"),
                cache_write_tokens=value_int(usage, "cache_creation_input_tokens", "cache_write_input_tokens"),
            )
            call["context_tokens"] = context_tokens_for(
                "claude", call["input_tokens"], call["cache_read_tokens"], call["cache_write_tokens"]
            )
            add_call(conn, call)
            stats["calls"] += 1
            stats["parsed"] += 1
    conn.commit()


def codex_turn_id(payload: dict[str, Any], item: dict[str, Any] | None = None) -> str | None:
    item = item or {}
    metadata = item.get("internal_chat_message_metadata_passthrough")
    if not isinstance(metadata, dict):
        metadata = payload.get("internal_chat_message_metadata_passthrough")
    for value in (payload.get("turn_id"), payload.get("root_turn_id"),
                  metadata.get("turn_id") if isinstance(metadata, dict) else None):
        if value is not None:
            return str(value)
    return None


def codex_user_text(payload: dict[str, Any], item: dict[str, Any], record_type: str) -> str:
    if record_type == "response_item":
        metadata = item.get("internal_chat_message_metadata_passthrough")
        if isinstance(metadata, dict):
            kinds = metadata.get("content_item_kinds") or []
            if kinds and not any(str(kind) in {"user.text", "user_message", "user.prompt"} for kind in kinds):
                return ""
        if item.get("role") != "user" and str(item.get("type")) not in {"UserMessage", "user_message"}:
            return ""
        return content_text(item.get("content", item.get("text", ""))).strip()
    event_type = str(payload.get("type") or payload.get("event_type") or "")
    if event_type not in {"user_message", "user_prompt"}:
        return ""
    return content_text(payload.get("message", payload.get("content", payload.get("text", "")))).strip()


def parse_codex(path: Path, cutoff: dt.datetime, conn, stats: dict[str, Any]) -> None:
    file_id = path.stem
    session_id = file_id
    project = "unknown"
    model = "unknown"
    turns: list[dict[str, Any]] = []
    pending_tools: list[dict[str, Any]] = []
    pending_response: list[str] = []
    usage_records: list[tuple[int, Any, dict[str, Any], str, list[dict[str, Any]], str]] = []
    active_turn: int | None = None
    current_model = "unknown"
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        stats["unreadable"] += 1
        return
    with handle:
        def replace_pending_wrapper(fact: dict[str, Any]) -> bool:
            for index in range(len(pending_tools) - 1, -1, -1):
                if pending_tools[index].get("_wrapper"):
                    pending_tools[index] = fact
                    return True
            if usage_records and usage_records[-1][0] == active_turn:
                recorded_facts = usage_records[-1][4]
                for index in range(len(recorded_facts) - 1, -1, -1):
                    if recorded_facts[index].get("_wrapper"):
                        recorded_facts[index] = fact
                        return True
            return False

        for line in handle:
            stats["lines"] += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                stats["malformed"] += 1
                continue
            if not isinstance(record, dict):
                stats["unknown"] += 1
                continue
            typ = record.get("type")
            if typ not in KNOWN_CODEX_TYPES:
                stats["unknown"] += 1
                stats["unknown_types"][str(typ)] += 1
                continue
            timestamp = record.get("timestamp")
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            if typ == "session_meta":
                session_id = str(payload.get("id") or payload.get("session_id") or file_id)
                project = payload.get("cwd") or project
                stats["parsed"] += 1
                continue
            if typ == "turn_context":
                current_model = str(payload.get("model") or current_model)
                model = current_model
                project = payload.get("cwd") or project
                stats["parsed"] += 1
                continue
            if typ == "token_usage_record":
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                if active_turn is None:
                    turns.append(base_turn(
                        f"codex:{file_id}:human:{len(turns) + 1}", "codex", session_id,
                        project, timestamp, "", len(turns) + 1,
                    ))
                    active_turn = len(turns) - 1
                usage_records.append((
                    active_turn, timestamp, usage, current_model,
                    list(pending_tools), "\n".join(pending_response),
                ))
                pending_tools.clear()
                pending_response.clear()
                stats["parsed"] += 1
                continue
            if typ == "event_msg":
                text = codex_user_text(payload, {}, typ)
                if text:
                    turns.append(base_turn(
                        f"codex:{file_id}:human:{len(turns) + 1}", "codex", session_id,
                        project, timestamp, text, len(turns) + 1,
                    ))
                    active_turn = len(turns) - 1
                item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
                if item.get("type") in {"CommandExecution", "FileChange"}:
                    fact = tool_fact_from_codex(item, payload)
                    if not replace_pending_wrapper(fact):
                        pending_tools.append(fact)
                stats["parsed"] += 1
                continue
            if typ != "response_item":
                stats["ignored"] += 1
                continue
            item = payload.get("item") if isinstance(payload.get("item"), dict) else payload
            item_type = str(item.get("type") or payload.get("type") or "")
            text = codex_user_text(payload, item, typ)
            if text:
                turns.append(base_turn(
                    f"codex:{file_id}:human:{len(turns) + 1}", "codex", session_id,
                    project, timestamp, text, len(turns) + 1,
                ))
                active_turn = len(turns) - 1
                pending_tools.clear()
                pending_response.clear()
            elif item_type in {"AgentMessage", "agent_message"} or item.get("role") == "assistant":
                if active_turn is None:
                    turns.append(base_turn(
                        f"codex:{file_id}:human:{len(turns) + 1}", "codex", session_id,
                        project, timestamp, "", len(turns) + 1,
                    ))
                    active_turn = len(turns) - 1
                pending_response.append(content_text(item.get("content", item.get("text", ""))))
            elif item_type in {"CommandExecution", "FileChange"}:
                if active_turn is None:
                    turns.append(base_turn(
                        f"codex:{file_id}:human:{len(turns) + 1}", "codex", session_id,
                        project, timestamp, "", len(turns) + 1,
                    ))
                    active_turn = len(turns) - 1
                fact = tool_fact_from_codex(item, payload)
                if not replace_pending_wrapper(fact):
                    pending_tools.append(fact)
            elif (item_type in {"function_call", "custom_tool_call"}
                  or payload.get("type") in {"function_call", "custom_tool_call"}
                  or item_type in {"Extension", "WebSearch", "ImageView"}):
                if active_turn is None:
                    turns.append(base_turn(
                        f"codex:{file_id}:human:{len(turns) + 1}", "codex", session_id,
                        project, timestamp, "", len(turns) + 1,
                    ))
                    active_turn = len(turns) - 1
                fact = tool_fact_from_codex(item, payload)
                original_name = str(item.get("name") or payload.get("name") or item_type).lower()
                if original_name in {"exec", "command", "shell"}:
                    fact["_wrapper"] = True
                pending_tools.append(fact)
            stats["parsed"] += 1
    call_index = 0
    added_turns: set[int] = set()
    for call_index, (turn_index, timestamp, usage, record_model, facts, response) in enumerate(usage_records, start=1):
        if not in_window(timestamp, cutoff):
            continue
        turn = turns[turn_index]
        if not in_window(turn.get("timestamp") or timestamp, cutoff):
            continue
        if turn_index not in added_turns:
            add_turn(conn, turn)
            added_turns.add(turn_index)
            stats["turns"] += 1
        facts = [{key: value for key, value in fact.items() if not key.startswith("_")} for fact in facts]
        tools = [fact["name"] for fact in facts]
        call = empty_call(
            f"codex:{file_id}:call:{call_index}", turn, timestamp,
            record_model or model, tools, response, False, facts,
        )
        call.update(
            input_tokens=value_int(usage, "input_tokens"), output_tokens=value_int(usage, "output_tokens"),
            cache_read_tokens=value_int(usage, "cached_input_tokens", "cache_read_input_tokens"),
            cache_write_tokens=value_int(usage, "cache_write_input_tokens", "cache_creation_input_tokens"),
        )
        call["context_tokens"] = context_tokens_for(
            "codex", call["input_tokens"], call["cache_read_tokens"], call["cache_write_tokens"]
        )
        add_call(conn, call)
        stats["calls"] += 1
    for turn_index, turn in enumerate(turns):
        if turn_index not in added_turns and in_window(turn.get("timestamp"), cutoff):
            add_turn(conn, turn)
            stats["turns"] += 1
    conn.commit()


def finalize_call_positions(conn) -> None:
    """Assign stable per-request positions after both providers have been parsed."""
    turn_ids = [row[0] for row in conn.execute("SELECT id FROM user_turns")]
    for turn_id in turn_ids:
        rows = conn.execute(
            "SELECT id FROM model_calls WHERE user_turn_id=? ORDER BY timestamp,id", (turn_id,)
        ).fetchall()
        for index, row in enumerate(rows):
            call = conn.execute("SELECT source,input_tokens,cache_read_tokens,cache_write_tokens FROM model_calls WHERE id=?", (row[0],)).fetchone()
            context_tokens = context_tokens_for(
                call["source"], call["input_tokens"], call["cache_read_tokens"], call["cache_write_tokens"]
            )
            conn.execute(
                "UPDATE model_calls SET request_index=?,is_first=?,is_last=?,context_tokens=?,final_text_chars=length(response_text) WHERE id=?",
                (index, int(index == 0), int(index == len(rows) - 1), context_tokens, row[0]),
            )


def scan(days: int) -> dict[str, Any]:
    conn = connect()
    reset_data(conn)
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=max(0, days))
    stats: dict[str, Any] = {"claude": source_stats(), "codex": source_stats(), "cutoff": cutoff.isoformat(), "files": {"claude": 0, "codex": 0}}
    roots = [Path.home() / ".claude" / "projects"]
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        roots.append(Path(os.environ["CLAUDE_CONFIG_DIR"]).expanduser() / "projects")
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            if path in seen:
                continue
            seen.add(path)
            stats["files"]["claude"] += 1
            parse_claude(path, cutoff, conn, stats["claude"])
    codex_root = Path.home() / ".codex" / "sessions"
    if codex_root.exists():
        for path in codex_root.glob("*/*/*/rollout-*.jsonl"):
            stats["files"]["codex"] += 1
            parse_codex(path, cutoff, conn, stats["codex"])
    finalize_call_positions(conn)
    serializable = json.loads(json.dumps(stats, default=dict))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('last_scan',?)", (safe_json(serializable),))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('last_scan_at',?)", (now_iso(),))
    conn.commit()
    conn.close()
    return serializable


def format_scan(stats: dict[str, Any]) -> str:
    lines = [f"Scanned {stats['files']['claude']} Claude files and {stats['files']['codex']} Codex files."]
    for source in ("claude", "codex"):
        item = stats[source]
        unknown = ", ".join(f"{key}={value}" for key, value in sorted(item["unknown_types"].items())) or "none"
        lines.append(f"{source}: {item['calls']} calls, {item['turns']} turns, {item['parsed']} parsed lines, {item['ignored']} ignored, {item['unknown']} unknown, {item['malformed']} malformed (unknown types: {unknown}).")
    return "\n".join(lines)
