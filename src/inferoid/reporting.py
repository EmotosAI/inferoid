from __future__ import annotations

import collections
import html
import json
import statistics
import time

from .analysis import (
    classify_step,
    detect_machine,
    json_facts,
    local_chat,
    ollama_chat,
    step_is_local_capable,
    validate_local_endpoint,
)
from .parsers import tool_fact
from .storage import (
    CLASSES,
    LOCAL_HEURISTIC_CLASSES,
    PRICES_PATH,
    STEP_CLASSES,
    STEP_LOCAL_CLASSES,
    connect,
    db_home,
    estimated_cost,
    is_local_model,
    load_json,
    now_iso,
    price_for,
    tier_for_ram,
    tier_spec,
    turn_calls,
)


def _ensure_step_classes(conn) -> None:
    for call in conn.execute("SELECT * FROM model_calls"):
        step_class = classify_step(call)
        if call["step_class"] != step_class:
            conn.execute("UPDATE model_calls SET step_class=? WHERE id=?", (step_class, call["id"]))
    conn.commit()


def _context_bucket(tokens: int) -> str:
    if tokens < 16_000:
        return "under 16k"
    if tokens < 32_000:
        return "16k-32k"
    if tokens < 64_000:
        return "32k-64k"
    if tokens < 128_000:
        return "64k-128k"
    return "128k+"


def _replay_stats(conn, tier: str) -> dict[str, dict[str, object]]:
    result = {
        step_class: {"n": 0, "agreement": 0, "median_ms": None, "established": False}
        for step_class in STEP_CLASSES
    }
    rows = conn.execute(
        "SELECT step_class,agreement,wall_time_ms FROM replay_results WHERE tier=? ORDER BY id", (tier,)
    ).fetchall()
    times: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        step_class = row["step_class"] or "other / unknown"
        if step_class not in result:
            result[step_class] = {"n": 0, "agreement": 0, "median_ms": None, "established": False}
        result[step_class]["n"] += 1
        result[step_class]["agreement"] += int(row["agreement"] or 0)
        if row["wall_time_ms"] is not None:
            times[step_class].append(float(row["wall_time_ms"]))
    for step_class, values in times.items():
        result[step_class]["median_ms"] = round(statistics.median(values), 1)
    for value in result.values():
        value["established"] = value["n"] >= 20
    return result


def local_savings(calls) -> str:
    """Local-model calls priced as if the paid model used most in this history had done them (API list price)."""
    local = [c for c in calls if is_local_model(c["model"])]
    if not local:
        return "Local model work: none recorded yet."
    paid: dict[str, int] = {}
    for c in calls:
        if not is_local_model(c["model"]) and price_for(c["model"]):
            paid[c["model"]] = paid.get(c["model"], 0) + 1
    tokens_in = sum((c["input_tokens"] or 0) + (c["cache_read_tokens"] or 0) for c in local)
    tokens_out = sum(c["output_tokens"] or 0 for c in local)
    if not paid:
        return f"Local model work: {len(local)} calls, {tokens_in:,} input and {tokens_out:,} output tokens, $0."
    ref = max(paid, key=paid.get)
    price = price_for(ref)
    usd = (sum(c["input_tokens"] or 0 for c in local) * float(price.get("input_per_million", 0))
           + sum(c["cache_read_tokens"] or 0 for c in local) * float(price.get("cache_read_per_million", 0))
           + tokens_out * float(price.get("output_per_million", 0))) / 1e6
    return (f"Local model work: {len(local)} calls, {tokens_in:,} input and {tokens_out:,} output tokens, $0. "
            f"The same tokens on {ref} would be about ${usd:.2f} at API list price.")


def report(tier: str | None = None) -> tuple[str, str, dict]:
    conn = connect()
    _ensure_step_classes(conn)
    machine_info = detect_machine()
    selected_tier = str(tier or tier_for_ram(int(machine_info.get("ram_gb", 0) or 0)))
    if selected_tier not in {"16", "48", "128"}:
        raise ValueError("tier must be 16, 48, or 128")
    max_context = int(tier_spec(selected_tier)["max_context_tokens"])
    turns = conn.execute("SELECT * FROM user_turns").fetchall()
    by_id = {turn["id"]: turn for turn in turns}
    calls = conn.execute("SELECT * FROM model_calls ORDER BY timestamp,id").fetchall()
    by_class = {
        key: {"calls": 0, "tokens": 0, "cost": 0.0, "unknown_cost": 0, "local_calls": 0}
        for key in CLASSES
    }
    step_by_class = {
        key: {"calls": 0, "cost": 0.0, "local_calls": 0, "local_cost": 0.0}
        for key in STEP_CLASSES
    }
    context_distribution = {key: 0 for key in ("under 16k", "16k-32k", "32k-64k", "64k-128k", "128k+")}
    local_model_calls = 0
    api_priced_calls = 0
    for call in calls:
        turn = by_id.get(call["user_turn_id"])
        category = turn["task_class"] if turn and turn["task_class"] in CLASSES else "unclassified"
        bucket = by_class[category]
        bucket["calls"] += 1
        bucket["tokens"] += call["input_tokens"] + call["output_tokens"]
        bucket["cost"] += estimated_cost(call)
        if is_local_model(call["model"]):
            local_model_calls += 1
        elif not price_for(call["model"]):
            bucket["unknown_cost"] += 1
        else:
            api_priced_calls += 1
        if turn and turn["task_class"] in LOCAL_HEURISTIC_CLASSES and turn["size"] in {"small", "medium"}:
            bucket["local_calls"] += 1

        step_class = call["step_class"] or classify_step(call)
        step = step_by_class.setdefault(step_class, {"calls": 0, "cost": 0.0, "local_calls": 0, "local_cost": 0.0})
        cost = estimated_cost(call)
        step["calls"] += 1
        step["cost"] += cost
        context_distribution[_context_bucket(int(call["context_tokens"] or 0))] += 1
        if step_class in STEP_LOCAL_CLASSES and int(call["context_tokens"] or 0) <= max_context:
            step["local_calls"] += 1
            step["local_cost"] += cost

    saved_note = local_savings(calls)
    total_calls = len(calls)
    total_tokens = sum(call["input_tokens"] + call["output_tokens"] for call in calls)
    total_cost = sum(estimated_cost(call) for call in calls)
    request_local_calls = sum(item["local_calls"] for item in by_class.values())
    unclassified_calls = by_class["unclassified"]["calls"]
    classified_calls = total_calls - unclassified_calls
    request_headline_share = request_local_calls / (classified_calls or 1) * 100
    step_local_calls = sum(item["local_calls"] for item in step_by_class.values())
    step_local_cost = sum(item["local_cost"] for item in step_by_class.values())
    step_headline_share = step_local_calls / (total_calls or 1) * 100
    step_dollar_share = step_local_cost / (total_cost or 1) * 100
    unknown_cost_calls = sum(item["unknown_cost"] for item in by_class.values())
    subagent_calls = sum(call["is_subagent"] for call in calls)
    replay = _replay_stats(conn, selected_tier)
    replay_n = sum(int(value["n"]) for value in replay.values())
    replay_agreement = sum(int(value["agreement"]) for value in replay.values())
    replay_percent = replay_agreement / (replay_n or 1) * 100
    prices = load_json(PRICES_PATH, {})
    try:
        last_scan = json.loads(conn.execute("SELECT value FROM meta WHERE key='last_scan'").fetchone()[0])
    except (TypeError, KeyError, json.JSONDecodeError, IndexError):
        last_scan = {}
    cache_context_warning = any(
        bool(last_scan.get(source, {}).get("cache_context_warning")) for source in ("claude", "codex")
    )
    context_note = (
        "Context accounting warning: context may be under-reported for this backend because repeated "
        "assistant blocks for one request disagree about cache-read tokens."
        if cache_context_warning
        else "No cache-token consistency warning was found in the scan metadata."
    )
    ranking = sorted(
        ((key, value) for key, value in step_by_class.items() if value["calls"] and key in STEP_LOCAL_CLASSES),
        key=lambda pair: pair[1]["cost"], reverse=True,
    )[:5]
    class_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value['calls']}</td><td>${value['cost']:.2f}</td>"
        f"<td>{value['local_calls']}</td><td>${value['local_cost']:.2f}</td></tr>"
        for key, value in step_by_class.items() if value["calls"]
    )
    distribution_rows = "".join(
        f"<tr><td>{label}</td><td>{count}</td><td>{count / (total_calls or 1) * 100:.1f}%</td></tr>"
        for label, count in context_distribution.items()
    )
    replay_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value['n']}</td><td>"
        f"{(value['agreement'] / value['n'] * 100):.1f}%</td><td>"
        f"{'established' if value['established'] else 'not established'}</td><td>"
        f"{value['median_ms'] if value['median_ms'] is not None else 'n/a'}</td></tr>"
        for key, value in replay.items() if value["n"]
    )
    top_html = "".join(f"<li>{html.escape(key)}: {value['calls']} calls, ${value['cost']:.2f} estimate</li>" for key, value in ranking)
    tier_label = html.escape(str(tier_spec(selected_tier).get("ram_label", selected_tier)))
    html_doc = f"""<!doctype html><meta charset=utf-8><title>inferoid report</title>
<style>body{{font:16px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem;color:#202124}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.5rem;text-align:left}}small{{color:#666}}</style>
<h1>inferoid report</h1><p>Private local report. Every dollar and share below is an estimate.</p>
<h2>Request-level share</h2><p>{request_local_calls}/{classified_calls or 1} classified-request calls ({request_headline_share:.1f}%). This is retained for comparison with earlier reports.</p>
<h2>Step-level share</h2><p>Tier {selected_tier} ({tier_label}, maximum context {max_context:,}): {step_local_calls}/{total_calls or 1} steps ({step_headline_share:.1f}%) and ${step_local_cost:.2f}/${total_cost:.2f} priced API list-price equivalent ({step_dollar_share:.1f}%).</p>
<p>{total_calls} recorded model calls, {total_tokens} recorded input/output tokens, ${total_cost:.2f} API list-price equivalent for priced API models. {local_model_calls} local model ($0) calls are excluded from that denominator; {unknown_cost_calls} unknown-model calls are also excluded.</p>
<p>{html.escape(context_note)}</p>
<h2>Context-size distribution</h2><table><tr><th>Context tokens</th><th>Steps</th><th>Share</th></tr>{distribution_rows}</table>
<h2>Step classes</h2><table><tr><th>Class</th><th>Steps</th><th>Known cost</th><th>Local-capable steps</th><th>Local-capable cost</th></tr>{class_rows}</table>
<h2>Replay agreement by class</h2><p>Replay uses a neutral system prompt because provider system prompts are not in the logs. It sends reconstructed normalized messages and tool names; omitted tool results and unavailable definitions are not claimed to be exact. Dispatch agreement is objective; edit and short-text agreement is judged by the local model.</p><table><tr><th>Class</th><th>N</th><th>Agreement</th><th>Evidence threshold</th><th>Median wall time (ms)</th></tr>{replay_rows or '<tr><td colspan=5>No replay results for this tier.</td></tr>'}</table>
<h2>Request-level task classes</h2><p>Unclassified: {unclassified_calls} calls. Subagent: {subagent_calls}/{total_calls or 1} calls ({subagent_calls / (total_calls or 1) * 100:.1f}%).</p>
<h2>Top 5 step classes worth moving local</h2><ul>{top_html or '<li>No local-capable steps recorded.</li>'}</ul>
    <h2>Method</h2><p>Step classes are deterministic: read/search dispatch is only local read, grep, glob, ls, or search-style output; web/search tools, subagent/task dispatch, and unknown tool names are separate non-local-capable classes; shell dispatch is split into read-only first words such as ls, cat, git, rg, and mutating shell output; a small edit is one edit/write with at most 30 changed lines; a large edit is bigger or multi-file; short text is final text of at most 1,500 characters with no tool call; intermediate text is shorter text from a non-final request step; longer text is long text / planning; no tool and no text is reasoning only. Local-capable step classes are read/search dispatch, read-only shell dispatch, small edit, and short text, provided input plus cache-read plus cache-write context fits the selected tier. The limits are 16k for 16-32 GB, 32k for 48-64 GB, and 64k for 128 GB+ because prefill time and KV memory grow with context on Apple Silicon. Dollar shares use dated public list prices; local model ($0) and unknown models are excluded from API list-price denominators. Request-level numbers still use the prior deterministic task-class and size heuristic. A replay percentage is established per class only at N &gt;= 20.</p>"""
    output = db_home() / "report.html"
    output.write_text(html_doc, encoding="utf-8")
    replay_text = f"{replay_percent:.1f}% replay-agreed on {replay_n} steps"
    share = f"""{step_headline_share:.1f}% of the steps in my Claude Code + Codex sessions could likely run on my {machine_info['chip']} {machine_info['ram_gb']} GB ({replay_text})
{request_headline_share:.1f}% of the classified request-level calls could likely run locally (retained request-level estimate).

Tier: {selected_tier} ({tier_label}); step dollar share: {step_dollar_share:.1f}% (${step_local_cost:.2f}/${total_cost:.2f} priced API list-price equivalent)
Recorded steps: {total_calls}; local model ($0) calls: {local_model_calls}; unknown-model calls: {unknown_cost_calls}; subagent steps: {subagent_calls}
{saved_note}
Context distribution: """ + ", ".join(f"{key}={value}" for key, value in context_distribution.items()) + f"\nPrice table date: {prices.get('as_of', 'unknown')}. API list-price denominator excludes local model ($0) and unknown-model calls; aggregates only; subscription users pay a flat plan.\n{context_note}\n"
    # Compatibility for databases written by the round-2 schema before per-step facts existed.
    legacy_rows = calls and all(
        not call["context_tokens"] and not call["final_text_chars"] and not json_facts(call)
        for call in calls
    )
    if legacy_rows:
        share = f"""{request_headline_share:.1f}% of my Claude Code + Codex calls could likely run on my {machine_info['chip']} {machine_info['ram_gb']} GB; request-level compatibility view.
{step_headline_share:.1f}% of steps could likely run locally once per-call facts are available.
"""
    (db_home() / "share-card.md").write_text(share, encoding="utf-8")
    timed_rows = conn.execute("SELECT wall_time_ms FROM replay_results WHERE tier=? AND wall_time_ms IS NOT NULL", (selected_tier,)).fetchall()
    summary = {
        "calls": total_calls, "classified_calls": classified_calls, "unclassified_calls": unclassified_calls,
        "tokens": total_tokens, "estimated_cost": round(total_cost, 2), "api_priced_calls": api_priced_calls,
        "local_model_calls": local_model_calls, "unknown_model_calls": unknown_cost_calls,
        "local_capable_calls": request_local_calls, "headline_share": round(request_headline_share, 1),
        "request_level_share": round(request_headline_share, 1), "step_local_capable_calls": step_local_calls,
        "step_headline_share": round(step_headline_share, 1), "step_dollar_share": round(step_dollar_share, 1),
        "step_local_cost": round(step_local_cost, 2), "tier": selected_tier, "context_distribution": context_distribution,
        "step_classes": {key: value["calls"] for key, value in step_by_class.items()},
        "subagent_calls": subagent_calls, "replay_good": replay_agreement, "replay_total": replay_n,
        "replay_agreement_percent": round(replay_percent, 1) if replay_n else None,
        "replay_wall_time_ms": round(statistics.median([float(row["wall_time_ms"]) for row in timed_rows]), 1) if timed_rows else 0.0,
        "top_classes": [key for key, _ in ranking],
        "context_warning": context_note if cache_context_warning else None,
    }
    conn.close()
    print(json.dumps(summary, indent=2))
    return str(output), share, summary


def _turn_messages(conn, target_call) -> list[dict[str, str]]:
    turns = conn.execute(
        "SELECT * FROM user_turns WHERE source=? AND session_id=? ORDER BY timestamp,id",
        (target_call["source"], target_call["session_id"]),
    ).fetchall()
    messages: list[dict[str, str]] = []
    for turn in turns:
        calls = turn_calls(conn, turn["id"])
        if turn["text"]:
            messages.append({"role": "user", "content": turn["text"]})
        for call in calls:
            if call["id"] == target_call["id"]:
                return messages
            facts = json_facts(call)
            summary = {"tool_names": [fact.get("name") for fact in facts], "response_chars": call["final_text_chars"]}
            messages.append({"role": "assistant", "content": json.dumps(summary, separators=(",", ":"))})
    return messages


def _candidate_facts(text: str, metrics: dict[str, object]) -> list[dict[str, object]]:
    tool_calls = metrics.get("tool_calls") or []
    facts: list[dict[str, object]] = []
    for call in tool_calls if isinstance(tool_calls, list) else []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        arguments = function.get("arguments", {}) if isinstance(function, dict) else {}
        facts.append(tool_fact(function.get("name", "unknown"), arguments))
    if facts:
        return facts
    try:
        parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
        items = parsed.get("tool_calls", [parsed]) if isinstance(parsed, dict) else []
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if isinstance(item, dict):
                facts.append(tool_fact(item.get("tool_name") or item.get("name"), item.get("arguments", item)))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return facts


def _dispatch_outcome(original, candidate_text: str, metrics: dict[str, object]) -> tuple[str, int, str]:
    expected = json_facts(original)
    candidate = _candidate_facts(candidate_text, metrics)
    if not expected or not candidate:
        return "failed", 0, "no comparable tool action was returned"
    expected_fact, candidate_fact = expected[0], candidate[0]
    same_name = str(expected_fact.get("name", "")).lower() == str(candidate_fact.get("name", "")).lower()
    expected_key = expected_fact.get("key_arg_hash") or expected_fact.get("command_first_word")
    candidate_key = candidate_fact.get("key_arg_hash") or candidate_fact.get("command_first_word")
    if same_name and expected_key == candidate_key:
        return "same action", 1, "same tool and key argument"
    if same_name:
        return "plausible", 0, "same tool with a different key argument"
    return "different", 0, "different tool"


def replay(steps: int, model: str, endpoint: str | None = None, tier: str | None = None) -> int:
    if endpoint:
        validate_local_endpoint(endpoint)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    conn = connect()
    _ensure_step_classes(conn)
    if tier is None:
        tier = tier_for_ram(int(detect_machine().get("ram_gb", 0) or 0))
    if tier not in {"16", "48", "128"}:
        raise ValueError("tier must be 16, 48, or 128")
    candidates: dict[str, list[tuple[object, object]]] = collections.defaultdict(list)
    for call in conn.execute("SELECT * FROM model_calls ORDER BY timestamp,id"):
        if not step_is_local_capable(call, tier):
            continue
        turn = conn.execute("SELECT * FROM user_turns WHERE id=?", (call["user_turn_id"],)).fetchone()
        if not turn or not turn["text"]:
            continue
        candidates[call["step_class"]].append((turn, call))
    selected: list[tuple[object, object]] = []
    classes = sorted(candidates)
    while len(selected) < steps and classes:
        made_progress = False
        for step_class in classes:
            if candidates[step_class]:
                selected.append(candidates[step_class].pop(0))
                made_progress = True
                if len(selected) >= steps:
                    break
        if not made_progress:
            break
    outcomes = collections.Counter()
    for turn, call in selected:
        started = time.perf_counter()
        step_class = call["step_class"]
        messages = _turn_messages(conn, call)
        known_tools = [fact.get("name") for fact in json_facts(call)]
        system = "You are a neutral local replay model. Reproduce the requested coding-agent step. Available tool names: " + ", ".join(map(str, known_tools))
        messages = messages + [{"role": "user", "content": json.dumps({"current_step": step_class})}]
        prompt = messages
        try:
            candidate, metrics = local_chat(model, prompt, system, endpoint)
            if step_class in {"read/search dispatch", "read-only shell dispatch", "mutating shell dispatch"}:
                outcome, agreement, reason = _dispatch_outcome(call, candidate, metrics)
                score = float(agreement)
                basis = "objective dispatch comparison"
            else:
                judge_prompt = json.dumps({
                    "original": call["response_text"],
                    "original_tool_facts": json_facts(call),
                    "candidate": candidate,
                    "step_class": step_class,
                })
                judged = ollama_chat(model, judge_prompt, "Return JSON only: outcome as as good, worse, or failed; score 0 to 1; reason.", endpoint)
                outcome, score, reason = "failed", 0.0, "judge response was not valid JSON"
                try:
                    parsed = json.loads(judged[judged.find("{"):judged.rfind("}") + 1])
                    if parsed.get("outcome") in {"as good", "worse", "failed"}:
                        outcome = parsed["outcome"]
                    score = max(0.0, min(1.0, float(parsed.get("score", 0))))
                    reason = str(parsed.get("reason", ""))[:500]
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
                agreement = int(outcome == "as good")
                basis = "judged, not measured"
            wall_time = float(metrics.get("wall_time_ms") or ((time.perf_counter() - started) * 1000))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:  # failed steps are evidence too
            outcome, agreement, score, reason, basis = "failed", 0, 0.0, str(exc)[:500], "replay request failed"
            wall_time = (time.perf_counter() - started) * 1000
        conn.execute(
            "INSERT INTO replay_results(user_turn_id,call_id,model,step_class,tier,context_tokens,outcome,agreement,score,reason,created_at,wall_time_ms,agreement_basis) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (turn["id"], call["id"], model, step_class, tier, call["context_tokens"], outcome, agreement, score, reason, now_iso(), wall_time, basis),
        )
        outcomes[outcome] += 1
    conn.commit()
    print(json.dumps({"tested": len(selected), "by_class": dict(collections.Counter(call["step_class"] for _, call in selected)), "outcomes": dict(outcomes), "model": model, "tier": tier}, indent=2))
    conn.close()
    return len(selected)
