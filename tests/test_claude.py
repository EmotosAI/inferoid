import json
import time
from pathlib import Path

from inferoid import claude


def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "shadowhome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "dotclaude"))


def test_install_keeps_user_settings_and_is_idempotent(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    proj = tmp_path / "proj"
    (proj / ".claude").mkdir(parents=True)
    (proj / "CLAUDE.md").write_text("# My project\nKeep tests green.\n")
    user = {"statusLine": {"type": "command", "command": "my-status.sh"},
            "hooks": {"PostToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "fmt.sh"}]}]}}
    (proj / ".claude" / "settings.local.json").write_text(json.dumps(user))
    claude.install(proj)
    claude.install(proj)  # twice: no duplicates
    s = json.loads((proj / ".claude" / "settings.local.json").read_text())
    assert claude.MARK in s["statusLine"]["command"]
    post = s["hooks"]["PostToolUse"]
    assert len(post) == 2 and post[0]["hooks"][0]["command"] == "fmt.sh"
    assert s["hooks"]["StopFailure"][0]["matcher"] == "rate_limit"
    md = (proj / "CLAUDE.md").read_text()
    assert md.startswith("# My project") and md.count(claude.RULES_BEGIN) == 1 and "HANDOFF.md" in md
    claude.uninstall(proj)
    s = json.loads((proj / ".claude" / "settings.local.json").read_text())
    assert s["statusLine"]["command"] == "my-status.sh"
    assert s["hooks"] == {"PostToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "fmt.sh"}]}]}
    assert claude.RULES_BEGIN not in (proj / "CLAUDE.md").read_text()


def _status_input(pct5, pct7, resets=None):
    resets = resets or int(time.time()) + 3600
    return json.dumps({"session_id": "s1", "model": {"display_name": "Opus 5.5"},
                       "rate_limits": {"five_hour": {"used_percentage": pct5, "resets_at": resets},
                                       "seven_day": {"used_percentage": pct7, "resets_at": resets + 86400}}})


def test_statusline_records_usage_and_hook_warns_once(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    claude.install(tmp_path)
    out = claude.statusline(_status_input(42, 18))
    assert "5h 42%" in out and "7d 18%" in out
    assert claude.hook("UserPromptSubmit", json.dumps({"session_id": "s1"})) == ""
    claude.statusline(_status_input(91, 18))
    first = claude.hook("PostToolUse", json.dumps({"session_id": "s1"}))
    ctx = json.loads(first)["hookSpecificOutput"]
    assert ctx["hookEventName"] == "PostToolUse" and "91%" in ctx["additionalContext"] and "HANDOFF.md" in ctx["additionalContext"]
    assert claude.hook("UserPromptSubmit", json.dumps({"session_id": "s1"})) == ""  # once per window and level
    claude.statusline(_status_input(98, 18))
    assert "98%" in claude.hook("UserPromptSubmit", json.dumps({"session_id": "s1"}))  # 97% level warns again


def test_expired_window_is_ignored(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    claude.statusline(_status_input(99, 10, resets=int(time.time()) - 10))
    pct, window, _ = claude.current_usage()
    assert window == "seven_day" and pct == 10


def test_statusline_chains_previous(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    proj = tmp_path / "p"
    (proj / ".claude").mkdir(parents=True)
    (proj / ".claude" / "settings.local.json").write_text(json.dumps({"statusLine": {"type": "command", "command": "echo mine"}}))
    claude.install(proj)
    assert claude.statusline(_status_input(5, 5)) == "mine"


def test_stopfailure_records_limit(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    claude.hook("StopFailure", json.dumps({"session_id": "s9", "cwd": "/x"}))
    hit = json.loads((Path(tmp_path / "shadowhome" / "claude" / "limit-hit.json")).read_text())
    assert hit["session_id"] == "s9"


def test_local_command_and_model_pick(tmp_path):
    args, env, mode = claude.local_command(tmp_path, "nemotron-3.5-lightning:30b-a3b", resume=False)
    assert mode == "plain" and args[:3] == ["claude", "--model", "nemotron-3.5-lightning:30b-a3b"]
    picker = json.loads(args[args.index("--settings") + 1])["modelPicker"]
    assert picker["replaceBuiltInOptions"] and picker["options"][0]["model"] == "nemotron-3.5-lightning:30b-a3b"
    assert env["ANTHROPIC_BASE_URL"].startswith("http") and env["ANTHROPIC_AUTH_TOKEN"] == "ollama" and env["ANTHROPIC_API_KEY"] == ""
    (tmp_path / "HANDOFF.md").write_text("# HANDOFF\n1. [ ] step\n")
    args, _, mode = claude.local_command(tmp_path, "m", resume=False)
    assert mode == "handoff" and "HANDOFF.md" in args[-1]
    assert claude.local_command(tmp_path, "m", resume=True)[2] == "resume"
    assert claude.pick_model({"ram_gb": 24})[0] == "nemotron-3.5-lightning:30b-a3b"
    assert claude.pick_model({"ram_gb": 16})[0] == "nemotron-3.5-bolt:e4b"


def test_memory_hint_on_24gb(monkeypatch):
    monkeypatch.setattr(claude, "_ollama", lambda path: {"models": [{"name": "nemotron-3.5-lightning:30b-a3b"}]})
    monkeypatch.setattr(claude.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "65536")
    notes = claude.preflight("nemotron-3.5-lightning:30b-a3b", 19.0, {"ram_gb": 24.0, "wired_limit_gb": 0.0})
    assert len(notes) == 1 and notes[0].startswith("memory:") and "iogpu.wired_limit_mb=21504" in notes[0]
    assert claude.gpu_limit_needed(19.0, {"ram_gb": 24.0, "wired_limit_gb": 0.0}) == 21504
    assert claude.gpu_limit_needed(19.0, {"ram_gb": 48.0, "wired_limit_gb": 0.0}) == 0
    assert claude.preflight("nemotron-3.5-lightning:30b-a3b", 19.0, {"ram_gb": 24.0, "wired_limit_gb": 21.0}) == []
