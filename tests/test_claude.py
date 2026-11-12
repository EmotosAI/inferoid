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
    args, env, mode = claude.local_command(tmp_path, "nemotron-3.5-lightning:30b-a3b-mlx", resume=False)
    assert mode == "plain" and args[:3] == ["claude", "--model", "nemotron-3.5-lightning:30b-a3b-mlx"]
    picker = json.loads(args[args.index("--settings") + 1])["modelPicker"]
    assert json.loads(args[args.index("--settings") + 1])["disableAllHooks"] is True
    assert picker["replaceBuiltInOptions"] and picker["options"][0]["model"] == "nemotron-3.5-lightning:30b-a3b-mlx"
    assert "--strict-mcp-config" in args and args[args.index("--tools") + 1] == claude.LEAN_TOOLS
    assert "--strict-mcp-config" not in claude.local_command(tmp_path, "m", resume=False, full=True)[0]
    assert env["ANTHROPIC_BASE_URL"].startswith("http") and env["ANTHROPIC_AUTH_TOKEN"] == "ollama" and env["ANTHROPIC_API_KEY"] == ""
    (tmp_path / "HANDOFF.md").write_text("# HANDOFF\n1. [ ] step\n")
    args, _, mode = claude.local_command(tmp_path, "m", resume=False)
    assert mode == "handoff" and "HANDOFF.md" in args[-1]
    assert claude.local_command(tmp_path, "m", resume=True)[2] == "resume"
    # Default picks must fit macOS's default GPU budget (24 GB Mac froze with Nemotron on 2026-11-11).
    assert claude.pick_model({"ram_gb": 24})[0] == "nemotron-3.5-bolt:e4b"
    assert claude.pick_model({"ram_gb": 16})[0] == "nemotron-3.5-bolt:e4b"
    assert claude.pick_model({"ram_gb": 48})[0] == "nemotron-3.5-lightning:30b-a3b-mlx"
    assert claude.context_for({"ram_gb": 24}) == 32768 and claude.context_for({"ram_gb": 48}) == 65536
    assert claude.context_for({"ram_gb": 128}) == 131072


def test_memory_guards_on_24gb(monkeypatch):
    lightning = "nemotron-3.5-lightning:30b-a3b-mlx"
    monkeypatch.setattr(claude, "_ollama", lambda path: {"models": [{"name": lightning}, {"name": "nemotron-3.5-bolt:e4b"}]})
    monkeypatch.setattr(claude.shutil, "which", lambda name: "/usr/bin/" + name)
    mac24 = {"ram_gb": 24.0, "wired_limit_gb": 0.0}
    # 19 GB model + context + runtime would leave macOS < 4 GB: refuse, suggest the safe model, never offer sudo.
    assert claude.gpu_limit_needed(lightning, 19.0, mac24) == 0
    problem = claude.memory_problem(lightning, 19.0, mac24)
    assert "less than 4 GB" in problem and "nemotron-3.5-bolt:e4b" in problem and "sysctl" not in problem
    # Even with the limit already raised to 21 GB, it does not fit with the context cap.
    assert claude.memory_problem(lightning, 19.0, {"ram_gb": 24.0, "wired_limit_gb": 21.0})
    # Ollama context above the tier cap is flagged as a memory problem.
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "65536")
    notes = claude.preflight("nemotron-3.5-bolt:e4b", 7.5, mac24)
    assert len(notes) == 1 and notes[0].startswith("memory:") and "above the safe 32,768" in notes[0]
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "32768")
    assert claude.preflight("nemotron-3.5-bolt:e4b", 7.5, mac24) == []
    # On a 32 GB Mac a modest overshoot may be allowed by opt-in, keeping >= 4 GB for macOS.
    mac32 = {"ram_gb": 32.0, "wired_limit_gb": 0.0}
    assert claude.gpu_limit_needed("gpt-oss:20b", 23.0, mac32) > 0 or claude.memory_problem("gpt-oss:20b", 23.0, mac32) == ""


def test_ensure_loaded_restarts_once_on_memory_error():
    calls, restarts = [], []
    errs = iter(["model requires 17.7 GiB but only 15.5 GiB are available", ""])
    assert claude.ensure_loaded("m", restart=lambda: restarts.append(1) or True, load=lambda m: calls.append(m) or next(errs)) == ""
    assert len(calls) == 2 and len(restarts) == 1
    assert claude.ensure_loaded("m", restart=lambda: False, load=lambda m: "model requires 17.7 GiB") == "model requires 17.7 GiB"
    assert claude.ensure_loaded("m", restart=lambda: 1 / 0, load=lambda m: "") == ""  # no restart when it loads


def test_saved_old_default_moves_to_mlx_build(tmp_path, monkeypatch, capsys):
    _env(tmp_path, monkeypatch)
    (tmp_path / "shadowhome" / "config.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "shadowhome" / "config.json").write_text(json.dumps({"model": "nemotron-3.5-lightning:30b-a3b"}))
    monkeypatch.setattr(claude, "_ollama", lambda path: {"models": []})
    assert claude.local(tmp_path, None, resume=False, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "nemotron-3.5-lightning:30b-a3b-mlx" in out and "faster build" in out


def test_probe_context_picks_largest_measured_fit():
    mac24 = {"ram_gb": 24.0, "wired_limit_gb": 0.0}  # default budget ~16.1 GB, minus 2 GB margin
    sizes = {131072: 17.5, 98304: 15.2, 65536: 12.9, 49152: 11.7, 32768: 10.6}
    current = {}
    calls = []

    def gen(model, num_ctx, keep_alive):
        calls.append((num_ctx, keep_alive))
        if num_ctx:
            current["n"] = num_ctx
        return ""

    got = claude.probe_context("m", mac24, gen=gen, size=lambda m: sizes[current["n"]], loaded=lambda m: current["n"],
                               log=lambda s: None)
    assert got == 65536  # 98K measured 15.2 GB > 14.1 GB allowed
    assert (None, 0) in calls  # unloads between probes
    # A model whose real maximum is lower reports the context it actually loaded.
    got = claude.probe_context("m", {"ram_gb": 64.0}, candidates=(131072,), gen=gen, size=lambda m: 9.0,
                               loaded=lambda m: 40960, log=lambda s: None)
    assert got == 40960


def test_resolve_context_numbers_and_errors(monkeypatch):
    monkeypatch.setattr(claude, "probe_context", lambda model, memory, candidates=(), log=None: candidates[0] if candidates and candidates[0] <= 65536 else 32768)
    mac = {"ram_gb": 24.0}
    assert claude.resolve_context("m", mac, "64k") == 65536
    assert claude.resolve_context("m", mac, "131072") == 0
    assert claude.resolve_context("m", mac, "131072", force=True) == 131072
    assert claude.resolve_context("m", mac, "lots") == 0
    assert claude.resolve_context("m", mac, None) == 32768


def test_probe_distrusts_size_below_weights(monkeypatch):
    monkeypatch.setattr(claude, "weights_gb", lambda m: 7.2)
    monkeypatch.setattr(claude, "kv_estimate_gb", lambda m, n: 8.0 if n > 65536 else 3.0)
    cur = {}

    def gen(model, num_ctx, keep_alive):
        if num_ctx:
            cur["n"] = num_ctx
        return ""

    # Ollama reports 1.1 GB (impossible: weights are 7.2 GB) -> use weights + KV estimate instead.
    # Ollama reports 1.1 GB and system memory cannot be read -> worst-case estimate.
    got = claude.probe_context("gemma", {"ram_gb": 24.0}, gen=gen, size=lambda m: 1.1, loaded=lambda m: cur["n"],
                               log=lambda s: None, used=lambda: 0.0)
    assert got == 65536  # 7.2 + 8.0 + 2 > 14.1 at 96K/128K; 7.2 + 3.0 + 2 fits at 64K
    # The answer does not depend on noisy system-memory readings.
    noisy = iter([100.0, 117.2, 100.0, 112.6] * 5)
    got = claude.probe_context("gemma", {"ram_gb": 24.0}, gen=gen, size=lambda m: 1.1, loaded=lambda m: cur["n"],
                               log=lambda s: None, used=lambda: next(noisy))
    assert got == 65536


def test_warm_up_does_not_override_tag_context(monkeypatch):
    sent = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"{}"

    def fake_urlopen(req, timeout=0):
        sent["body"] = json.loads(req.data)
        return R()

    monkeypatch.setattr(claude.urllib.request, "urlopen", fake_urlopen)
    assert claude.warm_up("m-ctx128k") == "" and "options" not in sent["body"]


def test_loaded_footprint_counts_mmapped_weights():
    assert claude.loaded_footprint_gb(7.2, 13.5) == 13.5  # the change already includes the weights
    assert claude.loaded_footprint_gb(11.0, 1.6) == 12.6  # weights sat in the file cache; add them back
    assert claude.loaded_footprint_gb(11.0, -0.4) == 11.0


def test_measured_context_is_saved_and_reused(monkeypatch, tmp_path):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(claude, "probe_context", lambda model, memory, candidates=(), log=None: calls.append(1) or 98304)
    mac = {"ram_gb": 24.0}
    assert claude.resolve_context("m:latest", mac, "max") == 98304
    assert claude.resolve_context("m", mac, "max") == 98304 and len(calls) == 1  # reused, not measured again
    assert claude.resolve_context("m", mac, "remeasure") == 98304 and len(calls) == 2
    assert claude.resolve_context("m", mac, None) == 98304  # sessions without --context use it too


def test_local_env_caps_reply_tokens():
    _, env, _ = claude.local_command(Path("."), "m", False, False, 32768)
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"


def test_over_limit_run_lowers_saved_context(monkeypatch, tmp_path):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path))
    mac = {"ram_gb": 24.0}
    claude.save_context("m", mac, 131072)
    assert claude.lower_saved_context("m", mac, 131072) == 98304
    assert claude.saved_context("m", mac) == 98304


def test_kv_estimate_counts_only_attention_layers_in_hybrids(monkeypatch):
    info = {"qwen35moe.block_count": 40, "qwen35moe.full_attention_interval": 4, "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.key_length": 256, "qwen35moe.attention.value_length": 256}

    class R:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"model_info": info}).encode()

    monkeypatch.setattr(claude.urllib.request, "urlopen", lambda *a, **k: R())
    gb = claude.kv_estimate_gb("q", 65536)
    assert 1.2 < gb < 1.5  # 10 attention layers, not 40


def test_local_session_drops_supervisor_provider_settings(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "secret")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "us.anthropic.x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    _, env, _ = claude.local_command(Path("."), "m", False, False, 32768)
    for k in ("CLAUDE_CODE_USE_BEDROCK", "AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_DEFAULT_SONNET_MODEL", "OPENROUTER_API_KEY"):
        assert k not in env
    assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1")


def test_kv_estimate_sliding_window_layers(monkeypatch):
    info = {"gemma4.block_count": 30, "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2] * 5,
            "gemma4.attention.sliding_window_pattern": [True] * 5 + [False] + [True] * 5 + [False] + [True] * 5 + [False]
            + [True] * 5 + [False] + [True] * 5 + [False],
            "gemma4.attention.sliding_window": 1024, "gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
            "gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256}

    class R:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"model_info": info}).encode()

    monkeypatch.setattr(claude.urllib.request, "urlopen", lambda *a, **k: R())
    gb = claude.kv_estimate_gb("g", 65536)
    assert 1.4 < gb < 1.7  # 5 full layers at 64K + 25 windowed layers at 1K, not 30 full layers (~40 GB)
