import json

from inferoid import claude, setup


def _env(tmp_path, monkeypatch, models, ram=24.0):
    monkeypatch.setenv("INFEROID_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "dotclaude"))
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "65536")
    monkeypatch.setattr(setup, "machine_memory", lambda: {"ram_gb": ram, "wired_limit_gb": 0.0})
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(setup, "_ollama_models", lambda: set(models))


def test_setup_defaults_on_24gb(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, [])
    pulled, lines = [], []
    raised = []
    rc = setup.run(tmp_path / "proj", answers={}, out=lines.append, pull=lambda m: pulled.append(m) or 0,
                   claude_version=lambda: "2.1.322 (Claude Code)", bin_dir=tmp_path / "bin",
                   gpu_raiser=lambda mb, ask: raised.append(mb) or ask("q"), reporter=lambda: "REPORT-OK")
    text = "\n".join(lines)
    assert rc == 0 and pulled == [setup.WORKLOADS["coding"]["models"][0][0]] and raised == []
    assert "Learn which one to pick: https://inferoid.com/lab/free-agents" in text and "65,536 tokens" in text
    assert "32,768-token context cap" in text and "Memory:" not in text
    assert "REPORT-OK" in text and "inferoid scan && inferoid report" in text
    assert "You're all set." in text and "90% of the 5-hour or weekly limit" in text and "all projects" in text
    cfg = json.loads(setup.config_path().read_text())
    assert cfg["model"] == setup.WORKLOADS["coding"]["models"][0][0] and cfg["threshold"] == 90.0 and cfg["takeover"] == "manual"
    s = json.loads((tmp_path / "dotclaude" / "settings.json").read_text())
    assert claude.MARK in s["statusLine"]["command"]
    assert claude.RULES_BEGIN in (tmp_path / "dotclaude" / "CLAUDE.md").read_text()
    assert "claude local" in (tmp_path / "bin" / "claude-local").read_text()


def test_setup_custom_answers(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, ["gpt-oss:20b"], ram=48.0)
    lines = []
    rc = setup.run(tmp_path, answers={"workload": "2", "model_choice": "2", "threshold": "85", "scope": "here", "takeover": "2", "report": "n"},
                   out=lines.append, pull=lambda m: 1, claude_version=lambda: "2.1.322", bin_dir=tmp_path / "bin",
                   reporter=lambda: "SHOULD-NOT-RUN")
    text = "\n".join(lines)
    assert rc == 0 and "already downloaded" in text and "Automatic switching is not in this version" in text
    assert "SHOULD-NOT-RUN" not in text and "would refuse" not in text  # gpt-oss 14 GB fits in 16 GB
    cfg = json.loads(setup.config_path().read_text())
    assert cfg["model"] == "gpt-oss:20b" and cfg["threshold"] == 85.0 and cfg["scope"] == str(tmp_path)
    assert (tmp_path / ".claude" / "settings.local.json").exists()


def test_setup_stops_on_old_claude_or_no_ollama(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, [setup.WORKLOADS["coding"]["models"][0][0]])
    lines = []
    assert setup.run(tmp_path, answers={}, out=lines.append, claude_version=lambda: "2.1.100", bin_dir=tmp_path / "b") == 1
    assert "claude update" in lines[-1]
    monkeypatch.setattr(setup, "_ollama_models", lambda: None)
    lines.clear()
    assert setup.run(tmp_path, answers={}, out=lines.append, claude_version=lambda: "2.1.322", bin_dir=tmp_path / "b") == 1
    assert "not running" in lines[-1]


def test_setup_reports_sandbox_state(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, [])
    lines = []
    rc = setup.run(tmp_path / "proj", answers={}, out=lines.append, pull=lambda m: 0,
                   claude_version=lambda: "2.1.322 (Claude Code)", bin_dir=tmp_path / "bin",
                   gpu_raiser=lambda mb, ask: ask("q"), reporter=lambda: "", sandbox_checker=lambda: "verified")
    text = "\n".join(lines)
    assert rc == 0 and "Sandbox         verified on this machine" in text and "local_agent" in text


def test_setup_learn_option_links_benchmark_then_picks(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, [])
    lines = []
    rc = setup.run(tmp_path, answers={"workload": "1", "model_choice": "4", "model_choice_after_learn": "2", "report": "n"},
                   out=lines.append, pull=lambda m: 0, claude_version=lambda: "2.1.322", bin_dir=tmp_path / "bin")
    text = "\n".join(lines)
    assert rc == 0 and "Open https://inferoid.com/lab/free-agents" in text
    cfg = json.loads(setup.config_path().read_text())
    assert cfg["model"] == setup.WORKLOADS["coding"]["models"][1][0]


def test_setup_coding_on_16gb_falls_back(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, [], ram=16.0)
    lines = []
    rc = setup.run(tmp_path, answers={"report": "n"}, out=lines.append, pull=lambda m: 0,
                   claude_version=lambda: "2.1.322", bin_dir=tmp_path / "bin")
    text = "\n".join(lines)
    assert rc == 0 and "need more memory than this machine has" in text
    assert json.loads(setup.config_path().read_text())["model"] == "nemotron-3.5-bolt:e4b"


def test_nvidia_gpu_memory_decides_fit(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, [], ram=64.0)
    monkeypatch.setattr(setup, "machine_memory", lambda: {"ram_gb": 64.0, "wired_limit_gb": 0.0, "vram_gb": 24.0, "gpu": "NVIDIA A10G"})
    assert claude.default_budget({"ram_gb": 64.0, "vram_gb": 24.0}) == 23.0
    assert claude.context_for({"ram_gb": 64.0, "vram_gb": 24.0}) == 32768
    lines = []
    rc = setup.run(tmp_path, answers={"report": "n"}, out=lines.append, pull=lambda m: 0,
                   claude_version=lambda: "2.1.322", bin_dir=tmp_path / "bin")
    text = "\n".join(lines)
    assert rc == 0 and "GPU NVIDIA A10G with 24 GB" in text
    assert json.loads(setup.config_path().read_text())["model"] == setup.WORKLOADS["coding"]["models"][0][0]
