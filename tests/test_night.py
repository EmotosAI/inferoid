import json
import os
import stat
import subprocess
from pathlib import Path

from inferoid import night

CFG = {
    "agents": {"primary": {"model": "gpt-5.6-luna"}, "local": {"model": "auto"}},
    "resources": {"moss": {"mem_gb": 20, "commands": ["scripts/gpu_eval.py --model moss"]}},
    "tasks": [{"id": "t1", "kind": "observe", "brief": "b.md", "uses": ["moss"]}],
}


def plan_for(ram, wired=0.0):
    return night.hardware_plan(CFG, {"ram_gb": ram, "wired_limit_gb": wired})


def test_hardware_plan_by_memory():
    p16 = plan_for(16)
    assert p16["local_model"] == "nemotron-3.5-bolt:e4b"
    p24 = plan_for(24)
    assert p24["local_model"] == "gpt-oss:20b" and p24["resource_modes"]["moss"] == "split"
    p24_raised = plan_for(24, wired=21)
    assert p24_raised["resource_modes"]["moss"] == "split"  # 14 + 20 + 2 > 21
    p48 = plan_for(48)
    assert p48["gpu_budget_gb"] == 36.0 and p48["resource_modes"]["moss"] == "inline"
    p128 = plan_for(128)
    assert p128["local_model"] == "gpt-oss:120b" and p128["resource_modes"]["moss"] == "inline"


def test_prompt_memory_rule_only_when_split():
    task = CFG["tasks"][0]
    split = night.build_prompt("do it", "local", task, CFG, plan_for(24), shadow=True)
    assert night.HEAVY_FILE in split and "HEAVY_JOBS_QUEUED" in split
    assert night.HEAVY_FILE not in night.build_prompt("do it", "local", task, CFG, plan_for(48), shadow=True)
    assert night.HEAVY_FILE not in night.build_prompt("do it", "primary", task, CFG, plan_for(24), shadow=False)


def test_handover_and_fallback():
    cfg = {**CFG, "handover": {"min_runs": 3}}
    state = {"tasks": {}, "kind_owner": {}, "history": []}
    for _ in range(3):
        state["history"] += [{"kind": "observe", "role": "primary", "passed": True},
                             {"kind": "observe", "role": "local", "passed": True}]
    assert night.update_owners(cfg, state) and state["kind_owner"]["observe"] == "local"
    state["history"] += [{"kind": "observe", "role": "local", "passed": False}] * 3
    assert night.update_owners(cfg, state) and state["kind_owner"]["observe"] == "primary"
    never = {**cfg, "handover": {"min_runs": 3, "never_local": ["observe"]}}
    state2 = {"tasks": {}, "kind_owner": {}, "history": [{"kind": "observe", "role": "local", "passed": True}] * 5}
    assert not night.update_owners(never, state2)


FAKE_CODEX = """#!/usr/bin/env python3
import os, sys, json
args = sys.argv[1:]
wt = os.getcwd()
local = "--oss" in args
resumed = "resume" in args
print(json.dumps({"type": "thread.started", "thread_id": "abc"}))
heavy = os.path.join(wt, ".nightshift-heavy.sh")
if local and not resumed and "HEAVY_JOBS_QUEUED" in args[-1]:
    open(heavy, "w").write("echo heavy-ran > heavy.out\\n")
    sys.exit(0)
open(os.path.join(wt, "result.txt"), "w").write("local\\n" if local else "primary\\n")
os.makedirs(os.path.join(wt, "renders"), exist_ok=True)
open(os.path.join(wt, "renders", "mix.wav"), "w").write("x")
open(os.path.join(wt, "NIGHT_RESULT.md"), "w").write("done\\n")
"""


def _exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_full_night_with_fake_agents(tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "pool").mkdir()
    (repo / "pool" / "song.flac").write_text("audio")
    (repo / "b.md").write_text("Make the thing.")
    (repo / ".nightshift.toml").write_text("""
[project]
base_branch = "main"
shared_paths = ["pool"]
listen_globs = ["renders/*.wav"]
[agents.primary]
model = "gpt-5.6-luna"
[agents.local]
model = "auto"
[resources.moss]
mem_gb = 20
[[tasks]]
id = "t1"
kind = "observe"
brief = "b.md"
uses = ["moss"]
check = "test -f result.txt && test -f NIGHT_RESULT.md"
[[tasks]]
id = "t2"
kind = "plan"
brief = "b.md"
after = "t1"
check = "git log --oneline | grep -q 't1 (primary)'"
[[tasks]]
id = "t0"
kind = "never"
brief = "b.md"
check = "false"
[[tasks]]
id = "t3"
kind = "plan"
brief = "b.md"
after = "t0"
""")
    (repo / ".gitignore").write_text("pool/\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], check=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "codex", FAKE_CODEX)
    _exe(bin_dir / "ollama", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SHADOW_CODEX", str(bin_dir / "codex"))
    monkeypatch.setattr(night, "machine_memory", lambda: {"ram_gb": 24.0, "wired_limit_gb": 0.0})
    monkeypatch.setattr(night, "ollama_ready", lambda model: None)
    monkeypatch.setattr(night, "keep_awake", lambda: None)

    morning = night.run_night(repo)
    text = morning.read_text()
    assert "| t1 | observe | primary |" in text and "| t1 | observe | local |" in text
    assert text.count("| pass |") == 4  # t1 primary+local, t2 primary+local (both start from t1's primary branch)
    assert "waiting for t0" in text
    assert "renders/mix.wav" in text
    runs = [json.loads(line) for line in (repo / ".nightshift" / "runs.jsonl").read_text().splitlines()]
    local = next(r for r in runs if r["role"] == "local")
    assert local["heavy_rounds"] == 1
    wt = Path(local["worktree"])
    assert (wt / "heavy.out").read_text().strip() == "heavy-ran"
    assert (wt / "pool").is_symlink()
    tracked = subprocess.run(["git", "-C", str(wt), "ls-files"], capture_output=True, text=True, check=True).stdout
    assert "pool" not in tracked and "result.txt" in tracked
    assert ".nightshift/" in (repo / ".git" / "info" / "exclude").read_text()
    state = night.load_state(repo)
    assert state["tasks"]["t1"]["status"] == "review" and state["tasks"]["t2"]["status"] == "review"
    assert "t3" not in state["tasks"]
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout == ""
