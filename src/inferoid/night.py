"""Night shift: run project tasks overnight with a paid primary agent and a free local shadow agent.

Week 1 the primary does the work and the local model shadows it on a separate git worktree.
Each task kind moves to the local model once it has passed its checks as often as the primary.
Memory handling adapts to the machine: when the local model and a task's heavy tools cannot
fit in GPU memory together, the local agent queues heavy commands, the runner unloads the
model, runs them, and resumes the agent.

Shadow itself makes no network calls; the agents you configure (for example Codex) may.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

CONFIG_NAME = ".nightshift.toml"
STATE_DIR = ".nightshift"
HEAVY_FILE = ".nightshift-heavy.sh"
HEADROOM_GB = 2.0

# Local agent model by unified memory. Memory figures are 4-bit weights plus runtime at ~32K context.
LOCAL_MODELS = [
    # (min_ram_gb, ollama model, approx GPU GB)
    (96, "gpt-oss:120b", 66.0),
    (24, "gpt-oss:20b", 14.0),
    (0, "nemotron-3.5-bolt:e4b", 7.5),
]


# ---------------------------------------------------------------- hardware

def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_memory() -> dict[str, float]:
    """Total RAM and the share the GPU may use (macOS wired limit, else Apple's default share)."""
    ram = 0.0
    wired = 0.0
    if platform.system() == "Darwin":
        raw = _sysctl("hw.memsize")
        ram = int(raw) / 1024**3 if raw.isdigit() else 0.0
        limit = _sysctl("iogpu.wired_limit_mb")
        wired = int(limit) / 1024 if limit.isdigit() else 0.0
    else:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) / 1024**2
                    break
        except OSError:
            pass
    out = {"ram_gb": round(ram, 1), "wired_limit_gb": round(wired, 1)}
    vram, gpu = nvidia_gpu()
    if vram:
        out["vram_gb"], out["gpu"] = vram, gpu
    return out


def nvidia_gpu() -> tuple[float, str]:
    """Largest NVIDIA GPU's memory in GB and its name (Linux, WSL2, Windows), or (0, "") without one."""
    if platform.system() == "Darwin" or not shutil.which("nvidia-smi"):
        return 0.0, ""
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0, ""
    best = (0.0, "")
    for line in text.splitlines():
        mib, _, name = line.partition(",")
        try:
            best = max(best, (round(float(mib) / 1024, 1), name.strip()))
        except ValueError:
            continue
    return best


def gpu_budget_gb(ram_gb: float, wired_limit_gb: float = 0.0) -> float:
    if wired_limit_gb > 0:
        return wired_limit_gb
    return round(ram_gb * (0.67 if ram_gb <= 36 else 0.75), 1)


def pick_local_model(ram_gb: float) -> tuple[str, float]:
    for min_ram, model, mem in LOCAL_MODELS:
        if ram_gb >= min_ram:
            return model, mem
    return LOCAL_MODELS[-1][1], LOCAL_MODELS[-1][2]


def hardware_plan(cfg: dict[str, Any], mem: dict[str, float] | None = None) -> dict[str, Any]:
    mem = mem or machine_memory()
    budget = gpu_budget_gb(mem["ram_gb"], mem.get("wired_limit_gb", 0.0))
    local_cfg = cfg.get("agents", {}).get("local", {})
    model = local_cfg.get("model", "auto")
    if model == "auto":
        model, model_mem = pick_local_model(mem["ram_gb"])
    else:
        model_mem = float(local_cfg.get("mem_gb", 14.0))
    resources = {name: float(r.get("mem_gb", 0)) for name, r in cfg.get("resources", {}).items()}
    modes = {}
    advice = []
    for name, need in resources.items():
        together = model_mem + need + HEADROOM_GB
        if together <= budget:
            modes[name] = "inline"
        else:
            modes[name] = "split"
            if platform.system() == "Darwin" and mem.get("wired_limit_gb", 0) == 0 and together <= mem["ram_gb"] - 3:
                advice.append(f"{name}: raising the GPU limit (sudo sysctl iogpu.wired_limit_mb={int((mem['ram_gb'] - 3) * 1024)}) would let it run inline")
        if need + HEADROOM_GB > max(budget, mem["ram_gb"] - 3):
            advice.append(f"{name} needs about {need:g} GB, more than this machine can give the GPU; it may fall back to CPU")
    if model_mem + HEADROOM_GB > budget:
        advice.append(f"local model {model} (~{model_mem:g} GB) does not fit the GPU budget of {budget:g} GB")
    return {"ram_gb": mem["ram_gb"], "gpu_budget_gb": budget, "local_model": model, "local_model_gb": model_mem,
            "resource_modes": modes, "advice": advice}


def describe_plan(plan: dict[str, Any]) -> str:
    lines = [f"Memory: {plan['ram_gb']:g} GB RAM, about {plan['gpu_budget_gb']:g} GB usable by the GPU.",
             f"Local agent model: {plan['local_model']} (~{plan['local_model_gb']:g} GB)."]
    for name, mode in plan["resource_modes"].items():
        lines.append(f"  {name}: {'runs alongside the local model' if mode == 'inline' else 'model is unloaded while it runs (split mode)'}")
    lines += [f"  note: {a}" for a in plan["advice"]]
    return "\n".join(lines)


# ---------------------------------------------------------------- config and state

def load_config(project: Path) -> dict[str, Any]:
    path = project / CONFIG_NAME
    if not path.exists():
        raise SystemExit(f"no {CONFIG_NAME} in {project}")
    cfg = tomllib.loads(path.read_text())
    ids = [t["id"] for t in cfg.get("tasks", [])]
    if len(ids) != len(set(ids)):
        raise SystemExit("task ids must be unique")
    for task in cfg.get("tasks", []):
        for key in ("id", "kind", "brief"):
            if key not in task:
                raise SystemExit(f"task {task.get('id', '?')} is missing '{key}'")
    return cfg


def load_state(project: Path) -> dict[str, Any]:
    path = project / STATE_DIR / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"tasks": {}, "kind_owner": {}, "history": []}


def save_state(project: Path, state: dict[str, Any]) -> None:
    d = project / STATE_DIR
    d.mkdir(exist_ok=True)
    (d / "state.json").write_text(json.dumps(state, indent=2))


def kind_stats(history: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    rows = [h for h in history if h["kind"] == kind]
    def rate(role: str) -> tuple[int, float]:
        r = [h for h in rows if h["role"] == role]
        return len(r), (sum(h["passed"] for h in r) / len(r)) if r else 0.0
    n_p, p_rate = rate("primary")
    n_l, l_rate = rate("local")
    return {"primary_runs": n_p, "primary_pass": p_rate, "local_runs": n_l, "local_pass": l_rate}


def update_owners(cfg: dict[str, Any], state: dict[str, Any]) -> list[str]:
    hand = cfg.get("handover", {})
    min_runs = int(hand.get("min_runs", 5))
    gap = float(hand.get("max_gap", 0.0))
    never_local = set(hand.get("never_local", []))
    changes = []
    for kind in sorted({t["kind"] for t in cfg.get("tasks", [])}):
        owner = state["kind_owner"].get(kind, "primary")
        s = kind_stats(state["history"], kind)
        if owner == "primary" and kind not in never_local and s["local_runs"] >= min_runs \
                and s["local_pass"] >= s["primary_pass"] - gap:
            state["kind_owner"][kind] = "local"
            changes.append(f"{kind}: moved to the local model ({s['local_pass']:.0%} vs {s['primary_pass']:.0%} over {s['local_runs']} runs)")
        elif owner == "local":
            recent = [h for h in state["history"] if h["kind"] == kind and h["role"] == "local"][-3:]
            if len(recent) == 3 and not any(h["passed"] for h in recent):
                state["kind_owner"][kind] = "primary"
                changes.append(f"{kind}: back to the primary after 3 local failures in a row")
    return changes


# ---------------------------------------------------------------- prompts

def build_prompt(brief: str, role: str, task: dict[str, Any], cfg: dict[str, Any], plan: dict[str, Any], shadow: bool) -> str:
    rules = [
        "You are working unattended overnight in a git worktree created for this task.",
        "Never push, never merge, never delete branches. Do not modify files under the shared paths except the declared caches.",
        "Follow the project's own instructions files. Finish by writing NIGHT_RESULT.md in the worktree root: what you did, what you checked, open questions.",
    ]
    caches = cfg.get("project", {}).get("cache_paths", [])
    if caches:
        rules.append("Writable shared caches: " + ", ".join(caches) + ". Reuse cached results when present.")
    if shadow:
        rules.append("This is a shadow run for comparison; a different agent is doing the same task. Work as if you were the only one.")
    if role == "local":
        for name in task.get("uses", []):
            if plan["resource_modes"].get(name) == "split":
                cmds = cfg["resources"][name].get("commands", [])
                rules.append(
                    f"Memory rule: this machine cannot hold you and '{name}' at the same time. Do not run commands that load it"
                    + (f" (for example: {', '.join(cmds)})" if cmds else "")
                    + f". Instead append each such command as one line to {HEAVY_FILE} in the worktree root, then stop and reply"
                    " HEAVY_JOBS_QUEUED. The runner frees memory, runs them, and resumes you with their output."
                    + (" Prefer cached results if they already exist." if shadow else ""))
    return "\n".join(f"- {r}" for r in rules) + "\n\n# Task\n\n" + brief.strip() + "\n"


# ---------------------------------------------------------------- git

def git(project: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", str(project), *args], check=check, capture_output=True, text=True).stdout.strip()


def make_worktree(project: Path, cfg: dict[str, Any], branch: str, path: Path, base: str | None = None) -> None:
    base = base or cfg.get("project", {}).get("base_branch") or git(project, "rev-parse", "--abbrev-ref", "HEAD")
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(project, "worktree", "add", "-q", "-B", branch, str(path), base)
    for rel in cfg.get("project", {}).get("shared_paths", []):
        src = project / rel
        dst = path / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())


def commit_worktree(path: Path, message: str, shared_paths: list[str]) -> str:
    before = git(path, "rev-parse", "HEAD")
    excludes = [f":(exclude){rel}" for rel in shared_paths]
    subprocess.run(check=False, args=["git", "-C", str(path), "add", "-A", "--", ".", *excludes], capture_output=True)
    subprocess.run(check=False, args=["git", "-C", str(path), "-c", "user.name=nightshift", "-c", "user.email=nightshift@localhost",
                    "commit", "-q", "-m", message], capture_output=True)
    if git(path, "rev-parse", "HEAD") == before:
        return "no changes"
    return git(path, "diff", "--shortstat", before, "HEAD", check=False)


def exclude_state_dir(project: Path) -> None:
    """Keep .nightshift/ (worktrees, logs, state) out of the user's git status without editing tracked files."""
    info = Path(git(project, "rev-parse", "--git-common-dir"))
    info = (info if info.is_absolute() else project / info) / "info" / "exclude"
    info.parent.mkdir(parents=True, exist_ok=True)
    text = info.read_text() if info.exists() else ""
    if f"/{STATE_DIR}/" not in text.splitlines():
        info.write_text(text + ("" if text.endswith("\n") or not text else "\n") + f"/{STATE_DIR}/\n")


# ---------------------------------------------------------------- agents

def codex_bin() -> str:
    return os.environ.get("SHADOW_CODEX", "codex")


def run_agent(role: str, cfg: dict[str, Any], plan: dict[str, Any], wt: Path, prompt: str, log: Path,
              resume_id: str | None = None, timeout: int = 3 * 3600) -> tuple[int, str | None]:
    agent = cfg.get("agents", {}).get(role, {})
    cmd = [codex_bin(), "exec"]
    if resume_id:
        cmd += ["resume", resume_id]
    if role == "local":
        cmd += ["--oss", "--local-provider", agent.get("provider", "ollama"), "-m", plan["local_model"]]
    else:
        cmd += ["-m", agent.get("model", "gpt-5.6-luna")]
    if not resume_id:
        cmd += ["-C", str(wt), "-s", "workspace-write", "--skip-git-repo-check"]
        for rel in cfg.get("project", {}).get("cache_paths", []):
            target = (wt / rel)
            if target.exists():
                cmd += ["--add-dir", str(target.resolve())]
    cmd += ["--json", "-o", str(wt / ".nightshift-last.md"), prompt]
    with log.open("a") as fh:
        fh.write(f"\n### {role} {' '.join(cmd[:8])} ...\n")
        fh.flush()
        try:
            p = subprocess.run(check=False, args=cmd, cwd=wt, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, stdin=subprocess.DEVNULL)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = 124
    session = None
    try:
        for line in log.read_text().splitlines():
            if '"thread_id"' in line or '"session_id"' in line:
                obj = json.loads(line)
                session = obj.get("thread_id") or obj.get("session_id") or session
    except (OSError, json.JSONDecodeError):
        pass
    return code, session


def ollama_stop(model: str) -> None:
    subprocess.run(check=False, args=["ollama", "stop", model], capture_output=True)


def run_heavy(wt: Path, log: Path, timeout: int = 4 * 3600) -> bool:
    heavy = wt / HEAVY_FILE
    if not heavy.exists() or not heavy.read_text().strip():
        return False
    with log.open("a") as fh:
        fh.write(f"\n### heavy jobs\n{heavy.read_text()}\n")
        fh.flush()
        try:
            subprocess.run(check=False, args=["bash", str(heavy)], cwd=wt, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            fh.write("heavy jobs timed out\n")
    done = wt / (HEAVY_FILE + ".done")
    with done.open("a") as fh:
        fh.write(heavy.read_text())
    heavy.unlink()
    return True


def run_check(task: dict[str, Any], wt: Path) -> tuple[bool, str]:
    check = task.get("check")
    if not check:
        return False, "no check defined: needs review"
    try:
        p = subprocess.run(check=False, args=["bash", "-lc", check], cwd=wt, capture_output=True, text=True, timeout=1800)
        return p.returncode == 0, (p.stdout + p.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return False, "check timed out"


def dependency_branch(state: dict[str, Any], task: dict[str, Any]) -> tuple[bool, str | None]:
    """A task with `after` starts from the newest passing branch of that task (or waits)."""
    dep = task.get("after")
    if not dep:
        return True, None
    if state["tasks"].get(dep, {}).get("status") == "done":
        return True, None  # merged into the base branch by the user
    passing = [h for h in state["history"] if h["task"] == dep and h["passed"] and h.get("branch")]
    if not passing:
        return False, None
    best = [h for h in passing if h["role"] == "primary"] or passing
    return True, best[-1]["branch"]


def run_role(project: Path, cfg: dict[str, Any], plan: dict[str, Any], task: dict[str, Any], role: str,
             night: str, shadow: bool, base: str | None = None) -> dict[str, Any]:
    branch = f"nightshift/{night}/{task['id']}/{role}"
    wt = project / STATE_DIR / "worktrees" / night / task["id"] / role
    make_worktree(project, cfg, branch, wt, base)
    log = project / STATE_DIR / "logs" / night / f"{task['id']}-{role}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    brief = (project / task["brief"]).read_text()
    prompt = build_prompt(brief, role, task, cfg, plan, shadow)
    t0 = time.time()
    if role == "primary" and task.get("uses"):
        ollama_stop(plan["local_model"])  # free GPU memory for the task's own heavy tools
    code, session = run_agent(role, cfg, plan, wt, prompt, log)
    rounds = 0
    while role == "local" and rounds < 3 and (wt / HEAVY_FILE).exists():
        ollama_stop(plan["local_model"])
        if not run_heavy(wt, log):
            break
        rounds += 1
        code, session = run_agent(role, cfg, plan, wt, "The queued heavy jobs finished; their output is in the log and files. Continue the task.",
                                  log, resume_id=session or "--last")
    passed, check_out = run_check(task, wt)
    stat = commit_worktree(wt, f"nightshift {night}: {task['id']} ({role})", cfg.get("project", {}).get("shared_paths", []))
    listen = []
    for pattern in cfg.get("project", {}).get("listen_globs", []):
        listen += [str(p.relative_to(wt)) for p in wt.glob(pattern) if p.is_file() and p.stat().st_mtime >= t0]
    return {"night": night, "task": task["id"], "kind": task["kind"], "role": role, "shadow": shadow,
            "model": plan["local_model"] if role == "local" else cfg.get("agents", {}).get("primary", {}).get("model", "gpt-5.6-luna"),
            "exit": code, "passed": passed, "check_tail": check_out, "minutes": round((time.time() - t0) / 60, 1),
            "branch": branch, "worktree": str(wt), "diff": stat or "", "heavy_rounds": rounds, "listen": sorted(listen)}


# ---------------------------------------------------------------- night

def keep_awake() -> subprocess.Popen | None:
    if platform.system() == "Darwin" and shutil.which("caffeinate"):
        return subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
    return None


def ollama_ready(model: str) -> str | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
    except OSError:
        return "Ollama is not running on 127.0.0.1:11434"
    if not any(n == model or n.split(":")[0] == model.split(":")[0] and n.endswith(model.split(":")[-1]) for n in names):
        return f"local model {model} is not pulled (ollama pull {model})"
    return None


def run_night(project: Path, max_tasks: int = 0, dry_run: bool = False) -> Path:
    cfg = load_config(project)
    plan = hardware_plan(cfg)
    state = load_state(project)
    night = dt.datetime.now().strftime("%Y-%m-%d")
    pending = [t for t in cfg["tasks"] if state["tasks"].get(t["id"], {}).get("status", "pending") == "pending"]
    if max_tasks:
        pending = pending[:max_tasks]
    problem = ollama_ready(plan["local_model"]) if not dry_run else None
    if not dry_run:
        exclude_state_dir(project)
    results: list[dict[str, Any]] = []
    awake = None if dry_run else keep_awake()
    try:
        for task in pending:
            ready, base = dependency_branch(state, task)
            if not ready:
                results.append({"night": night, "task": task["id"], "kind": task["kind"], "role": "-", "shadow": False,
                                "passed": False, "skipped": f"waiting for {task['after']}"})
                continue
            owner = state["kind_owner"].get(task["kind"], "primary")
            roles = [("primary", False), ("local", True)] if owner == "primary" else [("local", False)]
            for role, shadow in roles:
                if role == "local" and problem:
                    results.append({"night": night, "task": task["id"], "kind": task["kind"], "role": role, "shadow": shadow,
                                    "passed": False, "skipped": problem})
                    continue
                if dry_run:
                    results.append({"night": night, "task": task["id"], "kind": task["kind"], "role": role, "shadow": shadow,
                                    "passed": False, "dry_run": True,
                                    "prompt": build_prompt((project / task["brief"]).read_text(), role, task, cfg, plan, shadow)})
                    continue
                r = run_role(project, cfg, plan, task, role, night, shadow, base)
                results.append(r)
                state["history"].append({k: r[k] for k in ("night", "task", "kind", "role", "passed", "branch")})
            if owner == "local" and not dry_run and results and results[-1]["role"] == "local" and not results[-1]["passed"]:
                r = run_role(project, cfg, plan, task, "primary", night, False, base)  # fallback
                r["fallback"] = True
                results.append(r)
                state["history"].append({k: r[k] for k in ("night", "task", "kind", "role", "passed", "branch")})
            if not dry_run:
                state["tasks"][task["id"]] = {"status": "review", "night": night}
                save_state(project, state)  # later tasks in the same night can depend on this one
    finally:
        if awake:
            awake.terminate()
    changes = [] if dry_run else update_owners(cfg, state)
    if not dry_run:
        save_state(project, state)
    return write_morning(project, cfg, plan, state, night, results, changes, problem)


def write_morning(project: Path, cfg: dict[str, Any], plan: dict[str, Any], state: dict[str, Any], night: str,
                  results: list[dict[str, Any]], changes: list[str], problem: str | None) -> Path:
    d = project / STATE_DIR / "morning"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{night}.md"
    out = [f"# Night shift {night}", "", describe_plan(plan), ""]
    if problem:
        out += [f"**Local agent skipped:** {problem}", ""]
    out += ["| Task | Kind | Agent | Shadow | Check | Minutes | Diff | Branch |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        check = "dry run" if r.get("dry_run") else (f"skipped: {r['skipped']}" if r.get("skipped") else ("pass" if r["passed"] else "FAIL"))
        out.append(f"| {r['task']} | {r['kind']} | {r['role']}{' (fallback)' if r.get('fallback') else ''} | {'yes' if r.get('shadow') else ''} "
                   f"| {check} | {r.get('minutes', '')} | {r.get('diff', '')} | {r.get('branch', '')} |")
    listen = [(r["task"], r["role"], f) for r in results for f in r.get("listen", [])]
    if listen:
        out += ["", "## Listening queue", ""] + [f"- {t} ({role}): `{f}`" for t, role, f in listen]
    out += ["", "## Who owns each kind of task", ""]
    for kind in sorted({t["kind"] for t in cfg["tasks"]}):
        s = kind_stats(state["history"], kind)
        out.append(f"- {kind}: **{state['kind_owner'].get(kind, 'primary')}** (primary {s['primary_pass']:.0%} of {s['primary_runs']}, "
                   f"local {s['local_pass']:.0%} of {s['local_runs']})")
    if changes:
        out += ["", "## Handover changes", ""] + [f"- {c}" for c in changes]
    failing = [r for r in results if not r.get("dry_run") and not r.get("skipped") and not r["passed"]]
    if failing:
        out += ["", "## Failed checks", ""] + [f"- {r['task']} ({r['role']}): `{r['check_tail'][-200:]}`" for r in failing]
    out += ["", "## Morning review", "",
            ("Ask your strongest model: read this file, each task's NIGHT_RESULT.md and `git diff` on both branches, "
             "say which result to keep, what to listen to first, and what to queue tonight. Then mark tasks with "
             "`inferoid night mark <task> done|pending`.")]
    path.write_text("\n".join(out) + "\n")
    with (project / STATE_DIR / "runs.jsonl").open("a") as fh:
        for r in results:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "prompt"}) + "\n")
    return path


def mark(project: Path, task_id: str, status: str) -> None:
    cfg = load_config(project)
    if task_id not in {t["id"] for t in cfg["tasks"]}:
        raise SystemExit(f"unknown task {task_id}")
    state = load_state(project)
    state["tasks"][task_id] = {"status": status, "marked": dt.datetime.now().isoformat(timespec="seconds")}
    save_state(project, state)


LAUNCHD = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key><array><string>{exe}</string><string>night</string><string>run</string><string>--project</string><string>{project}</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>{hour}</integer><key>Minute</key><integer>{minute}</integer></dict>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>{path}</string></dict>
  <key>StandardOutPath</key><string>{project}/.nightshift/launchd.log</string>
  <key>StandardErrorPath</key><string>{project}/.nightshift/launchd.log</string>
</dict></plist>
"""


def write_launchd(project: Path, at: str) -> Path:
    hour, minute = (int(x) for x in at.split(":"))
    label = f"local.shadow.night.{project.name}"
    exe = shutil.which("shadow") or sys.argv[0]
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(LAUNCHD.format(label=label, exe=exe, project=project.resolve(), hour=hour, minute=minute,
                                    path=os.environ.get("PATH", "/usr/bin:/bin")))
    return plist
