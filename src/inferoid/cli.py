from __future__ import annotations

import argparse
import json
import os

from . import APP_NAME
from .analysis import classify, machine
from .parsers import format_scan, scan
from .reporting import replay, report
from .storage import connect, now_iso


def export_json() -> None:
    conn = connect()
    turns = []
    for turn in conn.execute("SELECT * FROM user_turns ORDER BY timestamp,id"):
        data = dict(turn)
        data["calls"] = []
        for call in conn.execute("SELECT * FROM model_calls WHERE user_turn_id=? ORDER BY timestamp,id", (turn["id"],)):
            item = dict(call)
            item["tool_names"] = json.loads(item["tool_names"] or "[]")
            item["tool_facts"] = json.loads(item["tool_facts"] or "[]")
            item.pop("response_text", None)
            data["calls"].append(item)
        turns.append(data)
    print(json.dumps({"schema_version": 2, "user": os.environ.get("INFEROID_USER", "local"), "exported_at": now_iso(), "turns": turns}, indent=2, ensure_ascii=False))
    conn.close()


def night_main(args: argparse.Namespace) -> None:
    from pathlib import Path

    from . import night

    project = Path(args.project).resolve()
    if args.night_command == "plan":
        cfg = night.load_config(project)
        print(night.describe_plan(night.hardware_plan(cfg)))
    elif args.night_command == "run":
        print(night.run_night(project, args.max_tasks, args.dry_run))
    elif args.night_command == "status":
        cfg = night.load_config(project)
        state = night.load_state(project)
        for task in cfg["tasks"]:
            st = state["tasks"].get(task["id"], {}).get("status", "pending")
            owner = state["kind_owner"].get(task["kind"], "primary")
            print(f"{task['id']:<32} {task['kind']:<20} {st:<8} owner={owner}")
    elif args.night_command == "mark":
        night.mark(project, args.task, args.status)
        print(f"{args.task}: {args.status}")
    elif args.night_command == "schedule":
        plist = night.write_launchd(project, args.at)
        print(f"Wrote {plist}\nStart it with: launchctl load {plist}\nStop it with:  launchctl unload {plist}")


def claude_main(args: argparse.Namespace) -> int:
    import sys
    from pathlib import Path

    from . import claude

    project = Path(getattr(args, "project", ".")).resolve()
    if args.claude_command == "install":
        print("\n".join(claude.install(project, args.global_, args.threshold)))
    elif args.claude_command == "uninstall":
        print("\n".join(claude.uninstall(project, args.global_)))
    elif args.claude_command == "local":
        return claude.local(project, args.model, args.resume, args.dry_run, args.full)
    elif args.claude_command == "status":
        print("\n".join(claude.status(project)))
    elif args.claude_command == "statusline":
        print(claude.statusline(sys.stdin.read()))
    elif args.claude_command == "hook":
        out = claude.hook(args.event, sys.stdin.read())
        if out:
            print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Private local coding-agent usage report")
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("--days", type=int, default=30)
    commands.add_parser("machine")
    classify_parser = commands.add_parser("classify")
    classify_parser.add_argument("--local-model")
    classify_parser.add_argument("--endpoint", help="Local Ollama or OpenAI-compatible endpoint")
    replay_parser = commands.add_parser("replay", help="Experimental step-level local replay")
    replay_parser.add_argument("--steps", type=int, default=20)
    replay_parser.add_argument("--model", required=True)
    replay_parser.add_argument("--endpoint", help="Local Ollama or OpenAI-compatible endpoint")
    replay_parser.add_argument("--tier", choices=("16", "48", "128"))
    commands.add_parser("report")
    commands.choices["report"].add_argument("--tier", choices=("16", "48", "128"))
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--json", action="store_true", required=True)
    night = commands.add_parser("night", help="Overnight tasks: paid primary agent + free local shadow agent")
    night_cmds = night.add_subparsers(dest="night_command", required=True)
    for name in ("plan", "run", "status", "mark", "schedule"):
        sub = night_cmds.add_parser(name)
        sub.add_argument("--project", default=".")
    night_cmds.choices["run"].add_argument("--max-tasks", type=int, default=0)
    night_cmds.choices["run"].add_argument("--dry-run", action="store_true")
    night_cmds.choices["mark"].add_argument("task")
    night_cmds.choices["mark"].add_argument("status", choices=("pending", "done", "review"))
    night_cmds.choices["schedule"].add_argument("--at", default="01:00", help="HH:MM local time")
    st = commands.add_parser("setup", help="Pick and download a local model and connect it to your coding agent")
    st.add_argument("--project", default=".")
    cl = commands.add_parser("claude", help="Claude Code handover to a free local model")
    cl_cmds = cl.add_subparsers(dest="claude_command", required=True)
    for name in ("install", "uninstall"):
        sub = cl_cmds.add_parser(name)
        sub.add_argument("--project", default=".")
        sub.add_argument("--global", dest="global_", action="store_true", help="Use ~/.claude instead of this project")
    cl_cmds.choices["install"].add_argument("--threshold", type=float, default=90.0, help="Usage %% that triggers the HANDOFF.md reminder")
    loc = cl_cmds.add_parser("local", help="Continue in Claude Code on a local Ollama model")
    loc.add_argument("--project", default=".")
    loc.add_argument("--model", help="Ollama model (default: picked by memory)")
    loc.add_argument("--resume", action="store_true", help="Resume the last session instead of starting from HANDOFF.md")
    loc.add_argument("--dry-run", action="store_true")
    loc.add_argument("--full", action="store_true", help="Load MCP servers, skills and all tools (much slower first reply)")
    cl_cmds.add_parser("status").add_argument("--project", default=".")
    cl_cmds.add_parser("statusline", help="(run by Claude Code) record usage for the status line")
    cl_cmds.add_parser("hook", help="(run by Claude Code) usage reminder hook").add_argument("event")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        print(format_scan(scan(args.days)))
    elif args.command == "machine":
        machine()
    elif args.command == "classify":
        print(json.dumps(classify(args.local_model, args.endpoint), indent=2, sort_keys=True))
    elif args.command == "replay":
        replay(args.steps, args.model, args.endpoint, args.tier)
    elif args.command == "report":
        report(args.tier)
    elif args.command == "export":
        export_json()
    elif args.command == "night":
        night_main(args)
    elif args.command == "claude":
        return claude_main(args)
    elif args.command == "setup":
        from pathlib import Path

        from .setup import run

        return run(Path(args.project).resolve())
    return 0
