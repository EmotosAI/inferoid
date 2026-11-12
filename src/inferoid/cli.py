from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
        return claude.local(project, args.model, args.resume, args.dry_run, args.full, args.allow_gpu_limit,
                            args.context, args.clean, args.engine, args.engine_url, args.engine_model)
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
    st.add_argument("--supervisor", choices=("claude", "codex", "cursor", "all"))
    run_local = commands.add_parser("run", help="Delegate one checked chore to the local model")
    run_local.add_argument("task")
    run_local.add_argument("--check")
    run_local.add_argument("--files", action="append", default=[])
    run_local.add_argument("--model")
    run_local.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    run_local.add_argument("--engine-url", default="http://127.0.0.1:8080")
    run_local.add_argument("--engine-model")
    run_local.add_argument("--context")
    run_local.add_argument("--timeout", type=int, default=900)
    run_local.add_argument("--max-turns", type=int, default=200)
    run_local.add_argument("--allow-bash", action="append", default=[])
    run_local.add_argument("--deny-tool", action="append", default=[])
    run_local.add_argument("--share", action="store_true", help="Submit a metrics-only report to Forge")
    run_local.add_argument("--json", action="store_true")
    commands.add_parser("mcp", help="Run the stdio MCP server")
    bench = commands.add_parser("bench", help="Run isolated Claude Code benchmarks on local models")
    bench_cmds = bench.add_subparsers(dest="bench_command", required=True)
    run_bench = bench_cmds.add_parser("run")
    run_bench.add_argument("--repo", required=True)
    run_bench.add_argument("--brief", required=True)
    run_bench.add_argument("--check", required=True)
    run_bench.add_argument("--pack", required=True)
    run_bench.add_argument("--task", required=True)
    run_bench.add_argument("--kind", default="coding")
    run_bench.add_argument("--model")
    run_bench.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    run_bench.add_argument("--engine-url", default="http://127.0.0.1:8080")
    run_bench.add_argument("--engine-model", help="Model name expected by an OpenAI-compatible engine")
    run_bench.add_argument("--timeout", type=int, default=1800)
    run_bench.add_argument("--context", help="Context size: a number (65536 or 64k) or 'max' to measure the largest safe one")
    run_bench.add_argument("--max-turns", type=int, default=200)  # time, not turns, should end a bench
    run_bench.add_argument("--allow-bash", action="append", default=[])
    run_bench.add_argument("--deny-tool", action="append", default=[],
                           help="Hide a Claude Code tool from the model (e.g. Edit, Write) so the task's own commands are used")
    run_bench.add_argument("--reference")
    run_bench.add_argument("--no-submit", action="store_true")
    run_bench.add_argument("--force", action="store_true")
    run_bench.add_argument("--no-clean", dest="clean", action="store_false",
                           help="For Ollama, send Claude Code's requests unchanged (OpenAI mode always uses the translator)")
    run_bench.add_argument("--keep", action="store_true")
    run_bench.add_argument("--from-commit", action="store_true",
                           help="Benchmark the last commit instead of the current working tree")
    run_bench.add_argument("--share", action="append", default=[], metavar="PATH",
                           help="Project-relative path to link into the benchmark copy (data, audio, caches not in git)")
    submit_bench = bench_cmds.add_parser("submit")
    submit_bench.add_argument("files", nargs="+")
    bench_cmds.add_parser("list")
    forge_cmd = commands.add_parser("forge", help="Configure and inspect Forge metrics sharing")
    forge_cmds = forge_cmd.add_subparsers(dest="forge_command", required=True)
    login = forge_cmds.add_parser("login")
    login.add_argument("--token", required=True)
    login.add_argument("--url")
    forge_cmds.add_parser("status")
    forge_cmds.add_parser("logout")
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
    loc.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    loc.add_argument("--engine-url", default="http://127.0.0.1:8080")
    loc.add_argument("--engine-model", help="Model name expected by an OpenAI-compatible engine")
    loc.add_argument("--context", help="Context size: a number (65536 or 64k) or 'max' to measure the largest safe one")
    loc.add_argument("--resume", action="store_true", help="Resume the last session instead of starting from HANDOFF.md")
    loc.add_argument("--dry-run", action="store_true")
    loc.add_argument("--no-clean", dest="clean", action="store_false",
                     help="For Ollama, send requests unchanged (OpenAI mode always uses the translator)")
    loc.add_argument("--full", action="store_true", help="Load MCP servers, skills and all tools (much slower first reply)")
    loc.add_argument("--allow-gpu-limit", action="store_true",
                     help="Offer to raise the macOS GPU memory limit when the model needs it (only if macOS keeps 4 GB)")
    cl_cmds.add_parser("status").add_argument("--project", default=".")
    cl_cmds.add_parser("statusline", help="(run by Claude Code) record usage for the status line")
    cl_cmds.add_parser("hook", help="(run by Claude Code) usage reminder hook").add_argument("event")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)  # progress shows up even when output goes to a file
        except (AttributeError, ValueError):
            pass
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
        from .setup import run

        return run(Path(args.project).resolve(), supervisor=args.supervisor)
    elif args.command == "run":
        from .dispatch import run_task

        value = run_task(project=Path.cwd(), task=args.task, check=args.check, files=args.files, model=args.model,
                         engine=args.engine, engine_url=args.engine_url, engine_model=args.engine_model,
                         context=args.context, timeout=args.timeout, max_turns=args.max_turns,
                         allow_bash=args.allow_bash, deny_tools=args.deny_tool, share=args.share)
        if args.share and value.get("_report"):
            from . import forge

            metrics_report = value.pop("_report")
            try:
                ok, message, verdict = forge.submit(metrics_report)
                value.setdefault("notes", []).append(message if not ok else "Metrics report submitted")
                if verdict:
                    value["verdict"] = verdict
            except (OSError, TypeError, ValueError) as exc:
                value.setdefault("notes", []).append(f"Metrics report could not be submitted: {exc}")
        value.pop("_report", None)
        if args.json:
            print(json.dumps(value, sort_keys=True))
        else:
            print(f"{value['status']}: {len(value.get('files_changed', []))} file(s), {value.get('wall_s', 0):.1f}s")
            if value.get("notes"):
                print("; ".join(value["notes"][-3:]))
        return 0 if value.get("status") == "done" else 1
    elif args.command == "mcp":
        from .mcp import serve

        serve()
        return 0
    elif args.command == "bench":
        import signal

        def stop(signum, frame):  # treat kill/hangup like Ctrl+C: stop the run but still write the report
            raise KeyboardInterrupt
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        from . import bench

        if args.bench_command == "run":
            code, _ = bench.run(Path(args.repo), Path(args.brief), args.check, args.pack, args.task,
                                 args.kind, args.model, args.timeout, args.max_turns, args.allow_bash,
                                 Path(args.reference) if args.reference else None, args.no_submit, args.force, args.keep,
                                 args.share, args.from_commit, args.context, args.clean, args.deny_tool,
                                 args.engine, args.engine_url, args.engine_model)
            return code
        if args.bench_command == "submit":
            return bench.submit_files([Path(path) for path in args.files])
        return bench.list_reports()
    elif args.command == "forge":
        from . import forge

        if args.forge_command == "login":
            try:
                print(forge.login(args.token, args.url))
            except ValueError as exc:
                print(f"Forge login failed: {exc}")
                return 2
        elif args.forge_command == "status":
            print("\n".join(forge.status()))
        elif args.forge_command == "logout":
            print(forge.logout())
    return 0
