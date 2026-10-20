"""CLI entry point: parse arguments, wire the runtime, run.

Wiring only. Everything here is assembly of pieces that work without it —
deleting `cli/` leaves the runtime intact, which is the layering rule from
docs/contributing.md.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from lab_harness.agent import Agent
from lab_harness.agent.types import AgentEvent, MessageEnd, declarations_for
from lab_harness.cli.export import export_html
from lab_harness.cli.json_sink import json_sink
from lab_harness.cli.renderers import display_name
from lab_harness.cli.terminal import LogStream, Terminal, build_approval_gate
from lab_harness.config import Config, ConfigError, PromptsConfig, load_config
from lab_harness.llm.providers import route
from lab_harness.llm.types import AssistantMessage, TextBlock
from lab_harness.logs import LEVELS, configure_logging
from lab_harness.prompts import build_system_prompt, detect_environment, load_project_context
from lab_harness.session import (
    ResumeError,
    SessionRecorder,
    TranscriptError,
    find_session,
    latest_session,
    list_sessions,
    read_transcript,
    resume,
    start_recorder,
)
from lab_harness.tools import AgentTool, default_registry

_log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab-harness", description="A minimal agent harness in Python, built for research."
    )
    parser.add_argument("-p", "--prompt", help="Run one prompt and exit.")
    parser.add_argument("--json", action="store_true", help="Emit the raw event stream as JSONL.")
    parser.add_argument(
        "--model", help="Model id. Overrides $LAB_HARNESS_MODEL and the configured agent.model."
    )
    parser.add_argument(
        "--provider",
        help=(
            "Provider serving the model: 'anthropic' or a configured "
            "providers.openai profile. Normally inferred from the model id."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        metavar="LEVEL",
        help=(
            "How hard a reasoning model thinks, in its provider's terms, e.g. low or "
            "high. Overrides the model's and agent.reasoning_effort; an empty value "
            "sends none."
        ),
    )
    parser.add_argument(
        "--cwd", default=".", help="Working directory for tools, config, and project instructions."
    )
    parser.add_argument(
        "--tools",
        type=parse_tool_names,
        metavar="NAMES",
        help=(
            "Comma-separated tools to enable, e.g. read,ls,grep. Overrides "
            "tools.enabled; an empty value enables none."
        ),
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument(
        "--system-prompt",
        metavar="TEXT",
        help=(
            "Replaces prompts.system for this run. The other sections of the "
            "system prompt, project instructions included, are still sent."
        ),
    )
    prompt.add_argument(
        "--system-prompt-file",
        metavar="PATH",
        help="Like --system-prompt, with the text read from a file.",
    )
    parser.add_argument(
        "--log-level",
        # list(), not sorted(): LEVELS is in severity order, which is how the
        # choices read in --help. Alphabetical puts "error" second.
        choices=list(LEVELS),
        help="Diagnostics on stderr. Overrides $LAB_HARNESS_LOG_LEVEL and logging.level.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip tool approval prompts in interactive mode.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="SESSION_ID",
        help=(
            "Continue the session with this id; a unique prefix works. With no "
            "id, list this directory's sessions and exit."
        ),
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write a transcript for this run."
    )
    parser.add_argument(
        "--export",
        nargs="?",
        const="",
        metavar="SESSION_ID",
        help=(
            "Write a saved session as one HTML page beside its transcript, print "
            "its path, and exit. With no id, this directory's most recent session."
        ),
    )
    return parser


def parse_tool_names(value: str) -> tuple[str, ...]:
    """`--tools read, ls` as names. Blank entries are dropped, so "" is none."""
    return tuple(name for part in value.split(",") if (name := part.strip()))


def _select_tools(args: argparse.Namespace, config: Config) -> tuple[AgentTool, ...]:
    """The tools this run gets: `--tools` if given, else `tools.enabled`.

    The same rule in both modes. Leaving a tool out is the only thing that
    stops it running in one-shot mode, which has no approval gate.
    """
    if args.tools is not None:
        names, source = args.tools, "--tools"
    else:
        names, source = config.tools.enabled, "tools.enabled"
    try:
        tools = default_registry().select(names)
    except ValueError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
    _log.info("Tools from %s: %s", source, ", ".join(t.name for t in tools) or "none")
    return tools


def _prompts(args: argparse.Namespace, config: Config) -> PromptsConfig:
    """`prompts`, with `system` replaced if a flag says so."""
    if args.system_prompt is not None:
        system, source = args.system_prompt, "--system-prompt"
    elif args.system_prompt_file is not None:
        source = "--system-prompt-file"
        try:
            # Stripped, as project instructions are: an editor's final newline
            # is not part of the prompt.
            system = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(
                f"{source}: cannot read {args.system_prompt_file} ({exc.strerror})"
            ) from exc
        except UnicodeDecodeError as exc:
            raise ConfigError(f"{source}: {args.system_prompt_file} is not UTF-8 text") from exc
    else:
        system, source = config.prompts.system, "prompts.system"
    _log.info("System prompt from %s (%d characters)", source, len(system))
    return replace(config.prompts, system=system)


def _build_agent(
    args: argparse.Namespace,
    config: Config,
    *,
    before_tool_call: Callable[..., Awaitable[object]] | None = None,
    console: Console | None = None,
) -> Agent:
    tools = _select_tools(args, config)
    context = load_project_context(args.cwd, config.project)
    if context is not None and console is not None:
        # Project instructions change how the agent behaves; a silent feature
        # cannot be told apart from a broken one.
        note = f"Project instructions: {context.path.name}"
        if context.truncated:
            note += f" (truncated to {config.project.max_bytes} bytes)"
        console.print(Text(note, style=config.terminal.theme.muted))
    model, provider = route(args.model, config, args.provider)
    if args.reasoning_effort is not None:
        model = replace(model, reasoning_effort=args.reasoning_effort.strip())
        _log.info("Reasoning effort %r, from --reasoning-effort", model.reasoning_effort)
    environment = detect_environment(args.cwd, model.id)
    system_prompt = build_system_prompt(_prompts(args, config), environment, context)
    _log.debug("System prompt: %d characters", len(system_prompt))
    return Agent(
        model=model,
        provider=provider,
        system_prompt=system_prompt,
        tools=tools,
        config=config,
        cwd=args.cwd,
        before_tool_call=before_tool_call,  # type: ignore[arg-type]
    )


def _attach_session(
    agent: Agent, args: argparse.Namespace, config: Config, console: Console | None = None
) -> SessionRecorder | None:
    """Restore a prior transcript if asked, and start recording this run.

    Returns the recorder so the caller can close it. None means nothing is
    being persisted.
    """
    resumed = None
    if args.resume is not None:
        resumed = resume(
            cwd=args.cwd,
            config=config.session,
            session_id=args.resume,
            system_prompt=agent.state.system_prompt,
        )
        agent.restore(resumed.messages)
        note = f"Resumed {resumed.path.stem} — {len(resumed.messages)} message(s)"
        if console is not None:
            # A rule rather than a line: what follows is the restored history,
            # drawn exactly as new turns are, and the rule is the only thing
            # saying where it ends.
            console.print(Rule(note, style=config.terminal.theme.muted))
            for warning in resumed.warnings:
                console.print(Text(f"  {warning}", style=config.terminal.theme.warning))
        else:
            # One-shot mode has no terminal to show these in. Logging puts them
            # on stderr, clear of the answer on stdout and of the JSONL stream.
            _log.info(note)
            for warning in resumed.warnings:
                _log.warning("Resumed %s: %s", resumed.path.stem, warning)

    if args.no_save or not config.session.enabled:
        return None

    # Everything a request carries besides the messages, so that a transcript
    # rebuilds each one exactly — see `recorded_requests`.
    recorder = start_recorder(
        cwd=args.cwd,
        model=agent.state.model,
        system_prompt=agent.state.system_prompt,
        tools=declarations_for(agent.state.tools),
        config=config,
        restored=len(agent.state.messages),
        path=resumed.path if resumed is not None else None,
    )
    agent.subscribe(recorder)
    return recorder


async def run_once(args: argparse.Namespace, config: Config) -> int:
    """One-shot mode. No approval gate — there is nobody at the keyboard."""
    _log.info("One-shot mode: tools run without approval")
    agent = _build_agent(args, config)
    agent.subscribe(json_sink() if args.json else _text_sink())
    recorder = _attach_session(agent, args, config)

    try:
        messages = await agent.prompt(args.prompt)
    finally:
        if recorder is not None:
            recorder.close()

    # A failed turn is a failed command: the exit status has to say so, or a
    # script piping this cannot tell a refusal from an outage.
    failed = any(
        isinstance(m, AssistantMessage) and m.stop_reason in ("error", "aborted") for m in messages
    )
    return 1 if failed else 0


async def run_interactive(args: argparse.Namespace, config: Config) -> int:
    console = Console()
    if console.is_terminal and sys.stderr.isatty():
        # One screen for both, so records go through the console — above the
        # live region rather than through it. A redirected stderr keeps them.
        stream = LogStream(console, config.terminal.theme.muted)
        configure_logging(args.log_level, config.logging, stream=stream)
    terminal = Terminal(console=console, config=config)
    if args.yes:
        _log.info("Tool approval off: --yes")
    gate = None if args.yes else build_approval_gate(terminal, config)
    agent = _build_agent(args, config, before_tool_call=gate, console=console)
    terminal.attach(agent)
    terminal.welcome()
    recorder = _attach_session(agent, args, config, console)
    if args.resume is not None:
        # The history itself, so the session continues where the eye left off
        # rather than from the one-line note above it. One-shot skips this: its
        # stdout is the answer alone.
        terminal.renderer.replay(agent.state.messages)
    try:
        return await terminal.run()
    finally:
        if recorder is not None:
            recorder.close()


def export_session(args: argparse.Namespace, config: Config) -> int:
    """`--export`: write a session's HTML page and print where. Reads only."""
    try:
        if args.export:
            path = find_session(args.cwd, args.export, config.session)
        else:
            path = latest_session(args.cwd, config.session)
        if path is None:
            raise ResumeError(
                f"No session matching {args.export!r} for this directory."
                if args.export
                else "No saved sessions for this directory."
            )
        target = export_html(path, config)
    except (ValueError, TranscriptError) as exc:  # an ambiguous prefix, a bad file
        raise ResumeError(str(exc)) from exc
    print(target)
    return 0


def list_saved_sessions(args: argparse.Namespace, config: Config) -> int:
    """`--resume` with no id: print this directory's sessions, newest first.

    Continuing one takes its id. Picking the most recent on its own would
    continue the wrong conversation whenever the last run was not the one meant.
    """
    console = Console()
    paths = list_sessions(args.cwd, config.session)
    if not paths:
        console.print(
            Text("No saved sessions for this directory.", style=config.terminal.theme.muted)
        )
        return 0
    for path in paths:
        try:
            transcript = read_transcript(path)
        except (OSError, TranscriptError):
            console.print(Text(f"{path.stem}  (unreadable)", style=config.terminal.theme.error))
            continue
        console.print(
            f"{path.stem}  {len(transcript.messages):>4} msg  "
            f"{transcript.header.model}  {transcript.header.timestamp}"
        )
    console.print(
        Text(
            "Continue one with --resume SESSION_ID; a unique prefix works.",
            style=config.terminal.theme.muted,
        )
    )
    return 0


def _text_sink() -> Callable[[AgentEvent], Awaitable[None]]:
    """Print assistant text and a one-line note per tool call.

    Text goes to stdout and notices to stderr, so `lab-harness -p … > out.txt`
    captures the answer alone.
    """

    async def emit(event: AgentEvent) -> None:
        if isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            body = "".join(b.text for b in event.message.content if isinstance(b, TextBlock))
            if body:
                print(body, flush=True)
            if event.message.error_message:
                print(f"error: {event.message.error_message}", file=sys.stderr, flush=True)
        elif event.type == "tool_execution_start":
            print(
                f"⏺ {display_name(event.tool_name)}({_brief(event.arguments)})",
                file=sys.stderr,
                flush=True,
            )

    return emit


def _brief(arguments: dict[str, object]) -> str:
    """One-line argument preview for the tool-call notice."""
    parts = []
    for key, value in arguments.items():
        rendered = str(value).replace("\n", " ")
        if len(rendered) > 40:
            rendered = rendered[:40] + "…"
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Twice, deliberately. The first call uses the flag, the environment and
        # the schema defaults, so `--log-level debug` covers the loading of
        # config itself — which is exactly what you want when a setting appears
        # to do nothing. A level set *in* a config file cannot report its own
        # file being read; the second call hands control to it from here on.
        configure_logging(args.log_level)
        # Config is loaded relative to --cwd, so `lab-harness --cwd ../other` picks
        # up that project's .lab-harness/ rather than this directory's.
        config = load_config(args.cwd)
        configure_logging(args.log_level, config.logging)
        if args.export is not None:
            return export_session(args, config)
        # `--resume` alone: its value is the flag's const, "".
        if args.resume == "":
            return list_saved_sessions(args, config)
        if args.prompt:
            return asyncio.run(run_once(args, config))
        return asyncio.run(run_interactive(args, config))
    except (ConfigError, ResumeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
