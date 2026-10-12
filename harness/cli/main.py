"""CLI entry point: parse arguments, wire the runtime, run.

Wiring only. Everything here is assembly of pieces that work without it —
deleting `cli/` leaves the runtime intact, which is the layering rule from
docs/contributing.md.
"""

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence

from rich.console import Console
from rich.text import Text

from harness.agent import Agent
from harness.agent.types import AgentEvent, MessageEnd
from harness.cli.json_sink import json_sink
from harness.cli.terminal import Terminal, build_approval_gate
from harness.config import Config, ConfigError, load_config
from harness.llm.providers.anthropic import AnthropicProvider
from harness.llm.registry import resolve_model
from harness.llm.types import AssistantMessage, TextBlock
from harness.session import (
    ResumeError,
    SessionRecorder,
    TranscriptError,
    list_sessions,
    prompt_hash,
    read_transcript,
    resume,
    start_recorder,
)
from harness.tools import default_registry

SYSTEM_PROMPT = (
    "You are a helpful coding assistant running in a terminal. You have tools "
    "for reading, writing, editing, and listing files, running shell commands, "
    "searching by content or filename, and searching and fetching the web. "
    "Prefer reading a file before editing it, and prefer grep/glob over "
    "shelling out to find. web_search returns snippets, not pages — follow it "
    "with web_fetch when you need the actual content. Keep responses concise."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="A lightweight agent harness.")
    parser.add_argument("-p", "--prompt", help="Run one prompt and exit.")
    parser.add_argument("--json", action="store_true", help="Emit the raw event stream as JSONL.")
    parser.add_argument("--model", help="Model id. Defaults to config, then HARNESS_MODEL.")
    parser.add_argument("--cwd", default=".", help="Working directory for tools.")
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
        help="Continue a session. With no id, the most recent one for this directory.",
    )
    parser.add_argument(
        "--sessions", action="store_true", help="List saved sessions for this directory and exit."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write a transcript for this run."
    )
    return parser


def _build_agent(
    args: argparse.Namespace,
    config: Config,
    *,
    tool_names: Sequence[str] | None = None,
    before_tool_call: Callable[..., Awaitable[object]] | None = None,
) -> Agent:
    registry = default_registry()
    tools = registry.select(list(tool_names)) if tool_names is not None else tuple(registry)
    return Agent(
        model=resolve_model(args.model),
        provider=AnthropicProvider(config.providers.anthropic),
        system_prompt=SYSTEM_PROMPT,
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
            session_id=args.resume or None,
            system_prompt_hash=prompt_hash(agent.state.system_prompt),
        )
        agent.restore(resumed.messages)
        if console is not None:
            console.print(
                Text(
                    f"Resumed {resumed.path.stem} — {len(resumed.messages)} message(s)",
                    style="dim",
                )
            )
            for warning in resumed.warnings:
                console.print(Text(f"  {warning}", style="yellow"))

    if args.no_save or not config.session.enabled:
        return None

    recorder = start_recorder(
        cwd=args.cwd,
        model=agent.state.model,
        system_prompt=agent.state.system_prompt,
        config=config.session,
        path=resumed.path if resumed is not None else None,
    )
    agent.subscribe(recorder)
    return recorder


async def run_once(args: argparse.Namespace, config: Config) -> int:
    """One-shot mode. No approval gate — there is nobody at the keyboard."""
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
    terminal = Terminal(console=console, config=config)
    gate = None if args.yes else build_approval_gate(terminal, config)
    agent = _build_agent(
        args,
        config,
        tool_names=config.terminal.enabled_tools,
        before_tool_call=gate,
    )
    terminal.attach(agent)
    recorder = _attach_session(agent, args, config, console)
    try:
        return await terminal.run()
    finally:
        if recorder is not None:
            recorder.close()


def list_saved_sessions(args: argparse.Namespace, config: Config) -> int:
    """Print saved sessions for this directory, newest first."""
    console = Console()
    paths = list_sessions(args.cwd, config.session)
    if not paths:
        console.print(Text("No saved sessions for this directory.", style="dim"))
        return 0
    for path in paths:
        try:
            transcript = read_transcript(path)
        except (OSError, TranscriptError):
            console.print(Text(f"{path.stem}  (unreadable)", style="red"))
            continue
        console.print(
            f"{path.stem}  {len(transcript.messages):>4} msg  "
            f"{transcript.header.model}  {transcript.header.timestamp}"
        )
    return 0


def _text_sink() -> Callable[[AgentEvent], Awaitable[None]]:
    """Print assistant text and a one-line note per tool call.

    Text goes to stdout and notices to stderr, so `harness -p … > out.txt`
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
            print(f"● {event.tool_name}({_brief(event.arguments)})", file=sys.stderr, flush=True)

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
        # Config is loaded relative to --cwd, so `harness --cwd ../other` picks
        # up that project's .harness/ rather than this directory's.
        config = load_config(args.cwd)
        if args.sessions:
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
