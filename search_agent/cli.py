"""`python -m search_agent` — every query in the file, one agent each, in order.

Progress on the console and nothing else interactive: no terminal, no approval
gate, no keyboard. Each query writes `runs/<query-id>/` and the run ends with
what they cost together.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from lab_harness.agent.agent import Listener
from lab_harness.agent.types import AgentEvent
from lab_harness.cli.usage import format_cost, format_tokens, prompt_tokens
from lab_harness.config import Config, load_config
from lab_harness.llm.types import Usage
from search_agent.run import Outcome, run_query
from search_agent.settings import CONFIG_FILE, load_settings

#: The harness config this program runs on, inside its own directory. Keys
#: and the model belong here, so it is the gitignored one; absent by default,
#: which is not an error.
LOCAL_CONFIG = "config.local.toml"


def _listener(console: Console) -> Listener:
    """A subscriber printing what the terminal would, without the terminal."""

    async def report(event: AgentEvent) -> None:
        if event.type == "message_update" and event.kind == "text":
            console.print(event.delta, end="", markup=False, highlight=False)
        elif event.type == "tool_execution_start":
            console.print(f"\n→ {event.tool_name}({_brief(event.arguments)})", markup=False)

    return report


def _brief(arguments: dict[str, object], width: int = 90) -> str:
    shown = json.dumps(arguments, ensure_ascii=False)
    return shown[:width] + ("…" if len(shown) > width else "")


def _line(outcome: Outcome, config: Config, root: Path) -> str:
    """One query's line: what it spent, and where its files are."""
    parts = [
        f"{outcome.requests} request(s)",
        f"{outcome.tool_calls} tool call(s)",
        f"{format_tokens(prompt_tokens(outcome.usage))} in"
        f" / {format_tokens(outcome.usage.output)} out",
    ]
    if outcome.cost is not None:
        parts.append(format_cost(outcome.cost, config.terminal.currency_symbol))
    parts.append(f"{outcome.seconds:.1f}s")
    where = outcome.directory
    with suppress(ValueError):
        where = outcome.directory.relative_to(root)
    return f"{' · '.join(parts)} → {where}"


def _total(left: Usage, right: Usage) -> Usage:
    return Usage(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_write=left.cache_write + right.cache_write,
    )


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="search_agent", description="Run the agent over a corpus, one query at a time."
    )
    parser.add_argument("--queries", type=Path, help="the query file")
    parser.add_argument(
        "--query", action="append", default=[], help="only this query_id (repeatable)"
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after this many")
    parser.add_argument("--out-dir", type=Path, help="where the runs are written")
    parser.add_argument(
        "--settings", type=Path, default=CONFIG_FILE, help="this program's settings"
    )
    parser.add_argument("--config", help="harness config (default: config.local.toml)")
    args = parser.parse_args(argv)

    settings = load_settings(Path(args.settings))
    if args.queries:
        settings = replace(settings, queries=Path(args.queries).expanduser())
    if args.out_dir:
        settings = replace(settings, out=Path(args.out_dir).expanduser())

    config_file = Path(args.config).expanduser() if args.config else settings.cwd / LOCAL_CONFIG
    if args.config and not config_file.is_file():
        print(f"error: --config: {args.config} is not a file", file=sys.stderr)
        return 1
    # An absent default is normal: the rest of the chain still applies.
    config = load_config(settings.cwd, config_file=config_file if config_file.is_file() else None)

    if not settings.queries.is_file():
        print(f"error: no query file at {settings.queries}", file=sys.stderr)
        return 1
    queries = [
        json.loads(line)
        for line in settings.queries.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.query:
        queries = [q for q in queries if str(q.get("query_id")) in set(args.query)]
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("error: no queries to run", file=sys.stderr)
        return 1

    console = Console()
    console.print(
        f"{len(queries)} quer(y/ies) · {config.agent.model} · {', '.join(settings.tools)}",
        markup=False,
    )

    spent = Usage()
    cost = 0.0
    failed: list[str] = []
    for index, query in enumerate(queries, start=1):
        identifier = str(query.get("query_id", index))
        console.print(f"\n[bold][{index}/{len(queries)}] query {identifier}[/bold]")
        console.print(str(query.get("query", ""))[:200], style="dim", markup=False)
        try:
            outcome = await run_query(query, settings, config, notify=_listener(console))
        except Exception as exc:  # one bad query must not end the run
            failed.append(identifier)
            console.print(f"  {type(exc).__name__}: {exc}", style="red", markup=False)
            continue
        spent = _total(spent, outcome.usage)
        cost += outcome.cost or 0.0
        console.print(f"\n  {_line(outcome, config, settings.cwd)}")
        console.print(f"  answer: {outcome.answer[:200]}", markup=False)

    run = len(queries) - len(failed)
    console.print(
        f"\n{run}/{len(queries)} run · "
        f"{format_tokens(prompt_tokens(spent))} in / {format_tokens(spent.output)} out · "
        f"{format_cost(cost, config.terminal.currency_symbol)} · "
        f"trajectories in {settings.out}",
        markup=False,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
