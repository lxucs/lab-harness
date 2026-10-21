"""The interactive terminal.

prompt_toolkit reads input (history, multi-line, key bindings); rich writes
output. Both operate on ordinary stdout, so the session lands in real terminal
scrollback instead of a full-screen app you cannot scroll back through.

Behaviors worth knowing:

**Ctrl-C cancels the turn, not the session.** During a run it aborts the run
and leaves the transcript valid to continue from. At an idle prompt it clears
the input line. Ctrl-D exits. The two arrive differently: at a prompt,
prompt_toolkit has the terminal in raw mode and Ctrl-C is a keypress; during a
run it is a SIGINT, which the turn takes over from asyncio.run's handler.

**Ctrl-O toggles thinking** between shown in full and `terminal.thinking`, and
prints what the last run collapsed. It is a prompt key binding, so pressed
mid-run it waits in the input queue and takes effect when the prompt returns —
nothing reads keys during a run, which is also what keeps typed-ahead text
safe. Nothing echoes them either: typed-ahead text appears when the prompt
does, not in the live region — see `_keys_wait_quietly`.

**Typing during a run is buffered here, not queued in the runtime.** The agent
refuses a second concurrent `prompt()` by design (see `agent.py`), so the
terminal holds the text and submits it when the run returns. That keeps the
"one run at a time" rule in one place rather than spread across layers.

**Log records print through the console** when stderr is this terminal, so
they land above the live region rather than inside it — see `LogStream`.

**Slash commands are run here, not sent.** The terminal registers its own in
a `Commands` registry, which both runs them and lists them above the prompt as
one is typed — see `commands.py`. Each skill found for the directory is added
as one when the agent attaches, so `/commit` sends that skill's instructions
as a turn; `/skills` lists them. `/tools` and `/skills` also *choose* what
applies mid-session, and the system prompt is rebuilt to match — by a callable
the CLI hands over, being what knows the flags that shaped it.

**`/model` and `/effort` change what the next runs ask for.** Each replaces
the agent's model, so the next run's `agent_start` carries it and a recorded
session opens a new segment there. `/model` stays inside the provider this
session started on: the provider is chosen once, when the agent is built.

**A status sits at the right of the input**: the model, and how much of its
window the last request filled, dim until that passes
`terminal.context_warning` — which is also when a run's footer says so. Unlike
a bottom toolbar it is a float over the input's first row, so it costs no row
and steps aside for what is typed.

**Usage is read from the transcript, not tracked.** The footer after a run
and `/status` both fold over messages that already carry their `Usage` — see
`usage.py` — so there is no counter here to fall out of step with history.
"""

import asyncio
import io
import logging
import os
import signal
import sys
import threading
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.filters import Filter
from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from lab_harness.agent import Agent
from lab_harness.cli.approval import ApprovalGate
from lab_harness.cli.commands import (
    Command,
    CommandMenu,
    Commands,
    EndSession,
    UnknownCommand,
    menu_style,
)
from lab_harness.cli.renderers import GLYPH, EventRenderer, display_name
from lab_harness.cli.usage import (
    UsageSummary,
    context_share,
    format_footer,
    format_status,
    summarize,
)
from lab_harness.config import Config, ConfigError, lab_harness_home
from lab_harness.llm.providers import select_provider
from lab_harness.llm.registry import resolve_model
from lab_harness.llm.types import Message, Model
from lab_harness.tools import default_registry
from lab_harness.tools.skills import Skill, directories, index, load_skills, skill

#: One line of what the keys do, under the welcome box.
_TIPS = "/ for commands · Ctrl-C cancels · Ctrl-O toggles thinking"
#: Style class for the glyph alone. Not `prompt`: prompt_toolkit applies that
#: class to every prompt message, the approval question included.
_GLYPH_CLASS = "glyph"
_PROMPT_MESSAGE = FormattedText([(f"class:{_GLYPH_CLASS}", GLYPH)])
#: The status at the right of the input, dim, and warning when the window is
#: nearly full. Its own classes, and ansi colors rather than the theme's rich
#: style strings, because prompt_toolkit parses these — see `menu_style`.
_STATUS = "class:status"
_STATUS_WARNING = "class:status.warning"
_STATUS_STYLE = {"status": "fg:ansibrightblack", "status.warning": "fg:ansiyellow"}
_CANCELLED_NOTICE = "^C  turn cancelled"
#: What `/effort` takes to send no level, leaving it to the model. Not "none"
#: or "off": some APIs take those as levels, meaning no thinking.
_EFFORT_CLEAR = "default"
#: Answers accepted at the approval prompt, by first letter. Deliberately no
#: empty-string entry: a bare Enter is not consent, it is a reflex. Anything
#: unlisted falls through to "no".
_ANSWERS = {"y": "yes", "n": "no", "a": "always"}
#: What `/tools` and `/skills` take to choose nothing.
_NONE = "none"

_log = logging.getLogger(__name__)


@dataclass
class Terminal:
    """Console, input loop, and approval prompt for one interactive session.

    Constructed *before* the agent, because the agent needs an approval gate
    that asks through this terminal — so the agent attaches afterwards via
    `attach()` rather than being a constructor argument.
    """

    console: Console
    config: Config

    def __post_init__(self) -> None:
        theme = self.config.terminal.theme
        try:
            style = Style.from_dict(
                {_GLYPH_CLASS: theme.prompt, **menu_style(theme.prompt), **_STATUS_STYLE}
            )
        except ValueError as exc:
            raise ConfigError(f"terminal.theme.prompt: {exc}") from exc
        #: Every command the prompt runs and lists, in the order it lists them.
        #: The first is the one Enter runs on a bare `/`, so it changes nothing.
        self.commands = Commands(
            [
                Command("status", "Token, cache and cost totals so far", self._print_status),
                Command(
                    "effort",
                    f'Show or set the reasoning effort; "{_EFFORT_CLEAR}" sends none',
                    self._set_effort,
                    arguments="[level]",
                ),
                Command(
                    "model",
                    "Show or set the model, on this session's provider",
                    self._set_model,
                    arguments="[id]",
                ),
                Command(
                    "tools",
                    "Show the tools this session has, or choose them",
                    self._set_tools,
                    arguments="[names]",
                ),
                Command(
                    "skills",
                    "List the skills that apply, or choose them",
                    self._print_skills,
                    arguments="[names]",
                ),
                Command("clear", "Forget the conversation and start over", self._clear),
                Command("help", "List the commands", self._print_help),
                Command("exit", "Exit the session", self._exit, aliases=("quit",)),
            ]
        )
        self._menu = CommandMenu(self.commands, indent=len(GLYPH))
        self._session: PromptSession[str] = PromptSession(
            key_bindings=_bindings(self._toggle_thinking, self._menu.listing),
            style=style,
            history=_history(self.config),
        )
        self._menu.attach(self._session)
        self._agent: Agent | None = None
        self._renderer: EventRenderer | None = None
        #: Share of the context window the last request filled. None until one
        #: reports, since "0% of the window" before any run would be a lie.
        self._context_share: float | None = None
        #: Commands added for skills, so they can be dropped again when the
        #: selection changes.
        self._skill_commands: tuple[str, ...] = ()
        #: Rebuilds the system prompt around a skills list. The CLI sets it,
        #: being what knows the flags that shaped the prompt; None leaves the
        #: prompt as it was built.
        self.rebuild_prompt: Callable[[str], str] | None = None

    def attach(self, agent: Agent) -> None:
        """Bind the agent and start rendering its events."""
        self._agent = agent
        self._renderer = EventRenderer(self.console, self.config.terminal)
        agent.subscribe(self._renderer)
        self._sync_skill_commands(self._skills())

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("Terminal.attach(agent) must be called before run()")
        return self._agent

    @property
    def renderer(self) -> EventRenderer:
        if self._renderer is None:
            raise RuntimeError("Terminal.attach(agent) must be called before run()")
        return self._renderer

    def welcome(self) -> None:
        """The box at the top of the session: what is running, where, and what
        the keys do. Printed once, before anything else the session says — so
        it is a separate call from `run`, which a resumed session reaches only
        after its restored history has been printed."""
        model = self.agent.state.model
        theme = self.config.terminal.theme
        self.console.print(
            Panel(
                Text.assemble(
                    ("lab-harness", theme.accent),
                    (f" · {model.id} via {model.provider}\n", ""),
                    (f"{_short_home(self.agent.cwd)}\n", theme.muted),
                    (_TIPS, theme.muted),
                ),
                box=ROUNDED,
                border_style=theme.muted,
                expand=False,
                padding=(0, 1),
            )
        )

    async def run(self) -> int:
        while True:
            try:
                # patch_stdout keeps rich's output from colliding with the
                # prompt redraw while a run is streaming.
                with patch_stdout(raw=True), self._menu.offered():
                    line = await self._session.prompt_async(_PROMPT_MESSAGE, rprompt=self._rprompt)
            except KeyboardInterrupt:
                # Ctrl-C at an idle prompt clears the line rather than exiting.
                continue
            except EOFError:
                return 0

            text = line.strip()
            if not text:
                continue
            if text == "exit":
                return 0
            try:
                invoked = self.commands.parse(text)
            except UnknownCommand as exc:
                self.console.print(Text(str(exc), style=self.config.terminal.theme.warning))
                self.console.print()
                continue
            if invoked is None:
                await self._run_turn(text)
                continue
            command, arguments = invoked
            try:
                await command.run(arguments)
            except EndSession:
                return 0
            except Exception as exc:
                # As with a run: a failed command ends itself, not the session.
                self.console.print(Text(f"error: {exc}", style=self.config.terminal.theme.error))

    async def _run_turn(self, message: str) -> None:
        # The task exists before Ctrl-C is rerouted, so a callback queued by an
        # early Ctrl-C runs after the run has claimed its abort signal, not
        # before it, where `abort()` would find nothing to set.
        task = asyncio.create_task(self.agent.prompt(message))
        produced: list[Message] = []
        with _ctrl_c_calls(self._cancel_turn), _keys_wait_quietly(_stdin_fd()):
            try:
                produced = await asyncio.shield(task)
            except asyncio.CancelledError:
                self.agent.abort()
                raise
            except Exception as exc:
                # A crash in the runtime ends the turn, not the session.
                self.console.print(Text(f"error: {exc}", style=self.config.terminal.theme.error))
            finally:
                # A run that ended normally has already closed it. One that
                # crashed has not, and its region must not outlive it.
                self.renderer.close()
        summary = summarize(produced, self.config.models)
        self._context_share = context_share(summary, self.agent.state.model)
        if self.config.terminal.show_usage:
            self._print_footer(summary)
        self.console.print()

    def _cancel_turn(self) -> None:
        # Ask the run to stop and let `_run_turn` keep waiting: abort is
        # cooperative, and the loop still has to emit its aborted message.
        self.agent.abort()
        self.console.print(Text(_CANCELLED_NOTICE, style=self.config.terminal.theme.warning))

    def _toggle_thinking(self) -> None:
        if self._renderer is not None:
            self._renderer.toggle_thinking()

    def _rprompt(self) -> StyleAndTextTuples:
        """The status at the right of the input: the model, and how much of
        its window the last request filled."""
        if self._agent is None:
            return []
        share = self._context_share
        parts = [self._agent.state.model.id]
        if share is not None:
            parts.append(f"context {share:.0%}")
        full = share is not None and share >= self.config.terminal.context_warning
        return [(_STATUS_WARNING if full else _STATUS, " · ".join(parts))]

    def _print_footer(self, summary: UsageSummary) -> None:
        model = self.agent.state.model
        theme = self.config.terminal.theme
        footer = format_footer(summary, model, self.config.terminal.currency_symbol)
        if footer is None:
            return
        self.console.print(Text(footer, style=theme.muted))
        share = context_share(summary, model)
        if share >= self.config.terminal.context_warning:
            # Nothing compacts here, so say what the next step is instead.
            self.console.print(
                Text(
                    f"context {share:.0%} of the window — /clear to start over", style=theme.warning
                )
            )

    async def _print_status(self, _arguments: str) -> None:
        # The whole transcript, so a resumed session counts its history too.
        summary = summarize(self.agent.state.messages, self.config.models)
        lines = format_status(summary, self.agent.state.model, self.config.terminal.currency_symbol)
        self.console.print(Text("\n".join(lines)))
        self.console.print()

    async def _set_effort(self, level: str) -> None:
        """`/effort` shows the level, `/effort LEVEL` sets it for the runs after."""
        model = self.agent.state.model
        theme = self.config.terminal.theme
        if level:
            effort = "" if level == _EFFORT_CLEAR else level
            model = replace(model, reasoning_effort=effort)
            self.agent.state.model = model
        shown = model.reasoning_effort or f"{_EFFORT_CLEAR} (none sent)"
        self.console.print(Text(f"reasoning effort: {shown}", style=theme.muted))
        if model.reasoning_effort and not model.supports_reasoning:
            # The adapters send no level to a model flagged as not reasoning.
            self.console.print(
                Text(f"{model.id} does not reason; no level is sent", style=theme.warning)
            )
        self.console.print()

    async def _print_help(self, _arguments: str) -> None:
        self.console.print(
            Text("\n".join(self.commands.help_lines()), style=self.config.terminal.theme.muted)
        )
        self.console.print()

    async def _set_tools(self, names: str) -> None:
        """`/tools` lists what this session has; `/tools NAMES` replaces it.

        The same list `--tools` takes, changed between runs rather than at
        startup. The next run's `agent_start` carries the new declarations, so
        a recorded session opens a new segment there, as it does for `/model`.
        """
        theme = self.config.terminal.theme
        if not names:
            self.console.print(Text(_listing("tools", self._tool_names()), style=theme.muted))
            self.console.print()
            return
        try:
            tools = default_registry().select(_names(names))
        except ValueError as exc:  # a name no tool answers to
            self.console.print(Text(str(exc), style=theme.warning))
            self.console.print()
            return
        self.agent.state.tools = tools
        # The skills follow: they are loadable only while `skill` is among
        # these, and the prompt is what says which there are.
        self._refresh_skills()
        self.console.print(Text(_listing("tools", [t.name for t in tools]), style=theme.muted))
        self.console.print()

    async def _print_skills(self, names: str) -> None:
        """`/skills` lists those that apply; `/skills NAMES` chooses them.

        Choosing sets `skills.selected` for the rest of the session: what the
        `skill` tool loads, what the prompt lists, and what the `/` menu
        offers all read it.
        """
        theme = self.config.terminal.theme
        if names:
            self._select_skills(_names(names))
            return
        if not self.agent.config.skills.enabled:
            self.console.print(Text("Skills are off: skills.enabled is false", style=theme.muted))
            self.console.print()
            return
        found = self._skills()
        if not found:
            self.console.print(
                Text(f"No skills in {_where(self.agent.cwd, self.agent.config)}", style=theme.muted)
            )
            self.console.print()
            return
        for one in found:
            line = Text(f"{one.name} — {one.description}")
            if one.allowed_tools:
                line.append(f"   allows: {', '.join(one.allowed_tools)}", style=theme.muted)
            self.console.print(line)
        self.console.print(Text("Invoke one with /name, or ask for it by name.", style=theme.muted))
        self.console.print()

    def _select_skills(self, names: tuple[str, ...]) -> None:
        """Choose which skills apply, and bring the prompt and menu along."""
        theme = self.config.terminal.theme
        skills = self.agent.config.skills
        self.agent.config = replace(
            self.agent.config, skills=replace(skills, selected=names, enabled=bool(names))
        )
        self._refresh_skills()
        shown = [one.name for one in self._skills()]
        self.console.print(Text(_listing("skills", shown), style=theme.muted))
        if names and len(shown) < len(names):
            # Chosen and not there: said here as well as logged, since a name
            # that applied to nothing is otherwise invisible.
            self.console.print(Text(f"Not found: {', '.join(names)}", style=theme.warning))
        self.console.print()

    def _skills(self) -> tuple[Skill, ...]:
        """The skills that apply here: found, and selected."""
        return load_skills(self.agent.cwd, self.agent.config)

    def _offered_skills(self) -> tuple[Skill, ...]:
        """What the prompt may list: none unless the model can ask for one.

        The terminal's own menu and `/skills` are not so gated — invoking one
        from the keyboard is a message, not a tool call — but a prompt listing
        what the model cannot load is a promise unkept.
        """
        if not any(tool.name == skill.name for tool in self.agent.state.tools):
            return ()
        return self._skills()

    def _tool_names(self) -> list[str]:
        return [tool.name for tool in self.agent.state.tools]

    def _refresh_skills(self) -> None:
        """Rebuild the prompt and the menu around the skills that apply now.

        Both say what can be asked for, so both follow a change to the tools
        or to the selection: a menu offering one that is not selected, or a
        prompt listing one the model cannot load, is a promise unkept.
        """
        if self.rebuild_prompt is not None:
            self.agent.state.system_prompt = self.rebuild_prompt(index(self._offered_skills()))
        self._sync_skill_commands(self._skills())

    def _sync_skill_commands(self, skills: Iterable[Skill]) -> None:
        """One command per skill, so `/commit` runs one the way Claude Code does.

        Synced rather than added once, because the selection can change
        mid-session: a skill dropped from it loses its command. A name already
        taken, by `/help` for instance, is skipped with a warning — a menu
        entry that silently shadowed a built-in would be worse than one that
        is missing.
        """
        for name in self._skill_commands:
            self.commands.discard(name)
        added: list[str] = []
        for found in skills:
            try:
                self.commands.add(
                    Command(
                        found.name,
                        found.description,
                        self._run_skill(found.name),
                        arguments="[request]",
                    )
                )
            except ValueError as exc:  # not a command name, or one we already have
                _log.warning("No command for skill %s: %s", found.name, exc)
                continue
            added.append(found.name)
        self._skill_commands = tuple(added)

    def _run_skill(self, name: str) -> Callable[[str], Awaitable[None]]:
        """`/name`: send the skill's instructions as the next turn's message.

        Read again at invocation rather than kept from startup, so a skill
        edited mid-session is the one that runs.
        """

        async def run(request: str) -> None:
            theme = self.config.terminal.theme
            found = {one.name: one for one in load_skills(self.agent.cwd, self.config)}
            if name not in found:
                self.console.print(
                    Text(f"No skill named /{name} — /skills lists them", style=theme.warning)
                )
                self.console.print()
                return
            typed = f"/{name}" + (f" {request}" if request else "")
            # Echoed as a typed line, so the run does not start out of nowhere.
            self.console.print(Text.assemble((GLYPH, theme.muted), (typed, "")))
            message = f"/{name}\n\n{found[name].body}"
            if request:
                message += f"\n\n{request}"
            await self._run_turn(message)

        return run

    async def _clear(self, _arguments: str) -> None:
        """`/clear` forgets the conversation; the transcript keeps what came before."""
        self.agent.restore(())
        self._context_share = None
        self.console.print(
            Text(
                "Conversation cleared; the transcript keeps what came before.",
                style=self.config.terminal.theme.muted,
            )
        )
        self.console.print()

    async def _set_model(self, wanted: str) -> None:
        """`/model` shows the model; `/model ID` switches to one this session's
        provider serves, since the provider is chosen when the agent is built."""
        model = self.agent.state.model
        theme = self.config.terminal.theme
        if not wanted:
            self._print_models(model)
            return
        if wanted not in self.config.models:
            self.console.print(
                Text(f"{wanted} is not a configured model; /model lists them", style=theme.warning)
            )
            self.console.print()
            return
        served = select_provider(wanted, self.config)
        if served != model.provider:
            self.console.print(
                Text(
                    f"{wanted} is served by {served}, but this session is on {model.provider}"
                    " — start a new one with --model",
                    style=theme.warning,
                )
            )
            self.console.print()
            return
        # The session's provider, not the catalog's: --provider may have
        # pointed this id somewhere else when the agent was built.
        self.agent.state.model = replace(
            resolve_model(wanted, self.config), provider=model.provider
        )
        self.console.print(self._model_line(self.agent.state.model))
        self.console.print()

    def _print_models(self, model: Model) -> None:
        """`/model` with no argument: what is in use, and what it could become."""
        theme = self.config.terminal.theme
        self.console.print(self._model_line(model))
        others = sorted(
            id
            for id, entry in self.config.models.items()
            if id != model.id and entry.provider == model.provider
        )
        if others:
            self.console.print(
                Text(f"others on {model.provider}: {', '.join(others)}", style=theme.muted)
            )
        elsewhere = sum(
            1 for entry in self.config.models.values() if entry.provider != model.provider
        )
        if elsewhere:
            self.console.print(
                Text(f"{elsewhere} on other providers — --model ID to use one", style=theme.muted)
            )
        self.console.print()

    def _model_line(self, model: Model) -> Text:
        return Text(
            f"model: {model.id} via {model.provider}", style=self.config.terminal.theme.muted
        )

    async def _exit(self, _arguments: str) -> None:
        raise EndSession

    async def ask_approval(self, tool_name: str, arguments: dict[str, object]) -> str:
        """Prompt for tool approval. Anything but an explicit yes denies."""
        question = f"Allow {display_name(tool_name)}? [y]es / [n]o / [a]lways "
        with self._renderer.suspended() if self._renderer is not None else _nothing():
            if self._renderer is not None:
                self._renderer.show_call(tool_name, arguments)
            try:
                with patch_stdout(raw=True):
                    answer = await self._session.prompt_async(question)
            except (KeyboardInterrupt, EOFError):
                # Interrupting the question is a refusal, not an approval.
                return "no"
        return _ANSWERS.get(answer.strip().lower()[:1], "no")


class LogStream:
    """Where log records go when stderr is the terminal the console draws on.

    Written there directly, a record lands inside the live region: the next
    redraw overwrites it and leaves a stale line of the region behind. Printed
    through the console, it goes above the region like any other output.
    """

    def __init__(self, console: Console, style: str) -> None:
        self._console = console
        self._style = style

    def write(self, text: str) -> int:
        if record := text.rstrip("\n"):
            # Soft-wrapped: folded at the edge, not broken and indented by rich.
            self._console.print(Text(record, style=self._style), soft_wrap=True)
        return len(text)

    def flush(self) -> None:
        """Nothing to flush: each record is printed as it is written."""


def _bindings(toggle_thinking: Callable[[], object], listing: Filter) -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape", "enter", filter=~listing)
    def _newline(event: object) -> None:
        """Alt-Enter inserts a newline instead of submitting.

        Off while the command list is up, because there Esc alone closes it:
        with this one still active, prompt_toolkit would hold the Escape and
        wait for the rest of the sequence.
        """
        event.current_buffer.insert_text("\n")  # type: ignore[attr-defined]

    @bindings.add("c-o")
    def _thinking(event: object) -> None:
        """Ctrl-O toggles thinking. Ours is added last, so it wins over
        prompt_toolkit's emacs default for the key."""
        # Prints through the prompt rather than over it.
        run_in_terminal(toggle_thinking)

    return bindings


def _history(config: Config) -> History:
    """Where typed lines are kept, so Up recalls them in a later session.

    A file under the session home, which nothing has created yet on a first
    run — hence the directory here. In memory when the session config says
    nothing is persisted.
    """
    if not config.session.enabled:
        return InMemoryHistory()
    path = lab_harness_home(config.session) / config.session.history_file
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(path))


def _short_home(path: str) -> str:
    """`~/projects/my-project` rather than the whole path. Display only."""
    home = str(Path.home())
    return path.replace(home, "~", 1) if path.startswith(home) else path


def _names(value: str) -> tuple[str, ...]:
    """`/tools read, ls` as names. The word `none` is none, as for `--tools`."""
    text = value.strip()
    return () if text == _NONE else tuple(n for part in text.split(",") if (n := part.strip()))


def _listing(what: str, names: Iterable[str]) -> str:
    """`tools: read, ls`, or `tools: none`."""
    listed = ", ".join(names)
    return f"{what}: {listed}" if listed else f"{what}: none"


def _where(cwd: str, config: Config) -> str:
    """Where skills are looked for, for a message that names both."""
    return " or ".join(_short_home(str(directory)) for directory in directories(cwd, config))


@contextmanager
def _ctrl_c_calls(callback: Callable[[], object]) -> Iterator[None]:
    """Route Ctrl-C to `callback` for the duration, then restore the old handler.

    Needed because asyncio.run installs a handler of its own that cancels the
    main task, which would end the session from inside any turn.

    The handler only queues `callback` on the loop. A signal handler runs
    between any two bytecodes, and the loop may be asleep in select();
    `call_soon_threadsafe` both defers to a safe point and wakes it, which is
    how asyncio.run's own handler works.

    It is a Python-level handler rather than `loop.add_signal_handler`, because
    prompt_toolkit replaces and then removes the loop's SIGINT handler around
    every prompt, so one installed that way would be gone after the first
    approval question. What prompt_toolkit saves and restores is the
    Python-level handler, which is what this sets.
    """
    if threading.current_thread() is not threading.main_thread():
        # Only the main thread may handle signals. Run elsewhere, Ctrl-C keeps
        # whatever meaning it already had.
        yield
        return
    loop = asyncio.get_running_loop()
    previous = signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(callback))
    try:
        yield
    finally:
        # None means the old handler was not set from Python, so there is
        # nothing to hand back; Python's default is the nearest equivalent.
        signal.signal(signal.SIGINT, signal.default_int_handler if previous is None else previous)


@contextmanager
def _keys_wait_quietly(fd: int | None) -> Iterator[None]:
    """Keep keys pressed during a run off the screen and out of its output,
    for the duration, then restore the terminal.

    A run reads no keys, so the terminal is cooked for its whole length, and a
    cooked terminal acts on two of them itself. It echoes what is typed, into
    the live region: the next redraw erases it, but a line that wraps makes
    the region a line taller than rich drew, and each redraw after that
    leaves its top line behind. And on macOS and the BSDs Ctrl-O is DISCARD,
    which silently drops the program's output until it is pressed again —
    the key the collapsed-thinking line says to press.

    So echo is off and DISCARD disabled. Either way the keypress stays in the
    input queue, where the next prompt reads it: typed-ahead text appears
    there, and Ctrl-O is the toggle. Only DISCARD, rather than every extended
    character: turning off IEXTEN would also take Ctrl-W and Ctrl-V away from
    text typed ahead mid-run.
    """
    termios = _termios()
    if termios is None or fd is None or not os.isatty(fd):
        yield
        return
    saved = termios.tcgetattr(fd)
    changed = [*saved[:6], list(saved[6])]
    changed[3] &= ~termios.ECHO
    changed[6][termios.VDISCARD] = bytes([os.fpathconf(fd, "PC_VDISABLE")])
    # TCSANOW rather than TCSAFLUSH, which would throw away typed-ahead input.
    termios.tcsetattr(fd, termios.TCSANOW, changed)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)


def _termios() -> Any:
    """The termios module, or None on a platform without one, or without DISCARD."""
    try:
        import termios
    except ImportError:
        return None
    return termios if hasattr(termios, "VDISCARD") else None


def _stdin_fd() -> int | None:
    try:
        return sys.stdin.fileno()
    except (OSError, ValueError, io.UnsupportedOperation):
        # Replaced by something that is not a file — a test runner's capture.
        return None


@contextmanager
def _nothing() -> Iterator[None]:
    yield


def build_approval_gate(terminal: Terminal, config: Config) -> ApprovalGate:
    """An approval gate that asks through this terminal."""
    return ApprovalGate(config.tools.approval, terminal.ask_approval)
