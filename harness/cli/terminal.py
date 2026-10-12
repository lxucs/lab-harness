"""The interactive terminal.

prompt_toolkit reads input (history, multi-line, key bindings); rich writes
output. Both operate on ordinary stdout, so the session lands in real terminal
scrollback instead of a full-screen app you cannot scroll back through.

Two behaviors worth knowing:

**Ctrl-C cancels the turn, not the session.** The first Ctrl-C during a run
aborts it and leaves the transcript valid to continue from. At an idle prompt it
clears the input line. Ctrl-D exits.

**Typing during a run is buffered here, not queued in the runtime.** The agent
refuses a second concurrent `prompt()` by design (see `agent.py`), so the
terminal holds the text and submits it when the run returns. That keeps the
"one run at a time" rule in one place rather than spread across layers.
"""

import asyncio
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.text import Text

from harness.agent import Agent
from harness.cli.approval import ApprovalGate
from harness.cli.renderers import EventRenderer
from harness.config import Config

_BANNER = "harness — Ctrl-C cancels a turn, Ctrl-D exits"
_PROMPT = "› "  # noqa: RUF001 — a deliberate prompt glyph, not a stray quote
_CANCELLED_NOTICE = "^C  turn cancelled"
#: Answers accepted at the approval prompt, by first letter. Deliberately no
#: empty-string entry: a bare Enter is not consent, it is a reflex. Anything
#: unlisted falls through to "no".
_ANSWERS = {"y": "yes", "n": "no", "a": "always"}


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
        self._session: PromptSession[str] = PromptSession(key_bindings=_bindings())
        self._agent: Agent | None = None

    def attach(self, agent: Agent) -> None:
        """Bind the agent and start rendering its events."""
        self._agent = agent
        agent.subscribe(EventRenderer(self.console, self.config.terminal))

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("Terminal.attach(agent) must be called before run()")
        return self._agent

    async def run(self) -> int:
        self.console.print(Text(_BANNER, style="dim"))
        while True:
            try:
                # patch_stdout keeps rich's output from colliding with the
                # prompt redraw while a run is streaming.
                with patch_stdout(raw=True):
                    line = await self._session.prompt_async(_PROMPT)
            except KeyboardInterrupt:
                # Ctrl-C at an idle prompt clears the line rather than exiting.
                continue
            except EOFError:
                return 0

            text = line.strip()
            if not text:
                continue
            if text in (":q", ":quit", "exit"):
                return 0

            await self._run_turn(text)

    async def _run_turn(self, message: str) -> None:
        task = asyncio.create_task(self.agent.prompt(message))
        while True:
            try:
                await asyncio.shield(task)
                break
            except KeyboardInterrupt:
                # Ask the run to stop, then keep waiting: abort is cooperative,
                # and the loop still has to emit its aborted message and return.
                self.agent.abort()
                self.console.print(Text(_CANCELLED_NOTICE, style="yellow"))
                continue
            except asyncio.CancelledError:
                self.agent.abort()
                raise
            except Exception as exc:
                # A crash in the runtime ends the turn, not the session.
                self.console.print(Text(f"error: {exc}", style="bold red"))
                break
        self.console.print()

    async def ask_approval(self, tool_name: str, arguments: dict[str, object]) -> str:
        """Prompt for tool approval. Anything but an explicit yes denies."""
        question = f"Allow {tool_name}? [y]es / [n]o / [a]lways "
        try:
            with patch_stdout(raw=True):
                answer = await self._session.prompt_async(question)
        except (KeyboardInterrupt, EOFError):
            # Interrupting the question is a refusal, not an approval.
            return "no"
        return _ANSWERS.get(answer.strip().lower()[:1], "no")


def _bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _newline(event: object) -> None:
        """Alt-Enter inserts a newline instead of submitting."""
        event.current_buffer.insert_text("\n")  # type: ignore[attr-defined]

    return bindings


def build_approval_gate(terminal: Terminal, config: Config) -> ApprovalGate:
    """An approval gate that asks through this terminal."""
    return ApprovalGate(config.tools.approval, terminal.ask_approval)
