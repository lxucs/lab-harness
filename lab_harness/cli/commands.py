"""Slash commands: what the terminal does itself rather than send to the model.

A command is a name, a line saying what it does, and a handler given the rest
of the line. The terminal registers its own in a `Commands`, and everything
else reads that one registry — the dispatch that runs a line, and the menu
that lists commands as one is typed. Adding a command is registering it: it is
listed, completed and run with nothing else to change.

**The menu follows Claude Code.** A `/` at the start of the input lists every
command it could begin, by name or alias, each beside its description, with
the first highlighted. Up and Down move the highlight; Tab puts the
highlighted command in the input, ready for arguments; Enter runs it, so `/st`
and Enter is `/status`. Moving never edits the input — only Tab does.

**The list opens above the input**, which moves down to make room and back
up when it closes. On the bottom row of the terminal the room comes from
scrolling earlier output up, which nothing can scroll back down, so there the
input ends a few rows higher, with blank rows under it until the next output.

**A line is a command only if it looks like one:** a slash, a name, and
optionally arguments. A path such as `/tmp/notes.md` fails at its second slash
and goes to the model as a message; `/stauts` names nothing, and says so
rather than spending a request.
"""

import re
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, is_done
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.layout import ConditionalContainer, Dimension, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl

PREFIX = "/"
#: What a command name or alias may be.
_NAME = re.compile(r"[A-Za-z][\w-]*")
#: A line that invokes a command: its name, then any arguments after a space.
_INVOCATION = re.compile(rf"{PREFIX}({_NAME.pattern})(?:\s+(.*))?", re.DOTALL)
#: Rows the menu shows before it scrolls.
_MENU_ROWS = 10
#: Style classes for the menu; `menu_style` gives the rules.
_CURRENT = "class:command-menu.current"
_DESCRIPTION = "class:command-menu.description"


class EndSession(Exception):
    """Raised by a command to end the session, as `/exit` does."""


class UnknownCommand(Exception):
    """A line shaped like a command that names none."""


@dataclass(frozen=True, slots=True)
class Command:
    """One thing the terminal does itself."""

    #: Without the slash.
    name: str
    #: One line, shown beside the name in the menu.
    description: str
    #: Given the rest of the line, stripped — empty if there is none.
    run: Callable[[str], Awaitable[None]]
    #: What it takes, shown after the name in the menu: `[level]`.
    arguments: str = ""
    #: Other names that run it, also matched as one is typed.
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @property
    def label(self) -> str:
        """How the menu shows it: `/effort [level]`, `/exit (quit)`."""
        label = PREFIX + self.name
        if self.arguments:
            label += f" {self.arguments}"
        if self.aliases:
            label += f" ({', '.join(self.aliases)})"
        return label


class Commands:
    """The commands the terminal runs itself, in the order the menu lists them."""

    def __init__(self, commands: Iterable[Command] = ()) -> None:
        self._commands: list[Command] = []
        self._by_name: dict[str, Command] = {}
        for command in commands:
            self.add(command)

    def add(self, command: Command) -> None:
        for name in command.names:
            if not _NAME.fullmatch(name):
                raise ValueError(f"Not a command name: {name!r}")
            if name in self._by_name:
                raise ValueError(f"{PREFIX}{name} is already a command")
        self._commands.append(command)
        self._by_name.update(dict.fromkeys(command.names, command))

    def __iter__(self) -> Iterator[Command]:
        return iter(self._commands)

    def matching(self, text: str) -> list[Command]:
        """The commands `text` could be the start of, by name or alias.

        One that `text` names in full comes first, so Enter on a whole name
        runs that command rather than a longer one it begins. Empty for
        anything but a slash and the start of a name.
        """
        typed = text.removeprefix(PREFIX)
        if typed == text or (typed and not _NAME.fullmatch(typed)):
            return []
        found = [c for c in self._commands if any(n.startswith(typed) for n in c.names)]
        return sorted(found, key=lambda command: typed not in command.names)

    def parse(self, line: str) -> tuple[Command, str] | None:
        """The command `line` runs, and the arguments it is given.

        None if `line` is no command at all: a message, or a path. Raises
        UnknownCommand if it looks like one but names none.
        """
        match = _INVOCATION.fullmatch(line)
        if match is None:
            return None
        command = self._by_name.get(match[1])
        if command is None:
            raise UnknownCommand(f"Unknown command {PREFIX}{match[1]} — type {PREFIX} to list them")
        return command, (match[2] or "").strip()


class CommandMenu:
    """Lists `commands` above the prompt as one is typed, the way Claude Code does.

    Attached to a session once, and offered at the prompts run inside
    `offered` — the input prompt, rather than a question asked through the
    same session.
    """

    def __init__(self, commands: Commands, *, indent: int = 0) -> None:
        self._commands = commands
        #: Columns before each row: the prompt's width, to line up with the input.
        self._indent = indent
        self._session: PromptSession[str] | None = None
        self._offered = False
        #: Index, among the matching commands, of the highlighted one.
        self._highlight = 0

    def matching(self) -> list[Command]:
        """The commands the input could be the start of, if they are offered."""
        if not self._offered or self._session is None:
            return []
        return self._commands.matching(self._session.default_buffer.text)

    def rows(self) -> list[tuple[Command, bool]]:
        """What the list shows: each command, and whether it is highlighted."""
        matches = self.matching()
        top = max(0, self._highlight - _MENU_ROWS + 1)
        shown = matches[top : top + _MENU_ROWS]
        return [(command, top + i == self._highlight) for i, command in enumerate(shown)]

    @contextmanager
    def offered(self) -> Iterator[None]:
        """Offer the commands at the prompt run inside this."""
        self._offered, self._highlight = True, 0
        try:
            yield
        finally:
            self._offered = False

    def attach(self, session: PromptSession[str]) -> None:
        """Draw the list above `session`'s input, and give it the keys."""
        self._session = session
        session.default_buffer.on_text_changed += self._text_changed
        listing = Condition(lambda: bool(self.matching()))
        layout = session.app.layout
        layout.container = HSplit(
            [
                ConditionalContainer(
                    Window(
                        FormattedTextControl(self._fragments),
                        height=lambda: Dimension.exact(len(self.rows())),
                        dont_extend_height=True,
                        wrap_lines=False,
                    ),
                    filter=listing & ~is_done,
                ),
                layout.container,
            ]
        )
        # After the session's own, so these win while the list is up:
        # prompt_toolkit's Up and Down would walk the history, and its Enter
        # would submit what is typed.
        session.key_bindings = merge_key_bindings(
            [session.key_bindings or KeyBindings(), self._bindings(listing)]
        )

    def _text_changed(self, buffer: Buffer) -> None:
        # The matches are new, so the first is highlighted again.
        self._highlight = 0

    def _fragments(self) -> StyleAndTextTuples:
        rows = self.rows()
        width = max((len(command.label) for command, _ in rows), default=0) + 2
        fragments: StyleAndTextTuples = []
        for command, current in rows:
            if fragments:
                fragments.append(("", "\n"))
            fragments += [
                ("", " " * self._indent),
                (_CURRENT if current else "", command.label.ljust(width)),
                (_CURRENT if current else _DESCRIPTION, command.description),
            ]
        return fragments

    def _bindings(self, listing: Condition) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("down", filter=listing)
        def _down(event: KeyPressEvent) -> None:
            self._move(1)

        @bindings.add("up", filter=listing)
        def _up(event: KeyPressEvent) -> None:
            self._move(-1)

        @bindings.add("tab", filter=listing)
        def _complete(event: KeyPressEvent) -> None:
            event.current_buffer.document = Document(self._chosen() + " ")

        @bindings.add("enter", filter=listing)
        def _run(event: KeyPressEvent) -> None:
            buffer = event.current_buffer
            buffer.document = Document(self._chosen())
            buffer.validate_and_handle()

        return bindings

    def _move(self, step: int) -> None:
        """Move the highlight, wrapping at either end."""
        self._highlight = (self._highlight + step) % len(self.matching())

    def _chosen(self) -> str:
        """The highlighted command, as typed to run it."""
        return PREFIX + self.matching()[self._highlight].name


def menu_style(highlight: str) -> dict[str, str]:
    """prompt_toolkit style rules for the menu: descriptions dimmed, and the
    highlighted row in `highlight`."""
    return {
        "command-menu.description": "fg:ansibrightblack",
        "command-menu.current": highlight,
    }
