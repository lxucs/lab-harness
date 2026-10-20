"""Slash commands: the registry, and the menu that lists them as one is typed.

The menu's keys are driven through a real PromptSession on a pipe, so its
bindings are checked against prompt_toolkit's own rather than in isolation.
"""

from typing import cast

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.layout import ConditionalContainer, HSplit
from prompt_toolkit.output import DummyOutput

from lab_harness.cli.commands import Command, CommandMenu, Commands, UnknownCommand

DOWN = "\x1b[B"
UP = "\x1b[A"
ESCAPE = "\x1b"


async def _nothing(arguments: str) -> None:
    pass


def _command(name: str, description: str = "", **options: object) -> Command:
    return Command(name, description or f"does {name}", _nothing, **options)  # type: ignore[arg-type]


COMMANDS = Commands(
    [
        _command("status", "Totals so far"),
        _command("effort", "Show or set the effort", arguments="[level]"),
        _command("exit", "Exit the session", aliases=("quit",)),
    ]
)


def _names(commands: list[Command]) -> list[str]:
    return [command.name for command in commands]


# --- the registry ---------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("/", ["status", "effort", "exit"]),
        ("/e", ["effort", "exit"]),
        ("/ST", []),
        ("/eff", ["effort"]),
        ("/qu", ["exit"]),
        ("/x", []),
    ],
)
def test_commands_match_by_prefix_of_a_name_or_an_alias(typed: str, expected: list[str]) -> None:
    assert _names(COMMANDS.matching(typed)) == expected


def test_a_name_typed_in_full_comes_before_longer_ones_it_begins() -> None:
    commands = Commands([_command("resume"), _command("re")])

    assert _names(commands.matching("/re")) == ["re", "resume"]
    assert _names(commands.matching("/r")) == ["resume", "re"], "otherwise in order"


@pytest.mark.parametrize("text", ["", "status", "/tmp/notes.md", "/effort low", "/ ", "//"])
def test_text_that_cannot_begin_a_command_matches_none(text: str) -> None:
    assert COMMANDS.matching(text) == []


@pytest.mark.parametrize(
    ("line", "name", "arguments"),
    [
        ("/status", "status", ""),
        ("/effort low", "effort", "low"),
        ("/effort   high  ", "effort", "high"),
        ("/quit", "exit", ""),
        ("/effort two\nlines", "effort", "two\nlines"),
    ],
)
def test_a_command_line_parses_to_its_command_and_arguments(
    line: str, name: str, arguments: str
) -> None:
    parsed = COMMANDS.parse(line)

    assert parsed is not None
    assert (parsed[0].name, parsed[1]) == (name, arguments)


@pytest.mark.parametrize("line", ["hello", "/tmp/notes.md", "/tmp/notes.md what is this", "/ x"])
def test_a_message_or_a_path_is_not_a_command(line: str) -> None:
    assert COMMANDS.parse(line) is None


def test_a_line_shaped_like_a_command_that_names_none_is_unknown() -> None:
    with pytest.raises(UnknownCommand, match="/stauts"):
        COMMANDS.parse("/stauts")


def test_a_name_is_registered_once() -> None:
    commands = Commands([_command("exit", aliases=("quit",))])

    with pytest.raises(ValueError, match="/quit is already a command"):
        commands.add(_command("quit"))


@pytest.mark.parametrize("name", ["", "/status", "two words", "a/b", "1st"])
def test_a_name_must_be_one_a_line_can_invoke(name: str) -> None:
    with pytest.raises(ValueError, match="Not a command name"):
        Commands([_command(name)])


def test_the_menu_label_shows_arguments_and_aliases() -> None:
    labels = [command.label for command in COMMANDS]

    assert labels == ["/status", "/effort [level]", "/exit (quit)"]


def test_help_lines_align_every_description() -> None:
    assert COMMANDS.help_lines() == [
        "/status          Totals so far",
        "/effort [level]  Show or set the effort",
        "/exit (quit)     Exit the session",
    ]


# --- the menu -------------------------------------------------------------


def _session(
    pipe: PipeInput, commands: Commands = COMMANDS
) -> tuple[PromptSession[str], CommandMenu]:
    session: PromptSession[str] = PromptSession(input=pipe, output=DummyOutput())
    menu = CommandMenu(commands, indent=2)
    menu.attach(session)
    return session, menu


def _shown(menu: CommandMenu) -> list[tuple[str, bool]]:
    return [(command.label, current) for command, current in menu.rows()]


async def test_a_slash_lists_the_commands_it_could_begin_with_the_first_highlighted() -> None:
    with create_pipe_input() as pipe:
        session, menu = _session(pipe)
        with menu.offered():
            session.default_buffer.text = "/"
            assert _shown(menu) == [
                ("/status", True),
                ("/effort [level]", False),
                ("/exit (quit)", False),
            ]

            session.default_buffer.text = "/e"
            assert _shown(menu) == [("/effort [level]", True), ("/exit (quit)", False)]

            session.default_buffer.text = "/exit"
            assert _shown(menu) == [("/exit (quit)", True)], "a whole name stays listed"

            session.default_buffer.text = "/exit "
            assert _shown(menu) == [], "past the name, the list closes"


async def test_each_row_is_the_label_then_the_description_aligned_under_the_input() -> None:
    with create_pipe_input() as pipe:
        session, menu = _session(pipe)
        with menu.offered():
            session.default_buffer.text = "/e"
            text = "".join(fragment[1] for fragment in menu._fragments())

    assert text.splitlines() == [
        "  /effort [level]  Show or set the effort",
        "  /exit (quit)     Exit the session",
    ]


async def test_a_long_list_says_how_many_it_leaves_out() -> None:
    many = Commands([_command(f"c{i}", f"does {i}") for i in range(12)])
    with create_pipe_input() as pipe:
        session, menu = _session(pipe, many)
        with menu.offered():
            session.default_buffer.text = "/"
            assert menu.more() == 2
            last = "".join(fragment[1] for fragment in menu._fragments()).splitlines()[-1]
            # Ten rows plus the one saying what is left out.
            split = cast(HSplit, session.app.layout.container)
            window = cast(ConditionalContainer, split.children[0]).content
            height = window.preferred_height(100, 40).preferred

    assert last.strip() == "… 2 more"
    assert height == 11


async def test_nothing_is_listed_where_commands_are_not_offered() -> None:
    with create_pipe_input() as pipe:
        session, menu = _session(pipe)
        session.default_buffer.text = "/"

        assert menu.rows() == []


@pytest.mark.parametrize(
    ("typed", "submitted"),
    [
        ("/st\r", "/status"),
        ("/qu\r", "/exit"),
        ("/exit\r", "/exit"),
        # The highlight moves, wrapping at either end; the input stays as typed.
        ("/" + DOWN + DOWN + "\r", "/exit"),
        ("/" + DOWN * 3 + "\r", "/status"),
        ("/" + UP + "\r", "/exit"),
        ("/e" + DOWN + "\r", "/exit"),
        # Moving, then typing, highlights the first match again.
        ("/" + DOWN + "e\r", "/effort"),
        # Tab puts the command in the input, ready for its arguments.
        ("/ef\tlow\r", "/effort low"),
        # Esc closes the list, which stays closed until the input changes.
        ("/" + ESCAPE + "\r", "/"),
        ("/e" + ESCAPE + DOWN + "\r", "/e"),
        ("/e" + ESCAPE + "f\r", "/effort"),
        # Not the start of a command: submitted as typed.
        ("/effort low\r", "/effort low"),
        ("/tmp/notes.md\r", "/tmp/notes.md"),
        ("/x\r", "/x"),
        ("hello\r", "hello"),
    ],
)
async def test_keys_in_the_list(typed: str, submitted: str) -> None:
    with create_pipe_input() as pipe:
        session, menu = _session(pipe)
        pipe.send_text(typed)

        with menu.offered():
            assert await session.prompt_async("› ") == submitted  # noqa: RUF001


async def test_a_question_asked_through_the_same_session_is_answered_as_typed() -> None:
    """The approval question shares the session; it offers no commands."""
    with create_pipe_input() as pipe:
        session, _ = _session(pipe)
        pipe.send_text("/st\r")

        assert await session.prompt_async("Allow? ") == "/st"
