"""Logging setup: precedence, isolation, and what must never reach a record.

The level resolves the same way everything else in the project does — flag,
then environment, then config file — and the tests below pin each rung of that
ladder separately, because a precedence bug reads as "the setting does
nothing" rather than as a failure.

Records are captured by handing `configure_logging` a `StringIO` rather than by
using `caplog`. That runs the real handler and the real formatter, so these
tests assert on the bytes a user would actually see.
"""

import asyncio
import io
import logging
from pathlib import Path

import pytest

from lab_harness.agent import Agent
from lab_harness.agent.types import ExecutionMode, ToolContext, ToolResult
from lab_harness.config import Config, ConfigError, LoggingConfig, ProjectConfig, load_config
from lab_harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from lab_harness.llm.providers import build_provider, select_provider
from lab_harness.llm.providers._trace import request_shape
from lab_harness.llm.providers.anthropic import AnthropicProvider
from lab_harness.llm.providers.chat_completions import ChatCompletionsProvider
from lab_harness.llm.registry import resolve_model
from lab_harness.llm.types import Context, ToolCallBlock, UserMessage, text
from lab_harness.logs import LEVELS, ROOT_LOGGER, configure_logging, resolve_level
from lab_harness.prompts import load_project_context
from tests.http_fakes import Recorder


def sink() -> io.StringIO:
    return io.StringIO()


def emit(level: str = "warning", message: str = "hello") -> None:
    getattr(logging.getLogger(f"{ROOT_LOGGER}.sample"), level)(message)


# --- level resolution -----------------------------------------------------


def test_the_default_level_is_warning() -> None:
    """A run that works should print nothing but its answer."""
    assert resolve_level() == "warning"


def test_an_explicit_argument_wins_over_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_HARNESS_LOG_LEVEL", "error")
    assert resolve_level("debug", LoggingConfig(level="info")) == "debug"


def test_the_environment_beats_the_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_HARNESS_LOG_LEVEL", "debug")
    assert resolve_level(None, LoggingConfig(level="error")) == "debug"


def test_the_config_file_is_used_when_nothing_overrides_it() -> None:
    assert resolve_level(None, LoggingConfig(level="info")) == "info"


def test_the_env_var_name_comes_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The variable's *name* is a config key, so a deployment can rename it."""
    monkeypatch.setenv("OTHER_LEVEL", "debug")
    assert resolve_level(None, LoggingConfig(level_env="OTHER_LEVEL")) == "debug"


def test_the_default_env_var_is_ignored_once_renamed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_HARNESS_LOG_LEVEL", "debug")
    assert resolve_level(None, LoggingConfig(level_env="OTHER_LEVEL")) == "warning"


def test_an_empty_env_var_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LAB_HARNESS_LOG_LEVEL=` in a shell profile must not be a startup error."""
    monkeypatch.setenv("LAB_HARNESS_LOG_LEVEL", "")
    assert resolve_level(None, LoggingConfig(level="info")) == "info"


def test_a_whitespace_env_var_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_HARNESS_LOG_LEVEL", "   ")
    assert resolve_level(None, LoggingConfig(level="info")) == "info"


@pytest.mark.parametrize("written", ["DEBUG", "Debug", " debug "])
def test_level_names_are_case_and_space_insensitive(
    written: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env vars get typed in caps; that should not be a failure."""
    monkeypatch.setenv("LAB_HARNESS_LOG_LEVEL", written)
    assert resolve_level() == "debug"


def test_an_unknown_level_raises_rather_than_falling_back() -> None:
    with pytest.raises(ConfigError) as caught:
        resolve_level("verbose")
    message = str(caught.value)
    assert "verbose" in message
    # The message has to list the way out, like the config errors do.
    assert all(name in message for name in LEVELS)


def test_an_unknown_level_in_a_config_file_raises() -> None:
    with pytest.raises(ConfigError):
        resolve_level(None, LoggingConfig(level="loud"))


# --- handler behavior -----------------------------------------------------


def test_records_at_or_above_the_level_are_written() -> None:
    stream = sink()
    configure_logging("warning", stream=stream)
    emit("warning", "visible")
    assert "visible" in stream.getvalue()


def test_records_below_the_level_are_dropped() -> None:
    stream = sink()
    configure_logging("warning", stream=stream)
    emit("debug", "quiet")
    assert stream.getvalue() == ""


def test_debug_turns_the_quiet_records_on() -> None:
    stream = sink()
    configure_logging("debug", stream=stream)
    emit("debug", "now visible")
    assert "now visible" in stream.getvalue()


def test_the_format_comes_from_config() -> None:
    stream = sink()
    configure_logging("warning", LoggingConfig(format="[%(levelname)s] %(message)s"), stream=stream)
    emit("warning", "formatted")
    assert stream.getvalue().strip() == "[WARNING] formatted"


def test_the_default_format_names_the_level_and_the_module() -> None:
    """The first question about a stray line is which module emitted it."""
    stream = sink()
    configure_logging("warning", stream=stream)
    emit("warning", "traceable")
    written = stream.getvalue()
    assert "WARNING" in written
    assert f"{ROOT_LOGGER}.sample" in written


def test_configuring_twice_does_not_double_the_output() -> None:
    """`main()` calls this twice; a user must not see every line twice."""
    stream = sink()
    configure_logging("warning", stream=stream)
    configure_logging("warning", stream=stream)
    emit("warning", "once")
    assert stream.getvalue().count("once") == 1


def test_the_second_call_can_change_the_level() -> None:
    """The config file takes over from the defaults the first call used."""
    stream = sink()
    configure_logging("warning", stream=stream)
    configure_logging(None, LoggingConfig(level="debug"), stream=stream)
    emit("debug", "late")
    assert "late" in stream.getvalue()


def test_configure_returns_the_level_it_applied() -> None:
    assert configure_logging("info", stream=sink()) == "info"


def test_a_foreign_handler_is_left_alone() -> None:
    """Only handlers this module installed are replaced."""
    foreign = logging.StreamHandler(sink())
    logger = logging.getLogger(ROOT_LOGGER)
    logger.addHandler(foreign)
    configure_logging("warning", stream=sink())
    assert foreign in logger.handlers


def test_records_do_not_propagate_to_the_root_logger() -> None:
    """Otherwise an embedding application's handler prints every line twice."""
    configure_logging("warning", stream=sink())
    assert logging.getLogger(ROOT_LOGGER).propagate is False


def test_other_libraries_are_not_turned_on_by_our_debug_level() -> None:
    """`--log-level debug` means this program, not httpx and anthropic too."""
    stream = sink()
    configure_logging("debug", stream=stream)
    logging.getLogger("httpx").debug("not ours")
    assert stream.getvalue() == ""


def test_importing_the_package_installs_no_handler() -> None:
    """A library configures nothing; only an application calls configure.

    This also proves the suite's own restore fixture works — every test above
    attaches a handler, and this one would see it otherwise.
    """
    assert logging.getLogger(ROOT_LOGGER).handlers == []


# --- what the records say -------------------------------------------------


def test_config_loading_is_traced_at_debug(tmp_path: Path) -> None:
    """The answer to "which config file won" has to be obtainable."""
    project = tmp_path / ".lab-harness"
    project.mkdir()
    (project / "config.toml").write_text("[agent]\nmax_turns = 7\n")
    stream = sink()
    configure_logging("debug", stream=stream)
    load_config(tmp_path)
    assert str(project / "config.toml") in stream.getvalue()


def test_info_names_the_model_in_use() -> None:
    """`info` has to earn its place: one line per significant decision.

    A level documented in the config schema that produces no output at all
    would be a hollow setting.
    """
    stream = sink()
    configure_logging("info", stream=stream)
    resolve_model("claude-opus-5")
    assert "claude-opus-5" in stream.getvalue()


def test_info_is_quieter_than_debug(tmp_path: Path) -> None:
    """The tiers have to differ, or the level is decoration."""
    at_info, at_debug = sink(), sink()
    configure_logging("info", stream=at_info)
    load_config(tmp_path)
    resolve_model("claude-opus-5")
    configure_logging("debug", stream=at_debug)
    load_config(tmp_path)
    resolve_model("claude-opus-5")

    assert at_info.getvalue()
    assert len(at_debug.getvalue()) > len(at_info.getvalue())
    # Config file tracing is debug detail; the model in use is not.
    assert "No config file at" not in at_info.getvalue()
    assert "No config file at" in at_debug.getvalue()


def test_an_unknown_model_warns_at_the_default_level() -> None:
    """A typo'd model id otherwise surfaces as an opaque API error later."""
    stream = sink()
    configure_logging("warning", stream=stream)
    resolve_model("claude-sonnet-6-typo")
    written = stream.getvalue()
    assert "claude-sonnet-6-typo" in written
    assert "WARNING" in written


def test_a_known_model_does_not_warn() -> None:
    stream = sink()
    configure_logging("warning", stream=stream)
    resolve_model("claude-sonnet-5")
    assert stream.getvalue() == ""


def test_truncated_project_instructions_warn(tmp_path: Path) -> None:
    """`-p` mode has no screen to print the truncation notice on."""
    (tmp_path / "AGENTS.md").write_text("x" * 100)
    stream = sink()
    configure_logging("warning", stream=stream)
    load_project_context(tmp_path, ProjectConfig(max_bytes=10))
    written = stream.getvalue()
    assert "AGENTS.md" in written
    assert "truncated" in written


def test_untruncated_project_instructions_do_not_warn(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("short")
    stream = sink()
    configure_logging("warning", stream=stream)
    load_project_context(tmp_path)
    assert stream.getvalue() == ""


# --- hygiene --------------------------------------------------------------


def test_credential_values_never_reach_a_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the *name* of the variable a credential came from may be logged.

    Debug logging is the obvious place for a key to leak into a file someone
    later pastes into an issue, so this asserts on the whole provider setup
    path at the loudest level.
    """
    secret = "sk-ant-not-a-real-key-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://private-gateway.internal.example")
    stream = sink()
    configure_logging("debug", stream=stream)

    AnthropicProvider()

    written = stream.getvalue()
    assert secret not in written
    assert "private-gateway" not in written
    # The names are the point: they say which source won without exposing it.
    assert "ANTHROPIC_API_KEY" in written
    assert "ANTHROPIC_BASE_URL" in written


def test_a_profile_names_where_its_key_and_endpoint_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same rule for an OpenAI-family profile, through a whole request."""
    secret = "sk-not-a-real-deepseek-key-0123456789"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://private-gateway.internal.example/v1")
    stream = sink()
    configure_logging("debug", stream=stream)
    profile = Config().providers.openai["deepseek"]
    provider = ChatCompletionsProvider(
        profile, "deepseek", client=Recorder("data: [DONE]").client()
    )

    asyncio.run(_drain(provider))

    written = stream.getvalue()
    assert "key from DEEPSEEK_API_KEY, endpoint from DEEPSEEK_BASE_URL" in written
    assert "Request to" in written, "the request line ran, and it must not leak either"
    assert secret not in written
    assert "private-gateway" not in written


async def _drain(provider: ChatCompletionsProvider) -> None:
    context = Context(system_prompt="", messages=(UserMessage(content=text("hi")),), tools=())
    async for _ in provider.stream(FAKE_MODEL, context):
        pass


def test_a_profile_with_no_key_says_so() -> None:
    stream = sink()
    configure_logging("debug", stream=stream)
    build_provider("deepseek", Config())
    assert "key from nowhere" in stream.getvalue()


# --- routing --------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "override", "reason"),
    [
        ("claude-opus-5", "deepseek", "set by --provider"),
        ("deepseek-flash", None, "from its models entry"),
        ("not-a-catalogued-model", None, "agent.provider"),
    ],
)
def test_info_says_which_rule_routed_the_model(
    model_id: str, override: str | None, reason: str
) -> None:
    """Three rungs decide the provider; a wrong endpoint needs the one that fired."""
    stream = sink()
    configure_logging("info", stream=stream)
    select_provider(model_id, Config(), override)
    written = stream.getvalue()
    assert repr(model_id) in written
    assert reason in written


def test_info_says_where_the_model_id_came_from(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = sink()
    configure_logging("info", stream=stream)
    resolve_model("claude-opus-5")
    monkeypatch.setenv("LAB_HARNESS_MODEL", "claude-sonnet-5")
    resolve_model()
    monkeypatch.delenv("LAB_HARNESS_MODEL")
    resolve_model()
    sources = [line.rsplit("from ", 1)[1] for line in stream.getvalue().splitlines()]
    assert sources == ["--model", "LAB_HARNESS_MODEL", "agent.model"]


# --- what a run traces ----------------------------------------------------
#
# The trace has to answer "what happened" without becoming a second copy of
# the conversation. Ids tie each line to the transcript, which has the rest.

#: Planted in a tool's argument, output and details. None may reach a record.
_ARGUMENT = "rm -rf /tmp/planted-argument"
_OUTPUT = "planted-tool-output"


class _Probe:
    name = "probe"
    description = "a tool with something private in every field"
    parameters: dict[str, object] = {"type": "object", "properties": {}}  # noqa: RUF012
    execution_mode: ExecutionMode = "parallel"

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        details: dict[str, object] = {"command": _ARGUMENT, "exit_code": 3, "truncated": True}
        return ToolResult(content=text(_OUTPUT), details=details)


def _run(level: str) -> str:
    stream = sink()
    configure_logging(level, stream=stream)
    call = ToolCallBlock(id="call_7", name="probe", arguments={"command": _ARGUMENT})
    replies = [tool_call_response([call]), text_response("done", thinking="hmm")]
    agent = Agent(
        model=FAKE_MODEL, provider=FakeProvider(replies), tools=(_Probe(),), config=Config()
    )
    asyncio.run(agent.prompt("go"))
    return stream.getvalue()


def test_debug_traces_each_turn_response_and_tool_call() -> None:
    written = _run("debug")

    assert "Turn 1/50: 1 message(s) to fake-model" in written
    assert "Turn 2/50: 3 message(s) to fake-model" in written
    assert "Response stop=tool_use tool_calls=1 in=10" in written
    assert "Response stop=stop thinking=3 text=4 in=10" in written
    # Argument names, so a line says what kind of call it was.
    assert "Tool call call_7: probe(command)" in written
    # Numbers and flags from the details; their strings stay out.
    assert "Tool call call_7: ok in" in written
    assert "exit_code=3 truncated=True" in written


def test_info_is_one_line_for_the_whole_run() -> None:
    written = _run("info")

    assert "Run finished: requests=2 tool_calls=1 failed=0 stop=stop in=20 out=10" in written
    assert "Turn 1/" not in written
    assert "Tool call" not in written


def test_the_trace_carries_no_arguments_output_or_string_details() -> None:
    written = _run("debug")

    assert written, "nothing was traced, so this proves nothing"
    assert _ARGUMENT not in written
    assert _OUTPUT not in written


def test_a_request_is_logged_by_shape_with_its_settings_spelled_out() -> None:
    planted = "private conversation text"
    request = {
        "model": "m",
        "stream": True,
        "messages": [{"role": "user", "content": planted}] * 3,
        "system": planted,
        "tools": [],
        "max_tokens": 8192,
        "thinking": {"type": "adaptive", "display": "summarized"},
    }

    shape = request_shape(request)

    settings = 'max_tokens=8192 thinking={"type":"adaptive","display":"summarized"}'
    assert shape == f"messages=3 system={len(planted)} tools=0 {settings}"


def test_a_request_key_nobody_classified_is_logged_by_size_alone() -> None:
    """Private by default, so a key added to a request later cannot leak content."""
    planted = "text in a field added after this was written"
    assert request_shape({"prediction": planted}) == f"prediction={len(planted)}"
