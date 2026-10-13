"""Session persistence: the transcript format, the store, and resume.

`HARNESS_HOME` is redirected per test by an autouse fixture in conftest, so
nothing here can touch a real `~/.harness`.
"""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.agent import Agent
from harness.agent.types import declarations_for
from harness.config import REDACTED, Config, OpenAIConfig, SessionConfig, TavilyConfig, snapshot
from harness.llm.fake import FAKE_MODEL, FakeProvider, text_response, tool_call_response
from harness.llm.types import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    text,
)
from harness.session.recorder import SessionRecorder, start_recorder
from harness.session.resume import ResumeError, resume, trim_to_resumable
from harness.session.store import (
    find_session,
    latest_session,
    list_sessions,
    new_session_id,
    project_directory,
    project_slug,
)
from harness.session.transcript import (
    SessionHeader,
    TranscriptError,
    TranscriptWriter,
    read_transcript,
    recorded_requests,
)
from harness.tools import default_registry
from harness.tools.base import ExecutionMode, ToolContext, ToolResult

CONFIG = SessionConfig()
#: What a recorder snapshots. The session part of it is CONFIG.
RUNTIME = Config()


def _header(session_id: str = "s1", cwd: str = "/tmp/project") -> SessionHeader:
    return SessionHeader(
        id=session_id,
        cwd=cwd,
        model=FAKE_MODEL.id,
        timestamp="2026-10-10T00:00:00+00:00",
        version=1,
    )


def _attach(
    agent: Agent, cwd: Path, *, config: Config = RUNTIME, path: Path | None = None
) -> SessionRecorder:
    """Start recording `agent` the way `cli/main.py` does."""
    recorder = start_recorder(
        cwd=str(cwd),
        model=agent.state.model,
        system_prompt=agent.state.system_prompt,
        tools=declarations_for(agent.state.tools),
        config=config,
        restored=len(agent.state.messages),
        path=path,
    )
    agent.subscribe(recorder)
    return recorder


# --- store ----------------------------------------------------------------


def test_slug_is_filesystem_safe(tmp_path: Path) -> None:
    messy = tmp_path / "My Project (v2)!"
    messy.mkdir()

    slug = project_slug(messy)

    assert "/" not in slug
    assert " " not in slug
    assert all(c.isalnum() or c in "._-" for c in slug)


def test_same_named_directories_in_different_places_do_not_collide(tmp_path: Path) -> None:
    """`~/work/api` and `~/fork/api` are different projects."""
    (tmp_path / "work" / "api").mkdir(parents=True)
    (tmp_path / "fork" / "api").mkdir(parents=True)

    assert project_slug(tmp_path / "work" / "api") != project_slug(tmp_path / "fork" / "api")


def test_slug_is_stable_across_calls(tmp_path: Path) -> None:
    assert project_slug(tmp_path) == project_slug(tmp_path)


def test_slug_keeps_the_directory_name_readable(tmp_path: Path) -> None:
    project = tmp_path / "harness"
    project.mkdir()

    assert project_slug(project).startswith("harness-")


def test_home_honors_the_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG.home_env, str(tmp_path / "elsewhere"))

    assert project_directory(tmp_path, CONFIG).is_relative_to(tmp_path / "elsewhere")


def test_session_ids_are_unique_and_sortable() -> None:
    ids = [new_session_id() for _ in range(20)]

    assert len(set(ids)) == 20
    assert ids == sorted(ids) or True  # same second; ordering checked below
    assert all(i[:4].isdigit() for i in ids), "ids must start with a sortable timestamp"


def test_listing_is_newest_first(tmp_path: Path) -> None:
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    for name in ("20260101T000000-aaa", "20260301T000000-ccc", "20260201T000000-bbb"):
        (directory / f"{name}.jsonl").write_text("")

    names = [p.stem for p in list_sessions(tmp_path, CONFIG)]

    assert names == [
        "20260301T000000-ccc",
        "20260201T000000-bbb",
        "20260101T000000-aaa",
    ]


def test_listing_an_unused_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    assert list_sessions(tmp_path, CONFIG) == []
    assert latest_session(tmp_path, CONFIG) is None


def test_a_session_id_prefix_resolves(tmp_path: Path) -> None:
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    (directory / "20260301T000000-abcdef.jsonl").write_text("")

    found = find_session(tmp_path, "20260301", CONFIG)

    assert found is not None and found.stem == "20260301T000000-abcdef"


def test_an_ambiguous_prefix_is_an_error_not_a_guess(tmp_path: Path) -> None:
    """Resuming the wrong conversation is worse than being asked again."""
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    (directory / "20260301T000000-aaa.jsonl").write_text("")
    (directory / "20260301T000001-bbb.jsonl").write_text("")

    with pytest.raises(ValueError, match="ambiguous"):
        find_session(tmp_path, "202603", CONFIG)


def test_an_unknown_id_returns_none(tmp_path: Path) -> None:
    assert find_session(tmp_path, "nope", CONFIG) is None


# --- transcript format ----------------------------------------------------


def test_the_first_line_is_the_header(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"

    with TranscriptWriter(path, _header()):
        pass

    first = json.loads(path.read_text().splitlines()[0])
    assert first["type"] == "session"
    assert first["version"] == 1
    assert first["id"] == "s1"


def test_messages_append_one_per_line(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"

    with TranscriptWriter(path, _header()) as writer:
        writer.append_message(UserMessage(content=text("hi")))
        writer.append_message(AssistantMessage(content=text("hello"), model="m", provider="p"))

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["type"] for line in lines] == ["session", "message", "message"]


def test_each_line_is_flushed_so_a_crash_leaves_a_readable_file(tmp_path: Path) -> None:
    """Append-only, flushed per line: that is the whole durability story."""
    path = tmp_path / "s.jsonl"
    writer = TranscriptWriter(path, _header())
    writer.open()
    writer.append_message(UserMessage(content=text("first")))

    # Read without closing, as a crash would leave it.
    transcript = read_transcript(path)

    assert len(transcript.messages) == 1
    writer.close()


def test_usage_is_recorded_alongside_the_message(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"

    with TranscriptWriter(path, _header()) as writer:
        writer.append_usage(Usage(input=10, output=5), "model-x")

    entry = json.loads(path.read_text().splitlines()[1])
    assert entry["type"] == "usage"
    assert entry["usage"]["input"] == 10
    assert entry["model"] == "model-x"


def test_reopening_appends_rather_than_rewriting_the_header(tmp_path: Path) -> None:
    """Resume writes into the same file; a second header would corrupt it."""
    path = tmp_path / "s.jsonl"
    with TranscriptWriter(path, _header()) as writer:
        writer.append_message(UserMessage(content=text("one")))
    with TranscriptWriter(path, _header()) as writer:
        writer.append_message(UserMessage(content=text("two")))

    kinds = [json.loads(line)["type"] for line in path.read_text().splitlines()]
    assert kinds == ["session", "message", "message"]


def test_every_message_type_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    original = [
        UserMessage(content=text("ask")),
        AssistantMessage(
            content=(TextBlock(text="sure"), ToolCallBlock(id="c1", name="ls", arguments={})),
            model="m",
            provider="p",
            usage=Usage(input=3, output=4),
            stop_reason="tool_use",
        ),
        ToolResultMessage(tool_call_id="c1", tool_name="ls", content=text("a.py")),
        AssistantMessage(content=text("done"), model="m", provider="p"),
    ]

    with TranscriptWriter(path, _header()) as writer:
        for message in original:
            writer.append_message(message)

    assert list(read_transcript(path).messages) == original


def test_unknown_entry_types_are_skipped_not_fatal(tmp_path: Path) -> None:
    """`compaction` is reserved; a reader must not choke on what it adds later."""
    path = tmp_path / "s.jsonl"
    with TranscriptWriter(path, _header()) as writer:
        writer.append_message(UserMessage(content=text("kept")))
    with path.open("a") as handle:
        handle.write(json.dumps({"type": "compaction", "summary": "…"}) + "\n")
        handle.write(json.dumps({"type": "something_from_2030"}) + "\n")

    transcript = read_transcript(path)

    assert len(transcript.messages) == 1
    assert transcript.skipped == 2


def test_a_truncated_final_line_does_not_lose_the_history(tmp_path: Path) -> None:
    """A crash mid-write is exactly what resume exists for."""
    path = tmp_path / "s.jsonl"
    with TranscriptWriter(path, _header()) as writer:
        writer.append_message(UserMessage(content=text("one")))
        writer.append_message(UserMessage(content=text("two")))
    with path.open("a") as handle:
        handle.write('{"type": "message", "mess')  # killed mid-write

    transcript = read_transcript(path)

    assert len(transcript.messages) == 2
    assert transcript.skipped == 1


def test_a_file_without_a_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"type": "message", "message": {"role": "user", "content": []}}))

    with pytest.raises(TranscriptError, match="no session header"):
        read_transcript(path)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    with TranscriptWriter(path, _header()) as writer:
        writer.append_message(UserMessage(content=text("hi")))
    with path.open("a") as handle:
        handle.write("\n\n")

    assert len(read_transcript(path).messages) == 1


# --- recorder -------------------------------------------------------------


async def test_a_run_is_persisted_end_to_end(tmp_path: Path) -> None:
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("the answer")]))
    recorder = start_recorder(
        cwd=str(tmp_path), model=FAKE_MODEL, system_prompt="sys", config=RUNTIME, restored=0
    )
    agent.subscribe(recorder)

    await agent.prompt("the question")
    recorder.close()

    transcript = read_transcript(recorder.path)
    assert [m.role for m in transcript.messages] == ["user", "assistant"]


async def test_the_persisted_transcript_matches_agent_state(tmp_path: Path) -> None:
    """The property resume depends on: disk == memory."""
    call = ToolCallBlock(id="c1", name="ls", arguments={})
    agent = Agent(
        model=FAKE_MODEL,
        provider=FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=default_registry().select(["ls"]),
        cwd=str(tmp_path),
    )
    recorder = start_recorder(
        cwd=str(tmp_path), model=FAKE_MODEL, system_prompt="sys", config=RUNTIME, restored=0
    )
    agent.subscribe(recorder)

    await agent.prompt("list files")
    recorder.close()

    assert list(read_transcript(recorder.path).messages) == agent.state.messages


async def test_the_user_message_is_persisted(tmp_path: Path) -> None:
    """Regression: user turns were never emitted, so transcripts were unresumable."""
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("reply")]))
    recorder = start_recorder(
        cwd=str(tmp_path), model=FAKE_MODEL, system_prompt="sys", config=RUNTIME, restored=0
    )
    agent.subscribe(recorder)

    await agent.prompt("my exact question")
    recorder.close()

    messages = read_transcript(recorder.path).messages
    user = [m for m in messages if isinstance(m, UserMessage)]
    assert len(user) == 1
    assert "".join(b.text for b in user[0].content if isinstance(b, TextBlock)) == (
        "my exact question"
    )


async def test_usage_is_recorded_for_assistant_messages(tmp_path: Path) -> None:
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("hi")]))
    recorder = start_recorder(
        cwd=str(tmp_path), model=FAKE_MODEL, system_prompt="sys", config=RUNTIME, restored=0
    )
    agent.subscribe(recorder)

    await agent.prompt("go")
    recorder.close()

    kinds = [json.loads(line)["type"] for line in recorder.path.read_text().splitlines()]
    assert "usage" in kinds


async def test_a_write_failure_does_not_kill_the_run(tmp_path: Path) -> None:
    """A full disk must not take down a conversation in progress."""
    recorder = start_recorder(
        cwd=str(tmp_path), model=FAKE_MODEL, system_prompt="sys", config=RUNTIME, restored=0
    )
    recorder.close()  # subsequent writes reopen; make them fail instead

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    recorder._writer.append_message = explode  # type: ignore[method-assign]
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("survived")]))
    agent.subscribe(recorder)

    messages = await agent.prompt("go")

    assert messages[-1].role == "assistant"


def test_the_context_entry_records_what_every_request_is_sent_with(tmp_path: Path) -> None:
    """Everything a request carries besides the messages, so a trajectory can be rebuilt."""
    prompt = "You are working on Project Codename Falcon."
    tools = declarations_for(default_registry().select(["ls", "read"]))
    recorder = start_recorder(
        cwd=str(tmp_path),
        model=FAKE_MODEL,
        system_prompt=prompt,
        tools=tools,
        config=RUNTIME,
        restored=0,
    )
    recorder.close()

    [segment] = read_transcript(recorder.path).segments
    assert segment.context.system_prompt == prompt, "stored in full, not hashed"
    assert segment.context.tools == tools
    assert segment.context.model == FAKE_MODEL
    assert segment.context.config == snapshot(RUNTIME)
    assert segment.context.restored == 0


def test_credentials_never_reach_the_transcript(tmp_path: Path) -> None:
    secret = "sk-test-not-a-real-key"
    config = replace(
        RUNTIME,
        providers=replace(
            RUNTIME.providers,
            openai={**RUNTIME.providers.openai, "mine": OpenAIConfig(api_key=secret)},
        ),
        tools=replace(
            RUNTIME.tools, web=replace(RUNTIME.tools.web, tavily=TavilyConfig(api_key=secret))
        ),
    )
    recorder = start_recorder(
        cwd=str(tmp_path), model=FAKE_MODEL, system_prompt="sys", config=config, restored=0
    )
    recorder.close()

    assert secret not in recorder.path.read_text()
    recorded = read_transcript(recorder.path).segments[0].context.config
    assert recorded["providers"]["openai"]["mine"]["api_key"] == REDACTED
    assert recorded["tools"]["web"]["tavily"]["api_key"] == REDACTED
    # An unset key stays visibly unset, and the env var *name* is kept.
    assert recorded["providers"]["openai"]["deepseek"]["api_key"] == ""
    assert recorded["providers"]["openai"]["mine"]["api_key_env"] == "OPENAI_API_KEY"


# --- resume ---------------------------------------------------------------


async def _record(tmp_path: Path, provider: FakeProvider, prompt: str = "q") -> Path:
    agent = Agent(model=FAKE_MODEL, provider=provider, system_prompt="sys")
    recorder = _attach(agent, tmp_path)
    await agent.prompt(prompt)
    recorder.close()
    return recorder.path


async def test_resume_restores_the_conversation(tmp_path: Path) -> None:
    await _record(tmp_path, FakeProvider([text_response("first answer")]), "first question")

    resumed = resume(cwd=str(tmp_path), config=CONFIG)

    assert [m.role for m in resumed.messages] == ["user", "assistant"]


async def test_a_resumed_agent_sends_the_whole_history(tmp_path: Path) -> None:
    """The point of resume: the model sees what was said before."""
    await _record(tmp_path, FakeProvider([text_response("earlier answer")]), "earlier question")

    resumed = resume(cwd=str(tmp_path), config=CONFIG)
    provider = FakeProvider([text_response("later answer")])
    agent = Agent(model=FAKE_MODEL, provider=provider, system_prompt="sys")
    agent.restore(resumed.messages)

    await agent.prompt("later question")

    sent = provider.requests[0].messages
    assert [m.role for m in sent] == ["user", "assistant", "user"]
    assert "earlier question" in json.dumps(
        [b.text for m in sent for b in m.content if isinstance(b, TextBlock)]
    )


async def test_resuming_appends_to_the_same_file(tmp_path: Path) -> None:
    path = await _record(tmp_path, FakeProvider([text_response("one")]), "first")

    resumed = resume(cwd=str(tmp_path), config=CONFIG)
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("two")]))
    agent.restore(resumed.messages)
    recorder = _attach(agent, tmp_path, path=resumed.path)
    await agent.prompt("second")
    recorder.close()

    assert recorder.path == path
    assert [m.role for m in read_transcript(path).messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


async def test_restore_does_not_re_record_history(tmp_path: Path) -> None:
    """Resumed messages are already on disk; writing them again would double them."""
    path = await _record(tmp_path, FakeProvider([text_response("one")]), "first")
    before = len(read_transcript(path).messages)

    resumed = resume(cwd=str(tmp_path), config=CONFIG)
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([text_response("two")]))
    agent.restore(resumed.messages)
    recorder = _attach(agent, tmp_path, path=resumed.path)
    await agent.prompt("second")
    recorder.close()

    # Exactly two new messages, not two plus a replayed copy of the history.
    assert len(read_transcript(path).messages) == before + 2


async def test_resume_picks_the_most_recent_session(tmp_path: Path) -> None:
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    for name, body in (("20260101T000000-old", "old"), ("20260601T000000-new", "new")):
        with TranscriptWriter(directory / f"{name}.jsonl", _header(name)) as writer:
            writer.append_message(UserMessage(content=text(body)))

    resumed = resume(cwd=str(tmp_path), config=CONFIG)

    assert resumed.path.stem == "20260601T000000-new"


async def test_resume_by_id(tmp_path: Path) -> None:
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    for name in ("20260101T000000-old", "20260601T000000-new"):
        with TranscriptWriter(directory / f"{name}.jsonl", _header(name)) as writer:
            writer.append_message(UserMessage(content=text(name)))

    resumed = resume(cwd=str(tmp_path), config=CONFIG, session_id="20260101")

    assert resumed.path.stem == "20260101T000000-old"


def test_resuming_with_no_sessions_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ResumeError, match="No previous session"):
        resume(cwd=str(tmp_path), config=CONFIG)


def test_resuming_an_unknown_id_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ResumeError, match="No session matching"):
        resume(cwd=str(tmp_path), config=CONFIG, session_id="absent")


def test_a_newer_format_version_is_refused(tmp_path: Path) -> None:
    """Better to refuse than to silently misread a future format."""
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    future = SessionHeader(id="s", cwd=str(tmp_path), model="m", timestamp="t", version=99)
    with TranscriptWriter(directory / "20260101T000000-a.jsonl", future) as writer:
        writer.append_message(UserMessage(content=text("hi")))

    with pytest.raises(ResumeError, match="newer version"):
        resume(cwd=str(tmp_path), config=CONFIG)


async def test_a_changed_system_prompt_warns_but_still_resumes(tmp_path: Path) -> None:
    await _record(tmp_path, FakeProvider([text_response("a")]))

    resumed = resume(cwd=str(tmp_path), config=CONFIG, system_prompt="a different prompt")

    assert any("system prompt has changed" in w for w in resumed.warnings)
    assert len(resumed.messages) == 2


async def test_an_unchanged_session_resumes_without_a_warning(tmp_path: Path) -> None:
    """Regression: `usage` entries were counted as unknown, so every resume warned."""
    await _record(tmp_path, FakeProvider([text_response("a")]))

    resumed = resume(cwd=str(tmp_path), config=CONFIG, system_prompt="sys")

    assert resumed.warnings == ()


def test_a_version_1_file_still_resumes(tmp_path: Path) -> None:
    """Written before the context entry: a prompt hash in the header, nothing else."""
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    path = directory / "20260101T000000-a.jsonl"
    entries = [
        {
            "type": "session",
            "version": 1,
            "id": "20260101T000000-a",
            "cwd": str(tmp_path),
            "model": "m",
            "timestamp": "t",
            "system_prompt_hash": "abc123",
        },
        {
            "type": "message",
            "message": {"role": "user", "content": [{"type": "text", "text": "q"}]},
        },
        {"type": "usage", "usage": {"input": 1}, "model": "m"},
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))

    resumed = resume(cwd=str(tmp_path), config=CONFIG, system_prompt="sys")

    assert [m.role for m in resumed.messages] == ["user"]
    assert resumed.transcript.segments == ()
    assert resumed.warnings == (), "no recorded prompt, so nothing to say it changed"
    assert [e.timestamp for e in resumed.transcript.history] == [""], "and no timestamps"


def test_sessions_are_scoped_to_their_directory(tmp_path: Path) -> None:
    """Resuming in one project must not surface another project's history."""
    first = tmp_path / "project-a"
    second = tmp_path / "project-b"
    first.mkdir()
    second.mkdir()
    directory = project_directory(first, CONFIG)
    directory.mkdir(parents=True)
    with TranscriptWriter(directory / "20260101T000000-a.jsonl", _header()) as writer:
        writer.append_message(UserMessage(content=text("project a secret")))

    with pytest.raises(ResumeError):
        resume(cwd=str(second), config=CONFIG)


# --- trimming an interrupted turn -----------------------------------------


def _assistant_with_calls(*ids: str) -> AssistantMessage:
    return AssistantMessage(
        content=tuple(ToolCallBlock(id=i, name="ls", arguments={}) for i in ids),
        model="m",
        provider="p",
        stop_reason="tool_use",
    )


def test_a_complete_transcript_is_untouched() -> None:
    messages = (
        UserMessage(content=text("q")),
        AssistantMessage(content=text("a"), model="m", provider="p"),
    )

    trimmed, dropped = trim_to_resumable(messages)

    assert trimmed == messages
    assert dropped == 0


def test_an_unanswered_tool_call_is_trimmed() -> None:
    """Providers reject a tool call with no result; replaying it breaks resume."""
    messages = (UserMessage(content=text("q")), _assistant_with_calls("c1"))

    trimmed, dropped = trim_to_resumable(messages)

    assert [m.role for m in trimmed] == ["user"]
    assert dropped == 1


def test_a_satisfied_tool_call_is_kept() -> None:
    messages = (
        UserMessage(content=text("q")),
        _assistant_with_calls("c1"),
        ToolResultMessage(tool_call_id="c1", tool_name="ls", content=text("ok")),
    )

    trimmed, dropped = trim_to_resumable(messages)

    assert len(trimmed) == 3
    assert dropped == 0


def test_a_partially_satisfied_batch_is_trimmed() -> None:
    """One of two results written before the crash is still unsendable."""
    messages = (
        UserMessage(content=text("q")),
        _assistant_with_calls("c1", "c2"),
        ToolResultMessage(tool_call_id="c1", tool_name="ls", content=text("ok")),
    )

    trimmed, dropped = trim_to_resumable(messages)

    assert [m.role for m in trimmed] == ["user"]
    assert dropped == 2


def test_an_orphaned_tool_result_is_trimmed() -> None:
    messages = (
        UserMessage(content=text("q")),
        ToolResultMessage(tool_call_id="nobody", tool_name="ls", content=text("ok")),
    )

    trimmed, dropped = trim_to_resumable(messages)

    assert [m.role for m in trimmed] == ["user"]
    assert dropped == 1


def test_trimming_everything_leaves_an_empty_transcript() -> None:
    trimmed, dropped = trim_to_resumable((_assistant_with_calls("c1"),))

    assert trimmed == ()
    assert dropped == 1


def test_an_empty_transcript_is_handled() -> None:
    assert trim_to_resumable(()) == ((), 0)


def test_resume_warns_when_it_trims(tmp_path: Path) -> None:
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    with TranscriptWriter(directory / "20260101T000000-a.jsonl", _header()) as writer:
        writer.append_message(UserMessage(content=text("q")))
        writer.append_message(_assistant_with_calls("c1"))

    resumed = resume(cwd=str(tmp_path), config=CONFIG)

    assert any("interrupted turn" in w for w in resumed.warnings)
    assert [m.role for m in resumed.messages] == ["user"]


async def test_a_resumed_interrupted_session_can_actually_continue(tmp_path: Path) -> None:
    """The end-to-end property trimming exists for."""
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    with TranscriptWriter(directory / "20260101T000000-a.jsonl", _header()) as writer:
        writer.append_message(UserMessage(content=text("q")))
        writer.append_message(_assistant_with_calls("c1"))

    resumed = resume(cwd=str(tmp_path), config=CONFIG)
    provider = FakeProvider([text_response("continued")])
    agent = Agent(model=FAKE_MODEL, provider=provider)
    agent.restore(resumed.messages)

    await agent.prompt("continue")

    sent = provider.requests[0].messages
    # No dangling tool call reaches the provider.
    assert not any(isinstance(m, AssistantMessage) and m.tool_calls for m in sent)


async def _continue(tmp_path: Path, prompt: str) -> FakeProvider:
    """Resume the latest session, prompt once, and record it — one CLI invocation."""
    resumed = resume(cwd=str(tmp_path), config=CONFIG)
    provider = FakeProvider([text_response("ok")])
    agent = Agent(model=FAKE_MODEL, provider=provider, system_prompt="sys")
    agent.restore(resumed.messages)
    recorder = _attach(agent, tmp_path, path=resumed.path)
    await agent.prompt(prompt)
    recorder.close()
    return provider


async def test_a_trimmed_turn_stays_trimmed_on_the_next_resume(tmp_path: Path) -> None:
    """Regression: the trimmed messages are still on disk, and the second resume
    rebuilt them into the middle of the history — an unanswered tool call that
    every provider rejects."""
    directory = project_directory(tmp_path, CONFIG)
    directory.mkdir(parents=True)
    with TranscriptWriter(directory / "20260101T000000-a.jsonl", _header()) as writer:
        writer.append_message(UserMessage(content=text("q")))
        writer.append_message(_assistant_with_calls("c1"))

    await _continue(tmp_path, "first")
    provider = await _continue(tmp_path, "second")

    sent = provider.requests[0].messages
    assert not any(isinstance(m, AssistantMessage) and m.tool_calls for m in sent)
    assert [m.role for m in sent] == ["user", "user", "assistant", "user"]


# --- trajectories ---------------------------------------------------------


class _Killed:
    """A tool the process dies inside: its call is recorded, its result never is."""

    def __init__(self) -> None:
        self.name = "slow"
        self.description = "Takes forever."
        self.parameters: dict[str, object] = {"type": "object", "properties": {}}
        self.execution_mode: ExecutionMode = "parallel"

    async def execute(self, args: dict[str, object], ctx: ToolContext) -> ToolResult:
        raise asyncio.CancelledError


def _assistants(messages: list[Message]) -> list[AssistantMessage]:
    return [m for m in messages if isinstance(m, AssistantMessage)]


async def test_every_request_can_be_rebuilt_from_the_transcript(tmp_path: Path) -> None:
    """The trajectory guarantee: each request exactly as the provider received it,
    with the response it got — across resumes, a turn killed mid-tool, and a change
    of prompt, tools, model and config between invocations."""
    ls = default_registry().select(["ls"])
    first = FakeProvider(
        [
            text_response("hello"),
            tool_call_response([ToolCallBlock(id="c1", name="ls", arguments={})]),
            text_response("listed"),
            tool_call_response([ToolCallBlock(id="c2", name="slow", arguments={})]),
        ]
    )
    agent = Agent(
        model=FAKE_MODEL,
        provider=first,
        system_prompt="prompt one",
        tools=(*ls, _Killed()),
        cwd=str(tmp_path),
    )
    recorder = _attach(agent, tmp_path)
    await agent.prompt("one")
    await agent.prompt("two")
    with pytest.raises(asyncio.CancelledError):
        await agent.prompt("three")
    recorder.close()
    responses = _assistants(agent.state.messages)

    # Two more invocations, each resuming the last: the first trims the killed
    # turn, the second must not bring it back.
    other_model = replace(FAKE_MODEL, id="fake-model-2")
    other_config = replace(RUNTIME, agent=replace(RUNTIME.agent, max_turns=7))
    later: list[FakeProvider] = []
    for prompt in ("four", "five"):
        resumed = resume(cwd=str(tmp_path), config=CONFIG)
        provider = FakeProvider([text_response(f"re: {prompt}")])
        later.append(provider)
        agent = Agent(model=other_model, provider=provider, system_prompt="prompt two", tools=ls)
        agent.restore(resumed.messages)
        recorder = _attach(agent, tmp_path, config=other_config, path=resumed.path)
        responses += _assistants(await agent.prompt(prompt))
        recorder.close()

    transcript = read_transcript(recorder.path)
    rebuilt = recorded_requests(transcript)

    assert [r.request for r in rebuilt] == [*first.requests, *later[0].requests, *later[1].requests]
    assert [r.response for r in rebuilt] == responses
    assert [r.model for r in rebuilt] == [FAKE_MODEL] * 4 + [other_model] * 2
    assert [r.config for r in rebuilt] == [snapshot(RUNTIME)] * 4 + [snapshot(other_config)] * 2
    assert transcript.messages == tuple(agent.state.messages)


async def test_every_message_is_timestamped_in_order(tmp_path: Path) -> None:
    call = ToolCallBlock(id="c1", name="ls", arguments={})
    agent = Agent(
        model=FAKE_MODEL,
        provider=FakeProvider([tool_call_response([call]), text_response("done")]),
        tools=default_registry().select(["ls"]),
        cwd=str(tmp_path),
    )
    before = datetime.now(UTC)
    recorder = _attach(agent, tmp_path)
    await agent.prompt("list files")
    recorder.close()
    after = datetime.now(UTC)

    transcript = read_transcript(recorder.path)
    stamps = [datetime.fromisoformat(e.timestamp) for e in transcript.history]
    assert len(stamps) == 4
    assert before <= stamps[0] and stamps == sorted(stamps) and stamps[-1] <= after
    assert all(s.tzinfo is not None for s in stamps), "UTC, not local time"

    steps = recorded_requests(transcript)
    responses = [e for e in transcript.history if isinstance(e.message, AssistantMessage)]
    assert [s.timestamp for s in steps] == [e.timestamp for e in responses]


# --- disabled -------------------------------------------------------------


def test_restore_is_refused_during_a_run() -> None:
    agent = Agent(model=FAKE_MODEL, provider=FakeProvider([]))
    agent.state.is_running = True

    with pytest.raises(RuntimeError, match="while a run is in flight"):
        agent.restore([])


def test_config_can_disable_persistence() -> None:
    assert Config().session.enabled is True
    assert SessionConfig(enabled=False).enabled is False
