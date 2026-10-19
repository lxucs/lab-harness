"""The HTML session page: `lab-harness --export`.

Transcripts are written with the real `TranscriptWriter` and read back with
`read_transcript`, so the page is built from exactly what a session leaves on
disk. Assertions read the markup; the script and styles are checked only for
what keeps the page self-contained.
"""

import base64
import hashlib
import re
from dataclasses import replace
from pathlib import Path

from lab_harness.cli.export import export_html, render_html
from lab_harness.config import Config, PricingConfig, snapshot
from lab_harness.llm.fake import FAKE_MODEL
from lab_harness.llm.types import (
    AssistantMessage,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
    Usage,
    UserMessage,
    text,
)
from lab_harness.session import SessionContext, SessionHeader, TranscriptWriter, read_transcript
from tests.entries import model_entry

TOOLS = (
    ToolDeclaration(
        name="bash",
        description="Run a shell command.",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    ),
)


def _context(restored: int = 0, *, system_prompt: str = "Be brief.", tools=TOOLS) -> SessionContext:
    return SessionContext(
        timestamp="2026-10-10T09:00:00+00:00",
        restored=restored,
        system_prompt=system_prompt,
        tools=tools,
        model=FAKE_MODEL,
        config=snapshot(Config()),
    )


def _write(path: Path, *segments: tuple[SessionContext, tuple[Message, ...]]) -> Path:
    header = SessionHeader(
        id=path.stem,
        cwd="/path/to/project",
        model=FAKE_MODEL.id,
        timestamp="2026-10-10T09:00:00+00:00",
        version=3,
    )
    with TranscriptWriter(path, header) as writer:
        for context, messages in segments:
            writer.append_context(context)
            for message in messages:
                writer.append_message(message)
    return path


def _reply(*blocks: object, **fields: object) -> AssistantMessage:
    return AssistantMessage(content=blocks, model=FAKE_MODEL.id, provider="fake", **fields)  # type: ignore[arg-type]


def _page(tmp_path: Path, *messages: Message, config: Config | None = None) -> str:
    path = _write(tmp_path / "s1.jsonl", (_context(), messages))
    return render_html(read_transcript(path), config or Config())


def _main(page: str) -> str:
    """The conversation itself, without the turn index above it."""
    return page[page.index("<main>") :]


def _body(page: str) -> str:
    """The page without its script and styles: what the session put there."""
    return re.sub(r"<(script|style)>.*?</\1>", "", page, flags=re.DOTALL)


# --- self-contained and safe ----------------------------------------------


def test_the_page_loads_nothing_from_anywhere(tmp_path: Path) -> None:
    page = _page(tmp_path, UserMessage(content=text("hi")))

    assert "default-src 'none'" in page, "the policy must forbid every load by default"
    assert not re.search(r"<(link|iframe|object|embed)\b", page)
    assert not re.search(r"\bsrc=[\"']?(https?:)?//", page)
    assert page.count("<script") == 1, "one inline script, the template's own"


def test_the_policy_allows_exactly_the_templates_script(tmp_path: Path) -> None:
    """Hashed, not 'unsafe-inline': a script that got in some other way would not run."""
    page = _page(tmp_path, UserMessage(content=text("hi")))

    script = re.search(r"<script>(.*?)</script>", page, re.DOTALL)
    assert script is not None
    digest = base64.b64encode(hashlib.sha256(script.group(1).encode()).digest()).decode()
    assert f"script-src 'sha256-{digest}'" in page
    assert "unsafe-inline' ; script" not in page
    assert "script-src 'unsafe-inline'" not in page


def test_everything_from_the_session_is_escaped(tmp_path: Path) -> None:
    hostile = '<script>alert(1)</script><img src=x onerror="alert(2)">'
    call = ToolCallBlock(id="c1", name="bash", arguments={"command": hostile})
    page = _page(
        tmp_path,
        UserMessage(content=text(hostile)),
        _reply(ThinkingBlock(thinking=hostile), TextBlock(text=hostile), call),
        ToolResultMessage(tool_call_id="c1", tool_name="bash", content=text(hostile)),
        _reply(error_message=hostile, stop_reason="error"),
    )

    body = _body(page)
    assert "<script>alert" not in body
    assert "<img src=x" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body


def test_session_text_cannot_fill_a_template_slot(tmp_path: Path) -> None:
    page = _page(tmp_path, UserMessage(content=text("{{body}} and {{script_hash}}")))

    assert "{{body}} and {{script_hash}}" in page


def test_markdown_links_cannot_run_script(tmp_path: Path) -> None:
    page = _page(tmp_path, _reply(TextBlock(text="[click](javascript:alert(1))")))

    assert "javascript:" not in _body(page).replace("[click](javascript:alert(1))", "")


def test_an_image_is_inlined_and_an_odd_type_is_refused(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        UserMessage(
            content=(
                ImageBlock(data="aGk=", mime_type="image/png"),
                ImageBlock(data="aGk=", mime_type='text/html" onload="x'),
            )
        ),
    )

    assert 'src="data:image/png;base64,aGk="' in page
    assert "onload" not in _body(page)


# --- the conversation -----------------------------------------------------


def test_the_conversation_reads_in_order(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        UserMessage(content=text("first question")),
        _reply(TextBlock(text="first answer")),
        UserMessage(content=text("second question")),
        _reply(TextBlock(text="second answer")),
    )

    main = _main(page)
    order = [main.index(s) for s in ("first question", "first answer", "second question")]
    assert order == sorted(order)
    assert main.index("second question") < main.index("second answer")


def test_each_question_opens_a_turn_listed_in_the_index(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        UserMessage(content=text("first question")),
        _reply(TextBlock(text="a")),
        UserMessage(content=text("second question")),
        _reply(TextBlock(text="b")),
    )

    assert 'id="turn-1"' in page
    assert 'id="turn-2"' in page
    assert re.search(r'href="#turn-2".*?second question', page, re.DOTALL)


def test_answers_are_rendered_from_markdown(tmp_path: Path) -> None:
    page = _page(tmp_path, _reply(TextBlock(text="**bold** and `code`\n\n```py\nx = 1\n```")))

    assert "<strong>bold</strong>" in page
    assert "<code>code</code>" in page
    assert '<code class="language-py">x = 1' in page


def test_a_question_is_shown_as_typed(tmp_path: Path) -> None:
    """Plain text, not Markdown: the user typed it, the model did not write it."""
    page = _page(tmp_path, UserMessage(content=text("**not bold**")))

    assert "**not bold**" in page


def test_thinking_folds_and_says_how_long_it_was(tmp_path: Path) -> None:
    page = _page(
        tmp_path, _reply(ThinkingBlock(thinking="weigh the two options"), TextBlock(text="b"))
    )

    match = re.search(r'<details class="thinking">(.*?)</details>', page, re.DOTALL)
    assert match is not None
    assert "weigh the two options" in match.group(1)
    assert "4 words" in match.group(1)


def test_empty_thinking_leaves_nothing(tmp_path: Path) -> None:
    page = _page(
        tmp_path, _reply(ThinkingBlock(thinking="  ", signature="sig"), TextBlock(text="b"))
    )

    assert 'class="thinking"' not in _body(page)


def test_a_tool_call_carries_its_result(tmp_path: Path) -> None:
    call = ToolCallBlock(id="c1", name="bash", arguments={"command": "ls"})
    page = _page(
        tmp_path,
        _reply(call),
        ToolResultMessage(
            tool_call_id="c1", tool_name="bash", content=text("a.py\nb.py"), duration_ms=1500
        ),
    )

    card = re.search(r'<details class="tool-call ok">(.*?)</details>', page, re.DOTALL)
    assert card is not None
    assert "ls" in card.group(1)
    assert "a.py\nb.py" in card.group(1)
    assert "1.5s" in card.group(1)
    assert _body(page).count("a.py\nb.py") == 1, "the result is drawn once, under its call"


def test_a_failed_call_is_marked(tmp_path: Path) -> None:
    call = ToolCallBlock(id="c1", name="bash", arguments={"command": "false"})
    page = _page(
        tmp_path,
        _reply(call),
        ToolResultMessage(
            tool_call_id="c1", tool_name="bash", content=text("exit 1"), is_error=True
        ),
    )

    assert '<details class="tool-call error">' in page


def test_a_call_with_no_result_says_so(tmp_path: Path) -> None:
    """An interrupted turn: the call was made, nothing came back."""
    page = _page(tmp_path, _reply(ToolCallBlock(id="c1", name="bash", arguments={"command": "x"})))

    assert '<details class="tool-call pending">' in page
    assert "No result recorded" in page


def test_an_edit_shows_its_change_as_a_diff(tmp_path: Path) -> None:
    call = ToolCallBlock(
        id="c1", name="edit", arguments={"path": "a.py", "old": "x = 1", "new": "x = 2"}
    )
    page = _page(tmp_path, _reply(call))

    assert '<span class="del">-x = 1</span>' in page
    assert '<span class="add">+x = 2</span>' in page


def test_errors_and_cancellations_are_shown(tmp_path: Path) -> None:
    page = _page(
        tmp_path,
        _reply(error_message="HTTP 500: upstream said no", stop_reason="error"),
        _reply(TextBlock(text="partial"), stop_reason="aborted"),
    )

    assert "HTTP 500: upstream said no" in page
    assert "cancelled" in page


# --- segments, usage, and the overview --------------------------------------


def test_each_invocation_shows_what_it_ran_with(tmp_path: Path) -> None:
    first = (UserMessage(content=text("q")), _reply(TextBlock(text="a")))
    path = _write(
        tmp_path / "s1.jsonl",
        (_context(), first),
        (_context(2, system_prompt="Be thorough."), (UserMessage(content=text("q2")),)),
    )

    page = render_html(read_transcript(path), Config())

    assert "Be brief." in page
    assert "Be thorough." in page
    assert "Started" in page
    assert "Context changed" in page and "system prompt" in page
    assert "Run a shell command." in page, "the tool declarations are there to read"


def test_a_changed_effort_is_shown(tmp_path: Path) -> None:
    first = (UserMessage(content=text("q")), _reply(TextBlock(text="a")))
    harder = replace(_context(2), model=replace(FAKE_MODEL, reasoning_effort="high"))
    path = _write(
        tmp_path / "s1.jsonl", (_context(), first), (harder, (UserMessage(content=text("q2")),))
    )

    page = render_html(read_transcript(path), Config())

    assert "changed: reasoning effort" in page
    assert "effort high" in page


def test_a_plain_resume_is_called_one(tmp_path: Path) -> None:
    first = (UserMessage(content=text("q")), _reply(TextBlock(text="a")))
    path = _write(
        tmp_path / "s1.jsonl",
        (_context(), first),
        (_context(2), (UserMessage(content=text("q2")),)),
    )

    assert "Resumed" in render_html(read_transcript(path), Config())


def test_a_turn_resume_dropped_is_marked_where_it_happened(tmp_path: Path) -> None:
    interrupted = (
        UserMessage(content=text("q")),
        _reply(ToolCallBlock(id="c1", name="bash", arguments={"command": "sleep 99"})),
    )
    path = _write(
        tmp_path / "s1.jsonl",
        (_context(), interrupted),
        (_context(0), (UserMessage(content=text("again")),)),
    )

    page = render_html(read_transcript(path), Config())

    assert "dropped 2 message(s)" in page


def test_usage_and_cost_are_totalled_and_shown_per_reply(tmp_path: Path) -> None:
    usage = Usage(input=1_000, output=500, cache_read=1_000)
    price = PricingConfig(input=3.0, output=15.0, cache_read=0.3)
    config = replace(Config(), models={FAKE_MODEL.id: model_entry(price=price)})
    page = _page(
        tmp_path,
        UserMessage(content=text("q")),
        _reply(TextBlock(text="a"), usage=usage),
        config=config,
    )

    assert "2k in · 500 out · 50% cached" in page
    # 1000 * 3 + 500 * 15 + 1000 * 0.3, per million.
    assert "$0.0108" in page


def test_an_unpriced_model_shows_no_cost(tmp_path: Path) -> None:
    page = _page(tmp_path, _reply(TextBlock(text="a"), usage=Usage(input=10, output=5)))

    assert not re.search(r"\$\d", _body(page))
    assert ">Cost<" not in page


def test_an_empty_session_still_renders(tmp_path: Path) -> None:
    page = _page(tmp_path)

    assert "No messages" in page


# --- writing it -----------------------------------------------------------


def test_the_page_is_written_beside_its_transcript(tmp_path: Path) -> None:
    path = _write(tmp_path / "s1.jsonl", (_context(), (UserMessage(content=text("q")),)))
    before = path.read_bytes()

    target = export_html(path, Config())

    assert target == tmp_path / "s1.html"
    assert "q" in target.read_text(encoding="utf-8")
    assert path.read_bytes() == before, "exporting must never change the session"
