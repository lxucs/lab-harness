"""Export a saved session as JSONL: one line per request it made.

    uv run lab-harness -p "What does this project do?"     # record a session
    uv run python examples/trajectories.py > requests.jsonl
    uv run python examples/trajectories.py <session-id>    # a given session; a prefix works
    uv run python examples/trajectories.py --wire          # as the provider received it

Each line is exactly what the provider was sent — system prompt, messages,
tool declarations — beside what came back, rebuilt from the transcript by
`recorded_requests`. That holds across resumes and a changed prompt, model or
tool set, because each invocation records what it ran with. No API key needed:
this only reads files. More in docs/sessions.md.

`--wire` writes each request as the JSON body the adapter posted — for a Chat
Completions endpoint, OpenAI-format `messages` and `tools`, which is what the
server applied its chat template to and what training frameworks read — and
the reply as the same dialect writes it into the next request, reasoning
included. The adapter rebuilds both under the provider settings the session
recorded, not today's.
"""

import argparse
import json
import sys

from lab_harness.config import from_snapshot, load_config
from lab_harness.llm.providers import build_provider
from lab_harness.llm.serde import to_dict
from lab_harness.session import (
    RecordedRequest,
    find_session,
    latest_session,
    read_transcript,
    recorded_requests,
)


def main(session_id: str | None, *, wire: bool = False) -> int:
    config = load_config()
    try:
        if session_id:
            path = find_session(".", session_id, config.session)
        else:
            path = latest_session(".", config.session)
    except ValueError as exc:  # an ambiguous id prefix
        print(exc, file=sys.stderr)
        return 1
    if path is None:
        print("No saved session for this directory. Run lab-harness here first.", file=sys.stderr)
        return 1

    for step in recorded_requests(read_transcript(path)):
        # Errored and aborted turns are steps too; keep the ones that finished.
        if step.response.stop_reason in ("error", "aborted"):
            continue
        print(json.dumps(as_sent(step) if wire else as_recorded(step)))
    return 0


def as_recorded(step: RecordedRequest) -> dict[str, object]:
    return {
        "timestamp": step.timestamp,
        "model": step.model.id,
        "request": to_dict(step.request),
        "response": to_dict(step.response),
    }


def as_sent(step: RecordedRequest) -> dict[str, object]:
    adapter = build_provider(step.model.provider, from_snapshot(step.config))
    return {
        "timestamp": step.timestamp,
        "model": step.model.id,
        "provider": step.model.provider,
        "request": adapter.request_body(step.model, step.request),
        "reply": adapter.reply_entries(step.response),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a saved session's requests as JSONL.")
    parser.add_argument("session_id", nargs="?", help="a session id, or a unique prefix of one")
    parser.add_argument("--wire", action="store_true", help="as the provider received it")
    args = parser.parse_args()
    raise SystemExit(main(args.session_id, wire=args.wire))
