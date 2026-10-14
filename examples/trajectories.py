"""Export a saved session as JSONL: one line per request it made.

    uv run lab-harness -p "What does this project do?"     # record a session
    uv run python examples/trajectories.py > requests.jsonl
    uv run python examples/trajectories.py <session-id>    # a given session; a prefix works

Each line is exactly what the provider was sent — system prompt, messages,
tool declarations — beside what came back, rebuilt from the transcript by
`recorded_requests`. That holds across resumes and a changed prompt, model or
tool set, because each invocation records what it ran with. No API key needed:
this only reads files. More in docs/sessions.md.
"""

import json
import sys

from lab_harness.config import load_config
from lab_harness.llm.serde import to_dict
from lab_harness.session import find_session, latest_session, read_transcript, recorded_requests


def main(session_id: str | None) -> int:
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
        record = {
            "timestamp": step.timestamp,
            "model": step.model.id,
            "request": to_dict(step.request),
            "response": to_dict(step.response),
        }
        print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
