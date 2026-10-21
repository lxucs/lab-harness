# Examples

How to use lab-harness from the terminal and from Python. Run everything from
the repository root, after `uv sync`.

| Example | Shows | Needs a key |
|---|---|---|
| [terminal.md](terminal.md) | The CLI: interactive, one-shot, JSONL events, sessions and viewing them, other models, the system prompt, debug logs | yes |
| [scripted.py](scripted.py) | The real loop and a real tool, driven by a scripted model | **no** |
| [quickstart.py](quickstart.py) | An agent in a few lines, with the CLI's system prompt: stream the answer, watch the tool calls | yes |
| [custom_tool.py](custom_tool.py) | A new tool as one decorated function, beside a few built-ins | yes |
| [hooks.py](hooks.py) | All four hooks: a tool budget, redaction, request logging, a self-check turn | yes |
| [trajectories.py](trajectories.py) | A saved session's requests as JSONL, for analysis or training — `--wire` as the endpoint received them | **no** |

```bash
uv run python examples/scripted.py                    # no key: try this first
export ANTHROPIC_API_KEY=sk-...
uv run python examples/quickstart.py "Where does the agent loop stop?"
```

`trajectories.py` needs no key either — it only reads the sessions saved for
the current directory — but recording one does:

```bash
uv run lab-harness -p "What does this project do?"            # record a session
uv run python examples/trajectories.py > requests.jsonl       # the latest one, a line per request
uv run python examples/trajectories.py <session-id> --wire    # a given one, as the endpoint received it
```

The Python examples take their model from config — `$LAB_HARNESS_MODEL`, else
`agent.model` — so one script runs against any configured vendor, given its
key:

```bash
export DEEPSEEK_API_KEY=sk-...
LAB_HARNESS_MODEL=deepseek-flash uv run python examples/hooks.py
```

Each example is a single file, readable top to bottom, and each is run by
[`tests/test_examples.py`](../tests/test_examples.py) against a scripted model,
so none of them can quietly fall behind the code. For a whole program rather
than a single file — a search agent over a corpus of documents — see
[`search_agent/`](../search_agent/).
