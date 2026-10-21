# Search agent

A search agent example built on [lab-harness](../README.md) — it answers a question by
searching and reading a corpus of documents — and the repository's worked
example of driving the harness from Python. The agent is a few lines; the model
and the tool set are config; the transcript, the report and the numbers are the
harness's to produce.

One command runs it, with no arguments:

```bash
uv sync
uv run python -m search_agent                  # every query in data/queries.jsonl
uv run python -m search_agent --query 793 --limit 1
uv run python -m search_agent --help
```

**A real run is committed, if you would rather look than run:**
[`runs/793/report.html`](https://lxucs.github.io/lab-harness/search_agent/runs/793/report.html) is the whole exchange on one
page — turns, tool calls, token counts — beside the `trajectory.json` built
from it and the transcript the harness wrote. Paths in it read `<project>` and
`<home>`; that is this machine's, and it goes no further.

It needs a model, which it takes from config like anything else — see
[configuration](../docs/configuration.md).

## What it does

For each line of `data/queries.jsonl`, in order:

1. builds an `Agent` — the model from config, the corpus tools, its own system
   prompt — and sends one message: the request in `search_agent.toml` with the
   query in it;
2. streams the run to the console: `→ search_docs({...})` as tools fire, the
   answer as it is written;
3. writes `runs/<query-id>/` holding three files:

| File | What |
|---|---|
| `<query-id>.jsonl` | The session transcript — the same file `lab-harness` writes for a terminal session |
| `report.html` | The HTML report built from it, the same page `--export` produces |
| `trajectory.json` | The run as data: the query, the gold answer, the docids, the model and tools, the numbers, and every message |

There is **no scoring**. The gold answer is carried into `trajectory.json` next
to what the agent answered, so judgement stays a separate step — yours, or a
later script's.

Running a query again replaces its transcript rather than appending to it: a
re-run is a new run, not the second half of the last one.

## What it shows

How little of a search agent is left to write once lab-harness is underneath
it — all of it in `run.py`:

- `route(None, config)` — the model and provider, from config alone, so the
  same program runs on any vendor;
- `ToolRegistry` — built-ins chosen by name, plus one tool of this
  program's own;
- `Agent(model=..., provider=..., system_prompt=..., tools=..., cwd=...)` and
  one `await agent.prompt(...)` — the whole run;
- `start_recorder` subscribed to the agent, which is what makes the session
  transcript and the HTML report appear without this program writing either;
- reading the result: `summarize(messages, config.models)` for requests,
  tokens, cache hit rate and cost, and `to_dict` for each message.

That is the setup: a model from config, a tool list, a prompt, one call. What
comes back out — the transcript, the page, the numbers — is the harness's, not
this program's, and it is why `run.py` stays short.

`tools.py` is the second half: a tool is one decorated function with a Pydantic
parameter model, and this one is BM25 over the corpus in a dozen lines of
arithmetic. `read`, `grep` and `glob` find a document you can name; only a
search finds one you can describe.

## The data

The example ships a query from the [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) benchmark.
You can replace the data with your own.

- **`data/docs/`** — the corpus, one `<docid>.md` per document, front matter and all.
- **`data/queries.jsonl`** — one JSON object per line: `query_id`, `query`,
  `answer`, and the docids split into `gold_docids`, `evidence_docids` and `negative_docids`.

The agent's working directory is `search_agent/` itself — the directory holding
`search_agent.toml` — so `data/docs` in config, in the request and in a tool
call all mean the same thing.

## Configuration

Two files, both TOML, both in the shape
[docs/configuration.md](../docs/configuration.md) describes.

**`search_agent.toml`** - the task-related config: how to ask, and where things are:

```toml
[request]
template = """...{query}..."""     # the first turn; {query} is the line

[corpus]
docs = "data/docs"                 # relative to this file — and to the agent

[run]
queries = "data/queries.jsonl"
out_dir = "runs"
tools = ["ls", "glob", "grep", "read", "search_docs"]
```

**`config.local.toml`** — the agent-related config (gitignored). Everything about the agent: the
model, the keys, the tool set, the prompts. It is the same ordinary harness config also consumed by the terminal, see [docs/configuration.md](../docs/configuration.md).
