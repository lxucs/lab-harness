# AGENTS.md

Working agreements for AI coding agents in this repository.

**Read [docs/contributing.md](docs/contributing.md) before writing code** — it
holds the code conventions, the four architectural invariants, and the rules
against hardcoded values and leaky log records. This file covers environment,
configuration hygiene, testing, and git.

## What this project is

A minimal agent harness in Python, built for research. Terminal frontend,
modeled on [pi](https://github.com/earendil-works/pi). Researchers read and
change the code, so keep it small and readable: a new feature that fits a hook
or a subscriber does not belong in the loop.

- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Usage, terminal and Python: [`examples/`](examples/)
- What is built and what is not: [`README.md`](README.md#status)

## Environment

Use **uv**. The environment is the project-local **`.venv`** in the repository
root, created and managed by `uv sync`. There is no named or global environment
— do not create one, and do not `pip install` into the ambient interpreter.

```bash
uv sync                 # create/update .venv from pyproject.toml + uv.lock
uv sync --extra ddgs    # ...plus the optional keyless search backend
uv run pytest           # run tests
uv run lab-harness      # run the CLI
uv add <package>        # add a dependency
```

`uv run` selects `.venv` automatically; activating it manually is unnecessary.

Optional extras exist so a user is not made to install a dependency they will
never call. Code behind one must import it lazily and fail with the `uv sync`
command in the message.

## Configuration hygiene

**Nothing sensitive goes in a committed file.** No API keys or tokens, no
absolute paths (`/Users/...`, `/home/...`), no private URLs, hostnames, or org
ids — in config, the model catalog, the prompts, docs, tests, or examples.

| File | Committed | Holds |
|---|---|---|
| `.lab-harness/config.toml` | yes | Non-sensitive defaults |
| `.lab-harness/config.local.toml` | **no** — gitignored | Keys, absolute paths, private endpoints |
| `~/.lab-harness/config.local.toml` | **no** — outside the repo | The same, for every project |
| `.lab-harness/config.example.toml` | yes | Template, placeholders only |
| `lab_harness/llm/catalog.toml` | yes | Every LLM setting — public, documented vendor data only |
| `lab_harness/prompts.toml` | yes | Shipped prompts — no tool names, nothing about a particular machine |

Real runtime values belong in the `*.local.toml` layer, which `.gitignore`
excludes; it merges over the committed file, so it only needs the keys that
differ. Prefer environment variables for secrets outright — config then holds
the variable's *name*, which is itself a config key, and never its value.
If you add a config location, add its local variant to `.gitignore` in the
same change.

In anything committed, use placeholders that could not be real: `sk-...`,
`https://api.example.com`, `/path/to/project`.

The same list applies one step further to log records — a credential that
reaches a log is a credential published. That rule, and the rest of the
configuration conventions, are in
[docs/contributing.md](docs/contributing.md).

## Testing is offline

**The automated suite must never touch the network.** This is enforced:
`tests/conftest.py` installs autouse fixtures that block outbound socket
connections, redirect `LAB_HARNESS_HOME` to a temp directory, and unset every
credential variable — so a test reaching a real API fails loudly rather than
going flaky, and a developer who exports a key locally gets the same results as
CI, which exports none.

- Loop and agent tests get their model responses from
  `lab_harness.llm.fake.FakeProvider`, fed scripted stream events. Never call a
  real provider.
- The HTTP provider adapters (`chat_completions.py`, `responses.py`) and the web
  tools are tested through `httpx.MockTransport`, which exercises the real
  request-building and SSE-parsing code against canned bytes.
- `ddgs` is tested through an injected callable standing in for the package.
- Search backends are tested at the tool layer by registering a fake in
  `BACKENDS` under a config name, so the real selection path runs.
- Filesystem tools are tested against pytest's `tmp_path`.
- The scripts in `examples/` are run by `tests/test_examples.py`, with each
  one's `route` swapped for a `FakeProvider`. An example is documentation that
  executes: change one and its test together, and keep each a single file a
  reader can follow top to bottom.

Canned API payloads should be **shapes a live API actually returned**, not
invented ones, and should say so in a comment. A fixture invented from memory
tests the parser against your assumptions rather than against the API.

Real end-to-end verification against live APIs is **manual, done by the user**.
Do not add live-API tests, and do not ask the user for API keys to run the
suite.

## Git

**Do not create commits unless explicitly asked.** This includes `git commit`,
`git commit --amend`, and anything else that writes history. Do not push, create
branches, or open PRs on your own initiative.

Read-only git is always fine: `git status`, `git diff`, `git log`, `git show`,
`git blame`.

Leave finished work in the working tree and say what changed. The user commits.

## Before claiming work is done

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

Run them and read the output — do not infer success from having written the
code. If something fails, say so with the output. If part of the scope was
skipped, say that explicitly.
