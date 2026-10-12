# AGENTS.md

Working agreements for AI coding agents in this repository.

**Read [docs/contributing.md](docs/contributing.md) before writing code** — it
holds the code conventions, the four architectural invariants, and the
configuration rules. This file covers environment, testing, and git.

## What this project is

A lightweight agent harness. Python runtime, terminal frontend, modeled on
[pi](https://github.com/earendil-works/pi).

- Design spec: [`docs/superpowers/specs/2026-10-09-harness-runtime-design.md`](docs/superpowers/specs/2026-10-09-harness-runtime-design.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Current progress: the build-order checklist in [`README.md`](README.md)

## Environment

Use **uv**. The environment is the project-local **`.venv`** in the repository
root, created and managed by `uv sync`. There is no named or global environment
— do not create one, and do not `pip install` into the ambient interpreter.

```bash
uv sync                 # create/update .venv from pyproject.toml + uv.lock
uv sync --extra ddgs    # ...plus the optional keyless search backend
uv run pytest           # run tests
uv run harness          # run the CLI
uv add <package>        # add a dependency
```

`uv run` selects `.venv` automatically; activating it manually is unnecessary.

Optional extras exist so a user is not made to install a dependency they will
never call. Code behind one must import it lazily and fail with the `uv sync`
command in the message.

## Testing is offline

**The automated suite must never touch the network.** This is enforced:
`tests/conftest.py` installs autouse fixtures that block outbound socket
connections, redirect `HARNESS_HOME` to a temp directory, and unset every
credential variable — so a test reaching a real API fails loudly rather than
going flaky, and a developer who exports a key locally gets the same results as
CI, which exports none.

- Model responses come from `harness.llm.fake.FakeProvider`, fed scripted stream
  events. Never call a real provider.
- HTTP-based tools are tested through `httpx.MockTransport`, which exercises the
  real request-building and response-parsing code against canned bytes.
- `ddgs` is tested through an injected callable standing in for the package.
- Search backends are tested at the tool layer by registering a fake in
  `BACKENDS` under a config name, so the real selection path runs.
- Filesystem tools are tested against pytest's `tmp_path`.

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
