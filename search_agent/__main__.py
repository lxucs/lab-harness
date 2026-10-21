"""So `python -m search_agent` runs the CLI."""

import asyncio

from search_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
