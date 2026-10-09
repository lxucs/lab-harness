"""Configuration schema and defaults.

This module is the single list of every tunable in the system. Code reads
values from a resolved `Config`; it never hardcodes them. See
docs/contributing.md.

Config file loading (TOML, layered, with config.local.toml) arrives with the
CLI milestone. For now `load_config()` returns defaults, so the loop has a real
config object to depend on from the start.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AgentConfig:
    #: Hard ceiling on provider requests in one run. Guards against a model
    #: that calls tools forever.
    max_turns: int = 50
    #: Run independent tool calls from one message concurrently.
    parallel_tools: bool = True


@dataclass(frozen=True, slots=True)
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)


def load_config() -> Config:
    """Resolve configuration. Currently defaults only."""
    return Config()
