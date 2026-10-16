"""Config entries built the way the loader builds them.

A profile or a model a config file names starts from the catalog's defaults,
so a test's entry does too: `profile_entry(api_key=...)` is what
`[providers.openai.<name>] api_key = ...` loads as.
"""

from dataclasses import replace
from typing import Any

from lab_harness.config import ModelConfig, OpenAIConfig, model_defaults, profile_defaults


def profile_entry(**settings: Any) -> OpenAIConfig:
    """A profile that sets only these keys."""
    return replace(profile_defaults(), **settings)


def model_entry(**settings: Any) -> ModelConfig:
    """A model entry that sets only these keys."""
    return replace(model_defaults(), **settings)
