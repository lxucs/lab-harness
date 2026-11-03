"""Config entries built the way the loader builds them.

A profile or a model a config file names starts from the catalog's defaults,
so a test's entry does too: `profile_entry(api_key=...)` is what
`[providers.openai.<name>] api_key = ...` loads as.

With one deliberate exception: retries are **off** unless a test asks for
them. The shipped default is five, with a doubling backoff, so a suite that
inherited it would sit out 31 seconds for every test of a failure — and every
one of those tests is about the failure, not about the retrying. The tests that
are about retrying ask for them, with no delay. `test_retry_defaults` is what
keeps the shipped number itself covered.
"""

from dataclasses import replace
from typing import Any

from lab_harness.config import ModelConfig, OpenAIConfig, model_defaults, profile_defaults


def profile_entry(**settings: Any) -> OpenAIConfig:
    """A profile that sets only these keys. Retries off — see the module docstring."""
    return replace(profile_defaults(), **{"max_retries": 0, **settings})


def retrying_profile(**settings: Any) -> OpenAIConfig:
    """A profile that retries as the shipped default does, but without waiting.

    A test that is about the waiting sets the delays back.
    """
    return replace(
        profile_defaults(), **{"retry_initial_seconds": 0.0, "retry_max_seconds": 0.0, **settings}
    )


def model_entry(**settings: Any) -> ModelConfig:
    """A model entry that sets only these keys."""
    return replace(model_defaults(), **settings)
