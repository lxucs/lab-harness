"""Model catalog and resolution.

Deliberately small. pi ships a provider catalog with pricing and capability
matrices; this is a dict of the models we actually use. Adding one is a line.

Capability flags matter to the runtime — `supports_reasoning` gates the thinking
request, `supports_server_tools` decides whether `web_search` is forwarded to
the provider or executed locally — so they live with the model, not at the call
site.
"""

import os
from dataclasses import replace

from harness.llm.types import Model

_CONTEXT_1M = 1_000_000
_CONTEXT_200K = 200_000

MODELS: dict[str, Model] = {
    "claude-opus-5": Model(
        id="claude-opus-5",
        name="Claude Opus 5",
        provider="anthropic",
        context_window=_CONTEXT_200K,
        max_tokens=32_000,
        supports_reasoning=True,
        supports_images=True,
        supports_server_tools=True,
    ),
    "claude-sonnet-5": Model(
        id="claude-sonnet-5",
        name="Claude Sonnet 5",
        provider="anthropic",
        context_window=_CONTEXT_1M,
        max_tokens=64_000,
        supports_reasoning=True,
        supports_images=True,
        supports_server_tools=True,
    ),
    "claude-haiku-4-5-20251001": Model(
        id="claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        provider="anthropic",
        context_window=_CONTEXT_200K,
        max_tokens=8_192,
        supports_images=True,
        supports_server_tools=True,
    ),
}

DEFAULT_MODEL_ID = "claude-sonnet-5"

#: Env var that overrides the configured model. Documented in configuration.md.
MODEL_ENV = "HARNESS_MODEL"


def resolve_model(model_id: str | None = None) -> Model:
    """Resolve a model id to a `Model`, honoring the environment override.

    Precedence: explicit argument, then `HARNESS_MODEL`, then the default.

    An unknown id is not an error. Model names ship faster than this catalog
    does, so an unlisted one is assumed to be a current Anthropic model and
    given conservative limits — being unable to use a model released last week
    is worse than a slightly wrong `context_window`.
    """
    wanted = model_id or os.environ.get(MODEL_ENV) or DEFAULT_MODEL_ID
    known = MODELS.get(wanted)
    if known is not None:
        return known
    fallback = MODELS[DEFAULT_MODEL_ID]
    return replace(
        fallback,
        id=wanted,
        name=wanted,
        context_window=_CONTEXT_200K,
        max_tokens=8_192,
    )
