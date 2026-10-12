"""Model catalog and resolution.

Deliberately small. pi ships a provider catalog with pricing and capability
matrices; this is a dict of the models we actually use. Adding one is a line.

Capability flags matter to the runtime — `supports_reasoning` gates the thinking
request, `supports_server_tools` decides whether `web_search` is forwarded to
the provider or executed locally — so they live with the model, not at the call
site.
"""

import logging
import os
from dataclasses import replace

from harness.config import AgentConfig
from harness.llm.types import Model

_log = logging.getLogger(__name__)

_CONTEXT_1M = 1_000_000
_CONTEXT_200K = 200_000
_CONTEXT_128K = 128_000

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
    # OpenAI-compatible entries. `provider` names the config profile that
    # serves them, not the dialect — routing is by profile, and a config file
    # can re-point any of these at a different endpoint.
    #
    # Limits here are deliberately conservative. The catalog's real job is the
    # capability flags, which the runtime acts on; an id whose window is
    # understated costs a little context, while one that is overstated fails
    # the request.
    "deepseek-chat": Model(
        id="deepseek-chat",
        name="DeepSeek Chat",
        provider="deepseek",
        context_window=_CONTEXT_128K,
        max_tokens=8_192,
    ),
    "deepseek-reasoner": Model(
        id="deepseek-reasoner",
        name="DeepSeek Reasoner",
        provider="deepseek",
        context_window=_CONTEXT_128K,
        max_tokens=8_192,
        # Drives the thinking request, and makes the adapter read the
        # `reasoning_content` deltas this model streams.
        supports_reasoning=True,
    ),
    "glm-4.6": Model(
        id="glm-4.6",
        name="GLM-4.6",
        provider="glm",
        context_window=_CONTEXT_128K,
        max_tokens=8_192,
        supports_reasoning=True,
    ),
}

#: Catalog entry cloned to describe an id this module has never heard of. Only
#: its provider and capability flags survive the clone; the limits are replaced
#: with conservative ones. This is *not* the default model — that is
#: `agent.model` in the config schema, the single place defaults live.
_UNKNOWN_MODEL_TEMPLATE = "claude-sonnet-5"

#: Conservative limits for an unlisted id. Both are floors across the current
#: Anthropic line, so guessing them cannot exceed what the model accepts.
_UNKNOWN_CONTEXT_WINDOW = _CONTEXT_200K
_UNKNOWN_MAX_TOKENS = 8_192


def resolve_model(model_id: str | None = None, config: AgentConfig | None = None) -> Model:
    """Resolve a model id to a `Model`.

    Precedence, highest first: the explicit argument (`--model`), the env var
    named by `agent.model_env`, then `agent.model`. That is the project-wide
    order — a CLI flag beats the environment, which beats a config file.

    An unknown id is not an error. Model names ship faster than this catalog
    does, so an unlisted one is given conservative limits rather than refused —
    being unable to use a model released last week is worse than a slightly
    wrong `context_window`.

    The returned `provider` is a placeholder for such an id; routing decides
    the real one. See `llm/providers/route`.
    """
    config = config or AgentConfig()
    wanted = model_id or os.environ.get(config.model_env) or config.model
    _log.info("Using model %r", wanted)
    known = MODELS.get(wanted)
    if known is not None:
        return known
    # A typo looks identical to a model newer than the catalog, so this says
    # only what is certain: the limits are a guess. Which vendor serves the id
    # is decided later, by routing.
    _log.warning(
        "Model %r is not in the catalog; assuming conservative limits "
        "(%d-token context, %d max output tokens)",
        wanted,
        _UNKNOWN_CONTEXT_WINDOW,
        _UNKNOWN_MAX_TOKENS,
    )
    return replace(
        MODELS[_UNKNOWN_MODEL_TEMPLATE],
        id=wanted,
        name=wanted,
        context_window=_UNKNOWN_CONTEXT_WINDOW,
        max_tokens=_UNKNOWN_MAX_TOKENS,
    )
