"""Model catalog and resolution.

Deliberately small. pi ships a provider catalog with pricing and capability
matrices; this is a dict of the models we actually use. Adding one is a line.

Capability flags describe the model, so they live with it rather than at a
call site: `supports_reasoning` gates the thinking request an adapter sends
when a caller asks for reasoning, and `supports_server_tools` records a
capability nothing acts on, since provider-native `web_search` is not planned.
"""

import logging
import os
from dataclasses import replace

from lab_harness.config import AgentConfig
from lab_harness.llm.types import Model

_log = logging.getLogger(__name__)

_CONTEXT_1M = 1_000_000
_CONTEXT_200K = 200_000
#: DeepSeek's documented default output ceiling in thinking mode ("64K").
_DEEPSEEK_THINKING_MAX_TOKENS = 64_000
#: The maximum output Z.ai documents for every catalogued GLM model ("128K").
_GLM_MAX_TOKENS = 128_000

MODELS: dict[str, Model] = {
    "claude-opus-5": Model(
        id="claude-opus-5",
        name="Claude Opus 5",
        provider="anthropic",
        context_window=_CONTEXT_1M,
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
        # Extended thinking, the budgeted kind — see
        # `providers.anthropic.budget_thinking_models`.
        supports_reasoning=True,
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
    #
    # DeepSeek: the two ids its API accepts as of 2026-09. Both think by
    # default, and reasoning counts against max_tokens — so the ceiling is
    # DeepSeek's own default for thinking mode, not the 8K non-thinking one,
    # which a long chain of thought would exhaust before the answer began.
    "deepseek-flash": Model(
        id="deepseek-flash",
        name="DeepSeek Flash",
        provider="deepseek",
        context_window=_CONTEXT_1M,
        max_tokens=_DEEPSEEK_THINKING_MAX_TOKENS,
        # Gates `reasoning_effort` in the request. Reasoning *deltas* are read
        # whatever this says, since a model that streams them means them.
        supports_reasoning=True,
    ),
    "deepseek-v4-pro": Model(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        provider="deepseek",
        context_window=_CONTEXT_1M,
        max_tokens=_DEEPSEEK_THINKING_MAX_TOKENS,
        supports_reasoning=True,
    ),
    # GLM: the catalogued ids are the ones whose limits Z.ai documents on a
    # model page (2026-10). The rest of the lineup routes by the profile's
    # `models` list and gets the conservative unknown-model limits.
    "glm-5.3": Model(
        id="glm-5.3",
        name="GLM-5.3",
        provider="glm",
        context_window=_CONTEXT_1M,
        max_tokens=_GLM_MAX_TOKENS,
        supports_reasoning=True,
    ),
    "glm-5.3-flash": Model(
        id="glm-5.3-flash",
        name="GLM-5.3-Flash",
        provider="glm",
        context_window=_CONTEXT_1M,
        max_tokens=_GLM_MAX_TOKENS,
        supports_reasoning=True,
    ),
    "glm-5.3-flashx": Model(
        id="glm-5.3-flashx",
        name="GLM-5.3-FlashX",
        provider="glm",
        context_window=_CONTEXT_1M,
        max_tokens=_GLM_MAX_TOKENS,
        supports_reasoning=True,
    ),
    "glm-4.7": Model(
        id="glm-4.7",
        name="GLM-4.7",
        provider="glm",
        context_window=_CONTEXT_200K,
        max_tokens=_GLM_MAX_TOKENS,
        supports_reasoning=True,
    ),
    "glm-4.6": Model(
        id="glm-4.6",
        name="GLM-4.6",
        provider="glm",
        context_window=_CONTEXT_200K,
        max_tokens=_GLM_MAX_TOKENS,
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
#: The same two, as the warning for an unlisted id states them.
_UNKNOWN_LIMITS = f"{_UNKNOWN_CONTEXT_WINDOW}-token context, {_UNKNOWN_MAX_TOKENS} output tokens"


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
    from_env = os.environ.get(config.model_env)
    wanted = model_id or from_env or config.model
    source = "--model" if model_id else config.model_env if from_env else "agent.model"
    _log.info("Using model %r, from %s", wanted, source)
    known = MODELS.get(wanted)
    if known is not None:
        return known
    # A typo looks identical to a model newer than the catalog, so this says
    # only what is certain: the limits are a guess. Which vendor serves the id
    # is decided later, by routing.
    _log.warning("Model %r is not in the catalog; assuming %s", wanted, _UNKNOWN_LIMITS)
    return replace(
        MODELS[_UNKNOWN_MODEL_TEMPLATE],
        id=wanted,
        name=wanted,
        context_window=_UNKNOWN_CONTEXT_WINDOW,
        max_tokens=_UNKNOWN_MAX_TOKENS,
    )
