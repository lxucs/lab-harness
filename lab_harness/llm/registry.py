"""Model resolution: from an id to the `Model` the adapters read.

The catalog itself is data, not code: llm/catalog.toml, read as the default of
the `models` config section. A config file adds a model, or changes one, with
the same `[models.<id>]` table, so nothing about a model is hardcoded here.

Capability flags describe the model, so they live with it rather than at a
call site: `supports_reasoning` gates the thinking request an adapter sends,
and `supports_images` and `supports_server_tools` only record a capability.
"""

import logging
import os

from lab_harness.config import Config, ModelConfig
from lab_harness.llm.types import Model

_log = logging.getLogger(__name__)

#: What an unlisted id is given, as its warning states it.
_UNKNOWN = ModelConfig()
_UNKNOWN_LIMITS = f"{_UNKNOWN.context_window}-token context, {_UNKNOWN.max_tokens} output tokens"


def resolve_model(model_id: str | None = None, config: Config | None = None) -> Model:
    """Resolve a model id to a `Model`.

    Precedence, highest first: the explicit argument (`--model`), the env var
    named by `agent.model_env`, then `agent.model`. That is the project-wide
    order — a CLI flag beats the environment, which beats a config file.

    An unknown id is not an error. Model names ship faster than this catalog
    does, so an unlisted one is given conservative limits rather than refused —
    being unable to use a model released last week is worse than a slightly
    wrong `context_window`.

    The returned `provider` is the entry's, which is empty for an unlisted id;
    routing decides the real one. See `llm/providers/route`.
    """
    config = config or Config()
    agent = config.agent
    from_env = os.environ.get(agent.model_env)
    wanted = model_id or from_env or agent.model
    source = "--model" if model_id else agent.model_env if from_env else "agent.model"
    _log.info("Using model %r, from %s", wanted, source)
    entry = config.models.get(wanted)
    if entry is None:
        # A typo looks identical to a model newer than the catalog, so this
        # says only what is certain: the limits are a guess. Which vendor
        # serves the id is decided later, by routing.
        _log.warning("Model %r is not in the catalog; assuming %s", wanted, _UNKNOWN_LIMITS)
        entry = _UNKNOWN
    return Model(
        id=wanted,
        name=entry.name or wanted,
        provider=entry.provider,
        context_window=entry.context_window,
        max_tokens=entry.max_tokens,
        supports_reasoning=entry.supports_reasoning,
        supports_images=entry.supports_images,
        supports_server_tools=entry.supports_server_tools,
    )
