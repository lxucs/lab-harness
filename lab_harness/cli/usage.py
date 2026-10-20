"""Token, cache and cost accounting for the terminal.

Computed from messages rather than accumulated from events. Every assistant
message already carries the `Usage` of the request that produced it, so a fold
over the transcript *is* the totals — including history restored by
`--resume`, which `Agent.restore` installs silently and so never passes a
subscriber.

The four `Usage` counts mean the same thing whichever adapter filled them:
`input` is the *uncached* remainder, and cache reads and writes are counted
separately. The adapters do that normalization, which is what lets one
hit-rate formula and one cost formula serve every provider.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from lab_harness.config import ModelConfig, PricingConfig
from lab_harness.llm.types import AssistantMessage, Message, Model, Usage

#: Rates are quoted per million tokens, as every vendor's price list is.
_PER_MILLION = 1_000_000
#: Joins the parts of the one-line footer.
_SEPARATOR = " · "


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Totals over a span of messages — one run, or a whole session."""

    #: Provider requests made: one per assistant message, failed ones included.
    requests: int = 0
    usage: Usage = field(default_factory=Usage)
    #: None when nothing was spent, or when any request that used tokens ran
    #: on an unpriced model — a total that silently left requests out would
    #: read as the whole bill.
    cost: float | None = None
    #: Prompt plus output of the latest request that reported usage, which is
    #: roughly what the next request resends. Zero when none has.
    context_tokens: int = 0


def summarize(messages: Iterable[Message], models: Mapping[str, ModelConfig]) -> UsageSummary:
    requests = 0
    usage = Usage()
    cost: float | None = 0.0
    context = 0
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        requests += 1
        spent = message.usage
        if spent.total == 0:
            # A failed or aborted request reports nothing. It adds no tokens,
            # and cost nothing, so it cannot make the total unknowable either.
            continue
        usage = _add(usage, spent)
        context = spent.total
        price = price_for(message.model, models)
        cost = None if cost is None or price is None else cost + cost_of(spent, price)
    return UsageSummary(
        requests=requests,
        usage=usage,
        cost=cost if usage.total else None,
        context_tokens=context,
    )


def price_for(model_id: str, models: Mapping[str, ModelConfig]) -> PricingConfig | None:
    """The configured rates for a model, or None when it has none.

    A price with every rate at zero counts as none — which is also the default,
    so a model listed without a price is unpriced. TOML has no null, and a
    config layer merges over the shipped catalog rather than replacing it, so
    zeroing a price is the only way to switch one off.
    """
    entry = models.get(model_id)
    if entry is None or entry.price == PricingConfig():
        return None
    return entry.price


def cost_of(usage: Usage, price: PricingConfig) -> float:
    return (
        usage.input * price.input
        + usage.output * price.output
        + usage.cache_read * price.cache_read
        + usage.cache_write * price.cache_write
    ) / _PER_MILLION


def prompt_tokens(usage: Usage) -> int:
    """Everything the provider read, cached or not."""
    return usage.input + usage.cache_read + usage.cache_write


def cache_hit_rate(usage: Usage) -> float | None:
    """Share of prompt tokens served from cache; None when nothing was read.

    Cache *writes* sit in the denominator: they are prompt tokens the cache
    did not serve, even though they will be served next time.
    """
    prompt = prompt_tokens(usage)
    return usage.cache_read / prompt if prompt else None


# --- formatting -----------------------------------------------------------


def format_tokens(count: int) -> str:
    """`999`, `12.4k`, `1.2M` — short enough for a one-line footer."""
    for divisor, suffix in ((1_000_000, "M"), (1_000, "k")):
        if count >= divisor:
            return f"{count / divisor:.1f}".removesuffix(".0") + suffix
    return str(count)


def format_cost(amount: float, symbol: str) -> str:
    # Four places below one unit: a short session costs fractions of a cent,
    # and "$0.00" would say nothing at all.
    return f"{symbol}{amount:.4f}" if amount < 1 else f"{symbol}{amount:.2f}"


def format_footer(summary: UsageSummary, model: Model, symbol: str) -> str | None:
    """One line for the end of a run, or None when no request reported usage.

    None rather than a row of zeros: a provider with usage reporting turned
    off would otherwise print a line that looks like a measurement.
    """
    usage = summary.usage
    if usage.total == 0:
        return None
    parts = [f"{format_tokens(prompt_tokens(usage))} in", f"{format_tokens(usage.output)} out"]
    rate = cache_hit_rate(usage)
    if rate is not None:
        parts.append(f"{rate:.0%} cached")
    parts.append(f"context {_context(summary.context_tokens, model)}")
    if summary.cost is not None:
        parts.append(format_cost(summary.cost, symbol))
    return _SEPARATOR.join(parts)


def format_status(summary: UsageSummary, model: Model, symbol: str) -> list[str]:
    """Aligned `label  value` lines for `/status`."""
    usage = summary.usage
    rows = [
        ("model", f"{model.id} via {model.provider}"),
        *([("effort", model.reasoning_effort or "default")] if model.supports_reasoning else []),
        ("requests", str(summary.requests)),
        (
            "input",
            f"{format_tokens(prompt_tokens(usage))} ("
            f"{format_tokens(usage.input)} uncached{_SEPARATOR}"
            f"{format_tokens(usage.cache_read)} cache read{_SEPARATOR}"
            f"{format_tokens(usage.cache_write)} cache write)",
        ),
        ("output", format_tokens(usage.output)),
    ]
    rate = cache_hit_rate(usage)
    if rate is not None:
        rows.append(("cache hit", f"{rate:.0%}"))
    if summary.context_tokens:
        rows.append(("context", _context(summary.context_tokens, model)))
    if summary.cost is not None:
        rows.append(("cost", format_cost(summary.cost, symbol)))
    width = max(len(label) for label, _ in rows)
    return [f"{label:<{width}}  {value}" for label, value in rows]


def _context(tokens: int, model: Model) -> str:
    """How full the window is: `10.5k of 200k (5%)`."""
    share = tokens / model.context_window
    return f"{format_tokens(tokens)} of {format_tokens(model.context_window)} ({share:.0%})"


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(
        input=a.input + b.input,
        output=a.output + b.output,
        cache_read=a.cache_read + b.cache_read,
        cache_write=a.cache_write + b.cache_write,
    )
