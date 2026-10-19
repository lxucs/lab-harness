"""Token, cache and cost accounting.

Pure functions over messages, so they are tested on hand-built transcripts
rather than through a run. The terminal tests cover the wiring.
"""

import pytest

from lab_harness.cli.usage import (
    cache_hit_rate,
    format_cost,
    format_footer,
    format_status,
    format_tokens,
    price_for,
    summarize,
)
from lab_harness.config import PricingConfig
from lab_harness.llm.fake import FAKE_MODEL
from lab_harness.llm.types import AssistantMessage, Usage, UserMessage, text
from tests.entries import model_entry

PRICED = "priced-model"
UNPRICED = "unpriced-model"
#: Round rates so expected costs can be worked out by hand.
PRICE = PricingConfig(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
#: The models table `summarize` reads prices from, holding one priced model.
MODELS = {PRICED: model_entry(price=PRICE)}


def reply(usage: Usage, *, model: str = PRICED, stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=text("ok"),
        model=model,
        provider="fake",
        usage=usage,
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )


# --- summarize ------------------------------------------------------------


def test_usage_is_summed_across_requests() -> None:
    messages = [
        UserMessage(content=text("go")),
        reply(Usage(input=100, output=10, cache_read=0, cache_write=50)),
        reply(Usage(input=20, output=5, cache_read=150, cache_write=0)),
    ]

    summary = summarize(messages, MODELS)

    assert summary.requests == 2
    assert summary.usage == Usage(input=120, output=15, cache_read=150, cache_write=50)


def test_a_failed_request_counts_but_adds_no_tokens() -> None:
    """One request per assistant message, even one that reported nothing."""
    messages = [reply(Usage(input=10, output=5)), reply(Usage(), stop_reason="error")]

    summary = summarize(messages, MODELS)

    assert summary.requests == 2
    assert summary.usage == Usage(input=10, output=5)


def test_context_is_the_latest_request_not_the_sum() -> None:
    """Every request resends the history, so summing would count it again each turn."""
    messages = [
        reply(Usage(input=1_000, output=100)),
        reply(Usage(input=200, output=50, cache_read=1_100)),
    ]

    assert summarize(messages, MODELS).context_tokens == 200 + 50 + 1_100


def test_context_skips_a_trailing_request_that_reported_nothing() -> None:
    """An aborted turn has no usage; the context is still what came before it."""
    messages = [reply(Usage(input=300, output=20)), reply(Usage(), stop_reason="aborted")]

    assert summarize(messages, MODELS).context_tokens == 320


def test_cost_is_computed_from_every_rate() -> None:
    usage = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000)

    summary = summarize([reply(usage)], MODELS)

    assert summary.cost == pytest.approx(2.0 + 10.0 + 0.2 + 2.5)


def test_cost_is_summed_across_requests() -> None:
    messages = [reply(Usage(input=500_000)), reply(Usage(output=100_000))]

    assert summarize(messages, MODELS).cost == pytest.approx(1.0 + 1.0)


def test_one_unpriced_request_makes_the_total_unknown() -> None:
    """A total that silently left requests out would read as the whole bill."""
    messages = [reply(Usage(input=10)), reply(Usage(input=10), model=UNPRICED)]

    assert summarize(messages, MODELS).cost is None


def test_an_unpriced_request_with_no_tokens_does_not_hide_the_cost() -> None:
    """A failed request on another model cost nothing, so the total is still known."""
    messages = [reply(Usage(input=1_000_000)), reply(Usage(), model=UNPRICED)]

    assert summarize(messages, MODELS).cost == pytest.approx(2.0)


def test_no_usage_means_no_cost() -> None:
    assert summarize([reply(Usage())], MODELS).cost is None


# --- prices ---------------------------------------------------------------


def test_an_entry_without_a_price_is_unpriced() -> None:
    """An entry may give only limits or a provider; its default price is none."""
    assert price_for(PRICED, {PRICED: model_entry(provider="groq")}) is None


def test_an_all_zero_price_is_unpriced() -> None:
    """TOML has no null, and a merged table cannot delete a shipped entry."""
    assert price_for(PRICED, {PRICED: model_entry(price=PricingConfig())}) is None


def test_a_partly_priced_entry_is_priced() -> None:
    price = PricingConfig(input=1.0)

    assert price_for(PRICED, {PRICED: model_entry(price=price)}) == price


# --- cache hit rate -------------------------------------------------------


def test_cache_hit_rate_is_reads_over_everything_read() -> None:
    """Writes are part of the prompt too — they are tokens the cache did not serve."""
    usage = Usage(input=10, cache_read=80, cache_write=10)

    assert cache_hit_rate(usage) == pytest.approx(0.8)


def test_cache_hit_rate_is_none_without_a_prompt() -> None:
    assert cache_hit_rate(Usage()) is None


# --- formatting -----------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1k"),
        (12_400, "12.4k"),
        (200_000, "200k"),
        (1_000_000, "1M"),
        (1_250_000, "1.2M"),
    ],
)
def test_format_tokens(count: int, expected: str) -> None:
    assert format_tokens(count) == expected


def test_a_small_cost_keeps_enough_places_to_mean_something() -> None:
    """A short session costs fractions of a cent; "$0.00" would say nothing."""
    assert format_cost(0.0123, "$") == "$0.0123"


def test_a_large_cost_is_shown_in_cents() -> None:
    assert format_cost(12.346, "$") == "$12.35"


def test_the_currency_symbol_is_configurable() -> None:
    assert format_cost(1.5, "¥") == "¥1.50"


def test_the_footer_shows_tokens_cache_context_and_cost() -> None:
    summary = summarize(
        [reply(Usage(input=1_000, output=500, cache_read=9_000, cache_write=0))], MODELS
    )

    footer = format_footer(summary, FAKE_MODEL, "$")

    assert footer is not None
    assert "10k in" in footer
    assert "500 out" in footer
    assert "90% cached" in footer
    assert "context 10.5k of 200k (5%)" in footer
    # 1k x $2 + 500 x $10 + 9k x $0.20, per million.
    assert "$0.0088" in footer


def test_the_footer_omits_cost_for_an_unpriced_model() -> None:
    summary = summarize([reply(Usage(input=1_000, output=10), model=UNPRICED)], MODELS)

    footer = format_footer(summary, FAKE_MODEL, "$")

    assert footer is not None
    assert "$" not in footer


def test_there_is_no_footer_when_nothing_reported_usage() -> None:
    """A provider with usage reporting off would otherwise print a row of zeros."""
    summary = summarize([reply(Usage(), stop_reason="aborted")], MODELS)

    assert format_footer(summary, FAKE_MODEL, "$") is None


def test_status_lists_the_session_totals() -> None:
    summary = summarize(
        [reply(Usage(input=1_000, output=500, cache_read=3_000, cache_write=1_000))], MODELS
    )

    status = "\n".join(format_status(summary, FAKE_MODEL, "$"))

    assert FAKE_MODEL.id in status
    assert "requests" in status
    assert "5k" in status, "input includes cached tokens"
    assert "3k cache read" in status
    assert "1k cache write" in status
    assert "60%" in status
    assert "cost" in status


def test_status_omits_cost_for_an_unpriced_model() -> None:
    summary = summarize([reply(Usage(input=10), model=UNPRICED)], MODELS)

    status = "\n".join(format_status(summary, FAKE_MODEL, "$"))

    assert "cost" not in status


def test_status_before_any_request_says_so_without_a_rate() -> None:
    status = "\n".join(format_status(summarize([], MODELS), FAKE_MODEL, "$"))

    assert "requests" in status
    assert "cache hit" not in status
    assert "context" not in status
