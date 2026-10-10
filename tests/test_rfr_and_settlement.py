"""Tests for RFR option timing, OIS quotes and swaption settlement."""

from datetime import date, timedelta

import pytest

from fi_pricing import core as fc
from fi_pricing.bootstrap import (
    DepositQuote,
    OISSwapQuote,
    SwapQuote,
    bootstrap_single_curve,
)
from fi_pricing.cap_floor import price_cap
from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES
from fi_pricing.rfr import RfrConvention, backward_looking_variance_ratio
from fi_pricing.swaption import cash_settled_annuity, price_swaption

VALUE_DATE = date(2025, 1, 2)
MATURITIES = [(date(2027, 1, 4), 0.0255), (date(2030, 1, 2), 0.0265), (date(2035, 1, 2), 0.0275)]


@pytest.fixture
def curve() -> fc.DiscountCurve:
    return fc.DiscountCurve(
        CURVE_DAYS, CURVE_ZERO_RATES, VALUE_DATE, interpolation="log_linear_df"
    )


# ---------------------------------------------------------------------------
# Gap 1: OIS swap quotes that actually compound
# ---------------------------------------------------------------------------
def _deposit() -> list[DepositQuote]:
    return [DepositQuote(date(2025, 7, 2), 0.0250, basis="ACT/360")]


def test_explicit_compounding_matches_the_telescoping_shortcut() -> None:
    """With no lag and no lockout the two constructions must agree.

    This is the invariant that makes the old shortcut legitimate; it is also the
    only case in which it is.
    """
    compounded = bootstrap_single_curve(
        VALUE_DATE, deposits=_deposit(), ois_swaps=[OISSwapQuote(m, r) for m, r in MATURITIES]
    )
    shortcut = bootstrap_single_curve(
        VALUE_DATE,
        deposits=_deposit(),
        swaps=[SwapQuote(m, r, fixed_basis="ACT/360") for m, r in MATURITIES],
    )
    assert max(abs(e) for e in compounded.repricing_errors) < 1e-13
    for a, b in zip(
        compounded.discount_factors, shortcut.discount_factors, strict=True
    ):
        assert a == pytest.approx(b, abs=1e-12)


def test_lookback_moves_the_curve_so_the_shortcut_would_be_wrong() -> None:
    history = {VALUE_DATE - timedelta(days=k): 0.0250 for k in range(1, 25)}
    plain = bootstrap_single_curve(
        VALUE_DATE, deposits=_deposit(), ois_swaps=[OISSwapQuote(m, r) for m, r in MATURITIES]
    )
    lookback = bootstrap_single_curve(
        VALUE_DATE,
        deposits=_deposit(),
        ois_swaps=[
            OISSwapQuote(
                m, r,
                rfr=RfrConvention(lag_days=5, shift_method="lookback"),
                past_fixings=history,
            )
            for m, r in MATURITIES
        ],
    )
    assert max(abs(e) for e in lookback.repricing_errors) < 1e-13
    assert lookback.discount_factors[-1] != pytest.approx(
        plain.discount_factors[-1], abs=1e-9
    )


def test_lookback_without_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="past overnight fixings"):
        bootstrap_single_curve(
            VALUE_DATE,
            deposits=_deposit(),
            ois_swaps=[
                OISSwapQuote(m, r, rfr=RfrConvention(lag_days=5, shift_method="lookback"))
                for m, r in MATURITIES
            ],
        )


def test_ois_quotes_cannot_be_bootstrapped_against_an_external_discount_curve(
    curve: fc.DiscountCurve,
) -> None:
    with pytest.raises(ValueError, match="external discount curve"):
        bootstrap_single_curve(
            VALUE_DATE,
            deposits=_deposit(),
            ois_swaps=[OISSwapQuote(m, r) for m, r in MATURITIES],
            discount_curve=curve,
        )


# ---------------------------------------------------------------------------
# Gap 2: caps and floors on a compounded RFR
# ---------------------------------------------------------------------------
def test_backward_looking_effective_time_formula() -> None:
    def period(start: date, end: date) -> fc.CouponPeriod:
        return fc.CouponPeriod(1, start, end, start, end, end, start)

    # Future period: reset time plus a third of the accrual window.
    future = period(date(2027, 1, 2), date(2027, 7, 2))
    start = (future.start - VALUE_DATE).days / 365.0
    length = (future.end - future.start).days / 365.0
    assert backward_looking_variance_ratio(VALUE_DATE, future) == pytest.approx(
        start + length / 3.0
    )

    # Continuity at t == S: the two branches must agree.
    at_start = period(VALUE_DATE, date(2025, 7, 2))
    span = (at_start.end - VALUE_DATE).days / 365.0
    assert backward_looking_variance_ratio(VALUE_DATE, at_start) == pytest.approx(span / 3.0)

    # In progress: strictly decreasing towards zero as the period closes.
    half = backward_looking_variance_ratio(VALUE_DATE, period(date(2024, 10, 2), date(2025, 4, 2)))
    nearly_done = backward_looking_variance_ratio(
        VALUE_DATE, period(date(2024, 10, 2), date(2025, 1, 3))
    )
    assert 0.0 < nearly_done < half


def test_compounded_cap_is_worth_more_than_the_term_rate_cap(
    curve: fc.DiscountCurve,
) -> None:
    """The extra (T-S)/3 of variance must show up as extra option value."""
    kwargs = dict(
        value_date=VALUE_DATE, effective=date(2025, 4, 2), maturity=date(2030, 4, 2),
        notional=1_000_000, strike=0.032, frequency=4, flat_vol=0.20, curve=curve,
    )
    term = price_cap(**kwargs)
    compounded = price_cap(**kwargs, rfr=RfrConvention(index="ESTR"))
    assert compounded.npv > term.npv
    # Same expected underlying, different effective option time.
    for a, b in zip(term.caplets, compounded.caplets, strict=True):
        assert a.forward == pytest.approx(b.forward, rel=1e-10)
        assert b.ttm > a.ttm


def test_compounded_cap_floor_parity(curve: fc.DiscountCurve) -> None:
    kwargs = dict(
        value_date=VALUE_DATE, effective=date(2025, 4, 2), maturity=date(2030, 4, 2),
        notional=1_000_000, strike=0.032, frequency=4, flat_vol=0.20, curve=curve,
        rfr=RfrConvention(index="ESTR"),
    )
    cap = price_cap(**kwargs)
    floor = price_cap(**{**kwargs, "style": "FLOOR"})
    swap = sum(
        (c.forward - 0.032) * c.af * 1_000_000 * c.df for c in cap.caplets
    )
    assert cap.npv - floor.npv == pytest.approx(swap, abs=1e-8)


def test_rfr_cap_rejects_a_term_first_fixing(curve: fc.DiscountCurve) -> None:
    with pytest.raises(ValueError, match="first_fixing does not apply"):
        price_cap(
            VALUE_DATE, date(2024, 10, 2), date(2030, 4, 2), 1_000_000, 0.032, 4,
            first_fixing=0.03, flat_vol=0.20, curve=curve,
            rfr=RfrConvention(index="ESTR"),
        )


def test_fixings_without_an_rfr_convention_are_rejected(curve: fc.DiscountCurve) -> None:
    with pytest.raises(ValueError, match="require an rfr convention"):
        price_cap(
            VALUE_DATE, date(2025, 4, 2), date(2030, 4, 2), 1_000_000, 0.032, 4,
            flat_vol=0.20, curve=curve, past_fixings={VALUE_DATE: 0.03},
        )


# ---------------------------------------------------------------------------
# Gap 3: cash-settled swaptions
# ---------------------------------------------------------------------------
def test_cash_settled_annuity_formula() -> None:
    # Zero rate: the annuity is just n/q.
    assert cash_settled_annuity(0.0, 10, 2) == pytest.approx(5.0)
    # Explicit sum for a small case.
    rate, n, q = 0.04, 3, 1
    expected = sum(1.0 / (1.0 + rate) ** i for i in range(1, n + 1))
    assert cash_settled_annuity(rate, n, q) == pytest.approx(expected)
    # Decreasing in the settlement rate.
    assert cash_settled_annuity(0.05, 10, 1) < cash_settled_annuity(0.01, 10, 1)
    with pytest.raises(ValueError):
        cash_settled_annuity(0.03, 0, 1)


def _swaption(**overrides):
    kwargs = dict(
        value_date=VALUE_DATE, expiry=date(2027, 1, 4), swap_start=date(2027, 1, 4),
        swap_end=date(2032, 1, 4), strike=0.032, vol=0.20, style="Payer",
    )
    kwargs.update(overrides)
    return price_swaption(**kwargs)


def test_cash_and_physical_share_the_forward_but_not_the_annuity() -> None:
    physical = _swaption()
    cash = _swaption(settlement="cash")
    assert cash.forward_rate == pytest.approx(physical.forward_rate)
    assert cash.settlement == "cash"
    assert physical.settlement == "physical"
    # The physical annuity is reported on both, so the two are comparable.
    assert cash.curve_annuity == pytest.approx(physical.annuity)
    assert cash.annuity != pytest.approx(physical.annuity, rel=1e-6)


def test_cash_settled_payer_receiver_parity() -> None:
    payer = _swaption(settlement="cash", style="Payer")
    receiver = _swaption(settlement="cash", style="Receiver")
    expected = (payer.forward_rate - payer.strike) * payer.annuity
    assert payer.premium - receiver.premium == pytest.approx(expected, rel=1e-10)


def test_physical_settlement_is_unchanged_by_the_new_argument() -> None:
    assert _swaption().premium == pytest.approx(
        _swaption(settlement="physical").premium, abs=1e-12
    )


def test_unknown_settlement_is_rejected() -> None:
    with pytest.raises(ValueError, match="settlement must be"):
        _swaption(settlement="cash-settled")

def test_cash_settlement_is_explicitly_labelled_as_approximation() -> None:
    cash = _swaption(settlement="cash")
    assert cash.settlement_method == "frozen_forward_irr_annuity"
    assert _swaption().settlement_method == "physical_delivery"
