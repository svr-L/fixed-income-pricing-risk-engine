"""Synthetic negative-rate regime tests.

The fixture is generated deterministically inside the test module. It exercises
discount factors above one, negative swap forwards, a positive OIS/projection
basis, and normal-volatility option quotes without redistributing historical
market data.
"""

from datetime import date

import pytest

from fi_pricing import core as fc
from fi_pricing.arbitrage_checks import check_discount_factors
from fi_pricing.bootstrap import SwapQuote, bootstrap_single_curve
from fi_pricing.cap_floor import price_cap
from fi_pricing.curves import CurveSet
from fi_pricing.volatility import implied_volatility

VALUE_DATE = date(2025, 1, 2)
TENORS = tuple(range(1, 31))
DAYS = tuple(365 * t for t in TENORS)


def _ois_zero(t: int) -> float:
    return -0.0040 + 0.00045 * t


def _projection_zero(t: int) -> float:
    return _ois_zero(t) + 0.0034 + 0.00002 * t


OIS = fc.DiscountCurve(
    DAYS,
    [_ois_zero(t) for t in TENORS],
    VALUE_DATE,
    interpolation="log_linear_df",
)
PROJECTION = fc.DiscountCurve(
    DAYS,
    [_projection_zero(t) for t in TENORS],
    VALUE_DATE,
    interpolation="log_linear_df",
)


def _annual_par(curve: fc.DiscountCurve, maturity: int) -> float:
    dfs = [curve.df_days(365 * k) for k in range(1, maturity + 1)]
    return (1.0 - dfs[-1]) / sum(dfs)


ATM_SWAP_RATES = tuple(_annual_par(PROJECTION, t) for t in TENORS)


def _normal_vol(expiry: float, strike: float) -> float:
    return 0.0060 + 0.00010 * min(expiry, 10.0) + 0.035 * abs(strike + 0.0010)


def test_fixture_is_genuinely_negative_rate() -> None:
    assert any(OIS.df_days(d) > 1.0 for d in DAYS)
    assert min(ATM_SWAP_RATES) < 0.0


def test_admissibility_accepts_negative_forward_curve() -> None:
    dfs = [OIS.df_days(d) for d in DAYS]
    assert check_discount_factors(DAYS, dfs).passed
    assert not check_discount_factors(DAYS, dfs, min_forward_rate=0.0).passed


def test_projection_curve_par_rates_are_internally_consistent() -> None:
    for maturity, quote in zip(TENORS, ATM_SWAP_RATES, strict=True):
        assert _annual_par(PROJECTION, maturity) == pytest.approx(quote, abs=1e-14)


def test_projection_basis_is_positive_at_every_tenor() -> None:
    for tenor in TENORS:
        days = 365 * tenor
        assert (PROJECTION.zero(days) - OIS.zero(days)) * 1e4 > 0.0


def test_black_refuses_negative_forward_and_bachelier_prices_it() -> None:
    forward = ATM_SWAP_RATES[0]
    vol = _normal_vol(1.0, forward)
    assert forward < 0.0
    assert (
        fc.forward_option_premium(
            forward,
            forward,
            1.0,
            vol,
            1,
            model="bachelier",
        )
        > 0.0
    )
    with pytest.raises(ValueError, match="positive forward"):
        fc.forward_option_premium(forward, forward, 1.0, 0.20, 1, model="black")


def test_bachelier_to_shifted_black_option_conversion_round_trip() -> None:
    shift = 0.03
    for maturity in (1, 2, 3, 5, 7, 10):
        forward = ATM_SWAP_RATES[maturity - 1]
        normal_vol = _normal_vol(float(maturity), forward)
        target = fc.forward_option_premium(
            forward,
            forward,
            float(maturity),
            normal_vol,
            1,
            model="bachelier",
        )
        equivalent = implied_volatility(
            target,
            forward,
            forward,
            float(maturity),
            1,
            model="shifted_black",
            shift=shift,
        )
        rebuilt = fc.forward_option_premium(
            forward,
            forward,
            float(maturity),
            equivalent,
            1,
            model="shifted_black",
            shift=shift,
        )
        assert rebuilt == pytest.approx(target, rel=1e-10, abs=1e-14)


def test_cap_prices_with_negative_strike_under_bachelier() -> None:
    curves = CurveSet(OIS, {"SYNTH3M": PROJECTION})
    cap = price_cap(
        VALUE_DATE,
        date(2025, 4, 2),
        date(2030, 4, 2),
        1_000_000,
        -0.0020,
        4,
        flat_vol=0.0070,
        vol_model="bachelier",
        curve=curves,
        index="SYNTH3M",
    )
    floor = price_cap(
        VALUE_DATE,
        date(2025, 4, 2),
        date(2030, 4, 2),
        1_000_000,
        -0.0020,
        4,
        style="FLOOR",
        flat_vol=0.0070,
        vol_model="bachelier",
        curve=curves,
        index="SYNTH3M",
    )
    assert cap.npv > 0.0
    assert any(c.forward < 0.0 for c in cap.caplets)
    swap = sum(
        (c.forward + 0.0020) * c.af * 1_000_000 * c.df for c in cap.caplets
    )
    assert cap.npv - floor.npv == pytest.approx(swap, abs=1e-6)


def test_bootstrap_supports_negative_par_swap_quotes() -> None:
    ois_par = tuple(_annual_par(OIS, t) for t in TENORS[1:8])
    quotes = [
        SwapQuote(
            date(VALUE_DATE.year + maturity, VALUE_DATE.month, VALUE_DATE.day),
            rate,
            fixed_basis="ACT/365",
        )
        for maturity, rate in zip(TENORS[1:8], ois_par, strict=True)
    ]
    result = bootstrap_single_curve(VALUE_DATE, swaps=quotes)
    assert max(abs(e) for e in result.repricing_errors) < 1e-10
    assert all(c.passed for c in result.checks)
    assert result.curve.zero(365 * 2) < 0.0
