"""Regression tests for market conventions and risk invariants."""

from datetime import date

import pytest

from fi_pricing import core as fc
from fi_pricing.arbitrage_checks import (
    ArbitrageError,
    check_butterfly_convexity,
    check_discount_factors,
)
from fi_pricing.bootstrap import DepositQuote, SwapQuote, bootstrap_single_curve
from fi_pricing.cap_floor import price_cap
from fi_pricing.irs import Leg, price_irs, price_leg
from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES
from fi_pricing.risk import key_rate_convexity_matrix, key_rate_risk, parallel_curve_risk
from fi_pricing.volatility import VolSurface2D

VALUE_DATE = date(2025, 1, 6)


def _curve(value_date: date = VALUE_DATE) -> fc.DiscountCurve:
    return fc.DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, value_date)


# ---------------------------------------------------------------------------
# Past fixings on already-fixed but unpaid periods
# ---------------------------------------------------------------------------
def _lagged_float_leg(**overrides) -> Leg:
    kwargs = dict(
        notional=1_000_000,
        effective=date(2024, 7, 2),
        maturity=date(2030, 7, 2),
        frequency=2,
        basis="ACT/360",
        is_fixed=False,
        current_fixing=0.0290,
        schedule_spec=fc.ScheduleSpec(frequency=2, payment_lag_days=5),
    )
    kwargs.update(overrides)
    return Leg(**kwargs)


def test_settled_unpaid_period_requires_a_past_fixing() -> None:
    with pytest.raises(ValueError, match="past_fixings"):
        price_leg(_lagged_float_leg(), VALUE_DATE, _curve(), fc.TargetCalendar())


def test_past_fixing_is_used_instead_of_a_curve_forecast() -> None:
    leg = _lagged_float_leg(past_fixings={date(2024, 7, 2): 0.0345})
    result = price_leg(leg, VALUE_DATE, _curve(), fc.TargetCalendar())
    first = result.flows[0]
    assert first.end <= VALUE_DATE < first.payment_date
    assert first.rate == pytest.approx(0.0345)


def test_unused_past_fixings_are_rejected_not_ignored() -> None:
    leg = _lagged_float_leg(
        past_fixings={date(2024, 7, 2): 0.0345, date(2030, 1, 2): 0.03},
    )
    with pytest.raises(ValueError, match="no corresponding unpaid period"):
        price_leg(leg, VALUE_DATE, _curve(), fc.TargetCalendar())


def test_past_fixings_rejected_on_a_fixed_leg() -> None:
    with pytest.raises(ValueError, match="past_fixings"):
        Leg(
            1_000_000, date(2024, 7, 2), date(2030, 7, 2), 1, "30/360", True,
            coupon=0.03, past_fixings={date(2024, 7, 2): 0.03},
        )


# ---------------------------------------------------------------------------
# Theta decomposition
# ---------------------------------------------------------------------------
def test_cap_theta_splits_into_option_decay_and_discount_roll() -> None:
    value_date = date(2025, 1, 2)
    curve = _curve(value_date)
    res = price_cap(
        value_date, date(2025, 4, 2), date(2030, 4, 2),
        1_000_000, 0.032, 2, flat_vol=0.20, curve=curve,
    )
    assert res.theta == pytest.approx(res.option_theta + res.discount_theta, abs=1e-9)
    assert res.option_theta < 0.0
    assert res.discount_theta > 0.0

    dt = 1.0 / 365.0
    option_fd = 0.0
    discount_fd = 0.0
    for caplet in res.caplets:
        premium = fc.forward_option_premium(caplet.forward, 0.032, caplet.ttm, caplet.vol, 1)
        decayed = fc.forward_option_premium(
            caplet.forward, 0.032, max(caplet.ttm - dt, 0.0), caplet.vol, 1
        )
        option_fd += (decayed - premium) * caplet.af * 1_000_000 * caplet.df
        tenor = (caplet.payment_date - value_date).days
        discount_fd += (
            premium * caplet.af * 1_000_000 * (curve.df_days(tenor - 1) - curve.df_days(tenor))
        )
    assert res.option_theta * dt == pytest.approx(option_fd, rel=1e-3)
    assert res.discount_theta * dt == pytest.approx(discount_fd, rel=1e-3)


# ---------------------------------------------------------------------------
# Key-rate convexity reconciliation
# ---------------------------------------------------------------------------
def test_key_rate_convexity_matrix_reconciles_with_parallel_convexity() -> None:
    value_date = date(2025, 1, 2)
    curve = _curve(value_date)
    fixed = Leg(1e6, date(2025, 4, 2), date(2032, 4, 2), 1, "30/360", True, coupon=0.033)
    floating = Leg(1e6, date(2025, 4, 2), date(2032, 4, 2), 2, "ACT/360", False)

    def pricer(c: fc.DiscountCurve) -> float:
        return price_irs(value_date, value_date, fixed, floating, c, fc.TargetCalendar()).payer_pv

    parallel = parallel_curve_risk(pricer, curve)
    diagonal = key_rate_risk(pricer, curve)
    full = key_rate_convexity_matrix(pricer, curve)

    assert sum(b.dv01 for b in diagonal) == pytest.approx(parallel.dv01, rel=1e-6)
    assert full.total == pytest.approx(parallel.convexity, rel=1e-6)
    # The diagonal alone must NOT reconcile: that is exactly why the matrix exists.
    assert sum(b.convexity for b in diagonal) != pytest.approx(parallel.convexity, rel=1e-3)


# ---------------------------------------------------------------------------
# Surface expiry interpolation
# ---------------------------------------------------------------------------
def test_piecewise_constant_forward_vol_steps_where_linear_vol_ramps() -> None:
    expiries = [1.0, 2.0, 3.0]
    strikes = [0.02, 0.04]
    vols = [[0.25, 0.25], [0.25, 0.25], [0.20, 0.20]]
    linear = VolSurface2D(expiries, strikes, vols, expiry_interpolation="linear_vol")
    flat_fwd = VolSurface2D(
        expiries, strikes, vols, expiry_interpolation="piecewise_constant_forward_vol"
    )
    variance = VolSurface2D(
        expiries, strikes, vols, expiry_interpolation="linear_total_variance"
    )
    for surface in (linear, flat_fwd, variance):
        for t in expiries:
            assert surface.vol(t, 0.03) == pytest.approx(0.25 if t < 3.0 else 0.20)
    # Between the last two nodes the three rules genuinely disagree.
    assert linear.vol(2.5, 0.03) == pytest.approx(0.225)
    assert flat_fwd.vol(2.5, 0.03) != pytest.approx(linear.vol(2.5, 0.03), abs=1e-6)
    assert variance.vol(2.5, 0.03) != pytest.approx(linear.vol(2.5, 0.03), abs=1e-6)


def test_total_variance_interpolation_cannot_create_negative_forward_variance() -> None:
    surface = VolSurface2D(
        [1.0, 4.0], [0.02, 0.04], [[0.30, 0.30], [0.20, 0.20]],
        expiry_interpolation="linear_total_variance",
    )
    previous = 0.0
    for i in range(1, 41):
        t = 1.0 + 3.0 * i / 40.0
        variance = surface.vol(t, 0.03) ** 2 * t
        assert variance >= previous - 1e-12
        previous = variance


def test_stripped_surface_carries_its_construction_interpolation() -> None:
    from fi_pricing.volatility import strip_caplet_vol_surface

    value_date = date(2025, 1, 2)
    curve = fc.DiscountCurve([0, 3650], [0.03, 0.03], value_date, interpolation="log_linear_df")
    surface = strip_caplet_vol_surface(
        value_date, date(2025, 7, 2),
        [date(2027, 7, 2), date(2029, 7, 2)],
        [0.02, 0.04],
        [[0.22, 0.21], [0.22, 0.21]],
        2, curve,
    )
    assert surface.expiry_interpolation == "piecewise_constant_forward_vol"


# ---------------------------------------------------------------------------
# Spot-starting cap stripping
# ---------------------------------------------------------------------------
def test_stripping_supports_spot_starting_caps() -> None:
    from fi_pricing.volatility import strip_caplet_vol_surface

    value_date = date(2025, 4, 2)
    curve = fc.DiscountCurve([0, 3650], [0.03, 0.03], value_date, interpolation="log_linear_df")
    spec = fc.ScheduleSpec(2, business_day_convention="unadjusted", stub="none")
    surface = strip_caplet_vol_surface(
        value_date,
        date(2025, 1, 2),  # effective before the valuation date
        [date(2027, 1, 2), date(2029, 1, 2)],
        [0.02, 0.04],
        [[0.22, 0.21], [0.22, 0.21]],
        2, curve, schedule_spec=spec,
    )
    # The already-fixed caplet carries no optionality and is excluded.
    assert min(surface.expiries) > 0.0


# ---------------------------------------------------------------------------
# Arbitrage checks and bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_reports_and_enforces_discount_factor_monotonicity() -> None:
    value_date = date(2025, 1, 2)
    good = bootstrap_single_curve(
        value_date,
        deposits=[DepositQuote(date(2025, 7, 2), 0.027)],
        swaps=[SwapQuote(date(2030, 1, 2), 0.031), SwapQuote(date(2035, 1, 2), 0.0325)],
    )
    assert all(c.passed for c in good.checks)
    assert max(abs(e) for e in good.repricing_errors) < 1e-12

    with pytest.raises(ArbitrageError, match="discount factor"):
        bootstrap_single_curve(
            value_date,
            deposits=[DepositQuote(date(2025, 7, 2), 0.05), DepositQuote(date(2026, 1, 2), -0.20)],
        )


def test_bootstrap_spot_lag_shifts_the_swap_start() -> None:
    value_date = date(2025, 1, 2)
    quotes = dict(
        deposits=[DepositQuote(date(2025, 7, 2), 0.027)],
        swaps=[SwapQuote(date(2030, 1, 2), 0.031)],
    )
    spot = bootstrap_single_curve(value_date, **quotes)
    lagged = bootstrap_single_curve(value_date, spot_lag_days=2, **quotes)
    assert max(abs(e) for e in lagged.repricing_errors) < 1e-12
    assert lagged.discount_factors[-1] != pytest.approx(spot.discount_factors[-1], abs=1e-9)


def test_butterfly_convexity_check_detects_a_negative_butterfly() -> None:
    assert check_butterfly_convexity([0.02, 0.03, 0.04], [0.010, 0.005, 0.002]).passed
    # Middle call too expensive relative to its neighbours -> concave in strike.
    assert not check_butterfly_convexity([0.02, 0.03, 0.04], [0.010, 0.009, 0.002]).passed


def test_discount_factor_check_flags_implausible_forwards() -> None:
    assert check_discount_factors([0, 365, 730], [1.0, 0.97, 0.94]).passed
    # A mildly rising discount factor is a negative forward, which is a
    # real market state, not an arbitrage. Only an implausible one is flagged.
    assert check_discount_factors([0, 365, 730], [1.0, 0.97, 0.98]).passed
    result = check_discount_factors([0, 365, 730], [1.0, 0.97, 1.40])
    assert not result.passed
    assert "floor" in result.violations[0]
    # Requiring non-negative forwards is available, as an explicit assumption.
    strict = check_discount_factors([0, 365, 730], [1.0, 0.97, 0.98], min_forward_rate=0.0)
    assert not strict.passed
    assert not check_discount_factors([0, 365], [1.0, -0.5]).passed


# ---------------------------------------------------------------------------
# Day count and error messages
# ---------------------------------------------------------------------------
def test_act_act_isda_splits_across_calendar_years() -> None:
    # 2024 is a leap year, 2025 is not.
    assert fc.year_fraction(date(2024, 1, 1), date(2025, 1, 1), "ACT/ACT") == pytest.approx(1.0)
    assert fc.year_fraction(date(2025, 1, 1), date(2026, 1, 1), "ACT/ACT") == pytest.approx(1.0)
    mixed = fc.year_fraction(date(2024, 7, 1), date(2025, 7, 1), "ACT/ACT")
    assert mixed == pytest.approx(184 / 366 + 181 / 365)


def test_shifted_black_error_message_points_at_the_shift() -> None:
    with pytest.raises(ValueError, match="increase shift"):
        fc.forward_option_premium(-0.05, -0.02, 2.0, 0.3, 1, model="shifted_black", shift=0.03)
