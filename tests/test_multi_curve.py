"""Tests for multi-curve pricing and compounded RFR coupons."""

from datetime import date

import pytest

from fi_pricing import core as fc
from fi_pricing.bootstrap import DepositQuote, SwapQuote, bootstrap_multi_curve
from fi_pricing.curves import CurveSet, as_curve_set
from fi_pricing.irs import Leg, price_irs, price_leg
from fi_pricing.rfr import RfrConvention, compounded_rate, observation_schedule
from fi_pricing.risk import multi_curve_risk

VALUE_DATE = date(2025, 1, 2)
CAL = fc.TargetCalendar()


def _flat(rate: float) -> fc.DiscountCurve:
    return fc.DiscountCurve([0, 3650], [rate, rate], VALUE_DATE, interpolation="log_linear_df")


def _period(start: date, end: date) -> fc.CouponPeriod:
    return fc.CouponPeriod(1, start, end, start, end, end, start)


# ---------------------------------------------------------------------------
# CurveSet
# ---------------------------------------------------------------------------
def test_bare_curve_is_promoted_and_prices_identically() -> None:
    curve = _flat(0.03)
    fixed = Leg(1e6, date(2025, 4, 2), date(2030, 4, 2), 1, "30/360", True, coupon=0.03)
    floating = Leg(1e6, date(2025, 4, 2), date(2030, 4, 2), 2, "ACT/360", False)
    bare = price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, curve, CAL).payer_pv
    promoted = price_irs(
        VALUE_DATE, VALUE_DATE, fixed, floating, CurveSet.single(curve), CAL
    ).payer_pv
    assert bare == pytest.approx(promoted, abs=1e-12)
    assert as_curve_set(curve).is_single_curve


def test_unknown_index_falls_back_to_the_discount_curve() -> None:
    curves = CurveSet(_flat(0.025), {"EURIBOR6M": _flat(0.03)})
    assert curves.project("EURIBOR6M").zero(365) == pytest.approx(0.03)
    assert curves.project("EURIBOR3M").zero(365) == pytest.approx(0.025)
    assert curves.project(None).zero(365) == pytest.approx(0.025)
    assert not curves.has_projection("EURIBOR3M")


def test_bumping_discount_and_projection_are_independent() -> None:
    curves = CurveSet(_flat(0.025), {"EURIBOR6M": _flat(0.03)})
    only_discount = curves.bumped_discount(10.0)
    assert only_discount.discount.zero(365) == pytest.approx(0.025 + 0.001)
    assert only_discount.project("EURIBOR6M").zero(365) == pytest.approx(0.03)

    only_projection = curves.bumped_projection(10.0)
    assert only_projection.discount.zero(365) == pytest.approx(0.025)
    assert only_projection.project("EURIBOR6M").zero(365) == pytest.approx(0.03 + 0.001)

    both = curves.bumped(10.0)
    assert both.discount.zero(365) == pytest.approx(0.026)
    assert both.project("EURIBOR6M").zero(365) == pytest.approx(0.031)

    with pytest.raises(KeyError):
        curves.bumped_projection(1.0, "SOFR")


# ---------------------------------------------------------------------------
# RFR compounding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2025, 4, 2), date(2025, 7, 2)),   # both business days
        (date(2025, 4, 5), date(2025, 7, 5)),   # opens and closes on a Saturday
        (date(2025, 4, 5), date(2025, 7, 6)),   # closes on a Sunday
        (date(2025, 12, 25), date(2026, 3, 25)),  # opens on a TARGET holiday
    ],
)
def test_compounding_telescopes_to_the_curve_ratio(start: date, end: date) -> None:
    """With no lag and no lockout, compounded overnight forwards must telescope.

    This is the invariant that catches dropped or double-counted calendar days
    in the observation schedule.
    """
    curve = _flat(0.03)
    period = _period(start, end)
    convention = RfrConvention(lag_days=0, lockout_days=0)
    schedule = observation_schedule(period, convention, CAL)

    assert sum(o.days for o in schedule) == (end - start).days
    assert schedule[0].start == start
    assert schedule[-1].end == end

    rate = compounded_rate(period, convention, curve, VALUE_DATE, CAL)
    expected = (curve.df(start, VALUE_DATE) / curve.df(end, VALUE_DATE) - 1.0) * 360.0
    expected /= (end - start).days
    assert rate == pytest.approx(expected, rel=1e-12)


def test_lag_and_lockout_change_the_rate_in_the_expected_direction() -> None:
    # Upward-sloping curve: observing earlier rates must lower the coupon.
    curve = fc.DiscountCurve(
        [0, 180, 3650], [0.020, 0.025, 0.040], VALUE_DATE, interpolation="log_linear_df"
    )
    period = _period(date(2025, 4, 2), date(2025, 7, 2))
    plain = compounded_rate(period, RfrConvention(lag_days=0), curve, VALUE_DATE, CAL)
    lookback = compounded_rate(
        period, RfrConvention(lag_days=5, shift_method="lookback"), curve, VALUE_DATE, CAL
    )
    lockout = compounded_rate(
        period, RfrConvention(lockout_days=5), curve, VALUE_DATE, CAL
    )
    assert lookback < plain
    assert lockout < plain


def test_observation_shift_moves_the_whole_window() -> None:
    curve = fc.DiscountCurve(
        [0, 180, 3650], [0.020, 0.025, 0.040], VALUE_DATE, interpolation="log_linear_df"
    )
    period = _period(date(2025, 4, 2), date(2025, 7, 2))
    convention = RfrConvention(lag_days=5, shift_method="observation_shift")
    schedule = observation_schedule(period, convention, CAL)
    shifted_start = CAL.workday(period.start, -5)
    shifted_end = CAL.workday(period.end, -5)
    assert schedule[0].start == shifted_start
    assert schedule[-1].end == shifted_end
    # The shifted window still telescopes, now over the shifted dates.
    rate = compounded_rate(period, convention, curve, VALUE_DATE, CAL)
    expected = (
        curve.df(shifted_start, VALUE_DATE) / curve.df(shifted_end, VALUE_DATE) - 1.0
    ) * 360.0 / (shifted_end - shifted_start).days
    assert rate == pytest.approx(expected, rel=1e-12)


def test_missing_overnight_fixings_are_rejected_not_forecast() -> None:
    curve = _flat(0.03)
    period = _period(date(2024, 12, 2), date(2025, 3, 3))  # already accruing
    with pytest.raises(ValueError, match="past overnight fixings"):
        compounded_rate(period, RfrConvention(), curve, VALUE_DATE, CAL)


def test_observed_compound_factor_substitutes_for_daily_history() -> None:
    curve = _flat(0.03)
    period = _period(date(2024, 12, 2), date(2025, 3, 3))
    rate = compounded_rate(
        period,
        RfrConvention(),
        curve,
        VALUE_DATE,
        CAL,
        observed_compounding=(VALUE_DATE, 1.0025),
    )
    assert 0.0 < rate < 0.10


def test_compounded_rfr_leg_prices_and_rejects_a_term_fixing() -> None:
    curve = _flat(0.03)
    leg = Leg(
        1e6, date(2025, 4, 2), date(2030, 4, 2), 4, "ACT/360", False,
        rfr=RfrConvention(index="ESTR"),
    )
    result = price_leg(leg, VALUE_DATE, curve, CAL)
    assert result.pv > 0.0
    assert leg.is_compounded_rfr

    with pytest.raises(ValueError, match="past_fixings"):
        Leg(
            1e6, date(2025, 4, 2), date(2030, 4, 2), 4, "ACT/360", False,
            rfr=RfrConvention(), current_fixing=0.03,
        )


# ---------------------------------------------------------------------------
# Two-stage bootstrap
# ---------------------------------------------------------------------------
def _multi_curve():
    ois_deposits = [DepositQuote(date(2025, 7, 2), 0.0250)]
    ois_swaps = [
        SwapQuote(date(2027, 1, 4), 0.0255),
        SwapQuote(date(2030, 1, 2), 0.0265),
        SwapQuote(date(2035, 1, 2), 0.0275),
    ]
    projection = [
        SwapQuote(date(2027, 1, 4), 0.0290),
        SwapQuote(date(2030, 1, 2), 0.0302),
        SwapQuote(date(2035, 1, 2), 0.0315),
    ]
    return bootstrap_multi_curve(
        VALUE_DATE,
        ois_deposits,
        ois_swaps,
        {"EURIBOR6M": projection},
        projection_deposits={"EURIBOR6M": [DepositQuote(date(2025, 7, 2), 0.0285)]},
    )


def test_two_stage_bootstrap_reprices_both_quote_sets() -> None:
    result = _multi_curve()
    assert max(abs(e) for e in result.discount.repricing_errors) < 1e-12
    assert max(abs(e) for e in result.projection["EURIBOR6M"].repricing_errors) < 1e-12
    assert all(c.passed for c in result.discount.checks)


def test_projection_curve_sits_above_the_discount_curve() -> None:
    result = _multi_curve()
    for tenor in (365, 1825, 3650):
        basis = result.basis_bp("EURIBOR6M", tenor)
        assert 10.0 < basis < 100.0


def test_projection_swaps_reprice_to_par_under_the_built_curve_set() -> None:
    result = _multi_curve()
    curves = result.curves
    for maturity, par in ((date(2030, 1, 2), 0.0302), (date(2035, 1, 2), 0.0315)):
        # Start the swap on the first future date so no period is mid-accrual.
        fixed = Leg(1e6, date(2025, 1, 3), maturity, 1, "30/360", True, coupon=par)
        floating = Leg(1e6, date(2025, 1, 3), maturity, 2, "ACT/360", False, index="EURIBOR6M")
        npv = price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, curves, CAL).payer_pv
        assert abs(npv) < 250.0  # a one-day start shift, not a pricing error


def test_single_curve_bootstrap_is_unchanged_by_the_multi_curve_path() -> None:
    """Passing OIS quotes only must reproduce the plain single-curve build."""
    result = _multi_curve()
    from fi_pricing.bootstrap import bootstrap_single_curve

    plain = bootstrap_single_curve(
        VALUE_DATE,
        deposits=[DepositQuote(date(2025, 7, 2), 0.0250)],
        swaps=[
            SwapQuote(date(2027, 1, 4), 0.0255),
            SwapQuote(date(2030, 1, 2), 0.0265),
            SwapQuote(date(2035, 1, 2), 0.0275),
        ],
    )
    for a, b in zip(plain.discount_factors, result.discount.discount_factors, strict=True):
        assert a == pytest.approx(b, rel=1e-14)


# ---------------------------------------------------------------------------
# Multi-curve risk decomposition
# ---------------------------------------------------------------------------
def test_discount_and_projection_dv01_split_and_sum_to_the_total() -> None:
    result = _multi_curve()
    curves = result.curves
    fixed = Leg(1e6, date(2025, 4, 2), date(2030, 4, 2), 1, "30/360", True, coupon=0.03)
    floating = Leg(
        1e6, date(2025, 4, 2), date(2030, 4, 2), 2, "ACT/360", False, index="EURIBOR6M"
    )

    def pricer(c: CurveSet) -> float:
        return price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, c, CAL).payer_pv

    risk = multi_curve_risk(pricer, curves)
    assert risk.discount_dv01 + risk.basis_dv01 == pytest.approx(risk.total_dv01, rel=1e-6)
    # A swap's rate risk lives almost entirely in the projection curve.
    assert abs(risk.discount_dv01) < 0.05 * abs(risk.total_dv01)
    assert risk.projection_dv01["EURIBOR6M"] == pytest.approx(risk.basis_dv01, rel=1e-12)


def test_single_curve_set_has_no_basis_risk() -> None:
    curves = CurveSet.single(_flat(0.03))
    fixed = Leg(1e6, date(2025, 4, 2), date(2030, 4, 2), 1, "30/360", True, coupon=0.03)
    floating = Leg(1e6, date(2025, 4, 2), date(2030, 4, 2), 2, "ACT/360", False)

    def pricer(c: CurveSet) -> float:
        return price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, c, CAL).payer_pv

    risk = multi_curve_risk(pricer, curves)
    assert risk.basis_dv01 == pytest.approx(0.0, abs=1e-12)
    assert risk.discount_dv01 == pytest.approx(risk.total_dv01, rel=1e-9)
