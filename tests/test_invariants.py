from datetime import date

import pytest

from fi_pricing import core as fc
from fi_pricing.cap_floor import price_cap
from fi_pricing.irs import Leg, price_irs, price_leg
from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES
from fi_pricing.risk import key_rate_risk, parallel_curve_risk
from fi_pricing.swaption import price_swaption

VALUE_DATE = date(2025, 1, 2)
CURVE = fc.DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, VALUE_DATE)
CAL = fc.TargetCalendar()


def _future_swap(coupon: float) -> tuple[Leg, Leg]:
    fixed = Leg(1_000_000, date(2025, 7, 2), date(2030, 7, 2), 1, "30/360", True, coupon=coupon)
    floating = Leg(1_000_000, date(2025, 7, 2), date(2030, 7, 2), 2, "ACT/360", False)
    return fixed, floating


def test_forward_floating_leg_telescopes() -> None:
    _, floating = _future_swap(0.03)
    result = price_leg(floating, VALUE_DATE, CURVE, CAL)
    periods = fc.remaining_periods(
        floating.effective,
        floating.maturity,
        floating.frequency,
        VALUE_DATE,
        CAL,
    )
    expected = floating.notional * (
        CURVE.df(periods[0].start, VALUE_DATE) - CURVE.df(periods[-1].end, VALUE_DATE)
    )
    assert result.pv == pytest.approx(expected, abs=1e-8)


def test_swap_at_break_even_coupon_has_zero_npv() -> None:
    fixed0, floating = _future_swap(0.0)
    float_pv = price_leg(floating, VALUE_DATE, CURVE, CAL).pv
    annuity = price_leg(fixed0, VALUE_DATE, CURVE, CAL).annuity
    par = float_pv / annuity
    fixed, floating = _future_swap(par)
    result = price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, curve=CURVE, cal=CAL)
    assert result.payer_pv == pytest.approx(0.0, abs=1e-8)


def test_cap_floor_parity_matches_payer_swap_for_future_periods() -> None:
    strike = 0.0325
    cap = price_cap(
        VALUE_DATE,
        date(2025, 7, 2),
        date(2030, 7, 2),
        1_000_000,
        strike,
        2,
        style="CAP",
        flat_vol=0.20,
        curve=CURVE,
        cal=CAL,
    )
    floor = price_cap(
        VALUE_DATE,
        date(2025, 7, 2),
        date(2030, 7, 2),
        1_000_000,
        strike,
        2,
        style="FLOOR",
        flat_vol=0.20,
        curve=CURVE,
        cal=CAL,
    )
    fixed = Leg(1_000_000, date(2025, 7, 2), date(2030, 7, 2), 2, "ACT/360", True, coupon=strike)
    floating = Leg(1_000_000, date(2025, 7, 2), date(2030, 7, 2), 2, "ACT/360", False)
    swap = price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, curve=CURVE, cal=CAL)
    assert cap.npv - floor.npv == pytest.approx(swap.payer_pv, abs=1e-8)


def test_key_rate_partition_reproduces_parallel_dv01() -> None:
    fixed, floating = _future_swap(0.0325)

    def pricer(curve: fc.DiscountCurve) -> float:
        return price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, curve=curve, cal=CAL).payer_pv

    parallel = parallel_curve_risk(pricer, CURVE).dv01
    buckets = key_rate_risk(pricer, CURVE)
    assert sum(x.dv01 for x in buckets) == pytest.approx(parallel, abs=1e-5)


def test_accrued_uses_leg_day_count_not_calendar_proration() -> None:
    leg = Leg(
        1_000_000,
        date(2024, 7, 2),
        date(2030, 7, 2),
        1,
        "30/360",
        True,
        coupon=0.0325,
    )
    result = price_leg(leg, VALUE_DATE, CURVE, CAL)
    current_start = result.flows[0].start
    expected = leg.notional * leg.coupon * fc.year_fraction(current_start, VALUE_DATE, "30/360")
    assert result.accrued == pytest.approx(expected, abs=1e-10)


def test_fixings_are_never_silently_ignored() -> None:
    future_float = Leg(
        1_000_000,
        date(2026, 1, 2),
        date(2030, 1, 2),
        2,
        "ACT/360",
        False,
        current_fixing=0.99,
    )
    with pytest.raises(ValueError, match="no current floating period"):
        price_leg(future_float, VALUE_DATE, CURVE, CAL)

    with pytest.raises(ValueError, match="no current cap/floor period"):
        price_cap(
            VALUE_DATE,
            date(2026, 1, 2),
            date(2030, 1, 2),
            1_000_000,
            0.03,
            2,
            first_fixing=0.99,
            flat_vol=0.20,
            curve=CURVE,
        )


def test_swaption_contract_validation() -> None:
    with pytest.raises(ValueError, match="style"):
        price_swaption(
            VALUE_DATE,
            date(2026, 1, 2),
            date(2026, 4, 2),
            date(2030, 4, 2),
            0.03,
            0.20,
            "P",
            curve=CURVE,
        )
    with pytest.raises(ValueError, match="expiry cannot be after"):
        price_swaption(
            VALUE_DATE,
            date(2028, 1, 2),
            date(2027, 4, 2),
            date(2030, 4, 2),
            0.03,
            0.20,
            "Payer",
            curve=CURVE,
        )


def test_vanilla_irs_rejects_mismatched_notionals() -> None:
    fixed, floating = _future_swap(0.03)
    floating.notional = 2_000_000
    with pytest.raises(ValueError, match="same notional"):
        price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, curve=CURVE, cal=CAL)
