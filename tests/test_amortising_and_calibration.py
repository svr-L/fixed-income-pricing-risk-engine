"""Tests for amortising notionals and the vega-normalised calibration objective."""

from datetime import date

import numpy as np
import pytest

from fi_pricing import core as fc
from fi_pricing.cap_floor import price_cap
from fi_pricing.hull_white import (
    CalibrationInstrument,
    HullWhiteParams,
    calibrate_hull_white,
    swaption_price,
)
from fi_pricing.irs import Leg, price_leg
from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES

VALUE_DATE = date(2025, 1, 2)
EFFECTIVE = date(2025, 4, 2)
MATURITY = date(2030, 4, 2)
CAL = fc.TargetCalendar()


@pytest.fixture
def curve() -> fc.DiscountCurve:
    return fc.DiscountCurve(
        CURVE_DAYS, CURVE_ZERO_RATES, VALUE_DATE, interpolation="log_linear_df"
    )


def _linear_amortisation(frequency: int, notional: float = 1_000_000.0) -> dict:
    periods = fc.remaining_periods(EFFECTIVE, MATURITY, frequency, VALUE_DATE, CAL)
    return {p.start: notional * (1.0 - i / len(periods)) for i, p in enumerate(periods)}


# ---------------------------------------------------------------------------
# Amortising legs
# ---------------------------------------------------------------------------
def test_amortising_leg_uses_the_scheduled_notional_per_period(
    curve: fc.DiscountCurve,
) -> None:
    schedule = _linear_amortisation(1)
    leg = Leg(
        1_000_000, EFFECTIVE, MATURITY, 1, "30/360", True,
        coupon=0.03, amortisation=schedule,
    )
    result = price_leg(leg, VALUE_DATE, curve, CAL)
    outstanding = [f.notional for f in result.flows]
    assert outstanding == sorted(outstanding, reverse=True)
    assert outstanding[0] == pytest.approx(1_000_000.0)
    assert outstanding[-1] < outstanding[0]
    # The PV is the sum of the individual flows, each on its own notional.
    rebuilt = sum(
        f.payment * curve.df(f.payment_date, VALUE_DATE) for f in result.flows
    )
    assert result.pv == pytest.approx(rebuilt, rel=1e-12)


def test_bullet_and_amortising_agree_when_the_schedule_is_flat(
    curve: fc.DiscountCurve,
) -> None:
    periods = fc.remaining_periods(EFFECTIVE, MATURITY, 1, VALUE_DATE, CAL)
    flat = {p.start: 1_000_000.0 for p in periods}
    bullet = Leg(1_000_000, EFFECTIVE, MATURITY, 1, "30/360", True, coupon=0.03)
    scheduled = Leg(
        1_000_000, EFFECTIVE, MATURITY, 1, "30/360", True, coupon=0.03, amortisation=flat
    )
    a = price_leg(bullet, VALUE_DATE, curve, CAL)
    b = price_leg(scheduled, VALUE_DATE, curve, CAL)
    assert a.pv == pytest.approx(b.pv, abs=1e-12)
    assert a.annuity == pytest.approx(b.annuity, abs=1e-12)


def test_amortising_cap_scales_npv_and_vega_together(curve: fc.DiscountCurve) -> None:
    schedule = _linear_amortisation(2)
    common = dict(
        value_date=VALUE_DATE, effective=EFFECTIVE, maturity=MATURITY,
        notional=1_000_000, strike=0.032, frequency=2, flat_vol=0.20, curve=curve,
    )
    bullet = price_cap(**common)
    amortising = price_cap(**common, amortisation=schedule)
    assert amortising.npv < bullet.npv
    assert amortising.vega < bullet.vega
    assert sum(c.premium for c in amortising.caplets) == pytest.approx(
        amortising.npv, abs=1e-9
    )


def test_amortisation_dates_must_match_accrual_starts(curve: fc.DiscountCurve) -> None:
    leg = Leg(
        1_000_000, EFFECTIVE, MATURITY, 1, "30/360", True,
        coupon=0.03, amortisation={date(2099, 1, 1): 1.0},
    )
    with pytest.raises(ValueError, match="do not match any remaining accrual start"):
        price_leg(leg, VALUE_DATE, curve, CAL)

    with pytest.raises(ValueError, match="non-negative"):
        Leg(
            1_000_000, EFFECTIVE, MATURITY, 1, "30/360", True,
            coupon=0.03, amortisation={EFFECTIVE: -1.0},
        )


def test_a_fully_repaid_period_contributes_nothing(curve: fc.DiscountCurve) -> None:
    periods = fc.remaining_periods(EFFECTIVE, MATURITY, 1, VALUE_DATE, CAL)
    schedule = {p.start: (1_000_000.0 if i == 0 else 0.0) for i, p in enumerate(periods)}
    leg = Leg(
        1_000_000, EFFECTIVE, MATURITY, 1, "30/360", True, coupon=0.03,
        amortisation=schedule,
    )
    result = price_leg(leg, VALUE_DATE, curve, CAL)
    assert all(f.payment == 0.0 for f in result.flows[1:])
    first = result.flows[0]
    assert result.pv == pytest.approx(
        first.payment * curve.df(first.payment_date, VALUE_DATE), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Vega-normalised calibration
# ---------------------------------------------------------------------------
def _heterogeneous_targets(curve: fc.DiscountCurve, truth: HullWhiteParams):
    """Instruments whose premiums span more than an order of magnitude."""
    instruments = []
    for expiry, n in ((1.0, 1), (1.0, 2), (2.0, 1), (5.0, 10), (10.0, 20), (10.0, 25)):
        times = tuple(expiry + i for i in range(1, n + 1))
        accruals = (1.0,) * n
        annuity = sum(
            a * curve.df_days(t * 365.0) for t, a in zip(times, accruals, strict=True)
        )
        forward = (
            curve.df_days(expiry * 365.0) - curve.df_days(times[-1] * 365.0)
        ) / annuity
        price = swaption_price(truth, curve, expiry, times, accruals, forward)
        instruments.append(
            CalibrationInstrument(expiry, times, accruals, forward, price)
        )
    return instruments


def test_both_objectives_recover_the_truth_from_clean_targets(
    curve: fc.DiscountCurve,
) -> None:
    truth = HullWhiteParams(a=0.07, sigma=0.0115)
    instruments = _heterogeneous_targets(curve, truth)
    for objective in ("price", "vega_normalised"):
        result = calibrate_hull_white(
            instruments, curve, objective=objective, initial=HullWhiteParams(0.03, 0.02)
        )
        assert result.params.a == pytest.approx(truth.a, rel=1e-4)
        assert result.params.sigma == pytest.approx(truth.sigma, rel=1e-4)


def test_vega_normalisation_stops_large_premiums_dominating(
    curve: fc.DiscountCurve,
) -> None:
    """With noisy targets no exact fit exists, and the weighting starts to matter.

    Raw price residuals let the expensive long-dated swaptions set the fit, so
    the cheap short-dated ones absorb the error. Normalising by sensitivity to
    sigma puts them on comparable footing.
    """
    truth = HullWhiteParams(a=0.07, sigma=0.0115)
    clean = _heterogeneous_targets(curve, truth)
    premiums = [i.target_price for i in clean]
    assert max(premiums) / min(premiums) > 10.0

    rng = np.random.default_rng(7)
    noisy = [
        CalibrationInstrument(
            i.expiry, i.payment_times, i.accruals, i.strike,
            i.target_price * (1.0 + 0.02 * rng.standard_normal()),
        )
        for i in clean
    ]

    by_price = calibrate_hull_white(noisy, curve, objective="price")
    by_vega = calibrate_hull_white(noisy, curve, objective="vega_normalised")

    def relative(result) -> list[float]:
        return [abs(e) / i.target_price for i, e in zip(noisy, result.errors, strict=True)]

    assert max(relative(by_vega)) < max(relative(by_price))
    # The two cheapest instruments are the ones price matching neglects.
    cheap = sorted(range(len(noisy)), key=lambda k: noisy[k].target_price)[:2]
    assert sum(relative(by_vega)[k] for k in cheap) < sum(
        relative(by_price)[k] for k in cheap
    )


def test_unknown_objective_is_rejected(curve: fc.DiscountCurve) -> None:
    truth = HullWhiteParams(a=0.05, sigma=0.01)
    with pytest.raises(ValueError, match="objective must be"):
        calibrate_hull_white(
            _heterogeneous_targets(curve, truth), curve, objective="least_squares"
        )
