import math
from datetime import date

import pytest

from fi_pricing.core import (
    DiscountCurve,
    TargetCalendar,
    black76_forward_premium,
    black76_greeks,
    black76_taylor_pnl,
    norm_cdf,
    norm_pdf,
    year_fraction,
)
from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES


def test_normal_distribution_helpers() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))


def test_target_calendar_easter_holidays_and_roll() -> None:
    cal = TargetCalendar()
    assert not cal.is_business_day(date(2020, 4, 10))  # Good Friday
    assert not cal.is_business_day(date(2020, 4, 13))  # Easter Monday
    assert cal.roll_following(date(2020, 4, 10)) == date(2020, 4, 14)


def test_day_count_conventions() -> None:
    d1 = date(2020, 1, 31)
    d2 = date(2021, 1, 31)
    assert year_fraction(d1, d2, "ACT/360") == pytest.approx(366 / 360)
    assert year_fraction(d1, d2, "ACT/365") == pytest.approx(366 / 365)
    assert year_fraction(d1, d2, "30/360") == pytest.approx(1.0)


def test_discount_curve_flat_clamps_and_bumps() -> None:
    curve = DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, date(2020, 10, 28))
    bumped = curve.bumped(1.0)
    assert bumped.zero(365) - curve.zero(365) == pytest.approx(0.0001)
    assert curve.df(date(2020, 10, 29)) == pytest.approx(curve.df_days(1))


def test_black76_premium_and_greeks_are_well_behaved() -> None:
    prem = black76_forward_premium(F=0.04, K=0.035, T=2.0, vol=0.20, phi=1)
    greeks = black76_greeks(F=0.04, K=0.035, T=2.0, vol=0.20, phi=1)
    assert prem > 0
    assert 0 < greeks["delta"] < 1
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0
    assert math.isfinite(greeks["vanna"])
    assert math.isfinite(greeks["volga"])


def test_black76_intrinsic_fallback() -> None:
    assert black76_forward_premium(F=0.04, K=0.035, T=0.0, vol=0.20, phi=1) == pytest.approx(
        0.005
    )
    assert black76_forward_premium(F=0.04, K=0.035, T=0.0, vol=0.20, phi=-1) == 0.0


def test_black76_vanna_volga_improve_small_shock_pnl_explain() -> None:
    F, K, T, vol, phi = 0.04, 0.035, 2.0, 0.20, 1
    dF, dvol, dt = 0.0005, 0.01, 1 / 252
    base = black76_forward_premium(F, K, T, vol, phi)
    shocked = black76_forward_premium(F + dF, K, T - dt, vol + dvol, phi)
    full = shocked - base
    explain = black76_taylor_pnl(F, K, T, vol, phi, dF=dF, dvol=dvol, dt=dt)
    assert explain["vanna"] != 0.0
    assert explain["volga"] != 0.0
    assert explain["total"] == pytest.approx(full, abs=2e-6)
