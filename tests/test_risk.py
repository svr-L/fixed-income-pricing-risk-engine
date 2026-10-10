from datetime import date

import numpy as np
import pytest

from fi_pricing import core as fc
from fi_pricing.cap_floor import price_cap
from fi_pricing.irs import Leg, price_irs
from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES
from fi_pricing.risk import (
    bucketed_surface_vega,
    carry_roll_down,
    curve_factor_risk,
    explain_curve_pnl,
    key_rate_risk,
    short_rate_parameter_greeks,
)
from fi_pricing.short_rate import ShortRateParams
from fi_pricing.volatility import VolSurface2D

VALUE_DATE = date(2025, 1, 2)


def _legs() -> tuple[Leg, Leg]:
    fixed = Leg(
        notional=1_000_000,
        effective=date(2024, 7, 2),
        maturity=date(2030, 7, 2),
        frequency=1,
        basis="30/360",
        is_fixed=True,
        coupon=0.0325,
    )
    floating = Leg(
        notional=1_000_000,
        effective=date(2024, 7, 2),
        maturity=date(2030, 7, 2),
        frequency=2,
        basis="ACT/360",
        is_fixed=False,
        current_fixing=0.0290,
    )
    return fixed, floating


def _payer_pricer(curve: fc.DiscountCurve) -> float:
    fixed, floating = _legs()
    return price_irs(date(2024, 12, 30), VALUE_DATE, fixed, floating, curve=curve).payer_pv


def test_key_rate_dv01_sums_to_parallel_dv01() -> None:
    curve = fc.DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, VALUE_DATE)
    buckets = key_rate_risk(_payer_pricer, curve)
    fixed, floating = _legs()
    parallel = price_irs(date(2024, 12, 30), VALUE_DATE, fixed, floating, curve=curve).dv01
    assert sum(x.dv01 for x in buckets) == pytest.approx(parallel, rel=2e-5)
    assert any(abs(x.convexity) > 0.0 for x in buckets)


def test_curve_factor_risk_reports_level_slope_curvature() -> None:
    curve = fc.DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, VALUE_DATE)
    factors = curve_factor_risk(_payer_pricer, curve)
    assert [x.name for x in factors] == ["level", "slope", "curvature"]
    fixed, floating = _legs()
    parallel = price_irs(date(2024, 12, 30), VALUE_DATE, fixed, floating, curve=curve).dv01
    assert factors[0].dv01 == pytest.approx(parallel, rel=1e-10)


def test_bucketed_surface_vega_adds_to_cap_vega() -> None:
    surface = VolSurface2D(
        [0.5, 1.0, 2.0, 5.0, 10.0],
        [0.01, 0.02, 0.03, 0.04, 0.05],
        [
            [0.24, 0.22, 0.20, 0.19, 0.19],
            [0.23, 0.21, 0.195, 0.185, 0.185],
            [0.22, 0.20, 0.19, 0.18, 0.18],
            [0.20, 0.19, 0.18, 0.175, 0.175],
            [0.19, 0.18, 0.175, 0.17, 0.17],
        ],
    )

    def cap_from_surface(s):
        return price_cap(
            value_date=VALUE_DATE,
            effective=date(2024, 10, 2),
            maturity=date(2030, 10, 2),
            notional=1_000_000,
            strike=0.0325,
            frequency=2,
            first_fixing=0.0290,
            style="CAP",
            caplet_vol_surface=s,
        ).npv

    base = price_cap(
        value_date=VALUE_DATE,
        effective=date(2024, 10, 2),
        maturity=date(2030, 10, 2),
        notional=1_000_000,
        strike=0.0325,
        frequency=2,
        first_fixing=0.0290,
        style="CAP",
        caplet_vol_surface=surface,
    )
    buckets = bucketed_surface_vega(cap_from_surface, surface)
    assert sum(x.dv_dvol for x in buckets) == pytest.approx(base.vega, rel=2e-6)
    assert sum(abs(x.dv_dvol) > 1e-8 for x in buckets) > 4


def test_curve_pnl_explain_reconciles_small_multinode_shock() -> None:
    curve = fc.DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, VALUE_DATE)
    node_years = np.asarray(CURVE_DAYS) / 365.0
    shocks = 1.5 - 0.08 * node_years  # bp: mild bull-flattening style move
    shocked = curve.bumped_nodes(shocks)
    exp = explain_curve_pnl(_payer_pricer, curve, shocked, include_cross_gamma=True)
    assert abs(exp.actual_pnl) > 1.0
    assert abs(exp.residual) < 0.02


def test_deterministic_carry_roll_down_is_internally_additive() -> None:
    curve = fc.DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, VALUE_DATE)
    fixed, floating = _legs()

    def dated_pricer(vd, c):
        return price_irs(date(2024, 12, 30), vd, fixed, floating, curve=c).payer_pv

    res = carry_roll_down(dated_pricer, VALUE_DATE, curve, horizon_days=5)
    assert res.total_pnl == pytest.approx(res.roll_down + res.financing_carry)
    assert np.isfinite(res.total_pnl)


def test_short_rate_model_parameter_greeks_are_finite() -> None:
    params = ShortRateParams(r0=0.028, kappa=0.45, theta=0.041, sigma=0.012)
    greeks = short_rate_parameter_greeks(
        _payer_pricer,
        model="vasicek",
        params=params,
        value_date=VALUE_DATE,
        curve_days=CURVE_DAYS,
    )
    assert [g.parameter for g in greeks] == ["r0", "kappa", "theta", "sigma"]
    assert all(np.isfinite(g.delta) and np.isfinite(g.gamma) for g in greeks)


def test_cir_model_parameter_greeks_are_finite() -> None:
    params = ShortRateParams(r0=0.030, kappa=0.60, theta=0.045, sigma=0.080)
    greeks = short_rate_parameter_greeks(
        _payer_pricer,
        model="cir",
        params=params,
        value_date=VALUE_DATE,
        curve_days=CURVE_DAYS,
    )
    assert [g.parameter for g in greeks] == ["r0", "kappa", "theta", "sigma"]
    assert all(np.isfinite(g.delta) and np.isfinite(g.gamma) for g in greeks)
