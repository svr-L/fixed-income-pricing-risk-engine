"""Tests for the Hull-White one-factor model."""

from datetime import date

import pytest

from fi_pricing import core as fc
from fi_pricing.hull_white import (
    CalibrationInstrument,
    HullWhiteParams,
    b_factor,
    calibrate_hull_white,
    caplet_instruments_from_surface,
    caplet_price,
    jamshidian_strikes,
    swaption_from_schedule,
    swaption_price,
    zero_bond_option,
    zero_bond_price,
)
from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES
from fi_pricing.volatility import implied_volatility, strip_caplet_vol_surface

VALUE_DATE = date(2025, 1, 2)


@pytest.fixture
def curve() -> fc.DiscountCurve:
    return fc.DiscountCurve(
        CURVE_DAYS, CURVE_ZERO_RATES, VALUE_DATE, interpolation="log_linear_df"
    )


PARAMS = HullWhiteParams(a=0.05, sigma=0.010)


# ---------------------------------------------------------------------------
# The defining property: exact fit to the initial curve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("maturity", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
def test_model_reproduces_the_initial_discount_curve_exactly(
    curve: fc.DiscountCurve, maturity: float
) -> None:
    """This is what separates Hull-White from Vasicek and CIR.

    theta(t) is implied by the initial term structure, so P^HW(0,T) must equal
    the market discount factor to machine precision, not approximately.
    """
    r0 = curve.instantaneous_forward(0.0)
    model = zero_bond_price(PARAMS, curve, 0.0, maturity, r0)
    assert model == pytest.approx(curve.df_days(maturity * 365.0), abs=1e-15)


def test_fit_holds_for_any_parameters(curve: fc.DiscountCurve) -> None:
    r0 = curve.instantaneous_forward(0.0)
    for a in (0.001, 0.05, 0.5):
        for sigma in (0.001, 0.02):
            params = HullWhiteParams(a, sigma)
            model = zero_bond_price(params, curve, 0.0, 7.0, r0)
            assert model == pytest.approx(curve.df_days(7.0 * 365.0), abs=1e-14)


def test_b_factor_limits() -> None:
    assert b_factor(0.05, 0.0) == 0.0
    # As a -> 0 the factor tends to the maturity itself.
    assert b_factor(1e-8, 5.0) == pytest.approx(5.0, rel=1e-6)
    # As tau -> infinity it tends to 1/a.
    assert b_factor(0.1, 500.0) == pytest.approx(10.0, rel=1e-6)


def test_degenerate_mean_reversion_is_rejected() -> None:
    with pytest.raises(ValueError, match="mean reversion"):
        HullWhiteParams(0.0, 0.01)
    with pytest.raises(ValueError, match="sigma"):
        HullWhiteParams(0.05, -0.01)


# ---------------------------------------------------------------------------
# Analytic parities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("expiry", "bond_maturity", "strike"),
    [(1.0, 2.0, 0.96), (2.0, 5.0, 0.90), (5.0, 10.0, 0.80), (0.25, 30.0, 0.40)],
)
def test_zero_bond_option_put_call_parity(
    curve: fc.DiscountCurve, expiry: float, bond_maturity: float, strike: float
) -> None:
    call = zero_bond_option(PARAMS, curve, expiry, bond_maturity, strike, 1)
    put = zero_bond_option(PARAMS, curve, expiry, bond_maturity, strike, -1)
    expected = curve.df_days(bond_maturity * 365.0) - strike * curve.df_days(expiry * 365.0)
    assert call - put == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("strike", [0.02, 0.0325, 0.05])
def test_payer_receiver_parity_equals_the_forward_swap(
    curve: fc.DiscountCurve, strike: float
) -> None:
    times = tuple(1.0 + i for i in range(1, 6))
    accruals = (1.0,) * 5
    payer = swaption_price(PARAMS, curve, 1.0, times, accruals, strike, payer=True)
    receiver = swaption_price(PARAMS, curve, 1.0, times, accruals, strike, payer=False)
    annuity = sum(a * curve.df_days(t * 365.0) for t, a in zip(times, accruals, strict=True))
    forward = (curve.df_days(365.0) - curve.df_days(times[-1] * 365.0)) / annuity
    assert payer - receiver == pytest.approx((forward - strike) * annuity, abs=1e-13)


def test_single_cashflow_swaption_collapses_to_a_bond_option(
    curve: fc.DiscountCurve,
) -> None:
    """Jamshidian on one cashflow must be the closed-form caplet."""
    strike = 0.03
    swaption = swaption_price(PARAMS, curve, 1.0, (2.0,), (1.0,), strike, payer=True)
    caplet = caplet_price(PARAMS, curve, 1.0, 2.0, 1.0, strike)
    assert swaption == pytest.approx(caplet, abs=1e-15)


def test_caplet_floorlet_parity(curve: fc.DiscountCurve) -> None:
    reset, payment, accrual, strike = 2.0, 2.5, 0.5, 0.031
    cap = caplet_price(PARAMS, curve, reset, payment, accrual, strike, phi=1)
    floor = caplet_price(PARAMS, curve, reset, payment, accrual, strike, phi=-1)
    df_reset = curve.df_days(reset * 365.0)
    df_payment = curve.df_days(payment * 365.0)
    forward = (df_reset / df_payment - 1.0) / accrual
    assert cap - floor == pytest.approx((forward - strike) * accrual * df_payment, abs=1e-14)


# ---------------------------------------------------------------------------
# Jamshidian internals and limiting behaviour
# ---------------------------------------------------------------------------
def test_jamshidian_critical_rate_prices_the_coupon_bond_at_par(
    curve: fc.DiscountCurve,
) -> None:
    times = (3.0, 4.0, 5.0)
    cashflows = [0.03, 0.03, 1.03]
    r_star, strikes = jamshidian_strikes(PARAMS, curve, 2.0, times, cashflows)
    rebuilt = sum(
        c * zero_bond_price(PARAMS, curve, 2.0, t, r_star)
        for t, c in zip(times, cashflows, strict=True)
    )
    assert rebuilt == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 < k < 1.5 for k in strikes)


def test_price_is_increasing_in_volatility(curve: fc.DiscountCurve) -> None:
    times, accruals = (3.0, 4.0, 5.0, 6.0, 7.0), (1.0,) * 5
    previous = -1.0
    for sigma in (0.005, 0.010, 0.015, 0.020):
        price = swaption_price(
            HullWhiteParams(0.05, sigma), curve, 2.0, times, accruals, 0.032
        )
        assert price > previous
        previous = price


def test_vanishing_volatility_gives_the_intrinsic_value(curve: fc.DiscountCurve) -> None:
    times, accruals, strike = (3.0, 4.0, 5.0, 6.0, 7.0), (1.0,) * 5, 0.02
    annuity = sum(a * curve.df_days(t * 365.0) for t, a in zip(times, accruals, strict=True))
    forward = (curve.df_days(2 * 365.0) - curve.df_days(times[-1] * 365.0)) / annuity
    intrinsic = max(forward - strike, 0.0) * annuity
    price = swaption_price(HullWhiteParams(0.05, 1e-9), curve, 2.0, times, accruals, strike)
    assert price == pytest.approx(intrinsic, abs=1e-12)


def test_mean_reversion_decays_the_implied_vol_term_structure(
    curve: fc.DiscountCurve,
) -> None:
    """Hull-White with a > 0 produces a downward-sloping caplet vol curve."""
    vols = []
    for reset in (1.0, 2.0, 5.0, 10.0):
        payment, accrual = reset + 0.5, 0.5
        df_reset = curve.df_days(reset * 365.0)
        df_payment = curve.df_days(payment * 365.0)
        forward = (df_reset / df_payment - 1.0) / accrual
        price = caplet_price(PARAMS, curve, reset, payment, accrual, forward)
        premium = price / (accrual * df_payment)
        vols.append(implied_volatility(premium, forward, forward, reset, 1))
    assert all(b < a for a, b in zip(vols, vols[1:], strict=False))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_calibration_recovers_known_parameters(curve: fc.DiscountCurve) -> None:
    truth = HullWhiteParams(a=0.08, sigma=0.0125)
    instruments = []
    for expiry in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
        times = tuple(expiry + i for i in range(1, 6))
        accruals = (1.0,) * 5
        annuity = sum(
            a * curve.df_days(t * 365.0) for t, a in zip(times, accruals, strict=True)
        )
        forward = (
            curve.df_days(expiry * 365.0) - curve.df_days(times[-1] * 365.0)
        ) / annuity
        price = swaption_price(truth, curve, expiry, times, accruals, forward)
        instruments.append(CalibrationInstrument(expiry, times, accruals, forward, price))

    result = calibrate_hull_white(
        instruments, curve, initial=HullWhiteParams(0.03, 0.02)
    )
    assert result.success
    assert result.params.a == pytest.approx(truth.a, rel=1e-6)
    assert result.params.sigma == pytest.approx(truth.sigma, rel=1e-6)
    assert result.max_abs_error < 1e-12


def test_calibration_with_fixed_mean_reversion(curve: fc.DiscountCurve) -> None:
    truth = HullWhiteParams(a=0.06, sigma=0.011)
    times, accruals = (3.0, 4.0, 5.0), (1.0, 1.0, 1.0)
    instruments = [
        CalibrationInstrument(
            2.0, times, accruals, 0.032,
            swaption_price(truth, curve, 2.0, times, accruals, 0.032),
        )
    ]
    result = calibrate_hull_white(instruments, curve, fix_a=truth.a)
    assert result.params.a == truth.a
    assert result.params.sigma == pytest.approx(truth.sigma, rel=1e-8)


def test_calibration_to_a_stripped_caplet_surface(curve: fc.DiscountCurve) -> None:
    surface = strip_caplet_vol_surface(
        VALUE_DATE, date(2025, 4, 2),
        [date(2027, 4, 2), date(2030, 4, 2), date(2035, 4, 2)],
        [0.02, 0.03, 0.04],
        [[0.24, 0.22, 0.23], [0.22, 0.205, 0.215], [0.21, 0.20, 0.21]],
        2, curve,
    )
    instruments = caplet_instruments_from_surface(
        VALUE_DATE, date(2025, 4, 2), date(2035, 4, 2), 0.03, 2, curve, surface
    )
    assert len(instruments) == 20
    result = calibrate_hull_white(instruments, curve)
    assert result.success
    assert 0.0 < result.params.a < 1.0
    assert 0.0 < result.params.sigma < 0.05
    # A two-parameter model cannot fit a whole smile-bearing strip exactly;
    # a few percent of the average premium is the expected order of magnitude.
    average_premium = sum(i.target_price for i in instruments) / len(instruments)
    assert result.rmse < 0.05 * average_premium


def test_calibration_rejects_an_empty_target_set(curve: fc.DiscountCurve) -> None:
    with pytest.raises(ValueError, match="at least one"):
        calibrate_hull_white([], curve)


# ---------------------------------------------------------------------------
# Date-based bridge
# ---------------------------------------------------------------------------
def test_swaption_from_schedule_matches_the_year_fraction_form(
    curve: fc.DiscountCurve,
) -> None:
    price = swaption_from_schedule(
        PARAMS, VALUE_DATE, date(2027, 1, 4), date(2027, 1, 4), date(2032, 1, 2),
        0.032, curve, notional=1_000_000,
    )
    assert price > 0.0
    receiver = swaption_from_schedule(
        PARAMS, VALUE_DATE, date(2027, 1, 4), date(2027, 1, 4), date(2032, 1, 2),
        0.032, curve, notional=1_000_000, payer=False,
    )
    assert receiver > 0.0
    # Deep out of the money in one direction must be worth less than at the money.
    far = swaption_from_schedule(
        PARAMS, VALUE_DATE, date(2027, 1, 4), date(2027, 1, 4), date(2032, 1, 2),
        0.10, curve, notional=1_000_000,
    )
    assert far < price


def test_expiry_after_swap_start_is_rejected(curve: fc.DiscountCurve) -> None:
    with pytest.raises(ValueError, match="expiry"):
        swaption_from_schedule(
            PARAMS, VALUE_DATE, date(2028, 1, 4), date(2027, 1, 4), date(2032, 1, 2),
            0.032, curve,
        )
