import numpy as np

from fi_pricing.short_rate import (
    ShortRateParams,
    calibrate_cir_q,
    calibrate_vasicek_q,
    cir_zero_rate,
    fit_cir_p,
    fit_vasicek_p,
    simulate_cir,
    simulate_vasicek,
    vasicek_zero_rate,
)


def test_vasicek_q_reproduces_synthetic_curve():
    true = ShortRateParams(0.028, 0.45, 0.041, 0.012)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30.0])
    market = vasicek_zero_rate(maturities, true.r0, true.kappa, true.theta, true.sigma)
    res = calibrate_vasicek_q(
        maturities,
        market,
        initial=ShortRateParams(0.025, 0.20, 0.035, 0.02),
    )
    assert res.success
    assert res.rmse < 1e-8
    assert res.max_abs_error < 3e-8


def test_cir_q_reproduces_synthetic_curve_and_feller_condition():
    true = ShortRateParams(0.030, 0.60, 0.045, 0.080)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30.0])
    market = cir_zero_rate(maturities, true.r0, true.kappa, true.theta, true.sigma)
    res = calibrate_cir_q(
        maturities,
        market,
        initial=ShortRateParams(0.027, 0.30, 0.040, 0.06),
        enforce_feller=True,
    )
    assert res.success
    assert res.rmse < 5e-8
    assert res.params.feller_margin >= -1e-8


def test_vasicek_p_fit_on_exact_simulation_is_economically_close():
    true = ShortRateParams(0.030, 0.80, 0.040, 0.015)
    rates = simulate_vasicek(true, n_steps=8_000, dt=1 / 252, seed=7)
    res = fit_vasicek_p(rates, dt=1 / 252)
    assert res.success
    assert abs(res.params.theta - true.theta) < 0.008
    assert abs(res.params.sigma - true.sigma) < 0.002
    assert 0.35 < res.params.kappa < 1.40
    assert np.isfinite(res.log_likelihood)


def test_cir_p_fit_on_exact_simulation_is_admissible_and_close():
    true = ShortRateParams(0.030, 0.90, 0.045, 0.090)
    rates = simulate_cir(true, n_steps=3_000, dt=1 / 52, seed=11)
    res = fit_cir_p(rates, dt=1 / 52, enforce_feller=True)
    assert res.success
    assert res.params.kappa > 0.0
    assert res.params.theta > 0.0
    assert res.params.sigma > 0.0
    assert res.params.feller_margin >= -1e-8
    assert abs(res.params.theta - true.theta) < 0.015
    assert abs(res.params.sigma - true.sigma) < 0.025
    assert np.isfinite(res.log_likelihood)
