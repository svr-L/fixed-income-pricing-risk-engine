"""Synthetic P/Q short-rate calibration example."""

import numpy as np

from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES
from fi_pricing.short_rate import (
    ShortRateParams,
    calibrate_cir_q,
    calibrate_vasicek_q,
    fit_cir_p,
    fit_vasicek_p,
    simulate_cir,
    simulate_vasicek,
)

maturities = np.asarray(CURVE_DAYS, dtype=float) / 365.0
zero_rates = np.asarray(CURVE_ZERO_RATES, dtype=float)

vas_q = calibrate_vasicek_q(maturities, zero_rates)
cir_q = calibrate_cir_q(maturities, zero_rates)

print("Q-measure curve calibration")
print("Vasicek:", vas_q.params, "RMSE=", f"{vas_q.rmse:.3e}")
print("CIR:    ", cir_q.params, "RMSE=", f"{cir_q.rmse:.3e}")

vas_true = ShortRateParams(r0=0.030, kappa=0.80, theta=0.040, sigma=0.015)
vas_path = simulate_vasicek(vas_true, n_steps=5_000, dt=1 / 252, seed=7)
vas_p = fit_vasicek_p(vas_path, dt=1 / 252)

cir_true = ShortRateParams(r0=0.030, kappa=0.90, theta=0.045, sigma=0.090)
cir_path = simulate_cir(cir_true, n_steps=2_500, dt=1 / 52, seed=11)
cir_p = fit_cir_p(cir_path, dt=1 / 52)

print("\nP-measure estimation on synthetic histories")
print("Vasicek:", vas_p.params, "logL=", f"{vas_p.log_likelihood:.1f}")
print("CIR:    ", cir_p.params, "logL=", f"{cir_p.log_likelihood:.1f}")
