"""Deterministic synthetic fixtures used by examples and regression tests.

These numbers are generated analytically inside the package.  They are not
historical market observations and are not copied from any external dataset.
"""

from __future__ import annotations

import math

CURVE_DAYS = [1, 30, 90, 180, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300, 10950]


def _synthetic_zero_rate(days: int) -> float:
    years = days / 365.0
    return 0.025 + 0.010 * (1.0 - math.exp(-years / 2.5)) - 0.004 * (1.0 - math.exp(-years / 12.0))


CURVE_ZERO_RATES = [_synthetic_zero_rate(days) for days in CURVE_DAYS]

VOL_CUBE_EXPIRIES = [1, 2, 3, 5, 7, 10, 15, 20]
VOL_CUBE_STRIKES = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060]


def _synthetic_vol_percent(expiry: float, strike: float) -> float:
    term = 21.5 - 0.27 * expiry + 1.25 * math.exp(-expiry / 4.0)
    smile = 12_000.0 * (strike - 0.030) ** 2
    skew = -18.0 * (strike - 0.030)
    return max(8.0, term + smile + skew)


VOL_CUBE = [
    [_synthetic_vol_percent(expiry, strike) for strike in VOL_CUBE_STRIKES]
    for expiry in VOL_CUBE_EXPIRIES
]
