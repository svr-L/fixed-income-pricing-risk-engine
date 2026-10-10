"""Reusable no-arbitrage checks for curves and volatility surfaces.

The checks are deliberately small, dependency-light and return structured
diagnostics instead of only raising. Constructors (bootstrapper, caplet-vol
stripper) call them at the end of their build so that a bad quote set fails
loudly rather than producing a silently inadmissible object.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


class ArbitrageError(ValueError):
    """Raised when a constructed market object violates a no-arbitrage check."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single no-arbitrage check."""

    name: str
    passed: bool
    violations: tuple[str, ...] = field(default=())

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ArbitrageError(f"{self.name}: " + "; ".join(self.violations))


def check_discount_factors(
    days: Sequence[int],
    discount_factors: Sequence[float],
    *,
    tol: float = 1e-12,
    min_forward_rate: float = -0.05,
) -> CheckResult:
    """Discount factors must be positive, and forwards must be plausible.

    Positivity is a hard no-arbitrage condition. Monotonicity is **not**: a
    discount factor above one, or rising with maturity, simply means the implied
    forward rate is negative, which was the state of EUR rates from 2014 to
    2022. Rejecting those curves would reject the market.

    What is still worth catching is a forward so negative it can only come from
    a bad quote or a sign error. ``min_forward_rate`` is that floor, as a
    continuously compounded annual rate; the default of -5% passes any real
    negative-rate curve while still flagging a fat-fingered input. Set it to
    ``0.0`` to require non-negative forwards, which is a modelling assumption
    rather than an arbitrage condition.
    """
    if len(days) != len(discount_factors):
        raise ValueError("days and discount_factors must have equal length")
    bad: list[str] = []
    for t, df in zip(days, discount_factors, strict=True):
        if df <= 0.0:
            bad.append(f"non-positive discount factor {df:.6g} at day {t}")
    if bad:
        return CheckResult("discount factor admissibility", False, tuple(bad))

    pairs = list(zip(days, discount_factors, strict=True))
    for (t1, df1), (t2, df2) in zip(pairs, pairs[1:], strict=False):
        span = (t2 - t1) / 365.0
        if span <= 0.0:
            bad.append(f"non-increasing maturities at day {t1} and {t2}")
            continue
        forward = -math.log(df2 / df1) / span
        if forward < min_forward_rate - tol:
            bad.append(
                f"implied forward {forward:.4%} between day {t1} and day {t2} "
                f"is below the {min_forward_rate:.2%} floor"
            )
    return CheckResult("discount factor admissibility", not bad, tuple(bad))


def check_total_variance_monotone(
    expiries: Sequence[float],
    vols: Sequence[Sequence[float]],
    strikes: Sequence[float] | None = None,
    *,
    tol: float = 1e-12,
) -> CheckResult:
    """Total variance sigma^2 * T must be non-decreasing in expiry per strike.

    A decreasing total variance implies negative forward variance between two
    expiries, i.e. calendar-spread arbitrage.
    """
    bad: list[str] = []
    n_strikes = len(vols[0]) if vols else 0
    for j in range(n_strikes):
        label = f"strike {strikes[j]:.4f}" if strikes is not None else f"strike index {j}"
        previous_var = None
        previous_t = None
        for t, row in zip(expiries, vols, strict=True):
            variance = row[j] * row[j] * t
            if previous_var is not None and variance < previous_var - tol:
                bad.append(
                    f"{label}: total variance falls from {previous_var:.8g} at "
                    f"T={previous_t:.4f} to {variance:.8g} at T={t:.4f}"
                )
            previous_var = variance
            previous_t = t
    return CheckResult("calendar-spread (total variance) monotonicity", not bad, tuple(bad))


def check_butterfly_convexity(
    strikes: Sequence[float],
    prices: Sequence[float],
    *,
    label: str = "smile slice",
    tol: float = 1e-10,
) -> CheckResult:
    """Call prices must be convex in strike (non-negative butterfly spreads)."""
    if len(strikes) != len(prices):
        raise ValueError("strikes and prices must have equal length")
    bad: list[str] = []
    for i in range(1, len(strikes) - 1):
        k0, k1, k2 = strikes[i - 1], strikes[i], strikes[i + 1]
        c0, c1, c2 = prices[i - 1], prices[i], prices[i + 1]
        w = (k2 - k1) / (k2 - k0)
        butterfly = w * c0 + (1.0 - w) * c2 - c1
        if butterfly < -tol:
            bad.append(
                f"{label}: negative butterfly {butterfly:.3e} around strike {k1:.4f}"
            )
    return CheckResult("butterfly convexity in strike", not bad, tuple(bad))


def run_all(checks: Sequence[CheckResult], *, strict: bool = True) -> tuple[CheckResult, ...]:
    """Return every check result, raising on the first failure when ``strict``."""
    results = tuple(checks)
    if strict:
        for result in results:
            result.raise_if_failed()
    return results
