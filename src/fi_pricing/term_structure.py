"""Discount-curve representation used across pricing and risk modules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta


class DiscountCurve:
    """Continuously compounded zero curve.

    Parameters
    ----------
    days:
        Strictly increasing day-count pillars measured from ``value_date``.
    zero_rates:
        Continuously compounded annual zero rates at those pillars.
    interpolation:
        ``linear_zero`` or ``log_linear_df``.
    """

    def __init__(
        self,
        days: Sequence[int],
        zero_rates: Sequence[float],
        value_date: date | None = None,
        shift_bp: float = 0.0,
        interpolation: str = "linear_zero",
    ) -> None:
        if len(days) != len(zero_rates) or len(days) < 2:
            raise ValueError("days and zero_rates must have the same length >= 2")
        xs = [int(x) for x in days]
        ys = [float(y) for y in zero_rates]
        if xs[0] < 0 or any(b <= a for a, b in zip(xs, xs[1:], strict=False)):
            raise ValueError("curve days must be non-negative and strictly increasing")
        if not all(math.isfinite(y) for y in ys):
            raise ValueError("zero rates must be finite")
        mode = interpolation.strip().lower()
        if mode not in {"linear_zero", "log_linear_df"}:
            raise ValueError("interpolation must be 'linear_zero' or 'log_linear_df'")
        self.days = xs
        self.rates = ys
        self.value_date = value_date
        self.shift = float(shift_bp) / 10_000.0
        self.interpolation = mode

    @classmethod
    def from_discount_factors(
        cls,
        days: Sequence[int],
        discount_factors: Sequence[float],
        value_date: date | None = None,
        interpolation: str = "log_linear_df",
    ) -> "DiscountCurve":
        if len(days) != len(discount_factors) or len(days) < 2:
            raise ValueError("days and discount_factors must have the same length >= 2")
        xs = [int(x) for x in days]
        dfs = [float(x) for x in discount_factors]
        if any(x <= 0.0 or not math.isfinite(x) for x in dfs):
            raise ValueError("discount factors must be positive and finite")
        rates: list[float] = []
        proxy: float | None = None
        for t, df in zip(xs, dfs, strict=True):
            if t == 0:
                if not math.isclose(df, 1.0, abs_tol=1e-12):
                    raise ValueError("discount factor at day 0 must equal 1")
                rates.append(0.0)
            else:
                z = -365.0 * math.log(df) / t
                rates.append(z)
                if proxy is None:
                    proxy = z
        if xs[0] == 0 and proxy is not None:
            rates[0] = proxy
        return cls(xs, rates, value_date=value_date, interpolation=interpolation)

    def _segment(self, t_days: float) -> tuple[int, int]:
        if t_days <= self.days[0]:
            return 0, 0
        if t_days >= self.days[-1]:
            last = len(self.days) - 1
            return last, last
        lo = 0
        hi = len(self.days) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.days[mid] <= t_days:
                lo = mid
            else:
                hi = mid
        return lo, hi

    def _unshifted_zero(self, t_days: float) -> float:
        if t_days <= 0.0:
            idx = next((i for i, t in enumerate(self.days) if t > 0), 0)
            return self.rates[idx]
        i, j = self._segment(t_days)
        if i == j:
            return self.rates[i]
        t0, t1 = self.days[i], self.days[j]
        z0, z1 = self.rates[i], self.rates[j]
        weight = (t_days - t0) / (t1 - t0)
        if self.interpolation == "linear_zero":
            return z0 + weight * (z1 - z0)
        log_df0 = -z0 * t0 / 365.0
        log_df1 = -z1 * t1 / 365.0
        log_df = log_df0 + weight * (log_df1 - log_df0)
        return -365.0 * log_df / t_days

    def zero(self, t_days: float) -> float:
        return self._unshifted_zero(float(t_days)) + self.shift

    def df_days(self, t_days: float) -> float:
        t = float(t_days)
        return math.exp(-self.zero(t) * t / 365.0)

    def df(self, pay_date: date, value_date: date | None = None) -> float:
        anchor = value_date or self.value_date
        if anchor is None:
            raise ValueError("value_date required")
        return self.df_days((pay_date - anchor).days)

    def bumped(self, shift_bp: float) -> "DiscountCurve":
        return DiscountCurve(
            self.days,
            self.rates,
            self.value_date,
            shift_bp=self.shift * 10_000.0 + float(shift_bp),
            interpolation=self.interpolation,
        )

    def bumped_nodes(self, shifts_bp: Sequence[float]) -> "DiscountCurve":
        if len(shifts_bp) != len(self.days):
            raise ValueError("shifts_bp must have one value per curve node")
        shifted = [
            rate + self.shift + float(bump) / 10_000.0
            for rate, bump in zip(self.rates, shifts_bp, strict=True)
        ]
        return DiscountCurve(
            self.days,
            shifted,
            self.value_date,
            interpolation=self.interpolation,
        )

    def instantaneous_forward(self, t_days: float) -> float:
        """Numerical instantaneous forward ``-d log(P)/dt`` in annual units."""
        t = float(t_days)
        h = 1.0
        lo = max(0.0, t - h)
        hi = t + h
        log_df_lo = math.log(self.df_days(lo))
        log_df_hi = math.log(self.df_days(hi))
        return -(log_df_hi - log_df_lo) * 365.0 / (hi - lo)

    def roll_forward(self, horizon_days: int) -> "DiscountCurve":
        """Return the deterministic forward curve starting at ``horizon_days``."""
        if horizon_days < 0:
            raise ValueError("horizon_days must be non-negative")
        if horizon_days == 0:
            return DiscountCurve(
                self.days,
                [self.zero(t) for t in self.days],
                self.value_date,
                interpolation=self.interpolation,
            )
        p_h = self.df_days(horizon_days)
        new_rates: list[float] = []
        for tau in self.days:
            if tau == 0:
                new_rates.append(self.zero(horizon_days))
                continue
            forward_df = self.df_days(horizon_days + tau) / p_h
            new_rates.append(-365.0 * math.log(forward_df) / tau)
        new_date = self.value_date + timedelta(days=horizon_days) if self.value_date else None
        return DiscountCurve(
            self.days,
            new_rates,
            new_date,
            interpolation=self.interpolation,
        )
