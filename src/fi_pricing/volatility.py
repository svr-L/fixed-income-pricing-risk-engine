"""Volatility surfaces, implied-vol inversion and caplet-vol stripping."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from scipy.optimize import brentq

from . import arbitrage_checks as ac
from . import core as fc
from .curves import CurveSet, as_curve_set

EXPIRY_INTERPOLATIONS = ("linear_vol", "linear_total_variance", "piecewise_constant_forward_vol")


def validate_expiry_interpolation(mode: str) -> str:
    """Normalise and validate an expiry-interpolation rule name."""
    normalised = mode.strip().lower()
    if normalised not in EXPIRY_INTERPOLATIONS:
        raise ValueError(f"expiry_interpolation must be one of {EXPIRY_INTERPOLATIONS}")
    return normalised


def linear_interpolate(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Clamped linear interpolation."""
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            w = (x - xs[i]) / (xs[i + 1] - xs[i])
            return float(ys[i] + w * (ys[i + 1] - ys[i]))
    raise RuntimeError("interpolation bracket not found")


def interpolate_across_expiry(
    mode: str, expiry: float, expiries: Sequence[float], vols: Sequence[float]
) -> float:
    """Interpolate volatility across expiry under an explicit rule.

    Shared by :class:`VolSurface2D` and by the SABR swaption cube so the two
    cannot drift apart. Linear-in-vol is the naive default; the other two
    interpolate total variance, which is what prevents an interpolated point
    from carrying negative forward variance relative to its neighbours.
    """
    xs = tuple(float(x) for x in expiries)
    if mode == "linear_vol" or expiry <= xs[0] or expiry >= xs[-1]:
        return linear_interpolate(expiry, xs, vols)

    i = 0
    for k in range(len(xs) - 1):
        if xs[k] <= expiry <= xs[k + 1]:
            i = k
            break
    t1, t2 = xs[i], xs[i + 1]
    v1, v2 = float(vols[i]), float(vols[i + 1])
    w1, w2 = v1 * v1 * t1, v2 * v2 * t2

    if mode == "linear_total_variance":
        w = w1 + (expiry - t1) * (w2 - w1) / (t2 - t1)
    else:  # piecewise_constant_forward_vol
        forward_variance = (w2 - w1) / (t2 - t1) if t2 > t1 else 0.0
        w = w1 + forward_variance * (expiry - t1)
    if expiry <= 0.0:
        return v1
    return math.sqrt(max(w, 0.0) / expiry)


@dataclass(frozen=True)
class VolSurface2D:
    """Expiry/strike volatility surface with an explicit expiry-interpolation rule.

    Strike interpolation is always linear in volatility. The expiry rule matters
    and must match how the surface was built:

    ``linear_vol``
        Linear in volatility. Appropriate for quoted grids read at face value.
    ``linear_total_variance``
        Linear in sigma^2 * T, the arbitrage-friendlier default for smooth
        surfaces because it cannot create negative forward variance between
        two admissible nodes.
    ``piecewise_constant_forward_vol``
        Flat-forward between nodes: the volatility at an expiry between two
        nodes is the one implied by holding forward variance constant from the
        left node. This is the rule that :func:`strip_caplet_vol_surface`
        actually assumes, so stripped surfaces are returned with it.
    """

    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...]
    expiry_interpolation: str = "linear_vol"

    def __init__(
        self,
        expiries: Sequence[float],
        strikes: Sequence[float],
        vols: Sequence[Sequence[float]],
        expiry_interpolation: str = "linear_vol",
    ) -> None:
        exps = tuple(map(float, expiries))
        ks = tuple(map(float, strikes))
        grid = tuple(tuple(map(float, row)) for row in vols)
        mode = validate_expiry_interpolation(expiry_interpolation)
        if len(exps) < 2 or len(ks) < 2:
            raise ValueError("surface needs at least two expiries and strikes")
        if len(grid) != len(exps) or any(len(row) != len(ks) for row in grid):
            raise ValueError("vol grid shape must be expiry x strike")
        if any(x2 <= x1 for x1, x2 in zip(exps, exps[1:], strict=False)):
            raise ValueError("expiries must be strictly increasing")
        if any(x2 <= x1 for x1, x2 in zip(ks, ks[1:], strict=False)):
            raise ValueError("strikes must be strictly increasing")
        if any(v < 0.0 for row in grid for v in row):
            raise ValueError("volatilities must be non-negative")
        if any(t < 0.0 for t in exps):
            raise ValueError("expiries must be non-negative")
        object.__setattr__(self, "expiries", exps)
        object.__setattr__(self, "strikes", ks)
        object.__setattr__(self, "vols", grid)
        object.__setattr__(self, "expiry_interpolation", mode)

    @staticmethod
    def _linear(x: float, xs: tuple[float, ...], ys: Sequence[float]) -> float:
        if x <= xs[0]:
            return float(ys[0])
        if x >= xs[-1]:
            return float(ys[-1])
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                w = (x - xs[i]) / (xs[i + 1] - xs[i])
                return float(ys[i] + w * (ys[i + 1] - ys[i]))
        raise RuntimeError("interpolation bracket not found")

    # Kept as a public alias: strike interpolation is always linear in vol.
    _interp = _linear

    def _interp_expiry(self, expiry: float, ys: Sequence[float]) -> float:
        return interpolate_across_expiry(self.expiry_interpolation, expiry, self.expiries, ys)

    def vol(self, expiry: float, strike: float) -> float:
        by_expiry = [self._linear(strike, self.strikes, row) for row in self.vols]
        return self._interp_expiry(expiry, by_expiry)

    def bumped_node(self, expiry_index: int, strike_index: int, bump_abs: float) -> VolSurface2D:
        if bump_abs == 0.0:
            return self
        grid = [list(row) for row in self.vols]
        grid[expiry_index][strike_index] += bump_abs
        if grid[expiry_index][strike_index] < 0.0:
            raise ValueError("volatility bump produced a negative node")
        return VolSurface2D(
            self.expiries, self.strikes, grid, expiry_interpolation=self.expiry_interpolation
        )

    def calendar_arbitrage_check(self) -> ac.CheckResult:
        """Check that total variance is non-decreasing in expiry at every strike."""
        return ac.check_total_variance_monotone(self.expiries, self.vols, self.strikes)


def implied_volatility(
    target_price: float,
    F: float,
    K: float,
    T: float,
    phi: int,
    *,
    model: str = "black",
    shift: float = 0.0,
    initial_upper: float | None = None,
    max_upper: float = 100.0,
    tol: float = 1e-12,
) -> float:
    """Invert an undiscounted forward-option price to implied volatility.

    The function supports Black-76, shifted Black and Bachelier. Prices are per
    unit annuity/accrual, matching :func:`fi_pricing.core.forward_option_premium`.
    """
    if target_price < 0.0:
        raise ValueError("target_price must be non-negative")
    if T < 0.0:
        raise ValueError("T must be non-negative")
    intrinsic = max(phi * (F - K), 0.0)
    if target_price < intrinsic - tol:
        raise ValueError("target price is below intrinsic value")
    if abs(target_price - intrinsic) <= tol or T == 0.0:
        return 0.0

    def f(vol: float) -> float:
        return fc.forward_option_premium(F, K, T, vol, phi, model=model, shift=shift) - target_price

    name = model.strip().lower().replace("-", "_")
    upper = initial_upper if initial_upper is not None else (0.50 if name != "bachelier" else 0.02)
    upper = max(float(upper), 1e-10)
    while f(upper) < 0.0 and upper < max_upper:
        upper *= 2.0
    if f(upper) < 0.0:
        raise ValueError("could not bracket implied volatility")
    return float(brentq(f, 0.0, upper, xtol=tol, rtol=1e-12))


def implied_vol_black76(target_price: float, F: float, K: float, T: float, phi: int) -> float:
    return implied_volatility(target_price, F, K, T, phi, model="black")


def implied_vol_bachelier(target_price: float, F: float, K: float, T: float, phi: int) -> float:
    return implied_volatility(target_price, F, K, T, phi, model="bachelier")


def implied_vol_shifted_black(
    target_price: float,
    F: float,
    K: float,
    T: float,
    phi: int,
    shift: float,
) -> float:
    return implied_volatility(target_price, F, K, T, phi, model="shifted_black", shift=shift)


def _caplet_value(
    period: fc.CouponPeriod,
    value_date: date,
    curve: fc.DiscountCurve | CurveSet,
    strike: float,
    vol: float,
    basis: str,
    model: str,
    shift: float,
    index: str | None = None,
) -> float:
    """Discounted caplet value per unit notional.

    Projection and discounting are separated when a :class:`CurveSet` is
    supplied; a bare ``DiscountCurve`` preserves the single-curve path.
    """
    curves = as_curve_set(curve)
    discount = curves.discount
    projection = curves.project(index)
    af = fc.year_fraction(period.start, period.end, basis)
    df_start = projection.df(period.start, value_date)
    df_end = projection.df(period.end, value_date)
    forward = (df_start / df_end - 1.0) / af
    t = max((period.fixing_date - value_date).days / 365.0, 0.0)
    premium = fc.forward_option_premium(
        forward, strike, t, vol, 1, model=model, shift=shift
    )
    return premium * af * discount.df(period.payment_date, value_date)


def strip_caplet_vol_surface(
    value_date: date,
    effective: date,
    quote_maturities: Sequence[date],
    strikes: Sequence[float],
    cap_vols: Sequence[Sequence[float]],
    frequency: int,
    curve: fc.DiscountCurve | CurveSet,
    *,
    basis: str = "ACT/360",
    cal: fc.TargetCalendar | None = None,
    schedule_spec: fc.ScheduleSpec | None = None,
    model: str = "black",
    shift: float = 0.0,
    check_arbitrage: bool = True,
    index: str | None = None,
) -> VolSurface2D:
    """Strip a caplet-vol surface from quoted flat cap volatilities.

    Each cap-maturity interval shares one forward caplet volatility, so a quote
    that adds several caplets gives the whole new block a common stripped
    volatility. The returned surface therefore carries
    ``expiry_interpolation='piecewise_constant_forward_vol'``, matching the
    assumption used to build it: reading it with linear-in-vol interpolation
    would ramp across block boundaries instead of stepping.

    Spot-starting caps (``effective <= value_date``) are supported. Caplets whose
    fixing date is on or before the valuation date carry no optionality, so they
    are excluded from both the quote target and the stripped surface. Their value
    is vol-independent and cancels out of the stripping residual.
    """
    maturities = list(quote_maturities)
    ks = list(map(float, strikes))
    grid = [list(map(float, row)) for row in cap_vols]
    if len(maturities) < 2 or len(ks) < 2:
        raise ValueError("at least two cap maturities and strikes are required")
    if any(m2 <= m1 for m1, m2 in zip(maturities, maturities[1:], strict=False)):
        raise ValueError("quote maturities must be strictly increasing")
    if len(grid) != len(maturities) or any(len(row) != len(ks) for row in grid):
        raise ValueError("cap_vols shape must be maturity x strike")
    if any(v < 0.0 for row in grid for v in row):
        raise ValueError("cap volatilities must be non-negative")
    cal = cal or fc.TargetCalendar()
    spec = schedule_spec or fc.ScheduleSpec(frequency=frequency)
    if spec.frequency != frequency:
        raise ValueError("frequency and schedule_spec.frequency must match")

    def optionful(periods: Sequence[fc.CouponPeriod]) -> list[fc.CouponPeriod]:
        """Drop caplets whose fixing has already happened."""
        return [p for p in periods if p.fixing_date > value_date]

    final_periods = optionful(fc.build_schedule(effective, maturities[-1], spec, cal))
    if len(final_periods) < 2:
        raise ValueError("cap quotes must contain at least two live caplets to strip")
    period_keys = [(p.start, p.end) for p in final_periods]
    reset_expiries = [(p.fixing_date - value_date).days / 365.0 for p in final_periods]
    stripped_by_strike: list[dict[tuple[date, date], float]] = []

    for strike_index, strike in enumerate(ks):
        known: dict[tuple[date, date], float] = {}
        previous_keys: set[tuple[date, date]] = set()
        for maturity_index, maturity in enumerate(maturities):
            periods = optionful(fc.build_schedule(effective, maturity, spec, cal))
            current_keys = {(p.start, p.end) for p in periods}
            if not previous_keys.issubset(current_keys):
                raise ValueError("cap quote schedules must be nested across maturities")
            previous_keys = current_keys
            flat_vol = grid[maturity_index][strike_index]
            target = sum(
                _caplet_value(p, value_date, curve, strike, flat_vol, basis, model, shift, index)
                for p in periods
            )
            known_value = 0.0
            new_periods: list[fc.CouponPeriod] = []
            for p in periods:
                key = (p.start, p.end)
                if key in known:
                    known_value += _caplet_value(
                        p, value_date, curve, strike, known[key], basis, model, shift, index
                    )
                else:
                    new_periods.append(p)
            if not new_periods:
                raise ValueError("cap quote maturity did not add any new caplets")
            residual_target = target - known_value
            if residual_target < -1e-10:
                raise ValueError("cap quotes imply a negative residual option value")

            # Loop variables are bound as defaults so the closure never depends
            # on the enclosing loop state (ruff B023).
            def residual(
                vol: float,
                *,
                _periods: Sequence[fc.CouponPeriod] = tuple(new_periods),
                _strike: float = strike,
                _target: float = residual_target,
            ) -> float:
                priced = sum(
                    _caplet_value(p, value_date, curve, _strike, vol, basis, model, shift, index)
                    for p in _periods
                )
                return priced - _target

            if residual_target <= 1e-14:
                stripped = 0.0
            else:
                upper = 0.50 if model.strip().lower() != "bachelier" else 0.02
                while residual(upper) < 0.0 and upper < 100.0:
                    upper *= 2.0
                if residual(upper) < 0.0:
                    raise ValueError("could not bracket caplet-vol stripping root")
                stripped = float(brentq(residual, 0.0, upper, xtol=1e-13, rtol=1e-12))
            for p in new_periods:
                known[(p.start, p.end)] = stripped
        stripped_by_strike.append(known)

    out_grid = [
        [stripped_by_strike[j][key] for j in range(len(ks))] for key in period_keys
    ]
    surface = VolSurface2D(
        reset_expiries,
        ks,
        out_grid,
        expiry_interpolation="piecewise_constant_forward_vol",
    )
    if check_arbitrage:
        surface.calendar_arbitrage_check().raise_if_failed()
    return surface


@dataclass(frozen=True)
class CapSurfaceConversionResult:
    """Price-preserving conversion between flat cap-vol quote conventions."""

    surface: VolSurface2D
    source_prices: tuple[tuple[float, ...], ...]
    target_prices: tuple[tuple[float, ...], ...]
    price_errors: tuple[tuple[float, ...], ...]
    max_abs_price_error: float
    source_model: str
    target_model: str
    source_shift: float
    target_shift: float


def _flat_cap_price(
    value_date: date,
    effective: date,
    maturity: date,
    strike: float,
    vol: float,
    frequency: int,
    curve: fc.DiscountCurve | CurveSet,
    *,
    basis: str,
    cal: fc.TargetCalendar,
    schedule_spec: fc.ScheduleSpec,
    model: str,
    shift: float,
    index: str | None,
) -> float:
    """Flat-vol cap price per unit notional, excluding already-fixed caplets."""
    periods = [
        p
        for p in fc.build_schedule(effective, maturity, schedule_spec, cal)
        if p.fixing_date > value_date
    ]
    if not periods:
        raise ValueError("cap quote maturity must contain at least one live caplet")
    return sum(
        _caplet_value(
            p,
            value_date,
            curve,
            strike,
            vol,
            basis,
            model,
            shift,
            index,
        )
        for p in periods
    )


def convert_cap_vol_surface(
    value_date: date,
    effective: date,
    quote_maturities: Sequence[date],
    source_surface: VolSurface2D,
    frequency: int,
    curve: fc.DiscountCurve | CurveSet,
    *,
    basis: str = "ACT/360",
    cal: fc.TargetCalendar | None = None,
    schedule_spec: fc.ScheduleSpec | None = None,
    source_model: str = "bachelier",
    target_model: str = "shifted_black",
    source_shift: float = 0.0,
    target_shift: float = 0.0,
    index: str | None = None,
    tol: float = 1e-12,
    check_arbitrage: bool = False,
) -> CapSurfaceConversionResult:
    """Convert a flat cap-vol surface by preserving each quoted cap price.

    Conversion is performed at the cap-strip level, not caplet by caplet. For
    every quoted maturity/strike pair the source convention first produces a
    cap price; a single flat target volatility is then solved so the target
    convention reproduces exactly the same strip price.
    """
    maturities = tuple(quote_maturities)
    if len(maturities) != len(source_surface.expiries):
        raise ValueError("quote_maturities must align one-for-one with surface expiries")
    cal = cal or fc.TargetCalendar()
    spec = schedule_spec or fc.ScheduleSpec(frequency=frequency)
    if spec.frequency != frequency:
        raise ValueError("frequency and schedule_spec.frequency must match")

    expected_expiries = tuple(
        max((m - value_date).days / 365.0, 0.0) for m in maturities
    )
    for actual, expected in zip(
        source_surface.expiries, expected_expiries, strict=True
    ):
        if abs(actual - expected) > 5.0 / 365.0:
            raise ValueError(
                "surface expiry grid is inconsistent with quote_maturities"
            )

    converted: list[list[float]] = []
    source_prices: list[list[float]] = []
    target_prices: list[list[float]] = []
    errors: list[list[float]] = []

    for i, maturity in enumerate(maturities):
        row_vols: list[float] = []
        row_source: list[float] = []
        row_target: list[float] = []
        row_errors: list[float] = []
        for j, strike in enumerate(source_surface.strikes):
            source_vol = source_surface.vols[i][j]
            source_price = _flat_cap_price(
                value_date,
                effective,
                maturity,
                strike,
                source_vol,
                frequency,
                curve,
                basis=basis,
                cal=cal,
                schedule_spec=spec,
                model=source_model,
                shift=source_shift,
                index=index,
            )

            def residual(
                target_vol: float,
                *,
                _maturity: date = maturity,
                _strike: float = strike,
                _source_price: float = source_price,
            ) -> float:
                return (
                    _flat_cap_price(
                        value_date,
                        effective,
                        _maturity,
                        _strike,
                        target_vol,
                        frequency,
                        curve,
                        basis=basis,
                        cal=cal,
                        schedule_spec=spec,
                        model=target_model,
                        shift=target_shift,
                        index=index,
                    )
                    - _source_price
                )

            if source_price <= tol:
                target_vol = 0.0
            else:
                target_name = target_model.strip().lower().replace("-", "_")
                upper = 0.02 if target_name in {"bachelier", "normal"} else 0.50
                while residual(upper) < 0.0 and upper < 100.0:
                    upper *= 2.0
                if residual(upper) < 0.0:
                    raise ValueError(
                        "could not bracket target cap volatility for "
                        f"maturity={maturity}, strike={strike:.6g}"
                    )
                target_vol = float(
                    brentq(residual, 0.0, upper, xtol=tol, rtol=1e-12)
                )

            rebuilt = _flat_cap_price(
                value_date,
                effective,
                maturity,
                strike,
                target_vol,
                frequency,
                curve,
                basis=basis,
                cal=cal,
                schedule_spec=spec,
                model=target_model,
                shift=target_shift,
                index=index,
            )
            row_vols.append(target_vol)
            row_source.append(source_price)
            row_target.append(rebuilt)
            row_errors.append(rebuilt - source_price)
        converted.append(row_vols)
        source_prices.append(row_source)
        target_prices.append(row_target)
        errors.append(row_errors)

    surface = VolSurface2D(
        expected_expiries,
        source_surface.strikes,
        converted,
        expiry_interpolation=source_surface.expiry_interpolation,
    )
    if check_arbitrage:
        surface.calendar_arbitrage_check().raise_if_failed()
    frozen_errors = tuple(tuple(row) for row in errors)
    return CapSurfaceConversionResult(
        surface=surface,
        source_prices=tuple(tuple(row) for row in source_prices),
        target_prices=tuple(tuple(row) for row in target_prices),
        price_errors=frozen_errors,
        max_abs_price_error=max(abs(x) for row in frozen_errors for x in row),
        source_model=source_model,
        target_model=target_model,
        source_shift=float(source_shift),
        target_shift=float(target_shift),
    )
