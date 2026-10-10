"""Single-curve bootstrapping from deposit, FRA and par-swap quotes.

The module is intentionally compact and transparent. It is not a replacement for
an industrial multi-curve builder; it provides a quoted-instrument pipeline for
educational validation and for testing downstream pricing/risk analytics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from scipy.optimize import brentq

from .arbitrage_checks import CheckResult, check_discount_factors
from .core import (
    DiscountCurve,
    ScheduleSpec,
    TargetCalendar,
    build_schedule,
    year_fraction,
)
from .curves import CurveSet
from .rfr import RfrConvention, compounded_rate


@dataclass(frozen=True)
class DepositQuote:
    maturity: date
    rate: float
    basis: str = "ACT/360"


@dataclass(frozen=True)
class FRAQuote:
    start: date
    end: date
    rate: float
    basis: str = "ACT/360"


@dataclass(frozen=True)
class SwapQuote:
    """Par swap quote.

    ``float_frequency``/``float_schedule_spec`` are only used to locate the
    floating leg's first accrual start and final payment date, which matter once
    a payment lag is in play: the telescoped floating PV must discount the
    notional repayment to the actual payment date, not to the accrual end. When
    ``float_schedule_spec`` is omitted, the floating leg inherits the fixed
    leg's business-day, stub, EOM and lag conventions at ``float_frequency``.
    """

    maturity: date
    fixed_rate: float
    fixed_frequency: int = 1
    fixed_basis: str = "30/360"
    schedule_spec: ScheduleSpec | None = None
    float_frequency: int = 2
    float_schedule_spec: ScheduleSpec | None = None


@dataclass(frozen=True)
class OISSwapQuote:
    """Par OIS swap: fixed against a compounded overnight rate.

    Unlike :class:`SwapQuote`, the floating leg is priced by explicitly
    compounding overnight forwards through :mod:`fi_pricing.rfr`. With no lag
    and no lockout the compounding telescopes and this reproduces the
    ``SwapQuote`` shortcut exactly; with a lookback or a lockout it does not,
    and the shortcut would be silently wrong.
    """

    maturity: date
    fixed_rate: float
    fixed_frequency: int = 1
    fixed_basis: str = "ACT/360"
    float_frequency: int = 1
    rfr: RfrConvention | None = None
    schedule_spec: ScheduleSpec | None = None
    float_schedule_spec: ScheduleSpec | None = None
    past_fixings: Mapping[date, float] | None = None
    """Overnight history. Required when a lookback pulls the first period's
    observation window back before the valuation date."""


@dataclass(frozen=True)
class BootstrapResult:
    curve: DiscountCurve
    node_dates: tuple[date, ...]
    discount_factors: tuple[float, ...]
    repricing_errors: tuple[float, ...]
    checks: tuple[CheckResult, ...] = ()


def _curve_from_nodes(
    value_date: date,
    node_dates: Sequence[date],
    dfs: Sequence[float],
    interpolation: str,
) -> DiscountCurve:
    days = [(d - value_date).days for d in node_dates]
    return DiscountCurve.from_discount_factors(days, dfs, value_date, interpolation=interpolation)


def _append_node(node_dates: list[date], dfs: list[float], d: date, df: float) -> None:
    if d <= node_dates[-1]:
        raise ValueError("bootstrap instruments must add strictly increasing maturity nodes")
    if df <= 0.0:
        raise ValueError("bootstrapped discount factor must be positive")
    node_dates.append(d)
    dfs.append(float(df))


def bootstrap_single_curve(
    value_date: date,
    deposits: Sequence[DepositQuote] = (),
    fras: Sequence[FRAQuote] = (),
    swaps: Sequence[SwapQuote] = (),
    ois_swaps: Sequence[OISSwapQuote] = (),
    *,
    cal: TargetCalendar | None = None,
    interpolation: str = "log_linear_df",
    spot_lag_days: int = 0,
    check_arbitrage: bool = True,
    discount_curve: DiscountCurve | None = None,
) -> BootstrapResult:
    """Bootstrap a spot single curve from money-market and par-swap quotes.

    Instruments are processed by maturity. Deposits and FRAs have direct DF
    updates; swap terminal DFs are solved by root finding while intermediate
    coupon dates use the chosen curve interpolation.

    ``spot_lag_days`` shifts the swap start to the market spot date (T+2 for
    EUR) instead of assuming swaps start on the valuation date. The resulting
    discount factors are checked for positivity and strict monotonicity unless
    ``check_arbitrage`` is disabled.

    When ``discount_curve`` is supplied, the curve being built is a *projection*
    curve: quoted instruments are still repriced to par, but their cashflows are
    discounted off ``discount_curve`` (typically an OIS curve) rather than off
    the curve being solved for. This is the second stage of a multi-curve build.
    """
    cal = cal or TargetCalendar()
    if spot_lag_days < 0:
        raise ValueError("spot_lag_days must be non-negative")
    spot_date = cal.workday(value_date, spot_lag_days) if spot_lag_days else value_date
    events: list[tuple[date, str, object]] = []
    events += [(q.maturity, "deposit", q) for q in deposits]
    events += [(q.end, "fra", q) for q in fras]
    events += [(q.maturity, "swap", q) for q in swaps]
    events += [(q.maturity, "ois", q) for q in ois_swaps]
    events.sort(key=lambda x: x[0])
    if not events:
        raise ValueError("at least one market quote is required")

    node_dates = [value_date]
    dfs = [1.0]
    errors: list[float] = []

    for _, kind, raw in events:
        if kind == "deposit":
            q = raw
            assert isinstance(q, DepositQuote)
            if q.maturity <= value_date:
                raise ValueError("deposit maturity must be after value_date")
            alpha = year_fraction(value_date, q.maturity, q.basis)
            df = 1.0 / (1.0 + q.rate * alpha)
            _append_node(node_dates, dfs, q.maturity, df)
            errors.append(0.0)
            continue

        if kind == "fra":
            q = raw
            assert isinstance(q, FRAQuote)
            if not (value_date <= q.start < q.end):
                raise ValueError("FRA dates must satisfy value_date <= start < end")
            curve = _curve_from_nodes(value_date, node_dates, dfs, interpolation)
            p_start = 1.0 if q.start == value_date else curve.df(q.start, value_date)
            alpha = year_fraction(q.start, q.end, q.basis)
            df = p_start / (1.0 + q.rate * alpha)
            _append_node(node_dates, dfs, q.end, df)
            errors.append(0.0)
            continue

        if kind == "ois":
            q = raw
            assert isinstance(q, OISSwapQuote)
            if q.maturity <= value_date:
                raise ValueError("OIS swap maturity must be after value_date")
            if discount_curve is not None:
                raise ValueError(
                    "OIS swap quotes build the discount curve itself and cannot be "
                    "bootstrapped against an external discount curve"
                )
            fixed_spec = q.schedule_spec or ScheduleSpec(frequency=q.fixed_frequency)
            float_spec = q.float_schedule_spec or ScheduleSpec(
                frequency=q.float_frequency,
                business_day_convention=fixed_spec.business_day_convention,
                stub=fixed_spec.stub,
                end_of_month=fixed_spec.end_of_month,
                payment_lag_days=fixed_spec.payment_lag_days,
            )
            convention = q.rfr or RfrConvention()
            fixed_schedule = build_schedule(spot_date, q.maturity, fixed_spec, cal)
            float_schedule = build_schedule(spot_date, q.maturity, float_spec, cal)
            terminal_date = max(
                fixed_schedule[-1].payment_date, float_schedule[-1].payment_date
            )
            if terminal_date <= node_dates[-1]:
                raise ValueError("OIS swap quote does not extend the bootstrapped curve")

            def ois_residual(
                df_terminal: float,
                *,
                _terminal: date = terminal_date,
                _quote: OISSwapQuote = q,
                _fixed: list = fixed_schedule,
                _float: list = float_schedule,
                _convention: RfrConvention = convention,
                _fixings: Mapping[date, float] | None = q.past_fixings,
            ) -> float:
                trial = _curve_from_nodes(
                    value_date,
                    [*node_dates, _terminal],
                    [*dfs, df_terminal],
                    interpolation,
                )
                annuity = sum(
                    year_fraction(p.start, p.end, _quote.fixed_basis)
                    * trial.df(p.payment_date, value_date)
                    for p in _fixed
                )
                # The floating leg is compounded explicitly rather than
                # telescoped, so a lookback or lockout changes the answer.
                float_pv = 0.0
                for p in _float:
                    rate = compounded_rate(
                        p,
                        _convention,
                        trial,
                        value_date,
                        cal,
                        past_fixings=dict(_fixings) if _fixings else None,
                    )
                    accrual = year_fraction(
                        p.start, p.end, f"ACT/{_convention.basis_days}"
                    )
                    float_pv += rate * accrual * trial.df(p.payment_date, value_date)
                return _quote.fixed_rate * annuity - float_pv

            lo, hi = 1e-6, 3.0
            f_lo, f_hi = ois_residual(lo), ois_residual(hi)
            if f_lo * f_hi > 0.0:
                raise ValueError("could not bracket the OIS swap discount factor")
            df_terminal = float(brentq(ois_residual, lo, hi, xtol=1e-16, rtol=1e-15))
            # Evaluate the residual before appending: the closure rebuilds the
            # curve from node_dates plus the terminal date, so appending first
            # would duplicate that node.
            err = ois_residual(df_terminal)
            _append_node(node_dates, dfs, terminal_date, df_terminal)
            errors.append(err)
            continue

        q = raw
        assert isinstance(q, SwapQuote)
        if q.maturity <= value_date:
            raise ValueError("swap maturity must be after value_date")
        spec = q.schedule_spec or ScheduleSpec(frequency=q.fixed_frequency)
        if spec.frequency != q.fixed_frequency:
            raise ValueError("fixed_frequency and schedule_spec.frequency must match")
        float_spec = q.float_schedule_spec or ScheduleSpec(
            frequency=q.float_frequency,
            business_day_convention=spec.business_day_convention,
            stub=spec.stub,
            end_of_month=spec.end_of_month,
            payment_lag_days=spec.payment_lag_days,
            fixing_lag_days=spec.fixing_lag_days,
        )
        if float_spec.frequency != q.float_frequency:
            raise ValueError("float_frequency and float_schedule_spec.frequency must match")
        if q.maturity <= spot_date:
            raise ValueError("swap maturity must be after the spot date")
        schedule = build_schedule(spot_date, q.maturity, spec, cal)
        float_schedule = build_schedule(spot_date, q.maturity, float_spec, cal)
        terminal_date = max(schedule[-1].payment_date, float_schedule[-1].payment_date)
        if terminal_date <= node_dates[-1]:
            raise ValueError("swap quote does not extend the bootstrapped curve")

        # Loop variables are bound as defaults so the closure is independent of
        # the enclosing loop state (ruff B023).
        def residual(
            df_terminal: float,
            *,
            _terminal: date = terminal_date,
            _quote: SwapQuote = q,
            _schedule: list = schedule,
            _float_schedule: list = float_schedule,
        ) -> float:
            trial_dates = [*node_dates, _terminal]
            trial_dfs = [*dfs, df_terminal]
            projection = _curve_from_nodes(value_date, trial_dates, trial_dfs, interpolation)
            # Single curve: discount off the curve being solved. Multi-curve:
            # discount off the exogenous (OIS) curve while projecting off it.
            disc = discount_curve or projection
            annuity = sum(
                year_fraction(p.start, p.end, _quote.fixed_basis)
                * disc.df(p.payment_date, value_date)
                for p in _schedule
            )
            if discount_curve is None:
                # Telescoping shortcut is only valid when projection == discounting.
                float_pv = disc.df(_float_schedule[0].start, value_date) - disc.df(
                    _float_schedule[-1].payment_date, value_date
                )
            else:
                float_pv = 0.0
                for p in _float_schedule:
                    af = year_fraction(p.start, p.end, "ACT/360")
                    fwd = (
                        projection.df(p.start, value_date) / projection.df(p.end, value_date)
                        - 1.0
                    ) / af
                    float_pv += fwd * af * disc.df(p.payment_date, value_date)
            return _quote.fixed_rate * annuity - float_pv

        lo, hi = 1e-6, 3.0
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0.0:
            raise ValueError("could not bracket swap bootstrap root")
        df_terminal = brentq(residual, lo, hi, xtol=1e-14, rtol=1e-13)
        err = residual(df_terminal)
        _append_node(node_dates, dfs, terminal_date, df_terminal)
        errors.append(err)

    curve = _curve_from_nodes(value_date, node_dates, dfs, interpolation)
    days = [(d - value_date).days for d in node_dates]
    checks = (check_discount_factors(days, dfs),)
    if check_arbitrage:
        for result in checks:
            result.raise_if_failed()
    return BootstrapResult(
        curve, tuple(node_dates), tuple(dfs), tuple(errors), checks
    )


@dataclass(frozen=True)
class MultiCurveResult:
    """Result of a two-stage OIS-discounted multi-curve build."""

    curves: CurveSet
    discount: BootstrapResult
    projection: dict[str, BootstrapResult]

    def basis_bp(self, index: str, tenor_days: int) -> float:
        """Projection-minus-discount zero-rate basis in basis points."""
        projection_curve = self.curves.project(index)
        return (projection_curve.zero(tenor_days) - self.curves.discount.zero(tenor_days)) * 1e4


def bootstrap_multi_curve(
    value_date: date,
    ois_deposits: Sequence[DepositQuote] = (),
    ois_swaps: Sequence[SwapQuote] = (),
    projection_quotes: Mapping[str, Sequence[SwapQuote]] | None = None,
    projection_deposits: Mapping[str, Sequence[DepositQuote]] | None = None,
    projection_fras: Mapping[str, Sequence[FRAQuote]] | None = None,
    *,
    cal: TargetCalendar | None = None,
    interpolation: str = "log_linear_df",
    spot_lag_days: int = 0,
    check_arbitrage: bool = True,
) -> MultiCurveResult:
    """Build an OIS discount curve and one projection curve per index.

    Stage one bootstraps the discount (OIS) curve from its own quotes. Stage two
    bootstraps each index projection curve from par swaps on that index, pricing
    them with OIS discounting. This is the standard post-2008 construction: the
    discount curve is shared, and each tenor basis lives in its own projection
    curve.
    """
    cal = cal or TargetCalendar()
    discount_result = bootstrap_single_curve(
        value_date,
        deposits=ois_deposits,
        swaps=ois_swaps,
        cal=cal,
        interpolation=interpolation,
        spot_lag_days=spot_lag_days,
        check_arbitrage=check_arbitrage,
    )

    projection_results: dict[str, BootstrapResult] = {}
    projection_curves: dict[str, DiscountCurve] = {}
    for index, swaps in (projection_quotes or {}).items():
        result = bootstrap_single_curve(
            value_date,
            deposits=(projection_deposits or {}).get(index, ()),
            fras=(projection_fras or {}).get(index, ()),
            swaps=swaps,
            cal=cal,
            interpolation=interpolation,
            spot_lag_days=spot_lag_days,
            check_arbitrage=check_arbitrage,
            discount_curve=discount_result.curve,
        )
        projection_results[index] = result
        projection_curves[index] = result.curve

    return MultiCurveResult(
        CurveSet(discount_result.curve, projection_curves),
        discount_result,
        projection_results,
    )
