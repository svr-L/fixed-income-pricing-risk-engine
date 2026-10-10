from datetime import date

import pytest
from scipy.optimize import brentq

from fi_pricing import core as fc
from fi_pricing.arbitrage_checks import ArbitrageError
from fi_pricing.cap_floor import price_cap
from fi_pricing.volatility import (
    VolSurface2D,
    implied_vol_bachelier,
    implied_vol_black76,
    implied_vol_shifted_black,
    strip_caplet_vol_surface,
)


def test_implied_vol_round_trip_black_bachelier_and_shifted_black() -> None:
    black_price = fc.forward_option_premium(0.035, 0.03, 2.0, 0.22, 1, model="black")
    assert implied_vol_black76(black_price, 0.035, 0.03, 2.0, 1) == pytest.approx(0.22, rel=1e-10)

    normal_price = fc.forward_option_premium(-0.005, 0.0, 1.5, 0.012, 1, model="bachelier")
    assert implied_vol_bachelier(normal_price, -0.005, 0.0, 1.5, 1) == pytest.approx(0.012, rel=1e-10)

    shifted_price = fc.forward_option_premium(
        -0.005, 0.0, 1.5, 0.30, 1, model="shifted_black", shift=0.02
    )
    assert implied_vol_shifted_black(shifted_price, -0.005, 0.0, 1.5, 1, 0.02) == pytest.approx(
        0.30, rel=1e-10
    )


def test_bachelier_cap_handles_negative_rates() -> None:
    value_date = date(2025, 1, 2)
    curve = fc.DiscountCurve([0, 3650], [-0.005, -0.005], value_date)
    result = price_cap(
        value_date=value_date,
        effective=date(2025, 7, 2),
        maturity=date(2028, 7, 2),
        notional=1_000_000,
        strike=-0.002,
        frequency=2,
        flat_vol=0.01,
        vol_model="bachelier",
        curve=curve,
    )
    assert result.npv > 0.0
    assert result.vega > 0.0
    with pytest.raises(ValueError, match="Bachelier or shifted Black"):
        price_cap(
            value_date=value_date,
            effective=date(2025, 7, 2),
            maturity=date(2028, 7, 2),
            notional=1_000_000,
            strike=-0.002,
            frequency=2,
            flat_vol=0.20,
            vol_model="black",
            curve=curve,
        )


def test_caplet_vol_stripping_reconstructs_cap_quotes() -> None:
    value_date = date(2025, 1, 2)
    effective = date(2025, 7, 2)
    maturities = [date(2026, 7, 2), date(2027, 7, 2), date(2028, 7, 2)]
    strikes = [0.02, 0.03, 0.04]
    curve = fc.DiscountCurve([0, 3650], [0.03, 0.03], value_date, interpolation="log_linear_df")
    spec = fc.ScheduleSpec(2, business_day_convention="unadjusted", stub="none")
    periods = fc.build_schedule(effective, maturities[-1], spec)
    expiries = [(p.fixing_date - value_date).days / 365.0 for p in periods]

    # Piecewise-constant forward caplet vols by annual cap-maturity block.
    rows = []
    for i, _ in enumerate(periods):
        block = i // 2
        # Levels are chosen so that total variance is non-decreasing in expiry:
        # a naive declining term structure would be calendar-arbitrageable.
        level = [0.20, 0.23, 0.215][block]
        rows.append([level + 0.02, level, level - 0.01])
    true_surface = VolSurface2D(expiries, strikes, rows)

    cap_vol_grid = []
    for maturity in maturities:
        row = []
        for strike in strikes:
            target = price_cap(
                value_date,
                effective,
                maturity,
                1_000_000,
                strike,
                2,
                caplet_vol_surface=true_surface,
                curve=curve,
                schedule_spec=spec,
            ).npv

            def residual(
                vol: float,
                *,
                _maturity: date = maturity,
                _strike: float = strike,
                _target: float = target,
            ) -> float:
                return price_cap(
                    value_date,
                    effective,
                    _maturity,
                    1_000_000,
                    _strike,
                    2,
                    flat_vol=vol,
                    curve=curve,
                    schedule_spec=spec,
                ).npv - _target

            row.append(brentq(residual, 1e-8, 1.0))
        cap_vol_grid.append(row)

    stripped = strip_caplet_vol_surface(
        value_date,
        effective,
        maturities,
        strikes,
        cap_vol_grid,
        2,
        curve,
        schedule_spec=spec,
    )

    for expiry in expiries:
        for strike in strikes:
            assert stripped.vol(expiry, strike) == pytest.approx(
                true_surface.vol(expiry, strike), abs=2e-9
            )

    # Market-facing check: stripped caplet vols reproduce every quoted cap price.
    for i, maturity in enumerate(maturities):
        for j, strike in enumerate(strikes):
            quoted = price_cap(
                value_date,
                effective,
                maturity,
                1_000_000,
                strike,
                2,
                flat_vol=cap_vol_grid[i][j],
                curve=curve,
                schedule_spec=spec,
            ).npv
            rebuilt = price_cap(
                value_date,
                effective,
                maturity,
                1_000_000,
                strike,
                2,
                caplet_vol_surface=stripped,
                curve=curve,
                schedule_spec=spec,
            ).npv
            assert rebuilt == pytest.approx(quoted, abs=2e-7)


def test_stripping_rejects_calendar_arbitrageable_cap_quotes() -> None:
    """A declining vol term structure can imply falling total variance.

    The stripper must reject it rather than return an inadmissible surface.
    """
    value_date = date(2025, 1, 2)
    effective = date(2025, 7, 2)
    maturities = [date(2026, 7, 2), date(2027, 7, 2), date(2028, 7, 2)]
    strikes = [0.02, 0.03, 0.04]
    curve = fc.DiscountCurve([0, 3650], [0.03, 0.03], value_date, interpolation="log_linear_df")
    spec = fc.ScheduleSpec(2, business_day_convention="unadjusted", stub="none")
    periods = fc.build_schedule(effective, maturities[-1], spec)
    expiries = [(p.fixing_date - value_date).days / 365.0 for p in periods]

    rows = []
    for i, _ in enumerate(periods):
        level = [0.20, 0.23, 0.19][i // 2]
        rows.append([level + 0.02, level, level - 0.01])
    bad_surface = VolSurface2D(expiries, strikes, rows)
    assert not bad_surface.calendar_arbitrage_check().passed

    cap_vol_grid = []
    for maturity in maturities:
        row = []
        for strike in strikes:
            target = price_cap(
                value_date, effective, maturity, 1_000_000, strike, 2,
                caplet_vol_surface=bad_surface, curve=curve, schedule_spec=spec,
            ).npv

            def residual(
                vol: float,
                *,
                _maturity: date = maturity,
                _strike: float = strike,
                _target: float = target,
            ) -> float:
                return price_cap(
                    value_date, effective, _maturity, 1_000_000, _strike, 2,
                    flat_vol=vol, curve=curve, schedule_spec=spec,
                ).npv - _target

            row.append(brentq(residual, 1e-8, 1.0))
        cap_vol_grid.append(row)

    with pytest.raises(ArbitrageError, match="total variance"):
        strip_caplet_vol_surface(
            value_date, effective, maturities, strikes, cap_vol_grid, 2, curve,
            schedule_spec=spec,
        )

    # The same quotes strip fine when the admissibility check is opted out of.
    permissive = strip_caplet_vol_surface(
        value_date, effective, maturities, strikes, cap_vol_grid, 2, curve,
        schedule_spec=spec, check_arbitrage=False,
    )
    assert not permissive.calendar_arbitrage_check().passed
