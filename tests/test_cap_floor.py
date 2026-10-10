from datetime import date

import pytest

from fi_pricing.cap_floor import price_cap, vol_cube_bilinear


def test_synthetic_cap_regression_values() -> None:
    res = price_cap(
        value_date=date(2025, 1, 2),
        effective=date(2024, 10, 2),
        maturity=date(2030, 10, 2),
        notional=1_000_000,
        strike=0.0325,
        frequency=2,
        first_fixing=0.0290,
        style="CAP",
    )

    assert len(res.caplets) == 12
    assert res.vol_used == pytest.approx(0.2035755522, abs=1e-10)
    assert res.npv == pytest.approx(21_187.6191, abs=1e-4)
    assert res.delta == pytest.approx(2_613_293.7116, abs=1e-4)
    assert res.vega == pytest.approx(96_797.5955, abs=1e-4)
    # Theta is split into option decay plus discount-factor roll.
    assert res.option_theta == pytest.approx(-4_218.6210, abs=1e-4)
    assert res.theta == pytest.approx(res.option_theta + res.discount_theta, abs=1e-9)
    assert res.discount_theta > 0.0
    assert res.vanna != 0.0
    assert res.volga != 0.0


def test_flat_vol_mode_is_supported() -> None:
    res = price_cap(
        value_date=date(2025, 1, 2),
        effective=date(2024, 10, 2),
        maturity=date(2030, 10, 2),
        notional=1_000_000,
        strike=0.0325,
        frequency=2,
        first_fixing=0.0290,
        style="CAP",
        flat_vol=0.20,
    )

    assert res.vol_used == pytest.approx(0.20)
    assert res.npv > 0.0


def test_vol_cube_bilinear_returns_decimal_vol() -> None:
    assert vol_cube_bilinear(5.0, 0.030) == pytest.approx(0.20508130996)
