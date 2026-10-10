from datetime import date

import pytest

from fi_pricing.core import DiscountCurve
from fi_pricing.swaption import price_swaption


def test_synthetic_swaption_regression_values() -> None:
    res = price_swaption(
        value_date=date(2025, 1, 2),
        expiry=date(2027, 1, 4),
        swap_start=date(2027, 4, 2),
        swap_end=date(2032, 4, 2),
        strike=0.0325,
        vol=0.20,
        style="Receiver",
        notional=1_000_000,
        shift_bp=0.0,
    )

    assert res.forward_rate == pytest.approx(0.0343261571, abs=1e-10)
    assert res.annuity == pytest.approx(4_226_384.0183, abs=1e-4)
    assert res.premium == pytest.approx(12_337.4533, abs=1e-4)
    assert res.delta == pytest.approx(-1_559_333.6222, abs=1e-4)
    assert res.vega == pytest.approx(77_499.3034, abs=1e-4)
    assert res.gamma > 0.0
    assert res.vanna != 0.0
    assert res.volga != 0.0
    assert res.d2 < res.d1


def test_bachelier_swaption_supports_negative_forward_rates() -> None:
    value_date = date(2025, 1, 2)
    curve = DiscountCurve([0, 3650], [-0.004, -0.004], value_date)
    res = price_swaption(
        value_date=value_date,
        expiry=date(2026, 1, 2),
        swap_start=date(2026, 4, 2),
        swap_end=date(2030, 4, 2),
        strike=-0.002,
        vol=0.01,
        style="Payer",
        curve=curve,
        vol_model="bachelier",
    )
    assert res.forward_rate < 0.0
    assert res.premium > 0.0
    assert res.vega > 0.0
