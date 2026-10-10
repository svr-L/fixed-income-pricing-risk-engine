from datetime import date

import pytest

from fi_pricing.irs import Leg, price_irs


def test_synthetic_irs_regression_values() -> None:
    value_date = date(2025, 1, 2)
    fixed = Leg(
        notional=1_000_000,
        effective=date(2024, 7, 2),
        maturity=date(2030, 7, 2),
        frequency=1,
        basis="30/360",
        is_fixed=True,
        coupon=0.0325,
    )
    floating = Leg(
        notional=1_000_000,
        effective=date(2024, 7, 2),
        maturity=date(2030, 7, 2),
        frequency=2,
        basis="ACT/360",
        is_fixed=False,
        current_fixing=0.0290,
    )

    res = price_irs(date(2024, 12, 30), value_date, fixed, floating)

    assert res.fixed.pv == pytest.approx(177_779.8119, abs=1e-4)
    assert res.floating.pv == pytest.approx(164_310.0973, abs=1e-4)
    assert res.fixed.annuity == pytest.approx(5_470_148.0582, abs=1e-4)
    assert res.par_coupon == pytest.approx(0.0300375960, abs=1e-10)
    assert res.receiver_pv == pytest.approx(13_469.7146, abs=1e-4)
    assert res.payer_pv == pytest.approx(-13_469.7146, abs=1e-4)
    assert res.dv01 == pytest.approx(462.1292, abs=1e-4)
    assert abs(res.curve_convexity) > 0.0
    assert fixed.coupon * res.fixed.annuity == pytest.approx(res.fixed.pv, abs=1e-8)
