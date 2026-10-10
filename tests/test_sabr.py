"""SABR and swaption-cube tests."""

import numpy as np
import pytest

from fi_pricing.sabr import (
    SABRParams,
    SwaptionVolCube,
    calibrate_sabr_cube,
    calibrate_sabr_smile,
    sabr_lognormal_vol,
)


def test_sabr_smile_calibration_recovers_synthetic_slice() -> None:
    truth = SABRParams(0.022, 0.5, -0.35, 0.55, 0.02)
    forward = 0.018
    strikes = np.linspace(-0.005, 0.045, 9)
    vols = [sabr_lognormal_vol(forward, float(k), 4.0, truth) for k in strikes]
    result = calibrate_sabr_smile(
        forward,
        4.0,
        strikes,
        vols,
        beta=truth.beta,
        shift=truth.shift,
    )
    assert result.success
    assert result.rmse_vol < 1e-8
    assert result.params.alpha == pytest.approx(truth.alpha, rel=1e-4)
    assert result.params.rho == pytest.approx(truth.rho, rel=1e-4)
    assert result.params.nu == pytest.approx(truth.nu, rel=1e-4)


def test_sabr_rejects_insufficient_shift() -> None:
    with pytest.raises(ValueError, match="shift greater"):
        sabr_lognormal_vol(
            -0.005,
            0.01,
            2.0,
            SABRParams(0.02, 0.5, 0.0, 0.3, 0.001),
        )


def test_swaption_cube_interpolation_and_sabr_diagnostics() -> None:
    expiries = (1.0, 5.0)
    tenors = (2.0, 10.0)
    strikes = (-0.005, 0.005, 0.015, 0.025, 0.035)
    forwards = np.array([[0.010, 0.015], [0.012, 0.018]])
    grid = []
    for i, expiry in enumerate(expiries):
        slab = []
        for j, _tenor in enumerate(tenors):
            params = SABRParams(
                0.020 + 0.002 * i + 0.001 * j,
                0.5,
                -0.2,
                0.40,
                0.02,
            )
            slab.append(
                [
                    sabr_lognormal_vol(
                        float(forwards[i, j]), strike, expiry, params
                    )
                    for strike in strikes
                ]
            )
        grid.append(slab)
    cube = SwaptionVolCube(expiries, tenors, strikes, grid)
    assert cube.vol(3.0, 6.0, 0.015) > 0.0
    calibrated = calibrate_sabr_cube(cube, forwards, beta=0.5, shift=0.02)
    assert calibrated.rmse_vol < 1e-7
    assert calibrated.max_abs_vol_error < 1e-6
