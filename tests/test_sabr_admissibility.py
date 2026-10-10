"""Tests for SABR smile admissibility and explicit cube expiry interpolation."""

import pytest

from fi_pricing.arbitrage_checks import ArbitrageError
from fi_pricing.sabr import (
    SABRParams,
    SwaptionVolCube,
    calibrate_sabr_cube,
    calibrate_sabr_smile,
    sabr_lognormal_vol,
    sabr_smile_arbitrage_check,
)
from fi_pricing.volatility import (
    EXPIRY_INTERPOLATIONS,
    VolSurface2D,
    interpolate_across_expiry,
)

FORWARD = 0.03
SHIFT = 0.02
STRIKES = [0.005, 0.015, 0.03, 0.05, 0.08]


def _smile(params: SABRParams, expiry: float) -> list[float]:
    return [sabr_lognormal_vol(FORWARD, k, expiry, params) for k in STRIKES]


# ---------------------------------------------------------------------------
# Hagan breakdown is an error, not a negative volatility
# ---------------------------------------------------------------------------
def test_hagan_breakdown_raises_instead_of_returning_a_negative_vol() -> None:
    """The (1 + correction*T) term is a truncated expansion, not a bound."""
    broken = SABRParams(alpha=0.30, beta=0.5, rho=-0.85, nu=1.20, shift=SHIFT)
    with pytest.raises(ValueError, match="broke down"):
        sabr_lognormal_vol(FORWARD, 0.001, 15.0, broken)


def test_ordinary_parameters_are_unaffected_by_the_guard() -> None:
    params = SABRParams(alpha=0.056, beta=0.5, rho=-0.30, nu=0.40, shift=SHIFT)
    for strike in STRIKES:
        assert sabr_lognormal_vol(FORWARD, strike, 5.0, params) > 0.0


def test_atm_branch_is_continuous_across_its_threshold() -> None:
    params = SABRParams(alpha=0.056, beta=0.5, rho=-0.30, nu=0.40, shift=SHIFT)
    near = sabr_lognormal_vol(FORWARD, FORWARD + 1e-9, 5.0, params)
    at = sabr_lognormal_vol(FORWARD, FORWARD, 5.0, params)
    assert near == pytest.approx(at, rel=1e-7)


# ---------------------------------------------------------------------------
# Butterfly admissibility of the fitted smile
# ---------------------------------------------------------------------------
def test_a_benign_smile_is_reported_arbitrage_free() -> None:
    params = SABRParams(0.056, 0.5, -0.30, 0.40, SHIFT)
    result = calibrate_sabr_smile(
        FORWARD, 5.0, STRIKES, _smile(params, 5.0), beta=0.5, shift=SHIFT
    )
    assert result.rmse_vol < 1e-10
    assert result.arbitrage_free
    assert result.arbitrage is not None and result.arbitrage.passed


@pytest.mark.parametrize(
    ("rho", "nu", "expiry"),
    [(-0.50, 0.60, 10.0), (-0.70, 0.90, 10.0)],
)
def test_a_perfectly_fitted_smile_can_still_be_arbitrageable(
    rho: float, nu: float, expiry: float
) -> None:
    """Fit quality and admissibility are different questions.

    These are ordinary EUR-looking parameters, the fit is exact to machine
    precision, and the resulting smile still prices concave call payoffs inside
    the quoted strike range. That is Hagan's wing behaviour, not a bad fit,
    which is why the diagnostic is reported rather than raised.
    """
    params = SABRParams(0.056, 0.5, rho, nu, SHIFT)
    result = calibrate_sabr_smile(
        FORWARD, expiry, STRIKES, _smile(params, expiry), beta=0.5, shift=SHIFT
    )
    assert result.rmse_vol < 1e-10
    assert not result.arbitrage_free
    assert result.arbitrage is not None and result.arbitrage.violations


def test_strict_mode_turns_the_diagnostic_into_an_error() -> None:
    params = SABRParams(0.056, 0.5, -0.70, 0.90, SHIFT)
    vols = _smile(params, 10.0)
    with pytest.raises(ArbitrageError, match="butterfly"):
        calibrate_sabr_smile(
            FORWARD, 10.0, STRIKES, vols, beta=0.5, shift=SHIFT, strict_arbitrage=True
        )
    # The same fit succeeds when the check is not enforced.
    ok = calibrate_sabr_smile(FORWARD, 10.0, STRIKES, vols, beta=0.5, shift=SHIFT)
    assert ok.success


def test_the_check_can_be_switched_off_entirely() -> None:
    params = SABRParams(0.056, 0.5, -0.70, 0.90, SHIFT)
    result = calibrate_sabr_smile(
        FORWARD, 10.0, STRIKES, _smile(params, 10.0),
        beta=0.5, shift=SHIFT, check_arbitrage=False,
    )
    assert result.arbitrage is None
    assert result.arbitrage_free  # nothing was checked, so nothing is claimed


def test_arbitrage_check_stays_inside_the_quoted_strike_range() -> None:
    params = SABRParams(0.056, 0.5, -0.30, 0.40, SHIFT)
    check = sabr_smile_arbitrage_check(params, FORWARD, 5.0, STRIKES, refinement=6)
    assert check.name.startswith("butterfly")
    with pytest.raises(ValueError, match="at least three"):
        sabr_smile_arbitrage_check(params, FORWARD, 5.0, [0.02, 0.03])


def test_cube_calibration_reports_which_slices_are_inadmissible() -> None:
    expiries = [1.0, 10.0]
    tenors = [5.0, 10.0]
    benign = SABRParams(0.056, 0.5, -0.30, 0.40, SHIFT)
    nasty = SABRParams(0.056, 0.5, -0.70, 0.90, SHIFT)
    vols = [
        [_smile(benign, 1.0), _smile(benign, 1.0)],
        [_smile(nasty, 10.0), _smile(nasty, 10.0)],
    ]
    cube = SwaptionVolCube(expiries, tenors, STRIKES, vols)
    forwards = [[FORWARD, FORWARD], [FORWARD, FORWARD]]
    result = calibrate_sabr_cube(cube, forwards, beta=0.5, shift=SHIFT)
    assert not result.arbitrage_free
    assert (1, 0) in result.arbitrage_violations
    assert (0, 0) not in result.arbitrage_violations


# ---------------------------------------------------------------------------
# Explicit expiry interpolation on the cube
# ---------------------------------------------------------------------------
def _flat_cube(mode: str) -> SwaptionVolCube:
    expiries = [1.0, 2.0, 5.0]
    tenors = [2.0, 5.0]
    strikes = [0.02, 0.03, 0.04]
    vols = [
        [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
        [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
        [[0.20, 0.20, 0.20], [0.20, 0.20, 0.20]],
    ]
    return SwaptionVolCube(expiries, tenors, strikes, vols, expiry_interpolation=mode)


def test_cube_reproduces_its_nodes_under_every_rule() -> None:
    for mode in EXPIRY_INTERPOLATIONS:
        cube = _flat_cube(mode)
        assert cube.expiry_interpolation == mode
        for expiry, expected in ((1.0, 0.25), (2.0, 0.25), (5.0, 0.20)):
            assert cube.vol(expiry, 2.0, 0.03) == pytest.approx(expected)


def test_the_rules_disagree_between_nodes() -> None:
    linear = _flat_cube("linear_vol").vol(3.5, 2.0, 0.03)
    variance = _flat_cube("linear_total_variance").vol(3.5, 2.0, 0.03)
    assert linear == pytest.approx(0.225)
    assert variance != pytest.approx(linear, abs=1e-6)


def test_unknown_expiry_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="expiry_interpolation must be one of"):
        _flat_cube("cubic")


def test_cube_and_surface_share_one_implementation() -> None:
    """The lesson learned on VolSurface2D must not be re-learnable on the cube."""
    expiries = [1.0, 2.0, 5.0]
    strikes = [0.02, 0.03, 0.04]
    rows = [[0.25] * 3, [0.25] * 3, [0.20] * 3]
    for mode in EXPIRY_INTERPOLATIONS:
        surface = VolSurface2D(expiries, strikes, rows, expiry_interpolation=mode)
        cube = _flat_cube(mode)
        assert surface.vol(3.5, 0.03) == pytest.approx(cube.vol(3.5, 2.0, 0.03))
        column = [row[1] for row in rows]
        assert interpolate_across_expiry(mode, 3.5, expiries, column) == pytest.approx(
            cube.vol(3.5, 2.0, 0.03)
        )


def test_cube_calendar_arbitrage_check_is_available() -> None:
    assert _flat_cube("linear_vol").calendar_arbitrage_check(2.0).passed
    falling = SwaptionVolCube(
        [1.0, 4.0], [2.0, 5.0], [0.02, 0.03, 0.04],
        [
            [[0.40, 0.40, 0.40], [0.40, 0.40, 0.40]],
            [[0.10, 0.10, 0.10], [0.10, 0.10, 0.10]],
        ],
    )
    assert not falling.calendar_arbitrage_check(2.0).passed
