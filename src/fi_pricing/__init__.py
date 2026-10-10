"""Fixed-income pricing and quantitative risk analytics."""

from .arbitrage_checks import (
    ArbitrageError,
    CheckResult,
    check_butterfly_convexity,
    check_discount_factors,
    check_total_variance_monotone,
)
from .bootstrap import (
    BootstrapResult,
    DepositQuote,
    FRAQuote,
    MultiCurveResult,
    OISSwapQuote,
    SwapQuote,
    bootstrap_multi_curve,
    bootstrap_single_curve,
)
from .cap_floor import CapResult, Caplet, default_vol_surface, price_cap, vol_cube_bilinear
from .core import (
    DiscountCurve,
    ScheduleSpec,
    TargetCalendar,
    bachelier_forward_premium,
    bachelier_greeks,
    black76_forward_premium,
    black76_greeks,
    black76_taylor_pnl,
    build_schedule,
    forward_option_greeks,
    forward_option_premium,
    shifted_black_forward_premium,
    shifted_black_greeks,
)
from .curves import CurveSet, as_curve_set
from .hull_white import (
    CalibrationInstrument,
    CalibrationResult,
    HullWhiteParams,
    calibrate_hull_white,
    caplet_instruments_from_surface,
    caplet_price,
    jamshidian_strikes,
    swaption_from_schedule,
    swaption_price,
    zero_bond_option,
    zero_bond_price,
)
from .irs import CashFlow, IRSResult, Leg, LegResult, price_irs, price_leg
from .rfr import (
    RfrConvention,
    accrued_compounded_rate,
    backward_looking_variance_ratio,
    compounded_rate,
    observation_schedule,
)
from .risk import (
    KeyRateConvexity,
    MultiCurveRisk,
    bucketed_surface_vega,
    carry_roll_down,
    curve_factor_risk,
    explain_curve_pnl,
    key_rate_convexity_matrix,
    key_rate_risk,
    multi_curve_risk,
    parallel_curve_risk,
    short_rate_parameter_greeks,
)
from .sabr import (
    SABRCalibrationResult,
    SABRCubeCalibration,
    SABRParams,
    SwaptionVolCube,
    calibrate_sabr_cube,
    calibrate_sabr_smile,
    sabr_lognormal_vol,
    sabr_smile_arbitrage_check,
)
from .short_rate import (
    CurveCalibrationResult,
    PhysicalFitResult,
    ShortRateParams,
    calibrate_cir_q,
    calibrate_vasicek_q,
    cir_zero_rate,
    fit_cir_p,
    fit_vasicek_p,
    simulate_cir,
    simulate_vasicek,
    vasicek_zero_rate,
)
from .swaption import SwaptionResult, cash_settled_annuity, price_swaption
from .volatility import (
    CapSurfaceConversionResult,
    VolSurface2D,
    convert_cap_vol_surface,
    implied_vol_bachelier,
    implied_vol_black76,
    implied_vol_shifted_black,
    implied_volatility,
    interpolate_across_expiry,
    linear_interpolate,
    strip_caplet_vol_surface,
    validate_expiry_interpolation,
)

__all__ = [
    "ArbitrageError", "BootstrapResult", "CalibrationInstrument", "CalibrationResult",
    "CapResult", "CapSurfaceConversionResult", "Caplet", "CashFlow", "CheckResult",
    "CurveCalibrationResult", "CurveSet", "DepositQuote", "DiscountCurve", "FRAQuote",
    "HullWhiteParams", "IRSResult", "KeyRateConvexity", "Leg", "LegResult",
    "MultiCurveResult", "MultiCurveRisk", "OISSwapQuote", "PhysicalFitResult",
    "RfrConvention", "SABRCalibrationResult", "SABRCubeCalibration", "SABRParams",
    "ScheduleSpec", "ShortRateParams", "SwapQuote", "SwaptionResult", "SwaptionVolCube",
    "TargetCalendar", "VolSurface2D", "accrued_compounded_rate", "as_curve_set",
    "bachelier_forward_premium", "bachelier_greeks", "backward_looking_variance_ratio",
    "black76_forward_premium", "black76_greeks", "black76_taylor_pnl",
    "bootstrap_multi_curve", "bootstrap_single_curve", "bucketed_surface_vega",
    "build_schedule", "calibrate_cir_q", "calibrate_hull_white", "calibrate_sabr_cube",
    "calibrate_sabr_smile", "calibrate_vasicek_q", "caplet_instruments_from_surface",
    "caplet_price", "carry_roll_down", "cash_settled_annuity", "check_butterfly_convexity",
    "check_discount_factors", "check_total_variance_monotone", "cir_zero_rate",
    "compounded_rate", "convert_cap_vol_surface", "curve_factor_risk", "default_vol_surface",
    "explain_curve_pnl", "fit_cir_p", "fit_vasicek_p", "forward_option_greeks",
    "forward_option_premium", "implied_vol_bachelier", "implied_vol_black76",
    "implied_vol_shifted_black", "implied_volatility", "interpolate_across_expiry",
    "jamshidian_strikes", "key_rate_convexity_matrix", "key_rate_risk",
    "linear_interpolate", "multi_curve_risk", "observation_schedule", "parallel_curve_risk",
    "price_cap", "price_irs", "price_leg", "price_swaption", "sabr_lognormal_vol",
    "sabr_smile_arbitrage_check", "shifted_black_forward_premium", "shifted_black_greeks",
    "short_rate_parameter_greeks", "simulate_cir", "simulate_vasicek",
    "strip_caplet_vol_surface", "swaption_from_schedule", "swaption_price",
    "validate_expiry_interpolation", "vasicek_zero_rate", "vol_cube_bilinear",
    "zero_bond_option", "zero_bond_price",
]
