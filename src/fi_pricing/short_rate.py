"""Compatibility facade for Vasicek/CIR short-rate models."""
from .models.short_rate import (
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

__all__ = [
    "CurveCalibrationResult",
    "PhysicalFitResult",
    "ShortRateParams",
    "calibrate_cir_q",
    "calibrate_vasicek_q",
    "cir_zero_rate",
    "fit_cir_p",
    "fit_vasicek_p",
    "simulate_cir",
    "simulate_vasicek",
    "vasicek_zero_rate",
]
