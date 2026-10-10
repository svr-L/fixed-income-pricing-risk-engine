"""Compatibility facade for the public fixed-income primitives.

The implementation lives in focused modules:
- :mod:`fi_pricing.conventions`
- :mod:`fi_pricing.term_structure`
- :mod:`fi_pricing.options`
"""

from .conventions import (
    CouponPeriod,
    ScheduleSpec,
    TargetCalendar,
    add_months,
    build_schedule,
    easter_sunday,
    generate_schedule,
    remaining_periods,
    target_holidays,
    year_fraction,
)
from .options import (
    bachelier_forward_premium,
    bachelier_greeks,
    black76_forward_premium,
    black76_greeks,
    black76_taylor_pnl,
    forward_option_greeks,
    forward_option_premium,
    norm_cdf,
    norm_pdf,
    shifted_black_forward_premium,
    shifted_black_greeks,
)
from .term_structure import DiscountCurve

__all__ = [
    "CouponPeriod",
    "DiscountCurve",
    "ScheduleSpec",
    "TargetCalendar",
    "add_months",
    "bachelier_forward_premium",
    "bachelier_greeks",
    "black76_forward_premium",
    "black76_greeks",
    "black76_taylor_pnl",
    "build_schedule",
    "easter_sunday",
    "forward_option_greeks",
    "forward_option_premium",
    "generate_schedule",
    "norm_cdf",
    "norm_pdf",
    "remaining_periods",
    "shifted_black_forward_premium",
    "shifted_black_greeks",
    "target_holidays",
    "year_fraction",
]
