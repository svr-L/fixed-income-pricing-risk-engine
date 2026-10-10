"""Closed-form forward-option models used by rates products."""

from __future__ import annotations

import math

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _validate_common(T: float, vol: float, phi: int) -> None:
    if phi not in {-1, 1}:
        raise ValueError("phi must be +1 or -1")
    if T < 0.0:
        raise ValueError("option expiry T must be non-negative")
    if vol < 0.0:
        raise ValueError("volatility must be non-negative")


def _black_state(F: float, K: float, T: float, vol: float) -> tuple[float, float, float]:
    sd = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / sd
    return d1, d1 - sd, sd


def black76_forward_premium(F: float, K: float, T: float, vol: float, phi: int) -> float:
    """Undiscounted Black-76 option value per unit annuity."""
    _validate_common(T, vol, phi)
    if T == 0.0 or vol == 0.0:
        return max(phi * (F - K), 0.0)
    if F <= 0.0 or K <= 0.0:
        raise ValueError(
            "Black-76 requires positive forward and strike; use Bachelier or shifted Black"
        )
    d1, d2, _ = _black_state(F, K, T, vol)
    return phi * (F * norm_cdf(phi * d1) - K * norm_cdf(phi * d2))


def black76_greeks(F: float, K: float, T: float, vol: float, phi: int) -> dict[str, float]:
    """Delta, Gamma, Vega, calendar Theta, Vanna and Volga under Black-76."""
    _validate_common(T, vol, phi)
    if T == 0.0 or vol == 0.0:
        return {
            "d1": float("nan"),
            "d2": float("nan"),
            "delta": float(phi if phi * (F - K) > 0.0 else 0.0),
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "vanna": 0.0,
            "volga": 0.0,
        }
    if F <= 0.0 or K <= 0.0:
        raise ValueError(
            "Black-76 requires positive forward and strike; use Bachelier or shifted Black"
        )
    d1, d2, sd = _black_state(F, K, T, vol)
    density = norm_pdf(d1)
    sqrt_t = math.sqrt(T)
    vega = F * density * sqrt_t
    return {
        "d1": d1,
        "d2": d2,
        "delta": phi * norm_cdf(phi * d1),
        "gamma": density / (F * sd),
        "vega": vega,
        "theta": -0.5 * F * density * vol / sqrt_t,
        "vanna": -density * d2 / vol,
        "volga": vega * d1 * d2 / vol,
    }


def shifted_black_forward_premium(
    F: float,
    K: float,
    T: float,
    vol: float,
    phi: int,
    shift: float,
) -> float:
    _validate_common(T, vol, phi)
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    shifted_f = F + shift
    shifted_k = K + shift
    if T > 0.0 and vol > 0.0 and (shifted_f <= 0.0 or shifted_k <= 0.0):
        minimum = max(-F, -K)
        raise ValueError(
            "shifted Black requires F+shift>0 and K+shift>0; "
            f"increase shift above {minimum:.6f} or use the Bachelier model"
        )
    return black76_forward_premium(shifted_f, shifted_k, T, vol, phi)


def shifted_black_greeks(
    F: float,
    K: float,
    T: float,
    vol: float,
    phi: int,
    shift: float,
) -> dict[str, float]:
    _validate_common(T, vol, phi)
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    shifted_f = F + shift
    shifted_k = K + shift
    if T > 0.0 and vol > 0.0 and (shifted_f <= 0.0 or shifted_k <= 0.0):
        minimum = max(-F, -K)
        raise ValueError(
            "shifted Black requires F+shift>0 and K+shift>0; "
            f"increase shift above {minimum:.6f} or use the Bachelier model"
        )
    return black76_greeks(shifted_f, shifted_k, T, vol, phi)


def bachelier_forward_premium(F: float, K: float, T: float, vol: float, phi: int) -> float:
    """Undiscounted Bachelier value per unit annuity for normal volatility."""
    _validate_common(T, vol, phi)
    if T == 0.0 or vol == 0.0:
        return max(phi * (F - K), 0.0)
    sd = vol * math.sqrt(T)
    d = (F - K) / sd
    return phi * (F - K) * norm_cdf(phi * d) + sd * norm_pdf(d)


def bachelier_greeks(F: float, K: float, T: float, vol: float, phi: int) -> dict[str, float]:
    _validate_common(T, vol, phi)
    if T == 0.0 or vol == 0.0:
        return {
            "d1": float("nan"),
            "d2": float("nan"),
            "delta": float(phi if phi * (F - K) > 0.0 else 0.0),
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "vanna": 0.0,
            "volga": 0.0,
        }
    sqrt_t = math.sqrt(T)
    sd = vol * sqrt_t
    d = (F - K) / sd
    density = norm_pdf(d)
    vega = sqrt_t * density
    return {
        "d1": d,
        "d2": float("nan"),
        "delta": phi * norm_cdf(phi * d),
        "gamma": density / sd,
        "vega": vega,
        "theta": -0.5 * vol * density / sqrt_t,
        "vanna": -density * d / vol,
        "volga": vega * d * d / vol,
    }


def forward_option_premium(
    F: float,
    K: float,
    T: float,
    vol: float,
    phi: int,
    model: str = "black",
    shift: float = 0.0,
) -> float:
    key = model.strip().lower().replace("-", "_")
    if key in {"black", "black76", "lognormal"}:
        return black76_forward_premium(F, K, T, vol, phi)
    if key in {"shifted_black", "shifted_lognormal"}:
        return shifted_black_forward_premium(F, K, T, vol, phi, shift)
    if key in {"bachelier", "normal"}:
        return bachelier_forward_premium(F, K, T, vol, phi)
    raise ValueError("model must be black, shifted_black or bachelier")


def forward_option_greeks(
    F: float,
    K: float,
    T: float,
    vol: float,
    phi: int,
    model: str = "black",
    shift: float = 0.0,
) -> dict[str, float]:
    key = model.strip().lower().replace("-", "_")
    if key in {"black", "black76", "lognormal"}:
        return black76_greeks(F, K, T, vol, phi)
    if key in {"shifted_black", "shifted_lognormal"}:
        return shifted_black_greeks(F, K, T, vol, phi, shift)
    if key in {"bachelier", "normal"}:
        return bachelier_greeks(F, K, T, vol, phi)
    raise ValueError("model must be black, shifted_black or bachelier")


def black76_taylor_pnl(
    F: float,
    K: float,
    T: float,
    vol: float,
    phi: int,
    dF: float = 0.0,
    dvol: float = 0.0,
    dt: float = 0.0,
) -> dict[str, float]:
    """Second-order local P&L expansion in forward, volatility and time."""
    g = black76_greeks(F, K, T, vol, phi)
    components = {
        "delta": g["delta"] * dF,
        "gamma": 0.5 * g["gamma"] * dF * dF,
        "vega": g["vega"] * dvol,
        "theta": g["theta"] * dt,
        "vanna": g["vanna"] * dF * dvol,
        "volga": 0.5 * g["volga"] * dvol * dvol,
    }
    components["total"] = sum(components.values())
    return components
