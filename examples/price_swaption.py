from datetime import date

from fi_pricing.swaption import price_swaption


def main() -> None:
    res = price_swaption(
        value_date=date(2025, 1, 2), expiry=date(2027, 1, 4),
        swap_start=date(2027, 4, 2), swap_end=date(2032, 4, 2),
        strike=0.0325, vol=0.20, style="Receiver", notional=1_000_000, shift_bp=0.0,
    )
    print("Receiver swaption synthetic example")
    print(f"Forward swap rate  : {res.forward_rate * 100:.6f}%")
    print(f"Annuity            : {res.annuity:,.2f}")
    print(f"Premium            : {res.premium:,.2f}")
    print(f"Delta              : {res.delta:,.2f}")
    print(f"Vega/Gamma/Theta   : {res.vega:,.2f} / {res.gamma:,.2e} / {res.theta:,.2f}")
    print(f"Vanna / Volga      : {res.vanna:,.2f} / {res.volga:,.2f}")


if __name__ == "__main__":
    main()
