from datetime import date

from fi_pricing.cap_floor import price_cap


def main() -> None:
    res = price_cap(
        value_date=date(2025, 1, 2), effective=date(2024, 10, 2),
        maturity=date(2030, 10, 2), notional=1_000_000, strike=0.0325,
        frequency=2, first_fixing=0.0290, style="CAP",
    )
    print("Cap synthetic example")
    print(f"Caplets            : {len(res.caplets)}")
    print(f"Vol used           : {res.vol_used * 100:.6f}%")
    print(f"NPV                : {res.npv:,.2f}")
    print(f"Delta/Gamma/Vega   : {res.delta:,.0f} / {res.gamma:,.2e} / {res.vega:,.0f}")
    print(f"Theta              : {res.theta:,.0f}")
    print(f"Vanna / Volga      : {res.vanna:,.2f} / {res.volga:,.2f}")


if __name__ == "__main__":
    main()
