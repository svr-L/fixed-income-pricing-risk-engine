# Market conventions and schedule safety

`ScheduleSpec` makes contract conventions explicit:

- coupon frequency;
- Following / Modified Following / Preceding / unadjusted rolls;
- short/long front or back stubs;
- end-of-month generation;
- payment lag;
- fixing lag.

`CouponPeriod` keeps unadjusted accrual boundaries, adjusted accrual dates, fixing dates
and payment dates separate. Schedules never extend outside the contractual effective and
maturity dates.

Supported day counts include ACT/360, ACT/365F, US 30/360, 30E/360 and ACT/ACT ISDA.
Accrued interest uses the leg's own day-count convention.

Floating coupons that have already fixed but remain unpaid require historical fixings.
Fixings that do not correspond to an applicable period are rejected rather than silently
ignored.

Black-76 enforces its positive forward/strike domain. Negative-rate instruments should
use Bachelier or an adequate shifted-lognormal specification.

Cash-settled swaptions use an explicit frozen-forward IRR-annuity approximation; the
result object exposes the settlement method because the true cash annuity is stochastic.
Amortising legs/caps accept outstanding notionals keyed by accrual-period start date.
