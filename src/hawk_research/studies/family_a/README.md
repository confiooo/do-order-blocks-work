# Family A: the pre-registered test (S1, S5)

Stages: `hawk-research run s3r s1s5`. Pre-registration:
[`docs/archive/preregistration.md`](../../../../docs/archive/preregistration.md); results as written at
the time: [`docs/archive/s1_s5_results.md`](../../../../docs/archive/s1_s5_results.md).

**Question.** After round 1 found no edge in the detector's direction, could a more careful use of the
same zones work? Five systems (S1-S5) were specified in advance with frozen parameters, costs, gates
and stop rules. Two were run:

- **S1, level mechanism.** Rest a maker limit at the far edge of the zone; 8h horizon; target 1.0 and
  stop 0.5 times the 8h volatility; at most 5 open positions. Needs "Pass R", a replay with the
  detector's swing filter switched off (`s3r`), so zones near and far from a swing can be compared.
- **S5, zone vs matched control.** S3's signed rule applied identically to zones and to matched
  non-zone bars (up to 3 per zone, exact bins on BTC range, ATR, liquidity tier, time block and
  residual sign); the question is whether being a zone adds anything.

**Result: both FALSIFIED** on the development tier. The holdout tier was never read.

| | Value |
|---|---|
| S1 filled trades | 45,988 |
| S1 mean net per trade | −33.5 bps (gross −1.4 bps) |
| S1 CPCV paths positive | 0 |
| S5 zone arm vs control arm, net | −33.2 vs −40.6 bps per trade |
| S5 zone minus control | +7.4 bps [+1.4, +13.4] |

S5's zone-minus-control difference is positive and its interval excludes zero, but both arms lose
heavily after costs; being a zone makes a losing trade lose a little less. Under the
pre-registration that difference authorised running S3 next. It was not run.

**What changed in this repository.** Annualisation now uses 365.25 days like the rest of the code
(the original used 365): annualised Sharpe ratios scale by 1.00034. Hansen's SPA and Romano-Wolf
StepM (pre-registered Gate 4) were not implemented, because with two of five systems run the test is
undefined; Gate 4 is recorded as failed. See [`docs/deviations.md`](../../../../docs/deviations.md).
