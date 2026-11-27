# Results — "Asymmetric Reversals"

**Paper:** Federico Baldi-Lanfranchi, Pierre Collin-Dufresne and Kent Daniel (2026),
"Asymmetric Reversals", NBER Working Paper 35840, October 2026,
<http://www.nber.org/papers/w35840>.

All numbers below are read from the files in `results/` written by the final
run of `python -m exp.<name>`. Returns, alphas and slopes are in decimal units
per month unless stated (0.0039 = 0.39%/month); t-statistics use Newey-West
standard errors with automatic lag selection.

**Executed sample.** Daily decomposition January 2011 – December 2022
(3,020 trading days; daily bars loaded from January 2008 for the lookback
windows); monthly predictive tests use formation months
January 2011 – November 2022 (143 months); the impulse responses and the ASRES
regressions start in January 2012 (after the 252-day lag window); ASRES
portfolios are formed December 2012 – November 2022 and held January 2013 –
December 2022 (120 months) after a two-year burn-in. The investable universe
holds 1,084–1,780 stocks per month
(2,479 distinct securities). The paper's sample is January 1972 – December
2022; see "Deviations from the paper".

**Verdict rule.** "reproduced": within the paper's reported range, or within
10% of the paper's value with the same sign, and the paper's qualitative
statement holds; "not reproduced": opposite sign, more than 50% away from the
paper's value, or the paper's qualitative statement (significance, ranking,
monotonicity) fails; otherwise "partially reproduced".

---

## Experiment 1 — Daily Cross-Sectional Return Decomposition and Goodness-of-Fit Evaluation

Daily WLS cross-sectional regressions of excess returns on rank-standardized
characteristics (Section 3, Eq. 1–4) and Fama-French 49 industry dummies, with
weights equal to beginning-of-month market equity winsorized at the 80th NYSE
percentile; returns in the [-1, +1] earnings-announcement window are moved to
the systematic component. Annual adjusted R² (Section 3.1, Figure 1) is
compared with stock-by-stock FF3 time-series regressions estimated within each
month. Files: `exp1_summary.json`, `exp1_adj_r2.png`.

| Metric (average annual adjusted R², WLS) | Paper | Replication | Verdict |
|---|---|---|---|
| Characteristics + industries | ≈20% | 29.1% | partially reproduced |
| Characteristics only | ≈17% | 21.9% | partially reproduced |
| Stock-by-stock FF3 (implied: 20% − 11 pp) | ≈9% | 13.6% | not reproduced |
| Gap char + ind minus FF3 | ≈11 pp | 15.6 pp | partially reproduced |
| Gap char only minus FF3 | ≈8 pp | 8.4 pp | reproduced |
| Peak R² near the end of the sample | ≈40% | 39.9% (2022) | reproduced |
| Ranking char+ind > char > FF3 | holds | holds (also OLS: 20.8% > 16.1% > 11.5%) | reproduced |

The paper's averages cover 1972–2022, during which R² rises over time; the
replication covers only 2011–2022, the late part of the paper's sample where
the paper's Figure 1 shows higher values, so higher levels are expected. The
paper's average for this sub-period is not stated in the text, so the level
difference is not attributed to the sample period.

## Experiment 2 — Short-Term Reversal in Residual Returns vs. Momentum in Systematic Returns

Monthly RES and SYS (sums of the daily components, skipping the last trading
day of the month) sorted into NYSE-breakpoint quintiles, value-weighted with
winsorized market equity (Section 4, Table 1). Files: `exp2_table1_sorts.csv`,
`exp2_summary.json`, `exp2_quintile_returns.png`.

| Metric | Paper | Replication | Verdict |
|---|---|---|---|
| RES High−Low RET_{m+1} | −1.11% (t = −8.51) | −0.39% (t = −3.27) | not reproduced |
| RES High−Low FF6 alpha | −1.08% (t = −7.42) | −0.33% (t = −3.67) | not reproduced |
| SYS High−Low RET_{m+1} | +0.86% (t = 5.35) | +0.25% (t = 0.84) | not reproduced |
| SYS High−Low FF6 alpha | +0.84% (t = 4.39) | +0.40% (t = 1.17) | not reproduced |
| RES quintile returns monotonic (reversal) | monotonic | 1.10, 1.06, 0.93, 0.87, 0.71% — monotonic | reproduced |
| SYS quintile returns (continuation) | monotonic | 0.77, 0.97, 0.97, 1.05, 1.02% — not monotonic | not reproduced |
| SUE median, SYS Low → High | −0.01 → 0.06, monotonic | 0.197, 0.244, 0.267, 0.247, 0.235 — not monotonic | not reproduced |
| SUE median flat across RES quintiles | flat | 0.207, 0.255, 0.275, 0.251, 0.222 — hump-shaped | partially reproduced |

The direction of both effects holds (residual reversal, systematic
continuation), but the magnitudes are about one third of the paper's, and the
SYS spread is not significant over 143 months. For comparison, the
Fama-French short-term reversal factor averages 0.165%/month over 2013–2022
against 0.421%/month over 1974–2022 (ratio 0.39; `exp6_meta.json`), so reversal profits were
weaker in the replication period than in the paper's full sample; this was
checked on the factor itself, not on the paper's RES portfolios for 2011–2022,
which the paper does not report, so the gap is not attributed to the sample
period alone.

## Experiment 3 — Spanning Idiosyncratic Volatility (IVOL) and Mispricing with Asymmetric Idiosyncratic Reversal (AIR)

RES⁺, RES⁻, IVOL and concentration measures (Section 5.1, Eq. 14–15);
Fama-MacBeth regressions of next-month returns (Table 2, WLS with the six
Fama-French controls; OLS and no-control versions also saved), IVOL horizon
dependence (Figure 2), regressions within mispricing quintiles (Table 3,
2011–2016, 65–68 months per quintile) and concavity (Table 4). Files: `exp3_eq15_ivol_on_res.csv`,
`exp3_table2_spanning_ivol.csv`, `exp3_figure2_ivol_horizon.csv/.png`,
`exp3_table3_misp.csv`, `exp3_table4_concavity.csv`, `exp3_summary.json`.

| Metric | Paper | Replication | Verdict |
|---|---|---|---|
| Eq. 15 slope on RES⁺ | +0.078 | +0.077 (t = 99.0) | reproduced |
| Eq. 15 slope on RES⁻ | −0.067 | −0.075 (t = −73.3) | partially reproduced |
| Eq. 15 average adjusted R² | 81.6% | 72.3% | partially reproduced |
| Table 2: IVOL alone | −0.195 (t = −5.28) | −0.049 (t = −0.46) | not reproduced |
| Table 2: RES⁺ (joint with RES⁻) | −0.063 (t = −12.36) | −0.030 (t = −2.62) | not reproduced |
| Table 2: RES⁻ (joint with RES⁺) | −0.035 (t = −4.28) | −0.012 (t = −0.88) | not reproduced |
| Table 2: RES⁺ − RES⁻ asymmetry | t = −3.54 | −0.018 (t = −0.87) | not reproduced |
| Table 2: IVOL with RES⁺, RES⁻ | +0.050 (t = 1.26), insignificant | +0.155 (t = 0.60), insignificant | partially reproduced |
| Figure 2: H = 1, 1-day holding, IVOL t-stat | +10.3 | −0.52 | not reproduced |
| Figure 2: positive for short H at the 1-day horizon, negative at longer horizons | holds | 1-day t-stats are negative for H ≤ 7 (−0.13 to −0.66) and small positive for H ≥ 12 (all \|t\| < 0.7); 5- and 21-day t-stats are negative for all H except 21-day H ≥ 19 (min −1.88) | not reproduced |
| Table 3: IVOL alone, Overpriced quintile | −0.342 (t = −6.11) | −0.147 (t = −0.54) | not reproduced |
| Table 3: IVOL with RES⁺, RES⁻, Overpriced | +0.038 (t = 0.40) | +0.254 (t = 0.38) | partially reproduced |
| Table 3: IVOL insignificant with RES⁺, RES⁻ in all 5 quintiles | holds | t = −1.27, 1.20, −0.27, −2.07, 0.38 (quintile 4 significant) | not reproduced |
| Table 4: CONC⁺ | +0.007 (t = 3.68) | +0.004 (t = 0.71) | not reproduced |
| Table 4: CONC⁻ | +0.013 (t = 5.40) | +0.004 (t = 0.66) | not reproduced |
| Table 4: CONC (with SYS) | +0.010 (t = 6.41) | +0.006 (t = 1.86) | not reproduced |

The cross-sectional link between IVOL and AIR (Eq. 15) is reproduced. In the
predictive regressions the signs agree with the paper (RES⁺ more negative than
RES⁻, CONC positive, IVOL loses its sign once RES⁺ and RES⁻ are added), but
only RES⁺ is significant: over 2011–2022 the standalone IVOL effect is itself
insignificant, so the spanning result cannot be tested with power.

## Experiment 4 — Spanning the Short-Term Anomaly Zoo with AIR, Concentration, and Systematic Returns

Thirteen one-day-skipped short-horizon characteristics (Appendix E.1) regressed
on RES⁺/RES⁻ (Table 6 Panel A) and used in WLS Fama-MacBeth return
regressions with the six controls alone (Panel B) and with RES⁺, RES⁻
(Panel C), + CONC (Panel D), + SYS (Panel E). Files:
`exp4_table6_spanning.csv`, `exp4_summary.json`, `exp4_table6_tstats.png`.

| Metric | Paper | Replication | Verdict |
|---|---|---|---|
| Panel A adjusted R², IVOL^FF3 / Ret Vol / MAX / MAX5 / Lag Ret | 50–70% | 57.1% / 54.1% / 39.3% / 55.3% / 52.8% | partially reproduced (4 of 5 in range) |
| Panel B: IVOL^FF3, Ret Vol, MAX, MAX5, SMAX, Lag Ret, Turnover significantly negative | 7 significant | 0 significant (t: −0.09, 0.24, −1.22, −1.47, −1.72, −0.78, 0.19) | not reproduced |
| Panel C: Skew, ISkew, MAX, MAX5, SMAX, Lag Ret turn significantly positive | 6 significant | 0 significant | not reproduced |
| Panel C: IVOL^FF3, Ret Vol, Coskew, Amihud, Zero Trades, Spread, Turnover insignificant | insignificant | insignificant (max \|t\| = 1.43) | reproduced |
| Panel E: no characteristic significant | 0 of 12 | 0 of 12 | reproduced |
| RES⁺, RES⁻, CONC, SYS significant in all specifications | holds | median t across characteristics in Panel E: RES⁺ −1.39, RES⁻ 0.11, CONC 1.89, SYS −0.22 | not reproduced |

Because no characteristic predicts returns on its own in 2011–2022 (Panel B),
the insignificance in Panels C–E does not demonstrate spanning; the Panel A
fits, which do not depend on return predictability, are close to the paper's.

## Experiment 5 — Daily Impulse-Response Dynamics of Signed Residual and Systematic Shocks

Daily WLS Fama-MacBeth regressions of returns on 252 lags of ε̃⁺, ε̃⁻, r̃⁺,
r̃⁻ (dense, Eq. 19, 1,008 slopes) and on the 27-knot degree-1 spline
projection (Eq. 22–24, 81 slopes); impulse responses compounded as in
Eq. 20–21 and averaged over 2,516 start dates. Files: `exp5_irf_summary.csv`,
`exp5_monthly_aggregation.csv`, `exp5_meta.json`, `exp5_irf.png`,
`exp5_monthly_aggregation.png`. "Reversal" is the largest move of the price
path back toward zero relative to the 1% shock.

| Metric | Paper | Replication (dense / spline) | Verdict |
|---|---|---|---|
| Rebound after −1% ε̃ shock | ≈21 bp | 10.0 bp / 8.5 bp | not reproduced |
| Horizon of the ε̃⁻ rebound | ≈1.5 months (≈32 days) | 48 / 54 days | not reproduced |
| Share of the ε̃⁻ reversal on day 1 | 29% | 13.5% / 12.7% | not reproduced |
| Fall after +1% ε̃ shock | ≈19 bp | 11.0 bp / 7.9 bp | partially reproduced (dense) / not reproduced (spline) |
| Horizon of the ε̃⁺ reversal | ≈2.5 months (≈53 days) | 52 / 46 days | reproduced (dense) / partially reproduced (spline) |
| Share of the ε̃⁺ reversal on day 1 | 4% | 17.8% / 29.2% | not reproduced |
| First-week reversal ε̃⁻ ≈ 2 × ε̃⁺ | ≈2× | 0.94× / 0.52× | not reproduced |
| Systematic short-term reversal | 3–5 bp | r̃⁺ 3.7 bp, r̃⁻ 5.2 bp / 0.3 bp | reproduced (dense) / not reproduced (spline) |
| Systematic response at 252 days, +1% | +1.13% | +0.87% / +1.13% | partially reproduced / reproduced |
| Systematic response at 252 days, −1% | −1.19% | −1.21% / −1.13% | reproduced / reproduced |
| Spline bands narrower than dense bands | holds | 95% band width at τ = 21: ε̃⁺ 3.8 vs 5.2 bp, ε̃⁻ 3.9 vs 4.8 bp, r̃⁺ 4.9 vs 18.1 bp | reproduced |
| Triangular monthly weights peak at lags 21–22 | 21–22 | 21–22 | reproduced |
| Monthly slopes implied by the daily spline: RES⁺ dominates RES⁻ | holds | RES⁺ −0.019, RES⁻ −0.039 | not reproduced |

In the 252-lag spline regressions, which also control for systematic shocks
and longer lags, the day-1 coefficient on ε̃⁺ (−0.022) exceeds that on ε̃⁻
(−0.011) (`exp5_monthly_aggregation.csv`), while the implied monthly slope is
larger in magnitude for RES⁻ (−0.039 vs −0.019, `exp5_meta.json`). The cause
of the slower ε̃⁻ reversal relative to the paper was not identified.

## Experiment 6 — Out-of-Sample Asymmetric Residual Return Factor (ASRES) Construction and Performance Evaluation

Daily WLS spline regressions of 21-day forward excess returns on lags 2–252 of
ε̃⁺, ε̃⁻ and r̃ (Eq. 25), expanding-window averages of the implied lag
coefficients using only regressions whose 21-day return is realized before
formation (Eq. 26–27), ASRES = μ̂²¹ + 0.40 (μ̂^full − μ̂²¹) (Eq. 28), and
NYSE-breakpoint quintile long-short factors (Tables 7–9). Files:
`exp6_table7_factors.csv`, `exp6_table8_loadings.csv`,
`exp6_table9_robustness.csv`, `exp6_lambda_bar_final.csv`, `exp6_meta.json`,
`exp6_factors_cumulative.png`, `exp6_lambda_bar.png`.

| Metric (Table 7) | Paper | Replication | Verdict |
|---|---|---|---|
| STREV mean / SR | 0.26% / 0.22 | 0.09% (t = 0.35) / 0.09 | not reproduced |
| RES Char mean / SR | 0.88% / 1.52 | 0.30% (t = 1.55) / 0.44 | not reproduced |
| RES Char+Ind mean / SR | 0.95% / 1.82 | 0.24% (t = 1.52) / 0.44 | not reproduced |
| RES Char+Ind+ZeroEAD mean / SR | 1.07% / 2.12 | 0.35% (t = 2.56) / 0.70 | not reproduced |
| ASRES mean | 1.18% (t = 10.56) | 0.37% (t = 2.23) | not reproduced |
| ASRES annualized volatility | 5.6% | 7.2% | partially reproduced |
| ASRES annualized SR | 2.55 | 0.61 | not reproduced |
| Sharpe ranking STREV < RES variants < ASRES | holds | 0.09 < 0.44 ≈ 0.44 < 0.70, ASRES 0.61 below ZeroEAD | not reproduced |
| ASRES CAPM alpha | 1.21% (t = 15.02) | 0.10% (t = 0.73) | not reproduced |
| ASRES FF6 alpha | 1.08% (t = 8.04) | 0.26% (t = 2.53) | not reproduced |
| ASRES FF6+REV alpha | 0.98% (t = 13.69) | 0.20% (t = 1.95) | not reproduced |

| Loading (Table 8) | Paper | Replication | Verdict |
|---|---|---|---|
| REV | +0.23 (t = 8.21) | +0.36 (t = 5.90) | not reproduced (56% larger) |
| RMW | +0.17 | −0.09 (t = −1.65) | not reproduced |
| HML | +0.11 | −0.03 (t = −0.48) | not reproduced |
| UMD | +0.08 | +0.00 (t = 0.03) | not reproduced |
| SMB | −0.11 | +0.21 (t = 3.70) | not reproduced |

| Group (Table 9) | Paper mean / SR | Replication mean / SR | Verdict |
|---|---|---|---|
| Micro | 1.35% / 1.18 | 0.60% (t = 2.20) / 0.62 | not reproduced |
| Small | 1.43% / 2.43 | 0.37% (t = 2.02) / 0.57 | not reproduced |
| Large | 1.16% / 2.03 | 0.23% (t = 1.21) / 0.35 | not reproduced |
| Mega | 0.89% / 1.74 | 0.29% (t = 2.26) / 0.59 | not reproduced |
| Post-decimalization | 0.60% (t = 6.38) / 1.41 | 0.37% (t = 2.23) / 0.61 | partially reproduced (mean) / not reproduced (SR) |
| Pre-decimalization | 1.65% / 3.66 | — | not reproduced (outside the sample) |

The replication period lies entirely in the paper's post-decimalization
subsample, whose ASRES mean (0.60%) is the relevant benchmark; the replicated
0.37% is 38% lower. The ASRES and RES factors remain positive and, with FF6
alpha t = 2.5–2.9 for ASRES and RES ZeroEAD, the zeroing of earnings
announcement windows raises the RES factor's Sharpe ratio from 0.44 to 0.70,
in line with the paper's progression across RES variants.

---

## Deviations from the paper

- **Data source and sample period.** The paper uses CRSP (CIZ), Compustat and
  the Jensen–Kelly–Pedersen (JKP) characteristics for January 1972 – December
  2022. Here daily bars come from the QuantX Dataloader (Massive data,
  split- and dividend-adjusted), reference data, shares outstanding, quarterly
  statements, SIC codes, dividends, splits and EDGAR filing dates from Massive
  REST, and Fama-French factors and the 49-industry SIC map from Kenneth
  French's data library. Massive daily bars start in September 2003 and its
  quarterly statements in 2009–2010; bars before 2003 are available only for
  surviving tickers, which would introduce survivorship bias. The executed
  sample is therefore bars from January 2008, decomposition and monthly
  tests from January 2011 to December 2022 (impulse responses and ASRES from
  January 2012), the last date being the paper's. No series in the load came
  from a source other than Massive (`non_massive_source` is empty); 58 of
  10,750 requested tickers returned no bars and are excluded (delisted or
  renamed names without Massive bars in the window; their share of the
  universe cannot be computed without bars, and no gap is attributed to
  them). The Dataloader returned load warnings for 3,710 tickers (series
  that start after the window start, i.e. later listings).
- **Universe.** Common stocks (Massive type `CS`) with primary exchange XNYS,
  XNAS or XASE, active and delisted, collected from year-end reference
  snapshots 2010–2022, with the paper's 0.003% entry / 0.002% exit
  market-cap rule (Section 2). The CRSP share-code, issuer-type and
  incorporation filters have no Massive equivalent. Market equity uses
  dividend-unadjusted closes times basic shares outstanding from the latest
  quarterly statement (restated share counts converted to the bars' split
  basis); share records implying a trailing median monthly turnover below 3%
  or jumping by more than a factor of 10 are rejected, and sub-cent closes are
  treated as missing (thresholds not specified in the paper). The resulting
  universe holds 1,060–1,780 stocks per month.
- **Characteristics.** The paper uses 151 JKP characteristics; 51 JKP-named
  characteristics that can be built from bars and Massive statements are used
  here (market, momentum, volatility, liquidity, valuation, profitability,
  investment and accruals groups), of which 27–50 pass the paper's coverage
  and collinearity filters each month. The ranking, normalization and filter
  rules follow Section 3.
- **Industries.** SIC codes are the current Massive codes for each security,
  not the paper's point-in-time Compustat/CRSP SIC history; unmatched codes go
  to industry 49 (Other).
- **Earnings announcement dates.** Compustat RDQ is not available. The
  announcement date is the first 8-K filed between 10 and 100 days after the
  fiscal quarter end and not after the quarter's 10-Q/10-K, else the 10-Q/10-K
  filing date (earnings releases are furnished on 8-K). The [-1, +1]
  trading-day window is applied as in Section 3.
- **SUE** uses quarterly EPS from Massive statements (4-quarter change over the
  standard deviation of the previous 8 quarters, Section 4).
- **Mispricing score (Table 3).** The Stambaugh–Yu–Yuan (2015) score is
  rebuilt from 8 of its 11 anomalies that the available data allow (net stock
  issuance, operating accruals, net operating assets, asset growth,
  investment-to-assets, momentum, gross profitability, return on assets),
  averaging cross-sectional percentile ranks with at least 5 available; the
  test window is formation months 2011–2016 (65–68 months per quintile)
  instead of 1972–2016.
- **FF3 benchmark (Figure 1).** The paper does not detail the benchmark's
  window; separate regressions per stock and calendar month with four
  parameters per stock-month are used.
- **Figure 2 lookback.** For H = 1 the demeaning uses the mean of the H
  residuals with divisor max(H − 1, 1), and the lookback ends on the last
  trading day before the return window.
- **ASRES spline (Eq. 25).** The paper does not state how the lag-1 knot is
  treated when lags start at 2; the knot at lag 1 is dropped, leaving 26 tent
  functions per shock. The two-year burn-in starts at the first decomposition
  month (January 2011), and the first forecast regressions (January 2012) need
  252 days of lagged shocks, so the expanding window holds one year of
  regressions at the first formation date; holding months are January 2013 –
  December 2022 (120 months) instead of January 1974 – December 2022.
- **Newey-West lags** for Table 1 are not specified in the paper; automatic
  lag selection floor(4 (T/100)^(2/9)) is used throughout.
- **Magnitude gaps.** Reversal and ASRES returns are about one third of the
  paper's. The Fama-French short-term reversal factor itself earned 0.165% per
  month in 2013–2022 against 0.421% in 1974–2022 (`exp6_meta.json`), so the replication window
  has weaker reversal profits than the paper's full sample; whether the paper's
  own estimates for 2011–2022 are similarly lower is not reported in the
  paper, and the cause of the gap was not otherwise identified.

## Not reproduced

- Results for 1972–2010 (including Table 9's pre-decimalization subsample and
  Table E.2's original-publication-sample windows): Massive daily bars start in
  2003 and its quarterly statements in 2009–2010, so CRSP/Compustat-quality
  data for that period are unavailable from the data sources used.
- All other comparisons are reported above with their verdicts.
