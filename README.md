# Asymmetric Reversals — replication

**Paper:** Federico Baldi-Lanfranchi, Pierre Collin-Dufresne and Kent Daniel (2026),
"Asymmetric Reversals", NBER Working Paper 35840, October 2026.
<http://www.nber.org/papers/w35840>. The paper does not link code or data.

## What is replicated

The paper splits each stock's daily excess return into a systematic part
(fitted by daily WLS cross-sectional regressions on firm characteristics and
Fama-French 49 industry dummies, Section 3) and a residual part, moves returns
inside the [-1, +1] earnings-announcement window to the systematic part, and
shows that monthly residual returns reverse strongly while systematic returns
continue; positive residual shocks reverse more than negative ones at the
monthly horizon (AIR), which spans the IVOL anomaly and most short-horizon
anomalies; daily impulse responses explain the asymmetry; and an out-of-sample
factor built from spline-estimated lag coefficients (ASRES) earns a high Sharpe
ratio.

| Script | Paper content |
|---|---|
| `exp/exp1_decomposition_fit.py` | Section 3, 3.1, Figure 1: annual adjusted R² of the characteristics and characteristics + industries WLS models vs. stock-by-stock FF3 (WLS and OLS) |
| `exp/exp2_res_sys_sorts.py` | Section 4, Table 1: quintile sorts on RES and SYS |
| `exp/exp3_ivol_air.py` | Section 5.1–5.4: Eq. (15), Table 2, Figure 2, Table 3 (MISP quintiles), Table 4 (concentration) |
| `exp/exp4_zoo_spanning.py` | Section 5.5, Table 6: 13 short-term characteristics vs. RES+, RES−, CONC, SYS |
| `exp/exp5_irf.py` | Section 6: dense (Eq. 19) and spline (Eqs. 22–24) impulse responses, Figure 7 aggregation weights |
| `exp/exp6_asres.py` | Section 7, Tables 7–9: STREV, RES and ASRES factors |

Results and the comparison with the paper are in `results/RESULTS.md`.

## Data

- Daily OHLCV bars: QuantX Dataloader REST API (Massive data), split- and
  dividend-adjusted; dividends from Massive are used to recover prices without
  dividend adjustment for market equity.
- Universe: Massive reference tickers (common stocks, type `CS`, primary
  exchange XNYS/XNAS/XASE), active and delisted, collected point in time.
- Shares outstanding, quarterly fundamentals, SIC codes, splits, dividends and
  EDGAR filing dates: Massive REST.
- Fama-French daily/monthly factors and the 49-industry SIC map: Kenneth
  French's data library.

The paper uses CRSP, Compustat and the JKP characteristics for 1972–2022.
Massive bars and statements cover the later part of that period only; the
sample used here and all other differences are listed in
`results/RESULTS.md` ("Deviations from the paper").

## Setup and usage

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
python -m pytest -m "not market_data"   # unit tests (no API calls)
python -m pytest                        # also the market-data slice tests
python -m exp.exp1_decomposition_fit
python -m exp.exp2_res_sys_sorts
python -m exp.exp3_ivol_air
python -m exp.exp4_zoo_spanning
python -m exp.exp5_irf
python -m exp.exp6_asres
```

`bash reproduce.sh` runs the tests and all experiments in this order. The
first experiment downloads all data and builds the shared panel (about 75
minutes of download and 5 minutes of processing); downloads and panels are
cached under `/tmp/asymrev_cache` and reused by the other experiments.
Parameters are in `src/config.py`.

## Layout

- `data/data_loader.py` — all API access (QuantX Dataloader, Massive REST, Ken French library).
- `src/` — universe and market equity (`universe.py`, `build.py`), characteristics
  (`characteristics.py`, `fundamentals.py`), daily decomposition (`decomposition.py`),
  monthly signals (`signals.py`), Fama-MacBeth / Newey-West / sorts (`stats.py`),
  short-term zoo (`zoo.py`), impulse responses and splines (`irf.py`), ASRES (`asres.py`),
  shared panel (`pipeline.py`), outputs (`reporting.py`), parameters (`config.py`).
- `exp/` — one script per experiment; `tests/` — pytest suite; `results/` — tables, figures, `RESULTS.md`.
