# AGENTS.md

Replication of Baldi-Lanfranchi, Collin-Dufresne and Daniel (2026),
"Asymmetric Reversals", NBER Working Paper 35840
(<http://www.nber.org/papers/w35840>): daily WLS return decomposition into
systematic and residual parts, RES/SYS sorts, AIR vs. IVOL, the short-term
anomaly zoo, daily impulse responses and the out-of-sample ASRES factor.

## Layout

- `data/data_loader.py` — the only place that calls external data (QuantX
  Dataloader `POST /fetch-ohlcvs` for bars, Massive REST for reference data,
  shares, fundamentals, splits, dividends, Ken French library for factors and
  the FF49 SIC map). Caches go to `/tmp/asymrev_cache`, never to the repository.
- `src/` — flat library modules; `src/config.py` holds every parameter with
  its source. `src/pipeline.py` builds the shared panel used by all experiments.
- `exp/` — one script per experiment (`exp1_decomposition_fit` …
  `exp6_asres`); each writes aggregate tables (`.csv`/`.json`) and `.png`
  figures to `results/`.
- `tests/` — pytest suite; `tests/conftest.py` registers the `market_data`
  marker and the session fixture that loads a small cached slice.
- `results/RESULTS.md` — comparison with the paper, deviations, items not reproduced.

## Commands (from the repository root)

```bash
pip install -r requirements.txt
cp .env.example .env                      # DATALOADER_API_KEY, MASSIVE_API_KEY
python -m pytest -m "not market_data"     # unit tests, no API calls
python -m pytest                          # plus market-data slice tests (needs keys)
python -m exp.exp1_decomposition_fit      # first run downloads data and builds the panel
python -m exp.exp2_res_sys_sorts
python -m exp.exp3_ivol_air
python -m exp.exp4_zoo_spanning
python -m exp.exp5_irf
python -m exp.exp6_asres
bash reproduce.sh                         # tests and all experiments
```

There is a single configuration (the one in `src/config.py`); its sample is
set by the coverage of the data source, as described in `results/RESULTS.md`.

## Conventions

- Parameters only in `src/config.py` (module-level constants with a comment
  on meaning, unit and source).
- Data only through `data/data_loader.py`; no loaders in `src/` or `exp/`.
- Unit tests use synthetic inputs; tests on market data are marked
  `market_data` and skipped without `DATALOADER_API_KEY`.
- API keys only in an uncommitted `.env`. Never commit market data or
  date-indexed series, caches or `.html` files; `results/` holds only
  aggregate tables and static figures.
- Rerunning an experiment overwrites its files in `results/`; the numbers and
  verdicts in `results/RESULTS.md` must then be checked against the new files
  and updated by hand.
