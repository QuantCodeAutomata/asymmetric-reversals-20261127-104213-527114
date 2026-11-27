"""Market and reference data for the replication of "Asymmetric Reversals".

* Daily OHLCV bars: QuantX Dataloader REST API (``POST /fetch-ohlcvs``).
* Point-in-time ticker universe, SIC codes, splits, quarterly fundamentals and
  the SEC EDGAR filing index: Massive REST (https://api.massive.com).
* Fama-French factors and the 49-industry SIC map: Kenneth French's data library
  (the source named in Section 2 of the paper).

All downloads are cached under ``src.config.CACHE_DIR`` (outside the repository).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv

from src import config

log = logging.getLogger(__name__)

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
MASSIVE_ROOT = "https://api.massive.com"
FRENCH_ROOT = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15  # tickers per Dataloader request; batches are split by ticker, never by date
MAX_PARALLEL = 4  # concurrent Dataloader requests
LOAD_CHUNK = 1000  # tickers per cached load in load_universe_bars
RETRY_STATUSES = {429, 500, 502, 503, 504}
CACHE_DIR = Path(config.CACHE_DIR)


class DataloaderError(RuntimeError):
    """A failed data request (credentials, HTTP error, or empty load)."""


# --------------------------------------------------------------------- helpers
def _keys() -> tuple[str, str]:
    load_dotenv(find_dotenv(usecwd=True))
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return dataloader_key, massive_key


def _cache_path(name: str, payload: dict) -> Path:
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_{key}.parquet"


def _request(method: str, route: str, body: dict | None = None) -> dict:
    dataloader_key, massive_key = _keys()
    headers = {"Authorization": f"Bearer {dataloader_key}"}
    if route == "/fetch-ohlcvs":
        headers["X-Massive-Api-Key"] = massive_key  # the Massive key goes to fetch-ohlcvs only
    for attempt in range(4):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        try:
            response = requests.request(method, API_ROOT + route, json=body, headers=headers, timeout=900)
        except requests.ConnectionError:
            if attempt == 3:
                raise
            time.sleep(5 * 2**attempt)
            continue
        if response.status_code not in RETRY_STATUSES or attempt == 3:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2**attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _massive_get(path: str, params: dict | None = None) -> list[dict]:
    """GET a Massive REST endpoint and follow ``next_url`` until exhausted."""
    _, massive_key = _keys()
    headers = {"Authorization": f"Bearer {massive_key}"}
    url, out, seen = MASSIVE_ROOT + path, [], set()
    while url:
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=120)
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 * 2**attempt)
                continue
            if resp.status_code in RETRY_STATUSES and attempt < 4:
                time.sleep(float(resp.headers.get("Retry-After") or 2 * 2**attempt))
                continue
            break
        if resp.status_code == 404:
            return out
        if resp.status_code != 200:
            raise DataloaderError(f"Massive {path}: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        res = data.get("results")
        if isinstance(res, dict):
            return [res]
        out.extend(res or [])
        nxt = data.get("next_url")
        if not nxt or nxt in seen:
            break
        seen.add(nxt)
        url, params = nxt, None
    return out


# ------------------------------------------------------------------- OHLCV bars
def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    """The wide table of one answer: inline rows, or one CSV export when stored."""
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)  # signed link; never logged
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    """Wide columns to rows of date, ticker, open..volume (one per bar)."""
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    dates = stamps.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        listed = [c for c in (detail.get("columns") or []) if isinstance(c, str)]
        for field in FIELDS:
            cands = [f"{ticker}_{field}", f"{alias}_{field}", f"{alias.lower()}_{field}"]
            cands += [c for c in listed if c.lower().endswith(f"_{field}")] + [c.lower() for c in listed
                                                                              if c.lower().endswith(f"_{field}")]
            if len(details) == 1:
                cands.append(field)
            for name in cands:
                if name in wide.columns:
                    columns[name] = field
                    break
        if len(columns) != len(FIELDS):
            log.warning("%s: columns not found in the answer; ticker skipped", ticker)
            detail["error"] = {"message": "columns not found in answer"}
            continue
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "date", dates)
        frames.append(bars.dropna(subset=["close"]))  # an absent bar is not a zero bar
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict]]:
    try:
        answer = _request("POST", "/fetch-ohlcvs", request)
    except DataloaderError as exc:
        # a batch in which no ticker loads (e.g. only invalid symbols) is answered with an
        # error status; its tickers are reported as failed instead of aborting the load
        if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
            raise
        log.warning("batch %d failed: %s", batch, exc)
        return pd.DataFrame(), [{"ticker": t, "error": {"message": str(exc)[:200]}} for t in request["tickers"]]
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    if not loaded:
        return pd.DataFrame(), details
    return _tidy(_bars_of(answer, request, batch), details), details


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = False,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Daily bars for ``tickers`` (one identity date) and load metadata.

    Returns rows of date, ticker, open, high, low, close, volume and a metadata
    dict with ``ticker_details``, ``failed``, ``warnings`` and ``non_massive_source``.
    """
    request = {
        "timeframe": timeframe, "start_date": start_date, "end_date": end_date,
        "source": "massive", "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    path = _cache_path("ohlcv", {**request, "tickers": sorted(tickers)})
    meta_path = path.with_suffix(".json")
    if use_cache and path.exists() and meta_path.exists():
        return pd.read_parquet(path), json.loads(meta_path.read_text())
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]), enumerate(batches)))
    frames = [bars for bars, _ in results if not bars.empty]
    details = [d for _, batch_details in results for d in batch_details]
    failed = {d["ticker"]: (d.get("error") or {}).get("message") for d in details if d.get("error")}
    bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", *FIELDS])
    meta = {
        "request": request,
        "ticker_details": [{k: d.get(k) for k in ("ticker", "data_source", "security_id", "warnings", "error")}
                           for d in details],
        "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {d["ticker"]: d.get("data_source") for d in details
                               if not d.get("error") and d.get("data_source") not in (None, "massive")},
    }
    bars.to_parquet(path, index=False)
    meta_path.write_text(json.dumps(meta, default=str))
    return bars, meta


def load_universe_bars(universe: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Dividend-adjusted daily bars for every ticker-security of ``universe``.

    Tickers are requested with ``identity_date`` = the last year-end on which the
    reference data lists them (a date inside their trading life), grouped by that
    date. Bars carry the column ``security`` (ticker-security key of ``universe``).
    """
    frames, meta_all = [], {"failed": {}, "warnings": {}, "non_massive_source": {}, "n_requested": 0}
    sec_cat = pd.CategoricalDtype(sorted(universe["security"].astype(str).unique()))
    for ident, grp in universe.groupby("last_seen"):
        key = grp.drop_duplicates("ticker").set_index("ticker")["security"].astype(str)
        tickers = sorted(key.index)
        # chunks are cached separately so that a failed request never discards a finished chunk
        for i in range(0, len(tickers), LOAD_CHUNK):
            chunk = tickers[i:i + LOAD_CHUNK]
            t0 = time.time()
            bars, meta = load_ohlcv(chunk, start, end, identity_date=ident)
            if not bars.empty:
                bars["security"] = bars["ticker"].map(key).astype(sec_cat)
                bars = bars.drop(columns="ticker")
                for c in FIELDS:
                    bars[c] = bars[c].astype("float32" if c != "volume" else "float64")
                frames.append(bars)
            for k in ("failed", "warnings", "non_massive_source"):
                meta_all[k].update({f"{t}@{ident}": v for t, v in meta[k].items()})
            meta_all["n_requested"] += len(chunk)
            log.info("bars identity %s: %d tickers, %d rows, %d failed, %.0fs", ident, len(chunk),
                     len(bars), len(meta["failed"]), time.time() - t0)
    bars = pd.concat(frames, ignore_index=True)
    del frames
    bars = bars.drop_duplicates(["security", "date"], keep="last")
    return bars, meta_all


def load_dividends(start: str, end: str) -> pd.DataFrame:
    """Cash dividends with ex-date in [start, end] (``/stocks/v1/dividends``)."""
    path = _cache_path("dividends", {"s": start, "e": end})
    if path.exists():
        return pd.read_parquet(path)
    years = range(int(start[:4]), int(end[:4]) + 1)

    def fetch(y: int) -> list[dict]:
        return _massive_get("/stocks/v1/dividends", {"ex_dividend_date.gte": f"{y}-01-01",
                                                     "ex_dividend_date.lte": f"{y}-12-31", "limit": 5000})

    rows = []
    with ThreadPoolExecutor(4) as pool:
        for res in pool.map(fetch, years):
            rows.extend(res)
    df = pd.DataFrame(rows)
    if df.empty:
        raise DataloaderError("No dividends returned")
    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------- reference universe
def load_reference_universe(years: list[int], exchanges: tuple[str, ...] = config.EXCHANGES) -> pd.DataFrame:
    """Point-in-time common-stock universe from Massive ``/v3/reference/tickers``.

    For every year-end date in ``years`` the tickers of type CS listed on the
    paper's exchanges (NYSE, Amex, NASDAQ) on that date are collected, so
    names delisted later are included. One row per (ticker, composite_figi/cik)
    with the first/last year-end it was seen and its primary exchange.
    """
    path = _cache_path("universe", {"years": years, "ex": exchanges})
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    jobs = []
    for y in years:
        d = pd.Timestamp(f"{y}-12-31")
        d = d - pd.offsets.BDay(0) if d.dayofweek < 5 else d - pd.offsets.BDay(1)
        for ex in exchanges:
            jobs.append((d.strftime("%Y-%m-%d"), ex))

    def fetch(job: tuple[str, str]) -> list[dict]:
        date, ex = job
        res = _massive_get("/v3/reference/tickers", {
            "type": "CS", "market": "stocks", "exchange": ex, "date": date, "limit": 1000})
        return [{**r, "asof": date} for r in res]

    with ThreadPoolExecutor(6) as pool:
        for res in pool.map(fetch, jobs):
            rows.extend(res)
    if not rows:
        raise DataloaderError("Empty reference universe")
    df = pd.DataFrame(rows)
    df["composite_figi"] = df.get("composite_figi").fillna("")
    df["cik"] = df.get("cik").fillna("")
    df["security"] = np.where(df["composite_figi"] != "", df["composite_figi"],
                              np.where(df["cik"] != "", "CIK" + df["cik"], "TK" + df["ticker"]))
    df = df.sort_values("asof")
    g = df.groupby(["ticker", "security"])
    out = g.agg(first_seen=("asof", "min"), last_seen=("asof", "max"), cik=("cik", "last"),
                name=("name", "last"), primary_exchange=("primary_exchange", "last"),
                delisted_utc=("delisted_utc", "last") if "delisted_utc" in df else ("asof", "last")).reset_index()
    out.to_parquet(path, index=False)
    log.info("reference universe: %d ticker-securities, exchanges %s", len(out),
             out["primary_exchange"].value_counts().to_dict())
    return out


def load_ticker_details(tickers: list[str], dates: list[str]) -> pd.DataFrame:
    """SIC code and CIK per ticker as of its identity date (``/v3/reference/tickers/{ticker}``)."""
    path = _cache_path("details", {"t": list(zip(tickers, dates))})
    if path.exists():
        return pd.read_parquet(path)

    def fetch(job: tuple[str, str]) -> dict:
        t, d = job
        res = _massive_get(f"/v3/reference/tickers/{t}", {"date": d})
        r = res[0] if res else {}
        return {"ticker": t, "identity_date": d, "sic_code": r.get("sic_code"), "cik_details": r.get("cik"),
                "list_date": r.get("list_date")}

    with ThreadPoolExecutor(8) as pool:
        out = pd.DataFrame(list(pool.map(fetch, list(zip(tickers, dates)))))
    out.to_parquet(path, index=False)
    return out


def load_splits(start: str, end: str) -> pd.DataFrame:
    """All stock splits executed in [start, end] (``/stocks/v1/splits``)."""
    path = _cache_path("splits", {"s": start, "e": end})
    if path.exists():
        return pd.read_parquet(path)
    res = _massive_get("/stocks/v1/splits", {"execution_date.gte": start, "execution_date.lte": end,
                                             "limit": 5000, "sort": "execution_date.asc"})
    df = pd.DataFrame(res)
    if df.empty:
        raise DataloaderError("No splits returned")
    df = df[["ticker", "execution_date", "split_from", "split_to"]]
    df.to_parquet(path, index=False)
    return df


def load_fundamentals(kind: str, timeframe: str, start: str, end: str, fields: list[str]) -> pd.DataFrame:
    """Quarterly/TTM statements from ``/stocks/financials/v1/{kind}`` with period_end in [start, end]."""
    path = _cache_path(f"fund_{kind}", {"tf": timeframe, "s": start, "e": end, "f": sorted(fields)})
    if path.exists():
        return pd.read_parquet(path)
    base = ["cik", "tickers", "period_end", "filing_date", "fiscal_year", "fiscal_quarter", "timeframe"]
    months = pd.date_range(start, end, freq="QS")
    jobs = [(m.strftime("%Y-%m-%d"), (m + pd.offsets.QuarterEnd(0)).strftime("%Y-%m-%d")) for m in months]

    def fetch(job: tuple[str, str]) -> list[dict]:
        return _massive_get(f"/stocks/financials/v1/{kind}", {
            "timeframe": timeframe, "period_end.gte": job[0], "period_end.lte": job[1],
            "limit": 50000, "fields": ",".join(base + fields)})

    rows = []
    with ThreadPoolExecutor(4) as pool:
        for res in pool.map(fetch, jobs):
            rows.extend(res)
    df = pd.DataFrame(rows)
    if df.empty:
        raise DataloaderError(f"No {kind} returned")
    df["tickers"] = df["tickers"].apply(lambda x: ",".join(x) if isinstance(x, (list, np.ndarray)) else (x or ""))
    for c in fields:
        if c not in df:
            df[c] = np.nan
    df.to_parquet(path, index=False)
    return df


def load_filing_index(form_types: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """SEC EDGAR filing index rows (cik, form_type, filing_date) from ``/stocks/filings/vX/index``."""
    path = _cache_path("filings", {"f": form_types, "s": start, "e": end})
    if path.exists():
        return pd.read_parquet(path)
    months = pd.date_range(start, end, freq="MS")
    jobs = [(f, m.strftime("%Y-%m-%d"), (m + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"))
            for f in form_types for m in months]

    def fetch(job: tuple[str, str, str]) -> list[dict]:
        return _massive_get("/stocks/filings/vX/index", {
            "form_type": job[0], "filing_date.gte": job[1], "filing_date.lte": job[2], "limit": 10000,
            "fields": "cik,form_type,filing_date"})

    rows = []
    with ThreadPoolExecutor(6) as pool:
        for res in pool.map(fetch, jobs):
            rows.extend(res)
    df = pd.DataFrame(rows)
    if df.empty:
        raise DataloaderError("Empty filing index")
    df = df[["cik", "form_type", "filing_date"]].drop_duplicates()
    df.to_parquet(path, index=False)
    return df


# ------------------------------------------------------- Kenneth French library
def _french_csv(name: str) -> str:
    path = CACHE_DIR / f"french_{name}.txt"
    if path.exists():
        return path.read_text()
    resp = requests.get(f"{FRENCH_ROOT}/{name}.zip", timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def _parse_french(text: str, daily: bool) -> pd.DataFrame:
    """First data block of a French CSV (percent units) -> decimal returns."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",") or l.lower().startswith(",mkt")
                 or (l.count(",") >= 1 and not l.strip()[0:1].isdigit() and i + 1 < len(lines)
                     and lines[i + 1].strip()[:6].isdigit()))
    header = [h.strip() for h in lines[start].split(",")]
    recs = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != (8 if daily else 6):
            break
        recs.append(parts)
    df = pd.DataFrame(recs, columns=["date"] + header[1:])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d" if daily else "%Y%m")
    if not daily:
        df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df = df.set_index("date").astype(float) / 100.0
    df.columns = [c.replace("-", "_").replace(" ", "_").upper() for c in df.columns]
    return df


def load_ff_daily() -> pd.DataFrame:
    """Daily MKT_RF, SMB, HML, RF (decimal) from French's library."""
    return _parse_french(_french_csv("F-F_Research_Data_Factors_daily_CSV"), daily=True)


def load_ff_monthly() -> pd.DataFrame:
    """Monthly MKT_RF, SMB, HML, RMW, CMA, RF, UMD, REV (decimal) from French's library."""
    ff5 = _parse_french(_french_csv("F-F_Research_Data_5_Factors_2x3_CSV"), daily=False)
    mom = _parse_french(_french_csv("F-F_Momentum_Factor_CSV"), daily=False)
    rev = _parse_french(_french_csv("F-F_ST_Reversal_Factor_CSV"), daily=False)
    mom.columns, rev.columns = ["UMD"], ["REV"]
    return ff5.join(mom, how="inner").join(rev, how="inner")


def load_ff49_sic_map() -> list[tuple[int, int, int]]:
    """(industry number 1..49, sic_low, sic_high) ranges from French's Siccodes49."""
    text = _french_csv("Siccodes49")
    out, ind = [], None
    for l in text.splitlines():
        s = l.strip()
        if not s:
            continue
        tok = s.split()
        if tok[0].isdigit() and len(tok) > 1 and not tok[1][0].isdigit():
            ind = int(tok[0])
        elif "-" in tok[0] and ind is not None:
            lo, hi = tok[0].split("-")
            out.append((ind, int(lo), int(hi)))
    if len({i for i, _, _ in out}) != config.N_INDUSTRIES:
        raise DataloaderError("Siccodes49 parsing failed")
    return out
