"""All experiment parameters (module-level constants).

Each constant states its meaning, unit and source: a section/equation of
Baldi-Lanfranchi, Collin-Dufresne and Daniel (2026), "Asymmetric Reversals",
or "not specified in the paper".
"""

from pathlib import Path

# ---------------------------------------------------------------- data / sample
CACHE_DIR = Path("/tmp/asymrev_cache")  # on-disk cache of downloaded and derived panels (outside the repository)
PAPER_SAMPLE_START = "1972-01-01"  # paper sample start (Section 2)
PAPER_SAMPLE_END = "2022-12-31"  # paper sample end (Section 2)
PRICE_START = "2008-01-01"  # first date of daily bars loaded (lookback for momentum/beta windows); Massive bars start Sep 2003
PRICE_END = "2022-12-30"  # last trading day of the paper sample (Section 2)
FUND_START = "2008-01-01"  # first fundamental period_end loaded; Massive statements start ~2009-2010
DECOMP_START = "2011-01-01"  # first month of the daily decomposition (first month with full shares-outstanding coverage)
DECOMP_END = PRICE_END  # last day of the decomposition (Section 2)
IDENTITY_DATE = PRICE_END  # as-of date that identifies active tickers (last date of the sample)
EXCHANGES = ("XNYS", "XNAS", "XASE")  # primary exchanges N, A, Q (Section 2)
NYSE_EXCHANGE = "XNYS"  # exchange used for NYSE breakpoints (Section 2)
FUND_LAG_MONTHS = 4  # months after period end until a quarterly statement is public (JKP convention; Section 2 cites JKP)
SHARES_STALE_MONTHS = 12  # maximum age (months) of the shares-outstanding record used for market equity; not specified in the paper
SPLITS_END = "2026-11-01"  # splits loaded through this date; splits after PRICE_END convert restated share counts to the bars' split basis
MIN_MONTHLY_TURNOVER = 0.03  # fraction; shares records implying a trailing-12-month median monthly turnover below 3% are treated as mis-scaled; not specified in the paper
SHARES_JUMP_FACTOR = 10.0  # a quarterly shares record more than this factor away from the median of the previous four is rejected as a unit error; not specified in the paper
MIN_BAR_PRICE = 0.01  # USD; daily bars with a sub-cent close are treated as missing (data errors); not specified in the paper
FILINGS_START = "2009-01-01"  # first EDGAR filing date loaded for the earnings-announcement proxy
FILINGS_END = "2023-03-31"  # last EDGAR filing date loaded (covers 2022 Q4 announcements)
FILING_FORMS = ("8-K", "10-Q", "10-K")  # EDGAR forms used for the earnings-announcement proxy
FUND_INCOME_FIELDS = ["revenue", "cost_of_revenue", "gross_profit", "selling_general_administrative", "research_development",
                      "operating_income", "interest_expense", "income_before_income_taxes", "income_taxes",
                      "consolidated_net_income_loss", "net_income_loss_attributable_common_shareholders",
                      "basic_earnings_per_share", "diluted_earnings_per_share", "basic_shares_outstanding",
                      "diluted_shares_outstanding", "ebitda"]  # quarterly income-statement items used by the characteristics
FUND_BALANCE_FIELDS = ["cash_and_equivalents", "short_term_investments", "receivables", "inventories",
                       "total_current_assets", "property_plant_equipment_net", "goodwill", "intangible_assets_net",
                       "total_assets", "accounts_payable", "debt_current", "total_current_liabilities",
                       "long_term_debt_and_capital_lease_obligations", "total_liabilities", "retained_earnings_deficit",
                       "total_equity_attributable_to_parent", "total_equity", "treasury_stock"]  # balance-sheet items
FUND_CASHFLOW_FIELDS = ["net_income", "depreciation_depletion_and_amortization", "net_cash_from_operating_activities",
                        "purchase_of_property_plant_and_equipment", "dividends", "net_cash_from_financing_activities",
                        "long_term_debt_issuances_repayments"]  # cash-flow items
REFERENCE_YEARS = list(range(2010, 2023))  # year-ends at which the point-in-time common-stock universe is collected

# ------------------------------------------------------------- investable universe
UNIV_ENTRY_SHARE = 0.003 / 100  # add a stock when ME > 0.003% of aggregate market cap (Section 2)
UNIV_EXIT_SHARE = 0.002 / 100  # drop a stock when ME < 0.002% of aggregate market cap (Section 2)
WEIGHT_WINSOR_PCT = 0.80  # regression/portfolio weights: ME winsorized at the 80th NYSE percentile (Section 3)

# ---------------------------------------------------------------- characteristics
CHAR_MIN_COVERAGE_IN = 0.40  # include a characteristic when >= 40% of stocks have it (Section 3)
CHAR_MIN_COVERAGE_OUT = 0.20  # remove a characteristic when coverage < 20% (Section 3)
CHAR_CORR_DROP = 0.99  # drop the shorter-tenor characteristic when |corr| >= 99% (Section 3)
CHAR_CORR_READD = 0.975  # reintroduce it when |corr| <= 97.5% (Section 3)
N_INDUSTRIES = 49  # Fama-French 49 industries (Section 3)
EA_WINDOW = (-1, 1)  # trading days around an earnings announcement assigned to SYS (Section 3)
EA_8K_MIN_LAG_DAYS = 10  # earnings 8-K searched from period end + 10 days; not specified in the paper (EA-date proxy)
EA_SEARCH_MAX_DAYS = 100  # last day after period end searched for the EA proxy; not specified in the paper

# --------------------------------------------------------------- exp 2 portfolios
N_QUANTILES = 5  # quintile portfolios with NYSE breakpoints (Section 4, Table 1)
NW_LAGS_PORT = None  # Newey-West lags for portfolio t-stats: None = automatic floor(4 (T/100)^(2/9)); not specified for Table 1

# --------------------------------------------------------------- exp 3 / exp 4
MIN_DAILY_OBS = 10  # minimum valid daily observations for monthly short-term characteristics (Appendix E.1)
IVOL_H_GRID = tuple(range(1, 22))  # lookback H = 1..21 days for IVOL horizon dependence (Figure 2)
HOLD_HORIZONS = (1, 5, 21)  # holding horizons in trading days (Figure 2)
MISP_END = "2016-12-31"  # last month with the Stambaugh-Yu-Yuan mispricing score (Table 3)
BETA_MONTHS = 60  # months in the FF-control market beta (Section 5.2)
BETA_MIN_MONTHS = 24  # minimum months for the beta control; not specified in the paper
AMIHUD_SCALE = 1e6  # Amihud illiquidity scaling (Table 1, Appendix E.1)
ZERO_TRADES_DEFLATOR = 480_000  # Liu (2006) deflator in Zero Trades (Appendix E.1)
SMAX_VOL_DAYS = 252  # trailing days for the SMAX volatility scaling (Appendix E.1)
SKEW_SCALE = 100.0  # Skew, ISkew, Coskew are divided by 100 in regressions (Table 6)

# -------------------------------------------------------------- exp 5 / exp 6 IRF
N_LAGS = 252  # daily lags in the impulse-response regressions (Equation 19)
SPLINE_KNOTS = (1, 2, 3, 4, 5, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63,
                74, 95, 116, 137, 158, 179, 200, 221, 242, 252)  # 27 degree-1 knots (Figure A.5)
SHOCK_SIZE = 0.01  # 1% shock in the impulse responses (Equations 20-21)
FORECAST_HORIZON = 21  # trading days of the ASRES forecast target R^21 (Equation 25)
ASRES_FIRST_LAG = 2  # first lag used by ASRES (1-day implementation gap, Equation 25)
ASRES_SHORT_LAG = 22  # last lag of the 1-month component (Equation 27)
ASRES_LONG_WEIGHT = 0.40  # weight on the longer-horizon component (Equation 28)
ASRES_BURN_IN_YEARS = 2  # initial burn-in for the expanding window (Section 7)
DECIMALIZATION_DATE = "2001-04-01"  # pre/post decimalization split (Table 9)
SIZE_GROUPS = {"Micro": (0.01, 0.20), "Small": (0.20, 0.50), "Large": (0.50, 0.80), "Mega": (0.80, 1.00)}  # NYSE ME percentile groups (Table 9)
