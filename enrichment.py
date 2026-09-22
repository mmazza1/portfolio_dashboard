"""Resolve DEGIRO positions (by ISIN) to Yahoo Finance ticker, asset type,
sector, region, and (for ETFs) top holdings and TER.

Both sector and region are expressed as a `_weights` dict mapping a bucket
name to a weight (a stock is `{bucket: 1.0}`; an ETF is its real look-through
breakdown). This lets the dashboard sum weighted exposure the same way
regardless of asset type. ETF sector_weightings keys come back snake_cased
from Yahoo (e.g. "consumer_cyclical") and are remapped to match the Title
Case strings Yahoo uses for a stock's `sector` field, so the two sources join
on the same key.

Region is inherently an approximation for stocks: no free API cleanly
answers "where does this company's business actually happen," and a stock's
HQ country from Yahoo is often just its legal domicile (Nebius is HQ'd in
Amsterdam but isn't a Dutch business). This is an accepted limitation, rare
enough to fix by hand via overrides.csv rather than infer generically. ETFs
don't have this problem: their real country look-through is scraped from
justETF (keyed by ISIN, via the unofficial `justetf-scraping` package) since
Yahoo Finance has no country/region weighting field for funds. Results are
cached in `fund_regions.csv`, scraped once per ISIN, ever, and reused on
every later run. A scrape failure is marked `{"Unknown": 1.0}` rather than
crashing the pipeline, and is not cached, so it's retried on the next run.

TER (total expense ratio) comes from the same justETF call, as a plain
percentage number (0.2 meaning 0.20%, not a 0-1 fraction), cached separately
in `fund_ter.csv`. The two caches are checked independently rather than
gated on one flag, so an ISIN whose region was cached before TER tracking
existed still gets backfilled on the next run instead of permanently
missing it.

Dividend yield is computed from Yahoo's per-payment `.dividends` history
(trailing 12 months, summed, divided by the current price), not from
Yahoo's own `dividendYield`/`dividendRate` `.info` fields -- those came
back `None` for a real distributing ETF (VWRL.AS) in a live test, so
they're unreliable for ETFs specifically even though they work for stocks.
Deriving it the same way for both asset types keeps the two sources
comparable, same rationale as region/sector already being pulled through
one shared shape. `None` means the lookup itself failed; `0.0` is a
genuine result (no dividends in the last year -- e.g. an accumulating ETF
that reinvests internally, or a non-dividend-paying stock), not an error,
so the two aren't conflated.
"""

import csv
import os
import time

import justetf_scraping
import pandas as pd
import yfinance as yf

_OVERRIDES_PATH = os.path.join(os.path.dirname(__file__), "overrides.csv")
_FUND_REGIONS_PATH = os.path.join(os.path.dirname(__file__), "fund_regions.csv")
_FUND_REGIONS_FIELDS = ["isin", "country", "percentage"]
_FUND_TER_PATH = os.path.join(os.path.dirname(__file__), "fund_ter.csv")
_FUND_TER_FIELDS = ["isin", "ter"]

ASSET_TYPE_LABELS = {
    "EQUITY": "Stock",
    "ETF": "ETF",
    "MUTUALFUND": "Fund",
    "CRYPTOCURRENCY": "Crypto",
    "CURRENCY": "Currency",
    "FUTURE": "Future",
}

_SECTOR_KEY_MAP = {
    "realestate": "Real Estate",
    "consumer_cyclical": "Consumer Cyclical",
    "basic_materials": "Basic Materials",
    "consumer_defensive": "Consumer Defensive",
    "technology": "Technology",
    "communication_services": "Communication Services",
    "financial_services": "Financial Services",
    "utilities": "Utilities",
    "industrials": "Industrials",
    "energy": "Energy",
    "healthcare": "Healthcare",
}

# justETF's own sector taxonomy (ICB-based), used only as a fallback when
# Yahoo's `funds_data.sector_weightings` comes back empty -- see
# `_justetf_sector_weights`. Different vocabulary than Yahoo's GICS-based
# one above, so this remaps justETF's names onto the same Title Case buckets
# a stock's own Yahoo-sourced `sector` field uses, the same reason
# `_SECTOR_KEY_MAP` exists for Yahoo's own snake_case ETF keys -- otherwise
# a fallback-sourced ETF wouldn't sum into the same sector buckets as
# everything else on the Allocation donut and concentration check. A few
# justETF categories fold into one GICS bucket where GICS doesn't split
# them as finely (its "Business Services" -> GICS's "Industrials", both
# "Consumer Cyclicals" and "Consumer Services" -> GICS's "Consumer
# Cyclical", which already covers retail/leisure). "Other" is deliberately
# left unmapped -- it passes through as its own bucket, same convention
# used elsewhere in this app (e.g. region continent bucketing).
_JUSTETF_SECTOR_MAP = {
    "Technology": "Technology",
    "Finance": "Financial Services",
    "Industrials": "Industrials",
    "Business Services": "Industrials",
    "Consumer Non-Cyclicals": "Consumer Defensive",
    "Healthcare": "Healthcare",
    "Non-Energy Materials": "Basic Materials",
    "Consumer Cyclicals": "Consumer Cyclical",
    "Consumer Services": "Consumer Cyclical",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Telecommunication": "Communication Services",
    "Real Estate": "Real Estate",
}

def _load_overrides() -> dict[str, dict]:
    if not os.path.exists(_OVERRIDES_PATH):
        return {}
    with open(_OVERRIDES_PATH, newline="", encoding="utf-8") as f:
        return {
            row["isin"].strip(): {
                "ticker": (row.get("ticker") or "").strip() or None,
                "region": (row.get("region") or "").strip() or None,
            }
            for row in csv.DictReader(f)
        }


OVERRIDES = _load_overrides()


def _load_fund_regions_cache() -> dict[str, dict[str, float]]:
    if not os.path.exists(_FUND_REGIONS_PATH):
        return {}
    cache: dict[str, dict[str, float]] = {}
    with open(_FUND_REGIONS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache.setdefault(row["isin"], {})[row["country"]] = float(row["percentage"]) / 100
    return cache


FUND_REGIONS_CACHE = _load_fund_regions_cache()


def _append_fund_regions_cache(isin: str, region_weights: dict[str, float]) -> None:
    file_exists = os.path.exists(_FUND_REGIONS_PATH)
    with open(_FUND_REGIONS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FUND_REGIONS_FIELDS)
        if not file_exists:
            writer.writeheader()
        for country, weight in region_weights.items():
            writer.writerow({"isin": isin, "country": country, "percentage": weight * 100})


def _load_fund_ter_cache() -> dict[str, float]:
    if not os.path.exists(_FUND_TER_PATH):
        return {}
    with open(_FUND_TER_PATH, newline="", encoding="utf-8") as f:
        return {row["isin"]: float(row["ter"]) for row in csv.DictReader(f) if row["ter"]}


FUND_TER_CACHE = _load_fund_ter_cache()


def _append_fund_ter_cache(isin: str, ter: float) -> None:
    file_exists = os.path.exists(_FUND_TER_PATH)
    with open(_FUND_TER_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FUND_TER_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"isin": isin, "ter": ter})


def _resolve_etf_overview(isin: str) -> tuple[dict[str, float], float | None]:
    """Region weights and TER (total expense ratio, as a plain percentage
    number -- justETF's own `ter` field is e.g. 0.2 meaning 0.20%, not a
    0-1 fraction) for one ETF, both scraped from justETF in the same call
    and cached separately in their own CSVs. Cached independently (not "if
    either is cached, skip") so an ISIN whose region was already cached from
    before TER was tracked still gets a one-time scrape to backfill TER,
    rather than silently missing it forever.
    """
    region_cached = isin in FUND_REGIONS_CACHE
    ter_cached = isin in FUND_TER_CACHE
    if region_cached and ter_cached:
        return FUND_REGIONS_CACHE[isin], FUND_TER_CACHE[isin]

    try:
        overview = justetf_scraping.get_etf_overview(isin)
        region_weights = {c["name"]: c["percentage"] / 100 for c in overview["countries"]}
        ter = overview.get("ter")
    except Exception:
        return FUND_REGIONS_CACHE.get(isin, {"Unknown": 1.0}), FUND_TER_CACHE.get(isin)

    if not region_cached:
        FUND_REGIONS_CACHE[isin] = region_weights
        _append_fund_regions_cache(isin, region_weights)
    if not ter_cached and ter is not None:
        FUND_TER_CACHE[isin] = ter
        _append_fund_ter_cache(isin, ter)

    return FUND_REGIONS_CACHE[isin], FUND_TER_CACHE.get(isin)


def _justetf_sector_weights(isin: str) -> dict[str, float]:
    """Fallback sector breakdown, from justETF, used only when Yahoo's own
    `funds_data.sector_weightings` comes back empty for an ETF -- confirmed
    to happen for every held ETF at once on a rate-limited/blocked cloud
    IP (see PROJECT_PLAN.md's Rebalancing calculator bug), which used to
    crash the Rebalance tab outright with no sector data available at all.
    Deliberately *not* cached to a CSV the way region/TER are (see
    `_resolve_etf_overview`) -- caching a fallback value would mean it
    stays stuck even after Yahoo recovers, defeating the point of
    preferring Yahoo whenever it's actually working; this just re-fetches
    from justETF live each time it's needed; a fresh justETF failure here
    too just means an empty dict, same as an unresolved Yahoo lookup.
    """
    try:
        overview = justetf_scraping.get_etf_overview(isin)
    except Exception:
        return {}
    weights: dict[str, float] = {}
    for entry in overview.get("sectors") or []:
        bucket = _JUSTETF_SECTOR_MAP.get(entry["name"], entry["name"])
        weights[bucket] = weights.get(bucket, 0.0) + entry["percentage"] / 100
    return weights


def _resolve_dividend_yield(ticker_obj: yf.Ticker) -> float | None:
    """Trailing-12-month dividend yield (a 0-1 fraction), derived from the
    per-payment history rather than Yahoo's own dividendYield field (see
    module docstring). Per-share dividend and price are in the same native
    currency, so this ratio is currency-agnostic -- the caller can multiply
    it straight by a position's EUR value without any FX conversion of the
    dividend amount itself.
    """
    try:
        dividends = ticker_obj.dividends
        if dividends.empty:
            return 0.0
        cutoff = pd.Timestamp.now(tz=dividends.index.tz) - pd.Timedelta(days=365)
        ttm_dividends = dividends[dividends.index >= cutoff].sum()
        if ttm_dividends <= 0:
            return 0.0
        last_price = ticker_obj.fast_info.last_price
        if not last_price:
            return None
        return ttm_dividends / last_price
    except Exception:
        return None


# Attempts and base delay (seconds, doubled each retry) for _resolve_quote's
# retry loop -- see its own docstring for why this exists at all.
_TICKER_RESOLVE_ATTEMPTS = 3
_TICKER_RESOLVE_RETRY_DELAY = 0.75


def _resolve_quote(isin: str) -> dict | None:
    """Resolve an ISIN to its best-match Yahoo Finance quote via yf.Search,
    retried a few times with a short backoff before giving up. Confirmed in
    real use: this call fails intermittently and inconsistently on
    Streamlit Community Cloud's shared IP range -- an ISIN that resolves
    every single time locally (e.g. AST SpaceMobile's US00217D1000) can
    come back with zero quotes on Cloud, then resolve fine again a minute
    later. Yahoo's rate-limit response for this endpoint has been observed
    to come back as a quietly empty result, not a raised error, so this
    retries on an empty result too, not just on an exception. Deliberately
    bounded (3 tries, doubling delay) rather than unbounded -- a genuinely
    unresolvable ISIN should still fail within a couple of seconds, not
    stall the whole enrichment run.
    """
    query = OVERRIDES.get(isin, {}).get("ticker") or isin
    for attempt in range(_TICKER_RESOLVE_ATTEMPTS):
        try:
            quotes = yf.Search(query, max_results=8).quotes
        except Exception:
            quotes = []
        if quotes:
            return quotes[0]
        if attempt < _TICKER_RESOLVE_ATTEMPTS - 1:
            time.sleep(_TICKER_RESOLVE_RETRY_DELAY * (2**attempt))
    return None


def _enrich_stock(ticker: str) -> dict:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    sector = info.get("sector")
    region = info.get("country")

    return {
        "sector_weights": {sector: 1.0} if sector else {},
        "region_weights": {region: 1.0} if region else {},
        "top_holdings": [],
        "dividend_yield": _resolve_dividend_yield(ticker_obj),
    }


def _enrich_etf(ticker: str, isin: str) -> dict:
    ticker_obj = yf.Ticker(ticker)
    fund = ticker_obj.funds_data

    sector_weights = {}
    for key, weight in (fund.sector_weightings or {}).items():
        sector_weights[_SECTOR_KEY_MAP.get(key, key)] = weight
    if not sector_weights:
        # Yahoo's own sector lookup came back empty -- fall back to
        # justETF's, remapped onto the same buckets (see
        # _JUSTETF_SECTOR_MAP). Only reached in this failure case, not on
        # every ETF, so the common (Yahoo-works) path pays no extra cost.
        sector_weights = _justetf_sector_weights(isin)

    top_holdings = list(fund.top_holdings.index) if fund.top_holdings is not None else []
    region_weights, ter = _resolve_etf_overview(isin)

    return {
        "sector_weights": sector_weights,
        "region_weights": region_weights,
        "top_holdings": top_holdings,
        "ter": ter,
        "dividend_yield": _resolve_dividend_yield(ticker_obj),
    }


def enrich_isin(isin: str) -> dict:
    """Resolve one ISIN via Yahoo Finance. Never raises; failures are
    reported through the `error` field so a bad lookup doesn't stop the run.
    """
    result = {
        "isin": isin,
        "ticker": None,
        "name": None,
        "asset_type": None,
        "sector_weights": {},
        "region_weights": {},
        "top_holdings": [],
        "ter": None,
        "dividend_yield": None,
        "error": None,
    }

    quote = _resolve_quote(isin)
    if quote is None:
        result["error"] = "Could not resolve ISIN to a Yahoo Finance ticker"
        return result

    ticker = quote["symbol"]
    quote_type = quote.get("quoteType", "")
    name = quote.get("longname") or quote.get("shortname")
    result.update(
        ticker=ticker,
        name=name,
        asset_type=ASSET_TYPE_LABELS.get(quote_type, quote_type or None),
    )

    try:
        if quote_type == "ETF":
            result.update(_enrich_etf(ticker, isin))
        else:
            result.update(_enrich_stock(ticker))
    except Exception as exc:
        result["error"] = f"Resolved to {ticker} but enrichment lookup failed: {exc}"

    region_override = OVERRIDES.get(isin, {}).get("region")
    if region_override:
        result["region_weights"] = {region_override: 1.0}

    return result


def enrich_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """Enrich a parser.parse_degiro_csv() DataFrame with ticker/asset type/
    sector/region/top holdings, one Yahoo Finance lookup per unique ISIN.
    Cash rows are labeled directly, without a lookup.
    """
    cache: dict[str, dict] = {}
    records = []

    for row in positions.itertuples(index=False):
        if row.is_cash:
            records.append(
                {
                    "isin": None,
                    "ticker": None,
                    "name": row.product,
                    "asset_type": "Cash",
                    "sector_weights": {},
                    "region_weights": {},
                    "top_holdings": [],
                    "ter": None,
                    "dividend_yield": None,
                    "error": None,
                }
            )
            continue

        if row.isin not in cache:
            cache[row.isin] = enrich_isin(row.isin)
        records.append(cache[row.isin])

    enrichment_df = pd.DataFrame(records).add_prefix("enrich_")
    return pd.concat([positions.reset_index(drop=True), enrichment_df], axis=1)


if __name__ == "__main__":
    from parser import parse_degiro_csv

    positions = parse_degiro_csv(r"C:\Users\Matteo\Downloads\Portfolio.csv")
    enriched = enrich_positions(positions)

    cols = ["product", "isin", "enrich_ticker", "enrich_asset_type", "enrich_error"]
    print(enriched[cols].to_string())
    print()
    for row in enriched.itertuples():
        if row.enrich_sector_weights:
            print(f"{row.enrich_ticker}: {row.enrich_sector_weights}")
    print()
    for row in enriched.itertuples():
        if row.enrich_region_weights:
            print(f"{row.enrich_ticker}: {row.enrich_region_weights}")
    print()
    for row in enriched.itertuples():
        if row.enrich_top_holdings:
            print(f"{row.enrich_ticker} top holdings: {row.enrich_top_holdings}")
