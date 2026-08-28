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
"""

import csv
import os

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


def _resolve_quote(isin: str) -> dict | None:
    query = OVERRIDES.get(isin, {}).get("ticker") or isin
    try:
        quotes = yf.Search(query, max_results=8).quotes
    except Exception:
        return None
    return quotes[0] if quotes else None


def _enrich_stock(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    sector = info.get("sector")
    region = info.get("country")

    return {
        "sector_weights": {sector: 1.0} if sector else {},
        "region_weights": {region: 1.0} if region else {},
        "top_holdings": [],
    }


def _enrich_etf(ticker: str, isin: str) -> dict:
    fund = yf.Ticker(ticker).funds_data

    sector_weights = {}
    for key, weight in (fund.sector_weightings or {}).items():
        sector_weights[_SECTOR_KEY_MAP.get(key, key)] = weight

    top_holdings = list(fund.top_holdings.index) if fund.top_holdings is not None else []
    region_weights, ter = _resolve_etf_overview(isin)

    return {
        "sector_weights": sector_weights,
        "region_weights": region_weights,
        "top_holdings": top_holdings,
        "ter": ter,
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
