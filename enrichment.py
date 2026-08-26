"""Resolve DEGIRO positions (by ISIN) to Yahoo Finance ticker, asset type,
sector, region, and (for ETFs) top holdings.

Sector is always expressed as a `sector_weights` dict mapping a GICS-style
sector name to a weight (a stock is `{sector: 1.0}`; an ETF is its real
look-through breakdown). This lets the dashboard sum weighted sector exposure
the same way regardless of asset type. ETF sector_weightings keys come back
snake_cased from Yahoo (e.g. "consumer_cyclical") and are remapped to match
the Title Case strings Yahoo uses for a stock's `sector` field, so the two
sources join on the same key.
Region is inherently an approximation: no free API cleanly answers "where
does this company's business actually happen." A stock's HQ country from
Yahoo is often just its legal domicile (Netherlands, Ireland, Luxembourg,
Bermuda, Cayman Islands, Switzerland are common tax/incorporation shells),
which can be unrelated to where it trades or operates. For those domiciles,
we fall back to the region of its primary listing exchange instead. Genuine
edge cases (a resolvable ticker with a still-misleading region, or an ISIN
Yahoo's search can't find at all) can be corrected by hand in overrides.csv.
"""

import csv
import os

import pandas as pd
import yfinance as yf

_OVERRIDES_PATH = os.path.join(os.path.dirname(__file__), "overrides.csv")

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

# Ordered most-specific-first: an ETF's proxy region is the first keyword
# match found in its fund name/category text.
_REGION_KEYWORDS = [
    ("Emerging Markets", ["emerging market", "em imi", "msci em"]),
    ("Japan", ["japan", "topix", "nikkei"]),
    ("China", ["china", "csi 300", "hang seng"]),
    ("Europe", ["europe", "stoxx", "eurozone", "euro stoxx"]),
    ("Asia Pacific", ["asia", "pacific", "asean"]),
    ("United States", ["s&p 500", "russell", "nasdaq", "united states"]),
    ("Global", ["all-world", "all world", "acwi", "world"]),
]

# HQ countries that are common tax/incorporation domiciles rather than a
# meaningful signal of where a business actually operates.
_DOMICILE_ONLY_COUNTRIES = {
    "Netherlands",
    "Ireland",
    "Luxembourg",
    "Bermuda",
    "Cayman Islands",
    "Switzerland",
}

# Yahoo's short exchange code (quote["exchange"] / info["exchange"]) to a
# region label, used as the domicile-only fallback above.
_EXCHANGE_REGION_MAP = {
    "NMS": "United States",
    "NYQ": "United States",
    "NGM": "United States",
    "NCM": "United States",
    "PCX": "United States",
    "ASE": "United States",
    "BTS": "United States",
    "LSE": "United Kingdom",
    "GER": "Germany",
    "FRA": "Germany",
    "STU": "Germany",
    "DUS": "Germany",
    "BER": "Germany",
    "MUN": "Germany",
    "HAM": "Germany",
    "AMS": "Netherlands",
    "PAR": "France",
    "MIL": "Italy",
    "MCE": "Spain",
    "EBS": "Switzerland",
    "VTX": "Switzerland",
    "STO": "Sweden",
    "CPH": "Denmark",
    "OSL": "Norway",
    "HEL": "Finland",
    "BRU": "Belgium",
    "LIS": "Portugal",
    "VIE": "Austria",
    "ISE": "Ireland",
    "WSE": "Poland",
    "ATH": "Greece",
    "TOR": "Canada",
    "TSX": "Canada",
    "TSXV": "Canada",
    "MEX": "Mexico",
    "SAO": "Brazil",
    "ASX": "Australia",
    "NZE": "New Zealand",
    "HKG": "Hong Kong",
    "SHH": "China",
    "SHZ": "China",
    "TAI": "Taiwan",
    "TWO": "Taiwan",
    "JPX": "Japan",
    "TYO": "Japan",
    "KSC": "South Korea",
    "KOE": "South Korea",
    "SES": "Singapore",
    "SGX": "Singapore",
    "NSI": "India",
    "BSE": "India",
    "JNB": "South Africa",
    "TLV": "Israel",
    "IST": "Turkey",
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


def _resolve_quote(isin: str) -> dict | None:
    query = OVERRIDES.get(isin, {}).get("ticker") or isin
    try:
        quotes = yf.Search(query, max_results=8).quotes
    except Exception:
        return None
    return quotes[0] if quotes else None


def _classify_region_from_text(text: str) -> str:
    text = (text or "").lower()
    for region, keywords in _REGION_KEYWORDS:
        if any(kw in text for kw in keywords):
            return region
    return "Unclassified"


def _enrich_stock(ticker: str, exchange: str | None) -> dict:
    info = yf.Ticker(ticker).info
    sector = info.get("sector")

    region = info.get("country")
    if region in _DOMICILE_ONLY_COUNTRIES:
        region = _EXCHANGE_REGION_MAP.get(exchange, region)

    return {
        "sector_weights": {sector: 1.0} if sector else {},
        "region": region,
        "top_holdings": [],
    }


def _enrich_etf(ticker: str, fallback_name: str) -> dict:
    fund = yf.Ticker(ticker).funds_data

    sector_weights = {}
    for key, weight in (fund.sector_weightings or {}).items():
        sector_weights[_SECTOR_KEY_MAP.get(key, key)] = weight

    overview = fund.fund_overview or {}
    region_text = " ".join(filter(None, [fallback_name, overview.get("categoryName")]))
    region = _classify_region_from_text(region_text)

    top_holdings = list(fund.top_holdings.index) if fund.top_holdings is not None else []

    return {
        "sector_weights": sector_weights,
        "region": region,
        "top_holdings": top_holdings,
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
        "region": None,
        "top_holdings": [],
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
            result.update(_enrich_etf(ticker, name))
        else:
            result.update(_enrich_stock(ticker, quote.get("exchange")))
    except Exception as exc:
        result["error"] = f"Resolved to {ticker} but enrichment lookup failed: {exc}"

    region_override = OVERRIDES.get(isin, {}).get("region")
    if region_override:
        result["region"] = region_override

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
                    "region": None,
                    "top_holdings": [],
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

    cols = ["product", "isin", "enrich_ticker", "enrich_asset_type", "enrich_region", "enrich_error"]
    print(enriched[cols].to_string())
    print()
    for row in enriched.itertuples():
        if row.enrich_sector_weights:
            print(f"{row.enrich_ticker}: {row.enrich_sector_weights}")
    print()
    for row in enriched.itertuples():
        if row.enrich_top_holdings:
            print(f"{row.enrich_ticker} top holdings: {row.enrich_top_holdings}")
