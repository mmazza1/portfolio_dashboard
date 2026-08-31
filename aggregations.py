"""Shared data-aggregation helpers: weighted breakdowns by sector/region/
country/continent/asset-type/currency, donut-chart data shaping, and the
concentration/overlap/drift flag logic. Used by more than one view
(Allocation, Rebalance, plus app.py's own sidebar/banner code), so this is
where they live rather than inside any one view module -- pure functions
over already-enriched DataFrames, no Streamlit rendering of their own.
"""

import pandas as pd

from formatting import format_pct


def _weighted_breakdown(non_cash: pd.DataFrame, weights_col: str) -> pd.DataFrame:
    """Weighted sum of value_eur per category, with a synthetic "Unresolved"
    bucket absorbing whatever fraction of each row's value isn't classified.

    A row's weights can come in slightly over 100% as well as under --
    justETF's region percentages and Yahoo's sector_weightings both round at
    the source, so summing them doesn't always land on exactly 1.0.
    Undershoot is a real signal (this much of the holding genuinely isn't
    classified) and stays in Unresolved, same as before. Overshoot has no
    such meaning -- weights can't legitimately exceed 100% of a holding --
    so rather than silently dropping the excess (the old behavior, which
    shrank the reported total below the portfolio's real value, by a
    different amount depending on which weighting source overshot) or
    letting Unresolved go negative (which a pie/donut chart can't render as
    a sensible slice), an overshooting row's weights are rescaled down to
    sum to exactly 1 before being distributed. Either way, every row's
    contribution sums to exactly its own value_eur, so the breakdown's
    total always exactly equals sum(value_eur) for the input rows.
    """
    totals: dict[str, float] = {}
    for row in non_cash.itertuples():
        weights = getattr(row, weights_col) or {}
        weight_sum = sum(weights.values())
        if weight_sum > 1:
            weights = {bucket: weight / weight_sum for bucket, weight in weights.items()}
            weight_sum = 1.0
        for bucket, weight in weights.items():
            totals[bucket] = totals.get(bucket, 0.0) + row.value_eur * weight
        leftover = 1 - weight_sum
        if leftover > 1e-9:
            totals["Unresolved"] = totals.get("Unresolved", 0.0) + row.value_eur * leftover
    return pd.DataFrame(totals.items(), columns=["bucket", "value_eur"]).sort_values("value_eur", ascending=False)


def sector_breakdown(non_cash: pd.DataFrame) -> pd.DataFrame:
    return _weighted_breakdown(non_cash, "enrich_sector_weights")


def region_breakdown(non_cash: pd.DataFrame) -> pd.DataFrame:
    return _weighted_breakdown(non_cash, "enrich_region_weights")


# Countries not listed here (Canada, Mexico, Australia, all of Africa, "Other"
# and "Unresolved"/"Unknown" from the enrichment pipeline itself) fall into
# the "Other" bucket. The Middle East is grouped under Asia geographically.
_CONTINENT_MAP = {
    "United States": "United States",
    # South America
    "Argentina": "South America", "Bolivia": "South America", "Brazil": "South America",
    "Chile": "South America", "Colombia": "South America", "Ecuador": "South America",
    "Guyana": "South America", "Paraguay": "South America", "Peru": "South America",
    "Suriname": "South America", "Uruguay": "South America", "Venezuela": "South America",
    # Europe
    "Albania": "Europe", "Andorra": "Europe", "Austria": "Europe", "Belarus": "Europe",
    "Belgium": "Europe", "Bosnia and Herzegovina": "Europe", "Bulgaria": "Europe",
    "Croatia": "Europe", "Cyprus": "Europe", "Czech Republic": "Europe", "Czechia": "Europe",
    "Denmark": "Europe", "Estonia": "Europe", "Finland": "Europe", "France": "Europe",
    "Germany": "Europe", "Greece": "Europe", "Hungary": "Europe", "Iceland": "Europe",
    "Ireland": "Europe", "Italy": "Europe", "Kosovo": "Europe", "Latvia": "Europe",
    "Liechtenstein": "Europe", "Lithuania": "Europe", "Luxembourg": "Europe", "Malta": "Europe",
    "Moldova": "Europe", "Monaco": "Europe", "Montenegro": "Europe", "Netherlands": "Europe",
    "North Macedonia": "Europe", "Norway": "Europe", "Poland": "Europe", "Portugal": "Europe",
    "Romania": "Europe", "Russia": "Europe", "San Marino": "Europe", "Serbia": "Europe",
    "Slovakia": "Europe", "Slovenia": "Europe", "Spain": "Europe", "Sweden": "Europe",
    "Switzerland": "Europe", "Ukraine": "Europe", "United Kingdom": "Europe",
    "Vatican City": "Europe",
    # Asia (including the Middle East)
    "Afghanistan": "Asia", "Armenia": "Asia", "Azerbaijan": "Asia", "Bahrain": "Asia",
    "Bangladesh": "Asia", "Bhutan": "Asia", "Brunei": "Asia", "Cambodia": "Asia",
    "China": "Asia", "Georgia": "Asia", "Hong Kong": "Asia", "India": "Asia",
    "Indonesia": "Asia", "Iran": "Asia", "Iraq": "Asia", "Israel": "Asia", "Japan": "Asia",
    "Jordan": "Asia", "Kazakhstan": "Asia", "Kuwait": "Asia", "Kyrgyzstan": "Asia",
    "Laos": "Asia", "Lebanon": "Asia", "Macau": "Asia", "Malaysia": "Asia",
    "Maldives": "Asia", "Mongolia": "Asia", "Myanmar": "Asia", "Nepal": "Asia",
    "North Korea": "Asia", "Oman": "Asia", "Pakistan": "Asia", "Palestine": "Asia",
    "Philippines": "Asia", "Qatar": "Asia", "Saudi Arabia": "Asia", "Singapore": "Asia",
    "South Korea": "Asia", "Sri Lanka": "Asia", "Syria": "Asia", "Taiwan": "Asia",
    "Tajikistan": "Asia", "Thailand": "Asia", "Timor-Leste": "Asia", "Turkey": "Asia",
    "Turkmenistan": "Asia", "United Arab Emirates": "Asia", "Uzbekistan": "Asia",
    "Vietnam": "Asia", "Yemen": "Asia",
}


def continent_breakdown(non_cash: pd.DataFrame) -> pd.DataFrame:
    # "Unresolved" (region_breakdown's own leftover bucket -- no region data
    # at all, e.g. a justETF overview with a genuinely empty countries list)
    # and "Unknown" (a justETF scrape that errored outright) both need to
    # stay their own buckets here, not fall through _CONTINENT_MAP's default
    # into "Other" -- that would silently conflate "we have no idea" with
    # "a real, specific country that just isn't one of our named continents"
    # (Canada, Mexico, Australia, ...), hiding a genuine data gap inside a
    # normal-looking bucket instead of surfacing it.
    by_country = region_breakdown(non_cash)
    totals: dict[str, float] = {}
    for _, r in by_country.iterrows():
        country = r["bucket"]
        bucket = country if country in ("Unresolved", "Unknown") else _CONTINENT_MAP.get(country, "Other")
        totals[bucket] = totals.get(bucket, 0.0) + r["value_eur"]
    return pd.DataFrame(totals.items(), columns=["bucket", "value_eur"]).sort_values("value_eur", ascending=False)


def continent_weights(weights: dict[str, float]) -> dict[str, float]:
    """Re-bucket one position's {country: weight} dict into continent buckets.
    Used for the rebalance calculator, where a per-country target would mean
    30+ number inputs -- continent-level keeps the target form usable.
    """
    result: dict[str, float] = {}
    for country, weight in (weights or {}).items():
        bucket = _CONTINENT_MAP.get(country, "Other")
        result[bucket] = result.get(bucket, 0.0) + weight
    return result


# A small, hand-picked set of buckets for the rebalance calculator's target
# form (same rationale as continent_weights: keeps the target form usable
# instead of 11 GICS sectors). Consumer Cyclical and Consumer Defensive
# merge into "Consumer Goods"; Energy and Utilities merge into one bucket.
# Everything else (Real Estate, Basic Materials, Industrials, Communication
# Services) falls into "Other".
_COMMON_SECTOR_MAP = {
    "Technology": "Technology",
    "Healthcare": "Healthcare",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Goods",
    "Consumer Defensive": "Consumer Goods",
    "Energy": "Energy + Utilities",
    "Utilities": "Energy + Utilities",
}


def common_sector_weights(weights: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for sector, weight in (weights or {}).items():
        bucket = _COMMON_SECTOR_MAP.get(sector, "Other")
        result[bucket] = result.get(bucket, 0.0) + weight
    return result


def round_percentages(values: pd.Series) -> pd.Series:
    """Round percentages to whole numbers so they still sum to 100, using the
    largest-remainder method. Rounding each value independently (e.g. round())
    can drift the total off by a point or two, which then falsely trips the
    "targets don't sum to 100%" check before the user has edited anything.
    """
    floors = values.astype(int)
    remainder = int(round(values.sum())) - int(floors.sum())
    order = (values - floors).sort_values(ascending=False).index
    result = floors.copy()
    for idx in order[:remainder]:
        result.loc[idx] += 1
    return result


def asset_type_breakdown(enriched: pd.DataFrame) -> pd.DataFrame:
    return (
        enriched.assign(bucket=enriched["enrich_asset_type"].fillna("Unresolved"))
        .groupby("bucket")["value_eur"]
        .sum()
        .reset_index()
        .sort_values("value_eur", ascending=False)
    )


def positions_breakdown(enriched: pd.DataFrame) -> pd.DataFrame:
    return enriched.assign(bucket=enriched["enrich_name"])[["bucket", "value_eur"]].sort_values(
        "value_eur", ascending=False
    )


def currency_breakdown(enriched: pd.DataFrame) -> pd.DataFrame:
    return (
        enriched.assign(bucket=enriched["currency"])
        .groupby("bucket")["value_eur"]
        .sum()
        .reset_index()
        .sort_values("value_eur", ascending=False)
    )


def _donut_data(breakdown: pd.DataFrame) -> list[dict]:
    return [{"label": row.bucket, "value": row.value_eur} for row in breakdown.itertuples()]


# Only the views we actually have data for: no per-security industry
# classification (GICS sector only) and no market cap pulled in enrichment,
# so those two are left out rather than shown empty.
DONUT_VIEWS = ["Type", "Positions", "Sectors", "Regions", "Countries", "Currencies"]
DONUT_SCOPE_AWARE_VIEWS = {"Sectors", "Regions", "Countries"}


def donut_data_for_view(view: str, display_source: pd.DataFrame, sector_region_source: pd.DataFrame) -> list[dict]:
    """`display_source` is whatever scope Type/Positions/Currencies should
    reflect (the whole portfolio, or ETF-only when scoped); `sector_region_source`
    likewise for Sectors/Regions/Countries -- same split as concentration_flags.
    """
    if view == "Type":
        return _donut_data(asset_type_breakdown(display_source))
    if view == "Positions":
        return _donut_data(positions_breakdown(display_source))
    if view == "Sectors":
        return _donut_data(sector_breakdown(sector_region_source))
    if view == "Regions":
        return _donut_data(continent_breakdown(sector_region_source))
    if view == "Countries":
        return _donut_data(region_breakdown(sector_region_source))
    if view == "Currencies":
        return _donut_data(currency_breakdown(display_source))
    return []


def concentration_flags(
    position_source: pd.DataFrame,
    sector_region_source: pd.DataFrame,
    total_value: float,
    asset_threshold: float,
    sector_threshold: float,
    region_threshold: float,
) -> list[str]:
    """`position_source` is whatever set of rows the asset-level check should
    consider (e.g. whole portfolio minus ETFs, or ETFs only when scoped);
    `sector_region_source` likewise for the sector/region breakdowns.
    `total_value` is the denominator for all three checks, so it should match
    the scope those sources represent.
    """
    flags = []

    for row in position_source.itertuples():
        weight = row.value_eur / total_value
        if weight > asset_threshold:
            name = row.enrich_name or row.product
            flags.append(f"Position **{name}** is **{format_pct(weight)}** of your portfolio.")

    for label, breakdown_fn, threshold in (
        ("Sector", sector_breakdown, sector_threshold),
        ("Region", continent_breakdown, region_threshold),
    ):
        breakdown = breakdown_fn(sector_region_source)
        for _, r in breakdown.iterrows():
            weight = r["value_eur"] / total_value
            if weight > threshold:
                flags.append(f"{label} **{r['bucket']}** is **{format_pct(weight)}** of your portfolio.")

    return flags


def target_drift_flags(
    breakdown: pd.DataFrame, targets: dict[str, float], total_value: float, threshold: float
) -> list[tuple[str, float]]:
    """Returns (message, abs_drift) pairs so callers can pick a severity level from abs_drift."""
    actual = dict(zip(breakdown["bucket"], breakdown["value_eur"] / total_value))
    flags = []
    for bucket in sorted(set(actual) | set(targets)):
        actual_weight = actual.get(bucket, 0.0)
        target_weight = targets.get(bucket, 0.0)
        drift = actual_weight - target_weight
        if abs(drift) > threshold:
            direction = "above" if drift > 0 else "below"
            message = (
                f"**{bucket}** is **{format_pct(actual_weight)}** of your portfolio, "
                f"{format_pct(abs(drift))} {direction} its **{format_pct(target_weight)}** target."
            )
            flags.append((message, abs(drift)))
    return flags


def etf_overlap_flags(enriched: pd.DataFrame) -> list[str]:
    stocks = enriched[enriched["enrich_asset_type"] == "Stock"]
    stock_by_ticker = dict(zip(stocks["enrich_ticker"], stocks["enrich_name"]))
    etfs = enriched[enriched["enrich_asset_type"] == "ETF"]

    flags = []
    for row in etfs.itertuples():
        overlap = set(stock_by_ticker) & set(row.enrich_top_holdings or [])
        for ticker in sorted(overlap):
            flags.append(
                f"**{ticker}** ({stock_by_ticker[ticker]}) is in your **{row.enrich_name}** holding, "
                "and you also hold it directly."
            )
    return flags
