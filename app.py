import json
import os

import pandas as pd
import streamlit as st

import rebalance
from components.donut_chart import render_donut
from enrichment import enrich_positions
from formatting import format_eur, format_number, format_pct
from parser import parse_degiro_csv

# TESTING ONLY -- when True, skips the file uploader and loads DEV_DATA_PATH
# from disk instead, so local dev/testing doesn't need a re-upload every run.
# Once enrichment has run once, its result is cached to DEV_SNAPSHOT_PATH and
# reused on later runs too, skipping ticker resolution / price fetching /
# justETF scraping entirely -- that network round-trip, not the file upload,
# is what actually made repeated local testing slow. Use the sidebar's
# "Refresh dev snapshot" button after changing the CSV or the enrichment
# pipeline itself. MUST be set back to False before deploying the public
# version or sharing this app with anyone else -- it bypasses the upload
# flow entirely.
DEV_MODE = False
DEV_DATA_PATH = "dev_data/Portfolio.csv"
DEV_SNAPSHOT_PATH = "dev_data/enriched_snapshot.parquet"
# Columns holding Python dicts/lists with a different shape per row
# (sector_weights, region_weights, top_holdings). Parquet can't round-trip
# these natively: pyarrow infers one struct type unioning every key seen
# anywhere in the column, so a row whose dict was originally empty comes
# back with every key present and set to None -- silently corrupting every
# downstream weighted sum rather than raising. JSON-encoding them as plain
# strings before writing sidesteps that struct inference entirely.
_DEV_SNAPSHOT_JSON_COLUMNS = ["enrich_sector_weights", "enrich_region_weights", "enrich_top_holdings"]
# Every enrich_* field enrich_positions() currently produces. Checked against
# a loaded snapshot's columns so a schema change (a field added or renamed
# since the snapshot was built -- this bit while adding `ter`, hence this
# check existing at all) is treated as a cache miss and rebuilt automatically,
# instead of surfacing as a cryptic KeyError deep in the dashboard code with
# no obvious link back to "the cached snapshot is just stale."
_EXPECTED_ENRICH_COLUMNS = {
    "enrich_isin",
    "enrich_ticker",
    "enrich_name",
    "enrich_asset_type",
    "enrich_sector_weights",
    "enrich_region_weights",
    "enrich_top_holdings",
    "enrich_ter",
    "enrich_error",
}


def _save_dev_snapshot(enriched: pd.DataFrame, path: str) -> None:
    snapshot = enriched.copy()
    for col in _DEV_SNAPSHOT_JSON_COLUMNS:
        snapshot[col] = snapshot[col].apply(json.dumps)
    snapshot.to_parquet(path)


def _load_dev_snapshot(path: str) -> pd.DataFrame | None:
    """Returns None (a cache miss) rather than a mismatched dataframe if the
    snapshot predates the current enrichment schema."""
    snapshot = pd.read_parquet(path)
    if not _EXPECTED_ENRICH_COLUMNS.issubset(snapshot.columns):
        return None
    for col in _DEV_SNAPSHOT_JSON_COLUMNS:
        snapshot[col] = snapshot[col].apply(json.loads)
    return snapshot


st.set_page_config(page_title="Portfolio Diversification Dashboard", layout="wide")

# Fixed-order categorical palette (never cycled by rank), one step per mode --
# validated for adjacent-pair colorblind-safe separation. Light/dark are the
# same eight hues stepped for their respective surface, not separate palettes.
_CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def _chart_theme() -> dict:
    """Colors/ink for the current Streamlit theme. Falls back to dark (this
    app's default) when the theme can't be read, e.g. outside a live session.
    """
    theme_type = st.context.theme.get("type") or "dark"
    if theme_type == "light":
        return {"colors": _CATEGORICAL_LIGHT, "ink": "#0b0b0b", "muted": "#52514e"}
    return {"colors": _CATEGORICAL_DARK, "ink": "#ffffff", "muted": "#c3c2b7"}


@st.cache_data(show_spinner="Parsing CSV and enriching positions via Yahoo Finance (first run can take a minute)...")
def load_and_enrich(file_bytes: bytes) -> pd.DataFrame:
    positions = parse_degiro_csv(file_bytes)
    return enrich_positions(positions)


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


def _continent_weights(weights: dict[str, float]) -> dict[str, float]:
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
# form (same rationale as _continent_weights: keeps the target form usable
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


def _common_sector_weights(weights: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for sector, weight in (weights or {}).items():
        bucket = _COMMON_SECTOR_MAP.get(sector, "Other")
        result[bucket] = result.get(bucket, 0.0) + weight
    return result


def _round_percentages(values: pd.Series) -> pd.Series:
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
_DONUT_VIEWS = ["Type", "Positions", "Sectors", "Regions", "Countries", "Currencies"]
_DONUT_SCOPE_AWARE_VIEWS = {"Sectors", "Regions", "Countries"}


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


st.title("Portfolio Diversification Dashboard")

if DEV_MODE:
    st.sidebar.subheader("Dev mode")
    if st.sidebar.button("Refresh dev snapshot", help="Deletes the cached enriched snapshot and re-runs enrichment fresh."):
        if os.path.exists(DEV_SNAPSHOT_PATH):
            os.remove(DEV_SNAPSHOT_PATH)

    enriched = _load_dev_snapshot(DEV_SNAPSHOT_PATH) if os.path.exists(DEV_SNAPSHOT_PATH) else None
    if enriched is not None:
        st.info("Dev mode: loaded cached enriched snapshot, skipping enrichment.")
    else:
        st.info(f"Dev mode: loaded {DEV_DATA_PATH} from disk, running enrichment once to build a snapshot.")
        with open(DEV_DATA_PATH, "rb") as f:
            file_bytes = f.read()
        enriched = load_and_enrich(file_bytes)
        _save_dev_snapshot(enriched, DEV_SNAPSHOT_PATH)
else:
    uploaded_file = st.file_uploader("Upload your DEGIRO Portfolio.csv export", type="csv")
    if uploaded_file is None:
        st.info("Upload a DEGIRO Portfolio.csv export to get started.")
        st.stop()
    enriched = load_and_enrich(uploaded_file.getvalue())

st.sidebar.subheader("Concentration warning thresholds")
asset_threshold = (
    st.sidebar.slider(
        "Asset",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Flags any single non-ETF position that makes up more than this share of your portfolio. "
        "ETFs are excluded since they're inherently diversified.",
    )
    / 100
)
sector_threshold = (
    st.sidebar.slider(
        "Sector",
        min_value=5,
        max_value=50,
        value=25,
        step=5,
        help="Flags any sector that makes up more than this share of your portfolio.",
    )
    / 100
)
region_threshold = (
    st.sidebar.slider(
        "Region",
        min_value=5,
        max_value=50,
        value=40,
        step=5,
        help="Flags any region that makes up more than this share of your portfolio.",
    )
    / 100
)

enriched["enrich_name"] = enriched["enrich_name"].fillna(enriched["product"])
total_value = enriched["value_eur"].sum()
non_cash = enriched[~enriched["is_cash"]]
etf_only = enriched[enriched["enrich_asset_type"] == "ETF"]
cash_value = float(enriched.loc[enriched["is_cash"], "value_eur"].sum())

scope = "Whole portfolio"
if not etf_only.empty:
    scope = st.segmented_control(
        "Scope",
        ["Whole portfolio", "ETFs only"],
        default="Whole portfolio",
        required=True,
        key="scope",
        help="ETFs only scopes the sector and region charts to just your ETF holdings, renormalized to 100% of ETF value.",
    )
sector_region_source = etf_only if scope == "ETFs only" else non_cash
# Everything that isn't specifically sector/region/country-shaped (the
# Positions tab table, and the Type/Positions/Currencies donut views) scopes
# down to this instead, so "ETFs only" actually means ETFs only everywhere,
# not just on the three charts that used to respect it.
display_source = etf_only if scope == "ETFs only" else enriched
display_total_value = display_source["value_eur"].sum()
scope_suffix = " (ETFs only)" if scope == "ETFs only" else ""
# ETFs are inherently diversified, so a large ETF holding isn't the same kind
# of risk as a large single-stock position -- excluded from the asset check
# in whole-portfolio scope, and skipped entirely (no asset check at all) in
# ETFs-only scope, which only warns on sector/region concentration.
position_source = enriched.iloc[0:0] if scope == "ETFs only" else enriched[enriched["enrich_asset_type"] != "ETF"]
concentration_total_value = etf_only["value_eur"].sum() if scope == "ETFs only" else total_value

lookup_errors = enriched.loc[~enriched["is_cash"] & enriched["enrich_error"].notna(), ["product", "enrich_error"]]
if not lookup_errors.empty:
    with st.expander(f"{len(lookup_errors)} position(s) could not be fully resolved via Yahoo Finance"):
        for _, r in lookup_errors.iterrows():
            st.write(f"- **{r['product']}**: {r['enrich_error']}")

st.sidebar.subheader("Target allocation")
st.sidebar.caption("Defaults to your current weights. Edit to set your actual targets.")
target_dimension = st.sidebar.selectbox("Compare", ["Asset type", "Region"])
target_breakdown = asset_type_breakdown(enriched) if target_dimension == "Asset type" else region_breakdown(non_cash)
# Asset type buckets (including Cash) sum to the whole portfolio; region buckets
# only cover non-cash holdings, so they're compared against their own total
# rather than total_value, or they'd never be able to reach 100%.
target_total_value = target_breakdown["value_eur"].sum()
target_breakdown = target_breakdown.assign(
    current_pct=_round_percentages(target_breakdown["value_eur"] / target_total_value * 100)
)

target_table = st.sidebar.data_editor(
    target_breakdown[["bucket", "current_pct"]].rename(columns={"current_pct": "target_pct"}),
    key=f"target_editor_{target_dimension}",
    hide_index=True,
    column_config={
        "bucket": st.column_config.TextColumn("Bucket", disabled=True),
        "target_pct": st.column_config.NumberColumn("Target %", min_value=0, max_value=100, step=5),
    },
)
target_pct_sum = target_table["target_pct"].sum()
if abs(target_pct_sum - 100) > 0.01:
    st.sidebar.error(f"Targets sum to {format_number(target_pct_sum, 0)}%, not 100%.")

drift_threshold = (
    st.sidebar.slider(
        "Drift warning threshold",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Flags any bucket whose actual weight differs from its target by more than this.",
    )
    / 100
)
targets = dict(zip(target_table["bucket"], target_table["target_pct"] / 100))

flags = concentration_flags(
    position_source, sector_region_source, concentration_total_value, asset_threshold, sector_threshold, region_threshold
)
overlaps = etf_overlap_flags(enriched)
drift_flags = target_drift_flags(target_breakdown, targets, target_total_value, drift_threshold)

if flags or overlaps:
    banner_col1, banner_col2 = st.columns(2)
    with banner_col1:
        if flags:
            st.warning(
                f"**Concentration warning{scope_suffix}** (asset >{format_pct(asset_threshold)}, "
                f"sector >{format_pct(sector_threshold)}, region >{format_pct(region_threshold)}):\n\n"
                + "\n\n".join(f"- {f}" for f in flags)
            )
    with banner_col2:
        if overlaps:
            st.info("**ETF overlap:**\n\n" + "\n\n".join(f"- {f}" for f in overlaps))

if drift_flags:
    max_drift = max(drift for _, drift in drift_flags)
    banner_fn = st.error if max_drift > 0.15 else st.warning if max_drift > 0.10 else st.info
    banner_fn(
        f"**Target allocation drift** ({target_dimension}, above {format_pct(drift_threshold)}):\n\n"
        + "\n\n".join(f"- {message}" for message, _ in drift_flags)
    )

# Rebalance only makes sense scoped to ETFs (see the Rebalancing calculator
# section in PROJECT_PLAN.md), so it's only shown as a tab at all when the
# Scope toggle above is set to ETFs only.
tab_labels = ["Positions", "Allocation"]
if scope == "ETFs only":
    tab_labels.append("Rebalance")
tabs = st.tabs(tab_labels)
tab_positions, tab_allocation = tabs[0], tabs[1]
tab_rebalance = tabs[2] if scope == "ETFs only" else None

with tab_positions:
    table = display_source.assign(pct=display_source["value_eur"] / display_total_value)[
        ["enrich_name", "enrich_asset_type", "quantity", "value_eur", "pct", "enrich_isin", "enrich_ter"]
    ].rename(columns={"enrich_name": "Name", "enrich_asset_type": "Type", "quantity": "Quantity"})

    positions_page_size = 10
    # Read before the table is built (so this render reflects the toggle's
    # current state) even though the button itself is drawn after the table
    # below -- clicking it flips session_state and triggers a fresh rerun,
    # which reads the updated value here from the top, same result either way.
    show_all_positions = st.session_state.get("positions_show_all", False)
    # NumberColumn's built-in "euro"/"percent" formats are fixed US-style
    # grouping with no way to swap separators -- pre-formatting Value/% as
    # European-style strings (like formatting.py does everywhere else) fixed
    # the *display*, but broke sorting: a string column sorts lexicographically
    # ("10" before "9"), not numerically. "localized" keeps the column
    # genuinely numeric (so the built-in sort stays correct) and renders
    # using the *viewer's own browser locale* instead of a hardcoded one --
    # for a European-locale browser that's period-thousands/comma-decimal
    # already, just not a hardcoded guarantee the way formatting.py is
    # elsewhere. "localized" has no currency/percent symbol of its own, so
    # that's carried by the column label instead, and pct needs to be
    # pre-scaled to 0-100 here since (unlike the "percent" format) it
    # doesn't multiply by 100 itself.
    displayed_table = table.assign(pct=table["pct"] * 100)
    # printf-style, not "localized": "localized" has no way to add a
    # currency/percent symbol and defaults to showing up to 3 decimals
    # (Streamlit's own docs example is "1,234.567"), neither of which
    # this format string has that problem with -- but printf-style is
    # hardcoded to comma-thousands/period-decimal with no locale
    # awareness (Streamlit's own docs: "Use `,` for thousand
    # separators"), so these two cells specifically revert to that
    # convention rather than the European one used everywhere else in
    # the app. Still a genuinely numeric column either way, so sorting
    # stays correct.
    column_config = {
        "Quantity": st.column_config.NumberColumn("Quantity", width="small"),
        "value_eur": st.column_config.NumberColumn("Value", format="€%.2f"),
        "pct": st.column_config.NumberColumn("% of portfolio", format="%.2f%%"),
    }
    # TER and a link out to justETF only make sense once every row actually
    # is an ETF -- in whole-portfolio scope the table also has stocks/cash,
    # which have neither.
    if scope == "ETFs only":
        displayed_table = displayed_table.assign(
            justetf_url="https://www.justetf.com/en/etf-profile.html?isin=" + displayed_table["enrich_isin"]
        ).rename(columns={"enrich_ter": "TER"})
        # The icon sits in its own column immediately after Name, as close to
        # "attached to the name" as st.dataframe actually allows -- a single
        # cell can only be one column type (text, link, ...), not a plain-text
        # name plus a separately-clickable icon layered inside it, so this is
        # the closest legitimate approximation rather than a hack that
        # abuses URL fragments to fake two things in one cell.
        displayed_table = displayed_table[
            ["Name", "Type", "Quantity", "value_eur", "pct", "TER", "justetf_url"]
        ]
        column_config["justetf_url"] = st.column_config.LinkColumn(
            "More info", display_text="ⓘ", width="small", help="Open this ETF's justETF page for more information"
        )
        column_config["TER"] = st.column_config.NumberColumn("TER", format="%.2f%%", width="small")
    else:
        displayed_table = displayed_table.drop(columns=["enrich_isin", "enrich_ter"])

    # The whole table is always sent to the widget, even when collapsed --
    # st.dataframe's column-header sort is a purely client-side feature, so
    # if only the first 10 (unsorted) rows were ever sent, sorting could only
    # ever reorder that arbitrary slice, never bring a genuinely-larger row
    # up from beyond the cutoff. `height` is what actually implements
    # "collapsed": short enough to show ~10 rows, tall enough to show every
    # row with no inner scroll when expanded. Sorting while collapsed now
    # scrolls to see rows past 10 -- the trade-off for sort being correct
    # instead of just being correct-looking on whatever happened to be sent.
    visible_rows = len(table) if show_all_positions else min(len(table), positions_page_size)
    st.dataframe(
        displayed_table,
        width="stretch",
        hide_index=True,
        height=38 + 35 * visible_rows + 3,
        column_config=column_config,
    )

    if len(table) > positions_page_size:
        button_label = "Show less" if show_all_positions else "Show more"
        _, button_col = st.columns([7, 1])
        with button_col:
            # The table above was already built from the pre-click value of
            # show_all_positions (it has to be, since this button renders
            # below it) -- updating session_state alone doesn't trigger a
            # second rerun on its own, so without an explicit st.rerun()
            # here the table only catches up on the *next* interaction,
            # meaning every click looks like it needs to be pressed twice.
            if st.button(button_label, width=110):
                st.session_state["positions_show_all"] = not show_all_positions
                st.rerun()

with tab_allocation:
    chart_theme = _chart_theme()
    available_views = [v for v in _DONUT_VIEWS if v not in _DONUT_SCOPE_AWARE_VIEWS or not sector_region_source.empty]
    donut_view = st.segmented_control(
        "View",
        available_views,
        default=available_views[0],
        required=True,
        key="donut_view",
        label_visibility="collapsed",
    )

    data = donut_data_for_view(donut_view, display_source, sector_region_source)
    if not data:
        st.info("No data to show for this view.")
    else:
        center_label = f"{donut_view}{scope_suffix}" if donut_view in _DONUT_SCOPE_AWARE_VIEWS else donut_view
        render_donut(center_label, data, colors=chart_theme["colors"], ink=chart_theme["ink"], muted=chart_theme["muted"])
        if donut_view in ("Regions", "Countries"):
            st.caption(
                "Region for stocks is an approximation (Yahoo Finance HQ country); "
                "for ETFs it's a real country look-through scraped from justETF."
            )

if tab_rebalance is not None:
    with tab_rebalance:
        if etf_only.empty:
            st.info("No ETF holdings to rebalance.")
        else:
            st.metric("Currently uninvested", format_eur(cash_value))
            rebalance_dimension = st.segmented_control(
                "Target dimension",
                ["Sector", "Region"],
                default="Sector",
                required=True,
                key="rebalance_dimension",
            )
            weights_col = "enrich_sector_weights" if rebalance_dimension == "Sector" else "enrich_region_weights"
            rebalance_source = etf_only
            if rebalance_dimension == "Sector":
                rebalance_source = etf_only.assign(
                    enrich_sector_weights=etf_only["enrich_sector_weights"].apply(_common_sector_weights)
                )
            else:
                rebalance_source = etf_only.assign(
                    enrich_region_weights=etf_only["enrich_region_weights"].apply(_continent_weights)
                )
            matrix, values = rebalance.build_weight_matrix(rebalance_source, weights_col)
            prices, shares = rebalance.build_trade_inputs(rebalance_source)
            raw_pct = pd.Series(matrix.values.T @ values.values / values.sum() * 100, index=matrix.columns)
            default_pct = _round_percentages(raw_pct)

            with st.form("rebalance_target_form"):
                st.caption("Set your target mix, then calculate. Defaults to your current ETF-only weights.")
                target_pct = {}
                input_cols = st.columns(min(4, len(matrix.columns)))
                for i, category in enumerate(matrix.columns):
                    with input_cols[i % len(input_cols)]:
                        target_pct[category] = st.number_input(
                            category,
                            min_value=0,
                            max_value=100,
                            value=int(default_pct[category]),
                            step=5,
                            key=f"rebalance_target_{rebalance_dimension}_{category}",
                        )
                total_pct = sum(target_pct.values())
                if total_pct == 100:
                    st.caption(f"Total: {format_number(total_pct, 0)}%")
                else:
                    st.caption(f"Total: {format_number(total_pct, 0)}% -- doesn't sum to 100%")
                additional_cash = st.number_input(
                    "Additional cash to invest",
                    min_value=0.0,
                    value=0.0,
                    step=50.0,
                    width=220,
                    help=(
                        "Only used by Buy-to-target. Combined with your currently uninvested cash into a "
                        "budget -- spent optimally, not maximally: money that wouldn't improve the fit is "
                        "left unspent. Leave at 0 to use your existing cash only (with a fallback suggestion "
                        "if that's too little to meaningfully help)."
                    ),
                )
                submitted = st.form_submit_button("Calculate")

            calculated_key = f"rebalance_calculated_{rebalance_dimension}"
            if submitted:
                st.session_state[calculated_key] = True

            if st.session_state.get(calculated_key, False):
                target = {category: pct / 100 for category, pct in target_pct.items()}
                buy_result = rebalance.buy_to_target(matrix, values, target, prices, cash_value, additional_cash)
                quick_result = rebalance.quick_rebalance(matrix, values, target, prices, shares)
                suggestions = rebalance.new_etf_suggestions(buy_result["unreachable"], rebalance_dimension.lower())

                name_by_ticker = etf_only.set_index("enrich_ticker")["enrich_name"]

                mode = st.segmented_control(
                    "View",
                    ["Buy-to-target", "Quick rebalance", "New ETF suggestions"],
                    default="Buy-to-target",
                    required=True,
                    key="rebalance_mode",
                )

                def _trade_table(shares_amounts: pd.Series, eur_amounts: pd.Series, eur_label: str) -> pd.DataFrame:
                    # Numeric, not pre-formatted: a pre-formatted string column
                    # sorts lexicographically ("10" before "9"), not numerically
                    # -- see the positions table for the same fix and why.
                    # Rendered via NumberColumn(format="localized") at each call
                    # site instead, which keeps the column sortable and renders
                    # using the viewer's own browser locale.
                    return pd.DataFrame(
                        {
                            "Name": [name_by_ticker[t] for t in shares_amounts.index],
                            "Shares": shares_amounts.values,
                            eur_label: eur_amounts.reindex(shares_amounts.index).values,
                        }
                    )

                if mode == "Buy-to-target":
                    if buy_result["unreachable"]:
                        st.warning(
                            "Target includes categories no held ETF provides at all: "
                            + ", ".join(f"**{c}**" for c in buy_result["unreachable"])
                            + ". Buying more of your current ETFs can't close this gap -- see New ETF suggestions."
                        )
                    if buy_result["used_fallback_cap"]:
                        estimate = buy_result["unconstrained_estimate"]
                        estimate_text = (
                            f"at least {format_eur(estimate, 0)}"
                            if buy_result["estimate_capped"]
                            else f"about {format_eur(estimate, 0)}"
                        )
                        st.info(
                            f"Your uninvested cash ({format_eur(cash_value)}) is too little to meaningfully close this gap, "
                            f"so this suggestion instead allows up to {format_eur(buy_result['budget_used'])} as an example "
                            f"budget -- it actually spends {format_eur(float(buy_result['buy_eur'].sum()))} of that, only "
                            "what improves the fit. Fully reaching this target would require "
                            f"{estimate_text} in new purchases. Enter additional cash above and recalculate to use "
                            "a specific amount instead."
                        )
                    else:
                        st.caption(
                            f"Budget: {format_eur(buy_result['budget_used'])} "
                            f"(spent {format_eur(float(buy_result['buy_eur'].sum()))} -- only what improves the fit)."
                        )
                    if buy_result["buy_shares"].empty:
                        st.info("Your current holdings are already at (or above) target -- no purchases needed.")
                    else:
                        st.dataframe(
                            _trade_table(buy_result["buy_shares"], buy_result["buy_eur"], "Buy (EUR)"),
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                                "Buy (EUR)": st.column_config.NumberColumn("Buy (EUR)", format="localized"),
                            },
                        )

                elif mode == "Quick rebalance":
                    if not quick_result["converged"]:
                        st.warning("The solver didn't find an optimal whole-share solution -- treat this result as approximate.")
                    if quick_result["unreachable"]:
                        st.warning(
                            "Target includes categories no held ETF provides at all: "
                            + ", ".join(f"**{c}**" for c in quick_result["unreachable"])
                            + ". No amount of buying or selling your current ETFs can close this gap -- "
                            "see New ETF suggestions."
                        )
                    if quick_result["delta_shares"].empty:
                        st.info("Your current holdings are already at target -- no trades needed.")
                    else:
                        share_trades = quick_result["delta_shares"]
                        eur_trades = quick_result["delta_eur"]
                        sell_mask = share_trades < 0
                        sells = _trade_table(-share_trades[sell_mask], -eur_trades[sell_mask], "Sell (EUR)")
                        buys = _trade_table(share_trades[~sell_mask], eur_trades[~sell_mask], "Buy (EUR)")
                        if not sells.empty:
                            st.write("**Sell**")
                            st.dataframe(
                                sells,
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                                    "Sell (EUR)": st.column_config.NumberColumn("Sell (EUR)", format="localized"),
                                },
                            )
                        if not buys.empty:
                            st.write("**Buy**")
                            st.dataframe(
                                buys,
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                                    "Buy (EUR)": st.column_config.NumberColumn("Buy (EUR)", format="localized"),
                                },
                            )
                        st.caption(
                            "Net cash flow is kept as close to zero as achievable in whole shares -- "
                            "not always exactly zero."
                        )

                else:
                    if suggestions:
                        st.warning("\n\n".join(f"- {s}" for s in suggestions))
                    else:
                        st.info("Your current ETF holdings can already reach this target -- no new fund categories needed.")
