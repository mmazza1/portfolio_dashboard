import pandas as pd
import plotly.express as px
import streamlit as st

import rebalance
from enrichment import enrich_positions
from parser import parse_degiro_csv

st.set_page_config(page_title="Portfolio Diversification Dashboard", layout="wide")


@st.cache_data(show_spinner="Parsing CSV and enriching positions via Yahoo Finance (first run can take a minute)...")
def load_and_enrich(file_bytes: bytes) -> pd.DataFrame:
    positions = parse_degiro_csv(file_bytes)
    return enrich_positions(positions)


def _weighted_breakdown(non_cash: pd.DataFrame, weights_col: str) -> pd.DataFrame:
    totals: dict[str, float] = {}
    for row in non_cash.itertuples():
        weights = getattr(row, weights_col) or {}
        for bucket, weight in weights.items():
            totals[bucket] = totals.get(bucket, 0.0) + row.value_eur * weight
        leftover = 1 - sum(weights.values())
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
    by_country = region_breakdown(non_cash)
    totals: dict[str, float] = {}
    for _, r in by_country.iterrows():
        bucket = _CONTINENT_MAP.get(r["bucket"], "Other")
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
            flags.append(f"Position **{name}** is **{weight:.0%}** of your portfolio.")

    for label, breakdown_fn, threshold in (
        ("Sector", sector_breakdown, sector_threshold),
        ("Region", continent_breakdown, region_threshold),
    ):
        breakdown = breakdown_fn(sector_region_source)
        for _, r in breakdown.iterrows():
            weight = r["value_eur"] / total_value
            if weight > threshold:
                flags.append(f"{label} **{r['bucket']}** is **{weight:.0%}** of your portfolio.")

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
                f"**{bucket}** is **{actual_weight:.0%}** of your portfolio, "
                f"{abs(drift):.0%} {direction} its **{target_weight:.0%}** target."
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

uploaded_file = st.file_uploader("Upload your DEGIRO Portfolio.csv export", type="csv")
if uploaded_file is None:
    st.info("Upload a DEGIRO Portfolio.csv export to get started.")
    st.stop()

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

enriched = load_and_enrich(uploaded_file.getvalue())
enriched["enrich_name"] = enriched["enrich_name"].fillna(enriched["product"])
total_value = enriched["value_eur"].sum()
non_cash = enriched[~enriched["is_cash"]]
etf_only = enriched[enriched["enrich_asset_type"] == "ETF"]

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
    st.sidebar.error(f"Targets sum to {target_pct_sum:g}%, not 100%.")

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
                f"**Concentration warning{scope_suffix}** (asset >{asset_threshold:.0%}, "
                f"sector >{sector_threshold:.0%}, region >{region_threshold:.0%}):\n\n"
                + "\n\n".join(f"- {f}" for f in flags)
            )
    with banner_col2:
        if overlaps:
            st.info("**ETF overlap:**\n\n" + "\n\n".join(f"- {f}" for f in overlaps))

if drift_flags:
    max_drift = max(drift for _, drift in drift_flags)
    banner_fn = st.error if max_drift > 0.15 else st.warning if max_drift > 0.10 else st.info
    banner_fn(
        f"**Target allocation drift** ({target_dimension}, above {drift_threshold:.0%}):\n\n"
        + "\n\n".join(f"- {message}" for message, _ in drift_flags)
    )

tab_positions, tab_allocation, tab_rebalance = st.tabs(["Positions", "Allocation", "Rebalance"])

with tab_positions:
    table = enriched.assign(pct=enriched["value_eur"] / total_value)[
        ["enrich_name", "enrich_asset_type", "quantity", "value_eur", "pct"]
    ].rename(columns={"enrich_name": "Name", "enrich_asset_type": "Type", "quantity": "Quantity"})
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "value_eur": st.column_config.NumberColumn("Value", format="euro"),
            "pct": st.column_config.NumberColumn("% of portfolio", format="percent"),
        },
    )

with tab_allocation:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            px.pie(asset_type_breakdown(enriched), names="bucket", values="value_eur", title="By asset type"),
            width="stretch",
        )
    if sector_region_source.empty:
        with col2:
            st.info("No ETF holdings to show.")
    else:
        with col2:
            st.plotly_chart(
                px.pie(
                    sector_breakdown(sector_region_source),
                    names="bucket",
                    values="value_eur",
                    title=f"By sector{scope_suffix}",
                ),
                width="stretch",
            )
        with col3:
            region_view = st.segmented_control(
                "View",
                ["Region", "Country"],
                default="Region",
                required=True,
                key="region_view",
            )
            breakdown = (
                continent_breakdown(sector_region_source)
                if region_view == "Region"
                else region_breakdown(sector_region_source)
            )
            st.plotly_chart(
                px.pie(breakdown, names="bucket", values="value_eur", title=f"By region{scope_suffix}"),
                width="stretch",
            )
            st.caption(
                "Region for stocks is an approximation (Yahoo Finance HQ country); "
                "for ETFs it's a real country look-through scraped from justETF."
            )

with tab_rebalance:
    if etf_only.empty:
        st.info("No ETF holdings to rebalance.")
    else:
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
                st.caption(f"Total: {total_pct}%")
            else:
                st.caption(f"Total: {total_pct}% -- doesn't sum to 100%")
            submitted = st.form_submit_button("Calculate")

        calculated_key = f"rebalance_calculated_{rebalance_dimension}"
        if submitted:
            st.session_state[calculated_key] = True

        if st.session_state.get(calculated_key, False):
            target = {category: pct / 100 for category, pct in target_pct.items()}
            buy_result = rebalance.buy_to_target(matrix, values, target)
            quick_result = rebalance.quick_rebalance(matrix, values, target)
            suggestions = rebalance.new_etf_suggestions(buy_result["unreachable"], rebalance_dimension.lower())

            price_per_unit = etf_only.set_index("enrich_ticker")["value_eur"] / etf_only.set_index("enrich_ticker")[
                "quantity"
            ]
            name_by_ticker = etf_only.set_index("enrich_ticker")["enrich_name"]

            mode = st.segmented_control(
                "View",
                ["Buy-to-target", "Quick rebalance", "New ETF suggestions"],
                default="Buy-to-target",
                required=True,
                key="rebalance_mode",
            )

            def _trade_table(amounts: pd.Series, amount_label: str) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "Name": [name_by_ticker[t] for t in amounts.index],
                        amount_label: amounts.values,
                        "Units": (amounts / price_per_unit.reindex(amounts.index)).values,
                    }
                )

            if mode == "Buy-to-target":
                if buy_result["unreachable"]:
                    st.warning(
                        "Target includes categories no held ETF provides at all: "
                        + ", ".join(f"**{c}**" for c in buy_result["unreachable"])
                        + ". Buying more of your current ETFs can't close this gap -- see New ETF suggestions."
                    )
                if buy_result["buy_eur"].empty:
                    st.info("Your current holdings are already at (or above) target -- no purchases needed.")
                else:
                    st.dataframe(
                        _trade_table(buy_result["buy_eur"], "Buy (EUR)"),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Buy (EUR)": st.column_config.NumberColumn("Buy (EUR)", format="euro"),
                            "Units": st.column_config.NumberColumn("Units", format="%.2f"),
                        },
                    )

            elif mode == "Quick rebalance":
                if quick_result["unreachable"]:
                    st.warning(
                        "Target includes categories no held ETF provides at all: "
                        + ", ".join(f"**{c}**" for c in quick_result["unreachable"])
                        + ". No amount of buying or selling your current ETFs can close this gap -- "
                        "see New ETF suggestions."
                    )
                if not quick_result["converged"]:
                    st.warning("The solver didn't fully converge -- treat this result as approximate.")
                if quick_result["delta_eur"].empty:
                    st.info("Your current holdings are already at target -- no trades needed.")
                else:
                    trades = quick_result["delta_eur"]
                    sells = _trade_table(-trades[trades < 0], "Sell (EUR)")
                    buys = _trade_table(trades[trades > 0], "Buy (EUR)")
                    if not sells.empty:
                        st.write("**Sell**")
                        st.dataframe(
                            sells,
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "Sell (EUR)": st.column_config.NumberColumn("Sell (EUR)", format="euro"),
                                "Units": st.column_config.NumberColumn("Units", format="%.2f"),
                            },
                        )
                    if not buys.empty:
                        st.write("**Buy**")
                        st.dataframe(
                            buys,
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "Buy (EUR)": st.column_config.NumberColumn("Buy (EUR)", format="euro"),
                                "Units": st.column_config.NumberColumn("Units", format="%.2f"),
                            },
                        )

            else:
                if suggestions:
                    st.warning("\n\n".join(f"- {s}" for s in suggestions))
                else:
                    st.info("Your current ETF holdings can already reach this target -- no new fund categories needed.")
