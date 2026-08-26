import pandas as pd
import plotly.express as px
import streamlit as st

from enrichment import enrich_positions
from parser import parse_degiro_csv

st.set_page_config(page_title="Portfolio Diversification Dashboard", layout="wide")


@st.cache_data(show_spinner="Parsing CSV and enriching positions via Yahoo Finance (first run can take a minute)...")
def load_and_enrich(file_bytes: bytes) -> pd.DataFrame:
    positions = parse_degiro_csv(file_bytes)
    return enrich_positions(positions)


def sector_breakdown(non_cash: pd.DataFrame) -> pd.DataFrame:
    totals: dict[str, float] = {}
    for row in non_cash.itertuples():
        weights = row.enrich_sector_weights or {}
        for sector, weight in weights.items():
            totals[sector] = totals.get(sector, 0.0) + row.value_eur * weight
        leftover = 1 - sum(weights.values())
        if leftover > 1e-9:
            totals["Unresolved"] = totals.get("Unresolved", 0.0) + row.value_eur * leftover
    return pd.DataFrame(totals.items(), columns=["bucket", "value_eur"]).sort_values("value_eur", ascending=False)


def region_breakdown(non_cash: pd.DataFrame) -> pd.DataFrame:
    region = non_cash["enrich_region"].fillna("Unresolved")
    return (
        non_cash.assign(bucket=region)
        .groupby("bucket")["value_eur"]
        .sum()
        .reset_index()
        .sort_values("value_eur", ascending=False)
    )


def asset_type_breakdown(enriched: pd.DataFrame) -> pd.DataFrame:
    return (
        enriched.assign(bucket=enriched["enrich_asset_type"].fillna("Unresolved"))
        .groupby("bucket")["value_eur"]
        .sum()
        .reset_index()
        .sort_values("value_eur", ascending=False)
    )


def concentration_flags(enriched: pd.DataFrame, total_value: float, threshold: float) -> list[str]:
    flags = []

    for row in enriched.itertuples():
        weight = row.value_eur / total_value
        if weight > threshold:
            name = row.enrich_name or row.product
            flags.append(f"Position **{name}** is **{weight:.0%}** of your portfolio.")

    non_cash = enriched[~enriched["is_cash"]]
    for label, breakdown_fn in (("Sector", sector_breakdown), ("Region", region_breakdown)):
        breakdown = breakdown_fn(non_cash)
        for _, r in breakdown.iterrows():
            weight = r["value_eur"] / total_value
            if weight > threshold:
                flags.append(f"{label} **{r['bucket']}** is **{weight:.0%}** of your portfolio.")

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

threshold = (
    st.sidebar.slider(
        "Concentration warning threshold",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        help="Flags any single position, sector, or region that makes up more than this share of your portfolio.",
    )
    / 100
)

enriched = load_and_enrich(uploaded_file.getvalue())
enriched["enrich_name"] = enriched["enrich_name"].fillna(enriched["product"])
total_value = enriched["value_eur"].sum()

lookup_errors = enriched.loc[~enriched["is_cash"] & enriched["enrich_error"].notna(), ["product", "enrich_error"]]
if not lookup_errors.empty:
    with st.expander(f"{len(lookup_errors)} position(s) could not be fully resolved via Yahoo Finance"):
        for _, r in lookup_errors.iterrows():
            st.write(f"- **{r['product']}**: {r['enrich_error']}")

flags = concentration_flags(enriched, total_value, threshold)
overlaps = etf_overlap_flags(enriched)

if flags or overlaps:
    banner_col1, banner_col2 = st.columns(2)
    with banner_col1:
        if flags:
            st.warning(
                "**Concentration warning** (above {:.0%}):\n\n".format(threshold) + "\n\n".join(f"- {f}" for f in flags)
            )
    with banner_col2:
        if overlaps:
            st.info("**ETF overlap:**\n\n" + "\n\n".join(f"- {f}" for f in overlaps))

tab_positions, tab_allocation = st.tabs(["Positions", "Allocation"])

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
    non_cash = enriched[~enriched["is_cash"]]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            px.pie(asset_type_breakdown(enriched), names="bucket", values="value_eur", title="By asset type"),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            px.pie(sector_breakdown(non_cash), names="bucket", values="value_eur", title="By sector"),
            width="stretch",
        )
    with col3:
        st.plotly_chart(
            px.pie(region_breakdown(non_cash), names="bucket", values="value_eur", title="By region"),
            width="stretch",
        )
        st.caption("Region is approximate: based on exchange listing or headquarters, not confirmed revenue exposure.")
