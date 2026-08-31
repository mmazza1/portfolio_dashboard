"""Shared per-page setup for the multipage app.

A Streamlit page registered via `st.Page` (see app.py) is its own
independent script run -- it does not inherit local variables from app.py
or from any other page's run, only `st.session_state` persists across
navigation. So rather than compute the shared filtered dataframes once and
have pages "receive" them, every page calls `load_dashboard_state()` at
the top of its own script and gets them fresh -- correctness (both pages
always reflecting the current Scope toggle and the currently uploaded
file) matters more here than shaving off recomputation. The actual
expensive part (Yahoo Finance / justETF lookups in `load_and_enrich`) is
still only ever paid once, `@st.cache_data`-cached and shared across pages
in the same session regardless of how many times this function itself
runs -- what's being "recomputed" per page is just re-deriving a few
cheap, pandas-only slices from that one cached result, plus re-drawing
the sidebar controls so they show up identically on every page.
"""

import json
import os
from dataclasses import dataclass

import pandas as pd
import streamlit as st

import aggregations
from enrichment import enrich_positions
from formatting import format_number, format_pct
from parser import parse_degiro_csv
from views import performance

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
DEV_MODE = True
DEV_DATA_PATH = "dev_data/Portfolio.csv"
DEV_SNAPSHOT_PATH = "dev_data/enriched_snapshot.parquet"
# Same dev-mode convenience, for the optional Performance view: if this
# exists, dev mode enables performance tracking automatically instead of
# waiting on a manual upload. No caching layer here like DEV_SNAPSHOT_PATH --
# reconstructing value history is one @st.cache_data call already (see
# views/performance.py's load_and_reconstruct_value_history), cheap enough
# not to need a second one.
DEV_ACCOUNT_PATH = "dev_data/Account.csv"
# Hidden, not deleted: the Dividends section (views/dividends.py) isn't
# behaving as intended yet -- Matteo asked for it off the dashboard
# entirely until that's sorted, but kept in the codebase rather than torn
# out, so re-enabling it later is a flip back to True, not a rewrite. The
# `dividends` module still gets imported either way -- only the call that
# renders it is gated.
SHOW_DIVIDENDS_SECTION = False
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
    "enrich_dividend_yield",
    "enrich_error",
}


@dataclass
class DashboardState:
    """Everything a page needs to render its own content, computed fresh
    (from cached underlying data) at the top of every page's script."""

    enriched: pd.DataFrame
    display_source: pd.DataFrame
    sector_region_source: pd.DataFrame
    etf_only: pd.DataFrame
    cash_value: float
    scope: str
    scope_suffix: str
    display_total_value: float
    account_dividends: pd.DataFrame | None


def _sync_dev_data(path: str, file_bytes: bytes) -> None:
    """Writes a just-uploaded file's bytes into dev_data/ too, so DEV_MODE
    -- a separate on/off switch, not tied to whether an upload happened --
    always tests against whatever was most recently actually uploaded
    instead of a fixture that silently drifts out of date. Per Matteo's
    request: DEV_MODE off doesn't mean dev_data stops being useful, it's
    used the moment DEV_MODE is flipped back on. `dev_data/` isn't
    guaranteed to exist yet (a fresh checkout, or `.gitignore`d entirely),
    so this creates it rather than failing on a missing directory.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(file_bytes)


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


@st.cache_data(show_spinner="Parsing CSV and enriching positions via Yahoo Finance (first run can take a minute)...")
def load_and_enrich(file_bytes: bytes) -> pd.DataFrame:
    positions_df = parse_degiro_csv(file_bytes)
    return enrich_positions(positions_df)


def load_dashboard_state() -> DashboardState:
    """Renders the title/upload button, the Performance sidebar section, the
    Portfolio.csv upload-or-dev-load flow, the sidebar concentration and
    target-allocation controls, and the top-of-page banners -- then returns
    the filtered dataframes every page's own content needs. Call this once
    at the very top of each page's render function, before that page draws
    anything of its own.
    """
    # Once a file's been uploaded once, its bytes are cached in session_state
    # so the uploader widget itself can be hidden on later reruns -- otherwise
    # there's no way to tell "user hasn't uploaded yet" from "user uploaded,
    # widget just isn't holding onto it across reruns" apart from re-reading
    # the widget's own live value, which is exactly what disappearing the
    # widget would lose. `show_upload_panel` is the one flag that brings both
    # uploaders (Portfolio.csv here, Account.csv in the sidebar below) back --
    # a single button covers both, since re-uploading one commonly means
    # re-exporting the other too (a fresher DEGIRO CSV pair).
    st.session_state.setdefault("show_upload_panel", True)

    title_col, upload_button_col = st.columns([6, 1])
    with title_col:
        st.title("Portfolio Diversification Dashboard")
    with upload_button_col:
        if not DEV_MODE and st.session_state.get("portfolio_bytes") is not None:
            st.write("")  # nudge the button down to roughly the title's baseline
            if st.button("📤 Upload new files", width="stretch"):
                st.session_state["show_upload_panel"] = True

    account_dividends = performance.render_performance_section(DEV_MODE, DEV_ACCOUNT_PATH)

    if DEV_MODE:
        st.sidebar.subheader("Dev mode")
        if st.sidebar.button(
            "Refresh dev snapshot", help="Deletes the cached enriched snapshot and re-runs enrichment fresh."
        ):
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
        show_portfolio_uploader = (
            st.session_state.get("portfolio_bytes") is None or st.session_state["show_upload_panel"]
        )
        if show_portfolio_uploader:
            uploaded_file = st.file_uploader("Upload your DEGIRO Portfolio.csv export", type="csv")
            if uploaded_file is not None:
                st.session_state["portfolio_bytes"] = uploaded_file.getvalue()
                st.session_state["show_upload_panel"] = False
                _sync_dev_data(DEV_DATA_PATH, st.session_state["portfolio_bytes"])
                # The cached snapshot was built from the *previous*
                # Portfolio.csv -- stale now that dev_data's own copy just
                # changed underneath it, so it's invalidated here rather
                # than left to silently serve old classification/prices
                # against a portfolio that no longer matches, the next time
                # DEV_MODE is on.
                if os.path.exists(DEV_SNAPSHOT_PATH):
                    os.remove(DEV_SNAPSHOT_PATH)
                st.rerun()
        if st.session_state.get("portfolio_bytes") is None:
            st.info("Upload a DEGIRO Portfolio.csv export to get started.")
            st.stop()
        enriched = load_and_enrich(st.session_state["portfolio_bytes"])

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
    target_breakdown = (
        aggregations.asset_type_breakdown(enriched)
        if target_dimension == "Asset type"
        else aggregations.region_breakdown(non_cash)
    )
    # Asset type buckets (including Cash) sum to the whole portfolio; region buckets
    # only cover non-cash holdings, so they're compared against their own total
    # rather than total_value, or they'd never be able to reach 100%.
    target_total_value = target_breakdown["value_eur"].sum()
    target_breakdown = target_breakdown.assign(
        current_pct=aggregations.round_percentages(target_breakdown["value_eur"] / target_total_value * 100)
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

    flags = aggregations.concentration_flags(
        position_source,
        sector_region_source,
        concentration_total_value,
        asset_threshold,
        sector_threshold,
        region_threshold,
    )
    overlaps = aggregations.etf_overlap_flags(enriched)
    drift_flags = aggregations.target_drift_flags(target_breakdown, targets, target_total_value, drift_threshold)

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

    return DashboardState(
        enriched=enriched,
        display_source=display_source,
        sector_region_source=sector_region_source,
        etf_only=etf_only,
        cash_value=cash_value,
        scope=scope,
        scope_suffix=scope_suffix,
        display_total_value=display_total_value,
        account_dividends=account_dividends,
    )
