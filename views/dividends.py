"""The Dividends section: a collapsed expander below the tabs (not a tab of
its own -- an occasional check, not a primary view), showing dividend
income.

A forward-looking per-position estimate derived from Yahoo Finance's own
dividend payment history (`enrichment.py`'s `dividend_yield`), needing only
the base Portfolio.csv upload -- the only data source now that the
Performance section (and the Account Statement upload / `ledger.py` /
`cost_basis.py` it depended on) has been removed entirely. This used to be
one of two picked automatically depending on what was available -- real
yearly income actually received, from the optional Account Statement
upload -- but that source no longer exists.
"""

import pandas as pd
import streamlit as st

from formatting import format_eur, format_pct


def dividend_breakdown(display_source: pd.DataFrame) -> pd.DataFrame:
    """Per-position trailing-12-month dividend income and yield estimate.
    Cash is excluded (it doesn't pay dividends -- showing it as a zero-row
    would be noise, not information). `enrich_dividend_yield` is `None`
    for a failed lookup and `0.0` for a genuine no-dividends result (e.g.
    an accumulating ETF, or a stock that just doesn't pay one) -- both
    display the same way here (€0 / 0%), since the lookup-error expander
    elsewhere already surfaces which rows failed outright.
    """
    non_cash = display_source[~display_source["is_cash"]]
    yield_fraction = non_cash["enrich_dividend_yield"].fillna(0.0)
    return (
        non_cash.assign(yield_fraction=yield_fraction, annual_income_eur=yield_fraction * non_cash["value_eur"])[
            ["enrich_name", "enrich_asset_type", "annual_income_eur", "yield_fraction"]
        ]
        .rename(columns={"enrich_name": "Name", "enrich_asset_type": "Type"})
        .sort_values("annual_income_eur", ascending=False)
    )


def _render_estimated_income_view(display_source: pd.DataFrame, display_total_value: float) -> None:
    dividends = dividend_breakdown(display_source)
    total_income = float(dividends["annual_income_eur"].sum())
    portfolio_yield = total_income / display_total_value if display_total_value else 0.0

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Estimated annual income", format_eur(total_income))
    metric_col2.metric("Portfolio dividend yield", format_pct(portfolio_yield))

    st.dataframe(
        dividends.assign(yield_pct=dividends["yield_fraction"] * 100).drop(columns=["yield_fraction"]),
        width="stretch",
        hide_index=True,
        column_config={
            "annual_income_eur": st.column_config.NumberColumn("Annual income", format="€%.2f"),
            "yield_pct": st.column_config.NumberColumn("Yield", format="%.2f%%"),
        },
    )
    st.caption(
        "Trailing-12-month dividend income and yield, derived per position from Yahoo Finance's dividend "
        "payment history."
    )


def render_dividends_section(display_source: pd.DataFrame, display_total_value: float, scope_suffix: str) -> None:
    # A collapsed section below the tabs rather than a tab of its own -- this
    # isn't a primary view you switch to like Positions/Allocation/Rebalance,
    # just an occasional check you scroll down and open.
    with st.expander(f"Dividends{scope_suffix}"):
        _render_estimated_income_view(display_source, display_total_value)
