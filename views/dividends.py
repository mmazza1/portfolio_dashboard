"""The Dividends section: a collapsed expander below the tabs (not a tab of
its own -- an occasional check, not a primary view), showing dividend
income.

Two different data sources, picked automatically depending on what's
available: real yearly income actually received, from the optional
Account Statement upload (see `ledger.dividend_events` /
`cost_basis.yearly_dividend_income`) when it's been uploaded; otherwise a
forward-looking per-position estimate derived from Yahoo Finance's own
dividend payment history (`enrichment.py`'s `dividend_yield`), which needs
only the base Portfolio.csv upload. The estimate view is the original
implementation, kept as a fallback so this section still works without
the optional second upload -- same "additive, never breaks the base
experience" pattern as the Performance section itself.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import cost_basis
import theme
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


def _render_yearly_income_view(account_dividend_events: pd.DataFrame) -> None:
    """Real dividend income actually received, by calendar year -- a
    getquin-style bar per year, at Matteo's request. Only years with real
    payment history are shown; a forecast for future years (getquin shows
    one) isn't built yet -- deliberately deferred, per Matteo's own "can be
    a later option."
    """
    yearly = cost_basis.yearly_dividend_income(account_dividend_events)
    st.metric("Total received", format_eur(float(yearly.sum())))

    chart_theme = theme.chart_theme()
    years = [str(year) for year in yearly.index]
    fig = go.Figure(
        go.Bar(
            x=years,
            y=yearly.values,
            marker=dict(color=chart_theme["colors"][0]),
            text=[format_eur(v) for v in yearly.values],
            textposition="outside",
            hovertemplate="%{x}<br>€%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(t=30, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=chart_theme["muted"], family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        xaxis=dict(showgrid=False, color=chart_theme["muted"]),
        yaxis=dict(showgrid=False, color=chart_theme["muted"], showticklabels=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Actual dividend income received into your account each calendar year (gross payments minus withheld "
        "tax), from your Account Statement."
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


def render_dividends_section(
    display_source: pd.DataFrame,
    display_total_value: float,
    scope_suffix: str,
    account_dividend_events: pd.DataFrame | None = None,
) -> None:
    # A collapsed section below the tabs rather than a tab of its own -- this
    # isn't a primary view you switch to like Positions/Allocation/Rebalance,
    # just an occasional check you scroll down and open.
    with st.expander(f"Dividends{scope_suffix}"):
        if account_dividend_events is not None and not account_dividend_events.empty:
            _render_yearly_income_view(account_dividend_events)
        else:
            _render_estimated_income_view(display_source, display_total_value)
