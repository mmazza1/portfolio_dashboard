"""The optional Performance section: value-over-time chart, FIFO P&L for
whatever Timeframe pill is selected, and the sidebar controls (checkbox +
Account.csv uploader, or dev-mode auto-load) that feed it. Entirely
additive -- uploading only Portfolio.csv still gets the full app; see
PROJECT_PLAN.md's Performance view section.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import cost_basis
import ledger
import performance
import theme
from formatting import format_eur

_PERFORMANCE_TIMEFRAMES = ["1M", "YTD", "1Y", "Max"]
_PERFORMANCE_TIMEFRAME_DAYS = {"1M": 30, "1Y": 365}


@st.cache_data(show_spinner="Reconstructing historical portfolio value from your Account Statement...")
def load_and_reconstruct_value_history(
    file_bytes: bytes,
) -> tuple[pd.Series, float, float, list[str], pd.DataFrame, pd.DataFrame]:
    account_ledger = ledger.parse_account_csv(file_bytes)
    events = ledger.extract_share_events(account_ledger)
    history = ledger.share_history(events)
    value, _flows, unpriced, current_value_addon, unresolved = performance.portfolio_value_history(
        history, events, account_ledger["date"].min(), pd.Timestamp.now().normalize()
    )
    # `events` is returned alongside the value series so the caller can
    # compute FIFO P&L (see cost_basis.py) windowed to whatever timeframe
    # is selected, without re-parsing the CSV. `dividend_events` likewise,
    # for the Dividends section's yearly-actual-income view -- a separate
    # extraction from the same raw ledger (dividends aren't share events).
    dividends = ledger.dividend_events(account_ledger)
    return value, unpriced, current_value_addon, unresolved, events, dividends


@st.cache_data(show_spinner=False)
def compute_pnl_snapshot(events: pd.DataFrame, as_of: str) -> dict:
    """Cumulative FIFO P&L (see cost_basis.py) as of one date, cached on
    (events, as_of) rather than on a whole window -- every timeframe's
    *end* snapshot is the same date (the latest day with data), so caching
    at this granularity means that shared snapshot is only ever actually
    computed once per session, no matter how many of the four Timeframe
    pills get clicked. `compute_windowed_pnl` composes two calls to this
    for its start/end difference, rather than calling `cost_basis.
    windowed_pnl` (which would hide this sharing inside one bigger,
    less-cacheable call).
    """
    return cost_basis.total_pnl(events, as_of=pd.Timestamp(as_of))


def compute_windowed_pnl(events: pd.DataFrame, window_start: str, window_end: str) -> dict:
    """P&L within [window_start, window_end], built from two independently
    -cached snapshots (see `compute_pnl_snapshot`) -- mirrors `cost_basis.
    windowed_pnl`'s own start/end-difference math, just composed so
    Streamlit's cache can share the end snapshot across every timeframe.
    FIFO cost-basis accounting, not time-weighting -- confirmed on real
    test data to land far closer to DEGIRO's own reported "Total P&L"
    (within ~EUR 56 on a ~EUR 1,635 total) than a time-weighted approach
    did (~EUR 300 off), since it correctly counts the realized loss on a
    fully closed but otherwise-unpriceable position (the AIRWA chain) that
    a value-change-based figure has no way to see at all.
    """
    window_start_ts = pd.Timestamp(window_start) - pd.Timedelta(days=1)
    end_pnl = compute_pnl_snapshot(events, window_end)
    start_pnl = compute_pnl_snapshot(events, window_start_ts.isoformat())
    return {
        "realized_eur": end_pnl["realized_eur"] - start_pnl["realized_eur"],
        "unrealized_eur": end_pnl["unrealized_eur"] - start_pnl["unrealized_eur"],
        "total_eur": end_pnl["total_eur"] - start_pnl["total_eur"],
        "unpriced_open_isins": end_pnl["unpriced_open_isins"],
    }


def timeframe_cutoff(timeframe: str, latest_date: pd.Timestamp, earliest_date: pd.Timestamp) -> pd.Timestamp:
    if timeframe in _PERFORMANCE_TIMEFRAME_DAYS:
        return latest_date - pd.Timedelta(days=_PERFORMANCE_TIMEFRAME_DAYS[timeframe])
    if timeframe == "YTD":
        return pd.Timestamp(year=latest_date.year, month=1, day=1)
    return earliest_date  # "Max"


def prefetch_all_timeframe_pnl(events: pd.DataFrame, value_history: pd.Series) -> None:
    """Warms `compute_pnl_snapshot`'s cache for every Timeframe pill right
    when the Account Statement loads, instead of paying the cost the first
    time each pill is actually clicked. Confirmed in real test data: a
    single timeframe's first P&L computation took 12+ seconds (live
    ticker/currency/price lookups for every open position) before this;
    clicking through all four pills used to mean paying a large chunk of
    that four times over. Prefetching moves the wait to one place -- right
    after upload, under the same spinner already shown for reconstructing
    the value history -- so every Timeframe pill feels instant afterward.
    """
    latest_date, earliest_date = value_history.index.max(), value_history.index.min()
    as_of_dates = {latest_date}
    for tf in _PERFORMANCE_TIMEFRAMES:
        cutoff = timeframe_cutoff(tf, latest_date, earliest_date)
        as_of_dates.add(cutoff - pd.Timedelta(days=1))
    for as_of in as_of_dates:
        compute_pnl_snapshot(events, as_of.isoformat())


def render_performance_header(
    value_history: pd.Series,
    events: pd.DataFrame,
    unpriced_cash_flow: float,
    current_value_addon: float,
    unresolved_isins: list[str],
) -> None:
    """The one new element the Performance tracking to-do item asked for:
    total value + a timeframe-filtered chart, positioned above everything
    else on the page (matching where getquin puts its own equivalent
    panel) -- everything else on the page is untouched, still laid out
    exactly as it already was, per Matteo's explicit "keep all the
    functionalities as they are" scoping call.

    P&L is FIFO cost-basis accounting (`cost_basis.windowed_pnl`), computed
    fresh for whatever window the Timeframe pill selects -- not a fixed
    lifetime figure. An earlier version showed a separate time-weighted
    %/EUR figure and a deposits breakdown alongside this; Matteo asked for
    those removed and for P&L itself to follow the Timeframe selector
    instead, which this does.
    """
    chart_theme = theme.chart_theme()
    timeframe = st.segmented_control(
        "Timeframe",
        _PERFORMANCE_TIMEFRAMES,
        default="Max",
        required=True,
        key="performance_timeframe",
        label_visibility="collapsed",
    )
    latest_date = value_history.index.max()
    cutoff = timeframe_cutoff(timeframe, latest_date, value_history.index.min())
    windowed = value_history[value_history.index >= cutoff]

    # Only reject a window with no data *at all* to show (nothing priced
    # yet, or the account is truly empty right now) -- the *first* day of
    # "Max" is legitimately 0 (before the first purchase, share count is
    # 0 by construction), which isn't "not enough history," it's the
    # correct starting point. Rejecting on that used to hide the Max view
    # outright; only the %-change line below needs a nonzero baseline, and
    # is skipped on its own if the window happens to start at 0.
    if windowed.empty or windowed.iloc[-1] == 0:
        st.info("Not enough price history yet for this timeframe.")
        return

    current, baseline = windowed.iloc[-1], windowed.iloc[0]
    value_line = f"<div style='font-size:2.1rem;font-weight:700;color:{chart_theme['ink']}'>{format_eur(current)}</div>"

    st.markdown(value_line, unsafe_allow_html=True)

    # FIFO cost-basis P&L for exactly this window (see cost_basis.py) --
    # the day before the window starts vs. the window's last day, so it
    # reflects only what happened *within* the selected Timeframe, same as
    # the value figure above it. Cached on (events, start, end) so flipping
    # between pills doesn't re-fetch live prices for a window already seen
    # this session.
    window_pnl = compute_windowed_pnl(events, windowed.index[0].isoformat(), windowed.index[-1].isoformat())
    total_pnl_eur = window_pnl["total_eur"]
    pnl_color = "#1baf7a" if total_pnl_eur >= 0 else "#e34948"
    pnl_sign = "+" if total_pnl_eur >= 0 else ""
    # % alongside the EUR figure, at Matteo's request -- FIFO P&L over the
    # window's *starting* holdings value (`baseline`, already computed above
    # for the chart's own dotted reference line and green/red coloring, so
    # this reuses it rather than a second, differently-scoped figure).
    # Deliberately not "gain / cost basis" -- cost basis needs its own FIFO
    # walk per window and isn't otherwise on hand here, while `baseline` is
    # already the exact same "where the window started" this P&L figure is
    # itself measuring change *from*, so the two stay conceptually paired.
    # Skipped (not shown as a bogus 0% or an error) when the window starts
    # at 0 -- the "Max" timeframe legitimately does, before the first
    # purchase existed, and there's no meaningful percentage of nothing.
    pnl_pct_text = ""
    if baseline > 0:
        pnl_pct = total_pnl_eur / baseline * 100
        pnl_pct_text = f" <span style='color:{pnl_color}'>({pnl_sign}{pnl_pct:.1f}%)</span>"
    pnl_line = (
        f"<div style='margin-top:6px;font-size:0.95rem'>"
        f"<span style='color:{chart_theme['muted']}'>P&amp;L ({timeframe}):</span> "
        f"<span style='color:{pnl_color};font-weight:600'>{pnl_sign}{format_eur(total_pnl_eur)}</span>"
        f"{pnl_pct_text}</div>"
    )
    st.markdown(pnl_line, unsafe_allow_html=True)
    if window_pnl["unpriced_open_isins"]:
        st.caption(
            f"{len(window_pnl['unpriced_open_isins'])} currently-held position(s) couldn't be priced and are "
            "excluded from P&L above."
        )

    # Green when the line ends above where it started in this window, red
    # otherwise -- matches what the line itself visibly does, independent
    # of the performance/deposits split in the text above (that's about
    # *why* the value moved, this is just "did the line go up or down").
    line_color = "#1baf7a" if current >= baseline else "#e34948"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[windowed.index[0], windowed.index[-1]],
            y=[baseline, baseline],
            mode="lines",
            line=dict(color=chart_theme["muted"], width=1, dash="dot"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=windowed.index,
            y=windowed.values,
            mode="lines",
            line=dict(color=line_color, width=2),
            hovertemplate="%{x|%d %b %Y}<br>€%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=chart_theme["muted"], family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        xaxis=dict(showgrid=False, color=chart_theme["muted"]),
        yaxis=dict(showgrid=False, color=chart_theme["muted"]),
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Short line always visible; everything else (why, and the unresolved-
    # position caveats) sits behind the "ⓘ" popover instead of being
    # printed in full underneath the chart every time -- Matteo flagged
    # the always-on version as too long for what's actually necessary at a
    # glance ("market value of your holding, cash not included").
    caption_col, info_col = st.columns([20, 1])
    with caption_col:
        st.caption("Market value of your holdings only, cash isn't included.")
    with info_col:
        with st.popover("ⓘ", width="content"):
            details = (
                "Reconstructed from your Account Statement's buy/sell history × historical daily closing prices."
            )
            # if unresolved_isins:
            #     details += (
            #         f"\n\n{len(unresolved_isins)} position(s) with no resolvable price history (e.g. options, "
            #         "delisted tickers) are excluded."
            #     )
            #     # A EUR amount, not a %, since there's no value series to
            #     # compute a return against for something this pipeline
            #     # can't price at all -- a transparency note so real money
            #     # (an option's premium, a delisted stock's cost basis)
            #     # doesn't just silently vanish from every figure above.
            #     # Covers the full account history, not just the selected
            #     # timeframe -- windowing it would need per-timeframe
            #     # recomputation this isn't worth adding for a footnote.
            #     if abs(unpriced_cash_flow) > 0.005:
            #         direction = "spent on" if unpriced_cash_flow >= 0 else "received from"
            #         details += (
            #             f" {format_eur(abs(unpriced_cash_flow))} net {direction} those positions (across your "
            #             "full history) isn't reflected in the value or performance figures above."
            #         )
            #     if current_value_addon > 0.005:
            #         # The current *total* does include this -- only the
            #         # chart's history doesn't (it can't be reconstructed
            #         # for these positions), so there's a small, real,
            #         # deliberate jump on the most recent day rather than a
            #         # smoothly wrong history.
            #         details += (
            #             f" Of that, {format_eur(current_value_addon)} is still held today and *is* included in "
            #             "today's total above (just not in the chart's history)."
            #         )
            #     elif len(unresolved_isins) > 1:
            #         # Only worth a note when it's plausibly relevant -- a
            #         # single unresolved ISIN with no ticker-sharing
            #         # sibling either got excluded for a reason unrelated
            #         # to this (e.g. it's genuinely not held anymore) or
            #         # already got its current value restored above
            #         # (current_value_addon would be > 0 in that case).
            #         details += (
            #             " Some of these share a ticker with each other (a rename/reverse-split chain the "
            #             "broker's own records never fully closed out) -- which one, if any, is still genuinely "
            #             "held can't be told apart from this data alone, so none of them are credited a current "
            #             "value either."
            #         )
            st.markdown(details)


def render_performance_section(dev_mode: bool, dev_account_path: str) -> pd.DataFrame | None:
    """Sidebar checkbox + Account.csv uploader (or dev-mode auto-load) that
    feed `render_performance_header`. Entirely optional and additive:
    unchecked (or no file yet), nothing renders and the rest of the app is
    unaffected -- see the module docstring.

    Returns the real dividend cash flows extracted from the same upload
    (`None` if this section isn't enabled or no file's been provided yet).
    `views/dividends.py` would use this for actual yearly income instead
    of its Yahoo-estimate fallback, but the Dividends section is currently
    hidden entirely (`app.py`'s `SHOW_DIVIDENDS_SECTION`) -- this return
    value is unused for now, kept rather than removed since re-enabling
    Dividends shouldn't require re-plumbing this. Account.csv is genuinely
    this section's own concern (the upload control, the caching, the
    session-state wiring all live here already), so returning its richer
    data back to the caller is simpler than duplicating an independent
    upload flow in `views/dividends.py`.

    Briefly relabeled around the data source ("Account Statement") while
    that dividend-sharing was visible in the UI, since a "performance
    tracking" checkbox was a non-obvious gate for an unrelated-sounding
    dividends feature -- reverted back to "performance tracking" now that
    Dividends is hidden and that ambiguity doesn't apply, per Matteo.
    """
    st.sidebar.subheader("Performance tracking (beta)")
    st.sidebar.caption(
        "Optional: upload your DEGIRO Account Statement (Account.csv) to add a value-over-time chart above. "
    )
    dev_account_available = dev_mode and os.path.exists(dev_account_path)
    # Off by default, deliberately -- even in dev mode with a fixture ready
    # to auto-load, per Matteo's request. Still remembers a real, explicit
    # choice within the session: once Matteo (or a real user) has actually
    # uploaded Account.csv, `account_bytes` is set and the box stays
    # checked across reruns rather than reverting to off on its own.
    performance_enabled = st.sidebar.checkbox(
        "Enable performance tracking",
        value=st.session_state.get("account_bytes") is not None,
    )
    value_history, unpriced_cash_flow, current_value_addon, events_history, dividend_events = (
        None, 0.0, 0.0, None, None,
    )
    unresolved_price_isins = []
    if performance_enabled:
        if dev_account_available:
            with open(dev_account_path, "rb") as f:
                account_bytes = f.read()
        else:
            show_account_uploader = (
                st.session_state.get("account_bytes") is None or st.session_state["show_upload_panel"]
            )
            if show_account_uploader:
                uploaded_account = st.sidebar.file_uploader("Upload Account.csv", type="csv", key="account_uploader")
                if uploaded_account is not None:
                    st.session_state["account_bytes"] = uploaded_account.getvalue()
                    st.session_state["show_upload_panel"] = False
                    # Keeps dev_data/Account.csv in sync with whatever was
                    # most recently actually uploaded, same as state.py does
                    # for Portfolio.csv (`_sync_dev_data` there) -- not
                    # reused directly from here since state.py already
                    # imports this module (`from views import performance`),
                    # so the reverse import would be circular; duplicated
                    # rather than factored out to a third module for three
                    # lines. Unlike Portfolio.csv, there's no derived
                    # on-disk cache to invalidate alongside this one --
                    # `load_and_reconstruct_value_history` is `@st.
                    # cache_data`-cached on the file's bytes directly, not a
                    # separate snapshot file, so a changed dev_account_path
                    # is automatically a fresh cache key next time regardless.
                    os.makedirs(os.path.dirname(dev_account_path), exist_ok=True)
                    with open(dev_account_path, "wb") as f:
                        f.write(st.session_state["account_bytes"])
                    st.rerun()
            account_bytes = st.session_state.get("account_bytes")

        if account_bytes is not None:
            (
                value_history, unpriced_cash_flow, current_value_addon,
                unresolved_price_isins, events_history, dividend_events,
            ) = load_and_reconstruct_value_history(account_bytes)
            with st.spinner("Computing profit/loss for every timeframe..."):
                prefetch_all_timeframe_pnl(events_history, value_history)

    if value_history is not None:
        render_performance_header(
            value_history, events_history, unpriced_cash_flow, current_value_addon, unresolved_price_isins
        )

    return dividend_events
