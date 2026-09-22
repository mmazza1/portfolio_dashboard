"""The Overview page (the multipage app's default/landing page): Positions
and a compact Allocation donut side-by-side, plus Rebalance below when
scoped to ETFs only -- replaces the old Positions/Allocation/Rebalance
st.tabs() layout, at Matteo's request, modeled after getquin's dashboard
home screen.
"""

import streamlit as st

import aggregations
import theme
from components.donut_chart import render_donut
from state import SHOW_DIVIDENDS_SECTION, load_dashboard_state
from views import dividends, positions, rebalance_view


def render_overview_page() -> None:
    dashboard = load_dashboard_state()

    # Positions still gets the wider column -- it's a table, Allocation's
    # compact donut here is just a glance, not the full chart+list (see the
    # Allocation page, linked below, for that) -- but not as wide as the
    # original 2:1 split: at 1:2, the six-option view selector (Type/
    # Positions/Sectors/Regions/Countries/Currencies) wrapped onto two rows
    # once "Currencies" didn't fit the remaining width, and
    # st.segmented_control has no font-size/padding knob to shrink it
    # instead. 3:2 gives the selector enough room to stay on one row.
    positions_col, allocation_col = st.columns([3, 2])

    with positions_col:
        positions.render_positions_tab(dashboard.display_source, dashboard.display_total_value, dashboard.scope)

    with allocation_col:
        chart_theme = theme.chart_theme()
        available_views = [
            v for v in aggregations.DONUT_VIEWS
            if v not in aggregations.DONUT_SCOPE_AWARE_VIEWS or not dashboard.sector_region_source.empty
        ]
        # Centered, not left-aligned -- st.segmented_control has no
        # horizontal_alignment of its own, so it's wrapped in a container
        # that centers it. This has to match how the chart below centers
        # itself (see donut_chart.py's compact-mode comment) rather than
        # both just happening to sit at the same spot: a left-aligned
        # selector and a left-aligned or centered chart only lined up by
        # coincidence at one column width and drifted apart at others
        # (e.g. after collapsing the sidebar) -- centering *both* the same
        # way keeps their centers matched at any column width, and reads as
        # more centered in the column overall rather than pinned to its
        # left edge, per Matteo's request.
        with st.container(horizontal_alignment="center"):
            # Its own session_state key ("overview_donut_view"), deliberately
            # not shared with the Allocation page's "donut_view" -- switching
            # views here for a quick glance shouldn't also jump the full
            # Allocation page to a different view than you left it on.
            # Defaults to "Type" (Matteo's original ask), but is otherwise
            # freely switchable now, same six views as the full page.
            donut_view = st.segmented_control(
                "View",
                available_views,
                default="Type",
                required=True,
                key="overview_donut_view",
                label_visibility="collapsed",
            )
        data = aggregations.donut_data_for_view(donut_view, dashboard.display_source, dashboard.sector_region_source)
        if not data:
            st.info("No data to show for this view.")
        else:
            center_label = (
                f"{donut_view}{dashboard.scope_suffix}"
                if donut_view in aggregations.DONUT_SCOPE_AWARE_VIEWS
                else donut_view
            )
            # Bigger than before (320 vs 260) -- Matteo asked for the chart
            # itself to read as larger, not just better-centered.
            render_donut(
                center_label,
                data,
                colors=chart_theme["colors"],
                ink=chart_theme["ink"],
                muted=chart_theme["muted"],
                height=320,
                compact=True,
            )
        # The Page object is stashed in session_state by app.py (see its
        # own comment) rather than imported here -- importing it from
        # app.py directly would re-run app.py's own top-level code
        # (st.navigation(...).run() included) as a side effect of the
        # import, and views/allocation_page.py importing this module back
        # to get at it would be a circular import besides.
        allocation_page = st.session_state.get("_allocation_page")
        if allocation_page is not None:
            with st.container(horizontal_alignment="center"):
                st.page_link(allocation_page, label="Show more →")

    # Rebalance only makes sense scoped to ETFs (see the Rebalancing
    # calculator section in PROJECT_PLAN.md), so it's only shown at all
    # when the Scope toggle above is set to ETFs only -- previously its own
    # tab; now a plain section below the two-column layout, since a single
    # lone tab once Positions/Allocation aren't tabs anymore wouldn't make
    # sense.
    if dashboard.scope == "ETFs only":
        st.subheader("Rebalance")
        rebalance_view.render_rebalance_tab(dashboard.etf_only, dashboard.cash_value)

    if SHOW_DIVIDENDS_SECTION:
        dividends.render_dividends_section(dashboard.display_source, dashboard.display_total_value, dashboard.scope_suffix)
