"""The Allocation tab: the interactive donut across Type/Positions/Sectors/
Regions/Countries/Currencies, scoped by the top-of-page Scope toggle.
"""

import pandas as pd
import streamlit as st

import aggregations
import theme
from components.donut_chart import render_donut


def render_allocation_tab(
    display_source: pd.DataFrame, sector_region_source: pd.DataFrame, scope_suffix: str
) -> None:
    chart_theme = theme.chart_theme()
    available_views = [
        v for v in aggregations.DONUT_VIEWS
        if v not in aggregations.DONUT_SCOPE_AWARE_VIEWS or not sector_region_source.empty
    ]
    donut_view = st.segmented_control(
        "View",
        available_views,
        default=available_views[0],
        required=True,
        key="donut_view",
        label_visibility="collapsed",
    )

    data = aggregations.donut_data_for_view(donut_view, display_source, sector_region_source)
    if not data:
        st.info("No data to show for this view.")
    else:
        center_label = (
            f"{donut_view}{scope_suffix}" if donut_view in aggregations.DONUT_SCOPE_AWARE_VIEWS else donut_view
        )
        render_donut(center_label, data, colors=chart_theme["colors"], ink=chart_theme["ink"], muted=chart_theme["muted"])
        if donut_view in ("Regions", "Countries"):
            st.caption(
                "Region for stocks is an approximation (Yahoo Finance HQ country); "
                "for ETFs it's a real country look-through scraped from justETF."
            )
