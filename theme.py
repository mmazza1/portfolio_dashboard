"""Chart color theme -- the "preamble" of visual constants shared by every
view that draws a chart (currently the Allocation donut and the Performance
line chart). `formatting.py` plays the same role for number formatting;
this is its counterpart for color.
"""

import streamlit as st

# Fixed-order categorical palette (never cycled by rank), one step per mode --
# validated for adjacent-pair colorblind-safe separation. Light/dark are the
# same eight hues stepped for their respective surface, not separate palettes.
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def chart_theme() -> dict:
    """Colors/ink for the current Streamlit theme. Falls back to dark (this
    app's default) when the theme can't be read, e.g. outside a live session.
    """
    theme_type = st.context.theme.get("type") or "dark"
    if theme_type == "light":
        return {"colors": CATEGORICAL_LIGHT, "ink": "#0b0b0b", "muted": "#52514e"}
    return {"colors": CATEGORICAL_DARK, "ink": "#ffffff", "muted": "#c3c2b7"}
