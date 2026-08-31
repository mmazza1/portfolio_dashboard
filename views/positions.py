"""The Positions tab: a paginated table of holdings with a Show more/less
toggle, TER + justETF link columns added when scoped to ETFs only.
"""

import pandas as pd
import streamlit as st


def render_positions_tab(display_source: pd.DataFrame, display_total_value: float, scope: str) -> None:
    table = display_source.assign(
        pct=display_source["value_eur"] / display_total_value,
    )[["enrich_name", "enrich_asset_type", "quantity", "value_eur", "pct", "enrich_isin", "enrich_ter"]].rename(
        columns={"enrich_name": "Name", "enrich_asset_type": "Type", "quantity": "Quantity"}
    )

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
    # Sized off the header label itself (character count * an average glyph
    # width + fixed padding for the cell's own margin and the sort-arrow
    # gutter), not off the data -- Matteo asked for these three specifically
    # to match their column *name's* length rather than being eyeballed
    # pixel values, so the width is genuinely derived from the label text
    # instead of a number that happens to look right for one label and
    # wrong for the next. Calibrated against "Type" (4 chars) reading
    # comfortably at 55px, which is where the 24px padding + ~7.7px/char
    # figures below come from.
    def _header_width(label: str) -> int:
        return round(len(label) * 7.7 + 24)

    column_config = {
        # Type keeps its own fixed width (short, fixed-vocabulary content:
        # "Stock"/"ETF"/"Cash") rather than the header formula -- its label
        # is already about as short as its longest value.
        "Type": st.column_config.TextColumn("Type", width=90),
        "Quantity": st.column_config.NumberColumn("Quantity", width=_header_width("Quantity")),
        "value_eur": st.column_config.NumberColumn("Total Value", format="€%.2f", width=_header_width("Total Value")),
        "pct": st.column_config.NumberColumn("% of portfolio", format="%.2f%%", width=_header_width("% of portfolio")),
    }
    column_order = ["Name", "Type", "Quantity", "value_eur", "pct"]
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
            "More info", display_text="ⓘ", width=50, help="Open this ETF's justETF page for more information"
        )
        column_config["TER"] = st.column_config.NumberColumn("TER", format="%.2f%%", width=55)
        column_order = column_order + ["TER", "justetf_url"]
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
    # Back to "stretch", not "content": "content" (the previous fix for
    # "table is too wide") kept the table sized to just its own columns --
    # correct on its own, but it stopped filling the positions column, which
    # broke two things Matteo flagged: freeing up space (e.g. collapsing the
    # sidebar) left a dead gap next to the table instead of the table using
    # it, and the "Show less" button -- positioned via a plain
    # `st.columns([7, 1])` split of the *column's* width -- ended up
    # floating to the right of the table's real (narrower) edge instead of
    # sitting under it. "stretch" fixes both by making the table genuinely
    # fill the column again, which is also what the button's column split
    # assumes. The tradeoff: `width="stretch"` grows every column by an
    # equal share of the leftover space, fixed pixel widths included, not
    # just Name -- confirmed in the compiled frontend bundle (every
    # non-pinned column gets `grow:1` when stretched). So the tight
    # Type/Quantity/%-of-portfolio widths above are a *starting* point, not
    # a hard cap, once there's slack to fill -- an acceptable tradeoff since
    # the starting widths are now genuinely tight (header-fit, not the old
    # oversized "small" preset), so the growth reads as reasonable column
    # padding rather than the original wide-columns complaint.
    st.dataframe(
        displayed_table,
        width="stretch",
        hide_index=True,
        height=38 + 35 * visible_rows + 3,
        column_config=column_config,
        column_order=column_order,
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
