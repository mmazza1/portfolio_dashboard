"""Entry point. Deliberately thin: page config and Streamlit's multipage
navigation (st.navigation + st.Page, callable-based -- no pages/ directory
needed), nothing else. Each page is its own independent script run and
calls `state.load_dashboard_state()` itself to get the shared setup
(loading, sidebar controls, banners) -- see state.py's own docstring for
why that's called per-page rather than done once here.
"""

import streamlit as st

from views.allocation_page import render_allocation_page
from views.overview import render_overview_page

st.set_page_config(page_title="Portfolio Diversification Dashboard", layout="wide")

overview_page = st.Page(render_overview_page, title="Overview", icon="📊", default=True)
allocation_page = st.Page(render_allocation_page, title="Allocation", icon="🥯")

# Stashed in session_state so views/overview.py's "Show more" st.page_link
# can reference the Allocation Page object directly, without importing it
# from this file -- importing app.py from a views/ module would re-run
# this file's own top-level code (st.navigation(...).run() included) as a
# side effect of the import, and the reverse import (this file pulling the
# page object back out of views/overview.py) isn't possible either, since
# the object doesn't exist until st.Page() is called here.
st.session_state["_allocation_page"] = allocation_page

pg = st.navigation([overview_page, allocation_page])
pg.run()
