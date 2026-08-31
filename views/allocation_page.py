"""The Allocation page: the full Allocation view (view selector, donut,
holdings list, region caption) -- what the Overview page's compact donut's
"Show more" link navigates to. Renders exactly what the old Allocation tab
did, via the same `render_allocation_tab`, just called from a dedicated
page instead of inside a tab.
"""

from state import load_dashboard_state
from views import allocation


def render_allocation_page() -> None:
    dashboard = load_dashboard_state()
    allocation.render_allocation_tab(dashboard.display_source, dashboard.sector_region_source, dashboard.scope_suffix)
