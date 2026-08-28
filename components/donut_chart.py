"""Reusable interactive donut chart, rendered as raw Plotly.js embedded via
`st.components.v1.html()` rather than `st.plotly_chart()` / `px.pie()`.

Streamlit's plotly_chart wrapper has no client-side event hooks -- a
hover-driven UI (fade every slice but the hovered one, swap the center
label, highlight the matching row in an adjacent list) would need a full
server rerun per hover, which can't feel instant. Embedding the JS directly
keeps the hover interaction entirely client-side, synced both ways between
the donut and the holdings list next to it: Python only decides *what*
dataset to show (the caller picks a view and aggregates the data
beforehand), never how the hover state updates.

Deliberately `st.components.v1.html()`, not the newer `st.iframe()`: this
component was briefly on `st.iframe()` to dodge the deprecation warning, but
`st.iframe()` hardcodes `scrolling=True` on the underlying <iframe> with no
way to turn it off, which is exactly what caused a spurious *outer*
scrollbar around the whole chart+list block (on top of the *inner* list's
own, intentional one) whenever the rendered content came out even a pixel
taller than the declared height. `components.v1.html()`'s `scrolling`
parameter defaults to False, which -- per its own docs -- crops any
overflow instead of ever showing a scrollbar. That's a hard guarantee
regardless of how precise the height math below is, not just a smaller
chance of the bug; the deprecation warning is the tradeoff for it.

This module knows nothing about portfolios, ETFs, or sectors -- it just
draws a donut + list from {label, value} pairs. The domain-specific "which
aggregation goes with which tab" logic belongs with the caller.
"""

import json
import uuid

import streamlit.components.v1 as components

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"


def render_donut(
    view_name: str,
    data: list[dict],
    colors: list[str] | None = None,
    ink: str = "#ffffff",
    muted: str = "#c3c2b7",
    height: int = 420,
) -> None:
    """Render an interactive donut chart with a holdings list beside it.

    `data` is a list of {"label": str, "value": float} pairs, already
    aggregated and ordered by the caller (e.g. one row per sector, per
    position, ...). `view_name` labels the center total when nothing is
    hovered.

    Hovering a slice *or* a list row highlights that one item and keeps it
    in place -- every other slice fades to a lower opacity rather than the
    hovered slice pulling outward, and the center text swaps to that item's
    label/value. There's no small color-dot marker for "this is the
    selected one"; the distinction is the name text itself (bold, full
    ink color) against the rest (muted, regular weight). Mouse-out on
    either side reverts both to their resting state.

    The list is a visually separate panel (a divider line, its own scroll
    area) fixed to the same `height` as the donut, so the component's
    overall size never changes between a 3-row view and a 30-row one --
    only the list scrolls internally, the donut stays put.
    """
    chart_id = f"donut-{uuid.uuid4().hex}"
    labels = [d["label"] for d in data]
    values = [d["value"] for d in data]
    total = sum(values)
    colors = colors or []

    # `components.v1.html()` embeds this string as a full srcdoc document,
    # not an inline snippet -- an un-styled <body> carries the browser's
    # default ~8px margin on all sides, which on its own is enough to push
    # the real content past a height estimate that assumes zero margin.
    # Resetting it here means the container's actual rendered height is
    # just `height` (both children are explicitly that tall), not
    # `height` + whatever the browser's default chrome happened to add.
    html = f"""
    <style>html, body {{ margin:0; padding:0; overflow:hidden; }}</style>
    <div style="display:flex; gap:8px; align-items:flex-start; font-family:system-ui,-apple-system,'Segoe UI',sans-serif;">
        <div id="{chart_id}" style="flex:0 0 46%; height:{height}px; min-width:260px;"></div>
        <div id="{chart_id}-list" style="flex:1 1 auto; height:{height}px; overflow-y:auto; padding-left:20px; padding-right:6px;"></div>
    </div>
    <script src="{_PLOTLY_CDN}"></script>
    <script>
    (function() {{
        const labels = {json.dumps(labels)};
        const values = {json.dumps(values)};
        const total = {json.dumps(total)};
        const centerLabel = {json.dumps(view_name)};
        const palette = {json.dumps(colors)};
        const inkColor = {json.dumps(ink)};
        const mutedColor = {json.dumps(muted)};
        const chartId = {json.dumps(chart_id)};

        // Cycle the palette out to one color per slice up front, so the
        // donut and the list bars always agree on which color is whose,
        // regardless of how many categories this view has.
        const sliceColors = labels.map((_, i) => palette.length ? palette[i % palette.length] : '#3987e5');

        // 'nl-NL' rather than the browser's own locale (the `undefined`
        // default) -- European-style grouping (period thousands, comma
        // decimal) regardless of the viewer's own locale settings, to match
        // the rest of the app's formatting.py-driven number formatting.
        function fmtEuro(v) {{
            return '€' + v.toLocaleString('nl-NL', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
        }}

        function fmtPct(v) {{
            return v.toLocaleString('nl-NL', {{minimumFractionDigits: 2, maximumFractionDigits: 2}}) + '%';
        }}

        // Labels come from portfolio data (product names, sector/region
        // buckets) built into HTML strings via concatenation, not text
        // nodes -- escape before embedding so a name containing `<`, `>`,
        // `&`, or `"` can't break the markup.
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function cardText(label, value) {{
            return '<span style="font-size:13px;color:' + mutedColor + '">' + escapeHtml(label) + '</span><br>' +
                   '<span style="font-size:24px;color:' + inkColor + '"><b>' + fmtEuro(value) + '</b></span>';
        }}

        function hexToRgba(hex, alpha) {{
            const h = hex.replace('#', '');
            const r = parseInt(h.substring(0, 2), 16);
            const g = parseInt(h.substring(2, 4), 16);
            const b = parseInt(h.substring(4, 6), 16);
            return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
        }}

        const baseText = cardText(centerLabel, total);
        const FADED_OPACITY = 0.32;

        const trace = [{{
            type: 'pie',
            hole: 0.72,
            labels: labels,
            values: values,
            textinfo: 'none',
            hoverinfo: 'none',
            marker: {{ colors: sliceColors, line: {{ color: 'rgba(0,0,0,0.4)', width: 2 }} }},
            sort: false
        }}];

        const layout = {{
            showlegend: false,
            margin: {{t: 20, b: 20, l: 20, r: 20}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: "system-ui, -apple-system, 'Segoe UI', sans-serif" }},
            annotations: [{{
                text: baseText,
                showarrow: false,
                align: 'center',
                font: {{ size: 16, color: inkColor }}
            }}]
        }};

        Plotly.newPlot(chartId, trace, layout, {{displayModeBar: false, responsive: true}});
        const chartDiv = document.getElementById(chartId);

        // Build the holdings list: name, a weight bar (colored to match its
        // slice, filled to its share of the total), the percentage, and the
        // euro amount. No bullet-dot identity marker -- color lives on the
        // bar itself, and the "currently hovered" distinction lives on the
        // name text (bold + full ink vs. muted + regular weight), not a
        // shape.
        const listEl = document.getElementById(chartId + '-list');
        listEl.style.borderLeft = '1px solid ' + hexToRgba(inkColor, 0.12);
        const rows = labels.map((label, i) => {{
            const pct = total > 0 ? (values[i] / total * 100) : 0;
            const safeLabel = escapeHtml(label);
            const row = document.createElement('div');
            row.style.padding = '7px 8px';
            row.style.borderRadius = '6px';
            row.style.marginBottom = '2px';
            row.style.transition = 'background-color 120ms ease';
            row.innerHTML =
                '<div style="display:flex; justify-content:space-between; align-items:baseline; gap:10px;">' +
                    '<span class="donut-name" style="color:' + mutedColor + '; font-weight:600; ' +
                        'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="' + safeLabel + '">' +
                        safeLabel +
                    '</span>' +
                    '<span style="white-space:nowrap; text-align:right; flex-shrink:0;">' +
                        '<span style="color:' + inkColor + '; font-size:16px; font-weight:700;">' + fmtPct(pct) + '</span>' +
                        '<span style="color:' + mutedColor + '; font-size:14px;"> &middot; ' + fmtEuro(values[i]) + '</span>' +
                    '</span>' +
                '</div>' +
                '<div style="background:' + hexToRgba(inkColor, 0.08) + '; border-radius:3px; height:6px; max-width:60%; margin-top:2px; overflow:hidden;">' +
                    '<div class="donut-bar" style="width:' + pct + '%; height:100%; background:' + sliceColors[i] + '; border-radius:3px;"></div>' +
                '</div>';
            row.addEventListener('mouseenter', () => setActive(i));
            row.addEventListener('mouseleave', () => clearActive());
            listEl.appendChild(row);
            return row;
        }});

        function setActive(i) {{
            const fadedColors = sliceColors.map((c, idx) => idx === i ? c : hexToRgba(c, FADED_OPACITY));
            Plotly.restyle(chartId, {{ 'marker.colors': [fadedColors] }});
            Plotly.relayout(chartId, {{ 'annotations[0].text': cardText(labels[i], values[i]) }});
            rows.forEach((row, idx) => {{
                const nameEl = row.querySelector('.donut-name');
                if (idx === i) {{
                    nameEl.style.color = inkColor;
                    nameEl.style.fontWeight = '700';
                    row.style.backgroundColor = 'rgba(255,255,255,0.07)';
                }} else {{
                    nameEl.style.color = mutedColor;
                    nameEl.style.fontWeight = '600';
                    row.style.backgroundColor = 'transparent';
                }}
            }});
        }}

        function clearActive() {{
            Plotly.restyle(chartId, {{ 'marker.colors': [sliceColors] }});
            Plotly.relayout(chartId, {{ 'annotations[0].text': baseText }});
            rows.forEach((row) => {{
                const nameEl = row.querySelector('.donut-name');
                nameEl.style.color = mutedColor;
                nameEl.style.fontWeight = '600';
                row.style.backgroundColor = 'transparent';
            }});
        }}

        chartDiv.on('plotly_hover', function(evt) {{ setActive(evt.points[0].pointNumber); }});
        chartDiv.on('plotly_unhover', function() {{ clearActive(); }});
    }})();
    </script>
    """
    # With the body margin reset above, the flex row's real rendered height
    # is exactly `height` (both children are explicitly that tall) -- this
    # small buffer is just rounding-error insurance, not the thing actually
    # preventing an outer scrollbar (scrolling=False below is).
    components.html(html, height=height + 8, scrolling=False)
