# Portfolio Diversification Dashboard, Project Plan

## Purpose
A tool to give insight into portfolio diversification: which regions, which sectors, which equity types (stock/ETF/commodity). Not a live price tracker, not a performance tool. Built from DEGIRO CSV exports.

## Build order
1. **Local version first.** Adds snapshot saving and a comparison tab between two dates. Uses SQLite. Single user, no auth needed.
2. **Public version later.** Same enrichment pipeline, wrapped statelessly, no database, no saving. Hosted on Streamlit Community Cloud so friends can upload their own CSV and get a snapshot with nothing stored.

## The DEGIRO CSV format
Columns: Product, Symbool/ISIN, Aantal (quantity), Slotkoers (close price), currency, local value, Waarde in EUR. Uses European number formatting (comma as decimal separator). No ticker symbol and no asset type in the raw export, both need to be resolved separately. One row is a cash line with no ISIN.

## Enrichment pipeline (per position)
1. Resolve ISIN to a Yahoo Finance ticker (via Yahoo's search endpoint). Fail gracefully when it can't be resolved, don't crash the whole run.
2. Pull asset type via `quoteType` (Stock, ETF, etc). This needs to generalize to any friend's holdings, not just mine, so don't hardcode by product name where avoidable.
3. Pull sector for stocks directly from Yahoo Finance. For ETFs, use `funds_data.sector_weightings` for a look-through breakdown instead of a single sector.
4. Pull region:
   - **Stocks**: headquarters country via Yahoo Finance `.info`. Known limitation: this picks up legal domicile, not actual business exposure (e.g. Nebius is HQ'd in Amsterdam but isn't a Dutch business). Treated as an accepted approximation for stocks, edge cases are rare.
   - **ETFs**: real country look-through via the `justetf-scraping` package (unofficial, scrapes justETF), keyed by ISIN. This is the accurate source since it reflects actual fund composition, not domicile or listing exchange. Yahoo Finance has no country/region weighting field for funds, and Financial Modeling Prep's equivalent endpoint requires a paid plan, so justETF is the only free, precise option for European-listed UCITS ETFs.
   - Results are cached in `fund_regions.csv` (ISIN, Country, Percentage), one scrape per ISIN, ever, reused on every later run instead of re-scraping. Wrap the scrape call in try/except; on failure, mark region as "Unknown" rather than crashing the pipeline. Only call this for rows where AssetType is ETF.
5. Pull top holdings for ETFs (`funds_data.top_holdings`) to support the overlap check.

## Dashboard features
- Positions table: name, quantity, value, % of portfolio.
- Allocation by asset type (stock/ETF/commodity/cash).
- Sector allocation (stocks, using GICS sector; ETFs via look-through).
- Region allocation.
- **Concentration warning**: flag any position, sector, or region bucket above a threshold, each independently adjustable via its own sidebar slider (asset default 10%, sector default 25%, region default 40%). The region check uses continent-level buckets (same grouping as the Allocation tab's region pie chart), not raw countries -- a portfolio spread across many European countries shouldn't dodge the check just because no single country crosses the threshold. Respects the top-of-page Scope toggle: in whole-portfolio scope the asset check excludes ETFs (inherently diversified, not a concentration risk the way a single stock is) and sector/region cover the whole portfolio; in ETFs-only scope the asset check is skipped entirely (no per-position warning), and only sector/region concentration is checked, scoped to ETF-only value -- same as the Allocation tab's charts. Shown as an ambient banner, not a separate tab.
- **ETF overlap check**: compare individual stock holdings against each ETF's top holdings list. Surface direct overlaps plainly (e.g. "AAPL is in your Vanguard All-World holding, and you also hold it directly"). Not a full weighted recalculation, just a plain flag.
- **Target allocation drift**: let the user define a target mix (e.g. 70% stock / 20% bond / 10% cash, or target region weights), then flag drift from it. Reuses the same threshold-flag pattern as the concentration warning, just comparing actual vs. user-defined target instead of actual vs. a fixed cap. Works off a single snapshot, no history needed, so this fits both the local and public versions.
- **ESG score surfacing**: Yahoo Finance exposes `esgScores` per stock. Show as an optional column in the positions table, not as a judgment or rating, just visible data the user can choose to look at or ignore.

## ETF-only view (built)
A "Scope" filter button (Whole portfolio / ETFs only) sits at the top of the page, right after the file is loaded. When ETFs only is selected, the sector and region charts on the Allocation tab recompute from ETF holdings alone; renormalization to 100% of ETF-only value falls out automatically from the same weighted-breakdown math, no extra logic needed. Reuses the same sector/region enrichment data already pulled per ETF. The toggle is hidden entirely for a portfolio with no ETF holdings, and the charts show "No ETF holdings to show" rather than an empty chart if selected anyway. Only affects sector/region (asset type and the positions table stay whole-portfolio, since ETF-only asset type would trivially be 100% ETF).

## Rebalancing calculator (built)
Three related tools, all driven by a user-defined target distribution (by sector or by region). All three operate on ETFs only, since sector/region tilting only makes sense for funds, not individual stocks. Each uses the sector/region weight vectors already pulled per ETF in the enrichment step. Pure computation lives in `rebalance.py` (no API calls, no caching), tested against made-up weight vectors before ever touching live data. Both dimensions are bucketed down for a usable target form: Region uses continent-level buckets (United States/South America/Europe/Asia/Other, same map as the Allocation tab's Region view) instead of 30+ raw countries; Sector uses 6 buckets instead of all 11 GICS sectors: Technology, Healthcare, Financials, Consumer Goods (merges Consumer Cyclical + Consumer Defensive), Energy + Utilities (merged), and Other (Real Estate, Basic Materials, Industrials, Communication Services). Only the rebalance calculator is scoped down this way -- the Allocation tab's sector chart and the concentration/drift checks still use the full 11 GICS sectors.

In the app, this is its own "Rebalance" tab (Positions / Allocation / Rebalance) rather than a sidebar widget. Inside: a Sector/Region dimension toggle, then per-category number inputs (defaulting to current ETF-only weights, largest-remainder rounded) with a running total, wrapped in an `st.form` with a "Calculate" button so the solve only runs on submit, not on every slider tick. Below the form, a Buy-to-target / Quick rebalance / New ETF suggestions selector switches between views of the same calculated result.

1. **Buy-to-target calculator** (built): given the target distribution and current ETF holdings, calculates non-negative EUR purchase amounts per ETF (new money only, nothing sold) to move the portfolio as close as possible to the target. Solved as non-negative least squares (`scipy.optimize.nnls`): each ETF's weight vector as a column, solving for the purchase amounts that best close the EUR gap between current and target category exposure. **Feasibility caveat** (built): categories with target weight > 0 that no held ETF provides at all are flagged before the results table, not after.

2. **Quick rebalance** (built): same target and weight data, but allows both buying and selling across existing ETFs, with total value held fixed and no position going negative. This needs both bounds and a linear equality constraint, which plain NNLS can't express, so it's solved with `scipy.optimize.minimize(method="SLSQP")` instead. Non-obvious gotcha hit during build: SLSQP's finite-difference gradient badly underflows when the optimization variable is raw EUR (hundreds/thousands) but the objective is squared-weight units (~0.01-0.1) -- it silently "converges" at the zero-trade starting point without taking a single real step. Fixed by optimizing in weight-fraction space (delta / total value) instead of raw EUR, then rescaling the result back to EUR.

3. **New ETF suggestions** (built): identifies the categories #1 and #2 can't reach and matches them against a small hand-built reference table, kept separate per dimension (sector categories vs. continent categories) since they use different vocabularies. Output stays categorical ("a Technology sector ETF would help close this gap"), never a specific ticker, to stay clear of investment-advice territory.

## Local-version-specific requirements
- SQLite for storage.
- Save the **enriched** data per snapshot, not the raw CSV, so sector/region classifications stay comparable across dates even if Yahoo's classifications drift later.
- ISIN is the stable join key for comparing two snapshots.
- Comparison tab: pick two saved snapshots, diff them by ISIN, show what changed (new positions, removed positions, value/weight changes).
- Trend charts: once more than two snapshots exist, a line chart of sector/region weight over time across all saved snapshots, not just a two-point before/after comparison. Natural extension of the comparison tab once enough history has built up.

## Explicitly out of scope
Live pricing, performance tracking, returns, or risk metrics.

## Suggested repo structure
```
portfolio-dashboard/
  app.py              # Streamlit app (local version)
  enrichment.py        # ISIN resolution, sector/region/type lookup, ETF overlap logic
  rebalance.py          # Pure-computation rebalancing calculator (no API calls, no caching)
  storage.py            # SQLite read/write for snapshots (local version only)
  fund_regions.csv     # Cache: ISIN -> country/region weightings, scraped from justETF
  requirements.txt
  venv/
```

## Current status
CSV parsing, enrichment (ticker/asset type/sector/region/top holdings), and the dashboard are built in `parser.py`, `enrichment.py`, and `app.py`, plus rebalancing math in `rebalance.py`: positions table, allocation tab (asset type/sector/region, with a Region-vs-Country filter button on the region chart, and a Scope toggle for whole-portfolio vs. ETFs-only), concentration warning, ETF overlap check, target allocation drift (sidebar-editable target table per asset-type or region, with a sum-must-equal-100% check and a severity-colored drift banner: info under 10%, orange 10-15%, red above 15%), and the Rebalance tab (Buy-to-target / Quick rebalance / New ETF suggestions, `scipy`-based, see Rebalancing calculator). ETF region uses real justETF look-through (`justetf-scraping`, cached in `fund_regions.csv`); stock region is plain Yahoo HQ country, corrected via `overrides.csv` where needed. Region is modeled as a `region_weights` dict for both asset types, same pattern as `sector_weights`. ESG score surfacing is blocked on Yahoo Finance no longer returning ESG data (see Dashboard features). Next open item: the local-version SQLite snapshot/comparison/trend-chart work (`storage.py`, not yet started).