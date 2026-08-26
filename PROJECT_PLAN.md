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
4. Pull region: headquarters country for stocks, fund name/category as a proxy for ETFs.
5. Pull top holdings for ETFs (`funds_data.top_holdings`) to support the overlap check.

## Dashboard features
- Positions table: name, quantity, value, % of portfolio.
- Allocation by asset type (stock/ETF/commodity/cash).
- Sector allocation (stocks, using GICS sector; ETFs via look-through).
- Region allocation.
- **Concentration warning**: flag any position, sector, or region bucket above a threshold (default ~15%). Shown as an ambient banner, not a separate tab.
- **ETF overlap check**: compare individual stock holdings against each ETF's top holdings list. Surface direct overlaps plainly (e.g. "AAPL is in your Vanguard All-World holding, and you also hold it directly"). Not a full weighted recalculation, just a plain flag.

## Local-version-specific requirements
- SQLite for storage.
- Save the **enriched** data per snapshot, not the raw CSV, so sector/region classifications stay comparable across dates even if Yahoo's classifications drift later.
- ISIN is the stable join key for comparing two snapshots.
- Comparison tab: pick two saved snapshots, diff them by ISIN, show what changed (new positions, removed positions, value/weight changes).

## Explicitly out of scope
Live pricing, performance tracking, returns, or risk metrics.

## Suggested repo structure
```
portfolio-dashboard/
  app.py              # Streamlit app (local version)
  enrichment.py        # ISIN resolution, sector/region/type lookup, ETF overlap logic
  storage.py            # SQLite read/write for snapshots (local version only)
  requirements.txt
  venv/
```

## Current status
Project folder, virtual environment, and empty placeholder files (app.py, enrichment.py, requirements.txt) are set up. Requirements installed: streamlit, yfinance, pandas, plotly. Next step: build the CSV parser for the DEGIRO format.
