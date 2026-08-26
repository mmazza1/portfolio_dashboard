# Portfolio Diversification Dashboard

A Streamlit dashboard that gives insight into portfolio diversification — which regions, sectors, and asset types (stock/ETF/commodity/cash) your holdings are spread across. Built from DEGIRO CSV exports.

This is not a live price tracker, performance tool, or risk metrics tool. It's purely about diversification.

## Features

- **Positions table**: name, quantity, value, % of portfolio.
- **Allocation by asset type**: stock / ETF / commodity / cash.
- **Sector allocation**: GICS sector for stocks, look-through breakdown for ETFs.
- **Region allocation**: headquarters country for stocks, fund name/category as a proxy for ETFs.
- **Concentration warning**: flags any position, sector, or region above an adjustable threshold (default 15%).
- **ETF overlap check**: flags when a stock you hold directly also shows up in one of your ETFs' top holdings.

## How it works

1. Upload a DEGIRO `Portfolio.csv` export.
2. The CSV is parsed into a clean table of positions (`parser.py`).
3. Each position's ISIN is resolved to a Yahoo Finance ticker, then enriched with asset type, sector, region, and (for ETFs) top holdings (`enrichment.py`).
4. The dashboard renders the enriched data as tables, charts, and warning banners (`app.py`).

Region and sector classifications are best-effort approximations (see `overrides.csv` for manual corrections to specific ISINs where the automated lookup gets it wrong).

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

Then upload your DEGIRO `Portfolio.csv` export in the browser tab that opens.

## Project status

Currently a stateless, single-run dashboard (no data is saved between runs). See `PROJECT_PLAN.md` for the roadmap, including a planned local version with SQLite-backed snapshot saving and a comparison tab between two dates.

## Out of scope

Live pricing, performance tracking, returns, and risk metrics.
