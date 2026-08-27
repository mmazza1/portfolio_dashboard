# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

A tool to give insight into portfolio diversification (regions, sectors, equity types) built from DEGIRO CSV exports. Not a live price tracker, not a performance/returns tool — see "Explicitly out of scope" below.

## Commands

Run the app:
```
streamlit run app.py
```

There is no test framework or linter configured. `parser.py` and `enrichment.py` each have a smoke test in their `if __name__ == "__main__"` block, run directly:
```
python parser.py
python enrichment.py
```
Note: these smoke tests hardcode a Windows path (`C:\Users\Matteo\Downloads\Portfolio.csv`) and specific expected values (ISINs, prices) from that file — they only work with that exact sample export present at that path, not as general-purpose tests.

## Architecture

Three-module pipeline, wired together in `app.py`:

1. **`parser.py`** — `parse_degiro_csv()` turns a DEGIRO `Portfolio.csv` export into a DataFrame. DEGIRO's header row is misleading (a blank currency-column header shifts "Lokale waarde" one position off from the column it actually labels), so columns are assigned **positionally**, not by trusting the header text. Numbers are European-formatted (`.` thousands, `,` decimal) and need conversion. The cash line has no ISIN and is flagged via `is_cash`.

2. **`enrichment.py`** — `enrich_positions()` takes the parsed DataFrame and does one Yahoo Finance lookup per unique ISIN (`enrich_isin()`), caching by ISIN so repeated positions aren't looked up twice. Never raises on a bad lookup; failures are carried in an `error` field per row so one broken ISIN doesn't stop the whole run. Key design points:
   - **Sector** is always represented as a `sector_weights` dict (`{sector: weight}`), even for a plain stock (`{sector: 1.0}`). This lets the dashboard sum weighted exposure the same way regardless of whether the position is a single-sector stock or a look-through ETF breakdown. ETF sector keys come back snake_cased from Yahoo and are remapped (`_SECTOR_KEY_MAP`) to the Title Case strings a stock's `sector` field uses, so both sources join on the same key.
   - **Region** is a best-effort approximation, not authoritative. A stock's HQ country from Yahoo is often just a tax/incorporation domicile (Netherlands, Ireland, Luxembourg, Bermuda, Cayman Islands, Switzerland) rather than where the business operates — for those, the code falls back to the region of the primary listing exchange (`_EXCHANGE_REGION_MAP`) instead. For ETFs, region is inferred by keyword-matching the fund name/category against `_REGION_KEYWORDS` (checked most-specific-first, e.g. "Emerging Markets" before "Global").
   - **`overrides.csv`** (`isin,ticker,region`) is the manual escape hatch for cases the automated logic gets wrong or can't resolve at all — a hand-supplied ticker skips/steers the Yahoo search, a hand-supplied region overrides the computed one. Check it when a lookup looks wrong before trying to fix the general-case logic.
   - ETF **top holdings** are pulled for the overlap check (see below).

3. **`app.py`** — Streamlit UI. Loads and enriches (cached via `@st.cache_data` on file bytes, since enrichment does live network calls), then renders:
   - A positions table and three allocation breakdowns (asset type, sector, region), all computed by summing `value_eur` weighted by `sector_weights`/`region`/`asset_type` across positions.
   - **Concentration warning**: flags any single position, sector, or region bucket above a user-adjustable threshold (sidebar slider, default 15%).
   - **ETF overlap check**: flags when a directly-held stock's ticker also appears in an owned ETF's top-holdings list (plain flag, not a weighted look-through recalculation).

Data flows one direction: `parser` → `enrichment` → `app`. There is no `storage.py` yet (see below).

## Roadmap context (see `PROJECT_PLAN.md`)

This is currently the **stateless/public-version** shape only. The plan calls for a local version first (SQLite-backed, snapshot saving, a comparison tab diffing two saved snapshots by ISIN) before a public version deployed statelessly to Streamlit Community Cloud. If asked to add snapshot/comparison features, check `PROJECT_PLAN.md` for the intended design (e.g. save *enriched* data per snapshot, not raw CSV, so sector/region stay comparable even if Yahoo's classifications drift later; ISIN is the stable join key).

## Explicitly out of scope

Live pricing, performance tracking, returns, or risk metrics — don't add these even if they seem like natural extensions.
