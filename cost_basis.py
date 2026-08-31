"""FIFO realized profit/loss, per ISIN, computed directly from the
ledger's own recorded cash effects (`ledger.extract_share_events`'
`cash_amount`) -- not from market prices at all.

This is what makes it possible to get a real number for a position
`performance.py` can't price -- an option, a delisted reverse-split-chain
ISIN -- as long as it's fully closed: a closed position's entire economic
outcome already happened in cash, visible directly in the ledger, no
yfinance lookup required. Confirmed as the missing piece behind a real
gap: Matteo's DEGIRO app reported a Total P&L that the dashboard's
value-change-based figure couldn't match, and the unexplained remainder
traced to exactly this -- a fully-closed, unpriceable position (the AIRWA
reverse-split chain) contributing a real loss that was simply invisible
everywhere, neither counted nor excluded-and-flagged.

FIFO, not average cost -- Matteo's explicit choice when this feature was
first scoped (see PROJECT_PLAN.md's Performance view section): the oldest
open lot is always consumed first by a sale.

A SPLIT AANPASSING pair (see ledger.py's docstring, and its real-bug-fix
entry) is treated the same as an ordinary sell-then-buy: the closing leg
(`is_reset=True`, `cash_amount` positive) realizes whatever gain/loss the
old ISIN's lots had built up, using its `cash_amount` as proceeds and
fully liquidating every open lot for that ISIN (confirmed in real test
data: a closing leg's described quantity always matched the ISIN's entire
pre-close balance, so "close everything" is exact here, not an
approximation). The paired opening leg (`is_reset=True`, `cash_amount`
negative) opens a fresh lot on the *new* ISIN, cost basis = that same
cash -- so a rename chain's cost basis carries through unbroken across
however many ISINs it touches, rather than resetting to zero at each hop.

Per-trade fees aren't attributed to one specific lot -- a fee row's exact
same-timestamp ordering relative to its trade isn't reliably recoverable
once the ledger's been sorted, and DEGIRO's own export doesn't carry an
Order Id link through `ledger.py`'s event extraction. Instead, each ISIN's
total lifetime fees are split between realized and unrealized by the
fraction of that ISIN's lifetime volume that's been sold vs. is still
held. This can slightly misattribute a fee to the wrong *individual* lot
for a partially-sold position with very unevenly-sized fees, but the
ISIN's combined total (realized + unrealized) is exact regardless of how
the split lands.
"""

from collections import deque

import pandas as pd


def _build_fx_cache(events: pd.DataFrame) -> dict:
    """One historical `<currency>EUR=X` series per non-EUR currency
    appearing anywhere in `events`, each covering the *entire* dataset's
    date range -- not a per-row window. `fifo_pnl` processes ISINs via
    `groupby`, not chronologically, so a cache built lazily around
    whichever row happens to trigger it first would only cover dates near
    that one row; every other row needing the same currency, at any other
    date, would then look up nothing and silently get dropped. Caught this
    exact bug in real test data: it zeroed out a real, unrelated Apple
    trade's contribution to `fifo_pnl` entirely.
    """
    from performance import _cached_daily_series  # local import: avoids a hard cross-module dependency at load time

    start = events["date"].min() - pd.Timedelta(days=14)
    end = events["date"].max() + pd.Timedelta(days=1)
    currencies = {c for c in events["cash_currency"].dropna().unique() if c and c != "EUR"}
    # `_cached_daily_series`, not `_daily_series` directly: `fifo_pnl` (and
    # so this) gets called once per P&L snapshot -- several per Timeframe
    # pill, several pills per session -- and this already requests the
    # *widest* possible range (the whole dataset), so routing it through
    # the shared cache means every other, narrower request for the same
    # currency pair is served from this one fetch instead of triggering
    # its own.
    return {currency: _cached_daily_series(f"{currency}EUR=X", start, end) for currency in currencies}


def _to_eur(amount: float, currency: str | None, date: pd.Timestamp, fx_cache: dict) -> float | None:
    if not currency or currency == "EUR":
        return amount
    fx = fx_cache.get(currency)
    if fx is None:
        return None
    rate = fx.asof(date)
    if pd.isna(rate):
        return None
    return amount * rate


def fifo_pnl(events: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict[str, dict]:
    """Per ISIN: `{"realized_eur", "open_quantity", "open_cost_eur"}`, as
    of `as_of` (default: everything, i.e. "now") -- only events on or
    before that date are considered, so this doubles as a cumulative
    snapshot at any point in the account's history, not just today.
    `open_cost_eur` is the remaining EUR cost basis of currently-held lots
    -- the caller compares it against market value at that same date (from
    `performance.py`, for whatever ISINs it can price) to get unrealized
    P&L on top of this module's realized figure.
    """
    if as_of is not None:
        events = events[events["date"] <= as_of]
    if events.empty:
        return {}
    fx_cache = _build_fx_cache(events)
    result: dict[str, dict] = {}

    for isin, group in events.groupby("isin"):
        group = group.sort_values("date")
        lots: deque[list] = deque()  # each item: [quantity, cost_eur]
        realized = 0.0
        total_fees_eur = 0.0
        total_bought = 0.0
        total_sold = 0.0

        for row in group.itertuples():
            if row.cash_amount is None or pd.isna(row.cash_amount):
                continue
            cash_eur = _to_eur(row.cash_amount, row.cash_currency, row.date, fx_cache)
            if cash_eur is None:
                continue

            if row.delta == 0 and not row.is_reset:
                # A fee row -- see module docstring for why this isn't
                # attributed to one lot directly.
                total_fees_eur += cash_eur
                continue

            if row.delta > 0:
                # Opening: a Koop, or a SPLIT AANPASSING opening leg.
                lots.append([row.delta, -cash_eur])
                total_bought += row.delta
                continue

            # Closing: a Verkoop (delta < 0, an exact quantity) or a SPLIT
            # AANPASSING closing leg (delta forced to 0 by the ledger fix,
            # identified here by is_reset -- fully liquidates every lot).
            qty_to_close = -row.delta if row.delta < 0 else sum(lot[0] for lot in lots)
            if qty_to_close <= 0:
                continue
            proceeds_total = cash_eur
            remaining = qty_to_close
            while remaining > 1e-9 and lots:
                lot = lots[0]
                take = min(lot[0], remaining)
                cost_fraction = lot[1] * (take / lot[0]) if lot[0] else 0.0
                proceeds_fraction = proceeds_total * (take / qty_to_close) if qty_to_close else 0.0
                realized += proceeds_fraction - cost_fraction
                lot[0] -= take
                lot[1] -= cost_fraction
                remaining -= take
                if lot[0] <= 1e-9:
                    lots.popleft()
            if remaining > 1e-9:
                # Sold more than was ever bought via a Koop/opening-leg
                # lot -- confirmed real: an options contract sold to
                # *open* (writing/selling a put for premium, no prior
                # purchase) has nothing to match against. There's no cost
                # basis for a quantity never "bought" in this model, so
                # the unmatched proceeds are pure realized gain, not
                # silently dropped.
                realized += proceeds_total * (remaining / qty_to_close) if qty_to_close else 0.0
            total_sold += qty_to_close  # both the matched and unmatched portions are now fully closed out

        # A pure short (sold to open, e.g. a written option -- nothing
        # ever bought) has total_bought == 0 but is still fully closed, so
        # its fees should land entirely in "realized," not stuck at a
        # nonsensical 0/0-implied "0% realized" that would otherwise route
        # them all into an open-cost figure for a position with 0 open
        # quantity.
        sold_fraction = (total_sold / total_bought) if total_bought else (1.0 if total_sold > 0 else 0.0)
        realized += total_fees_eur * sold_fraction  # total_fees_eur <= 0, so this only ever reduces realized

        open_quantity = sum(lot[0] for lot in lots)
        open_cost_eur = sum(lot[1] for lot in lots) - total_fees_eur * (1 - sold_fraction)

        result[isin] = {
            "realized_eur": realized,
            "open_quantity": open_quantity,
            "open_cost_eur": open_cost_eur,
        }

    return result


def total_pnl(events: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict:
    """Cumulative P&L as of `as_of` (default: now) -- not timeframe-
    filtered on its own, but a *difference* between two calls at different
    dates gives a windowed figure (see `windowed_pnl`), matching the shape
    of DEGIRO's own "Total P&L," the original point of comparison that
    motivated building this module. Combines `fifo_pnl`'s realized figure
    with unrealized P&L (market value at `as_of` minus remaining cost
    basis) for every open position this pipeline can still price. An open
    position it *can't* price (none in real test data -- everything
    unpriceable in this dataset turned out to already be fully closed once
    the SPLIT AANPASSING sign bug was fixed, see ledger.py) is flagged in
    `unpriced_open_isins` rather than silently left out of the total, same
    policy as `performance.py`.
    """
    # avoids a hard cross-module dependency at load time
    from performance import _cached_daily_series, _resolve_currency, _resolve_ticker

    as_of = (as_of or pd.Timestamp.now()).normalize()
    pnl = fifo_pnl(events, as_of=as_of)
    realized_eur = sum(v["realized_eur"] for v in pnl.values())
    unrealized_eur = 0.0
    unpriced_open_isins: list[str] = []

    for isin, v in pnl.items():
        if v["open_quantity"] <= 1e-9:
            continue
        ticker = _resolve_ticker(isin)
        if ticker is None:
            unpriced_open_isins.append(isin)
            continue
        currency = _resolve_currency(ticker)
        # `_cached_daily_series`, not `_daily_series` directly: `total_pnl`
        # gets called once per Timeframe pill (a different `as_of` each
        # time), so without a shared per-ticker cache the same stock's
        # price gets re-fetched from Yahoo on every single pill click --
        # confirmed to matter in real test data (see performance.py).
        prices = _cached_daily_series(ticker, as_of - pd.Timedelta(days=10), as_of + pd.Timedelta(days=1))
        price_as_of = prices.asof(as_of) if prices is not None else None
        if price_as_of is None or pd.isna(price_as_of):
            unpriced_open_isins.append(isin)
            continue
        if currency and currency != "EUR":
            fx = _cached_daily_series(f"{currency}EUR=X", as_of - pd.Timedelta(days=10), as_of + pd.Timedelta(days=1))
            rate = fx.asof(as_of) if fx is not None else None
            if rate is None or pd.isna(rate):
                unpriced_open_isins.append(isin)
                continue
            price_as_of *= rate
        unrealized_eur += v["open_quantity"] * price_as_of - v["open_cost_eur"]

    return {
        "realized_eur": realized_eur,
        "unrealized_eur": unrealized_eur,
        "total_eur": realized_eur + unrealized_eur,
        "unpriced_open_isins": unpriced_open_isins,
    }


def windowed_pnl(events: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> dict:
    """P&L that accrued strictly within [`window_start`, `window_end`] --
    the difference between two cumulative `total_pnl` snapshots, one as of
    `window_end` and one as of the day *before* `window_start` (so a
    position's gain on the first day of the window is still counted, not
    excluded by an off-by-one). For a window starting before any account
    activity, the "before" snapshot is naturally all-zero, so this reduces
    to plain cumulative P&L -- consistent with the "Max" timeframe meaning
    "everything."
    """
    end_pnl = total_pnl(events, as_of=window_end)
    start_pnl = total_pnl(events, as_of=window_start - pd.Timedelta(days=1))
    return {
        "realized_eur": end_pnl["realized_eur"] - start_pnl["realized_eur"],
        "unrealized_eur": end_pnl["unrealized_eur"] - start_pnl["unrealized_eur"],
        "total_eur": end_pnl["total_eur"] - start_pnl["total_eur"],
        "unpriced_open_isins": end_pnl["unpriced_open_isins"],
    }


def yearly_dividend_income(dividend_events: pd.DataFrame) -> pd.Series:
    """Net EUR dividend income actually received (gross payments minus
    withheld tax -- see `ledger.dividend_events`), grouped by calendar
    year. Real historical cash flow, unlike the existing per-position
    Dividends table (`views/dividends.py`'s fallback, `enrichment.py`'s
    `dividend_yield`), which is a forward-looking estimate derived from
    Yahoo's own payment history, not anything this specific account has
    actually received. Reuses `_build_fx_cache`/`_to_eur` -- the same
    ledger-cash-to-EUR conversion `fifo_pnl` uses, just grouped by year
    instead of matched into FIFO lots.
    """
    if dividend_events.empty:
        return pd.Series(dtype=float)
    fx_cache = _build_fx_cache(dividend_events)
    totals: dict[int, float] = {}
    for row in dividend_events.itertuples():
        eur = _to_eur(row.cash_amount, row.cash_currency, row.date, fx_cache)
        if eur is None:
            continue
        totals[row.date.year] = totals.get(row.date.year, 0.0) + eur
    return pd.Series(totals).sort_index()


if __name__ == "__main__":
    from ledger import extract_share_events, parse_account_csv

    ledger = parse_account_csv(r"C:\Users\Matteo\Downloads\Account.csv")
    events = extract_share_events(ledger)
    pnl = fifo_pnl(events)

    for isin, v in sorted(pnl.items(), key=lambda kv: -abs(kv[1]["realized_eur"])):
        product = events.loc[events["isin"] == isin, "product"].iloc[-1]
        print(
            f"{product} ({isin}): realized EUR {v['realized_eur']:.2f}, "
            f"open {v['open_quantity']:.2f} shares, open cost EUR {v['open_cost_eur']:.2f}"
        )

    print()
    result = total_pnl(events)
    print(f"Realized:   EUR {result['realized_eur']:.2f}")
    print(f"Unrealized: EUR {result['unrealized_eur']:.2f}")
    print(f"TOTAL P&L:  EUR {result['total_eur']:.2f}")
    if result["unpriced_open_isins"]:
        print(f"Unpriced open positions (not included above): {result['unpriced_open_isins']}")
