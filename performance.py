"""Historical portfolio value reconstruction for the Performance view.

Turns `ledger.share_history()` (per-ISIN share counts over time) into a
single daily portfolio-value-in-EUR series, by pulling historical daily
closes per ISIN from Yahoo Finance and converting non-EUR prices via
historical `<currency>EUR=X` rates (also from Yahoo).

Value here means **market value of holdings only, not total net worth** --
it does not include cash. Reconstructing a historical cash balance needs
the full transaction ledger (deposits, dividends, fees, FX legs, interest,
cash-sweep transfers), a separate, much larger effort than this chart --
see PROJECT_PLAN.md's Performance tracker to-do notes. The caller is
expected to caption the chart accordingly.

An ISIN yfinance can't resolve a ticker for, or has no price history for
(an expired option, a delisted intermediate ISIN from a reverse-split
chain), is silently excluded from the total rather than raising -- see
`unresolved_isins` on the result, which the caller should surface, not
hide.

A ticker can also resolve successfully and have a full price history that
is still wrong for this purpose: confirmed in real test data, a penny
stock (ISIN US8314454088, "Connexa Sports Technologies Inc.") correctly
resolves to its current ticker after a rename ("YYAI"), but Yahoo's own
historical price series for that ticker has been retroactively rebased by
one or more reverse splits across its *entire* history -- a EUR 250+
"close" shows up for a date the stock actually traded around EUR 0.15,
inflating that one position's reconstructed value by roughly 1,700x for
that period. There's no split-ratio data available to correct this, so
instead every resolved ticker's price series is cross-checked against the
*real* prices actually paid (Koop/Verkoop rows in the ledger carry the
executed price) on the same dates; a mismatch beyond `_PRICE_TOLERANCE`
means the whole series is untrusted and the ISIN is excluded, same as an
unresolved one.

A position excluded this way can still be genuinely held today, though --
confirmed in real test data: two ISINs in the same reverse-split chain
(US8314455077, US8314456067, both also resolving to "YYAI") were still
held (81 and 2 shares) when the export was taken, and their *current*
market value (~EUR 64.39 combined) is real -- only the ticker's *history*
is corrupted, not its live/latest quote, since a rebase only ever distorts
the past relative to today's scale, never the reverse. Fully excluding
such a position (the original behavior) meant the reconstructed "current"
total quietly undercounted a real, currently-held position's value --
confirmed against Matteo's own DEGIRO app: EUR 9,214.48 reconstructed vs.
EUR 9,272.82 real, closing to within EUR 6 once this fix restored the
missing ~EUR 64. `portfolio_value_history` now folds each such position's
today-only value into the *last* day of the returned series (via
`_current_value_contribution`) -- the historical shape of the chart still
excludes it entirely (that part genuinely can't be reconstructed), so
there's a small, deliberate, real jump on the most recent day rather than
a smoothly wrong history. The added amount is also returned on its own
(`current_value_addon`) so the caller can caption it, not just silently
patch the number.

This restoration is skipped, falling back to full exclusion, whenever the
resolved ticker is ambiguous -- mapped to more than one still-held ISIN at
once. Confirmed necessary in real test data: the *same* reverse-split
chain above actually spans four ISINs, not two, and three of them showed
a nonzero "current" balance simultaneously (833 combined shares) -- a
real broker never lets the same position exist under multiple live ISINs
at once, so this is DEGIRO's own ledger never having cleanly closed out
the two older, superseded ones. Guessing which single ISIN (if any) is
the genuine current holding isn't attempted; see `ambiguous_tickers`.

Alongside the value series, `portfolio_value_history` also returns a daily
`flows` series -- the net EUR actually spent that day (positive) or
received (negative), straight from the ledger's own recorded amounts, not
yfinance's close or a re-derived price x quantity. `time_weighted_return_index`
uses it to back new principal (and costs) out of the % and EUR gain
figures: buying more shares raises the *value* series (correctly -- the
account really is worth more), but that's not "growth," it's just added
capital, and naively computing change on the raw value series conflates
the two. `flows` includes per-trade fees as well as the trades themselves
(both real cash moved, from `ledger.extract_share_events`'s `cash_amount`)
-- confirmed to matter in real test data: omitting fees overstated the
reconstructed gain by roughly the total fees paid, since a DEGIRO-reported
figure Matteo compared against is fee-inclusive. This still doesn't need
the full cash ledger (deposits, dividends, interest, cash-sweep transfers)
-- only the buy/sell/fee rows `ledger.py` already parses.
"""

from collections import Counter

import pandas as pd
import yfinance as yf

from enrichment import _resolve_quote


_PRICE_TOLERANCE = 5.0

# ISIN -> ticker, and ticker -> native currency, both static facts (an
# ISIN's ticker and a ticker's listing currency don't change mid-session)
# but each backed by a live network call (`yf.Search`, `yf.Ticker(...).
# fast_info`) with no caching of its own. Both get looked up repeatedly
# across this module and cost_basis.py -- once per ISIN per P&L snapshot,
# and there can be several snapshots per timeframe (see cost_basis.
# windowed_pnl) and several timeframes per session. Confirmed to matter in
# real test data: switching the Performance view's Timeframe pill took
# 12+ seconds the first time, almost entirely repeated identical lookups.
# A plain module-level dict is enough -- this is a well-known "cache
# forever within the process" case, not something that needs
# st.cache_data's invalidation machinery.
_TICKER_CACHE: dict[str, str | None] = {}
_CURRENCY_CACHE: dict[str, str | None] = {}


def _resolve_ticker(isin: str) -> str | None:
    if isin not in _TICKER_CACHE:
        quote = _resolve_quote(isin)
        _TICKER_CACHE[isin] = quote["symbol"] if quote else None
    return _TICKER_CACHE[isin]


def _resolve_currency(ticker: str) -> str | None:
    if ticker not in _CURRENCY_CACHE:
        try:
            _CURRENCY_CACHE[ticker] = yf.Ticker(ticker).fast_info.currency
        except Exception:
            _CURRENCY_CACHE[ticker] = None
    return _CURRENCY_CACHE[ticker]


def _price_series_is_plausible(prices: pd.Series, isin_events: pd.DataFrame) -> bool:
    """True unless a resolved ticker's historical price disagrees with a
    real executed price (from the ledger) by more than `_PRICE_TOLERANCE`x
    on the same date -- see module docstring. `prices` must still be in the
    ticker's native currency here (not yet FX-converted), since the ledger's
    `price`/`trade_currency` are the native execution price too.
    """
    for row in isin_events.itertuples():
        if row.price is None or pd.isna(row.price):
            continue
        yahoo_price = prices.asof(row.date)
        if pd.isna(yahoo_price) or yahoo_price <= 0:
            continue
        ratio = yahoo_price / row.price
        if ratio > _PRICE_TOLERANCE or ratio < 1 / _PRICE_TOLERANCE:
            return False
    return True


def _daily_series(ticker: str, start, end) -> pd.Series | None:
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if hist.empty:
            return None
        closes = hist["Close"]
        closes.index = closes.index.tz_localize(None).normalize()
        return closes
    except Exception:
        return None


# ticker/FX-pair -> the widest daily series fetched so far this process.
# `cost_basis.py` asks for the same ticker's price at several different
# as_of dates (one P&L snapshot per Timeframe pill) -- without this, each
# snapshot re-fetches its own narrow ~10-day window from Yahoo even though
# they're all well within a range already fetched for an earlier snapshot.
# Confirmed to matter in real test data: caching ticker/currency lookups
# alone (see `_TICKER_CACHE`/`_CURRENCY_CACHE`) still left each new
# Timeframe pill taking 3-4 seconds on top of the first one's 12+; this is
# the rest of that cost.
_WIDE_PRICE_CACHE: dict[str, pd.Series] = {}


def _cached_daily_series(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """Same contract as `_daily_series`, but shares one widening cache per
    ticker across however many (start, end) windows get asked for over the
    life of the process, instead of each call fetching its own.
    """
    cached = _WIDE_PRICE_CACHE.get(ticker)
    # `end` is routinely a day or more in the future (callers pad it so an
    # `.asof()` lookup for "today" doesn't fall exactly on the last index
    # entry) -- Yahoo never actually returns bars past today regardless of
    # what `end` is requested, so `cached.index.max()` can never literally
    # reach a future `end`. Checking against it directly meant this cache
    # never once hit in real test data: every call looked like it needed
    # fresher data and re-fetched. Capping the comparison at "today" (never
    # further) is what the cache can actually satisfy.
    effective_end = min(end, pd.Timestamp.now().normalize())
    if cached is not None and start >= cached.index.min() and effective_end <= cached.index.max():
        return cached
    fetch_start = min(start, cached.index.min()) if cached is not None else start
    fetch_end = max(end, cached.index.max()) if cached is not None else end
    # A little extra buffer beyond exactly what's needed now, so a nearby
    # future request (a slightly earlier/later as_of date) still hits this
    # same cache entry instead of triggering another fetch a few days on.
    fresh = _daily_series(ticker, fetch_start - pd.Timedelta(days=10), fetch_end + pd.Timedelta(days=10))
    if fresh is None:
        return cached  # keep whatever was cached before rather than discarding it on a transient failure
    _WIDE_PRICE_CACHE[ticker] = fresh
    return fresh


def unpriced_cash_flow(events: pd.DataFrame, unresolved_isins: list[str]) -> float:
    """Net EUR that moved through real trades (Koop/Verkoop, not SPLIT
    AANPASSING resets) in ISINs that couldn't be priced at all --
    positive means net money spent on them, negative means net money
    received back (e.g. an option sold for premium). Uses the ledger's own
    `cash_amount` directly, not price x quantity -- see
    `ledger.extract_share_events`'s docstring for why that matters for
    something like an options contract, where the plain "N @ P" text
    doesn't reveal a per-unit multiplier. Not part of `flows` / the value
    series at all -- this is money the rest of this module has no way to
    show performance for, surfaced separately so it isn't just silently
    dropped from the picture.
    """
    trade_rows = events[events["isin"].isin(unresolved_isins) & (~events["is_reset"]) & events["cash_amount"].notna()]
    if trade_rows.empty:
        return 0.0

    total = 0.0
    fx_cache: dict[str, pd.Series | None] = {}
    for row in trade_rows.itertuples():
        if not row.cash_currency or row.cash_currency == "EUR":
            total -= row.cash_amount
            continue
        if row.cash_currency not in fx_cache:
            fx_cache[row.cash_currency] = _daily_series(
                f"{row.cash_currency}EUR=X", row.date - pd.Timedelta(days=14), row.date + pd.Timedelta(days=1)
            )
        fx = fx_cache[row.cash_currency]
        rate = fx.asof(row.date) if fx is not None else None
        if rate is None or pd.isna(rate):
            continue
        total -= row.cash_amount * rate
    return total


def _current_value_contribution(
    shares: pd.Series, prices: pd.Series, currency: str | None, fetch_end: pd.Timestamp, today: pd.Timestamp
) -> float:
    """Real EUR value *today* of a position whose historical price series
    just failed the plausibility check (see the caller) -- uses only
    `prices`' single most recent point, not the untrusted series as a
    whole. Confirmed safe in real test data: a rebase only ever distorts
    *past* prices relative to today's real scale (the whole reason the
    check flags old dates as implausible), never the reverse -- `prices`'
    own latest point matched a live quote for the same ticker.
    """
    if shares.empty or shares.iloc[-1] == 0:
        return 0.0
    price_today = prices.asof(today)
    if pd.isna(price_today):
        return 0.0
    if currency and currency != "EUR":
        fx = _daily_series(f"{currency}EUR=X", today - pd.Timedelta(days=7), fetch_end)
        rate = fx.asof(today) if fx is not None else None
        if rate is None or pd.isna(rate):
            return 0.0
        price_today *= rate
    return shares.iloc[-1] * price_today


def portfolio_value_history(
    share_history: dict[str, pd.Series], events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Series, pd.Series, float, float, list[str]]:
    """Daily EUR market value of holdings from `start` to `end` inclusive,
    a same-indexed daily series of net EUR invested that day (see module
    docstring), the net EUR moved through unpriceable positions (see
    `unpriced_cash_flow`), the real current EUR value of unpriceable
    positions still held today (see `_current_value_contribution` --
    already folded into the returned value series' last day, returned
    again on its own purely so the caller can caption it), and the list of
    ISINs that couldn't be priced -- ticker resolution failed, yfinance had
    no history for it, or its price history didn't survive the
    plausibility check against real executed prices (`events`, from
    `ledger.extract_share_events`).
    """
    calendar = pd.date_range(start.normalize(), end.normalize(), freq="D")
    total = pd.Series(0.0, index=calendar)
    flows = pd.Series(0.0, index=calendar)
    unresolved: list[str] = []
    current_value_addon = 0.0

    ticker_for_isin = {isin: _resolve_ticker(isin) for isin in share_history}
    # Same ticker mapped to more than one *still-held* ISIN means a rename
    # or reverse-split chain the broker's own ledger never cleanly closed
    # out -- confirmed in real test data: three different ISINs for the
    # same underlying stock all showed a nonzero "current" balance at once
    # (833 combined shares), which can't be real; DEGIRO's ledger simply
    # never zeroed the superseded ones. There's no reliable way to tell
    # from this data alone which ISIN (if any single one) is the genuine
    # current holding, so current-value restoration below is skipped
    # entirely for every ISIN sharing an ambiguous ticker -- back to the
    # safe default (fully excluded, undercounting by a small, at least
    # bounded amount) rather than a confident but possibly badly wrong one.
    _held_tickers = [
        ticker_for_isin[isin]
        for isin, shares in share_history.items()
        if ticker_for_isin[isin] is not None and not shares.empty and shares.iloc[-1] != 0
    ]
    ambiguous_tickers = {t for t, count in Counter(_held_tickers).items() if count > 1}

    for isin, shares in share_history.items():
        ticker = ticker_for_isin[isin]
        if ticker is None:
            unresolved.append(isin)
            continue

        currency = _resolve_currency(ticker)

        # A few days of slack before the earliest date we actually need --
        # both this ISIN's first event and the caller's own `start` -- since
        # yfinance has no bar on a weekend/holiday and the reindex below
        # needs something to forward-fill from on day one.
        fetch_start = min(shares.index.min(), calendar[0]) - pd.Timedelta(days=7)
        fetch_end = calendar[-1] + pd.Timedelta(days=1)

        prices = _daily_series(ticker, fetch_start, fetch_end)
        if prices is None:
            unresolved.append(isin)
            continue

        # Every event for this ISIN -- trades, fees, splits -- not
        # pre-filtered by `price`, so fee rows (which have none, only a
        # real `cash_amount`) aren't accidentally dropped before they ever
        # reach the flows loop below. The plausibility check specifically
        # needs a real per-unit price to compare against, so it filters
        # its own input rather than filtering `isin_events` itself.
        isin_events = events[events["isin"] == isin]
        if not _price_series_is_plausible(prices, isin_events[isin_events["price"].notna()]):
            unresolved.append(isin)
            # The *historical* series is untrustworthy (that's what just
            # failed), but rebasing only ever distorts the past relative to
            # today's real scale, never the reverse -- confirmed in real
            # test data, `prices`' own most recent point matched a live
            # quote for the same ticker. So if shares are still held today
            # *and* this ticker isn't ambiguous (see `ambiguous_tickers`),
            # their current value is real and worth keeping even though the
            # chart can't show how it got there.
            if ticker not in ambiguous_tickers:
                current_value_addon += _current_value_contribution(shares, prices, currency, fetch_end, calendar[-1])
            continue

        fx = None
        if currency and currency != "EUR":
            fx = _daily_series(f"{currency}EUR=X", fetch_start, fetch_end)
            if fx is None:
                unresolved.append(isin)
                continue
            fx_daily = fx.reindex(calendar.union(fx.index)).ffill().reindex(calendar)
            prices_eur = prices * fx_daily.reindex(prices.index.union(fx_daily.index)).ffill().reindex(prices.index)
        else:
            prices_eur = prices

        shares_daily = shares.reindex(calendar.union(shares.index)).ffill().reindex(calendar).fillna(0.0)
        prices_daily = prices_eur.reindex(calendar.union(prices_eur.index)).ffill().reindex(calendar)

        total = total.add((shares_daily * prices_daily).fillna(0.0), fill_value=0.0)

        # Trades and their fees, not SPLIT AANPASSING resets (a corporate
        # action isn't new money), contribute to `flows`, using the
        # ledger's own recorded cash effect (`cash_amount`) rather than
        # re-deriving price x quantity -- the exact real amount, not an
        # approximation, and consistent with `unpriced_cash_flow`'s own use
        # of the same field. Negated so a buy or a fee (money leaving cash,
        # in exchange for holdings or for nothing) is a positive flow, a
        # sell negative -- the sign `time_weighted_return_index` expects to
        # back both out. A fee genuinely does reduce performance the same
        # way a bad trade would: it's cash spent that produced no value,
        # confirmed to matter in real test data (see module docstring).
        trade_rows = isin_events[~isin_events["is_reset"]]
        for row in trade_rows.itertuples():
            date_key = row.date.normalize()
            if date_key not in flows.index:
                continue
            fx_rate = 1.0
            if fx is not None:
                rate = fx.asof(date_key)
                if pd.isna(rate):
                    continue
                fx_rate = rate
            flows[date_key] += -row.cash_amount * fx_rate

    if current_value_addon:
        total[calendar[-1]] += current_value_addon

    unpriced = unpriced_cash_flow(events, unresolved)
    return total, flows, unpriced, current_value_addon, unresolved


def time_weighted_return_index(value: pd.Series, flows: pd.Series) -> pd.Series:
    """A cumulative growth index (starting at 1.0, same index as `value`) of
    pure price return -- the actual performance of the assets held, with
    the effect of adding or withdrawing principal (`flows`) backed out
    before compounding, so buying more shares with newly deposited cash
    doesn't read as "growth." The caller reads the ratio of this index
    between any two dates as that period's return, e.g.
    `index[end] / index[start] - 1`.

    Daily-linked, one flow per day: each day's return is
    `(V_t - flow_t) / V_(t-1) - 1`, chained multiplicatively (a
    Modified-Dietz-style approximation -- exact sub-daily timing of a trade
    within its day isn't known, only daily granularity is, same tradeoff as
    the rest of this pipeline). A day with no prior value to compare against
    (before the first holding existed) contributes a 0% leg rather than
    raising, so the index starts flat until there's actually something to
    measure a return on.
    """
    index = pd.Series(1.0, index=value.index)
    running = 1.0
    prev_value = None
    for date in value.index:
        v = value[date]
        if prev_value is not None and prev_value > 0:
            daily_return = (v - flows[date]) / prev_value - 1.0
            running *= 1 + daily_return
        index[date] = running
        prev_value = v
    return index


if __name__ == "__main__":
    from ledger import extract_share_events, parse_account_csv, share_history as build_share_history

    ledger = parse_account_csv(r"C:\Users\Matteo\Downloads\Account.csv")
    events = extract_share_events(ledger)
    history = build_share_history(events)

    value, flows, unpriced, current_value_addon, unresolved = portfolio_value_history(
        history, events, ledger["date"].min(), ledger["date"].max()
    )
    print(f"Unresolved ISINs ({len(unresolved)}): {unresolved}")
    print(f"Net EUR moved through them: {unpriced:.2f}")
    print(f"Of which, real current value still held: {current_value_addon:.2f}")
    print()
    print(value.tail(20).to_string())
    print(f"\nLatest value: EUR {value.iloc[-1]:.2f}")

    twr = time_weighted_return_index(value, flows)
    naive_pct = (value.iloc[-1] / value[value > 0].iloc[0] - 1) * 100
    twr_pct = (twr.iloc[-1] / twr[value > 0].iloc[0] - 1) * 100
    print(f"\nNaive value change (includes deposits): {naive_pct:.1f}%")
    print(f"Time-weighted return (deposits excluded): {twr_pct:.1f}%")
