"""Parse a DEGIRO Account Statement (Account.csv) export into a transaction
ledger, and reconstruct historical share counts per ISIN from it -- the
data Portfolio.csv (a single point-in-time snapshot) can't provide on its
own, needed for the Performance view's value-over-time chart.

Like Portfolio.csv (see parser.py), DEGIRO's Account.csv header is
misleading: "Mutatie" and "Saldo" each actually label a *currency* column,
with the real amount column immediately after (blank in the header).
Parsed positionally against the confirmed real export layout:

    Datum, Tijd, Valutadatum, Product, ISIN, Omschrijving, FX,
    <mutatie currency>, <mutatie amount>, <saldo currency>, <saldo amount>, Order Id

Three row shapes matter for reconstructing share history and real cash
movement, all in the Omschrijving column, all Dutch:
    "Koop N @ P CCY"     -- bought N shares at price P
    "Verkoop N @ P CCY"  -- sold N shares at price P
A third, "SPLIT AANPASSING: N <name> @ P CCY (ISIN)", covers reverse
splits / ticker changes -- DEGIRO represents these as the *old* ISIN's
holding dropping to some count (often 0) and a *new* ISIN's holding being
set, rather than a clean buy/sell -- confirmed in real data as a company
cycling through several ISINs via repeated reverse splits. Handled by
treating N as the new *absolute* share count for that ISIN as of that
date, not a delta -- the alternative (tracking one logical security across
however many ISINs a split chain touches) isn't attempted; each ISIN's own
history is what matters for pricing anyway, and yfinance typically has no
price history left for a delisted intermediate ISIN regardless (see
`performance.py`, which already tolerates and flags unpriceable ISINs).

A fourth row type, "DEGIRO Transactiekosten en/of kosten van derden" (the
per-trade fee, a separate ledger line from the trade itself), carries no
share-count information but is real money spent -- confirmed in real test
data to matter: `performance.py`'s reconstructed gain overstated a real
DEGIRO-reported figure by roughly the total fees paid, since fees were
previously invisible to the whole pipeline. Captured as its own event
(`delta=0`, no share effect, but a real `cash_amount`) so it flows through
the same real-cash-moved accounting as a trade, without pretending it
bought or sold anything. DEGIRO's separate annual/exchange connection fees
("DEGIRO Aansluitingskosten...") have no ISIN on their row and are
excluded here for that reason alone (this module only ever looks at rows
with an ISIN) -- a deliberate-enough omission not to need special-casing:
they're a flat account cost, not tied to any one position's performance.

Every other row type in the export (cash sweeps, dividends, FX conversion
legs, interest, options expiry) is irrelevant to share counts and
deliberately ignored here -- this module only answers "how many shares of
ISIN X were held on date Y" and "how much real cash moved through it."
Options rows (Product/ISIN filled, but the ISIN isn't a real equity/ETF)
aren't special-cased either: they match the same Koop/Verkoop pattern as a
normal trade, but the option ISIN will simply fail ticker resolution
downstream in `performance.py` and get excluded from pricing there, same
as a delisted split-chain ISIN -- its real cash effect still gets counted,
just via `performance.unpriced_cash_flow` instead of the ordinary path.

Real external cash movement -- what actually crossed the boundary between
your bank and DEGIRO, as opposed to money moving between DEGIRO's own
internal trading balance and its "flatex" cash-sweep sub-account -- is a
separate, non-ISIN concern handled by `cash_deposit_events` (see its own
docstring), not this function.
"""

import io
import re

import pandas as pd

_COLUMNS = [
    "date", "time", "value_date", "product", "isin", "description",
    "fx_rate", "mutation_currency", "mutation_amount",
    "balance_currency", "balance_amount", "order_id",
]

_TRADE_PATTERN = re.compile(r"^(Koop|Verkoop) (\d+) @ ([\d.,]+) (\w+)")
_SPLIT_PATTERN = re.compile(r"^SPLIT AANPASSING: (\d+) ")
_FEE_DESCRIPTION = "DEGIRO Transactiekosten en/of kosten van derden"

# Real external cash movement only -- money that actually crossed the
# boundary between the user's bank and DEGIRO. "iDEAL Deposit" is a
# completed incoming transfer; "SEPA Instant Terugstorting" a real
# withdrawal back out (confirmed present in real test data, EUR 500).
# Deliberately excludes "Reservation iDEAL" (an internal reservation/
# settlement pairing for the *same* iDEAL Deposit money -- DEGIRO credits
# a pending deposit's funds as usable immediately via a positive
# Reservation iDEAL, then reverses it with a negative one once the iDEAL
# Deposit itself posts a day or two later; the two net to ~0 over time, so
# counting them too would double the real deposit) and "Degiro Cash Sweep
# Transfer" / "Overboeking ... flatexDEGIRO Bank" (internal transfers
# between DEGIRO's own trading balance and its interest-bearing
# sub-account, not new or returned money -- the existing PROJECT_PLAN.md
# to-do notes already flagged these as needing to net to zero, not count
# as deposits/withdrawals).
_DEPOSIT_DESCRIPTIONS = {"iDEAL Deposit", "SEPA Instant Terugstorting"}


def _parse_european_number(value):
    """Convert a European-formatted number ("1.234,56") to float. Same
    convention as parser.py's helper -- duplicated rather than imported,
    keeping this module independent (same pattern as formatting.py /
    rebalance.py elsewhere in this codebase)."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_account_csv(file) -> pd.DataFrame:
    """Parse a DEGIRO Account.csv export. `file` can be a path, file-like
    object, or bytes. One row per ledger line, sorted oldest first (the raw
    export is newest-first).
    """
    if isinstance(file, bytes):
        file = io.BytesIO(file)

    try:
        raw = pd.read_csv(file, dtype=str, header=0, encoding="utf-8-sig", keep_default_na=False)
    except UnicodeDecodeError:
        if hasattr(file, "seek"):
            file.seek(0)
        raw = pd.read_csv(file, dtype=str, header=0, encoding="latin-1", keep_default_na=False)

    if len(raw.columns) != len(_COLUMNS):
        raise ValueError(
            f"Expected {len(_COLUMNS)} columns in DEGIRO Account.csv, found {len(raw.columns)}: "
            f"{list(raw.columns)}"
        )

    out = raw.copy()
    out.columns = _COLUMNS

    out["date"] = pd.to_datetime(out["date"], format="%d-%m-%Y")
    out["value_date"] = pd.to_datetime(out["value_date"], format="%d-%m-%Y")
    out["product"] = out["product"].str.strip()
    out["isin"] = out["isin"].str.strip().replace("", None)
    out["description"] = out["description"].str.strip()
    for col in ("mutation_amount", "balance_amount", "fx_rate"):
        out[col] = out[col].apply(_parse_european_number)

    # "time" (HH:MM, zero-padded) sorts correctly as a plain string -- used
    # only as the tiebreaker within a date, so several trades logged the
    # same day still replay in the order they actually happened.
    return out.sort_values(["date", "time"], kind="stable").reset_index(drop=True)


def extract_share_events(ledger: pd.DataFrame) -> pd.DataFrame:
    """Buy/sell/split/fee rows, as columns (date, isin, product, delta,
    is_reset, price, trade_currency, cash_amount, cash_currency) -- `delta`
    is a signed share change for a Koop/Verkoop row (0 for a fee row, which
    moves money but no shares), or the new absolute share count for a SPLIT
    AANPASSING row (`is_reset`).

    `price`/`trade_currency` are the per-*unit* executed price straight from
    the description text (`None` for a fee or split row, which have no such
    price) -- they're the ground truth `performance.py` cross-checks a
    resolved ticker's historical price series against, catching the case
    where the ticker "resolves" and has full price history but that history
    has been silently rebased by a reverse split (confirmed in real test
    data). `cash_amount`/`cash_currency` are the ledger's own recorded cash
    effect of the row instead (its `Mutatie` column) -- for a plain
    stock/ETF trade this is just `price * delta` (modulo rounding), but for
    anything with a per-unit contract multiplier other than 1 it isn't:
    confirmed in real test data, a 1-lot option trade's description read
    "Verkoop 1 @ 4,2 EUR" (implying a EUR 4.20 cash effect) but the
    ledger's actual `Mutatie` for that row was EUR 42.00, a 10x multiplier
    the plain "N @ P" text never reveals. Only `cash_amount` is trustworthy
    as "how much money this row actually moved."

    Every other row (no ISIN, or an ISIN but no matching description -- a
    dividend, an FX leg, a connection fee with no ISIN of its own) is
    dropped.
    """
    events = []
    for row in ledger.itertuples():
        if not row.isin or not row.description:
            continue
        if row.description == _FEE_DESCRIPTION:
            events.append(
                {
                    "date": row.date,
                    "isin": row.isin,
                    "product": row.product,
                    "delta": 0.0,
                    "is_reset": False,
                    "price": None,
                    "trade_currency": None,
                    "cash_amount": row.mutation_amount,
                    "cash_currency": row.mutation_currency,
                }
            )
            continue
        split_match = _SPLIT_PATTERN.match(row.description)
        if split_match:
            # A SPLIT AANPASSING pair is DEGIRO closing the old ISIN out
            # (selling it, receiving cash -- a positive mutation) and
            # opening the new one with the proceeds (buying it, a negative
            # mutation), always at the same timestamp -- confirmed by
            # Matteo directly for a real example: a 2-share position was
            # "sold for $0.11" as part of a split, not resized to 2. The
            # row's own leading number ("N ... @ P") is the share count
            # *being sold* on the positive-mutation leg, not what remains
            # -- using it as the new balance there was the bug (it landed
            # on the pre-close count, e.g. treating "2 sold" as "now holds
            # 2"). Only the negative-mutation (buying) leg's N is really a
            # new opening balance.
            is_closing_leg = row.mutation_amount is not None and row.mutation_amount >= 0
            events.append(
                {
                    "date": row.date,
                    "isin": row.isin,
                    "product": row.product,
                    "delta": 0.0 if is_closing_leg else float(split_match.group(1)),
                    "is_reset": True,
                    "price": None,
                    "trade_currency": None,
                    "cash_amount": row.mutation_amount,
                    "cash_currency": row.mutation_currency,
                }
            )
            continue
        trade_match = _TRADE_PATTERN.match(row.description)
        if trade_match:
            action, quantity, price_str, trade_currency = trade_match.groups()
            quantity = float(quantity)
            events.append(
                {
                    "date": row.date,
                    "isin": row.isin,
                    "product": row.product,
                    "delta": quantity if action == "Koop" else -quantity,
                    "is_reset": False,
                    "price": _parse_european_number(price_str),
                    "trade_currency": trade_currency,
                    "cash_amount": row.mutation_amount,
                    "cash_currency": row.mutation_currency,
                }
            )
    return pd.DataFrame(
        events,
        columns=[
            "date", "isin", "product", "delta", "is_reset",
            "price", "trade_currency", "cash_amount", "cash_currency",
        ],
    )


def cash_deposit_events(ledger: pd.DataFrame) -> pd.Series:
    """Net real EUR deposited (positive) or withdrawn (negative) per date --
    see `_DEPOSIT_DESCRIPTIONS` for exactly which row types count and why.
    Always EUR: both real row types in real test data were EUR-denominated
    (DEGIRO's iDEAL/SEPA rails are EUR-only), so unlike `cash_amount`
    elsewhere in this module, no currency conversion is needed here.

    Distinct from `extract_share_events`'s `cash_amount` on a Koop/Verkoop
    row: that's money moving between DEGIRO's own cash balance and a
    specific position (what got invested), this is money moving between
    the user's own bank and DEGIRO (what got deposited) -- the same euro
    can show up in both, once as a deposit, once (later, often days later)
    as the funding for a purchase, and they answer different questions.
    """
    real = ledger[ledger["description"].isin(_DEPOSIT_DESCRIPTIONS)]
    return real.groupby(real["date"].dt.normalize())["mutation_amount"].sum()


# "Dividend" is the gross payment; "Dividendbelasting" the withheld tax on
# it (its own separate row, already negative) -- both real cash that
# actually moved, unlike enrichment.py's dividend_yield, which is a
# forward-looking estimate derived from Yahoo's payment history, not from
# anything this specific account received.
_DIVIDEND_DESCRIPTIONS = {"Dividend", "Dividendbelasting"}


def dividend_events(ledger: pd.DataFrame) -> pd.DataFrame:
    """Real dividend cash flows -- gross payments and withheld tax, as their
    own rows (date, isin, product, description, cash_amount, cash_currency).
    Every row has an ISIN in real test data (a dividend is always tied to a
    specific holding), but rows without one are dropped defensively anyway,
    consistent with `extract_share_events`.
    """
    rows = ledger[ledger["description"].isin(_DIVIDEND_DESCRIPTIONS) & ledger["isin"].notna()]
    return (
        rows[["date", "isin", "product", "description", "mutation_amount", "mutation_currency"]]
        .rename(columns={"mutation_amount": "cash_amount", "mutation_currency": "cash_currency"})
        .reset_index(drop=True)
    )


def share_history(events: pd.DataFrame) -> dict[str, pd.Series]:
    """Per ISIN, a running share-count series indexed by event date (one
    point per date something changed -- a step function). Callers reindex
    onto whatever daily calendar they need and forward-fill themselves,
    since "daily" means different things to different callers.
    """
    history = {}
    for isin, group in events.groupby("isin"):
        group = group.sort_values("date")
        running = 0.0
        points: dict = {}
        for row in group.itertuples():
            running = row.delta if row.is_reset else running + row.delta
            points[row.date] = running
        history[isin] = pd.Series(points).sort_index()
    return history


if __name__ == "__main__":
    ledger = parse_account_csv(r"C:\Users\Matteo\Downloads\Account.csv")
    print(f"{len(ledger)} ledger rows, {ledger['date'].min().date()} to {ledger['date'].max().date()}")

    events = extract_share_events(ledger)
    print(f"\n{len(events)} share events across {events['isin'].nunique()} ISINs")

    history = share_history(events)
    for isin, series in history.items():
        product = events.loc[events["isin"] == isin, "product"].iloc[-1]
        print(f"\n{product} ({isin}):")
        print(series.to_string())
