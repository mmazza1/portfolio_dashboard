"""Parse DEGIRO Portfolio.csv exports into a clean DataFrame.

DEGIRO's header row is misleading: "Lokale waarde" sits one position before
the column that actually holds the local value (the header for the currency
column in between is blank). So this parses positionally against the
confirmed real export layout rather than trusting header text:

    Product, Symbool/ISIN, Aantal, Slotkoers, currency, Lokale waarde, Waarde in EUR
"""

import io

import pandas as pd

_COLUMNS = ["product", "isin", "quantity", "close_price", "currency", "local_value", "value_eur"]
_NUMERIC_COLUMNS = ["quantity", "close_price", "local_value", "value_eur"]


def _parse_european_number(value):
    """Convert a European-formatted number ("1.234,56") to float."""
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


def parse_degiro_csv(file) -> pd.DataFrame:
    """Parse a DEGIRO Portfolio.csv export.

    `file` can be a path, file-like object, or bytes. Returns one row per
    line item, including the cash line (flagged via `is_cash`), with columns:
    product, isin, quantity, close_price, currency, local_value, value_eur,
    is_cash.
    """
    if isinstance(file, bytes):
        file = io.BytesIO(file)

    try:
        raw = pd.read_csv(file, dtype=str, header=0, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(file, "seek"):
            file.seek(0)
        raw = pd.read_csv(file, dtype=str, header=0, encoding="latin-1")

    if len(raw.columns) != len(_COLUMNS):
        raise ValueError(
            f"Expected {len(_COLUMNS)} columns in DEGIRO CSV, found {len(raw.columns)}: "
            f"{list(raw.columns)}"
        )

    out = raw.copy()
    out.columns = _COLUMNS

    out["product"] = out["product"].str.strip()
    out["isin"] = out["isin"].str.strip().replace("", None)
    out["currency"] = out["currency"].str.strip()
    for col in _NUMERIC_COLUMNS:
        out[col] = out[col].apply(_parse_european_number)

    out["is_cash"] = out["isin"].isna()

    return out[_COLUMNS + ["is_cash"]]


if __name__ == "__main__":
    df = parse_degiro_csv(r"C:\Users\Matteo\Downloads\Portfolio.csv")
    print(df.to_string())

    cash_row = df[df["is_cash"]].iloc[0]
    assert cash_row["value_eur"] == 361.62
    assert cash_row["currency"] == "EUR"

    alphabet = df[df["isin"] == "US02079K3059"].iloc[0]
    assert alphabet["quantity"] == 1.0
    assert alphabet["close_price"] == 346.96
    assert alphabet["currency"] == "USD"
    assert alphabet["local_value"] == 346.96
    assert alphabet["value_eur"] == 297.21

    nvidia = df[df["isin"] == "US67066G1040"].iloc[0]
    assert nvidia["currency"] == "EUR"
    assert nvidia["local_value"] == 1085.88

    assert df["is_cash"].sum() == 1
    assert len(df) == 11

    print("\nSmoke test passed.")
