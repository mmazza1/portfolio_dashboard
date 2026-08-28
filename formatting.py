"""Centralized European-style number/currency formatting -- period as the
thousands separator, comma as the decimal separator (e.g. "9.220,32") --
used everywhere a euro value or percentage is displayed.

Python's f-string formatting (`f"{x:,.2f}"`) and Streamlit's built-in
"euro"/"percent" `st.column_config.NumberColumn` formats both produce
US-style grouping ("9,220.32"), and NumberColumn's format spec has no way
to swap separators. `locale.setlocale()` is process-global, not always
correctly configured on hosting platforms, and would affect the whole
process rather than just these display strings. So this is a small
hand-rolled formatter instead, used consistently everywhere rather than
reformatted ad hoc at each call site -- if the format ever needs to change
again, it changes in one place.
"""

_THOUSANDS_PLACEHOLDER = "@"


def _swap_separators(formatted: str) -> str:
    """"9,220.32" (US grouping) -> "9.220,32" (European)."""
    swapped = formatted.replace(",", _THOUSANDS_PLACEHOLDER)
    swapped = swapped.replace(".", ",")
    swapped = swapped.replace(_THOUSANDS_PLACEHOLDER, ".")
    return swapped


def format_number(value: float, decimals: int = 2) -> str:
    """European-style grouped number, no currency or percent sign: "9.220,32"."""
    return _swap_separators(f"{value:,.{decimals}f}")


def format_eur(value: float, decimals: int = 2) -> str:
    """Euro-prefixed European-style amount: "EUR9.220,32" with the real euro sign."""
    return "€" + format_number(value, decimals)


def format_pct(value: float, decimals: int = 0) -> str:
    """Percent from a fraction (0-1), European-style: format_pct(0.1547) -> "15%"."""
    return format_number(value * 100, decimals) + "%"
