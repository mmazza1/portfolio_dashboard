"""Pure computation for the ETF rebalancing calculator: given a per-ETF
sector/region weight matrix and a user-defined target distribution, solve
for buy/sell amounts. No API calls and nothing cached here -- enrichment.py
owns pulling and caching external data; this module only does linear algebra
on numbers already sitting in memory, so it can be tested against made-up
weight vectors with no network access at all.

All three tools operate purely in EUR value terms (not unit counts) -- the
EUR-to-units conversion via current price is a display concern for the
caller, not part of the math here.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls


def build_weight_matrix(etf_df: pd.DataFrame, weights_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build a (ETF ticker x category) weight matrix and a matching current-value
    series from an ETF-only enriched dataframe. `weights_col` is the column
    holding each row's {category: weight} dict, e.g. "enrich_sector_weights".
    """
    categories = sorted({category for weights in etf_df[weights_col] for category in (weights or {})})

    matrix = pd.DataFrame(
        [[(row_weights or {}).get(c, 0.0) for c in categories] for row_weights in etf_df[weights_col]],
        index=etf_df["enrich_ticker"],
        columns=categories,
    )
    values = pd.Series(etf_df["value_eur"].values, index=etf_df["enrich_ticker"])
    return matrix, values


def unreachable_categories(matrix: pd.DataFrame, target: dict[str, float]) -> list[str]:
    """Target categories with a target weight above zero that no held ETF provides at all."""
    covered = set(matrix.columns[matrix.sum(axis=0) > 0])
    return [category for category, weight in target.items() if weight > 0 and category not in covered]


def buy_to_target(matrix: pd.DataFrame, values: pd.Series, target: dict[str, float]) -> dict:
    """New-money-only rebalance: non-negative EUR purchase amounts per ETF
    that move the portfolio's category weights as close as possible to
    `target`, without selling anything. Solved as non-negative least squares:
    each ETF's weight vector is a column, and we solve for the non-negative
    combination of purchases that best matches the EUR gap between current
    and target category exposure (target category value based on the
    *current* total, so the problem stays linear).
    """
    categories = matrix.columns
    total_value = values.sum()
    target_vec = np.array([target.get(c, 0.0) for c in categories])
    current_vec = matrix.values.T @ values.values
    gap = target_vec * total_value - current_vec

    buy, _residual = nnls(matrix.values.T, gap)
    buy_series = pd.Series(buy, index=matrix.index)

    new_values = values + buy_series
    achieved = pd.Series(matrix.values.T @ new_values.values / new_values.sum(), index=categories)

    return {
        "buy_eur": buy_series[buy_series > 1e-6].sort_values(ascending=False),
        "achieved_weights": achieved,
        "unreachable": unreachable_categories(matrix, target),
    }


def quick_rebalance(matrix: pd.DataFrame, values: pd.Series, target: dict[str, float]) -> dict:
    """Buy-and-sell rebalance: deltas per ETF (total value held stays fixed,
    no position goes negative) that move category weights as close as
    possible to `target`. Needs both bounds (can't sell more than you hold)
    and a linear equality constraint (deltas must sum to zero), which plain
    NNLS can't express -- this uses SLSQP instead.

    The optimization runs in weight-fraction space (delta / total_value),
    not raw EUR: SLSQP's finite-difference gradient badly underflows when the
    variable is O(1000) EUR but the objective is O(0.01) squared-weight units,
    which makes it wrongly "converge" at the starting point (delta=0) without
    taking a single real step. Rescaling both onto the same order of
    magnitude fixes that.
    """
    categories = matrix.columns
    total_value = values.sum()
    target_vec = np.array([target.get(c, 0.0) for c in categories])
    weights = matrix.values
    current_frac = values.values / total_value

    def objective(delta_frac):
        achieved = weights.T @ (current_frac + delta_frac)
        return float(np.sum((achieved - target_vec) ** 2))

    bounds = [(-cf, None) for cf in current_frac]
    constraints = [{"type": "eq", "fun": lambda delta_frac: np.sum(delta_frac)}]
    result = minimize(objective, x0=np.zeros(len(current_frac)), method="SLSQP", bounds=bounds, constraints=constraints)

    delta_series = pd.Series(result.x * total_value, index=matrix.index)
    new_values = values + delta_series
    achieved = pd.Series(weights.T @ new_values.values / total_value, index=categories)

    return {
        "delta_eur": delta_series[delta_series.abs() > 1e-6].sort_values(),
        "achieved_weights": achieved,
        "unreachable": unreachable_categories(matrix, target),
        "converged": bool(result.success),
    }


# Hand-built once, kept separate per dimension since sector and region use
# different category vocabularies. Categorical only -- never a specific
# ticker recommendation, to stay clear of investment-advice territory.
_SECTOR_ETF_SUGGESTIONS = {
    "Technology": "a Technology sector ETF",
    "Healthcare": "a Healthcare sector ETF",
    "Financials": "a Financials sector ETF",
    "Consumer Goods": "a Consumer Discretionary or Consumer Staples sector ETF",
    "Energy + Utilities": "an Energy or Utilities sector ETF",
    "Other": "a Real Estate, Materials, Industrials, or Communication Services sector ETF",
}

_REGION_ETF_SUGGESTIONS = {
    "United States": "a US-focused ETF",
    "South America": "a Latin America-focused ETF",
    "Europe": "a Europe-focused ETF",
    "Asia": "an Asia-Pacific or Emerging Markets ETF",
    "Other": "a broader global or frontier-markets ETF",
}


def new_etf_suggestions(unreachable: list[str], dimension: str) -> list[str]:
    """Categorical fund-category suggestions for target categories the
    current ETF holdings structurally can't reach. Never a specific ticker.
    """
    table = _SECTOR_ETF_SUGGESTIONS if dimension == "sector" else _REGION_ETF_SUGGESTIONS
    labels = [table.get(category, f"an ETF with {category} exposure") for category in unreachable]
    return [
        f"{label[:1].upper()}{label[1:]} would help close the **{category}** gap."
        for category, label in zip(unreachable, labels)
    ]
