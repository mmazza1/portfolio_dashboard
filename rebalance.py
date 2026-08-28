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
from scipy.optimize import Bounds, LinearConstraint, milp, minimize


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


def build_trade_inputs(etf_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """EUR price per share and current whole-share count held, both indexed
    by ticker -- needed by the whole-share solvers below, since you can only
    ever buy or sell in whole shares, not arbitrary EUR amounts.
    """
    prices = pd.Series(etf_df["value_eur"].values / etf_df["quantity"].values, index=etf_df["enrich_ticker"])
    shares = pd.Series(etf_df["quantity"].values, index=etf_df["enrich_ticker"]).round().astype(int)
    return prices, shares


def unreachable_categories(matrix: pd.DataFrame, target: dict[str, float]) -> list[str]:
    """Target categories with a target weight above zero that no held ETF
    provides meaningful exposure to (zero, or near enough to be noise).
    """
    covered = set(matrix.columns[matrix.sum(axis=0) > 1e-6])
    return [category for category, weight in target.items() if weight > 0 and category not in covered]


def _category_contribution_matrix(matrix: pd.DataFrame, prices: pd.Series) -> np.ndarray:
    """EUR contribution to each category per whole share traded of each ETF:
    shape (n_categories, n_etfs), M[j, i] = price_i * weight[i, j].
    """
    price = prices.reindex(matrix.index).values
    return (matrix.values * price[:, None]).T


def _solve_whole_shares(
    matrix: pd.DataFrame,
    values: pd.Series,
    target: dict[str, float],
    prices: pd.Series,
    lower_bounds: np.ndarray,
    cash_neutral: bool,
) -> tuple[np.ndarray, bool]:
    """Shared MILP core for both whole-share tools below. Solves for integer
    share deltas per ETF (bounded by `lower_bounds`, unbounded-ish above)
    that minimize the total *absolute* gap between achieved and target
    category weights.

    This is L1, not least-squares (L2): `scipy.optimize.milp` only supports
    linear objectives, and least-squares is quadratic, so it can't be handed
    to milp directly. L1 (sum of absolute deviations) is the standard
    linear-programming-compatible stand-in -- expressed here via one
    auxiliary non-negative variable per category (t_j >= |gap_j|, enforced
    by two inequality constraints, then minimize sum(t_j)) -- and still gets
    the mathematically optimal whole-share combination, just under a
    different norm than the continuous version used.

    When `cash_neutral`, net cash flow is bounded (can't exceed one share's
    worth of drift, since exact zero isn't generally achievable with whole
    shares of differing prices) *and* added to the objective with a tiny
    weight, purely as a tiebreaker: minimizing category gap alone leaves the
    solver free to land anywhere inside that cash-drift band, since nothing
    otherwise rewards landing closer to zero.
    """
    categories = matrix.columns
    tickers = matrix.index
    n_etf, n_cat = len(tickers), len(categories)

    total_value = values.sum()
    target_vec = np.array([target.get(c, 0.0) for c in categories])
    current_vec = matrix.values.T @ values.values
    gap = target_vec * total_value - current_vec

    price = prices.reindex(tickers).values
    category_matrix = _category_contribution_matrix(matrix, prices)
    n_extra = 1 if cash_neutral else 0

    # x = [n_1..n_k share deltas, t_1..t_m absolute category-gap auxiliaries,
    # (cash_neutral only) u = absolute net cash flow]
    c = np.concatenate([np.zeros(n_etf), np.ones(n_cat), np.full(n_extra, 1e-3)])
    integrality = np.concatenate([np.ones(n_etf), np.zeros(n_cat), np.zeros(n_extra)])

    # A share delta can't realistically exceed "spend the whole portfolio on
    # this one ETF" -- a generous but finite upper bound MILP needs to stay
    # well-posed.
    max_shares = np.ceil(total_value / np.where(price > 0, price, total_value))
    max_price = price.max() if len(price) else 0.0
    bounds = Bounds(
        lb=np.concatenate([lower_bounds, np.zeros(n_cat), np.zeros(n_extra)]),
        ub=np.concatenate([max_shares, np.full(n_cat, np.inf), np.full(n_extra, max_price)]),
    )

    identity = np.eye(n_cat)
    pad = np.zeros((n_cat, n_extra))
    constraints = [
        LinearConstraint(np.hstack([category_matrix, -identity, pad]), -np.inf, gap),
        LinearConstraint(np.hstack([-category_matrix, -identity, pad]), -np.inf, -gap),
    ]
    if cash_neutral:
        cash_row = np.concatenate([price, np.zeros(n_cat)])
        # u >= cash and u >= -cash together make u >= |cash|; minimizing u
        # (via its tiny objective weight above) then pins it to exactly
        # |cash| at the optimum, the standard L1 trick.
        constraints.append(LinearConstraint(np.concatenate([cash_row, [-1.0]]).reshape(1, -1), -np.inf, 0))
        constraints.append(LinearConstraint(np.concatenate([-cash_row, [-1.0]]).reshape(1, -1), -np.inf, 0))

    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not result.success:
        return np.zeros(n_etf), False
    return np.round(result.x[:n_etf]).astype(int), True


# Buy-to-target's fallback path, used only when no additional cash is
# entered and the current uninvested cash barely moves the needle: how much
# relative improvement counts as "enough" to skip the fallback, how big a
# hypothetical budget to try instead, and how generous a ceiling to use when
# estimating "how much would fully close the gap." All hand-picked, not
# derived.
_INSUFFICIENT_IMPROVEMENT_THRESHOLD = 0.15
_FALLBACK_BUDGET_FRACTION_OF_PORTFOLIO = 0.25
_ESTIMATE_BUDGET_CAP_MULTIPLE = 10


def _achieved_weights(current_vec: np.ndarray, weights: np.ndarray, purchases: np.ndarray, total_value: float) -> np.ndarray:
    new_total = total_value + purchases.sum()
    if new_total <= 0:
        return np.zeros_like(current_vec)
    return (current_vec + weights.T @ purchases) / new_total


def _distance(achieved: np.ndarray, target_vec: np.ndarray) -> float:
    return float(np.sum((achieved - target_vec) ** 2))


def _continuous_budgeted_solve(
    weights: np.ndarray, current_vec: np.ndarray, total_value: float, target_vec: np.ndarray, budget: float
) -> np.ndarray:
    """Continuous (fractional-EUR) non-negative purchases that minimize
    squared distance to target, bounded by `sum(purchases) <= budget` -- an
    *inequality*, not equality, so the solver can leave money unspent if
    spending more wouldn't improve the fit (optimal use of cash, not maximal
    use of it).

    Runs in fraction-of-budget units, not raw EUR: SLSQP's finite-difference
    gradient badly underflows when the variable is O(1000) EUR but the
    objective is O(0.01) squared-weight units, and silently "converges" at
    the zero-purchase starting point without taking a real step (same issue
    documented on quick_rebalance's SLQSP-era history). Scaling by budget
    keeps both on a comparable order of magnitude.
    """
    n = weights.shape[0]
    if budget <= 0:
        return np.zeros(n)

    def objective(x_frac):
        purchases = x_frac * budget
        return _distance(_achieved_weights(current_vec, weights, purchases, total_value), target_vec)

    bounds = [(0, None)] * n
    constraints = [{"type": "ineq", "fun": lambda x_frac: 1.0 - x_frac.sum()}]

    result = minimize(objective, x0=np.zeros(n), method="SLSQP", bounds=bounds, constraints=constraints)
    return np.maximum(result.x, 0.0) * budget


def _greedy_spend_leftover(
    weights: np.ndarray, current_vec: np.ndarray, total_value: float, target_vec: np.ndarray, price: np.ndarray, shares: np.ndarray, leftover: float
) -> np.ndarray:
    """Starting from a whole-share purchase vector, repeatedly buy whichever
    single additional share (across held ETFs) improves distance-to-target
    the most per euro spent, until no affordable share meaningfully helps.
    """
    shares = shares.copy()
    current_distance = _distance(_achieved_weights(current_vec, weights, shares * price, total_value), target_vec)

    while True:
        affordable = np.where(price <= leftover + 1e-9)[0]
        if len(affordable) == 0:
            break
        best_i, best_ratio, best_distance = None, 1e-9, current_distance
        for i in affordable:
            trial = shares.copy()
            trial[i] += 1
            distance = _distance(_achieved_weights(current_vec, weights, trial * price, total_value), target_vec)
            ratio = (current_distance - distance) / price[i]
            if ratio > best_ratio:
                best_i, best_ratio, best_distance = i, ratio, distance
        if best_i is None:
            break
        shares[best_i] += 1
        leftover -= price[best_i]
        current_distance = best_distance

    return shares


def buy_to_target(
    matrix: pd.DataFrame,
    values: pd.Series,
    target: dict[str, float],
    prices: pd.Series,
    cash_available: float,
    additional_cash: float = 0.0,
) -> dict:
    """New-money-only rebalance: whole shares to buy per ETF (non-negative,
    nothing sold) that move the portfolio's category weights as close as
    possible to `target`, spent from `cash_available` (the portfolio's
    existing uninvested cash) plus any `additional_cash` the user opts to
    contribute.

    Each candidate budget is solved in two steps: a continuous SLSQP solve
    (see `_continuous_budgeted_solve`) gets the fractional-EUR allocation
    that best fits the target under a `sum(purchases) <= budget` bound, then
    that's floored to whole shares and the leftover change is spent greedily
    (see `_greedy_spend_leftover`) -- you can only ever buy whole shares.

    If the user entered additional cash, that (plus existing cash) is the
    budget, no fallback logic involved. Otherwise: try the existing cash
    alone first; if that barely moves the needle (relative improvement under
    `_INSUFFICIENT_IMPROVEMENT_THRESHOLD`, or it can't even afford one
    helpful share), it's not a useful suggestion, so re-solve against a
    hypothetical budget instead (`_FALLBACK_BUDGET_FRACTION_OF_PORTFOLIO` of
    total portfolio value) and separately report what fully closing the gap
    would cost, as context (`unconstrained_estimate`, capped at a generous
    but finite multiple of portfolio value -- see the fallback branch below
    for why a truly unconstrained estimate isn't well-defined here).
    """
    categories = matrix.columns
    tickers = matrix.index
    total_value = values.sum()
    target_vec = np.array([target.get(c, 0.0) for c in categories])
    weights = matrix.values
    current_vec = weights.T @ values.values
    price = prices.reindex(tickers).values

    baseline_distance = _distance(current_vec / total_value, target_vec)

    def solve_and_refine(budget: float) -> np.ndarray:
        continuous = _continuous_budgeted_solve(weights, current_vec, total_value, target_vec, budget)
        floored = np.floor(continuous / np.where(price > 0, price, np.inf)).astype(int)
        leftover = max(budget - float((floored * price).sum()), 0.0)
        return _greedy_spend_leftover(weights, current_vec, total_value, target_vec, price, floored, leftover)

    unconstrained_estimate = None
    estimate_capped = False
    used_fallback_cap = False
    additional_cash = additional_cash or 0.0

    if additional_cash > 0:
        budget_used = cash_available + additional_cash
        final_shares = solve_and_refine(budget_used)
    else:
        budget_used = cash_available
        step1_shares = solve_and_refine(budget_used)
        distance_1 = _distance(_achieved_weights(current_vec, weights, step1_shares * price, total_value), target_vec)
        relative_improvement = (
            1.0 if baseline_distance <= 1e-9 else (baseline_distance - distance_1) / baseline_distance
        )
        can_afford_one_share = cash_available > 0 and (price <= cash_available).any()

        if relative_improvement >= _INSUFFICIENT_IMPROVEMENT_THRESHOLD and can_afford_one_share:
            final_shares = step1_shares
        else:
            used_fallback_cap = True
            budget_used = total_value * _FALLBACK_BUDGET_FRACTION_OF_PORTFOLIO
            final_shares = solve_and_refine(budget_used)

            # How much would actually be needed to fully close the gap, as
            # context alongside the capped suggestion above. There's no
            # finite "exact" answer in general -- buying only ever dilutes
            # existing holdings toward target, asymptotically, so driving
            # distance to exactly zero can require unbounded spend. Solving
            # against a large-but-finite ceiling instead of truly
            # unconstrained keeps this a sane number rather than an SLSQP
            # excursion into the millions; `estimate_capped` says whether
            # even that ceiling wasn't enough, i.e. this is a lower bound.
            estimate_cap = total_value * _ESTIMATE_BUDGET_CAP_MULTIPLE
            estimate_purchases = _continuous_budgeted_solve(weights, current_vec, total_value, target_vec, estimate_cap)
            unconstrained_estimate = float(estimate_purchases.sum())
            estimate_capped = unconstrained_estimate >= estimate_cap * 0.99

    buy_shares_series = pd.Series(final_shares, index=tickers)
    buy_eur_series = buy_shares_series * price

    new_values = values + buy_eur_series
    achieved = pd.Series(weights.T @ new_values.values / new_values.sum(), index=categories)

    mask = buy_shares_series > 0
    # Sort by EUR amount, then line shares up in that same order -- sorting
    # each independently could put them in different orders (a small share
    # count at a high price can be a bigger EUR amount than a large share
    # count at a low price), silently misaligning the two columns on display.
    buy_eur_sorted = buy_eur_series[mask].sort_values(ascending=False)
    return {
        "buy_shares": buy_shares_series.reindex(buy_eur_sorted.index),
        "buy_eur": buy_eur_sorted,
        "achieved_weights": achieved,
        "unreachable": unreachable_categories(matrix, target),
        "converged": True,
        "budget_used": budget_used,
        "used_fallback_cap": used_fallback_cap,
        "unconstrained_estimate": unconstrained_estimate,
        "estimate_capped": estimate_capped,
    }


def quick_rebalance(
    matrix: pd.DataFrame, values: pd.Series, target: dict[str, float], prices: pd.Series, shares: pd.Series
) -> dict:
    """Buy-and-sell rebalance: whole-share deltas per ETF (net cash flow kept
    near zero, no position sold below zero) that move category weights as
    close as possible to `target`. See `_solve_whole_shares` for the solver
    details.
    """
    categories = matrix.columns
    tickers = matrix.index
    lower_bounds = -shares.reindex(tickers).values
    delta_shares, converged = _solve_whole_shares(
        matrix, values, target, prices, lower_bounds=lower_bounds, cash_neutral=True
    )
    delta_shares_series = pd.Series(delta_shares, index=tickers)
    delta_eur_series = delta_shares_series * prices.reindex(tickers).values

    new_values = values + delta_eur_series
    achieved = pd.Series(matrix.values.T @ new_values.values / values.sum(), index=categories)

    mask = delta_shares_series != 0
    delta_eur_sorted = delta_eur_series[mask].sort_values()
    return {
        "delta_shares": delta_shares_series.reindex(delta_eur_sorted.index),
        "delta_eur": delta_eur_sorted,
        "achieved_weights": achieved,
        "unreachable": unreachable_categories(matrix, target),
        "converged": converged,
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
