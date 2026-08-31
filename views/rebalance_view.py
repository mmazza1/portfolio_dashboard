"""The Rebalance tab: target-mix form plus Buy-to-target / Quick rebalance /
New ETF suggestions results. Named `rebalance_view.py`, not `rebalance.py`,
to avoid colliding with the project-root `rebalance.py` -- the pure-
computation module (buy_to_target/quick_rebalance/etc.) this view calls
into, left untouched by this split.
"""

import pandas as pd
import streamlit as st

import aggregations
import rebalance
from formatting import format_eur, format_number


def render_rebalance_tab(etf_only: pd.DataFrame, cash_value: float) -> None:
    if etf_only.empty:
        st.info("No ETF holdings to rebalance.")
        return

    st.metric("Currently uninvested", format_eur(cash_value))
    rebalance_dimension = st.segmented_control(
        "Target dimension",
        ["Sector", "Region"],
        default="Sector",
        required=True,
        key="rebalance_dimension",
    )
    weights_col = "enrich_sector_weights" if rebalance_dimension == "Sector" else "enrich_region_weights"
    rebalance_source = etf_only
    if rebalance_dimension == "Sector":
        rebalance_source = etf_only.assign(
            enrich_sector_weights=etf_only["enrich_sector_weights"].apply(aggregations.common_sector_weights)
        )
    else:
        rebalance_source = etf_only.assign(
            enrich_region_weights=etf_only["enrich_region_weights"].apply(aggregations.continent_weights)
        )
    matrix, values = rebalance.build_weight_matrix(rebalance_source, weights_col)
    prices, shares = rebalance.build_trade_inputs(rebalance_source)
    raw_pct = pd.Series(matrix.values.T @ values.values / values.sum() * 100, index=matrix.columns)
    default_pct = aggregations.round_percentages(raw_pct)

    with st.form("rebalance_target_form"):
        st.caption("Set your target mix, then calculate. Defaults to your current ETF-only weights.")
        target_pct = {}
        input_cols = st.columns(min(4, len(matrix.columns)))
        for i, category in enumerate(matrix.columns):
            with input_cols[i % len(input_cols)]:
                target_pct[category] = st.number_input(
                    category,
                    min_value=0,
                    max_value=100,
                    value=int(default_pct[category]),
                    step=5,
                    key=f"rebalance_target_{rebalance_dimension}_{category}",
                )
        total_pct = sum(target_pct.values())
        if total_pct == 100:
            st.caption(f"Total: {format_number(total_pct, 0)}%")
        else:
            st.caption(f"Total: {format_number(total_pct, 0)}% -- doesn't sum to 100%")
        additional_cash = st.number_input(
            "Additional cash to invest",
            min_value=0.0,
            value=0.0,
            step=50.0,
            width=220,
            help=(
                "Only used by Buy-to-target. Combined with your currently uninvested cash into a "
                "budget -- spent optimally, not maximally: money that wouldn't improve the fit is "
                "left unspent. Leave at 0 to use your existing cash only (with a fallback suggestion "
                "if that's too little to meaningfully help)."
            ),
        )
        submitted = st.form_submit_button("Calculate")

    calculated_key = f"rebalance_calculated_{rebalance_dimension}"
    if submitted:
        st.session_state[calculated_key] = True

    if not st.session_state.get(calculated_key, False):
        return

    target = {category: pct / 100 for category, pct in target_pct.items()}
    buy_result = rebalance.buy_to_target(matrix, values, target, prices, cash_value, additional_cash)
    quick_result = rebalance.quick_rebalance(matrix, values, target, prices, shares)
    suggestions = rebalance.new_etf_suggestions(buy_result["unreachable"], rebalance_dimension.lower())

    name_by_ticker = etf_only.set_index("enrich_ticker")["enrich_name"]

    mode = st.segmented_control(
        "View",
        ["Buy-to-target", "Quick rebalance", "New ETF suggestions"],
        default="Buy-to-target",
        required=True,
        key="rebalance_mode",
    )

    def _trade_table(shares_amounts: pd.Series, eur_amounts: pd.Series, eur_label: str) -> pd.DataFrame:
        # Numeric, not pre-formatted: a pre-formatted string column
        # sorts lexicographically ("10" before "9"), not numerically
        # -- see the positions table for the same fix and why.
        # Rendered via NumberColumn(format="localized") at each call
        # site instead, which keeps the column sortable and renders
        # using the viewer's own browser locale.
        return pd.DataFrame(
            {
                "Name": [name_by_ticker[t] for t in shares_amounts.index],
                "Shares": shares_amounts.values,
                eur_label: eur_amounts.reindex(shares_amounts.index).values,
            }
        )

    if mode == "Buy-to-target":
        if buy_result["unreachable"]:
            st.warning(
                "Target includes categories no held ETF provides at all: "
                + ", ".join(f"**{c}**" for c in buy_result["unreachable"])
                + ". Buying more of your current ETFs can't close this gap -- see New ETF suggestions."
            )
        if buy_result["used_fallback_cap"]:
            estimate = buy_result["unconstrained_estimate"]
            estimate_text = (
                f"at least {format_eur(estimate, 0)}"
                if buy_result["estimate_capped"]
                else f"about {format_eur(estimate, 0)}"
            )
            st.info(
                f"Your uninvested cash ({format_eur(cash_value)}) is too little to meaningfully close this gap, "
                f"so this suggestion instead allows up to {format_eur(buy_result['budget_used'])} as an example "
                f"budget -- it actually spends {format_eur(float(buy_result['buy_eur'].sum()))} of that, only "
                "what improves the fit. Fully reaching this target would require "
                f"{estimate_text} in new purchases. Enter additional cash above and recalculate to use "
                "a specific amount instead."
            )
        else:
            st.caption(
                f"Budget: {format_eur(buy_result['budget_used'])} "
                f"(spent {format_eur(float(buy_result['buy_eur'].sum()))} -- only what improves the fit)."
            )
        if buy_result["buy_shares"].empty:
            st.info("Your current holdings are already at (or above) target -- no purchases needed.")
        else:
            st.dataframe(
                _trade_table(buy_result["buy_shares"], buy_result["buy_eur"], "Buy (EUR)"),
                width="stretch",
                hide_index=True,
                column_config={
                    "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                    "Buy (EUR)": st.column_config.NumberColumn("Buy (EUR)", format="localized"),
                },
            )

    elif mode == "Quick rebalance":
        if not quick_result["converged"]:
            st.warning("The solver didn't find an optimal whole-share solution -- treat this result as approximate.")
        if quick_result["unreachable"]:
            st.warning(
                "Target includes categories no held ETF provides at all: "
                + ", ".join(f"**{c}**" for c in quick_result["unreachable"])
                + ". No amount of buying or selling your current ETFs can close this gap -- "
                "see New ETF suggestions."
            )
        if quick_result["delta_shares"].empty:
            st.info("Your current holdings are already at target -- no trades needed.")
        else:
            share_trades = quick_result["delta_shares"]
            eur_trades = quick_result["delta_eur"]
            sell_mask = share_trades < 0
            sells = _trade_table(-share_trades[sell_mask], -eur_trades[sell_mask], "Sell (EUR)")
            buys = _trade_table(share_trades[~sell_mask], eur_trades[~sell_mask], "Buy (EUR)")
            if not sells.empty:
                st.write("**Sell**")
                st.dataframe(
                    sells,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                        "Sell (EUR)": st.column_config.NumberColumn("Sell (EUR)", format="localized"),
                    },
                )
            if not buys.empty:
                st.write("**Buy**")
                st.dataframe(
                    buys,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                        "Buy (EUR)": st.column_config.NumberColumn("Buy (EUR)", format="localized"),
                    },
                )
            st.caption(
                "Net cash flow is kept as close to zero as achievable in whole shares -- "
                "not always exactly zero."
            )

    else:
        if suggestions:
            st.warning("\n\n".join(f"- {s}" for s in suggestions))
        else:
            st.info("Your current ETF holdings can already reach this target -- no new fund categories needed.")
