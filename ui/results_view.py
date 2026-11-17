"""The results display shared by the Evaluate and Results pages."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from rcpsp_gp.evaluation import SUMMARY_COLUMNS, compare_methods, mean_ranks
from ui.explainers import METRIC_EXPLANATIONS

METRIC_TITLES = {
    "makespan_increase": "Makespan increase",
    "percent_over_lower_bound": "% over lower bound",
    "first_repair_increase": "Increase at 1st repair",
    "total_start_deviation": "Start deviation",
    "mean_delay": "Mean delay",
    "max_delay": "Max delay",
    "interruption_count": "Interruptions",
    "wasted_work": "Lost work",
    "disruption_count": "Disruptions met",
    "disrupted_days": "Disrupted days",
    "capacity_utilisation": "Utilisation",
    "mean_repair_milliseconds": "ms per repair",
}


def numeric_results(results: pd.DataFrame) -> pd.DataFrame:
    """Results with every metric column numeric (CSV blanks become NaN)."""
    results = results.copy()
    for key, _, _ in SUMMARY_COLUMNS:
        results[key] = pd.to_numeric(results[key], errors="coerce")
    return results


def method_summary(results: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key, _, _ in SUMMARY_COLUMNS]
    grouped = results.groupby("method")
    summary = grouped[keys].mean()
    summary.insert(0, "kind", grouped["method_kind"].first())
    summary.insert(1, "runs", grouped.size())
    ranks = mean_ranks(results.to_dict("records"))
    summary["Mean rank"] = [ranks[method] for method in summary.index]
    return summary.sort_values("makespan_increase").rename(columns=METRIC_TITLES).round(3)


def show_results(results: pd.DataFrame) -> None:
    if results.empty:
        st.warning("No rows to show.")
        return
    results = numeric_results(results)
    summary = method_summary(results)
    method_order = list(summary.index)
    chart_height = 22 * len(method_order) + 40

    st.subheader("Mean per method")
    st.dataframe(summary, alt="Mean metrics per method, best makespan increase first")
    with st.expander("What the columns mean", icon=":material/help:"):
        st.markdown(METRIC_EXPLANATIONS)

    bar_data = summary.reset_index()[["method", "kind", "Makespan increase"]]
    st.altair_chart(
        alt.Chart(bar_data).mark_bar().encode(
            x=alt.X("Makespan increase:Q", title="mean makespan increase (days)"),
            y=alt.Y("method:N", sort=method_order, title=None),
            color=alt.Color("kind:N", title="method kind"),
            tooltip=["method", "kind", alt.Tooltip("Makespan increase", format=".2f")])
        .properties(height=chart_height),
        alt=f"Mean makespan increase per method; best is {method_order[0]}")

    st.subheader("By disruption setting")
    st.caption("Mean makespan increase for each combination of fp (how often) and ATR (how long).")
    settings = [f"fp={fp:g}, ATR={atr:g}" for fp, atr in
                zip(results["failure_probability"], results["mean_repair_time"])]
    by_setting = (results.assign(setting=settings)
                  .groupby(["method", "setting"])["makespan_increase"].mean().reset_index())
    heatmap = alt.Chart(by_setting).mark_rect().encode(
        x=alt.X("setting:N", title=None),
        y=alt.Y("method:N", sort=method_order, title=None),
        color=alt.Color("makespan_increase:Q", title="mean increase",
                        scale=alt.Scale(scheme="orangered")),
        tooltip=["method", "setting", alt.Tooltip("makespan_increase", format=".2f")])
    labels = alt.Chart(by_setting).mark_text(fontSize=11).encode(
        x="setting:N", y=alt.Y("method:N", sort=method_order),
        text=alt.Text("makespan_increase:Q", format=".1f"))
    st.altair_chart((heatmap + labels).properties(height=chart_height),
                    alt="Mean makespan increase per method and disruption setting")

    st.subheader("Head-to-head")
    standard = [m for m in method_order if summary.loc[m, "kind"] == "standard"]
    reference = st.selectbox("Compare every method with", method_order,
                             index=method_order.index(standard[0]) if standard else 0,
                             help="Paired comparison on the same scenarios. Defaults to the "
                                  "best standard rule.")
    records = results.to_dict("records")
    comparisons = []
    for method in method_order:
        if method == reference:
            continue
        outcome = compare_methods(records, method, reference)
        comparisons.append({"method": method, "better": outcome["better"],
                            "equal": outcome["equal"], "worse": outcome["worse"],
                            "Wilcoxon p": round(outcome["p_value"], 4),
                            "significant (p < 0.05)": bool(outcome["p_value"] < 0.05)})
    if comparisons:
        st.dataframe(pd.DataFrame(comparisons), hide_index=True,
                     alt=f"Scenarios where each method beats, ties or loses to {reference}")
        long = pd.DataFrame(comparisons).melt(id_vars="method",
                                              value_vars=["better", "equal", "worse"],
                                              var_name="outcome", value_name="scenarios")
        st.altair_chart(
            alt.Chart(long).mark_bar().encode(
                x=alt.X("scenarios:Q", stack="normalize",
                        title=f"share of scenarios compared with {reference}"),
                y=alt.Y("method:N", sort=method_order, title=None),
                color=alt.Color("outcome:N", sort=["better", "equal", "worse"],
                                scale=alt.Scale(domain=["better", "equal", "worse"],
                                                range=["#54a24b", "#bab0ac", "#e45756"])),
                tooltip=["method", "outcome", "scenarios"])
            .properties(height=22 * len(comparisons) + 40),
            alt=f"Share of scenarios each method wins, ties or loses against {reference}")
        st.caption("Better = shorter makespan in the same scenario. The Wilcoxon signed-rank "
                   "test checks whether the difference could be chance.")

    pool_rows = results[(results["method_kind"] == "evolved_pool")
                        & results["pool_choice_counts"].notna()
                        & results["pool_choice_counts"].astype(str).str.strip().ne("")]
    if not pool_rows.empty:
        st.subheader("GP-pool: whose repair was adopted")
        counts = np.sum([[int(c) for c in str(text).split()]
                         for text in pool_rows["pool_choice_counts"]], axis=0)
        st.dataframe(pd.DataFrame({"heuristic": [f"GP#{i}" for i in range(1, len(counts) + 1)],
                                   "repairs": counts,
                                   "share": (counts / max(counts.sum(), 1)).round(3)}),
                     hide_index=True, alt="How often each pool member's repair was adopted")
        st.caption("A heuristic that is never adopted adds nothing to the pool, for example "
                   "because it always ranks activities like a better member.")

    st.download_button("Download these rows as CSV", results.to_csv(index=False),
                       file_name="results.csv", mime="text/csv", icon=":material/download:")
