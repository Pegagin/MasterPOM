"""Results: explore any saved results CSV."""
import pandas as pd
import streamlit as st

from ui.common import RESULTS_FOLDER, results_files
from ui.results_view import show_results

st.caption("Open a saved evaluation, filter it, and compare methods.")

files = results_files()
if not files:
    st.info("No results yet. Run an evaluation first.", title="Nothing to show")
    st.page_link("app_pages/evaluate.py", label="Go to Evaluate", icon=":material/science:")
    st.stop()

choice = st.selectbox("Results file (newest first)", [f.name for f in files],
                      help="CSV files in results/, written by the Evaluate page or "
                           "run_evaluation.py. One row per scenario and method.")
results = pd.read_csv(RESULTS_FOLDER / choice)

with st.expander("Filters", icon=":material/filter_list:"):
    all_methods = sorted(results["method"].unique())
    methods = st.multiselect("Methods", all_methods, default=all_methods,
                             help="Which methods to include in every table and chart.")
    failure_probabilities = sorted(results["failure_probability"].unique())
    chosen_fp = st.pills("fp (disruption chance per day)", failure_probabilities,
                         selection_mode="multi", default=failure_probabilities,
                         help="Keep only scenarios with these disruption frequencies.")
    repair_times = sorted(results["mean_repair_time"].unique())
    chosen_atr = st.pills("ATR (mean disruption length)", repair_times, selection_mode="multi",
                          default=repair_times,
                          help="Keep only scenarios with these disruption lengths.")
    instances = st.multiselect("Instances (empty = all)", sorted(results["instance"].unique()),
                               help="Restrict to single projects, for example to study one in "
                                    "the Schedule viewer afterwards.")

keep = (results["method"].isin(methods)
        & results["failure_probability"].isin(chosen_fp)
        & results["mean_repair_time"].isin(chosen_atr))
if instances:
    keep &= results["instance"].isin(instances)
st.caption(f"{int(keep.sum())} of {len(results)} rows · settings: "
           f"{results['preemption_mode'].iloc[0]}, {results['sgs'].iloc[0]} SGS, "
           f"pause by {results['pause_order'].iloc[0]}")
show_results(results[keep])
