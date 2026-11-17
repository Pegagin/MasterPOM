"""Evaluate: run standard rules and evolved heuristics on the same test scenarios."""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from rcpsp_gp.dataset import make_scenarios
from rcpsp_gp.evaluation import (Method, check_baseline_reproduction, evaluate_methods,
                                 evolved_methods, standard_methods, write_rows_csv)
from rcpsp_gp.rules import STANDARD_RULES
from rcpsp_gp.simulator import SimulationSettings
from ui.common import (RESULTS_FOLDER, about, data_inputs, heuristics_files,
                       heuristics_from, heuristics_table, projects_for, simulation_inputs)
from ui.explainers import HELP, METHODS_EXPLAINER
from ui.results_view import show_results

st.caption("Every method runs the same scenarios with the same disruption streams.")

NO_FILE = "(none: standard rules only)"
choice = st.selectbox("Evolved heuristics", [NO_FILE] + [f.name for f in heuristics_files()],
                      help="A file saved by training. Its settings become the defaults below.")
heuristics, trained_with = [], {}
if choice != NO_FILE:
    heuristics, document = heuristics_from(choice)
    trained_with = document.get("metadata", {}).get("simulation", {})
    with st.expander(f"{len(heuristics)} heuristics in {choice}", icon=":material/account_tree:"):
        st.dataframe(heuristics_table(heuristics, document), hide_index=True,
                     alt="Evolved heuristics in the selected file")

data_column, simulation_column, method_column = st.columns(3)
with data_column:
    st.subheader("Data")
    data = data_inputs("evaluate", subset="test")
    scenario_seed = int(st.number_input("Scenario seed", 0, 10 ** 9, 1000,
                                        help=HELP["scenario_seed"]))
with simulation_column:
    st.subheader("Simulation")
    if trained_with:
        st.caption("Defaults are the settings the heuristics were trained with.")
    simulation = simulation_inputs(f"evaluate-{choice}", SimulationSettings(**trained_with)
                                   if trained_with else SimulationSettings())
with method_column:
    st.subheader("Methods")
    about(METHODS_EXPLAINER)
    rule_names = st.multiselect("Standard rules", list(STANDARD_RULES),
                                default=list(STANDARD_RULES), help=HELP["rules"])
    top_k, use_single, use_pool = 0, False, False
    if heuristics:
        top_k = st.slider("Top k evolved heuristics", 1, len(heuristics),
                          min(5, len(heuristics)), help=HELP["top_k_evaluate"])
        use_single = st.checkbox("Each evolved heuristic alone", value=True, help=HELP["single"])
        use_pool = st.checkbox("Evolved pool", value=True, help=HELP["pool"])

methods: List[Method] = standard_methods(rule_names)
if heuristics:
    methods += evolved_methods(heuristics, top_k, single=use_single, pool=use_pool)
default_name = (f"{Path(choice).stem if heuristics else 'standard_rules'}_{data['subset']}_"
                f"{simulation.preemption_mode}_{simulation.sgs}")
with st.container(horizontal=True, vertical_alignment="bottom"):
    output_name = st.text_input("Save results as results/<name>.csv", default_name,
                                help="A file with the same name is overwritten.")
    start = st.button("Run evaluation", type="primary", icon=":material/play_arrow:",
                      disabled=not methods)


def run_evaluation(methods: List[Method], data: Dict, scenario_seed: int,
                   simulation: SimulationSettings, output_path: Path) -> None:
    projects = projects_for(data)
    scenarios = make_scenarios(projects, data["replications"], seed=scenario_seed)
    if not scenarios:
        st.error("No instances found. Check the folder, subset and filter.")
        return
    st.info(f"{len(projects)} projects, {len(scenarios)} scenarios, {len(methods)} methods")
    with st.spinner("Control run without disruptions ..."):
        problems = check_baseline_reproduction(methods, projects, simulation)
    if problems:
        st.error("; ".join(problems), title="Control run failed")
    else:
        st.success("Every method reproduces the baseline when there are no disruptions.",
                   title="Control run passed", icon=":material/check_circle:")
    progress_bar = st.progress(0.0)

    def show_progress(done: int, total: int) -> None:
        progress_bar.progress(done / total, text=f"Scenario {done}/{total}")

    rows = evaluate_methods(methods, scenarios, simulation, log=None, progress=show_progress)
    write_rows_csv(rows, output_path)
    st.session_state["evaluate_output"] = {"rows": rows, "path": output_path.name}


if start:
    run_evaluation(methods, data, scenario_seed, simulation,
                   RESULTS_FOLDER / f"{output_name}.csv")

if "evaluate_output" in st.session_state:
    output = st.session_state["evaluate_output"]
    st.success(f"Saved {len(output['rows'])} rows to results/{output['path']}",
               title="Evaluation finished")
    show_results(pd.DataFrame(output["rows"]))
