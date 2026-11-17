"""Train: evolve heuristics with GP, with live progress."""
from pathlib import Path
from typing import Dict, List

import streamlit as st

from rcpsp_gp.dataset import make_scenarios
from rcpsp_gp.gp import infix_formula, save_heuristics
from rcpsp_gp.simulator import SimulationSettings
from rcpsp_gp.training import TrainingSettings, train_heuristics
from ui.charts import fitness_chart, heuristic_tree
from ui.common import (HEURISTICS_FOLDER, data_inputs, heuristics_from, heuristics_table,
                       projects_for, simulation_inputs, training_inputs)

st.caption("The GP population is the repair pool: at every disruption each heuristic plans a "
           "what-if repair, the best plan is adopted, and the scores become the fitness.")

data_column, simulation_column, gp_column = st.columns(3)
with data_column:
    st.subheader("Data")
    data = data_inputs("train")
with simulation_column:
    st.subheader("Simulation")
    simulation = simulation_inputs("train")
with gp_column:
    st.subheader("Genetic programming")
    try:
        training = training_inputs()
    except ValueError as error:
        st.error(str(error), title="Invalid GP settings")
        st.stop()

default_name = (f"{Path(data['folder']).name}_{simulation.preemption_mode}_{simulation.sgs}_"
                f"{training.evolution_basis}_seed{training.seed}")
with st.container(horizontal=True, vertical_alignment="bottom"):
    output_name = st.text_input("Save heuristics as heuristics/<name>.json", default_name,
                                help="A file with the same name is overwritten.")
    start = st.button("Start training", type="primary", icon=":material/play_arrow:")


def run_training(data: Dict, simulation: SimulationSettings, training: TrainingSettings,
                 output_path: Path) -> None:
    training_projects = projects_for(data, "train")
    validation_projects = projects_for(data, "validation")
    training_scenarios = make_scenarios(training_projects, data["replications"],
                                        seed=training.seed)
    validation_scenarios = make_scenarios(validation_projects, data["replications"],
                                          seed=training.seed + 1)
    if not training_scenarios:
        st.error("No training instances found. Check the folder and the filter.")
        return
    st.info(f"{len(training_projects)} training projects ({len(training_scenarios)} scenarios), "
            f"{len(validation_projects)} validation projects "
            f"({len(validation_scenarios)} scenarios)")
    progress_bar = st.progress(0.0, text="Starting ...")
    chart_slot = st.empty()

    def show_progress(phase: str, done: int, total: int, history: List[Dict]) -> None:
        if phase == "training":
            progress_bar.progress(done / total, text=f"Training: project {done}/{total}, "
                                                     f"{len(history)} generations bred")
            if history:
                chart_slot.altair_chart(fitness_chart(history),
                                        alt="Fitness per generation during training")
        else:
            progress_bar.progress(done / total, text=f"Validating candidate {done}/{total}")

    result = train_heuristics(training_scenarios, validation_scenarios, simulation, training,
                              log=None, progress=show_progress)
    metadata = dict(result.settings, data=data["folder"], split=data["split"],
                    resource_constrained_only=data["constrained"],
                    replications=data["replications"], history=result.history)
    save_heuristics(output_path, result.heuristics, result.validation_scores, metadata)
    progress_bar.progress(1.0, text="Done")
    chart_slot.empty()
    st.session_state["train_output_file"] = output_path.name


if start:
    run_training(data, simulation, training, HEURISTICS_FOLDER / f"{output_name}.json")

if "train_output_file" in st.session_state:
    file_name = st.session_state["train_output_file"]
    heuristics, document = heuristics_from(file_name)
    st.success(f"Saved {len(heuristics)} heuristics to heuristics/{file_name}",
               title="Training finished")
    st.dataframe(heuristics_table(heuristics, document), hide_index=True,
                 alt="Best evolved heuristics with validation score, size and formula")
    best_column, history_column = st.columns(2)
    with best_column:
        st.markdown(f"**Best heuristic (GP#1)**: `{infix_formula(heuristics[0].tree)}`")
        st.mermaid_chart(heuristic_tree(heuristics[0].tree),
                         alt="Expression tree of the best evolved heuristic")
    with history_column:
        history = document.get("metadata", {}).get("history", [])
        if history:
            st.altair_chart(fitness_chart(history), alt="Fitness per generation")
            st.caption("Each generation is scored on a different project and disruption "
                       "stream, so the curve is noisy; the validation scores above are the "
                       "fair comparison.")
    st.page_link("app_pages/heuristics.py", label="Read all heuristics on the Heuristics page",
                 icon=":material/account_tree:")
