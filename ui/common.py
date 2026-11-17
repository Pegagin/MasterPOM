"""Shared UI pieces: folders, cached loading and the settings widgets used on several pages."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from rcpsp_gp.dataset import Project, load_projects
from rcpsp_gp.gp import infix_formula, load_heuristics
from rcpsp_gp.sgs import SCHEDULE_GENERATION_SCHEMES
from rcpsp_gp.simulator import PAUSE_ORDERS, SCORE_TYPES, SimulationSettings
from rcpsp_gp.split import SPLIT_SCHEMES
from rcpsp_gp.state import PREEMPTION_MODES
from rcpsp_gp.training import EVOLUTION_BASES, TrainingSettings
from ui.explainers import DATA_EXPLAINER, GP_EXPLAINER, HELP, SIMULATION_EXPLAINER

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_FOLDER = ROOT / "tests" / "data" / "j30"
BASELINE_FOLDER = ROOT / "baselines"
HEURISTICS_FOLDER = ROOT / "heuristics"
RESULTS_FOLDER = ROOT / "results"
SUBSETS = ["train", "validation", "test", "all"]


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading projects (first time: solving CP-SAT baselines) ...",
                   max_entries=32)
def cached_projects(data_folder: str, subset: str, split_scheme: str,
                    resource_constrained_only: bool, limit: Optional[int]) -> List[Project]:
    return load_projects(data_folder, subset, split_scheme, resource_constrained_only,
                         baseline_folder=str(BASELINE_FOLDER), limit=limit, log=None)


@st.cache_resource(max_entries=32)
def cached_heuristics(path: str, modified_time: float):
    """Heuristics and document of a file; ``modified_time`` makes edits reload it."""
    return load_heuristics(path)


def heuristics_from(file_name: str):
    path = HEURISTICS_FOLDER / file_name
    return cached_heuristics(str(path), path.stat().st_mtime)


def heuristics_files() -> List[Path]:
    return sorted(HEURISTICS_FOLDER.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def results_files() -> List[Path]:
    return sorted(RESULTS_FOLDER.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def heuristics_table(heuristics, document: Dict) -> pd.DataFrame:
    return pd.DataFrame([{"heuristic": f"GP#{entry['rank']}",
                          "validation score": entry["validation_score"],
                          "size": entry["size"],
                          "formula": infix_formula(heuristic.tree)}
                         for heuristic, entry in zip(heuristics, document["heuristics"])])


def about(text: str) -> None:
    """A collapsed "About these settings" box."""
    with st.expander("About these settings", icon=":material/help:"):
        st.markdown(text)


# ---------------------------------------------------------------- settings widgets
def data_inputs(key: str, subset: Optional[str] = None, show_limit: bool = True,
                show_replications: bool = True) -> Dict:
    """Widgets for choosing instances. ``subset=None`` hides the subset selector."""
    about(DATA_EXPLAINER)
    choices: Dict = {
        "folder": st.text_input("Instance folder", str(DEFAULT_DATA_FOLDER), key=f"{key}-folder",
                                help=HELP["folder"]),
        "split": st.selectbox("Split scheme", SPLIT_SCHEMES, key=f"{key}-split",
                              help=HELP["split"]),
        "subset": subset,
    }
    if subset is not None:
        choices["subset"] = st.selectbox("Subset", SUBSETS, index=SUBSETS.index(subset),
                                         key=f"{key}-subset", help=HELP["subset"])
    choices["constrained"] = st.checkbox("Only resource-constrained instances",
                                         value=True, key=f"{key}-constrained",
                                         help=HELP["constrained"])
    choices["limit"] = None
    if show_limit:
        limit = st.number_input("Max instances (0 = all)", 0, 10000, 0, key=f"{key}-limit",
                                help=HELP["limit"])
        choices["limit"] = int(limit) or None
    choices["replications"] = 1
    if show_replications:
        choices["replications"] = int(st.number_input(
            "Scenarios per instance", 1, 50, 3, key=f"{key}-replications",
            help=HELP["replications"]))
    return choices


def projects_for(choices: Dict, subset: Optional[str] = None) -> List[Project]:
    return cached_projects(choices["folder"], subset or choices["subset"], choices["split"],
                           choices["constrained"], choices["limit"])


def simulation_inputs(key: str, defaults: SimulationSettings = SimulationSettings()
                      ) -> SimulationSettings:
    about(SIMULATION_EXPLAINER)

    def choose(label: str, options, current: str, help_key: str) -> str:
        options = list(options)
        return st.selectbox(label, options, index=options.index(current),
                            key=f"{key}-{help_key}", help=HELP[help_key])

    return SimulationSettings(
        preemption_mode=choose("Preemption mode", PREEMPTION_MODES, defaults.preemption_mode,
                               "preemption_mode"),
        sgs=choose("Schedule generation scheme", SCHEDULE_GENERATION_SCHEMES, defaults.sgs, "sgs"),
        pause_order=choose("Pause order", PAUSE_ORDERS, defaults.pause_order, "pause_order"),
        no_early_start=st.checkbox("No early start", value=defaults.no_early_start,
                                   key=f"{key}-no-early-start", help=HELP["no_early_start"]),
        score_type=choose("What-if score", SCORE_TYPES, defaults.score_type, "score_type"),
    )


def training_inputs() -> TrainingSettings:
    about(GP_EXPLAINER)
    defaults = TrainingSettings()
    population_size = st.number_input("Population size", 4, 4096, defaults.population_size,
                                      step=4, help=HELP["population_size"])
    evolution_basis = st.selectbox("Evolution basis", EVOLUTION_BASES,
                                   help=HELP["evolution_basis"])
    epochs = st.number_input("Epochs", 1, 100, defaults.epochs, help=HELP["epochs"])
    top_k = st.number_input("Heuristics to keep (top k)", 1, 50, defaults.top_k,
                            help=HELP["top_k"])
    use_extra_terminals = st.checkbox("Extra terminals", value=False,
                                      help=HELP["extra_terminals"])
    seed = st.number_input("Random seed", 0, 10 ** 9, defaults.seed, help=HELP["seed"])
    with st.expander("Advanced", icon=":material/tune:"):
        tournament_size = st.number_input("Tournament size", 2, 50, defaults.tournament_size,
                                          help=HELP["tournament_size"])
        crossover_probability = st.slider("Crossover probability", 0.0, 1.0,
                                          defaults.crossover_probability,
                                          help=HELP["crossover_probability"])
        mutation_probability = st.slider("Mutation probability", 0.0, 1.0,
                                         defaults.mutation_probability,
                                         help=HELP["mutation_probability"])
        max_tree_depth = st.number_input("Max tree depth", 2, 17, defaults.max_tree_depth,
                                         help=HELP["max_tree_depth"])
        elite_count = st.number_input("Elites", 0, 100, defaults.elite_count,
                                      help=HELP["elite_count"])
    return TrainingSettings(
        population_size=int(population_size), evolution_basis=evolution_basis, epochs=int(epochs),
        tournament_size=int(tournament_size), crossover_probability=float(crossover_probability),
        mutation_probability=float(mutation_probability), max_tree_depth=int(max_tree_depth),
        elite_count=int(elite_count), use_extra_terminals=use_extra_terminals, top_k=int(top_k),
        seed=int(seed))
