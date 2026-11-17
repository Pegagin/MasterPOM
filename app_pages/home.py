"""Home: what the app is for, how to use it, and what is available on disk."""
import streamlit as st

from ui.common import (BASELINE_FOLDER, DEFAULT_DATA_FOLDER, heuristics_files, results_files)

st.caption("Evolve priority heuristics with genetic programming that repair a project plan when "
           "resources break down, and compare them with the classic rules on identical "
           "disruptions.")

# ---------------------------------------------------------------- status
instance_count = len(list(DEFAULT_DATA_FOLDER.glob("*.sm")))
baseline_count = len(list((BASELINE_FOLDER / DEFAULT_DATA_FOLDER.name).glob("*.json")))
with st.container(horizontal=True):
    st.metric("J30 instances found", instance_count, border=True,
              help=f"PSPLIB .sm files in {DEFAULT_DATA_FOLDER}")
    st.metric("Cached baselines", baseline_count, border=True,
              help="Near-optimal CP-SAT plans already solved; others are solved on first use.")
    st.metric("Heuristic files", len(heuristics_files()), border=True,
              help="Training results in heuristics/")
    st.metric("Result files", len(results_files()), border=True,
              help="Evaluation results in results/")
if instance_count == 0:
    st.warning(f"No instances in {DEFAULT_DATA_FOLDER}. Download J30 from PSPLIB and unzip it "
               "there, or enter another folder on the pages.", title="No data found")

# ---------------------------------------------------------------- workflow
st.subheader("How to use it")
st.mermaid_chart(
    """flowchart LR
    T["Train<br/>evolve heuristics"] --> H["Heuristics<br/>read the formulas"]
    T --> E["Evaluate<br/>compare on test instances"]
    E --> R["Results<br/>tables, charts, statistics"]
    S["Schedule viewer<br/>look inside one project"]
    H -.-> S
    R -.-> S""",
    alt="Workflow: train, then read the heuristics or evaluate them, then explore results; "
        "the schedule viewer inspects single runs")

steps = st.columns(3)
with steps[0].container(border=True, height="stretch"):
    st.markdown("**1. Train**")
    st.markdown("Evolve heuristics on the training instances. For a first try set *Max "
                "instances* to 4 and *Population size* to 16 (about a minute).")
    st.page_link("app_pages/train.py", label="Go to Train", icon=":material/model_training:")
with steps[1].container(border=True, height="stretch"):
    st.markdown("**2. Evaluate**")
    st.markdown("Pick the saved heuristics and compare them with the standard rules on the "
                "test instances. Every method meets the same disruptions.")
    st.page_link("app_pages/evaluate.py", label="Go to Evaluate", icon=":material/science:")
with steps[2].container(border=True, height="stretch"):
    st.markdown("**3. Understand**")
    st.markdown("Read the evolved formulas, explore the results and watch single projects "
                "being disrupted and repaired.")
    with st.container(horizontal=True):
        st.page_link("app_pages/heuristics.py", label="Heuristics", icon=":material/account_tree:")
        st.page_link("app_pages/results.py", label="Results", icon=":material/insights:")
        st.page_link("app_pages/schedule_viewer.py", label="Schedule viewer",
                     icon=":material/view_timeline:")

# ---------------------------------------------------------------- concepts
st.subheader("Key ideas")
plan_tab, disruption_tab, heuristic_tab, gp_tab, metrics_tab = st.tabs(
    ["Plan and execution", "Disruptions", "Heuristics", "Genetic programming", "Measuring"])
with plan_tab:
    st.markdown("""
A project (from **PSPLIB**) consists of activities with durations, precedence relations and
resource needs. Before execution a near-optimal **baseline plan** is computed with CP-SAT.

The project is then **executed day by day**. Without disruptions the plan is followed exactly.
When resources drop, running activities that no longer fit are **paused**, and the rest of the
project is **re-planned** (repaired) by a priority heuristic.
""")
with disruption_tab:
    st.markdown("""
Disruptions follow Chand et al. (2019):

* **fp**: chance per day that a disruption is discovered (0.15, 0.20 or 0.25).
* **ATR**: average number of days it lasts (1, 2 or 3).
* It hits a random number of resources, each losing a random share of its availability.

The random stream is fixed by a **seed**, so every method meets exactly the same disruptions.
""")
with heuristic_tab:
    st.markdown("""
A **priority heuristic** gives every activity that could start a number; the **lowest starts
first**. Standard rules are fixed formulas, for example **LFT** (latest finish time first) or
**WCS** (worst case slack). Evolved heuristics combine attributes such as LS, LF or AvgRReq
into new formulas. The *Heuristics* page draws them as trees.
""")
with gp_tab:
    st.markdown("""
**Genetic programming** starts with random formulas. At every disruption all of them plan a
what-if repair; the best plan is used and every formula is scored by how much it would
delay the project. Good formulas are combined and mutated into the next generation.

After training, the best formulas on the **validation** instances are saved. They are tested
alone (**GP#1 ... GP#k**) or together as a **pool** that picks the best repair each time.
""")
with metrics_tab:
    st.markdown("""
* **Makespan increase**: how many days later than planned the project finishes. Main result.
* **Stability**: how far start days moved from the plan.
* **Head-to-head**: per scenario better/equal/worse counts and the **Wilcoxon test**
  (p < 0.05 means the difference is unlikely to be chance).
""")

st.subheader("Tips")
st.markdown("""
* Hover over the :material/help: icons next to every setting for a short explanation, and open
  **About these settings** for more.
* Full runs take a few minutes and keep the page busy; use **Max instances** for quick tries.
* The command-line scripts (`run_training.py`, `run_evaluation.py`) and this app read and
  write the same files.
""")
