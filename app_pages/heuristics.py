"""Heuristics: see the evolved heuristics as formulas and trees."""
from collections import Counter

import pandas as pd
import streamlit as st

from rcpsp_gp.decision import TERMINAL_DESCRIPTIONS, TERMINAL_GROUPS, terminal_group
from rcpsp_gp.gp import infix_formula, terminal_usage, tree_depth
from ui.charts import fitness_chart, heuristic_tree, terminal_usage_chart
from ui.common import heuristics_files, heuristics_from, heuristics_table
from ui.explainers import HEURISTIC_READING_GUIDE

st.caption("What the evolved heuristics compute, drawn as expression trees.")

files = heuristics_files()
if not files:
    st.info("No evolved heuristics yet. Train some first.", title="Nothing to show")
    st.page_link("app_pages/train.py", label="Go to Train", icon=":material/model_training:")
    st.stop()

file_name = st.selectbox("Heuristics file (newest first)", [f.name for f in files],
                         help="Files in heuristics/, written by the Train page or "
                              "run_training.py. Heuristics are ranked by validation score.")
heuristics, document = heuristics_from(file_name)
metadata = document.get("metadata", {})
simulation = metadata.get("simulation", {})
training = metadata.get("training", {})

with st.expander("How to read a heuristic", icon=":material/help:"):
    st.markdown(HEURISTIC_READING_GUIDE)

with st.container(horizontal=True):
    st.metric("Preemption mode", simulation.get("preemption_mode", "?"), border=True,
              help="resume keeps finished work after an interruption, restart loses it")
    st.metric("Schedule generation", simulation.get("sgs", "?"), border=True,
              help="How the rest of the project was re-planned during training")
    st.metric("Evolution basis", training.get("evolution_basis", "?"), border=True,
              help="project: bred once per project; disruption: after every repair")
    st.metric("Population", training.get("population_size", "?"), border=True,
              help="Number of heuristics evolved together")
    st.metric("Generations", len(metadata.get("history", [])), border=True,
              help="How often the population bred")

st.subheader("All saved heuristics")
st.dataframe(heuristics_table(heuristics, document), hide_index=True,
             alt="Saved heuristics with validation score, size and readable formula")

# ---------------------------------------------------------------- one heuristic
st.subheader("Inspect one heuristic")
rank = st.segmented_control("Heuristic", list(range(1, len(heuristics) + 1)),
                            format_func=lambda r: f"GP#{r}", default=1, required=True,
                            help="Ranked by validation score; GP#1 is the best.")
heuristic = heuristics[rank - 1]
entry = document["heuristics"][rank - 1]

formula_column, tree_column = st.columns([2, 3])
with formula_column:
    with st.container(horizontal=True):
        st.metric("Validation score", f"{entry['validation_score']:.2f}"
                  if entry["validation_score"] is not None else "?", border=True,
                  help="Mean makespan increase over the baseline (days) on the validation "
                       "scenarios when this heuristic alone repairs every disruption.")
        st.metric("Size", heuristic.size, border=True, help="Number of nodes in the tree")
        st.metric("Depth", tree_depth(heuristic.tree), border=True,
                  help="Longest path from the top to a leaf")
    st.markdown("**Formula** (lowest value starts first)")
    st.code(infix_formula(heuristic.tree), language=None, wrap_lines=True)
    st.markdown("**As stored**")
    st.code(heuristic.expression, language=None, wrap_lines=True)
    occurrences = Counter(str(node.value) for node in heuristic.tree if hasattr(node, "value")
                          and str(node.value) in TERMINAL_DESCRIPTIONS)
    st.markdown("**Terminals used**")
    st.dataframe(pd.DataFrame([{"terminal": name, "times": count, "group": terminal_group(name),
                                "meaning": TERMINAL_DESCRIPTIONS[name]}
                               for name, count in occurrences.most_common()]),
                 hide_index=True, alt="Terminals in this heuristic and what they mean")
with tree_column:
    st.markdown("**Expression tree**")
    st.mermaid_chart(heuristic_tree(heuristic.tree),
                     alt=f"Expression tree of GP#{rank}: {infix_formula(heuristic.tree)}")
    st.markdown(" ".join(f":{colour}-badge[{group}]" for group, colour in
                         zip(TERMINAL_GROUPS, ("blue", "green", "orange", "violet")))
                + " :gray-badge[operator]")
    st.caption("Boxes are terminals (facts about an activity), circles are operators. The "
               "tree is evaluated from the leaves up.")

# ---------------------------------------------------------------- all heuristics
st.subheader("Which terminals do the heuristics rely on?")
top_k = st.slider("Count over the best k heuristics", 1, len(heuristics), len(heuristics),
                  help="Terminals that appear often in good heuristics are the attributes "
                       "that matter for repairing disruptions.")
st.altair_chart(terminal_usage_chart(terminal_usage(heuristics[:top_k])),
                alt="How often each terminal occurs in the best heuristics")
with st.expander("All terminals and their meaning", icon=":material/menu_book:"):
    st.dataframe(pd.DataFrame([{"terminal": name, "group": terminal_group(name),
                                "meaning": meaning}
                               for name, meaning in TERMINAL_DESCRIPTIONS.items()]),
                 hide_index=True, alt="Terminal reference")

history = metadata.get("history", [])
if history:
    st.subheader("Training progress")
    st.altair_chart(fitness_chart(history), alt="Best and mean fitness per generation")
    st.caption("Each generation is scored on a different project and disruption stream, so "
               "the curve is noisy. The validation score is the fair comparison.")
