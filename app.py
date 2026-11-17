"""Streamlit UI for training, evaluating and inspecting disruption-repair heuristics.

Start it from the repository root (with the virtual environment active):

    streamlit run app.py

The pages live in app_pages/, shared widgets, charts and texts in ui/.
"""
import streamlit as st

st.set_page_config(page_title="RCPSP disruption heuristics", page_icon=":material/schedule:",
                   layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True,
                help="What the app does, how to use it, key ideas and what is on disk."),
        st.Page("app_pages/train.py", title="Train", icon=":material/model_training:",
                help="Evolve repair heuristics with genetic programming, with a live progress "
                     "bar and fitness chart. Saves the best heuristics."),
        st.Page("app_pages/heuristics.py", title="Heuristics", icon=":material/account_tree:",
                help="Read the evolved heuristics as formulas and expression trees, and see "
                     "which terminals they rely on."),
        st.Page("app_pages/evaluate.py", title="Evaluate", icon=":material/science:",
                help="Compare evolved heuristics with the standard rules on the same test "
                     "scenarios and save the results."),
        st.Page("app_pages/results.py", title="Results", icon=":material/insights:",
                help="Explore saved results: summary table, charts per disruption setting and "
                     "head-to-head statistical tests."),
        st.Page("app_pages/schedule_viewer.py", title="Schedule viewer",
                icon=":material/view_timeline:",
                help="Run one project with one method and compare the baseline plan with what "
                     "happened, as Gantt and resource charts."),
    ],
    position="top",
)
st.title(page.title, icon=page.icon)
page.run()
