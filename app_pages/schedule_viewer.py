"""Schedule viewer: run one project with one method and compare plan and reality."""
import numpy as np
import pandas as pd
import streamlit as st

from rcpsp_gp.dataset import Project, Scenario
from rcpsp_gp.disruption import DisruptionSettings
from rcpsp_gp.gp import infix_formula
from rcpsp_gp.rules import STANDARD_RULES
from rcpsp_gp.simulator import ExecutionResult, run_scenario
from ui.charts import day_ranges, gantt_chart, resource_chart
from ui.common import (about, data_inputs, heuristics_files, heuristics_from, projects_for,
                       simulation_inputs)
from ui.explainers import DISRUPTION_EXPLAINER, HELP

st.caption("Run one project with one method and see the baseline plan next to what happened.")

project_column, simulation_column, method_column = st.columns(3)
with project_column:
    st.subheader("Project and disruptions")
    data = data_inputs("schedule", subset="test", show_limit=False, show_replications=False)
    projects = projects_for(data)
    if not projects:
        st.error("No instances found. Check the folder, subset and filter.")
        st.stop()
    names = [project.name for project in projects]
    project = projects[names.index(st.selectbox("Instance", names,
                                                help="PSPLIB instance to execute"))]
    st.caption(f"{project.instance.n_activities - 2} activities, "
               f"{project.instance.n_resources} resources, lower bound "
               f"{project.instance.lower_bound}, baseline makespan "
               f"{project.baseline_makespan} ({project.baseline_status})")
    about(DISRUPTION_EXPLAINER)
    failure_probability = st.slider("fp: disruption chance per day", 0.0, 1.0, 0.2, 0.05,
                                    help=HELP["fp"])
    mean_repair_time = st.slider("ATR: mean disruption length (days)", 0.5, 10.0, 2.0, 0.5,
                                 help=HELP["atr"])
    seed = int(st.number_input("Disruption seed", 0, 10 ** 9, 1, help=HELP["disruption_seed"]))
with simulation_column:
    st.subheader("Simulation")
    simulation = simulation_inputs("schedule")
with method_column:
    st.subheader("Method")
    kind = st.segmented_control("Repair with", ["standard rule", "evolved heuristic", "evolved pool"],
                                default="standard rule", required=True,
                                help="Who re-plans the project after each disruption.")
    if kind == "standard rule":
        rule_names = list(STANDARD_RULES)
        rule_name = st.selectbox("Rule", rule_names, index=rule_names.index("WCS"),
                                 help=HELP["rules"])
        heuristics, method_name = [STANDARD_RULES[rule_name]], rule_name
    else:
        files = heuristics_files()
        if not files:
            st.info("No evolved heuristics yet; train some first.")
            st.stop()
        file_name = st.selectbox("Heuristics file", [f.name for f in files])
        evolved, _ = heuristics_from(file_name)
        if kind == "evolved heuristic":
            rank = st.selectbox("Heuristic", range(1, len(evolved) + 1),
                                format_func=lambda r: f"GP#{r}: {infix_formula(evolved[r - 1].tree)}",
                                help="Ranked by validation score")
            heuristics, method_name = [evolved[rank - 1]], f"GP#{rank}"
        else:
            top_k = st.slider("Pool size (top k)", 1, len(evolved), min(5, len(evolved)),
                              help=HELP["pool"])
            heuristics, method_name = evolved[:top_k], f"GP-pool (top {top_k})"
    run = st.button("Run project", type="primary", icon=":material/play_arrow:", width="stretch")

if run:
    scenario = Scenario(project, DisruptionSettings(failure_probability, mean_repair_time), seed)
    st.session_state["schedule_output"] = {
        "project": project, "method": method_name,
        "heuristic_names": [getattr(h, "name", str(h)) for h in heuristics],
        "result": run_scenario(scenario, heuristics, simulation)}


def show_schedule(project: Project, result: ExecutionResult, method_name: str,
                  heuristic_names: list) -> None:
    instance = project.instance
    baseline_makespan = project.baseline_makespan
    real_activities = [int(a) for a in instance.real_activities if instance.durations[a] > 0]
    st.subheader(f"{project.name} repaired with {method_name}")

    with st.container(horizontal=True):
        st.metric("Baseline makespan", baseline_makespan, border=True)
        st.metric("Realised makespan", result.makespan, border=True,
                  delta=f"{result.makespan - baseline_makespan:+d} days", delta_color="inverse")
        st.metric("Disruptions", len(result.disruptions), border=True)
        st.metric("Interruptions", result.interruption_count, border=True,
                  help="Running activities paused because capacity dropped")
        st.metric("Lost work (days)", result.wasted_work, border=True,
                  help="Only in restart mode")
        st.metric("Start deviation", int(np.abs(result.start_day - project.baseline_start).sum()),
                  border=True, help="Sum of |realised start - planned start| over all activities")

    last_day = max(result.makespan, baseline_makespan) + 1
    planned = pd.DataFrame([{"activity": a, "start": int(project.baseline_start[a]),
                             "end": int(project.baseline_start[a] + instance.durations[a]),
                             "outcome": "planned"} for a in real_activities])
    realised = pd.DataFrame([{"activity": s.activity, "start": s.start_day, "end": s.end_day,
                              "outcome": s.outcome} for s in result.segments
                             if s.activity in real_activities])
    reduced = (result.available_capacity < result.capacities).any(axis=1)
    disrupted = pd.DataFrame(day_ranges(reduced), columns=["start", "end"])
    st.altair_chart(gantt_chart(planned, "Baseline plan", last_day, len(real_activities)),
                    alt="Gantt chart of the baseline plan")
    st.altair_chart(gantt_chart(realised, "What actually happened (red background: reduced "
                                          "capacity)", last_day, len(real_activities), disrupted),
                    alt="Gantt chart of the realised schedule with disrupted days shaded")

    st.markdown("**Resources**: nominal capacity, availability after disruptions, actual usage")
    columns = st.columns(2)
    for resource in range(instance.n_resources):
        with columns[resource % 2]:
            st.altair_chart(resource_chart(result, resource, last_day),
                            alt=f"Capacity, availability and usage of resource {resource}")

    repairs_tab, disruptions_tab = st.tabs(["Repairs", "Disruptions"])
    with repairs_tab:
        st.dataframe(pd.DataFrame([{
            "day": repair.day, "new disruptions": repair.new_disruption_count,
            "paused activities": ", ".join(map(str, repair.paused_activities)),
            "adopted repair from": heuristic_names[repair.chosen_heuristic],
            "planned makespan after": repair.adopted_makespan,
            "milliseconds": round(1000 * repair.runtime_seconds, 1)}
            for repair in result.disruption_repairs]), hide_index=True,
            alt="Every repair: day, paused activities and whose plan was adopted")
    with disruptions_tab:
        st.dataframe(pd.DataFrame([{
            "discovered on day": d.discovery_day, "start day": d.start_day,
            "duration": d.duration, "resources hit": ", ".join(str(r) for r in d.resources),
            "share of availability lost": ", ".join(f"{f:.0%}" for f in d.reduction_fractions)}
            for d in result.disruptions]), hide_index=True,
            alt="Every disruption met during the run")


if "schedule_output" in st.session_state:
    output = st.session_state["schedule_output"]
    show_schedule(output["project"], output["result"], output["method"], output["heuristic_names"])
