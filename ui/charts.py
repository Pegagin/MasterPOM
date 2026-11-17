"""Chart builders (Altair and Mermaid) used by the pages."""
from __future__ import annotations

from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd

from rcpsp_gp.decision import TERMINAL_DESCRIPTIONS, TERMINAL_GROUPS, terminal_group
from rcpsp_gp.gp import tree_to_mermaid
from rcpsp_gp.simulator import ExecutionResult

# One colour per terminal group, shared by the tree diagrams and the usage chart
GROUP_COLOURS = {"critical path": "#4c78a8", "precedence network": "#54a24b",
                 "resources": "#f58518", "baseline and disruptions": "#b279a2"}
GROUP_FILLS = {"critical path": "#d6e4f2", "precedence network": "#dcefd9",
               "resources": "#fde3c8", "baseline and disruptions": "#ecdcea"}
GROUP_CLASS_NAMES = {group: group.replace(" ", "_") for group in TERMINAL_GROUPS}

OUTCOME_COLOURS = {"planned": "#9e9e9e", "completed": "#4c78a8",
                   "paused": "#f58518", "lost": "#e45756"}
OUTCOME_LABELS = {"planned": "planned (baseline)", "completed": "ran to completion",
                  "paused": "interrupted, resumed later", "lost": "interrupted, work lost"}


# ---------------------------------------------------------------- heuristics
def heuristic_tree(tree) -> str:
    """Mermaid diagram of an evolved heuristic, terminals coloured by group."""
    terminal_classes = {name: GROUP_CLASS_NAMES[terminal_group(name)]
                        for name in TERMINAL_DESCRIPTIONS}
    class_styles = {GROUP_CLASS_NAMES[group]: f"fill:{GROUP_FILLS[group]},"
                                              f"stroke:{GROUP_COLOURS[group]},color:#1f1f1f"
                    for group in TERMINAL_GROUPS}
    class_styles["operator"] = "fill:#eeeeee,stroke:#888888,color:#1f1f1f"
    return tree_to_mermaid(tree, terminal_classes, class_styles)


def group_colour_scale() -> alt.Scale:
    return alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values()))


def terminal_usage_chart(usage: Dict[str, int]) -> alt.Chart:
    table = pd.DataFrame([{"terminal": name, "occurrences": count,
                           "group": terminal_group(name),
                           "meaning": TERMINAL_DESCRIPTIONS[name]}
                          for name, count in usage.items()]).sort_values("occurrences",
                                                                         ascending=False)
    return (alt.Chart(table).mark_bar().encode(
        x=alt.X("occurrences:Q", title="occurrences in the selected heuristics"),
        y=alt.Y("terminal:N", sort="-x", title=None),
        color=alt.Color("group:N", scale=group_colour_scale(), title="describes"),
        tooltip=["terminal", "meaning", "group", "occurrences"])
        .properties(height=24 * len(table) + 40))


def fitness_chart(history: List[Dict]) -> alt.Chart:
    table = pd.DataFrame(history).rename(columns={"best_fitness": "best in generation",
                                                  "mean_fitness": "population mean"})
    long = table.melt(id_vars="generation", value_vars=["best in generation", "population mean"],
                      var_name="series", value_name="fitness")
    return (alt.Chart(long).mark_line(point=len(table) < 80).encode(
        x=alt.X("generation:Q"),
        y=alt.Y("fitness:Q", title="mean makespan increase (days)"),
        color=alt.Color("series:N", title=None),
        tooltip=["generation", "series", alt.Tooltip("fitness", format=".2f")])
        .properties(height=280))


# ---------------------------------------------------------------- schedules
def day_ranges(mask: np.ndarray) -> List[tuple]:
    """Contiguous ``(start, end)`` day ranges where ``mask`` is True."""
    ranges, start = [], None
    for day, flagged in enumerate(list(mask) + [False]):
        if flagged and start is None:
            start = day
        elif not flagged and start is not None:
            ranges.append((start, day))
            start = None
    return ranges


def gantt_chart(rows: pd.DataFrame, title: str, last_day: int, n_activities: int,
                disrupted: Optional[pd.DataFrame] = None) -> alt.LayerChart:
    x_scale = alt.Scale(domain=[0, last_day])
    outcomes = [o for o in OUTCOME_COLOURS if o in set(rows["outcome"])]
    rows = rows.assign(meaning=rows["outcome"].map(OUTCOME_LABELS))
    bars = alt.Chart(rows).mark_bar(stroke="white", strokeWidth=0.5).encode(
        x=alt.X("start:Q", scale=x_scale, title="day"), x2="end:Q",
        y=alt.Y("activity:O", sort="ascending", title="activity"),
        color=alt.Color("meaning:N", title=None,
                        scale=alt.Scale(domain=[OUTCOME_LABELS[o] for o in outcomes],
                                        range=[OUTCOME_COLOURS[o] for o in outcomes])),
        tooltip=["activity", "start", "end", alt.Tooltip("meaning", title="what happened")])
    layers = []
    if disrupted is not None and not disrupted.empty:
        layers.append(alt.Chart(disrupted).mark_rect(color="#e45756", opacity=0.12).encode(
            x=alt.X("start:Q", scale=x_scale), x2="end:Q",
            tooltip=[alt.Tooltip("start", title="reduced capacity from day"),
                     alt.Tooltip("end", title="until day")]))
    layers.append(bars)
    return alt.layer(*layers).properties(title=title, height=max(200, 14 * n_activities))


def resource_chart(result: ExecutionResult, resource: int, last_day: int) -> alt.Chart:
    table = pd.DataFrame({"day": np.arange(len(result.available_capacity)),
                          "nominal capacity": result.capacities[resource],
                          "available": result.available_capacity[:, resource],
                          "used": result.executed_usage[:, resource]})
    long = table.melt(id_vars="day", var_name="series", value_name="units")
    return (alt.Chart(long).mark_line(interpolate="step-after").encode(
        x=alt.X("day:Q", scale=alt.Scale(domain=[0, last_day])),
        y=alt.Y("units:Q"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(
            domain=["nominal capacity", "available", "used"],
            range=["#9e9e9e", "#e45756", "#4c78a8"])),
        strokeDash=alt.condition(alt.datum.series == "nominal capacity",
                                 alt.value([4, 3]), alt.value([1, 0])),
        tooltip=["day", "series", "units"])
        .properties(title=f"Resource {resource}", height=180))
