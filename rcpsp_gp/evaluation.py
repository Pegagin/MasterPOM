"""Testing: compare standard rules and frozen evolved heuristics on identical disruptions.

Methods
-------
``standard``        one standard rule (EST, LFT, WCS, ...) used for every repair.
``evolved_single``  one evolved heuristic used for every repair (each of the top k is its
                    own method, ``GP#1`` = best on validation).
``evolved_pool``    the top k evolved heuristics consulted at every repair; the best
                    what-if plan is adopted (``GP-pool``).

Every method runs every scenario with the same seeded disruption stream. Until the first
disruption all methods follow the same baseline, so the first repair is an exact
head-to-head; later disruptions hit whatever each method's schedule looks like by then.

One row of results per (scenario, method) holds the metrics built in :func:`result_row`.
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon

from .dataset import Project, Scenario
from .metrics import percent_over_lower_bound, stability
from .rules import STANDARD_RULES
from .simulator import ExecutionResult, SimulationSettings, execute_project, run_scenario

METHOD_KINDS = ("standard", "evolved_single", "evolved_pool")


@dataclass
class Method:
    name: str
    kind: str
    heuristics: list            # one rule, or the pool of evolved heuristics


def standard_methods(rule_names: Optional[Sequence[str]] = None) -> List[Method]:
    names = list(STANDARD_RULES) if rule_names is None else list(rule_names)
    return [Method(name, "standard", [STANDARD_RULES[name]]) for name in names]


def evolved_methods(heuristics: Sequence, top_k: int = 5, single: bool = True,
                    pool: bool = True) -> List[Method]:
    """``GP#1 .. GP#k`` (each heuristic alone) and/or ``GP-pool`` (all k together)."""
    selected = list(heuristics)[:top_k]
    methods = []
    if single:
        methods += [Method(f"GP#{rank}", "evolved_single", [heuristic])
                    for rank, heuristic in enumerate(selected, start=1)]
    if pool and selected:
        methods.append(Method("GP-pool", "evolved_pool", selected))
    return methods


# ------------------------------------------------------------------------------ metrics
def result_row(scenario: Scenario, method: Method, result: ExecutionResult,
               settings: SimulationSettings) -> Dict:
    """All metrics of one run, as a flat dictionary (one CSV row)."""
    project = scenario.project
    instance = project.instance
    real = instance.real_activities
    delays = result.start_day[real] - project.baseline_start[real]
    disruption_repairs = result.disruption_repairs
    first_repair_makespan = (disruption_repairs[0].adopted_makespan if disruption_repairs
                             else result.makespan)
    reduced_days = (result.available_capacity < result.capacities).any(axis=1)
    available_units = result.available_capacity.sum()
    row = {
        # identity of the run
        "instance": project.name,
        "parameter_set": project.parameter_set,
        "seed": scenario.seed,
        "failure_probability": scenario.disruption_settings.failure_probability,
        "mean_repair_time": scenario.disruption_settings.mean_repair_time,
        "preemption_mode": settings.preemption_mode,
        "sgs": settings.sgs,
        "pause_order": settings.pause_order,
        "method": method.name,
        "method_kind": method.kind,
        # how similar the run was to other methods' runs
        "disruption_count": len(result.disruptions),
        "first_disruption_day": (result.disruptions[0].discovery_day if result.disruptions
                                 else ""),
        "disrupted_days": int(reduced_days.sum()),
        # outcome
        "lower_bound": instance.lower_bound,
        "baseline_makespan": project.baseline_makespan,
        "makespan": result.makespan,
        "makespan_increase": result.makespan - project.baseline_makespan,
        "percent_over_lower_bound": round(percent_over_lower_bound(instance, result.makespan), 3),
        "makespan_after_first_repair": first_repair_makespan,
        "first_repair_increase": first_repair_makespan - project.baseline_makespan,
        # plan stability
        "total_start_deviation": stability(result.start_day, project.baseline_start),
        "mean_delay": round(float(delays.mean()), 3),
        "max_delay": int(delays.max()),
        # disruption effects
        "interruption_count": result.interruption_count,
        "wasted_work": result.wasted_work,
        # resources and cost
        "capacity_utilisation": round(float(result.executed_usage.sum() / available_units), 4)
        if available_units > 0 else "",
        "repair_count": len(disruption_repairs),
        "mean_repair_milliseconds": round(1000 * float(np.mean(
            [repair.runtime_seconds for repair in disruption_repairs])), 3)
        if disruption_repairs else "",
        "pool_choice_counts": "",
    }
    if method.kind == "evolved_pool":
        counts = np.bincount([repair.chosen_heuristic for repair in disruption_repairs],
                             minlength=len(method.heuristics))
        row["pool_choice_counts"] = " ".join(str(int(c)) for c in counts)
    return row


def evaluate_methods(methods: Sequence[Method], scenarios: Sequence[Scenario],
                     settings: SimulationSettings = SimulationSettings(),
                     log: Optional[Callable[[str], None]] = print,
                     progress: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
    """Run every method on every scenario; returns one row per (scenario, method).

    ``progress(done, total)`` is called after every scenario (used by the UI)."""
    log = log or (lambda message: None)
    progress = progress or (lambda done, total: None)
    rows = []
    started = time.perf_counter()
    for index, scenario in enumerate(scenarios, start=1):
        for method in methods:
            result = run_scenario(scenario, method.heuristics, settings)
            rows.append(result_row(scenario, method, result, settings))
        log(f"scenario {index:>3}/{len(scenarios)} {scenario.label:<45} "
            f"({time.perf_counter() - started:6.1f}s)")
        progress(index, len(scenarios))
    return rows


def check_baseline_reproduction(methods: Sequence[Method], projects: Sequence[Project],
                                settings: SimulationSettings = SimulationSettings()) -> List[str]:
    """Control run without disruptions: every method must execute the baseline exactly.

    Returns a list of problems (empty = all good)."""
    problems = []
    for project in projects:
        for method in methods:
            result = execute_project(project.instance, project.baseline_start, None,
                                     method.heuristics, settings)
            if not np.array_equal(result.start_day, project.baseline_start):
                problems.append(f"{method.name} changed the baseline of {project.name}")
    return problems


def write_rows_csv(rows: Sequence[Dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ------------------------------------------------------------------------------ summary
SUMMARY_COLUMNS = [            # (row key, column title, decimals)
    ("makespan_increase", "mk+", 2),
    ("percent_over_lower_bound", "%LB", 1),
    ("first_repair_increase", "1st+", 2),
    ("total_start_deviation", "dev", 1),
    ("mean_delay", "delay", 2),
    ("max_delay", "maxdel", 1),
    ("interruption_count", "intr", 2),
    ("wasted_work", "waste", 2),
    ("disruption_count", "disr", 2),
    ("disrupted_days", "ddays", 1),
    ("capacity_utilisation", "util", 3),
    ("mean_repair_milliseconds", "ms/rep", 2),
]

COLUMN_EXPLANATIONS = """\
mk+     makespan - baseline makespan          %LB    % above critical-path lower bound
1st+    makespan increase planned at the first repair (identical situation for all methods)
dev     sum |start - baseline start|          delay  mean start delay per activity
maxdel  largest start delay                   intr   activities paused by disruptions
waste   activity-days lost to restarts        disr   disruptions met during the run
ddays   days with reduced capacity            util   used / available resource-days
ms/rep  milliseconds per repair               rank   mean rank on makespan per scenario (1 = best)"""


def _scenario_key(row: Dict):
    return (row["instance"], row["seed"], row["failure_probability"], row["mean_repair_time"],
            row["preemption_mode"], row["sgs"], row["pause_order"])


def _mean(values) -> float:
    numbers = [float(v) for v in values if v != ""]
    return float(np.mean(numbers)) if numbers else float("nan")


def _format_table(headers: List[str], table_rows: List[List]) -> str:
    texts = [[str(cell) for cell in row] for row in [headers] + table_rows]
    widths = [max(len(row[i]) for row in texts) for i in range(len(headers))]
    lines = ["  ".join(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
                       for i, cell in enumerate(row)) for row in texts]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def mean_ranks(rows: Sequence[Dict]) -> Dict[str, float]:
    """Mean rank of each method on makespan within each scenario (1 = best, ties averaged)."""
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[_scenario_key(row)].append(row)
    ranks = defaultdict(list)
    for scenario_rows in by_scenario.values():
        for row, rank in zip(scenario_rows, rankdata([r["makespan"] for r in scenario_rows])):
            ranks[row["method"]].append(rank)
    return {method: float(np.mean(values)) for method, values in ranks.items()}


def compare_methods(rows: Sequence[Dict], method: str, reference: str,
                    metric: str = "makespan") -> Dict:
    """Paired comparison on ``metric`` (lower is better): better/equal/worse counts and the
    two-sided Wilcoxon signed-rank p-value."""
    by_scenario = defaultdict(dict)
    for row in rows:
        by_scenario[_scenario_key(row)][row["method"]] = float(row[metric])
    pairs = [(values[method], values[reference]) for values in by_scenario.values()
             if method in values and reference in values]
    differences = np.array([a - b for a, b in pairs])
    p_value = float("nan")
    if len(differences) and np.any(differences != 0):
        p_value = float(wilcoxon(differences).pvalue)
    return {"better": int((differences < 0).sum()), "equal": int((differences == 0).sum()),
            "worse": int((differences > 0).sum()), "p_value": p_value}


def summary_report(rows: Sequence[Dict]) -> str:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    rows_of = {method: [row for row in rows if row["method"] == method] for method in methods}
    ranks = mean_ranks(rows)
    methods.sort(key=lambda method: _mean(r["makespan_increase"] for r in rows_of[method]))
    sections = []

    table = [[method, len(rows_of[method])]
             + [f"{_mean(r[key] for r in rows_of[method]):.{decimals}f}"
                for key, _, decimals in SUMMARY_COLUMNS]
             + [f"{ranks[method]:.2f}"] for method in methods]
    sections.append("Mean per method (sorted by makespan increase)\n"
                    + _format_table(["method", "runs"] + [title for _, title, _ in SUMMARY_COLUMNS]
                                    + ["rank"], table)
                    + "\n\n" + COLUMN_EXPLANATIONS)

    def disruption_setting(row):
        return row["failure_probability"], row["mean_repair_time"]

    groups = sorted({disruption_setting(row) for row in rows})
    table = []
    for method in methods:
        group_means = [_mean(r["makespan_increase"] for r in rows_of[method]
                             if disruption_setting(r) == group) for group in groups]
        table.append([method] + [f"{value:.2f}" for value in group_means])
    sections.append("Mean makespan increase by disruption setting (fp, ATR)\n"
                    + _format_table(["method"] + [f"fp={fp:g},ATR={atr:g}" for fp, atr in groups],
                                    table))

    standard = [m for m in methods if rows_of[m][0]["method_kind"] == "standard"]
    others = [m for m in methods if rows_of[m][0]["method_kind"] != "standard"]
    if standard and others:
        reference = standard[0]
        table = []
        for method in others:
            comparison = compare_methods(rows, method, reference)
            table.append([method, comparison["better"], comparison["equal"], comparison["worse"],
                          f"{comparison['p_value']:.4f}"])
        sections.append(f"Makespan per scenario compared with the best standard rule ({reference})\n"
                        + _format_table(["method", "better", "equal", "worse", "Wilcoxon p"],
                                        table))

    pool_rows = [row for row in rows if row["method_kind"] == "evolved_pool"
                 and row["pool_choice_counts"]]
    if pool_rows:
        totals = np.sum([[int(c) for c in row["pool_choice_counts"].split()]
                         for row in pool_rows], axis=0)
        shares = totals / max(totals.sum(), 1)
        sections.append("GP-pool: how often each heuristic's repair was adopted\n"
                        + _format_table(["heuristic", "repairs", "share"],
                                        [[f"GP#{i}", int(count), f"{share:.1%}"]
                                         for i, (count, share) in enumerate(zip(totals, shares),
                                                                            start=1)]))
    return "\n\n".join(sections)
