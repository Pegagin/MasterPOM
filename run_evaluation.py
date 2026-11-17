"""Test standard rules and evolved heuristics on the same disrupted executions.

Usage (from the project root, with the venv):
    python run_evaluation.py --heuristics heuristics/j30_resume_parallel_project_seed0.json
    python run_evaluation.py                                  # standard rules only
    python run_evaluation.py --heuristics FILE --evolved pool --rules LST,LFT,WCS
    python run_evaluation.py --heuristics FILE --limit-projects 5

Simulation settings default to the ones the heuristics were trained with; any flag given
here overrides them. Results go to a CSV file (one row per scenario and method) and a
summary is printed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rcpsp_gp.dataset import load_projects, make_scenarios
from rcpsp_gp.evaluation import (check_baseline_reproduction, evaluate_methods,
                                 evolved_methods, standard_methods, summary_report,
                                 write_rows_csv)
from rcpsp_gp.gp import load_heuristics
from rcpsp_gp.rules import STANDARD_RULES
from rcpsp_gp.simulator import PAUSE_ORDERS, SCORE_TYPES, SimulationSettings
from rcpsp_gp.sgs import SCHEDULE_GENERATION_SCHEMES
from rcpsp_gp.split import SPLIT_SCHEMES
from rcpsp_gp.state import PREEMPTION_MODES

DEFAULT_DATA = Path(__file__).parent / "tests" / "data" / "j30"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heuristics", default=None, help="JSON file from run_training.py")
    parser.add_argument("--evolved", default="single,pool",
                        help="'single', 'pool' or 'single,pool': how evolved heuristics are used")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rules", default=",".join(STANDARD_RULES),
                        help="comma-separated standard rules, or 'none'")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", choices=SPLIT_SCHEMES, default="chand")
    parser.add_argument("--subset", default="test", choices=("train", "validation", "test", "all"))
    parser.add_argument("--all-instances", action="store_true",
                        help="do not restrict to resource-constrained instances (RS<=0.5, RF>0.5)")
    parser.add_argument("--baseline-folder", default="baselines")
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--limit-projects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1000,
                        help="seed of the test scenarios (differs from training by default)")
    parser.add_argument("--preemption-mode", choices=PREEMPTION_MODES, default=None)
    parser.add_argument("--sgs", choices=list(SCHEDULE_GENERATION_SCHEMES), default=None)
    parser.add_argument("--pause-order", choices=PAUSE_ORDERS, default=None)
    parser.add_argument("--allow-early-start", action="store_true", default=None)
    parser.add_argument("--score", choices=SCORE_TYPES, default=None)
    parser.add_argument("--output", default=None, help="CSV file (default: results/<name>.csv)")
    return parser.parse_args()


def simulation_settings_from(arguments, trained_with: dict) -> SimulationSettings:
    """Settings the heuristics were trained with, overridden by explicit flags."""
    values = dict(SimulationSettings().__dict__)
    values.update(trained_with)
    overrides = {"preemption_mode": arguments.preemption_mode, "sgs": arguments.sgs,
                 "pause_order": arguments.pause_order, "score_type": arguments.score,
                 "no_early_start": (None if arguments.allow_early_start is None
                                    else not arguments.allow_early_start)}
    values.update({key: value for key, value in overrides.items() if value is not None})
    return SimulationSettings(**values)


def main():
    arguments = parse_arguments()
    heuristics, trained_with = [], {}
    if arguments.heuristics:
        heuristics, document = load_heuristics(arguments.heuristics)
        trained_with = document.get("metadata", {}).get("simulation", {})
    settings = simulation_settings_from(arguments, trained_with)
    print(f"simulation settings: {settings}")

    methods = [] if arguments.rules == "none" else standard_methods(arguments.rules.split(","))
    if heuristics:
        modes = set(arguments.evolved.split(","))
        methods += evolved_methods(heuristics, arguments.top_k, single="single" in modes,
                                   pool="pool" in modes)
    if not methods:
        raise SystemExit("nothing to evaluate: give --heuristics and/or --rules")

    projects = load_projects(arguments.data, arguments.subset, arguments.split,
                             resource_constrained_only=not arguments.all_instances,
                             baseline_folder=arguments.baseline_folder,
                             limit=arguments.limit_projects)
    scenarios = make_scenarios(projects, arguments.replications, seed=arguments.seed)
    print(f"{len(projects)} {arguments.subset} projects, {len(scenarios)} scenarios, "
          f"{len(methods)} methods: {', '.join(m.name for m in methods)}")
    for rank, heuristic in enumerate(heuristics[:arguments.top_k], start=1):
        print(f"  GP#{rank} = {heuristic.expression}")

    problems = check_baseline_reproduction(methods, projects, settings)
    print("control run without disruptions: "
          + ("every method reproduces the baseline" if not problems else "; ".join(problems)))

    rows = evaluate_methods(methods, scenarios, settings)
    name = Path(arguments.heuristics).stem if arguments.heuristics else "standard_rules"
    output = arguments.output or (f"results/{name}_{arguments.subset}_{settings.preemption_mode}_"
                                  f"{settings.sgs}.csv")
    write_rows_csv(rows, output)
    print("\n" + summary_report(rows))
    print(f"\nall rows saved to {output}")


if __name__ == "__main__":
    main()
