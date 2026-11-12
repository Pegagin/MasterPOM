"""Run the baseline schedulers on RCPSP .sm instances and print a summary.

Usage:
    python run_baseline.py                          # tests/data/toy.sm
    python run_baseline.py tests/data/j30           # every .sm in a folder
    python run_baseline.py a.sm b.sm --samples 500 --seed 1
    python run_baseline.py tests/data/j30 --method cpsat --time-limit 5
    python run_baseline.py tests/data/j30 --method cpsat --save baselines/j30
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rcpsp_gp.baseline import cpsat_solve, save_baseline, random_sampling_baseline, serial_sgs
from rcpsp_gp.instance import parse_sm
from rcpsp_gp.metrics import check_feasible, makespan, pct_dev_lb

DEFAULT = Path(__file__).parent / "tests" / "data" / "toy.sm"


def collect(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.sm")) if p.is_dir() else [p]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[str(DEFAULT)], help=".sm files or folders")
    ap.add_argument("--samples", type=int, default=200, help="random-sampling runs per instance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["sampling", "cpsat"], default="sampling",
                    help="how the 'best' baseline is built")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="CP-SAT seconds per instance; default scales with size (0.5 s/job, min 5). "
                         "CP-SAT stops earlier if it proves optimality")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="CP-SAT stops once within this relative gap of the bound, e.g. 0.02")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write each baseline schedule to DIR/<instance>.json "
                         "(load later with rcpsp_gp.baseline.load_baseline)")
    args = ap.parse_args()

    files = collect(args.paths)
    if not files:
        raise SystemExit("no .sm files found")

    print(f"{'instance':<14}{'jobs':>5}{'LB':>6}{'LFT':>6}{'best':>6}{'%dev LB':>9}{'bound':>7}  {'status':<9} feasible")
    devs = []
    for f in files:
        inst = parse_sm(f)
        lft = serial_sgs(inst)
        if args.method == "cpsat":
            best, status, bound = cpsat_solve(inst, time_limit=args.time_limit,
                                              seed=args.seed, gap=args.gap)
        else:
            best = random_sampling_baseline(inst, n_samples=args.samples, seed=args.seed)
            status, bound = "-", inst.lb
        if args.save:
            save_baseline(Path(args.save) / f"{inst.name}.json", inst, best,
                          args.method, status, bound)
        ok = not check_feasible(inst, best)
        dev = pct_dev_lb(inst, best)
        devs.append(dev)
        print(f"{f.stem:<14}{inst.n_jobs:>5}{inst.lb:>6}{makespan(inst, lft):>6}"
              f"{makespan(inst, best):>6}{dev:>9.2f}{bound:>7}  {status:<9} {'yes' if ok else 'NO'}")

    if len(devs) > 1:
        print(f"\nmean %dev LB over {len(devs)} instances: {sum(devs) / len(devs):.2f}")


if __name__ == "__main__":
    main()
