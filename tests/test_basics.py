from pathlib import Path

import numpy as np
import pytest

from rcpsp_gp.baseline import random_sampling_baseline, serial_sgs
from rcpsp_gp.generate import random_instance
from rcpsp_gp.instance import parse_sm
from rcpsp_gp.metrics import check_feasible, makespan, stability
from rcpsp_gp.rules import BENCHMARK_RULES, LFT, baseline_order
from rcpsp_gp.simulator import simulate

TOY = Path(__file__).parent / "data" / "toy.sm"


def test_parse_toy():
    inst = parse_sm(TOY)
    assert inst.n_jobs == 10 and inst.n_res == 2
    assert list(inst.d) == [0, 3, 2, 4, 2, 3, 5, 2, 1, 0]
    assert list(inst.cap) == [4, 3]
    assert inst.succ[0] == [1, 2, 3]
    assert inst.pred[8] == [4, 5, 6, 7]
    assert list(inst.req[3]) == [3, 1]


def test_static_attributes_toy():
    inst = parse_sm(TOY)
    assert inst.lb == 8
    assert list(inst.es) == [0, 0, 0, 0, 3, 3, 2, 4, 7, 8]
    assert list(inst.ef[[1, 4, 8]]) == [3, 5, 8]
    assert inst.lf[inst.sink] == inst.lb
    assert (inst.ls >= inst.es).all()          # no negative float


def test_serial_sgs_feasible_toy():
    inst = parse_sm(TOY)
    s = serial_sgs(inst)
    assert check_feasible(inst, s) == []
    assert makespan(inst, s) >= inst.lb


def test_check_feasible_detects_errors():
    inst = parse_sm(TOY)
    bad = np.zeros(inst.n_jobs, dtype=np.int64)     # everything at t=0
    assert check_feasible(inst, bad) != []


@pytest.mark.parametrize("seed", range(40))
def test_simulator_reproduces_baseline(seed):
    """No disruptions + baseline order + no early start => identical to baseline."""
    inst = random_instance(n=12 + seed % 10, n_res=1 + seed % 4, seed=seed)
    base = random_sampling_baseline(inst, n_samples=20, seed=seed)
    assert check_feasible(inst, base) == []
    res = simulate(inst, baseline_order(base), baseline=base, no_early_start=True)
    assert np.array_equal(res.start, base)
    assert res.makespan == makespan(inst, base)
    assert stability(res.start, base) == 0


@pytest.mark.parametrize("name", list(BENCHMARK_RULES))
@pytest.mark.parametrize("seed", range(10))
def test_simulator_feasible_for_all_rules(name, seed):
    inst = random_instance(n=15, n_res=3, seed=100 + seed)
    res = simulate(inst, BENCHMARK_RULES[name])          # no baseline, free dispatch
    assert check_feasible(inst, res.start) == []
    assert res.makespan >= inst.lb


def test_release_dates_respected():
    """With no_early_start, no activity starts before its baseline date and the result stays feasible."""
    inst = random_instance(n=20, n_res=3, seed=7)
    base = serial_sgs(inst, priority=np.random.default_rng(0).random(inst.n_jobs))  # deliberately poor
    held = simulate(inst, LFT, baseline=base, no_early_start=True)
    assert check_feasible(inst, held.start) == []
    assert (held.start >= base).all()
