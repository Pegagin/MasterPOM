from pathlib import Path

import numpy as np
import pytest

from rcpsp_gp.baseline import random_sampling_baseline, serial_sgs
from rcpsp_gp.generate import random_instance
from rcpsp_gp.instance import parse_sm, resource_factor, resource_strength
from rcpsp_gp.metrics import check_feasible, makespan

TOY = Path(__file__).parent / "data" / "toy.sm"
J30 = Path(__file__).parent / "data" / "j30"


def test_parse_toy():
    instance = parse_sm(TOY)
    assert instance.n_activities == 10 and instance.n_resources == 2
    assert list(instance.durations) == [0, 3, 2, 4, 2, 3, 5, 2, 1, 0]
    assert list(instance.capacities) == [4, 3]
    assert instance.successors[0] == [1, 2, 3]
    assert instance.predecessors[8] == [4, 5, 6, 7]
    assert list(instance.requirements[3]) == [3, 1]


def test_static_attributes_toy():
    instance = parse_sm(TOY)
    assert instance.lower_bound == 8
    assert list(instance.earliest_start) == [0, 0, 0, 0, 3, 3, 2, 4, 7, 8]
    assert list(instance.earliest_finish[[1, 4, 8]]) == [3, 5, 8]
    assert instance.latest_finish[instance.sink] == instance.lower_bound
    assert (instance.latest_start >= instance.earliest_start).all()     # no negative float


def test_total_successor_and_predecessor_counts_toy():
    instance = parse_sm(TOY)
    assert instance.total_successor_count[0] == instance.n_activities - 1
    assert instance.total_predecessor_count[instance.sink] == instance.n_activities - 1
    assert instance.total_successor_count[instance.sink] == 0
    assert instance.total_predecessor_count[8] == 8        # everything except the sink


def test_serial_sgs_feasible_toy():
    instance = parse_sm(TOY)
    start = serial_sgs(instance)
    assert check_feasible(instance, start) == []
    assert makespan(instance, start) >= instance.lower_bound


def test_check_feasible_detects_errors():
    instance = parse_sm(TOY)
    everything_on_day_zero = np.zeros(instance.n_activities, dtype=np.int64)
    assert check_feasible(instance, everything_on_day_zero) != []


@pytest.mark.parametrize("seed", range(10))
def test_random_sampling_baseline_feasible(seed):
    instance = random_instance(n=12 + seed, n_res=1 + seed % 4, seed=seed)
    assert check_feasible(instance, random_sampling_baseline(instance, n_samples=20, seed=seed)) == []


def test_save_load_baseline_roundtrip(tmp_path):
    from rcpsp_gp.baseline import load_baseline, save_baseline
    instance = parse_sm(TOY)
    start = serial_sgs(instance)
    path = save_baseline(tmp_path / "sub" / "toy.json", instance, start, "lft")
    data = load_baseline(path)
    assert (data["start"] == start).all()
    assert data["makespan"] == makespan(instance, start) and data["method"] == "lft"


def test_parse_instance_name():
    from rcpsp_gp.split import parse_instance_name, parse_name
    assert parse_name("j3021_4") == (21, 4)
    assert parse_instance_name("j12060_10") == (120, 60, 10)
    with pytest.raises(ValueError):
        parse_name("toy")


def test_chand_split_on_j30():
    from collections import Counter
    from rcpsp_gp.split import parse_instance_name, split_instances
    if not J30.exists():
        pytest.skip("J30 data not available")
    subsets = split_instances(J30, "chand")
    assert [len(subsets[s]) for s in ("train", "validation", "test")] == [96, 48, 336]
    numbers = {s: {parse_instance_name(f.stem).instance_number for f in subsets[s]}
               for s in subsets}
    assert numbers == {"train": {1, 2}, "validation": {3}, "test": set(range(4, 11))}
    per_set = Counter(parse_instance_name(f.stem).parameter_set for f in subsets["train"])
    assert set(per_set.values()) == {2}


def test_half_split_on_j30():
    from rcpsp_gp.split import split_instances
    if not J30.exists():
        pytest.skip("J30 data not available")
    subsets = split_instances(J30, "half")
    assert len(subsets["train"]) == len(subsets["test"]) == 240 and not subsets["validation"]


def test_design_parameters_match_psplib_parameter_sets():
    """Snapped RF/RS follow PSPLIB's order: RS varies fastest, then RF, then NC."""
    from rcpsp_gp.dataset import design_parameters
    from rcpsp_gp.split import parse_instance_name
    if not J30.exists():
        pytest.skip("J30 data not available")
    factor_levels, strength_levels = (0.25, 0.5, 0.75, 1.0), (0.2, 0.5, 0.7, 1.0)
    for path in sorted(J30.glob("*.sm")):
        name = parse_instance_name(path.stem)
        factor, strength = design_parameters(parse_sm(path), 30)
        position = (name.parameter_set - 1) % 16
        assert factor == factor_levels[position // 4], path.stem
        if factor > 0.25:            # RF 0.25 instances have noisy computed RS in PSPLIB
            assert strength == strength_levels[position % 4], path.stem


def test_resource_measures_toy_are_in_range():
    instance = parse_sm(TOY)
    assert 0 < resource_factor(instance) <= 1
    assert 0 <= resource_strength(instance) <= 1
