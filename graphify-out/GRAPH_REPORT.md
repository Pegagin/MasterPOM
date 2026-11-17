# Graph Report - MasterPOM  (2026-11-15)

## Corpus Check
- 25 files · ~13,066 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .ini 1, .sm 1)

## Summary
- 396 nodes · 1130 edges · 13 communities (7 shown, 6 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 136 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ca82507c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- numpy
- test_gp_and_evaluation.py
- evaluation.py
- test_simulation.py
- ProjectState
- parse_instance_name
- DecisionView
- CLAUDE.md
- EvolvingHeuristicPool

## God Nodes (most connected - your core abstractions)
1. `DecisionView` - 39 edges
2. `Instance` - 34 edges
3. `ProjectState` - 34 edges
4. `SimulationSettings` - 24 edges
5. `execute_project()` - 24 edges
6. `EvolvedHeuristic` - 22 edges
7. `makespan()` - 19 edges
8. `train_heuristics()` - 19 edges
9. `CapacityProfile` - 18 edges
10. `check_feasible()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `test_disruption_settings_of_scenarios_come_from_chand_values()` --uses--> `DisruptionSettings`  [INFERRED]
  tests/test_gp_and_evaluation.py → rcpsp_gp/disruption.py
- `test_without_disruptions_the_baseline_is_executed_exactly()` --uses--> `SimulationSettings`  [INFERRED]
  tests/test_simulation.py → rcpsp_gp/simulator.py
- `test_random_sampling_baseline_feasible()` --calls--> `random_sampling_baseline()`  [EXTRACTED]
  tests/test_basics.py → rcpsp_gp/baseline.py
- `make_projects()` --calls--> `random_sampling_baseline()`  [EXTRACTED]
  tests/test_gp_and_evaluation.py → rcpsp_gp/baseline.py
- `make_project()` --calls--> `random_sampling_baseline()`  [EXTRACTED]
  tests/test_simulation.py → rcpsp_gp/baseline.py

## Import Cycles
- None detected.

## Communities (13 total, 6 thin omitted)

### Community 0 - "numpy"
Cohesion: 0.06
Nodes (35): _cpsat(), cpsat_baseline(), cpsat_solve(), default_time_limit(), load_baseline(), random_sampling_baseline(), save_baseline(), serial_sgs() (+27 more)

### Community 1 - "test_gp_and_evaluation.py"
Cohesion: 0.06
Nodes (32): load_projects(), make_scenarios(), Project, evolved_methods(), standard_methods(), write_rows_csv(), build_primitive_set(), EvolvedHeuristic (+24 more)

### Community 2 - "evaluation.py"
Cohesion: 0.07
Nodes (18): check_baseline_reproduction(), compare_methods(), evaluate_methods(), _format_table(), _mean(), mean_ranks(), Method, result_row() (+10 more)

### Community 3 - "test_simulation.py"
Cohesion: 0.12
Nodes (18): Scenario, DisruptionGenerator, DisruptionSettings, random_instance(), schedule_from_scratch(), test_random_sampling_baseline_feasible(), make_project(), test_disrupted_runs_respect_precedence_and_availability() (+10 more)

### Community 4 - "ProjectState"
Cohesion: 0.05
Nodes (11): CapacityProfile, pad_days(), Disruption, pick_by_rule(), Plan, plan_parallel(), plan_serial(), _PlanningContext (+3 more)

### Community 5 - "parse_instance_name"
Cohesion: 0.18
Nodes (8): InstanceName, parse_instance_name(), parse_name(), split_instances(), subset_of(), test_chand_split_on_j30(), test_half_split_on_j30(), test_parse_instance_name()

### Community 6 - "DecisionView"
Cohesion: 0.10
Nodes (23): _availability_ratio(), _baseline_makespan(), DecisionView, _delay(), _lower_bound(), _needed_resources(), _planned_start(), _time_to_recovery() (+15 more)

## Knowledge Gaps
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ProjectState` connect `ProjectState` to `numpy`, `evaluation.py`, `test_simulation.py`, `DecisionView`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `DecisionView` (e.g. with `CapacityProfile` and `ProjectState`) actually correct?**
  _`DecisionView` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.06200411401704378 - nodes in this community are weakly interconnected._
- **Why does `DecisionView` connect `DecisionView` to `test_gp_and_evaluation.py`, `evaluation.py`, `ProjectState`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Instance` (e.g. with `cpsat_baseline()` and `cpsat_solve()`) actually correct?**
  _`Instance` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_gp_and_evaluation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06358024691358025 - nodes in this community are weakly interconnected._
- **Why does `Instance` connect `numpy` to `test_gp_and_evaluation.py`, `test_simulation.py`, `ProjectState`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._