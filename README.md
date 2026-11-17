# Evolving repair heuristics for RCPSP under resource disruptions

This repository evolves **priority heuristics with genetic programming (GP)** that repair a
project schedule when resources break down during execution. It compares them with the
classic priority rules on exactly the same disruptions.

The disruption model, terminals, operators and data split follow
Chand, Singh & Ray (2019), *Evolving heuristics for the resource constrained project
scheduling problem with dynamic resource disruptions*, Swarm and Evolutionary Computation 44,
897–912. The main difference from the paper is that execution starts from a
**near-optimal baseline plan** (CP-SAT), so stability against the plan can be measured too.

---

## The idea in one picture

```
 PSPLIB instance ──► CP-SAT baseline plan
                          │
                          ▼
        execute day by day ──► disruption discovered?
                          │            │ yes
                          │            ▼
                          │   pause activities that no longer fit
                          │            │
                          │            ▼
                          │   every heuristic in the pool plans the rest ("what-if")
                          │            │
                          │            ▼
                          │   adopt the plan with the smallest makespan increase
                          ◄────────────┘
                          ▼
                 project finished ──► metrics
```

During **training** the pool is the GP population. It learns from the what-if scores and
breeds after every project (or after every disruption). During **testing** the evolved
heuristics are frozen and compared with the standard rules on the same disruption streams.

Everything can be run from the command line or from a **browser app** (Streamlit) with
explanations for every setting, live progress, result charts, expression trees of the evolved
heuristics and Gantt charts of single runs. See [Using the app](#using-the-app).

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # numpy, ortools, deap, scipy, streamlit, pytest
```

### Data

The PSPLIB `.sm` files are not in the repository. Download J30 from the PSPLIB website and
unzip it to `tests/data/j30/`, so that files are named like `tests/data/j30/j3021_4.sm`.
Larger sets (J60, J90, J120) work the same way; point `--data` at their folder.

Baselines are solved with CP-SAT the first time an instance is used and cached in
`baselines/<data folder>/<instance>.json`.

---

## Quick start

Run all commands from the repository root with the virtual environment active.

```bash
# 1. Evolve heuristics (J30, default settings: about 5 minutes)
python run_training.py

# 2. Compare them with the standard rules on the test instances (about 3 minutes)
python run_evaluation.py --heuristics heuristics/j30_resume_parallel_project_seed0.json

# 3. Run the test suite
python -m pytest -q
```

For a quick try, add `--limit-projects 4` to both commands and `--population-size 16` to
training. `--help` lists every option.

For a quick start in the browser instead, see the next section.

---

## Using the app

### Start and stop

```bash
source .venv/bin/activate
streamlit run app.py
```

The app opens at <http://localhost:8501>. Stop it with `Ctrl+C` in the terminal.

* After changing files in `rcpsp_gp/`, **restart** the app: Streamlit reloads the page files
  automatically but may keep an old copy of the library in memory (symptom: an `ImportError`
  for a function that clearly exists).
* Changes to `app.py`, `app_pages/` or `ui/` show up after refreshing the browser.
* If the port is taken, an old instance is still running; stop it or use
  `streamlit run app.py --server.port 8502`.

### Pages

The pages are in the menu at the top.

| Page | What you do there |
|---|---|
| **Home** | Read how the app works and the key ideas (plan, disruptions, heuristics, GP, metrics). Status cards show how many instances, cached baselines, heuristic files and result files exist |
| **Train** | Set data, simulation and GP settings and press *Start training*. A progress bar and a live fitness chart show the run; afterwards the best heuristics are listed with their formulas, the tree of GP#1 is drawn, and the file is saved to `heuristics/` |
| **Heuristics** | Pick a heuristics file to see what it was trained with, every heuristic as a readable formula (e.g. `LF − AvgRReq`), and one heuristic at a time as an **expression tree** with terminals coloured by what they describe (critical path, precedence network, resources, baseline/disruptions). A chart shows which terminals the best heuristics rely on, plus the training progress |
| **Evaluate** | Pick a heuristics file (its training settings become the defaults), the standard rules, and whether to test each heuristic alone and/or as a pool. A control run checks that every method reproduces the baseline without disruptions; then all scenarios run and the results are shown and saved to `results/` |
| **Results** | Open any results CSV and filter by method, fp, ATR or instance: summary table, bar chart, heatmap per disruption setting, head-to-head against any method with better/equal/worse counts and the Wilcoxon test, and how often each pool member was adopted. Download the filtered rows as CSV |
| **Schedule viewer** | Choose an instance, disruption strength (fp, ATR, seed), simulation settings and a method (standard rule, one evolved heuristic, or the pool). Shows key numbers, Gantt charts of the baseline and of what actually happened (interrupted work orange, lost work red, disrupted days shaded), availability vs usage per resource, and tables of every repair and disruption |

### Help in the app

* Every setting has a **tooltip** (the help icon next to it).
* Every group of settings has a collapsed **About these settings** box with a longer
  explanation.
* The Results page explains every column under *What the columns mean*; the Heuristics page
  explains how to read a tree under *How to read a heuristic*.
* All texts live in `ui/explainers.py`, so they can be edited in one place.

### A first run (about 2 minutes)

1. **Train**: set *Max instances* to 4 and *Population size* to 16, press *Start training*.
2. **Heuristics**: select the new file and look at GP#1's formula and tree.
3. **Evaluate**: select the file, set *Max instances* to 4, press *Run evaluation*.
4. **Schedule viewer**: pick an instance and compare a standard rule (e.g. WCS) with
   GP#1 or the pool on the same disruption seed.

Full runs (all 120 resource-constrained J30 instances) take a few minutes each and keep the
page busy until they finish. The app and the scripts read and write the same files, so
heuristics trained in one can be evaluated in the other.

---

## How it works

### 1. Data and split (`dataset.py`, `split.py`)

* **Split (Chand scheme, default):** for every PSPLIB parameter set, instances 1–2 are
  used for training, 3 for validation and 4–10 for testing. J90 and J120 are test only.
  `--split half` uses instances 1–5 for training and 6–10 for testing.
* **Resource-constrained filter (default on):** only instances with resource strength
  RS ≤ 0.5 and resource factor RF > 0.5 are used, where the choice of heuristic matters.
  RS and RF are computed from the data and snapped to PSPLIB's design levels. On J30 this
  leaves 120 of 480 instances: 24 train, 12 validation, 84 test. `--all-instances` turns
  it off.
* **Scenarios:** each project is run 3 times (`--replications`), each with its own seed and
  with `fp` and ATR drawn from Chand's values.

### 2. Disruptions (`disruption.py`)

Three parameters, as in Chand et al. (Algorithm 1):

| Parameter | Paper | Meaning | Values used |
|---|---|---|---|
| `failure_probability` | fp | chance per day that a disruption is discovered | 0.15, 0.20, 0.25 |
| `mean_repair_time` | ATR | mean of the exponential disruption length (min. 1 day) | 1, 2, 3 |
| `mean_lead_time` | ATF | mean delay between discovery and start | 0 (same day) |

A disruption hits a random number of resources (1..K). Each loses a random fraction of its
current availability, rounded down to whole units, and overlapping disruptions compound.
The random stream of each day depends only on (seed, day). This guarantees that every
method meets exactly the same disruptions.

### 3. Execution and repair (`simulator.py`, `state.py`, `sgs.py`)

Each day the simulator:

1. marks finished activities,
2. applies newly discovered disruptions,
3. pauses running activities that no longer fit,
4. if something changed, lets every heuristic in the pool plan the rest of the project and
   adopts the best plan,
5. starts the activities planned for today.

Without disruptions the baseline is executed exactly; a control check verifies this for
every method.

| Setting | Options (default first) | Meaning |
|---|---|---|
| `preemption_mode` | `resume`, `restart` | a paused activity keeps its finished work / loses it |
| `sgs` | `parallel`, `serial` | how the rest of the project is planned (see below) |
| `pause_order` | `baseline_order`, `rule` | which running activities keep running: the ones planned earliest in the baseline (Chand's "initial priority"), or the ones the heuristic ranks best |
| `no_early_start` | on, off | activities never start before their baseline day |
| `score_type` | `makespan_increase`, `makespan` | what-if score of a repair (lower is better) |

**Schedule generation schemes**

* **Parallel** (as in Chand): moves a clock forward; on each day it starts the best-ranked
  activity that fits until nothing else fits.
* **Serial:** places one activity at a time, in priority order, at its earliest feasible day.

### 4. Heuristics (`decision.py`, `rules.py`, `gp.py`)

A heuristic receives the candidate activities of one decision and returns a priority for
each. **The lowest value starts first**; ties go to the lower activity id.

**Standard rules** (Chand Table 1): EST, EFT, LST, LFT, SPT, MTS, FIFO, GRPW, GRD,
WCS (worst case slack) and ACS (average case slack).

**Evolved heuristics** are expression trees, for example `sub(LF, AvgRReq)` (shown in the
app as the formula `LF − AvgRReq` and drawn as a tree on the *Heuristics* page), built from:

* **Operators:** `add`, `sub`, `mul`, `div` (protected: x/0 = 0), `max`, `min`, `neg`
* **Terminals (Chand Table 4)**, normalised to about 0..1:

| Terminal | Meaning |
|---|---|
| ES, EF, LS, LF | critical-path earliest/latest start/finish ÷ lower bound |
| TPC, TSC | number of direct + indirect predecessors / successors ÷ (n − 1) |
| RR | share of resource types the activity needs |
| AvgRReq, MaxRReq, MinRReq | requirement ÷ known availability over the activity's remaining duration: mean / max / min over resources |

**Optional extra terminals** (`--extra-terminals`), which use the baseline and the
disruption state: PlannedStart, Delay, Remaining, Interrupted, TimePressure,
AvailabilityRatio, TimeToRecovery. See `decision.py` for their definitions.

### 5. Training (`training.py`, `run_training.py`)

* The population (default 64) is the simulator's heuristic pool. At every disruption each
  heuristic makes a what-if repair and gets a score: its planned makespan minus the
  baseline makespan.
* **Evolution basis:**
  * `project` (default): fitness is the mean score over one project's disruptions, and the
    population breeds once per project.
  * `disruption`: the population breeds after every repair.
* **Breeding:** tournament selection (size 7), one-point crossover (0.9), uniform subtree
  mutation (0.1), max depth 6, 2 elites.
* **Selection:** after training, the final population and the best heuristic of every
  generation are run alone on the validation scenarios. The best `--top-k` (default 5) are
  saved to `heuristics/<name>.json` with their expressions and training settings.

### 6. Testing (`evaluation.py`, `run_evaluation.py`)

Methods compared on identical scenarios:

| Method | Meaning |
|---|---|
| `EST` … `ACS` | one standard rule used for every repair |
| `GP#1` … `GP#k` | one evolved heuristic used for every repair (#1 = best on validation) |
| `GP-pool` | all k evolved heuristics consulted at every repair; the best what-if plan is adopted |

Until the first disruption all methods follow the same baseline, so the **first repair**
is an exact head-to-head comparison.

The output is a CSV with one row per (scenario, method) in `results/`, and a printed summary:

* **Mean per method:** makespan increase (`mk+`), % above lower bound, increase at the first
  repair, start-day deviation and delays, interruptions, lost work, disruptions met,
  disrupted days, capacity utilisation, time per repair, mean rank.
* **By disruption setting:** mean makespan increase per (fp, ATR).
* **Against the best standard rule:** better / equal / worse counts and the Wilcoxon
  signed-rank p-value (p < 0.05 = significant difference).
* **GP-pool:** how often each heuristic's repair was adopted.

The evaluation reuses the settings stored with the heuristics; flags override them.

---

## Example result

Default training (J30, resume, parallel, project basis, seed 0), tested on 84 instances × 3
scenarios:

| Method | Makespan increase (days) | % over lower bound | Mean rank |
|---|---|---|---|
| GP-pool | **27.10** | **85.6** | **5.22** |
| GP#1 | 28.31 | 88.1 | 7.01 |
| WCS (best standard rule) | 28.91 | 89.3 | 7.47 |
| LFT | 28.94 | 89.3 | 7.75 |
| SPT (worst) | 35.01 | 100.9 | 13.10 |

The GP pool beats WCS in 126 scenarios, ties 71 and loses 55 (Wilcoxon p < 0.0001). A
single evolved heuristic is not yet significantly better than WCS (p = 0.13).

---

## Common options

| Goal | Option |
|---|---|
| Quick run | `--limit-projects 4` (+ `--population-size 16` for training) |
| Restart instead of resume | `--preemption-mode restart` |
| Serial SGS | `--sgs serial` |
| Breed after every disruption | `--evolution-basis disruption` |
| More training | `--epochs 3`, `--population-size 128` |
| Extra terminals | `--extra-terminals` |
| Pause by heuristic | `--pause-order rule` |
| All instances | `--all-instances` |
| Other data set | `--data path/to/j60` |
| Only some rules (evaluation) | `--rules LST,LFT,WCS` or `--rules none` |
| Only pool / only single GP (evaluation) | `--evolved pool` / `--evolved single` |

---

## Repository structure

```
rcpsp_gp/
  instance.py     PSPLIB parser, critical-path attributes, RS / RF
  split.py        train / validation / test split (chand, half)
  dataset.py      projects with cached CP-SAT baselines, filter, scenarios
  baseline.py     CP-SAT and sampling baselines, save / load
  disruption.py   Chand-style day-by-day disruption generator
  capacity.py     day x resource profiles that grow on demand
  state.py        execution state: availability, usage, activity status, plan
  sgs.py          parallel and serial schedule generation schemes
  simulator.py    day loop, pausing, what-if repairs, settings, results
  decision.py     what a rule sees at a decision + GP terminals
  rules.py        11 standard priority rules + baseline order
  gp.py           DEAP primitive set, evolved heuristics, save / load
  training.py     evolving population, training loop, validation
  evaluation.py   methods, metrics, control check, summary, Wilcoxon
  metrics.py      makespan, feasibility check, stability
  generate.py     small random instances for tests
run_baseline.py     build / compare baselines
run_training.py     train heuristics      -> heuristics/*.json
run_evaluation.py   test and compare      -> results/*.csv
app.py              Streamlit app entry point: page navigation (streamlit run app.py)
app_pages/          one file per page: home, train, heuristics, evaluate, results,
                    schedule_viewer
ui/
  common.py         folders, cached loading, settings widgets with tooltips
  explainers.py     every help text and explanation shown in the app
  charts.py         Altair charts (fitness, Gantt, resources, terminals), Mermaid trees
  results_view.py   results tables, charts and head-to-head tests
.streamlit/         app configuration (usage statistics switched off)
tests/              unit and integration tests (pytest)
```

Generated folders (`baselines/`, `heuristics/`, `results/`) are ignored by git.

---

## Differences from Chand et al. (2019)

| Aspect | Chand et al. | This repository |
|---|---|---|
| Initial schedule | built by the priority rule (parallel SGS) | near-optimal CP-SAT baseline |
| Disruption lead time | Exp(ATF), ATF ∈ {4, 6, 8} | 0 (same day) by default; `mean_lead_time` is available |
| Repair | the rule continues dispatching; paused activities are restarted first, in initial-priority order | each heuristic plans the rest of the project; the best what-if plan is adopted |
| Learning | offline: every rule is evaluated on all training instances per generation | online: the population is the repair pool and breeds per project or per disruption |
| Objectives | % over lower bound, plus tree size (NSGA-II) | makespan increase over the baseline (single objective) |
| SGS | parallel | parallel or serial |
| Stability | not measured | start-day deviation and delays against the baseline |

Table 4 of the paper defines MaxRReq with `Min` and MinRReq with `Max`. Here the names mean
what they say.
