"""All help texts and explanations shown in the UI, kept in one place."""

# ---------------------------------------------------------------- short tooltips (help=)
HELP = {
    # data
    "folder": "Folder with PSPLIB .sm files, for example tests/data/j30.",
    "split": "How instances are divided. chand: instances 1-2 of every parameter set train, "
             "3 validate, 4-10 test (Chand et al. 2019). half: 1-5 train, 6-10 test.",
    "subset": "Which part of the split to use. Test on instances the heuristics never saw.",
    "constrained": "Keep only instances with scarce resources (resource strength ≤ 0.5) where "
                   "activities need many resource types (resource factor > 0.5). The choice of "
                   "heuristic matters most there. On J30 this keeps 120 of 480 instances.",
    "limit": "Use only the first N instances. Handy for a quick try; 0 uses all.",
    "replications": "How many disruption scenarios per instance. Each has its own random seed "
                    "and draws fp and ATR from Chand's values (fp 0.15/0.20/0.25, ATR 1/2/3).",
    "scenario_seed": "Seed for the test scenarios. Keep it different from training so the "
                     "heuristics are tested on disruptions they have not seen.",
    # simulation
    "preemption_mode": "What happens to an activity that a disruption interrupts. resume: it "
                       "keeps its finished work and continues later. restart: the work is lost "
                       "and it starts over.",
    "sgs": "How the rest of the project is re-planned after a disruption. parallel: steps "
           "through the days and starts the best-ranked activity that fits (as in Chand). "
           "serial: places one activity at a time at its earliest feasible day.",
    "pause_order": "When capacity drops, which running activities may keep running. "
                   "baseline_order: the ones planned to start earliest in the baseline (Chand's "
                   "initial priority). rule: the ones the repairing heuristic ranks best.",
    "no_early_start": "Activities never start before their day in the baseline plan, even if "
                      "a repair would allow it. Keeps the schedule close to the plan.",
    "score_type": "How what-if repairs are compared (lower is better). makespan_increase: "
                  "planned makespan minus baseline makespan. makespan: planned makespan.",
    # genetic programming
    "population_size": "Number of heuristics evolved together. Larger = more variety but each "
                       "disruption takes longer, because every heuristic plans a repair.",
    "evolution_basis": "When the population breeds. project: once per project, using the mean "
                       "score over its disruptions. disruption: after every repair.",
    "epochs": "How often the training scenarios are run. More epochs = more generations.",
    "top_k": "How many of the best heuristics (ranked on the validation scenarios) to save.",
    "extra_terminals": "Lets heuristics also use the baseline and the disruption state: "
                       "PlannedStart, Delay, Remaining, Interrupted, TimePressure, "
                       "AvailabilityRatio, TimeToRecovery.",
    "seed": "Random seed of the evolution and the training scenarios. Same seed = same result.",
    "tournament_size": "Parents are chosen as the best of this many random heuristics. Larger "
                       "= stronger pressure towards the current best.",
    "crossover_probability": "Chance that two parents swap random subtrees.",
    "mutation_probability": "Chance that a random subtree is replaced by a new random one.",
    "max_tree_depth": "Deepest tree allowed after crossover or mutation. Limits bloat.",
    "elite_count": "The best heuristics copied unchanged into the next generation.",
    # methods
    "rules": "Classic priority rules to compare against (Chand Table 1).",
    "top_k_evaluate": "How many of the saved heuristics to use, best first.",
    "single": "Test each evolved heuristic on its own, used for every repair (GP#1 ... GP#k).",
    "pool": "Test the k heuristics together: at every disruption each plans a repair and the "
            "best plan is adopted (GP-pool).",
    # schedule viewer
    "fp": "Chance that a new disruption is discovered on a given day (Chand's fp).",
    "atr": "Average number of days a disruption lasts (Chand's ATR, exponential, at least 1).",
    "disruption_seed": "Seed of the disruption stream. The same seed gives exactly the same "
                       "disruptions for every method.",
}

# ---------------------------------------------------------------- section explainers
DATA_EXPLAINER = """
**Instances** come from PSPLIB (J30 = 30 activities). Each instance gets a
**baseline plan** solved with CP-SAT (near-optimal, cached in `baselines/`).

**Split** keeps training, validation and test instances apart, so heuristics are judged on
projects they never saw. Every PSPLIB parameter set contributes to every subset.

**Scenarios**: every instance is run several times, each with its own random disruptions.
"""

SIMULATION_EXPLAINER = """
The project is executed **day by day** following the baseline plan. When a disruption is
discovered, resources drop, running activities that no longer fit are **paused**, and the rest
of the project is **re-planned** by the heuristic(s).

* **Preemption mode**: does a paused activity keep its finished work (*resume*) or lose it
  (*restart*)?
* **Schedule generation scheme**: *parallel* (day by day, as in Chand) or *serial*
  (one activity at a time).
* **Pause order**: who keeps running when capacity drops.
* **No early start**: never start before the baseline day.
* **What-if score**: how repairs are compared; lower is better.
"""

GP_EXPLAINER = """
**Genetic programming** evolves formulas (trees) that rank activities. The whole population
acts as the repair pool: at every disruption each heuristic plans a what-if repair, the best
plan is used, and each heuristic is scored by its planned makespan increase.

The population then **breeds**: good heuristics are selected (tournament), combined
(crossover) and slightly changed (mutation). After training, candidates are tested alone on
the **validation** scenarios and the best *k* are saved.
"""

METHODS_EXPLAINER = """
* **Standard rules** (EST, LFT, WCS, ...) are fixed formulas from the literature.
* **GP#1 ... GP#k**: each evolved heuristic alone, used for every repair.
* **GP-pool**: all *k* heuristics consulted at every disruption; the best what-if plan is
  adopted. Compare it with the single heuristics to see whether combining them helps.

Every method runs exactly the same scenarios and meets the same disruptions. A **control run**
without disruptions first checks that every method executes the baseline unchanged.
"""

DISRUPTION_EXPLAINER = """
Every day a disruption is discovered with probability **fp**. It starts the same day, lasts
on average **ATR** days and hits a random number of resources, each losing a random share of
its availability. The **seed** fixes the random stream: the same seed gives exactly the same
disruptions for every method.
"""

METRIC_EXPLANATIONS = """
* **Makespan increase**: final makespan minus baseline makespan (days). The main result.
* **% over lower bound**: distance from the critical-path lower bound (Chand's measure).
* **Increase at 1st repair**: planned increase at the first disruption. All methods are in
  exactly the same situation there, so this is the fairest head-to-head.
* **Start deviation / mean delay / max delay**: how far start days moved from the baseline.
* **Interruptions / lost work**: activities paused by disruptions; days of work lost
  (restart only).
* **Disruptions met / disrupted days**: similar values mean the runs were comparable.
* **Utilisation**: used / available resource-days. **ms per repair**: time per repair.
* **Mean rank**: average placing per scenario on makespan (1 = best).
* **Wilcoxon p**: below 0.05 means the difference to the reference is unlikely to be chance.
"""

HEURISTIC_READING_GUIDE = """
**What a heuristic is.** At every scheduling decision there are several activities that could
start. The heuristic computes one number per activity; the activity with the **lowest value
starts first** (ties go to the lower activity id).

**How to read the tree.** Leaves (boxes) are **terminals**: facts about an activity, scaled to
roughly 0–1. Circles are **operators** (+, −, ×, ÷, min, max, negation). The tree is
evaluated from the leaves up; the top node gives the priority. ÷ is protected: x ÷ 0 = 0.

**Example.** `LF − AvgRReq` starts activities that must finish early (small LF) and, among
similar ones, prefers those that need a large share of the currently available resources.

**Validation score**: mean makespan increase over the baseline (days) when this heuristic
alone repairs every disruption of the validation scenarios. Lower is better.
**Size**: number of nodes; smaller trees are easier to interpret.
"""
