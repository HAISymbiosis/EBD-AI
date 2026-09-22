# Development context

## Repository and conventions

This repository is **BDI** (HAISymbiosis/EBD-AI), an independent rebase of
Ex-Fuzzy. Read [STATUS.md](STATUS.md) for versions, remotes, and recent work.

Two import packages ship from one pip distribution (`ebdai` 1.0.1):

- `ex_fuzzy/` — rebased Ex-Fuzzy API (`ex_fuzzy.__version__` is 3.2.0)
- `ebdai/` — BDI expansions that accept `ex_fuzzy` objects (`ebdai.__version__`
  is 1.0.1). Do not re-export Ex-Fuzzy names from `ebdai`.

Tests are in `tests/`, opt-in benchmarks in `benchmarks/`, examples in
`Demos/`, and Sphinx sources in `docs/source/`. Check packaging metadata for
current versions and dependencies.

Use PascalCase classes, snake_case functions, UPPER_CASE constants, leading
underscores for private helpers, type hints, and Google-style docstrings.
Modules import each other with relative imports only. Tests, benchmarks and
demos import through `ex_fuzzy` or `ebdai`, never the module files directly,
so every class and enum exists once per process. Both `ex_fuzzy/__init__.py`
and `ebdai/__init__.py` import lazily: register new submodules in
`_SUBMODULES` and new top-level names in `_EXPORTS`.
Load optional dependencies lazily and provide actionable installation errors.
Reject invalid input with `ValueError`, report recoverable problems with
`warnings.warn`, and keep `print` for verbose progress and reports.
Classifiers follow the scikit-learn `fit`/`predict` API: constructor arguments
are stored verbatim under their own names (derived state may be kept beside
them), `fit` returns the estimator, and `BaseFuzzyRulesClassifier` encodes
labels as consequent indexes for the search (`classes_`) and decodes them in
`predict`. Rule bases and `evalRuleBase` work on consequent indexes.
Never use interactive Git commands.

## Code map

All module paths below are relative to `ex_fuzzy/`.

| Area | Modules |
| --- | --- |
| Fuzzy sets and linguistic variables | `fuzzy_sets.py`, `utils.py`, `temporal.py`, `centroid.py` |
| Rule representation, inference, firing kernels | `rules.py` |
| Genetic classification | `evolutionary_fit.py`, `evolutionary_backends.py`, `evolutionary_search.py`, `_problem.py` (optimizer-independent problem base and pymoo wrapper) |
| Exact built-in objective | `_fitness.py`, `_array_fitness.py`, `_population_fitness.py` (CPU population batching), `_torch_fitness.py` (exact PyTorch population objective for EvoX devices) |
| Population route selection | `_dispatch_profile.py` |
| Regression | `evolutionary_fit_regression.py` |
| High-level classifiers and mining | `classifiers.py`, `rule_mining.py` |
| Evaluation and visualization | `eval_tools.py`, `eval_rules.py`, `vis_rules.py` |
| Statistical analysis | `bootstrapping_test.py`, `permutation_test.py`, `pattern_stability.py` |
| Persistence | `persistence.py` |
| Conformal prediction | `conformal.py` |
| Other learners | `ferl.py` (compact and medium FERL), `ferl_deep.py` (`DeepFERL`), `_evidence.py` (shared Dempster--Shafer combination), `ferl_partitions.py`, `tree_learning_new/`, `cognitive_maps.py` |

`ebdai/` (paths relative to `ebdai/`):

| Area | Modules |
| --- | --- |
| Version | `_version.py` (pip / `ebdai.__version__`, not `ex_fuzzy._version`) |
| Bias in data and inference | `bias.py` |
| Workshop tables | `datasets.py`, `data/*.csv` |
| Local demo app | `demo_app.py` (loopback server), `_demo_notebooks.py` (source and parameters), `_demo_worker.py` (Jupyter execution), `demo_static/` (browser UI) |

PyMoo is the default backend and supports checkpoints. The fitting problems
subclass `_problem.Problem`, not pymoo's; pymoo is imported only when the PyMoo
backend or the temporal classifier runs, which wrap problems with
`as_pymoo_problem`. Keep module-level pymoo imports out of the package
(`tests/test_optional_pymoo.py` checks it). EvoX classification scores
whole generations through `FitRuleBase._evaluate_gene_population`, sharing the
fit-local caches and batching, and on CUDA an exact PyTorch objective that each
fit verifies against the CPU before use. Consult
[the EvoX documentation](../docs/source/evox_backend.rst) and implementation for
current device support; old JAX descriptions and blanket GPU speed claims are stale.
Fuzzy sets include Type-1, interval Type-2, and general Type-2; fast paths support
only subsets of those cases. Conformal prediction requires a held-out calibration
set separate from training and final evaluation data.

## Performance invariants

Before changing `FitRuleBase`, `_fitness.py`, `_array_fitness.py`, or firing
kernels in `rules.py`, read [SPEED_UP.md](performance/SPEED_UP.md) and
[SPEED_UP_REVIEW.md](performance/SPEED_UP_REVIEW.md).

- Preserve objective values bit for bit, including reduction and pruning order.
  Numerical closeness is insufficient: small changes can alter the entire search.
- The object constructor and `_fitness.score_rulebase` remain the oracle and
  fallback, and build final models, checkpoints, and objects for custom losses.
  `FitRuleBase._array_score` returns `None` for unsupported cases.
- NumPy reduction layout and pairwise summation affect exactness. Compare new
  reductions with the reference across relevant types and array layouts.
- The PyTorch objective reproduces NumPy's pairwise sums and left-to-right
  products, and leaves non-integer MCC and penalty arithmetic to NumPy because
  CPU `torch.sqrt` is not correctly rounded. Its runtime verification is a
  safety net, not a substitute for parity tests.
- Fit-local fitness/firing caches and packed memberships must remain bounded and
  scoped to a fit, with cleanup on success and failure. Do not make them global.
- Respect fast-path eligibility and preserve logical evaluation counts even on
  cache hits. Keep custom-loss, checkpoint, and worker fallbacks intact.
- A rule is identified by its antecedents, consequent and modifiers, never by
  its score, weight or accuracy. `RuleSimple` equality and hashing, rule-base
  duplicate removal and the array decoder's per-class dedup must agree, and the
  first occurrence keeps its weight.
- Do not present kernel timings as complete-fit gains or benchmark prototypes
  as shipped capabilities.

## Validation

Install from source with `pip install -e .`; use `pip install -e ".[evox]"`
when the optional backend is needed. Prefer the mamba env `developer`. Run
relevant tests with `pytest tests/` or selected test files. Fixtures live in
`tests/conftest.py`. Bias helpers: `pytest tests/test_ebdai_bias.py`. Refresh
demo notebook outputs with `python Demos/run_notebooks.py`.

Demo Studio: install `pip install -e ".[demo]"` and run
`python -m ebdai.demo_app` from the repository root. `Demos/` remains the source
of truth. UI controls edit execution snapshots; only an explicit notebook save
changes source files. Preserve revision checks, fresh kernels, cancellation,
loopback binding, and the session token. Keep `ex_fuzzy` unchanged for GUI work.
`pytest tests/test_demo_app.py tests/test_ebdai_bias.py` covers the runner and bias
helpers. Kernel/server tests require local sockets. The optional browser suite is
`EBD_DEMO_BROWSER=1 pytest tests/test_demo_browser.py` after installing Playwright
and its Chromium browser. See `Demos/README.md` for the full workflow.

For evaluator changes, relevant suites include `test_array_evaluation.py`,
`test_genetic_fitness_semantics.py`, `test_fast_fitness.py`, cache tests,
`test_population_evaluation.py`, `test_route_dispatch.py` and
`test_evox_population.py`.
Measure complete seeded fits with
`python benchmarks/benchmark_evaluator_variants.py`, and EvoX fits with
`python benchmarks/benchmark_evox_routes.py`; both refuse timings unless
variants produce identical results. Avoid competing workloads during timing.
See the speedup records for workload-specific reproduction commands and limitations.

The README accuracy figure comes from `benchmarks/benchmark_keel.py`, one
result per (dataset, method) pair, spread over CERES by
`benchmarks/cluster/submit_keel.sh` and published by
`benchmarks/aggregate_keel.py`, which refuses to publish an incomplete grid.
Read [the KEEL methodology](../docs/performance/KEEL.md) before changing the
protocol or quoting its numbers. It measures defaults plus a stated GA search
budget, not tuned models.

`DeepFERL` ports `LearnedFuzzyTree` from `../fuzzy_greedy_tree` (FERL-deep in
that paper). `tests/test_ferl_deep.py` pins its trees and predictions against
a golden fixture, and runs a live bit-for-bit comparison when that checkout exists.
`tests/test_ferl_evidence.py` pins `FERL.predict_ds` to outputs recorded before
the combination rules moved into `_evidence.py`. Keep both pins passing.
