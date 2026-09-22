# Agent handoff

Last updated 2026-09-17. Check the working tree and `git log` before treating
this as current.

## What this repository is

[HAISymbiosis/EBD-AI](https://github.com/HAISymbiosis/EBD-AI) is an
**independent** public copy of [Fuminides/ex-fuzzy](https://github.com/Fuminides/ex-fuzzy),
not a GitHub-linked fork. Product name: **BDI: The Explainable By Design AI
Toolbox**. Author: Prof. Javier Andreu-Perez. It is a rebase of Ex-Fuzzy;
acknowledge that in user-facing text.

Remote `upstream` (if configured) points at `Fuminides/ex-fuzzy`. Fetch and
cherry-pick or re-implement. Do not `git merge upstream/main` and do not
convert this repo into a GitHub fork. Skip upstream README/LLM/agent files.
If a change touches `FitRuleBase`, `_fitness.py`, `_array_fitness.py`, or
firing kernels, keep objective values bit-for-bit.

As of 2026-09-22, `upstream/main` is `05fea6e` (dropped the temporal demo
notebook and occupancy dataset). That change is ported. The temporal library
code in `ex_fuzzy/temporal.py` stays. Do not merge `fix/python39-pymoo-install`
(Python 3.9) or `gh-pages` into `main`. Every other origin feature branch is
already contained in `main`.

## Two packages, two versions

`pip install ebdai` installs both. They are versioned separately.

| Import | Path | Version | Role |
| --- | --- | --- | --- |
| `ebdai` | `ebdai/` | **1.0.1** (`ebdai/_version.py`, pip metadata) | BDI expansions |
| `ex_fuzzy` | `ex_fuzzy/` | **3.2.0** (`ex_fuzzy/_version.py`) | Rebased Ex-Fuzzy API |

Do **not** re-export Ex-Fuzzy names from `ebdai`. Classes and enums must exist
once per process. Register new `ebdai` modules in `ebdai/__init__.py`
`_SUBMODULES` / `_EXPORTS`, same lazy pattern as `ex_fuzzy`.

## Bias work (on `main`)

Workshop source:
[rferper/WorkshopIgualdad2025](https://github.com/rferper/WorkshopIgualdad2025)
(Ex-Fuzzy 2.1.3 notebooks). Ported to current `ex_fuzzy` 3.2.0.

- Library: `ebdai/bias.py`, `ebdai/datasets.py`, CSVs in `ebdai/data/`
- Public names: `outcome_rates_by_group`, `fairness_report`,
  `winning_rules_by_group`, `reweigh_weights`, `weighted_mcc_loss`,
  `fairness_regularized_loss`, `load_titanic`, `load_heart_failure`,
  `load_loan_approval`
- Demos: `Demos/09_bias_titanic.ipynb`, `10_bias_heart_failure.ipynb`,
  `11_bias_loan_fairness.ipynb` (executed outputs stored)
- Tests: `tests/test_ebdai_bias.py`
- Docs: `docs/source/user-guide/ebdai.rst`, `docs/source/api/ebdai.rst`

Verified in the `developer` mamba env: unit tests passed; all three notebooks
ran with `python Demos/run_notebooks.py`.

## Environment

Install libraries in the **mamba env `developer`**
(`/home/ubuntu/miniforge3/envs/developer`): Python 3.12, numpy, pandas,
scikit-learn, matplotlib, pymoo, pytest, editable `ebdai`. GitHub CLI `gh`
is in that env; login is **jandreu**.

```bash
conda activate developer
pip install -e ".[test]"
python -m pytest tests/test_ebdai_bias.py
```

## User-facing docs vs this folder

Sphinx and README are for humans. Keep agent conventions here. User-facing
ebdai guide: `docs/source/user-guide/ebdai.rst`.

## Reasonable next work (not promised)

- More EBD method families under `ebdai.<name>` (concept bottlenecks,
  computing with words, prototypes) without renaming `ex_fuzzy`
- Publish `ebdai` 1.0.1 to PyPI when asked
- GitHub Settings → social preview: `docs/source/_static/og-image.png`
- After Pages rebuild, confirm `/user-guide/ebdai.html` and `/api/ebdai.html`
