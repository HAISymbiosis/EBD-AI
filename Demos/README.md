# BDI demos

Ten notebooks walk through the library, from a first classifier to
uncertainty and bias in data and inference. They are
executed with their outputs stored, so they read well on GitHub, and
every one of them runs in well under a minute on a laptop. Open them
with Jupyter after `pip install ebdai` (or `pip install -e .` from a
checkout), or read them online.

| Notebook | What it shows |
| --- | --- |
| [01 Getting started](01_getting_started.ipynb) | Fit, score, read the rules, probabilities, per-sample explanations, partition plots. |
| [02 Scikit-learn integration](02_scikit_learn_integration.ipynb) | Titanic data with categorical columns and missing values: a `Pipeline` with imputation, cross-validation, `GridSearchCV`. |
| [03 Rules and partitions](03_rules_and_partitions.ipynb) | Fuzzy sets and variables by hand, handmade rules, fixed versus optimised partitions, Type-2 sets, validation, inference modes, LaTeX export, saving and loading. |
| [04 Controlling the search](04_controlling_the_search.ipynb) | Budget and early stopping, custom objectives, checkpoints, mined candidate rules, and a comparison of every classifier on the Wine data. |
| [05 Regression](05_regression.ipynb) | `BaseFuzzyRulesRegressor` with crisp and Mamdani consequents on California housing, then inference by hand. |
| [06 Uncertainty](06_uncertainty.ipynb) | Conformal prediction sets with coverage evaluation and rule-level explanations, next to FERL and DeepFERL evidential outputs. |
| [07 Robustness](07_robustness.ipynb) | Pattern stability over repeated fits, and permutation and bootstrap validation of a fitted rule base. |
| [09 Bias in the data (Titanic)](09_bias_titanic.ipynb) | `ebdai` outcome rates and winning-rule firings by sex on the workshop Titanic table. |
| [10 Bias in heart-failure labels](10_bias_heart_failure.ipynb) | Same bias tools on the heart-failure death data. |
| [11 Bias in inference and mitigation (loans)](11_bias_loan_fairness.ipynb) | Demographic parity, Kamiran–Calders reweighing, and a fairness-regularised genetic loss. |

Notebooks 1 and 5 download Titanic and California housing through scikit-learn
on the first run and cache them in your home directory. Notebooks 9–11 load
tables shipped in `ebdai/data/` from the
[WorkshopIgualdad2025](https://github.com/rferper/WorkshopIgualdad2025) workshop.

Those three bias notebooks start with the workshop install cell for Google Colab:

```
!git clone -q https://github.com/HAISymbiosis/EBD-AI.git
%cd EBD-AI
!pip install -q .
```

Open them from GitHub:

- [09 Bias in the data (Titanic)](https://colab.research.google.com/github/HAISymbiosis/EBD-AI/blob/main/Demos/09_bias_titanic.ipynb)
- [10 Bias in heart-failure labels](https://colab.research.google.com/github/HAISymbiosis/EBD-AI/blob/main/Demos/10_bias_heart_failure.ipynb)
- [11 Bias in inference and mitigation (loans)](https://colab.research.google.com/github/HAISymbiosis/EBD-AI/blob/main/Demos/11_bias_loan_fairness.ipynb)

`evox_backend_demo.py` compares the PyMoo and EvoX backends and reports
whether EvoX and CUDA are available; run it as `python Demos/evox_backend_demo.py`.

To refresh the stored outputs after changing the library, run
`python Demos/run_notebooks.py` from the repository root.
