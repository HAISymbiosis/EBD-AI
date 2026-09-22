# BDI demos

Ten notebooks walk through the library, from a first classifier to
uncertainty and bias in data and inference. They are
executed with their outputs stored, so they read well on GitHub, and
every one of them runs in well under a minute on a laptop. Open them
with Jupyter after `pip install ebdai` (or `pip install -e .` from a
checkout), or read them online.

## Local browser app

From the repository root, in the same Python environment used for the library:

```bash
pip install -e ".[demo]"
python -m ebdai.demo_app
```

Open the session URL printed in the terminal (normally
`http://127.0.0.1:8765/#…`). **Demo Studio** has a menu of all ten notebooks
and the EvoX script, parameter controls, plots, tables, and a notebook editor.
No Node.js, frontend build, or external web service is needed.

- Choose an example, change the controls, and click **Run example**. Controls
  are discovered from literal keyword arguments, settings dictionaries, and
  named assignments. Each control identifies its cell and call; expressions
  and loops can be changed in **Notebook & code**.
- **Run through here** executes from the beginning up to that cell. Every run
  starts a fresh Jupyter kernel; it never depends on a previous run's variables.
- **Explore** initially shows the notebook's saved outputs. During execution it
  shows the run snapshot, with outputs appearing after each cell completes.
  **Inspect executed code** shows the actual parameter overrides used.
- Edit a notebook in Jupyter and save it: the app detects the change within a
  few seconds and refreshes the code and controls. It does not auto-execute edits.
  Browser edits remain in memory until **Save notebook**. A conflicting external
  edit is reported instead of overwritten. Saving code clears stale outputs.
- Changing controls affects only the run snapshot. **Download run** exports the
  latest run as a notebook with its code, parameters, and outputs. Temporary
  runs are removed when the server exits. Keep downloads you want to retain.
- Use **Stop** to cancel a run. One example executes at a time. The same port
  shares the current run across tabs.

The original notebook/script paths and standalone workflows are unchanged.
`ex_fuzzy` is unchanged; the GUI is an optional `ebdai` feature. The app runs
trusted local Python code with your environment's permissions, and binds only
to loopback. Open the full session URL, including its token.

Use `--port 8766` to choose another port, or `--demos /path/to/Demos` when
launching outside the repository root. Each cell has a ten-minute timeout.
Arbitrary JavaScript outputs and live Jupyter widgets are not rendered; use
Jupyter for those. HTML tables are displayed in isolated frames.

Browser validation is optional for contributors:

```bash
pip install playwright
python -m playwright install chromium
EBD_DEMO_BROWSER=1 pytest tests/test_demo_browser.py
```

On Linux, `python -m playwright install-deps chromium` may also be needed.
The regular integration suite is `pytest tests/test_demo_app.py` and requires
the `demo` extra for kernel execution tests.

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

Notebooks 2 and 5 download Titanic and California housing through scikit-learn
on the first run and cache them in your home directory. Notebooks 9–11 load
tables shipped in `ebdai/data/` from the
[WorkshopIgualdad2025](https://github.com/rferper/WorkshopIgualdad2025) workshop.

`evox_backend_demo.py` compares the PyMoo and EvoX backends and reports
whether EvoX and CUDA are available; run it as `python Demos/evox_backend_demo.py`.

To refresh the stored outputs after changing the library, run
`python Demos/run_notebooks.py` from the repository root.
