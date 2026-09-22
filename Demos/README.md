# BDI demos

Ten notebooks walk through the library, from a first classifier to
uncertainty and bias in data and inference. They are
executed with their outputs stored, so they read well on GitHub, and
every one of them runs in well under a minute on a laptop. Open them
with Jupyter after `pip install ebdai` (or `pip install -e .` from a
checkout), or read them online.

## Local browser app

### First-time setup

Use Python 3.10 or newer. Demo Studio currently lives on the
`feature/localhost-demo-studio` branch. For a new checkout:

```bash
git clone --branch feature/localhost-demo-studio https://github.com/HAISymbiosis/EBD-AI.git
cd EBD-AI
```

If you already have this branch checked out, open a terminal in its repository
root (the folder containing `pyproject.toml` and `Demos/`). Activate your existing
Python environment. Alternatively, create one:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead of the
`source` command. With the environment active, install the app dependencies:

```bash
python -m pip install -e ".[demo]"
```

Here is what that command means:

- `python -m pip` installs into the same Python environment that will run the app.
- `.` refers to the project in the current folder.
- `-e` installs it in editable mode, so source changes are available without
  reinstalling. Restart the app after editing its Python server code.
- `[demo]` selects the project's optional notebook execution dependencies:
  `nbclient`, `nbformat`, and `ipykernel`. Keep the quotes around `".[demo]"`.

Do this once per environment; repeat installation if dependencies change.
The `demo` extra runs notebooks but does not install the JupyterLab editor.
To edit them in JupyterLab too, optionally run `python -m pip install jupyterlab`
and then `python -m jupyterlab Demos` in a second terminal with the same
environment active.

### Start and stop

From the repository root, run:

```bash
python -m ebdai.demo_app
```

Open the session URL printed in the terminal (normally
`http://127.0.0.1:8765/#…`). **Demo Studio** has a menu of all ten notebooks
and the EvoX script, parameter controls, plots, tables, and a notebook editor.
No Node.js, frontend build, or external web service is needed.

Copy the **entire URL**, including the session token after `#`; the example
`#…` above is a placeholder. Keep the terminal running, and open the URL in a
browser on the same computer. Press **Ctrl+C** in the terminal to stop the app.

For later visits, activate the same environment, return to the repository root,
and run `python -m ebdai.demo_app` again. No reinstall is needed.

### Try an example

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

### Troubleshooting

| Message or situation | What to do |
| --- | --- |
| `No module named ebdai.demo_app` | Check that your checkout contains Demo Studio, activate the correct environment, and run `python -m pip install -e ".[demo]"` from its root. |
| `Install demo dependencies` | Run the installation command above in the same environment as the app. |
| `No demos found` | Start from the repository root, or run `python -m ebdai.demo_app --demos /path/to/EBD-AI/Demos`. |
| Port 8765 is already in use | Run `python -m ebdai.demo_app --port 8766` and open the newly printed URL. |
| The page asks for a session URL | Copy the full URL from the current server terminal, including its token. Restarting the server creates a new token. |
| A dataset download fails | Notebooks 02 and 05 need internet access on their first run. The bias examples 09–11 use bundled data. |

### Contributor checks

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
