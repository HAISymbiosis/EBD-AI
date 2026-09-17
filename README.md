<p align="center">
  <img src="docs/source/_static/logo.png" width="200" height="200" alt="BDI logo">
</p>

<h1 align="center">BDI: The Explainable By Design AI Toolbox</h1>

<p align="center">
  <i>A Python toolbox for explainable-by-design machine learning</i>
</p>

<p align="center">
  <a href="https://github.com/HAISymbiosis/EBD-AI/actions/workflows/tests.yml">
    <img alt="Tests" src="https://github.com/HAISymbiosis/EBD-AI/actions/workflows/tests.yml/badge.svg">
  </a>
  <a href="https://github.com/HAISymbiosis/EBD-AI/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/HAISymbiosis/EBD-AI?style=flat-square">
  </a>
  <a href="https://github.com/HAISymbiosis/EBD-AI/stargazers">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/HAISymbiosis/EBD-AI?style=flat-square">
  </a>
  <a href="https://haisymbiosis.github.io/EBD-AI/">
    <img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat-square">
  </a>
  <a href="https://www.sciencedirect.com/science/article/pii/S0925231224008191">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Neurocomputing-green?style=flat-square">
  </a>
</p>

---

## Overview

**BDI** is a Python toolbox for **explainable-by-design** machine learning:
models whose decisions are interpretable by construction, not explained after
the fact by a separate method.

It is meant for many families of that idea — rule systems, evidential and
conformal predictors, sparse or prototype models, constrained learners, and
other methods that keep the model itself readable — not only fuzzy logic.

The first shipped family is fuzzy rule learning, via a rebase of
[Ex-Fuzzy](https://github.com/Fuminides/ex-fuzzy). `import ex_fuzzy` is that
API. `import ebdai` is the home for further explainable-by-design methods that
compose with it.

### Why BDI?

- Explainable by design: the model is the explanation (rules, sets, prototypes, constraints), not a post-hoc saliency map.
- Broader than fuzzy logic: `ex_fuzzy` is the first method family; new EBD methods land in `ebdai`.
- Classification and regression with a scikit-learn-style `fit` / `predict` API.
- Built-in checks: visualizations, robustness, and uncertainty tools for the models you train.

## Features

### Explainable-by-design toolbox
- **Method families**, not a single algorithm: start with fuzzy rules, add other EBD learners under `ebdai` without changing the `ex_fuzzy` API.
- **Readable models**: human-inspectable structure (rules, trees, sets, scores) instead of explaining a black box later.
- **Uncertainty as a first-class output**: conformal prediction sets and evidential belief / plausibility where the method supports them.
- **Shared workflow**: scikit-learn compatible estimators, train/test evaluation, and optional GPU search.

### Currently available: fuzzy rule learning (`ex_fuzzy`)

This is the rebased Ex-Fuzzy stack, the first method family in the toolbox.

- **Fuzzy association rules**: classification and regression with genetic fine-tuning.
- **FERL rule trees**: greedy fuzzy rule learning with native belief, plausibility, ignorance, and set-valued predictions.
- **Complexity control**: number of rules, rule length, linguistic variables, with strong and soft constraints.
- **Statistical analysis**: confidence intervals for rule quality, repeated fits for robustness.
- **Conformal predictions**: rule classifiers with finite-sample coverage guarantees.
- **Visualisation and validation**: fuzzy-set and rule plots, partition meaning, pattern stability.
- **Search backends**: PyMoo (CPU) and EvoX (GPU/CPU) for evolutionary rule optimisation.
- **Fuzzy set types**: Type-1, interval Type-2, and general Type-2, with quantile-based linguistic variables.

## Quick Start

### Installation

The pip package is **`ebdai`**. That install provides two imports: `ex_fuzzy`
for the fuzzy rule-learning family, and `ebdai` for other explainable-by-design
methods as they are added.

```bash
pip install ebdai

# With GPU support (EvoX backend with PyTorch)
pip install "ebdai[evox]"
```

From GitHub, until the package is on PyPI:

```bash
pip install "ebdai @ git+https://github.com/HAISymbiosis/EBD-AI.git"
pip install "ebdai[evox] @ git+https://github.com/HAISymbiosis/EBD-AI.git"
```

Or from a checkout:

```bash
git clone https://github.com/HAISymbiosis/EBD-AI.git
cd EBD-AI
pip install -e .
pip install -e ".[evox]"
```

### Basic Usage

The example below trains a fuzzy rule classifier from `ex_fuzzy`, the first
method family. Further explainable-by-design learners will import from `ebdai`.

```python
import ebdai
from ex_fuzzy import BaseFuzzyRulesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train fuzzy classifier
classifier = BaseFuzzyRulesClassifier(
    nRules=15,
    nAnts=4,
    backend="pymoo"  # or "evox" for GPU acceleration
)

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate and visualize fuzzy partitions
from ex_fuzzy.eval_tools import eval_fuzzy_model
eval_fuzzy_model(classifier, X_train, y_train, X_test, y_test,
                plot_partitions=True)
```

### FERL Evidential Classification

`FERL` learns a fuzzy rule tree and derives Dempster--Shafer evidence directly
from rule firing strengths. It ships in `ex_fuzzy` and needs no separate
fuzzy-tree package.

```python
from ex_fuzzy import FERL

ferl = FERL(max_rules=15, random_state=0)
ferl.fit(X_train, y_train)

predictions = ferl.predict(X_test)
betp, belief, plausibility, ignorance = ferl.predict_credal(X_test)
prediction_sets = ferl.predict_set(X_test)
ferl.print_tree()
```

Use `split_mode="learned"` for data-driven soft split locations or
`partition="mdlp"` for supervised trapezoidal partitions. Native FERL sets are
calibration-free evidential outputs; use `ConformalFuzzyClassifier` when a
finite-sample marginal coverage guarantee is required.

For higher accuracy with the same evidential outputs, `DeepFERL` grows a deep
tree of learned, Gini-placed soft splits and votes over its leaves.

### Regression Usage

`BaseFuzzyRulesRegressor` learns interpretable Type-1 rules for continuous
targets. It supports crisp Takagi-Sugeno consequents and fuzzy Mamdani
consequents.

```python
from ex_fuzzy import BaseFuzzyRulesRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=500, n_features=5, noise=5.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

regressor = BaseFuzzyRulesRegressor(
    nRules=20,
    nAnts=3,
    consequent_type="crisp",  # use "fuzzy" for Mamdani consequents
    backend="pymoo",
)
regressor.fit(X_train, y_train, n_gen=50, pop_size=50)

predictions = regressor.predict(X_test)
print(f"Test R2: {regressor.score(X_test, y_test):.3f}")
regressor.print_rules()
```

## Visualizations

The `ex_fuzzy` family includes plots for partitions, rules, and stability:

<p align="center">
  <img src="https://github.com/user-attachments/assets/858ae72b-6504-4173-b81b-b11a3caf802f" height="280" title="Type-1 Fuzzy Sets">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6ffff71c-49e5-4437-94e3-3b821f799643" height="280" title="Type-2 Fuzzy Sets">
  <img src="https://github.com/Fuminides/ex-fuzzy/assets/12574757/b356a09f-4c66-45c9-8362-ebdbda684669" height="280" title="General Type-2 Fuzzy Sets">
</p>

### Statistical Analysis

Monitor pattern stability and variable usage across multiple runs:

<p align="center">
  <img src="https://github.com/user-attachments/assets/4e57469d-6cc6-4a9c-a256-dba052a91045" height="300" title="Usage per Class">
  <img src="https://github.com/user-attachments/assets/819f0988-deeb-4c8d-8cca-d8dd75e437f7" height="300" title="Usage per Variable">
</p>

### Bootstrap Confidence Intervals

Obtain statistical confidence intervals for your metrics:

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d5d9d77-4ac4-474e-8ac2-6a146085ae53" alt="Bootstrap Analysis" style="border: 2px solid #ddd; border-radius: 8px; padding: 10px;" />
</p>

## Performance

### Accuracy and model size on 67 KEEL datasets

![Test accuracy, rules per model and training time for BDI's Genetic Search Rules, Mine+Search and FERL learners against logistic regression, decision tree, random forest and gradient boosting baselines on 67 KEEL classification datasets](docs/performance/keel.svg)

### Training speed in the ex-fuzzy lineage (2.0 vs 3.0)

![T1 complete-fit scaling from 1,000 to 100,000 samples and 10 to 200 features](docs/performance/t1_scaling.svg)

This experiment crosses **1,000 / 10,000 / 100,000 samples**
with **10 / 50 / 200 features**, for both fixed and optimized partitions, using the CPU backend.

### EvoX GPU acceleration

A three-seed benchmark compared identical EvoX CPU and CUDA
searches on 100,000 samples and 200 features (Type-1, 20 rules, 4 antecedents,
population 40 and 5 generations):

![EvoX CPU vs GPU complete fit on 100,000 samples and 200 features: fixed partitions 521.3 s vs 22.8 s (22.87× faster), optimized partitions 729.2 s vs 37.6 s (19.38× faster)](docs/performance/evox_gpu.svg)

### Backend Comparison

The fuzzy rule search supports two evolutionary optimization backends:

| Backend | Hardware | Best For |
|---------|----------|----------|
| **PyMoo** | CPU | Classification/regression on small datasets, checkpoint support |
| **EvoX** | GPU/CPU | Batched classification/regression on large datasets |

### When to Use Each Backend

**Use PyMoo** when:
- Working with small to medium datasets
- Running on CPU-only environments
- Need checkpoint/resume functionality
- Memory is limited

**Use EvoX** when:
- Have GPU available (CUDA recommended)
- Working with large datasets (>10,000 samples)
- No checkpointing (Evox does not support checkpointing yet)

Both backends automatically batch operations to fit available memory and large datasets are processed in chunks to prevent out-of-memory errors.


## Examples

### Notebooks

Executed notebooks in [`Demos/`](Demos/README.md) walk through `ex_fuzzy` and `ebdai`; they render on GitHub with their outputs.

| Notebook | What it shows |
|----------|---------------|
| [Getting started](Demos/01_getting_started.ipynb) | Fit, score, read the rules, probabilities, explanations, partition plots |
| [Scikit-learn integration](Demos/02_scikit_learn_integration.ipynb) | Titanic data with categorical columns and missing values, pipelines, cross-validation, grid search |
| [Rules and partitions](Demos/03_rules_and_partitions.ipynb) | Fuzzy sets by hand, fixed versus optimised partitions, Type-2 sets, inference modes, saving and loading |
| [Controlling the search](Demos/04_controlling_the_search.ipynb) | Budget, early stopping, custom objectives, checkpoints, mined rules, all classifiers compared |
| [Regression](Demos/05_regression.ipynb) | Crisp and Mamdani consequents on California housing |
| [Uncertainty](Demos/06_uncertainty.ipynb) | Conformal prediction sets, FERL and DeepFERL evidential outputs |
| [Robustness](Demos/07_robustness.ipynb) | Pattern stability over repeated fits, permutation and bootstrap validation |
| [Temporal](Demos/08_temporal.ipynb) | Temporal fuzzy sets on the occupancy data |
| [Bias in the data (Titanic)](Demos/09_bias_titanic.ipynb) | `ebdai` outcome rates and winning-rule firings by sex |
| [Bias in heart-failure labels](Demos/10_bias_heart_failure.ipynb) | Same bias tools on clinical death labels |
| [Bias in inference and mitigation (loans)](Demos/11_bias_loan_fairness.ipynb) | Demographic parity, reweighing, and a fairness-regularised genetic loss |
| [EvoX backend](Demos/evox_backend_demo.py) | GPU-accelerated training with EvoX (script) |

Bias demos use `import ebdai` on the workshop tables:

```python
from ebdai import load_titanic, features_and_target, outcome_rates_by_group, fairness_report
from ex_fuzzy import BaseFuzzyRulesClassifier

frame, sensitive = load_titanic()
X, y = features_and_target(frame, 'Survived')
print(outcome_rates_by_group(y, X[sensitive]))
```

#### Real Applications
  - BDI in fNIRS data: https://github.com/jjcato9/ex_fuzzy_fnirs_demo
### Code Examples

<details>
<summary><b>Fuzzy Partition Visualization</b></summary>

```python
# Plot fuzzy variable partitions
classifier.plot_fuzzy_variables()
```
</details>

<details>
<summary><b>GPU-Accelerated Training (EvoX Backend)</b></summary>

```python
from ex_fuzzy import BaseFuzzyRulesClassifier, BaseFuzzyRulesRegressor

# Create classifier with EvoX backend for GPU acceleration
classifier = BaseFuzzyRulesClassifier(
    nRules=30,
    nAnts=4,
    backend='evox',  # Use GPU-accelerated EvoX backend
    verbose=True
)

# Train with GPU acceleration
classifier.fit(X_train, y_train, 
              n_gen=50,
              pop_size=100)

# Early stopping is enabled by default:
# patience=10, min_delta=1e-4

# Regression uses the same EvoX backend. Both crisp and fuzzy
# consequents are evaluated in memory-aware PyTorch batches.
regressor = BaseFuzzyRulesRegressor(
    nRules=30,
    nAnts=4,
    consequent_type="crisp",
    backend="evox",
)
regressor.fit(X_reg_train, y_reg_train, n_gen=50, pop_size=100)

# CUDA is selected automatically when available; otherwise EvoX uses CPU.
print(regressor.optimization_device_)  # "cuda" or "cpu"
print(regressor.gpu_accelerated_)      # True only when CUDA was used
```
</details>

<details>
<summary><b>Bootstrap Analysis</b></summary>

```python
from ex_fuzzy.bootstrapping_test import generate_bootstrap_samples

# Generate bootstrap samples
bootstrap_samples = generate_bootstrap_samples(X_train, y_train, n_samples=100)

# Evaluate model stability
bootstrap_results = []
for X_boot, y_boot in bootstrap_samples:
    classifier_boot = BaseFuzzyRulesClassifier(nRules=10)
    classifier_boot.fit(X_boot, y_boot)
    accuracy = classifier_boot.score(X_test, y_test)
    bootstrap_results.append(accuracy)

print(f"Bootstrap confidence interval: {np.percentile(bootstrap_results, [2.5, 97.5])}")
```
</details>

## Documentation

- **[ebdai user guide](https://haisymbiosis.github.io/EBD-AI/user-guide/ebdai.html)**: Bias in the data and in inference (`import ebdai`)
- **[ebdai API](https://haisymbiosis.github.io/EBD-AI/api/ebdai.html)**: `ebdai.bias` and `ebdai.datasets`
- **[User Guide](https://haisymbiosis.github.io/EBD-AI/user-guide/index.html)**: Fuzzy rule learning with `ex_fuzzy`
- **[API Reference](https://haisymbiosis.github.io/EBD-AI/api/index.html)**: Classes and functions
- **[Quick Start Guide](https://haisymbiosis.github.io/EBD-AI/getting-started.html)**: Get up and running fast
- **[Examples Gallery](https://github.com/HAISymbiosis/EBD-AI/tree/main/Demos)**: Executed notebooks, including [bias demos 09–11](Demos/README.md)

## Requirements

### Core Dependencies
- **Python** >= 3.10
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **PyMOO** >= 0.6.2

### Optional Dependencies
- **EvoX** >= 1.3.0 (for GPU-accelerated evolutionary optimization): `pip install "ebdai[evox]"`
- **PyTorch** >= 2.6.0 (required by EvoX)

## Contributing

We welcome contributions from the community! Here's how you can help:

### Bug Reports
Found a bug? Please [open an issue](https://github.com/HAISymbiosis/EBD-AI/issues) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information

### Feature Requests
Have an idea? [Submit a feature request](https://github.com/HAISymbiosis/EBD-AI/issues) with:
- Clear use case description
- Proposed API design
- Implementation considerations

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=ex_fuzzy --cov=ebdai --cov-report=html

# Run specific test file
pytest tests/test_fuzzy_sets_comprehensive.py -v
```


## License

This project is licensed under the **AGPL v3 License** - see the [LICENSE](LICENSE) file for details.

Copyright for this rebase and BDI branding: **Prof. Javier Andreu-Perez**.
The codebase is a rebase of [Ex-Fuzzy](https://github.com/Fuminides/ex-fuzzy); original copyright of that project remains with its authors.

## Citation

If you use BDI’s fuzzy rule learners in your research, please cite the original Ex-Fuzzy paper that this toolbox rebases:

```bibtex
@article{fumanalex2024,
  title = {Ex-Fuzzy: A library for symbolic explainable AI through fuzzy logic programming},
  journal = {Neurocomputing},
  pages = {128048},
  year = {2024},
  issn = {0925-2312},
  doi = {10.1016/j.neucom.2024.128048},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231224008191},
  author = {Javier Fumanal-Idocin and Javier Andreu-Perez}
}
```

## Author

- **[Prof. Javier Andreu-Perez](https://github.com/jandreu)** — Author

## Acknowledgments

BDI is an explainable-by-design toolbox. Its first method family is a rebase of **[Ex-Fuzzy](https://github.com/Fuminides/ex-fuzzy)** by Javier Fumanal-Idocin and Javier Andreu-Perez.

- Special thanks to all [Ex-Fuzzy contributors](https://github.com/Fuminides/ex-fuzzy/graphs/contributors)
- This research has been supported by EU Horizon Europe under the Marie Skłodowska-Curie COFUND grant No 101081327 YUFE4Postdocs.
---

<p align="center">
  <b>Star us on GitHub if you find BDI useful!</b><br>
  <a href="https://github.com/HAISymbiosis/EBD-AI/stargazers">
    <img src="https://img.shields.io/github/stars/HAISymbiosis/EBD-AI?style=social" alt="GitHub Stars">
  </a>
</p>
