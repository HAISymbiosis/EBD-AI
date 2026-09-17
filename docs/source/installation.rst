============
Installation
============

Install BDI in the Python environment where you want to use it. The pip
package name is ``ebdai``. That install provides ``import ex_fuzzy`` for the
rebased Ex-Fuzzy API and ``import ebdai`` for new BDI expansions. The
environment can come from ``venv``, conda, pyenv, system Python, or any other
Python manager.

Quick Install
=============

.. code-block:: bash

    python -m pip install ebdai

From GitHub, until the package is on PyPI:

.. code-block:: bash

    python -m pip install "ebdai @ git+https://github.com/HAISymbiosis/EBD-AI.git"

Using ``python -m pip`` keeps the command tied to the active Python interpreter,
which avoids installing into the wrong environment.

Optional Extras
===============

Install extras only when you need the corresponding feature:

.. code-block:: bash

    python -m pip install "ebdai[gpu]"   # PyTorch support for GPU tensors
    python -m pip install "ebdai[evox]"  # EvoX/PyTorch evolutionary backend
    python -m pip install "ebdai[docs]"  # Documentation build dependencies
    python -m pip install "ebdai[all]"   # All optional dependencies

Most users only need:

.. code-block:: bash

    python -m pip install ebdai

Environment Examples
====================

The install command stays the same once the environment is active.

With ``venv``:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install ebdai

On Windows, activate the environment with:

.. code-block:: bat

    .venv\Scripts\activate

With conda:

.. code-block:: bash

    conda create -n bdi python=3.11
    conda activate bdi
    python -m pip install ebdai

Development Install
===================

From a repository checkout:

.. code-block:: bash

    git clone https://github.com/HAISymbiosis/EBD-AI.git
    cd EBD-AI
    python -m pip install -e .

For development and documentation work:

.. code-block:: bash

    python -m pip install -e ".[dev]"
    python -m pip install -e ".[docs,evox]"

The editable install makes local source changes immediately available in the
active Python environment.

Requirements
============

BDI requires Python 3.10 or later. Core dependencies are installed
automatically by ``pip``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Purpose
   * - numpy
     - Numerical computations and array operations
   * - pandas
     - Data manipulation and analysis
   * - scikit-learn
     - Machine learning utilities and metrics
   * - matplotlib
     - Plotting and visualization
   * - pymoo
     - Evolutionary optimization

Verify Installation
===================

Check the installed version:

.. code-block:: bash

    python -c "import ex_fuzzy; print(ex_fuzzy.__version__)"

Create a classifier:

.. code-block:: python

    from ex_fuzzy import BaseFuzzyRulesClassifier

    classifier = BaseFuzzyRulesClassifier(nRules=3, verbose=False)
    print(type(classifier).__name__)

Backend Support
===============

The default backend is the CPU-based PyMoo optimizer. To use the optional EvoX
backend, install the EvoX extra in the same environment:

.. code-block:: bash

    python -m pip install "ebdai[evox]"

Then select it when creating a classifier or regressor:

.. code-block:: python

    from ex_fuzzy import BaseFuzzyRulesClassifier, BaseFuzzyRulesRegressor

    classifier = BaseFuzzyRulesClassifier(backend="evox")
    regressor = BaseFuzzyRulesRegressor(backend="evox")

For CUDA-specific PyTorch wheels, install PyTorch using the command recommended
by the PyTorch project for your platform, then install BDI with the EvoX
extra in the same environment.

Troubleshooting
===============

If installation fails with permission errors, create and activate a virtual
environment instead of installing into system Python.

If imports fail, confirm that ``pip`` and ``python`` point at the same
environment:

.. code-block:: bash

    python -m pip show ebdai
    python -c "import sys; print(sys.executable)"

If an optional backend import fails, install the matching extra in the active
environment:

.. code-block:: bash

    python -m pip install "ebdai[evox]"

Next Steps
==========

- :doc:`getting-started`: Learn the basics with a quick tutorial
- :doc:`examples/index`: See practical examples and use cases
- :doc:`user-guide/index`: Choose workflows and advanced features
