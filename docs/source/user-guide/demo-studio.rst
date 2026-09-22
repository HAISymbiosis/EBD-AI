===========
Demo Studio
===========

Demo Studio runs the existing BDI examples in your browser on your own
computer. Choose a notebook from the menu, adjust its hyperparameters, run it,
and inspect its code, tables, rules, and plots. The notebooks remain ordinary
Jupyter notebooks that also run independently of the app.

First-time setup
================

Use Python 3.10 or newer and a checkout containing Demo Studio. The app
currently lives on the ``feature/localhost-demo-studio`` branch. To obtain it:

.. code-block:: bash

    git clone --branch feature/localhost-demo-studio https://github.com/HAISymbiosis/EBD-AI.git
    cd EBD-AI

If you already have that checkout, open a terminal in its root: the folder
containing ``pyproject.toml`` and ``Demos/``. Activate your existing Python
environment, or create a virtual environment:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate

On Windows PowerShell, activate it with ``.venv\Scripts\Activate.ps1`` instead.
Then install the project and its demo dependencies:

.. code-block:: bash

    python -m pip install -e ".[demo]"

This command installs into the active Python environment:

- ``.`` means the project in the current folder.
- ``-e`` means editable installation: Python uses your source checkout, so
  changes do not require reinstalling. Restart the server after editing its code.
- ``[demo]`` adds the optional Jupyter execution dependencies: ``nbclient``,
  ``nbformat``, and ``ipykernel``. Keep the quotes around ``".[demo]"``.

Install once per environment, and repeat if the project's dependencies change.
The app needs the repository's ``Demos/`` folder; installing only the published
package does not supply those notebooks.

Start the app
=============

From the repository root, with the same environment active:

.. code-block:: bash

    python -m ebdai.demo_app

Open the **full URL printed in the terminal** in a browser on the same computer.
It normally begins with ``http://127.0.0.1:8765/#`` and ends with a session token.
Copy that token too. Keep the terminal running while using the app, and press
**Ctrl+C** in the terminal to stop it.

On later visits, activate the same environment, return to the repository root,
and run the launch command again. You do not need to reinstall.

If the port is occupied, choose another one:

.. code-block:: bash

    python -m ebdai.demo_app --port 8766

To launch from another folder, specify the notebook directory:

.. code-block:: bash

    python -m ebdai.demo_app --demos /path/to/EBD-AI/Demos

Explore and edit
================

1. Select an example from the menu. Initially, **Explore** shows its saved outputs.
2. Adjust the controls and click **Run example**. Each run uses a fresh kernel;
   results appear as cells complete. **Stop** cancels the current run.
3. Use **Inspect executed code** to see the parameter values used, or
   **Notebook & code** to edit cell source. Click **Save notebook** to write
   those edits to the original file. Saving clears outdated notebook outputs.
4. Use **Run through here** to execute from the beginning through a selected cell.
5. Use **Download run** to keep an executed notebook with its parameters and
   outputs. Only the latest run is retained temporarily by the app.

Changes to GUI controls affect a run snapshot, not the original notebook.
When you edit and save the original notebook in Jupyter or another editor,
the app refreshes its code and controls within a few seconds. It does not
automatically execute the new code. Conflicting unsaved browser edits are
reported instead of overwritten.

The demo dependencies do not include a notebook editor. To use JupyterLab
alongside the app, optionally install and launch it in a second terminal with
the same environment active:

.. code-block:: bash

    python -m pip install jupyterlab
    python -m jupyterlab Demos

Troubleshooting
===============

- **Module or dependency not found:** activate the environment used for setup
  and repeat ``python -m pip install -e ".[demo]"`` from the repository root.
  Check that your checkout contains ``ebdai/demo_app.py``.
- **No demos found:** run from the repository root or use ``--demos``.
- **Session URL requested:** open the full URL from the current terminal.
  Restarting the server creates a new token.
- **Dataset download fails:** examples 02 and 05 require internet access on
  their first run. Bias examples 09–11 load bundled data.
- **Widget output unavailable:** live Jupyter widgets and arbitrary JavaScript
  outputs need Jupyter. The app displays text, HTML tables, and PNG plots.

The app executes trusted local Python code with your environment's permissions
and binds only to loopback. It does not modify the rebased ``ex_fuzzy`` library.
