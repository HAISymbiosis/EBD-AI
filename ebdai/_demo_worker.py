"""Execute a snapshot in a fresh Jupyter kernel and publish cell progress."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback


def main() -> None:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager

    directory, working_directory = map(Path, sys.argv[1:3])
    notebook = nbformat.read(directory / 'input.ipynb', as_version=4)
    state = dict(status='running', cell=None, completed=0, total=len(notebook.cells))

    def publish(**updates):
        state.update(updates)
        temp = directory / 'state.tmp'
        temp.write_text(json.dumps(dict(**state, notebook=notebook)))
        os.replace(temp, directory / 'state.json')

    def started(cell, cell_index, **kwargs):
        publish(cell=cell_index)

    def completed(cell, cell_index, **kwargs):
        publish(completed=cell_index + 1)

    manager = AsyncKernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=600, record_timing=False,
                            on_cell_start=started, on_cell_executed=completed)
    try:
        # Inline figures are captured as notebook outputs, including bare expressions.
        client.execute(cwd=str(working_directory), cleanup_kc=True,
                       env={**os.environ, 'MPLBACKEND': 'module://matplotlib_inline.backend_inline'})
        publish(status='complete', completed=len(notebook.cells))
    except BaseException:
        publish(status='error', error=traceback.format_exc())
    finally:
        nbformat.write(notebook, directory / 'result.ipynb')


if __name__ == '__main__':
    main()
