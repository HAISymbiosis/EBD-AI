"""Notebook source preservation and real local demo execution."""
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ebdai._demo_notebooks import ConflictError, NotebookStore, apply_overrides, parameters
from ebdai.demo_app import DemoApplication, make_server


def notebook(text):
    return dict(nbformat=4, nbformat_minor=5, metadata={}, cells=[
        dict(id='test-cell', cell_type='code', source=text, metadata={}, outputs=[], execution_count=None)])


def test_parameter_overrides_preserve_source_and_unicode():
    original = notebook('# café\nx = dict(n_gen=30, pop_size=20, patience=None)  # keep comment\ny = dict(n_gen=x)\n')
    controls = parameters(original)
    assert [p['name'] for p in controls] == ['n_gen', 'pop_size', 'patience']
    result = apply_overrides(original, {controls[0]['id']: 2})
    assert 'n_gen=2,' in result['cells'][0]['source']
    assert '# café' in result['cells'][0]['source']
    assert '# keep comment' in result['cells'][0]['source']
    assert 'n_gen=30,' in original['cells'][0]['source']
    with pytest.raises(ValueError, match='stale'):
        apply_overrides(original, {'unknown': 1})


def test_settings_grid_and_assignment_controls():
    original = notebook("CONFIG = {'n_gen': 30}\npatience = None\ngrid = {'model__n_gen': [15, 30]}\n")
    controls = parameters(original)
    assert [p['name'] for p in controls] == ['n_gen', 'patience', 'model__n_gen']
    result = apply_overrides(original, {controls[-1]['id']: [2, 3]})
    assert "'model__n_gen': [2, 3]" in result['cells'][0]['source']
    with pytest.raises(ValueError, match='finite'):
        apply_overrides(original, {controls[0]['id']: float('nan')})


def test_store_detects_external_edits_and_keeps_metadata(tmp_path):
    path = tmp_path / '01_example.ipynb'
    data = notebook('print(1)')
    data['metadata']['custom'] = {'keep': True}
    path.write_text(json.dumps(data))
    store = NotebookStore(tmp_path)
    first = store.read(path.name)
    path.write_text(json.dumps(notebook('print(2)')))
    with pytest.raises(ConflictError):
        store.save(path.name, first['revision'], {'0': 'print(3)'})
    path.write_text(json.dumps(data))
    saved = store.save(path.name, first['revision'], {'0': 'print(4)'})
    assert saved['notebook']['metadata']['custom'] == {'keep': True}
    assert saved['notebook']['cells'][0]['source'] == 'print(4)'
    with pytest.raises(ValueError, match='Unknown'):
        store.read('../elsewhere.ipynb')


def test_all_existing_examples_discovered():
    store = NotebookStore(Path(__file__).resolve().parents[1] / 'Demos')
    expected = set(store.root.glob('[0-9]*.ipynb')) | set(store.root.glob('*_demo.py'))
    assert set(store.paths()) == expected
    assert len(expected) >= 10
    for path in store.paths():
        demo = store.read(path.name)
        assert demo['title'] and demo['notebook']['cells']
        assert demo['parameters'], path.name


def wait_until_done(app):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = app.state()
        if state['status'] not in ('starting', 'running'):
            # Result file is written immediately after completion state.
            app.process.wait(timeout=10)
            return state
        time.sleep(.1)
    app.stop()
    pytest.fail('Notebook execution timed out')


def test_real_kernel_outputs_errors_and_cancellation(tmp_path):
    pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    path = tmp_path / '01_example.ipynb'
    text = "import sys\nimport pandas as pd\nimport matplotlib.pyplot as plt\nsettings = dict(n_gen=30)\nprint('generations', settings['n_gen'])\nplt.plot([1,2], [3,4])\ndisplay(pd.DataFrame({'value':[7]}))\nprint(sys.executable)"
    path.write_text(json.dumps(notebook(text)))
    app = DemoApplication(tmp_path)
    try:
        document = app.store.read(path.name)
        app.run(dict(name=path.name, revision=document['revision'], overrides={document['parameters'][0]['id']: 2}))
        with pytest.raises(ConflictError):
            app.run(dict(name=path.name, revision=document['revision']))
        state = wait_until_done(app)
        assert state['status'] == 'complete', state
        outputs = state['notebook']['cells'][0]['outputs']
        assert any('generations 2' in o.get('text', '') for o in outputs)
        assert any('text/html' in o.get('data', {}) for o in outputs)
        assert any('image/png' in o.get('data', {}) for o in outputs)
        assert path.read_text() == json.dumps(notebook(text))
        assert (app.run_directory / 'result.ipynb').exists()
        previous_directory = app.run_directory
        document = app.store.save(path.name, document['revision'], {'0': "raise ValueError('intentional failure')"})
        app.run(dict(name=path.name, revision=document['revision']))
        assert not previous_directory.exists()
        assert wait_until_done(app)['status'] == 'error'
        document = app.store.save(path.name, document['revision'], {'0': 'import time\ntime.sleep(60)'})
        app.run(dict(name=path.name, revision=document['revision']))
        time.sleep(2)
        assert app.stop()['status'] == 'cancelled'
        assert app.process.poll() is not None
    finally:
        app.close()


def test_http_api_requires_token_and_rejects_other_origins(tmp_path):
    (tmp_path / '01_example.ipynb').write_text(json.dumps(notebook('print(1)')))
    app = DemoApplication(tmp_path)
    server = make_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        assert b'Demo Studio' in urlopen(base).read()
        with pytest.raises(HTTPError) as error:
            urlopen(base + '/api/demos')
        assert error.value.code == 403
        response = urlopen(Request(base + '/api/demos', headers={'X-Demo-Token': app.token}))
        assert json.load(response)[0]['name'] == '01_example.ipynb'
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + '/api/stop', data=b'{}', headers={
                'X-Demo-Token': app.token, 'Origin': 'https://example.com'}))
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_partial_runs_start_fresh_and_keep_later_cells_unexecuted(tmp_path):
    pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    data = notebook("assert 'prior_run' not in globals()\nprior_run = 1\nprint('first cell')")
    data['cells'].append(dict(id='later', cell_type='code', metadata={}, outputs=[],
                              execution_count=None, source="raise AssertionError('should not run')"))
    path = tmp_path / '01_partial.ipynb'
    path.write_text(json.dumps(data))
    app = DemoApplication(tmp_path)
    try:
        document = app.store.read(path.name)
        for _ in range(2):
            app.run(dict(name=path.name, revision=document['revision'], through=0))
            result = wait_until_done(app)
            assert result['status'] == 'complete', result
            assert len(result['notebook']['cells']) == 1
        assert len(json.loads(path.read_text())['cells']) == 2
    finally:
        app.close()
