"""Opt-in browser acceptance test: EBD_DEMO_BROWSER=1 pytest this file."""
import json
import os
from pathlib import Path
import shutil
import threading

import pytest

from ebdai.demo_app import DemoApplication, make_server

pytestmark = pytest.mark.skipif(os.environ.get('EBD_DEMO_BROWSER') != '1',
                                reason='Set EBD_DEMO_BROWSER=1 and install Playwright/Chromium')


def test_browser_controls_execution_editing_and_sync(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    demo_root = Path(__file__).resolve().parents[1] / 'Demos'
    for path in demo_root.glob('*.ipynb'):
        shutil.copy2(path, tmp_path / path.name)
    for path in demo_root.glob('*_demo.py'):
        shutil.copy2(path, tmp_path / path.name)
    editable = tmp_path / '00_browser_test.ipynb'
    fixture = dict(nbformat=4, nbformat_minor=5, metadata={}, cells=[
        dict(id='intro', cell_type='markdown', metadata={}, source='# Browser test\nA small **live** notebook.'),
        dict(id='code', cell_type='code', metadata={}, execution_count=None, outputs=[],
             source="settings = dict(n_gen=30)\nprint('generations', settings['n_gen'])")])
    editable.write_text(json.dumps(fixture))
    app = DemoApplication(tmp_path)
    server = make_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/#{app.token}')
            page.locator('#title').filter(has_text='Browser test').wait_for()
            assert page.locator('#menu button').count() == len(app.store.paths())
            page.locator('[data-parameter="n_gen"]').fill('2')
            page.locator('#run').click()
            page.get_by_text('generations 2', exact=True).wait_for(timeout=30000)
            page.get_by_text('Run complete', exact=True).wait_for()
            assert 'n_gen=30' in editable.read_text()
            with page.expect_download() as download_info:
                page.locator('#download').click()
            downloaded = json.loads(Path(download_info.value.path()).read_text())
            assert 'n_gen=2' in ''.join(downloaded['cells'][1]['source'])

            page.get_by_role('tab', name='Notebook & code').click()
            code = page.get_by_role('textbox', name='Cell 2 source', exact=True)
            code.fill("settings = dict(n_gen=7)\nprint('edited in browser')")
            page.locator('#save').click()
            page.get_by_text('Notebook saved. Controls now reflect your code.').wait_for()
            assert 'edited in browser' in editable.read_text()
            assert page.locator('[data-parameter="n_gen"]').input_value() == '7'

            # A save from Jupyter must reload controls without running any code.
            saved = json.loads(editable.read_text())
            saved['cells'][1]['source'] = "settings = dict(n_gen=9)\nprint('edited externally')"
            editable.write_text(json.dumps(saved))
            playwright.expect(page.locator('[data-parameter="n_gen"]')).to_have_value('9')
            assert 'edited externally' in code.input_value()
            code.fill("print('unsaved browser work')")
            saved['cells'][1]['source'] = "print('external conflict')"
            editable.write_text(json.dumps(saved))
            page.get_by_text('The notebook changed on disk.', exact=False).wait_for()
            assert code.input_value() == "print('unsaved browser work')"
            page.locator('#save').click()
            page.get_by_text('The file changed on disk.', exact=False).wait_for()
            assert 'external conflict' in editable.read_text()
            page.locator('#notice button').click()
            playwright.expect(code).to_have_value("print('external conflict')")

            # Execute a shipped example and verify rich outputs in the real UI.
            page.get_by_role('button', name='9. Bias in the data: Titanic survival', exact=True).click()
            page.locator('[data-parameter="n_gen"]').fill('2')
            page.locator('[data-parameter="pop_size"]').fill('8')
            page.locator('#run').click()
            page.get_by_text('Run complete', exact=True).wait_for(timeout=60000)
            assert page.locator('.output img').count() >= 2
            assert page.locator('.output.error').count() == 0
            page.screenshot(path='/tmp/ebdai-demo-studio.png', full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        app.close()
