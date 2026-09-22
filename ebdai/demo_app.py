"""Local notebook demo app. Run with ``python -m ebdai.demo_app``."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ._demo_notebooks import ConflictError, NotebookStore, apply_overrides


class DemoApplication:
    """Manage a single cancellable execution and notebook files."""

    def __init__(self, root: Path):
        self.store = NotebookStore(root)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.workspace = tempfile.TemporaryDirectory(prefix='ebdai-demo-')
        self.process = None
        self.run_directory = None
        self.run_info = None

    def state(self) -> dict:
        with self.lock:
            if self.run_info is None:
                return {'status': 'idle'}
            result = dict(self.run_info)
            path = self.run_directory / 'state.json'
            if path.exists():
                result.update(json.loads(path.read_text()))
            if self.run_info.get('status') == 'cancelled':
                result['status'] = 'cancelled'
            elif self.process.poll() is not None and result['status'] in ('starting', 'running'):
                result.update(status='error', error=(self.run_directory / 'worker.log').read_text()
                              or 'The notebook worker exited unexpectedly.')
            return result

    def run(self, body: dict) -> dict:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ConflictError('An example is already running. Stop it before starting another.')
            document = self.store.checked(body['name'], body['revision'])
            notebook = apply_overrides(document['notebook'], body.get('overrides', {}))
            through = body.get('through')
            if through is not None:
                if type(through) is not int or not 0 <= through < len(notebook['cells']):
                    raise ValueError('Invalid final cell')
                notebook['cells'] = notebook['cells'][:through + 1]
            run_id = secrets.token_hex(8)
            if self.run_directory is not None:
                shutil.rmtree(self.run_directory)
            self.run_directory = Path(self.workspace.name) / run_id
            self.run_directory.mkdir()
            (self.run_directory / 'input.ipynb').write_text(json.dumps(notebook))
            self.run_info = dict(id=run_id, name=body['name'], revision=document['revision'],
                                 status='starting', overrides=body.get('overrides', {}))
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent) + os.pathsep + environment.get('PYTHONPATH', '')
            environment['IPYTHONDIR'] = str(self.run_directory / 'ipython')
            environment['JUPYTER_RUNTIME_DIR'] = str(self.run_directory / 'jupyter')
            environment['MPLCONFIGDIR'] = str(self.run_directory / 'matplotlib')
            with (self.run_directory / 'worker.log').open('w') as log:
                self.process = subprocess.Popen(
                    [sys.executable, '-m', 'ebdai._demo_worker', str(self.run_directory), str(self.store.root)],
                    stdout=log, stderr=log, env=environment, start_new_session=True)
            return self.state()

    def stop(self) -> dict:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                # nbclient's SIGTERM handler cleans up its kernel, which may have its own session.
                self.process.terminate()
                try:
                    self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait()
                self.run_info['status'] = 'cancelled'
            return self.state()

    def close(self):
        self.stop()
        self.workspace.cleanup()


def make_server(app: DemoApplication, port: int = 8765) -> ThreadingHTTPServer:
    """Bind only to loopback; require a session token for every API operation."""
    static = Path(__file__).with_name('demo_static')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, status=200, content_type='application/json'):
            if content_type == 'application/json':
                data = json.dumps(data).encode()
            elif isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; frame-src 'self' about:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            host = self.headers.get('Host', '')
            expected = f'127.0.0.1:{self.server.server_port}'
            origin = self.headers.get('Origin')
            if host not in (expected, f'localhost:{self.server.server_port}'):
                self.respond({'error': 'Invalid host'}, 403)
                return False
            if origin and origin != f'http://{host}':
                self.respond({'error': 'Invalid origin'}, 403)
                return False
            if not secrets.compare_digest(self.headers.get('X-Demo-Token', ''), app.token):
                self.respond({'error': 'Open the session URL printed in your terminal.'}, 403)
                return False
            return True

        def do_GET(self):
            url = urlparse(self.path)
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if url.path in assets:
                name, kind = assets[url.path]
                return self.respond((static / name).read_bytes(), content_type=kind)
            if not self.authorized():
                return
            try:
                query = parse_qs(url.query)
                if url.path == '/api/demos':
                    documents = [app.store.read(p.name) for p in app.store.paths()]
                    return self.respond([dict(name=d['name'], title=d['title']) for d in documents])
                if url.path == '/api/notebook':
                    return self.respond(app.store.read(query['name'][0]))
                if url.path == '/api/revision':
                    return self.respond({'revision': app.store.read(query['name'][0])['revision']})
                if url.path == '/api/state':
                    return self.respond(app.state())
                if url.path == '/api/download':
                    with app.lock:
                        if app.run_directory is None:
                            raise ValueError('Run an example first')
                        path = app.run_directory / 'result.ipynb'
                        if not path.exists():
                            path = app.run_directory / 'input.ipynb'
                        return self.respond(path.read_bytes(), content_type='application/x-ipynb+json')
                self.respond({'error': 'Not found'}, 404)
            except (ValueError, KeyError, OSError) as error:
                self.respond({'error': str(error)}, 400)

        def do_POST(self):
            if not self.authorized():
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 10_000_000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Request must be a JSON object')
                with app.lock:
                    if self.path == '/api/run':
                        result = app.run(body)
                    elif self.path == '/api/stop':
                        result = app.stop()
                    elif self.path == '/api/save':
                        result = app.store.save(body['name'], body['revision'], body['cells'])
                    else:
                        return self.respond({'error': 'Not found'}, 404)
                self.respond(result)
            except ConflictError as error:
                self.respond({'error': str(error)}, 409)
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.respond({'error': str(error)}, 400)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--demos', type=Path, default=Path.cwd() / 'Demos', help='Folder containing the original demo notebooks')
    args = parser.parse_args()
    missing = [name for name in ('nbclient', 'nbformat', 'ipykernel') if importlib.util.find_spec(name) is None]
    if missing:
        parser.error('Install demo dependencies with: pip install -e ".[demo]"')
    app = DemoApplication(args.demos)
    if not app.store.paths():
        parser.error('No demos found. Run from the repository root or pass --demos /path/to/Demos')
    server = make_server(app, args.port)
    print(f'BDI Demo Studio: http://127.0.0.1:{server.server_port}/#{app.token}', flush=True)
    print('Keep this terminal running. Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == '__main__':
    main()
