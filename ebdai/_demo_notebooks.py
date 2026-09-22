"""Notebook discovery and source-preserving parameter overrides for the demo UI."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

# Only literal values are exposed. Expressions and arbitrary code remain in the editor.
PARAMETERS = {
    'nRules', 'nAnts', 'n_gen', 'pop_size', 'patience', 'random_state',
    'n_linguistic_variables', 'ds_mode', 'tolerance', 'alpha', 'beta', 'lam',
    'test_size', 'train_size', 'n_splits', 'cv', 'max_rules', 'max_depth',
    'n_output_lvs', 'support_threshold', 'n_bootstrap', 'n_permutations',
    'n_repeats', 'n_jobs', 'consequent_type', 'score_type', 'lower_height',
    'sbx_eta', 'mutation_eta', 'var_prob', 'large_dataset_samples',
    'large_dataset_features',
}


class ConflictError(ValueError):
    """The notebook changed since the browser last loaded it."""


def source(cell: dict) -> str:
    """Return source text from either notebook serialization format."""
    value = cell.get('source', '')
    return ''.join(value) if isinstance(value, list) else value


def revision(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parameter_nodes(text: str):
    """Yield literal settings in calls, dictionaries, and named assignments."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            candidates.extend((kw.arg, kw.value, ast.unparse(node.func)) for kw in node.keywords)
        elif isinstance(node, ast.Dict):
            candidates.extend((key.value, value, 'settings') for key, value in zip(node.keys, node.values)
                              if isinstance(key, ast.Constant) and isinstance(key.value, str))
        elif isinstance(node, ast.Assign):
            candidates.extend((target.id, node.value, 'assignment') for target in node.targets
                              if isinstance(target, ast.Name))
    seen = set()
    for name, node, call in sorted(candidates, key=lambda item: (item[1].lineno, item[1].col_offset)):
        if name is None or name.split('__')[-1] not in PARAMETERS:
            continue
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError):
            continue
        location = (node.lineno, node.col_offset)
        if valid_parameter(value) and location not in seen:
            seen.add(location)
            yield name, node, value, call


def valid_parameter(value: Any) -> bool:
    """Accept finite JSON scalar values and lists of those values."""
    if isinstance(value, (list, tuple)):
        return all(not isinstance(item, (list, tuple)) and valid_parameter(item) for item in value)
    return (value is None or type(value) in (int, bool, str)
            or type(value) is float and math.isfinite(value))


def parameters(notebook: dict) -> list[dict]:
    result = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        for name, node, value, call in parameter_nodes(source(cell)):
            result.append(dict(id=f'{index}:{node.lineno}:{node.col_offset}',
                               cell=index, name=name, value=value, call=call))
    return result


def apply_overrides(notebook: dict, overrides: dict[str, Any]) -> dict:
    """Replace only selected literal spans in a copy; never rewrite notebook files."""
    result = copy.deepcopy(notebook)
    known = {item['id']: item for item in parameters(notebook)}
    if set(overrides) - known.keys():
        raise ValueError('Parameter controls are stale; reload the notebook')
    for value in overrides.values():
        if not valid_parameter(value):
            raise ValueError('Parameters must be finite JSON values or lists of scalar values')
    for index, cell in enumerate(result['cells']):
        if cell['cell_type'] != 'code':
            continue
        original = source(cell)
        lines = original.encode('utf-8').splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        edits = []
        for _, node, _, _ in parameter_nodes(original):
            key = f'{index}:{node.lineno}:{node.col_offset}'
            if key in overrides:
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.end_lineno - 1] + node.end_col_offset
                edits.append((start, end, repr(overrides[key]).encode()))
        data = original.encode()
        for start, end, value in sorted(edits, reverse=True):
            data = data[:start] + value + data[end:]
        cell['source'] = data.decode()
        cell['outputs'] = []
        cell['execution_count'] = None
    return result


class NotebookStore:
    """Read demos in place and protect saves/runs with content revisions."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def paths(self) -> list[Path]:
        return sorted(self.root.glob('[0-9]*.ipynb')) + sorted(self.root.glob('*_demo.py'))

    def path(self, name: str) -> Path:
        candidates = {p.name: p for p in self.paths() if p.resolve().parent == self.root}
        if name not in candidates:
            raise ValueError('Unknown demo')
        return candidates[name]

    def read(self, name: str) -> dict:
        path = self.path(name)
        data = path.read_bytes()
        if path.suffix == '.py':
            notebook = dict(nbformat=4, nbformat_minor=5, metadata={}, cells=[
                dict(id='script', cell_type='code', metadata={}, source=data.decode(), outputs=[], execution_count=None)])
        else:
            notebook = json.loads(data)
        for cell in notebook['cells']:
            cell['source'] = source(cell)
        title = name.replace('_', ' ').rsplit('.', 1)[0]
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown' and source(cell).startswith('# '):
                title = source(cell).splitlines()[0].lstrip('# ')
                break
        return dict(name=name, title=title, revision=revision(data), notebook=notebook,
                    parameters=parameters(notebook))

    def checked(self, name: str, expected: str) -> dict:
        document = self.read(name)
        if document['revision'] != expected:
            raise ConflictError('The file changed on disk. Reload before saving or running.')
        return document

    def save(self, name: str, expected: str, cells: dict[str, str]) -> dict:
        document = self.checked(name, expected)
        notebook = document['notebook']
        for index, text in cells.items():
            if not isinstance(text, str):
                raise ValueError('Cell source must be text')
            index = int(index)
            if not 0 <= index < len(notebook['cells']):
                raise ValueError('Invalid cell index')
            notebook['cells'][index]['source'] = text
        # Changed code invalidates all stored outputs, including downstream cells.
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        path = self.path(name)
        text = (source(notebook['cells'][0]) if path.suffix == '.py'
                else json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root,
                                         suffix='.tmp', delete=False) as handle:
            temp = Path(handle.name)
            handle.write(text)
        try:
            self.checked(name, expected)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return self.read(name)
