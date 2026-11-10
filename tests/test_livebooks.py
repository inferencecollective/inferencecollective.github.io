"""Check notebook source integrity and execute the vision notebook end to end."""
import ast
import hashlib
from io import BytesIO
from pathlib import Path
import re
import tomllib
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / 'priv/static/aidatafoundation/notebooks'


@pytest.mark.parametrize('track', ['ml-foundations', 'pytorch', 'vision', 'mlx'])
def test_notebook_has_python_setup_and_parseable_cells(track):
    text = (NOTEBOOKS / f'{track}.livemd').read_text()
    project = tomllib.loads(re.search(r'```pyproject.toml\n(.*?)\n```', text, re.S)[1])
    assert project['project']['requires-python'] == '==3.12.*'
    cells = re.findall(r'```python\n(.*?)\n```', text, re.S)
    assert len(cells) >= 4
    for cell in cells:
        ast.parse(cell)
    assignments = ast.parse(cells[0]).body
    assets = next(ast.literal_eval(node.value) for node in assignments if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_workshop_assets' for t in node.targets))
    for name, digest in assets.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_vision_notebook_executes_with_verified_assets(tmp_path, monkeypatch):
    text = (NOTEBOOKS / 'vision.livemd').read_text()
    cells = re.findall(r'```python\n(.*?)\n```', text, re.S)
    def fetch(url, timeout):
        relative = url.split('.github.io/', 1)[1].split('/', 1)[1]
        return BytesIO((ROOT / relative).read_bytes())
    import sys
    monkeypatch.setattr(sys, 'path', list(sys.path))
    namespace = {}
    with patch('urllib.request.urlopen', side_effect=fetch), patch('tempfile.mkdtemp', return_value=str(tmp_path)):
        for cell in cells:
            exec(compile(cell, 'vision.livemd', 'exec'), namespace)
    assert (tmp_path / 'outputs/01-resize-comparison.png').is_file()
    assert (tmp_path / 'outputs/02-enhancement-grid.png').is_file()
