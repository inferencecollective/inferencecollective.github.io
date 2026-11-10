"""Generate portable .ipynb notebooks for the Colab-compatible workshop tracks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_REVISION = 'a9c6139d8d9ff9e6aae8b7b28d41e7f796639a95'
ASSETS = {
    'ml-foundations': ['labs/common.py', 'data/orders.csv', 'data/customers_churn.csv'],
    'pytorch': ['labs/common.py', 'data/pytorch-regression.pt', 'data/pytorch-moons.pt'],
    'vision': ['data/scene-01.png', 'data/scene-02.png'],
}


def generate():
    catalog = json.loads((ROOT / 'priv/static/aidatafoundation/data/workshops.json').read_text())
    destination = ROOT / 'priv/static/aidatafoundation/notebooks'
    for track in catalog['tracks']:
        if track['id'] not in ASSETS:
            continue
        assets = ASSETS[track['id']] + [s['module'].replace('.', '/') + '.py' for s in track['steps']]
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in assets}
        requirements = [line for line in (ROOT / f"requirements/workshops/{track['id']}.txt").read_text().splitlines() if line and not line.startswith('#')]
        bootstrap = f'''import hashlib
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

workshop_root = Path(tempfile.mkdtemp(prefix="ic-{track['id']}-"))
_workshop_assets = {hashes!r}
for _relative, _expected in _workshop_assets.items():
    _url = "https://raw.githubusercontent.com/inferencecollective/inferencecollective.github.io/{SOURCE_REVISION}/" + _relative
    with urlopen(_url, timeout=30) as _response:
        _payload = _response.read()
    if hashlib.sha256(_payload).hexdigest() != _expected:
        raise RuntimeError("Source verification failed: " + _relative)
    _target = workshop_root / _relative
    _target.parent.mkdir(parents=True, exist_ok=True)
    _target.write_bytes(_payload)
(workshop_root / "labs/__init__.py").touch()
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(workshop_root))
# Re-evaluating setup must import the new copies, not cached lab modules.
for _name in list(sys.modules):
    if _name == "labs" or _name.startswith("labs."):
        del sys.modules[_name]
print("Workshop files verified. Outputs:", workshop_root / "outputs")'''

        cells = []
        def md(text):
            cells.append(dict(cell_type='markdown', metadata={}, source=text + '\n'))
        def code(text, tags=None):
            cells.append(dict(cell_type='code', metadata={'tags': tags or []}, execution_count=None, outputs=[], source=text + '\n'))
        md(f"# {track['title']}\n\nInference Collective · ~{track['minutes']} minutes\n\n{track['description']}\n\n**Prerequisites:** {track['prerequisites']}\n\nOpen in Google Colab or use a Python 3.11–3.13 kernel in VS Code/Jupyter. Run the installation cell first; restart the runtime if prompted, then run the remaining cells in order. Save a copy before editing. Notebook content can be shared; runtime state and files are separate for each participant.")
        code('%pip install ' + ' '.join('"' + req + '"' for req in requirements), ['installation'])
        md('## Prepare the lab files\n\nThis cell downloads only the listed lab modules and synthetic data from a pinned source commit, then checks SHA-256 hashes. Inspect the source before running it. Outputs are stored in a temporary runtime directory: download files you want to keep before the runtime ends.')
        code(bootstrap)
        for index, step in enumerate(track['steps'], 1):
            md(f"## {index}. {step['title']}\n\n**Expected:** {step['outcome']}\n\n**Exercise:** {step['challenge']}")
            code(f"import runpy\nrunpy.run_module({step['module']!r}, run_name='__main__')\nprint('Outputs:', workshop_root / 'outputs')")
        md('## Inspect your output\n\nOpen the generated files or preview the first PNG below. Colab Files lets you download outputs. Change a parameter, rerun, and explain the difference to your group.')
        code("from PIL import Image\npng_files = sorted((workshop_root / 'outputs').glob('*.png'))\nImage.open(png_files[0]) if png_files else 'Run the labs first to generate images.'")
        for index, cell in enumerate(cells):
            cell['id'] = f"{track['id']}-{index:02d}"
        notebook = dict(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}, nbformat=4, nbformat_minor=5)
        (destination / f"{track['id']}.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    generate()
