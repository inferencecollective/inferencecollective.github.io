"""Generate standalone Livebook notebooks from the curated workshop catalog."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_REVISION = 'a9c6139d8d9ff9e6aae8b7b28d41e7f796639a95'  # Published commit containing all workshop modules and data.
ASSETS = {
    'ml-foundations': ['labs/common.py', 'data/orders.csv', 'data/customers_churn.csv'],
    'pytorch': ['labs/common.py', 'data/pytorch-regression.pt', 'data/pytorch-moons.pt'],
    'vision': ['data/scene-01.png', 'data/scene-02.png'],
    'mlx': ['labs/mlx_runtime.py'],
}


def generate():
    catalog = json.loads((ROOT / 'priv/static/aidatafoundation/data/workshops.json').read_text())
    destination = ROOT / 'priv/static/aidatafoundation/notebooks'
    destination.mkdir(exist_ok=True)
    for track in catalog['tracks']:
        assets = ASSETS[track['id']] + [s['module'].replace('.', '/') + '.py' for s in track['steps']]
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in assets}
        requirements = [line for line in (ROOT / f"requirements/workshops/{track['id']}.txt").read_text().splitlines() if line and not line.startswith('#')]
        header = f'''<!-- livebook:{{"default_language":"python"}} -->

# {track['title']}

```elixir
Mix.install([{{:pythonx, "~> 0.4.9"}}, {{:kino_pythonx, "~> 0.1.0"}}])
```

```pyproject.toml
[project]
name = "inference-collective-{track['id']}"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = {json.dumps(requirements)}
```

## Before you run

{track['description']}

Prerequisites: {track['prerequisites']}

Evaluate the setup and Python cells in order. The first Python cell downloads a small, explicit set of lab files and sample data from published commit `{SOURCE_REVISION}`. Every file is SHA-256 checked before execution. This needs network access the first time. Outputs stay in a temporary directory on the notebook runtime machine until you save them elsewhere.

For collaboration, open this notebook once on a shared, authenticated Livebook server and give your group the same session URL. Separate imports create independent notebooks. Only invite people you trust to edit and execute code in that runtime.
'''
        if track['id'] == 'mlx':
            header += '\nMLX requires this Livebook runtime to run on Apple silicon with macOS 14+ and native arm64 Python. A cloud runtime does not use the GPU in your browser’s Mac.\n'
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
        if track['id'] == 'mlx':
            bootstrap += '\n# Set to "cpu" or "gpu" to choose explicitly.\nos.environ["MLX_WORKSHOP_DEVICE"] = "auto"'
        notebook = header + '\n## Prepare your workspace\n\n```python\n' + bootstrap + '\n```\n'
        for index, step in enumerate(track['steps'], 1):
            notebook += f'''\n## {index}. {step['title']}

Inspect `{step['module'].replace('.', '/')}.py` in your downloaded workspace before running.

```python
import runpy
runpy.run_module({step['module']!r}, run_name="__main__")
print("Inspect outputs in", workshop_root / "outputs")
```

**Expected output:** {step['outcome']}

**Try it:** {step['challenge']}
'''
        notebook += '''
## Inspect and keep your outputs

```python
_output_files = sorted((workshop_root / "outputs").glob("*")) if (workshop_root / "outputs").exists() else []
for _file in _output_files:
    print(_file.name)
    if _file.suffix == ".json":
        print(_file.read_text())
```

Copy any output files you want to keep out of the temporary runtime directory. For image outputs, use the Python cell below to preview the first saved PNG.
'''
        if track['id'] != 'mlx':
            notebook += '''
```python
from PIL import Image
_png_files = list((workshop_root / "outputs").glob("*.png"))
Image.open(_png_files[0]) if _png_files else "No PNG outputs yet. Run the model or image lab first."
```
'''
        notebook += '\nShare one result, the parameter you changed, and your explanation with your group.\n'
        (destination / f"{track['id']}.livemd").write_text(notebook)


if __name__ == '__main__':
    generate()
