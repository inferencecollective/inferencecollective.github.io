"""Execute a Colab-compatible notebook with the invoking Python environment."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import json

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track', choices=['ml-foundations', 'pytorch', 'vision'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / f'priv/static/aidatafoundation/notebooks/{args.track}.ipynb', as_version=4)
    nbformat.validate(notebook)
    with TemporaryDirectory(prefix='ic-notebook-') as temporary:
        temporary = Path(temporary)
        kernel_dir = temporary / 'kernels/python3'
        kernel_dir.mkdir(parents=True)
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Workshop verification', 'language': 'python',
            'env': {'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(temporary / 'matplotlib'),
                    'IPYTHONDIR': str(temporary / 'ipython'), 'XDG_CACHE_HOME': str(temporary / 'cache'),
                    'OMP_NUM_THREADS': '2'},
        }))
        manager = KernelManager(kernel_name='python3', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temporary / 'kernels')]))
        # CI installs requirements before execution; users run the installation cell in Colab.
        client = NotebookClient(notebook, km=manager, timeout=180,
                                skip_cells_with_tag='installation',
                                resources={'metadata': {'path': str(temporary)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        images = [out for cell in notebook.cells if cell.cell_type == 'code' for out in cell.outputs if 'image/png' in out.get('data', {})]
        assert images, 'Expected a generated lab image to render in notebook output'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
        print(f'Verified {args.track}: all exercise cells executed; saved {args.output}')


if __name__ == '__main__':
    main()
