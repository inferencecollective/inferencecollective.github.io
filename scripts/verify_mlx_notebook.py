"""Execute mlx.ipynb in the invoking Python environment and check its results."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / 'priv/static/aidatafoundation/notebooks/mlx.ipynb', as_version=4)
    nbformat.validate(notebook)
    with TemporaryDirectory(prefix='ic-mlx-notebook-') as temporary:
        temporary = Path(temporary)
        kernel_dir = temporary / 'kernels/python3'
        kernel_dir.mkdir(parents=True)
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Workshop verification', 'language': 'python',
            'env': {'MLX_WORKSHOP_DEVICE': args.device, 'MPLCONFIGDIR': str(temporary / 'matplotlib'),
                    'IPYTHONDIR': str(temporary / 'ipython'), 'XDG_CACHE_HOME': str(temporary / 'cache')},
        }))
        manager = KernelManager(kernel_name='python3', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temporary / 'kernels')]))
        client = NotebookClient(notebook, km=manager, timeout=120, resources={'metadata': {'path': str(temporary)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        artifacts = temporary / 'outputs/mlx-notebook'
        report = json.loads((artifacts / f'report-{args.device}.json').read_text())
        assert report['device'] == args.device
        assert report['test_mse'] < 0.05
        assert abs(report['slope'] - 2.5) < 0.1
        assert abs(report['intercept'] + 0.8) < 0.1
        assert all(row['outputs_match'] and row['compiled_median_ms'] > 0 for row in report['benchmark'])
        images = [output for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs if 'image/png' in output.get('data', {})]
        assert len(images) >= 2, 'Expected training and latency plots in notebook output'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
        target = args.output.parent / f'mlx-results-{args.device}'
        shutil.copytree(artifacts, target, dirs_exist_ok=True)
        print(f"Verified {args.device}: held-out MSE {report['test_mse']:.5f}, {len(images)} plots; saved {args.output}")


if __name__ == '__main__':
    main()
