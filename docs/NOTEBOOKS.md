# Workshop notebooks

The website offers `.ipynb` downloads for all four tracks and Open in Google Colab buttons for ML foundations, PyTorch, and vision. These links work with the static GitHub Pages deployment; no notebook server is hosted by the website.

## Google Colab

Open a compatible track with its Colab button. Save a copy to your Drive, run the installation cell first, restart if package installation prompts you to, and then run the exercise cells in order. The notebooks download only an explicit list of source modules and sample data from a pinned repository commit, checking SHA-256 hashes before execution. Runtime files are temporary, so download outputs you want to keep.

You can share your saved notebook with collaborators. Colab shares notebook content and outputs; each participant's runtime state, installed packages, and generated files are separate. See [Colab's sharing documentation](https://research.google.com/colaboratory/faq.html#sharing).

## Local notebooks

Open the downloaded notebook in VS Code or Jupyter. Choose an environment containing the track dependencies and an IPython kernel. From the repository:

```bash
python -m pip install -r requirements/workshops/pytorch.txt ipykernel
```

Replace `pytorch` with `ml-foundations` or `vision` for other tracks. You can also run the notebook's installation cell in your selected environment. Restart the kernel and Run All for a reproducible execution.

## MLX

`mlx.ipynb` is self-contained and targets Apple silicon, macOS 14+, and native arm64 Python 3.11–3.13. Install `requirements/workshops/mlx-notebook.txt`, select that Python kernel, restart, and Run All. The notebook includes training, held-out evaluation, saving and reloading weights, inference, and compilation benchmarks with plots. This specific notebook is not offered for a hosted Colab runtime: it requires the Mac runtime, not just a browser on a Mac.

## Updating and validating notebooks

Generate portable notebooks with `python3 scripts/create_workshop_notebooks.py` and MLX with `python3 scripts/create_mlx_notebook.py`. The portable generator pins the lab source commit in `SOURCE_REVISION`; update that after publishing new lab code.

The download URLs use the local static assets. GitHub previews and Colab launchers use `WORKSHOP_NOTEBOOK_REVISION`, defaulting to the published notebook revision pinned in the application. When publishing updated notebook files, set that variable to their commit SHA (or `main` after merging), then rebuild the website so the links match the deployed source. This avoids broken Colab URLs for notebooks that are still on a branch.

Validation runs the actual notebooks in a fresh kernel using the invoking Python environment. Install `nbformat`, `nbclient`, and `ipykernel` alongside the selected track dependencies, then:

```bash
python scripts/verify_workshop_notebook.py vision --output outputs/vision.ipynb
python scripts/verify_workshop_notebook.py ml-foundations --output outputs/ml-foundations.ipynb
python scripts/verify_workshop_notebook.py pytorch --output outputs/pytorch.ipynb
python scripts/verify_mlx_notebook.py --device cpu --output outputs/mlx.ipynb
mix precommit
```

Portable verification skips the installation cell because the environment already contains its dependencies, executes the source-download and exercise cells, and verifies that an image was rendered. GitHub Actions runs these checks on Linux and Windows. MLX executes on the macOS job.
