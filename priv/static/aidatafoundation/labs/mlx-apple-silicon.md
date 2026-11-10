# MLX on Apple Silicon

Use MLX to train a small model and inspect how compilation affects execution on your Mac. Follow the [guided workshop](/workshops/mlx) for setup, progress tracking, and exercises.

## Recommended: mlx.ipynb

[Download the self-contained MLX notebook](/aidatafoundation/notebooks/mlx.ipynb). It includes training and held-out evaluation, saving and reloading weights for inference, compiled execution benchmarks, and plots. Open it in VS Code or Jupyter on an Apple silicon Mac with a native Python kernel. From this repository, install `requirements/workshops/mlx-notebook.txt`; then restart the kernel and Run All.

## Requirements and setup

This track targets Apple silicon Macs running macOS 14 or later with native arm64 Python 3.11–3.13. It uses the installed MLX package and generates small synthetic arrays locally. No model download or API key is needed. Linux and Windows learners can use the other [workshop tracks](/workshops).

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/workshops/mlx.txt
python -m labs.workshop mlx --check
python -m labs.workshop mlx
```

The check tests a real array operation before starting the labs. The default device is Metal GPU when available, otherwise CPU. To choose explicitly:

```bash
python -m labs.workshop mlx --device cpu
python -m labs.workshop mlx --device gpu
```

An explicitly requested GPU must be available; the runner reports an error rather than silently switching it to CPU. If Python reports `x86_64` on an M-series Mac, recreate the virtual environment using native arm64 Python instead of Rosetta.

## Lab 1: autograd and lazy evaluation

Run `python -m labs.workshop mlx --step 1`. Inspect `labs/lab01_mlx_autograd.py`: a small linear model fits the synthetic relation `y = 2.5x - 0.8` using `mx.value_and_grad`.

MLX builds computation graphs lazily. Calling `mx.eval(params)` after each update executes that graph and bounds the chain of pending updates. Reading `.item()` also materializes a scalar; avoid doing this unnecessarily in performance-sensitive inner loops.

Open `outputs/mlx-autograd.json`. The learned slope should approach 2.5, the intercept should approach -0.8, and the final mean squared error should be lower than the initial error. This clean, noiseless example teaches gradients; it does not establish generalization performance on real data.

Change the learning rate or iteration count in the source, rerun, and explain the resulting loss.

## Lab 2: compile and measure

Run `python -m labs.workshop mlx --step 2`. The source in `labs/lab02_mlx_compile.py` compares a normalization operation with its `mx.compile` counterpart. It verifies output agreement, warms up both paths, and calls `mx.eval` inside every timed iteration so pending device work is included.

Open `outputs/mlx-compile.json`. Compare the median milliseconds and observed ratio. These are warm execution measurements; they exclude initial compilation, input creation, and loading. There is no guaranteed speedup: results depend on array shape, device, load, and overhead. A ratio below one means the compiled path was slower in this run. For careful comparisons, repeat runs and alternate the measurement order to check for order effects.

Repeat on CPU and GPU separately and keep each JSON report. Keep the input shape and repeat count identical when comparing devices.

## Troubleshooting

- Missing package: install the requirements with the same interpreter used for execution.
- Backend check failure: try `--device cpu`, check the operating system and native Python architecture, then consult the installation documentation.
- No output window: these labs write JSON files into `outputs/`; they do not open a plotting window.
- A different MLX installation may support other platforms. This workshop's compatibility checks intentionally target its documented Apple silicon environment.

## Official documentation

- [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html)
- [MLX linear regression example](https://ml-explore.github.io/mlx/build/html/examples/linear_regression.html)
- [MLX compilation and timing](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
