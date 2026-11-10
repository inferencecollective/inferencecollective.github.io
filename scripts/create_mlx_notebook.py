"""Build the standalone MLX notebook without requiring Jupyter to generate it."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
cells = []


def markdown(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=dedent(text).strip() + '\n'))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=dedent(text).strip() + '\n'))


markdown('''
# MLX: from training to measured inference
**Inference Collective · Apple silicon workshop · 45–60 minutes**

Build a small regression model, use its learned weights on new inputs, then measure compiled inference on your CPU or Metal GPU. Every example is self-contained: no repository imports, API keys, external datasets, or model downloads.

**You will leave with:** learning curves, held-out predictions, latency measurements, a reloadable weight file, and a JSON report.

## Before you start
Use an Apple silicon Mac, macOS 14+, and native arm64 Python 3.11–3.13. In the cloned repository, install `requirements/workshops/mlx-notebook.txt` in your virtual environment. To use this notebook on its own, install:

```bash
python -m pip install "mlx>=0.32,<0.33" "numpy>=2.1,<3" "matplotlib>=3.10,<4" "ipykernel>=6.29,<8"
```

Open `mlx.ipynb` in VS Code or your existing Jupyter application and select that Python environment as the kernel. **Restart the kernel and Run All** to reproduce the workshop. A remote kernel uses the remote machine's hardware, not your browser's Mac. Shared editing remains available through the companion Livebook workshop.
''')
code('''
import json
import os
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

if platform.system() != "Darwin" or platform.machine() != "arm64":
    raise RuntimeError("Use native arm64 Python on an Apple silicon Mac for this workshop.")
if int(platform.mac_ver()[0].split(".")[0]) < 14 or not (3, 11) <= sys.version_info[:2] <= (3, 13):
    raise RuntimeError("This workshop requires macOS 14+ and Python 3.11–3.13.")

import mlx.core as mx
import numpy as np
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from IPython.display import display

# Change these settings, then restart and Run All.
requested_device = os.environ.get("MLX_WORKSHOP_DEVICE", "auto")  # auto, cpu, or gpu
learning_rate = 0.08
epochs = 150
seed = 42
batch_sizes = [1, 64, 4096]
repeats = 30
output_dir = Path("outputs/mlx-notebook")

if requested_device not in {"auto", "cpu", "gpu"}:
    raise ValueError("Choose auto, cpu, or gpu for requested_device.")
metal_available = mx.metal.is_available()
if requested_device == "gpu" and not metal_available:
    raise RuntimeError("Metal is unavailable. Choose cpu and rerun.")
device = "gpu" if requested_device == "gpu" or (requested_device == "auto" and metal_available) else "cpu"
mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
mx.eval(mx.array([1.0]) + 1)  # Test real execution, not just import.
mx.random.seed(seed)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Python {sys.version.split()[0]} · MLX device: {device} · seed: {seed}")
''')
markdown('''
## 1. Make a reproducible dataset

Our synthetic process is `y = 2.5x - 0.8 + noise`. Shuffle before separating training and test examples. Fit only the training split; keep test examples for the final evaluation. This small example teaches mechanics, not performance on real-world data.
''')
code('''
x = mx.linspace(-2, 2, 512)
y = 2.5 * x - 0.8 + 0.1 * mx.random.normal(x.shape)
order = mx.random.permutation(x.shape[0])
train_indices, test_indices = order[:400], order[400:]
x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indices], y[test_indices]
mx.eval(x_train, y_train, x_test, y_test)
print(f"Train: {x_train.size} examples · Test: {x_test.size} examples")
''')
markdown('''
## 2. Train with automatic differentiation

The model has two parameters: a slope and an intercept. `mx.value_and_grad` gives the loss and its gradient. MLX is lazy: `mx.eval(weights)` executes each update and prevents an ever-growing chain of pending computation. Recording a loss with `.item()` synchronizes too; this is an educational training loop, not a training-speed benchmark.
''')
code('''
def infer(weights, values):
    return weights[0] * values + weights[1]


def mse(weights, values, targets):
    return mx.mean(mx.square(infer(weights, values) - targets))


weights = mx.array([0.0, 0.0])
loss_and_grad = mx.value_and_grad(mse)
loss_history = []
initial_loss = mse(weights, x_train, y_train).item()
for epoch in range(epochs):
    loss, gradient = loss_and_grad(weights, x_train, y_train)
    weights = weights - learning_rate * gradient
    mx.eval(weights)
    loss_history.append(loss.item())
train_mse = mse(weights, x_train, y_train).item()
test_mse = mse(weights, x_test, y_test).item()
print(f"Slope: {weights[0].item():.4f} · Intercept: {weights[1].item():.4f}")
print(f"Training MSE: {initial_loss:.4f} → {train_mse:.4f} · Held-out MSE: {test_mse:.4f}")
assert train_mse < initial_loss and test_mse < 0.05, "Inspect the settings and learning curve."
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
axes[0].semilogy(loss_history, color="#059669")
axes[0].set(title="Training converges", xlabel="Update", ylabel="Mean squared error (log)")
axes[1].scatter(np.array(x_test), np.array(y_test), s=14, alpha=0.6, label="Held-out examples")
axes[1].plot(np.array(x), np.array(infer(weights, x)), color="#059669", label="Learned model")
axes[1].set(title=f"Held-out MSE: {test_mse:.4f}", xlabel="Input x", ylabel="Target / prediction")
axes[1].legend()
fig.savefig(output_dir / "learning-and-predictions.png", dpi=150)
plt.show()
''')
markdown('''
## 3. Use the trained model: this is inference

Training changed the parameters. Inference holds them fixed and computes predictions for new inputs. Save and reload the parameters before predicting to check that the model artifact is sufficient to reproduce the output. Keep the feature preparation and model formula with the weights in a real deployment.
''')
code('''
mx.savez(str(output_dir / "linear-model.npz"), weights=weights)
loaded_weights = mx.load(str(output_dir / "linear-model.npz"))["weights"]
new_inputs = mx.array([-1.5, 0.0, 1.5])
predictions = infer(loaded_weights, new_inputs)
mx.eval(predictions)
assert mx.allclose(predictions, infer(weights, new_inputs)).item()
for value, prediction in zip(new_inputs.tolist(), predictions.tolist()):
    print(f"x={value:5.1f} → prediction={prediction:7.3f}")
''')
markdown('''
## 4. Measure before optimizing

Compare ordinary MLX execution with `mx.compile(infer)` on the **same device, inputs, and weights**. A first call can include compilation and initialization; warm execution is measured separately. Each timed call includes `mx.eval` so it measures completed work instead of graph construction. Measurement order alternates to reduce order bias.

Small batches can be dominated by dispatch overhead. A GPU or compilation can be slower. These numbers describe this tiny regression workload, not LLM token generation or general MLX performance. P95 is an estimate from a small sample; repeat runs under comparable load.
''')
code('''
compiled_infer = mx.compile(infer)
benchmark_rows = []
for batch_size in batch_sizes:
    values = mx.random.normal((batch_size,))
    mx.eval(values, loaded_weights)
    start = perf_counter()
    compiled_output = compiled_infer(loaded_weights, values)
    mx.eval(compiled_output)
    first_call_ms = (perf_counter() - start) * 1000
    reference = infer(loaded_weights, values)
    assert mx.allclose(compiled_output, reference, rtol=1e-4, atol=1e-4).item()
    for _ in range(5):
        mx.eval(infer(loaded_weights, values))
        mx.eval(compiled_infer(loaded_weights, values))
    samples = {"uncompiled": [], "compiled": []}
    functions = {"uncompiled": infer, "compiled": compiled_infer}
    for iteration in range(repeats):
        names = ["uncompiled", "compiled"] if iteration % 2 == 0 else ["compiled", "uncompiled"]
        for name in names:
            start = perf_counter()
            mx.eval(functions[name](loaded_weights, values))
            samples[name].append((perf_counter() - start) * 1000)
    regular_ms, optimized_ms = (median(samples[name]) for name in ["uncompiled", "compiled"])
    benchmark_rows.append({
        "batch_size": batch_size, "device": device, "outputs_match": True,
        "compiled_first_call_ms": first_call_ms,
        "uncompiled_median_ms": regular_ms, "compiled_median_ms": optimized_ms,
        "compiled_p95_ms": float(np.percentile(samples["compiled"], 95)),
        "observed_ratio": regular_ms / optimized_ms,
        "compiled_examples_per_second": batch_size * 1000 / optimized_ms,
    })
    print(f"Batch {batch_size:4d} | ordinary {regular_ms:.3f} ms | compiled {optimized_ms:.3f} ms | ratio {regular_ms / optimized_ms:.2f}×")
''')
code('''
positions = np.arange(len(benchmark_rows))
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
ax.bar(positions - 0.18, [row["uncompiled_median_ms"] for row in benchmark_rows], width=0.36, label="Ordinary MLX", color="#64748b")
ax.bar(positions + 0.18, [row["compiled_median_ms"] for row in benchmark_rows], width=0.36, label="Compiled MLX", color="#059669")
ax.set_xticks(positions, [str(row["batch_size"]) for row in benchmark_rows])
ax.set(title=f"Warm inference latency · {device.upper()} · lower is faster", xlabel="Examples per batch", ylabel="Median milliseconds per batch")
ax.legend()
fig.savefig(output_dir / "inference-latency.png", dpi=150)
plt.show()
''')
markdown('''
## 5. Keep a reproducible report

The JSON records settings, quality, and timing results. Latency is per batch; throughput is examples per second. Neither is a token-generation metric. Compare CPU and GPU by changing `requested_device` and restarting the kernel; device-specific filenames keep both reports.
''')
code('''
report = {
    "device": device, "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
    "train_examples": int(x_train.size), "test_examples": int(x_test.size),
    "train_mse": train_mse, "test_mse": test_mse,
    "slope": weights[0].item(), "intercept": weights[1].item(),
    "repeats": repeats, "warmups": 5, "benchmark": benchmark_rows,
    "timing_scope": "Warm execution includes evaluation; first-call timings are recorded separately. No guaranteed speedup.",
}
report_path = output_dir / f"report-{device}.json"
report_path.write_text(json.dumps(report, indent=2) + "\\n")
print(f"Saved {report_path}")
print("Also saved: linear-model.npz, learning-and-predictions.png, inference-latency.png")
''')
markdown('''
## Workshop challenges

1. Change the learning rate. Explain convergence using the loss curve and held-out MSE.
2. Change the batch sizes. Explain why examples/second and latency answer different questions.
3. Run once on CPU and once on GPU. Keep both JSON reports and repeat the comparison before drawing conclusions.
4. Add a nonlinear term to the data. Does the linear model still fit? Explain model error versus serving performance.

### Troubleshooting

- **Missing module:** install the requirements using the interpreter selected by your notebook kernel.
- **No Metal device:** use `requested_device = "cpu"` on a supported Mac; the initial MLX import may still fail in restricted or headless environments. Consult MLX installation guidance.
- **Different results after editing cells:** restart the kernel and Run All. The notebook is designed for top-to-bottom execution.
- **Slow first run:** package import and compilation are setup costs. Compare warm measurements separately.

### Official references

- [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html)
- [Lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [Compilation and benchmarking](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [VS Code notebooks and kernels](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
''')

for index, cell in enumerate(cells):
    cell['id'] = f'mlx-{index:02d}'
notebook = dict(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3 (MLX environment)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12', 'file_extension': '.py', 'mimetype': 'text/x-python'},
}, nbformat=4, nbformat_minor=5)
if __name__ == '__main__':
    destination = ROOT / 'priv/static/aidatafoundation/notebooks/mlx.ipynb'
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
