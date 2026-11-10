"""Compare evaluated MLX operations with and without compilation on one device."""
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx
from labs.mlx_runtime import configure_device

OUTPUTS = Path(__file__).resolve().parent.parent / 'outputs'


def run(output_dir=OUTPUTS, repeats=20):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    device = configure_device()
    mx.random.seed(42)
    x = mx.random.normal((256, 1024))
    mx.eval(x)

    def normalize(values):
        centered = values - mx.mean(values, axis=-1, keepdims=True)
        return centered * mx.rsqrt(mx.mean(mx.square(centered), axis=-1, keepdims=True) + 1e-5)

    compiled = mx.compile(normalize)
    correct = mx.allclose(normalize(x), compiled(x), rtol=1e-4, atol=1e-4).item()
    if not correct:
        raise RuntimeError('Compiled output differs from the reference.')
    # Warm both paths before timing; mx.eval waits for each output to finish.
    for _ in range(5):
        mx.eval(normalize(x))
        mx.eval(compiled(x))
    timings = {'uncompiled': [], 'compiled': []}
    for _ in range(repeats):
        for name, function in [('uncompiled', normalize), ('compiled', compiled)]:
            start = perf_counter()
            mx.eval(function(x))
            timings[name].append((perf_counter() - start) * 1000)
    regular_ms, compiled_ms = (median(timings[name]) for name in timings)
    report = dict(device=device, shape=list(x.shape), repeats=repeats,
                  warmups=5, outputs_match=bool(correct),
                  uncompiled_median_ms=regular_ms, compiled_median_ms=compiled_ms,
                  observed_ratio=regular_ms / compiled_ms,
                  note='Warm execution only; excludes compile time. Results depend on hardware, load, and shape; compilation may be slower.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'mlx-compile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
