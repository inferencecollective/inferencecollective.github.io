"""Fit y = 2.5x - 0.8 using MLX autograd and explicit lazy evaluation."""
import json
from pathlib import Path

import mlx.core as mx
from labs.mlx_runtime import configure_device

OUTPUTS = Path(__file__).resolve().parent.parent / 'outputs'


def run(output_dir=OUTPUTS, steps=200, learning_rate=0.1):
    device = configure_device()
    x = mx.linspace(-2, 2, 256)
    y = 2.5 * x - 0.8
    params = mx.array([0.0, 0.0])

    def loss(weights):
        return mx.mean(mx.square(weights[0] * x + weights[1] - y))

    value_and_grad = mx.value_and_grad(loss)
    initial_loss = loss(params).item()
    for _ in range(steps):
        _, gradient = value_and_grad(params)
        params = params - learning_rate * gradient
        # MLX builds a lazy graph. Evaluate each update to bound graph growth.
        mx.eval(params)
    report = dict(device=device, steps=steps, learning_rate=learning_rate,
                  initial_loss=initial_loss, final_loss=loss(params).item(),
                  slope=params[0].item(), intercept=params[1].item(),
                  target_slope=2.5, target_intercept=-0.8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'mlx-autograd.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
