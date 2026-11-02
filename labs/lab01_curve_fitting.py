"""Lab 1: estimate exponential-decay parameters with least squares."""
import numpy as np
from scipy.optimize import least_squares


def fit_decay(time, observations):
    """Return (initial_amplitude, decay_rate) for amplitude * exp(-rate*t)."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(observations, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or t.size < 3:
        raise ValueError("time and observations must be 1D arrays of length >= 3")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(t < 0):
        raise ValueError("inputs must be finite; times must be nonnegative")

    def residuals(parameters):
        amplitude, rate = parameters
        return amplitude * np.exp(-rate * t) - y

    result = least_squares(residuals, x0=[max(y[0], 0.01), 0.4], bounds=(0, np.inf))
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


if __name__ == "__main__":
    rng = np.random.default_rng(14)
    times = np.linspace(0, 6, 40)
    measured = 4.0 * np.exp(-0.65 * times) + rng.normal(0, 0.04, times.size)
    amplitude, rate = fit_decay(times, measured)
    print(f"Estimated amplitude: {amplitude:.3f}; decay rate: {rate:.3f}")
    print("Challenge: change the noise level and see how the fit changes.")
