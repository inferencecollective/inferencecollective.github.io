"""Lab 2: simulate Newton's law of cooling with an ODE solver."""
import numpy as np
from scipy.integrate import solve_ivp


def simulate_cooling(initial_temp=90, ambient_temp=20, rate=0.2, minutes=30):
    """Return (times, temperatures) for exponential cooling."""
    if not np.all(np.isfinite([initial_temp, ambient_temp, rate, minutes])):
        raise ValueError("all inputs must be finite")
    if rate <= 0 or minutes <= 0:
        raise ValueError("rate and minutes must be positive")
    times = np.linspace(0, minutes, 61)

    def derivative(_t, state):
        return [-rate * (state[0] - ambient_temp)]

    solution = solve_ivp(derivative, (0, minutes), [initial_temp],
                         t_eval=times, rtol=1e-9, atol=1e-11)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y[0]


if __name__ == "__main__":
    times, temperatures = simulate_cooling()
    for step in (0, 10, 30, 60):
        print(f"Minute {times[step]:4.1f}: {temperatures[step]:.2f}°C")
    print("Challenge: change ambient temperature and the cooling rate.")
