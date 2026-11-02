"""Lab 3: compare two independent groups with Welch's t-test."""
import numpy as np
from scipy.stats import ttest_ind


def compare_groups(control, treatment):
    """Return descriptive statistics and a Welch t-test result.

    This is a teaching example, not a causal inference or production A/B framework.
    """
    a, b = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or min(a.size, b.size) < 2:
        raise ValueError("both samples must be 1D with >= 2 observations")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("samples must contain finite values")
    if np.var(a, ddof=1) == 0 and np.var(b, ddof=1) == 0:
        raise ValueError("at least one group must have variation")
    test = ttest_ind(b, a, equal_var=False)
    ci = test.confidence_interval(confidence_level=0.95)
    return {
        "control_mean": float(np.mean(a)),
        "treatment_mean": float(np.mean(b)),
        "difference": float(np.mean(b) - np.mean(a)),
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "ci_95": (float(ci.low), float(ci.high)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(25)
    control = rng.normal(50, 8, 120)
    treatment = rng.normal(53, 9, 110)
    report = compare_groups(control, treatment)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("Challenge: vary sample size; do not interpret a p-value as P(hypothesis true).")
