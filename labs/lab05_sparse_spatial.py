"""Lab 5: solve a sparse linear system and query nearest neighbors."""
import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree


def solve_smooth_chain(forces):
    """Solve a symmetric positive-definite tridiagonal system A*x=forces."""
    b = np.asarray(forces, dtype=float)
    if b.ndim != 1 or b.size < 2 or not np.all(np.isfinite(b)):
        raise ValueError("forces must be a finite 1D array with >= 2 values")
    n = b.size
    a = diags_array([
        -np.ones(n - 1), 3.0 * np.ones(n), -np.ones(n - 1)
    ], offsets=[-1, 0, 1], format="csr", dtype=float)
    return spsolve(a, b)


def find_nearest(points, query):
    """Return (index, distance) of nearest coordinate in points."""
    points = np.asarray(points, dtype=float)
    query = np.asarray(query, dtype=float)
    if (points.ndim != 2 or points.shape[0] == 0 or query.shape != (points.shape[1],)
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(query))):
        raise ValueError("provide finite 2D points and a matching 1D query")
    distance, index = KDTree(points).query(query)
    return int(index), float(distance)


if __name__ == "__main__":
    result = solve_smooth_chain([1, 0, 0, 1])
    print("Sparse system solution:", np.round(result, 3))
    index, distance = find_nearest([[0, 0], [2, 2], [5, 5]], [1.9, 2.2])
    print(f"Nearest point index={index}, distance={distance:.3f}")
    print("Challenge: expand the sparse system to 1,000 unknowns.")
