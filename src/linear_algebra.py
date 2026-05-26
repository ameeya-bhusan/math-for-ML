"""Linear algebra building blocks for machine learning.

Conventions
-----------
* Vectors are 1-D ``numpy.ndarray`` objects.
* Matrices are 2-D ``numpy.ndarray`` objects with shape ``(rows, cols)``.
* Functions never mutate their inputs.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "dot",
    "matmul",
    "transpose",
    "vector_norm",
    "normalize",
    "cosine_similarity",
    "is_symmetric",
    "power_iteration",
    "eigen_decomposition",
]


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Return the dot product ``a . b = sum_i a_i b_i``.

    The dot product is the workhorse of ML: it measures how much two vectors
    point in the same direction and underlies every linear layer, kernel and
    similarity score.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sum(a * b))


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices using the triple-loop definition.

    ``C[i, j] = sum_k A[i, k] * B[k, j]``.

    This is deliberately the textbook ``O(n^3)`` formulation so that the index
    bookkeeping is visible. For real work use ``A @ B``; the result here is
    validated against NumPy in the test-suite.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("matmul expects 2-D arrays")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"inner dimensions disagree: {A.shape} x {B.shape}")

    n, m = A.shape
    _, p = B.shape
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            acc = 0.0
            for k in range(m):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C


def transpose(A: np.ndarray) -> np.ndarray:
    """Return the transpose ``A^T`` where ``A^T[i, j] = A[j, i]``."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("transpose expects a 2-D array")
    return A.T.copy()


def vector_norm(v: np.ndarray, p: float = 2.0) -> float:
    """Return the L-p norm of a vector.

    * ``p = 1``: sum of absolute values (Manhattan / Lasso penalty).
    * ``p = 2``: Euclidean length (Ridge penalty, default).
    * ``p = inf``: largest absolute component.
    """
    v = np.asarray(v, dtype=float)
    if np.isinf(p):
        return float(np.max(np.abs(v)))
    if p <= 0:
        raise ValueError("p must be positive (or numpy.inf)")
    return float(np.sum(np.abs(v) ** p) ** (1.0 / p))


def normalize(v: np.ndarray) -> np.ndarray:
    """Scale a vector to unit L2 length. Returns a copy; raises on the zero vector."""
    v = np.asarray(v, dtype=float)
    length = vector_norm(v, 2.0)
    if length == 0.0:
        raise ValueError("cannot normalize the zero vector")
    return v / length


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors, in ``[-1, 1]``.

    ``cos(theta) = (a . b) / (||a|| ||b||)``. Unlike the raw dot product this is
    scale invariant, which is why it is the default similarity for text
    embeddings and many recommender systems.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = vector_norm(a), vector_norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot(a, b) / (na * nb)


def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Return ``True`` if ``A`` equals its transpose within ``tol``."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    return bool(np.allclose(A, A.T, atol=tol))


def power_iteration(
    A: np.ndarray,
    num_iters: int = 1000,
    tol: float = 1e-12,
    seed: int | None = 0,
) -> tuple[float, np.ndarray]:
    """Estimate the dominant eigenvalue/eigenvector pair by power iteration.

    Repeatedly applying ``A`` to a random vector amplifies the component along
    the eigenvector with the largest-magnitude eigenvalue. After normalising
    each step the iterate converges to that eigenvector, and the Rayleigh
    quotient ``v^T A v`` gives the eigenvalue.

    Returns
    -------
    (eigenvalue, eigenvector)
        ``eigenvector`` is unit length.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("power_iteration expects a square matrix")

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0])
    v = normalize(v)

    eigenvalue = 0.0
    for _ in range(num_iters):
        Av = A @ v
        new_v = normalize(Av)
        # Rayleigh quotient is the best eigenvalue estimate for the current v.
        new_eigenvalue = float(new_v @ (A @ new_v))
        # Fix the sign so the vector direction is stable across iterations.
        if dot(new_v, v) < 0:
            new_v = -new_v
        if abs(new_eigenvalue - eigenvalue) < tol:
            v, eigenvalue = new_v, new_eigenvalue
            break
        v, eigenvalue = new_v, new_eigenvalue
    return eigenvalue, v


def eigen_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full eigendecomposition of a symmetric matrix, eigenvalues descending.

    For symmetric ``A`` the eigenvalues are real and the eigenvectors are
    orthogonal, which is exactly the structure PCA relies on (the covariance
    matrix is symmetric). We use ``numpy.linalg.eigh`` for numerical stability
    and then reorder the results so the leading component comes first.
    """
    A = np.asarray(A, dtype=float)
    if not is_symmetric(A):
        raise ValueError("eigen_decomposition expects a symmetric matrix")
    values, vectors = np.linalg.eigh(A)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]