"""Reference implementation of the diffusion maps algorithm."""

from typing import Callable, List

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg


def _compute_kernel_matrix(data: np.ndarray,
                           kernel: Callable[[np.ndarray, np.ndarray], float],
                           epsilon: float) -> sparse.csr_matrix:
    n_samples: int = data.shape[0]

    kernel_data: List[float] = []
    kernel_row_ind: List[int] = []
    kernel_col_ind: List[int] = []

    def kernel_append(row_ind: int, col_ind: int, value: float) -> None:
        kernel_data.append(value)
        kernel_row_ind.append(row_ind)
        kernel_col_ind.append(col_ind)

    for i in range(n_samples):
        for j in range(i + 1):
            value = kernel(data[i], data[j])

            if value >= epsilon:
                kernel_append(i, j, value)
                if i != j:
                    kernel_append(j, i, value)

    return sparse.coo_matrix((kernel_data, (kernel_row_ind, kernel_col_ind)),
                             shape=(n_samples, n_samples)).tocsr()


def diffusion_maps(
        data: np.ndarray, n_components: int,
        kernel: Callable[[np.ndarray, np.ndarray], float],
        diffusion_time: float, *, kernel_epsilon: float = 1e-6,
        eig_solver_tol: float = 1e-6) -> np.ndarray:
    # Step 1: Compute the kernel matrix.

    kernel_matrix = _compute_kernel_matrix(data, kernel, kernel_epsilon)

    # Step 2: Compute the "symmetrised" kernel matrix
    # P' = D^(-1/2) * K * D^(-1/2) where D is the diagonal matrix that
    # normalises the rows of the kernel matrix.

    inv_sqrt_d = np.reshape(np.array(kernel_matrix.sum(axis=1)), -1) ** -0.5
    inv_sqrt_d_mat = sparse.diags(inv_sqrt_d)
    pprime = inv_sqrt_d_mat @ kernel_matrix @ inv_sqrt_d_mat

    # Step 3: Compute the eigenvectors of P'.

    w, v = slinalg.eigsh(pprime, k=n_components+1, sigma=1,
                         which='LM', tol=eig_solver_tol)

    # Step 4: Compute the diffusion map.

    diffusion_maps = inv_sqrt_d[:, np.newaxis] * np.flip(
        v[:, :-1] * w[:-1] ** diffusion_time, axis=1)

    return diffusion_maps


class Gaussian:
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    @classmethod
    def from_sigma(cls, sigma: float) -> 'Gaussian':
        return cls(1 / (2 * sigma ** 2))

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.exp(-self.gamma * np.linalg.norm(x - y) ** 2)
