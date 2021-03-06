"""Diffusion maps."""

from typing import Optional

import numpy as np

import _diffusion_maps


default_kernel_epsilon = 1e-6
default_eig_solver_tol = 1e-6
default_eig_solver_max_iter = 100000


def diffusion_maps(
        data: np.ndarray, n_components: int, kernel: str, diffusion_time: float,
        *, rng_seed: Optional[int] = None,
        kernel_epsilon: float = default_kernel_epsilon,
        eig_solver_tol: float = default_eig_solver_tol,
        eig_solver_max_iter: int = default_eig_solver_max_iter,
        **kwargs) -> np.ndarray:
    """Diffusion maps.

    Parameters
    ----------
    data : np.ndarray
        The data matrix where each row is a data point.
    n_components : int
        The dimension of the projected subspace.
    kernel : {'gaussian'}
        The kernel function.
    diffusion_time : float
        The diffusion time.
    rng_seed : int, optional
        The seed for the random number generator.
    kernel_epsilon : float, default 1e-6
        The value below which the output of the kernel would be treated as zero.
    eig_solver_tol : float, default 1e-6
        The tolerance of the eigendecomposition solver.
    eig_solver_max_iter : int, default 100000
        The maximum number of iterations of the eigendecomposition solver.
    **kwargs : dict, optional
        The keyword arguments of the kernel function.

    Returns
    -------
    np.ndarray
        The lower-dimensional embedding of the data in the diffusion space.

    Raises
    ------
    ValueError
        If the data matrix is not a two-dimensional array.
    ValueError
        If `n_components` is negative or greater than the number of data points
        minus 1.
    ValueError
        If the kernel is not supported.
    ValueError
        If the kernel parameters are not valid.
    ValueError
        If the diffusion time is negative.

    Kernels
    -------
    The following kernels are supported:

    - 'gaussian'
      The kernel parameter (`gamma` or `sigma`) can be specified as a keyword
      argument. If both are not specified, `gamma` defaults to
      1 / ``n_features``. If both are specified, raise a `ValueError`.
    """

    # Check the dimensions.
    if data.ndim != 2:
        raise ValueError('data must be a 2D array')

    # Check kernel.
    if kernel == 'gaussian':
        if 'gamma' in kwargs:
            if 'sigma' in kwargs:
                raise ValueError('cannot specify both gamma and sigma')
            else:
                gamma = kwargs['gamma']
        elif 'sigma' in kwargs:
            sigma = kwargs['sigma']
            gamma = 1 / (2 * sigma*sigma)
        else:
            # Use default value.
            n_features = data.shape[1]
            gamma = 1 / n_features

        kernel_obj = _diffusion_maps.kernel.Gaussian(gamma)
    else:
        raise ValueError(f'unknown kernel: {kernel}')

    return _diffusion_maps.diffusion_maps(data, n_components, kernel_obj,
                                          diffusion_time, rng_seed, kernel_epsilon,
                                          eig_solver_tol, eig_solver_max_iter)
