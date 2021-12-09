import numpy as np

import _diffusion_maps


def diffusion_maps(data: np.ndarray, n_components: int, kernel: str,
                   diffusion_time: float, *, kernel_epsilon: float = 1e-6,
                   eig_solver_tol: float = 1e-6,
                   eig_solver_max_iter: int = 1000,
                   eig_solver_max_restarts: int = 10, **kwargs) -> np.ndarray:
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
                                          diffusion_time, kernel_epsilon,
                                          eig_solver_tol, eig_solver_max_iter,
                                          eig_solver_max_restarts)


def dummy():
    return _diffusion_maps.dummy()
