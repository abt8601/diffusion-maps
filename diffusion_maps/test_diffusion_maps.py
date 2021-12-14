from .diffusion_maps import diffusion_maps

import numpy as np


def test_diffusion_maps_helix():
    """Tests diffusion maps on a helix."""

    # Generate data.

    n_samples = 1000
    t = np.linspace(0, 8 * np.pi, n_samples)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (4 * np.pi) - 1
    helix = np.column_stack((x, y, z))

    # Compute diffusion maps.

    result = diffusion_maps(helix, n_components=1,
                            kernel='gaussian', sigma=0.1, diffusion_time=1)

    # Check the dimensions.

    assert result.shape == (n_samples, 1)

    # Check that result is monotonic.

    diff = np.diff(result)
    assert np.all(diff >= 0) or np.all(diff <= 0)
