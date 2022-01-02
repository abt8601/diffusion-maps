import importlib
import os
import sys
import timeit

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

diffusion_maps = importlib.import_module('diffusion_maps').diffusion_maps
ref = importlib.import_module('diffusion_maps_ref')


NUMBER = 1
REPEAT = 5


def mean(a: list[float]) -> float:
    return sum(a) / len(a)


def main():
    n_samples = 1000
    t = np.linspace(0, 8 * np.pi, n_samples)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (4 * np.pi) - 1
    helix = np.column_stack((x, y, z))

    globals = {'diffusion_maps': diffusion_maps, 'ref': ref, 'helix': helix}

    comp_time = mean(timeit.repeat(
        '''diffusion_maps(helix, n_components=2, kernel="gaussian", sigma=0.1,
                          diffusion_time=1)''',
        number=NUMBER, repeat=REPEAT, globals=globals))
    print(f'Own implementation: {comp_time}')

    ref_time = mean(timeit.repeat(
        '''ref.diffusion_maps(helix, n_components=2,
                              kernel=ref.Gaussian.from_sigma(0.1),
                              diffusion_time=1)''',
        number=NUMBER, repeat=REPEAT, globals=globals))
    print(f'Reference implementation: {ref_time}')


if __name__ == '__main__':
    main()
