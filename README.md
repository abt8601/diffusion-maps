# diffusion-maps

A C++/Python library for diffusion maps,
a non-linear dimensionality reduction technique.

This is the final project of the course
[Numerical Software Development](https://yyc.solvcon.net/en/latest/nsd/).
You can find my project proposal in the link below.
Note that the API is different from that described in the proposal.

[Project Proposal](https://github.com/yungyuc/nsdhw_21au/blob/master/proposal/abt8601/README.rst)

## Building

### Requirements

To build the C++ library, you need:
- A version of GCC or Clang with C++17 and OpenMP support.
- Make

To build the Python module, you need in addition:
- [pybind11](https://pybind11.readthedocs.io/en/stable/)
- NumPy (only at runtime)

If you want to run the test suite, you also need:
- [Criterion](https://criterion.readthedocs.io/en/latest/intro.html)
  for the C++ tests
- [pytest](https://docs.pytest.org/en/latest/) for the Python tests

### Profiles

The build script defines multiple profiles with different compilation settings.
- `DEBUG`
  - With debug info, Address Sanitizer, and Undefined Behavior Sanitizer.
  - No optimisations.
  - Used in development.
- `TEST`
  - With debug info, Address Sanitizer, and Undefined Behavior Sanitizer.
  - Basic optimisations.
  - Used in some automated tests.
- `TEST_PAR`
  - With debug info and Thread Sanitizer.
  - Basic optimisations.
  - Originally intended to be used in automated tests
    to test the correctness of the parallel algorithms.
    However, since
    [libgomp does not support Thread Sanitizer](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55561),
    no such tests are enabled and this profile is currently unused.
    Binaries compiled with this profile may not work as intended.
- `RELEASE`
  - No debug info or sanitizers.
  - Full optimisations.
  - Used to create release binaries and in some automated tests.

### Build Commands

To build everything, including the C++ library and the Python module:
```shell
$ make PROFILE=<profile>
```
If `PROFILE=<profile>` is omitted, then the profile defaults to `DEBUG`.
The compiled C++ library is located at `build/<profile>/libdiffusion_maps.a`.

Or, to build just the C++ library:
```shell
$ make lib PROFILE=<profile>
```

Or, to build just the Python module:
```shell
$ make pymod PROFILE=<profile>
```

To run the test suite (optional):
```shell
$ make test
```

Or, to just run the C++ or Python tests,
replace `test` in the command with `cpptest` or `pytest`.

## Basic Usage

For more information about the API,
please refer to the documentation that comes with the source code.

### C++

```cpp
#include <random> // std::default_random_engine

#include "diffusion_maps/diffusion_maps.hpp" // diffusion_maps::diffusion_maps
#include "diffusion_maps/kernel.hpp" // diffusion_maps::kernel::Gaussian
#include "diffusion_maps/matrix.hpp" // diffusion_maps::Matrix

// Each row of data is a data point.
const diffusion_maps::Matrix data(n_samples, n_features);
// kernel can be any std::function<double(const diffusion_maps::Vector &,
//                                        const diffusion_maps::Vector &)>.
// Gaussian kernel can also be defined with the σ parameter, like so:
//   const auto kernel = diffusion_maps::kernel::Gaussian::with_sigma(sigma);
const diffusion_maps::kernel::Gaussian kernel(gamma);
// rng can be any UniformRandomBitGenerator. It must be reasonably good,
// otherwise the algorithm may not converge or produce erroneous results.
std::default_random_engine rng;

// result's dimension is n_samples × n_components, or it might have fewer
// columns than n_components if the computation does not converge. Each row of
// result is a data point.
diffusion_maps::Matrix result = diffusion_maps::diffusion_maps(
    data, n_components, kernel, diffusion_time, rng);
```

Statically link with the compiled C++ library `libdiffusion_maps.a`.

Please refer to [`tests/test_diffusion_maps.cpp`](tests/test_diffusion_maps.cpp)
for a complete example.

### Python

```python
import numpy as np

import diffusion_maps

# Each row of data is a data point.
data = np.empty((n_samples, n_features))

# result's dimension is n_samples × n_components, or it might have fewer columns
# than n_components if the computation does not converge. Each row of result is
# a data point.
# It is also possible to specify the σ parameter of the Gaussian kernel by the
# sigma=sigma keyword argument.
result = diffusion_maps.diffusion_maps(data, n_components, kernel='gaussian',
                                       gamma=gamma, diffusion_time=t)
```

When running the program,
the top-level directory of this project must be in `PYTHONPATH`.
Or if you would like to install the library,
copy the `diffusion_maps` directory
and the compiled binary module (name starting with `_diffusion_maps`)
to the installation destination.

Please refer to [`tests/test_diffusion_maps.py`](tests/test_diffusion_maps.py)
or [`examples/helix.ipynb`](examples/helix.ipynb) for complete examples.

## Directory Structure

- `include`: Header files for the C++ library
- `src`: Source code for the C++ library
- `pybind`: Source code for the compiled Python module, which uses pybind11
- `diffusion_maps`: The Python module,
  which is basically a wrapper around the compiled binary module
- `tests`: Automated tests for both the C++ library and the Python module
- `examples`: Some examples for using the library
- `diffusion_maps_ref.py`: Reference implementation written purely in Python,
  used in verification.

## Acknowledgement

Thanks to [Bo-Yu Cheng](https://github.com/Nemo1999)
and Professor [Yung-Yu Chen](https://yyc.solvcon.net)
for discussions that lead to
[a solution to a major problem in my eigendecomposition solver](https://github.com/abt8601/diffusion-maps/commit/4eba5dd).

## License

This project is licensed under [The Unlicense](LICENSE).
