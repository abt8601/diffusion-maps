#include "pybind11/pybind11.h"

#include "dummy.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_diffusion_maps, m) { m.def("dummy", &dummy); }
