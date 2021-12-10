#include <stdexcept>

#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "diffusion_maps/diffusion_maps.hpp"
#include "diffusion_maps/kernel.hpp"
#include "diffusion_maps/matrix.hpp"
#include "diffusion_maps/vector.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

static py::array_t<double> _diffusion_maps(
    const py::array_t<double> data, const std::size_t n_components,
    const std::function<double(const diffusion_maps::Vector &,
                               const diffusion_maps::Vector &)> &kernel,
    const double diffusion_time, const double kernel_epsilon,
    const double eig_solver_tol, const unsigned eig_solver_max_iter,
    const unsigned eig_solver_max_restarts) {
  const auto info = data.request();
  if (info.ndim != 2) {
    throw std::runtime_error("data must be a 2D array");
  }

  const diffusion_maps::Matrix data_matrix(
      reinterpret_cast<double *>(info.ptr), info.shape[0], info.shape[1],
      info.strides[0] / sizeof(double), info.strides[1] / sizeof(double));

  auto [dm_buffer, dm] = diffusion_maps::diffusion_maps(
      data_matrix, n_components, kernel, diffusion_time, kernel_epsilon,
      eig_solver_tol, eig_solver_max_iter, eig_solver_max_restarts);

  double *const dm_data = dm_buffer.release();
  return py::array_t<double>(
      {dm.n_rows(), dm.n_cols()},
      {dm.row_stride() * sizeof(double), dm.col_stride() * sizeof(double)},
      dm_data, py::capsule(dm_data, [](void *ptr) {
        delete[] reinterpret_cast<double *>(ptr);
      }));
}

PYBIND11_MODULE(_diffusion_maps, m) {
  m.def("diffusion_maps", &_diffusion_maps);

  py::class_<diffusion_maps::Vector>(m, "Vector");

  auto k = m.def_submodule("kernel");
  py::class_<diffusion_maps::kernel::Gaussian>(k, "Gaussian")
      .def(py::init<double>())
      .def("__call__", &diffusion_maps::kernel::Gaussian::operator());
}
