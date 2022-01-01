#include <optional>
#include <random>
#include <stdexcept>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "diffusion_maps/diffusion_maps.hpp"
#include "diffusion_maps/kernel.hpp"
#include "diffusion_maps/matrix.hpp"
#include "diffusion_maps/vector.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

class KernelBase {
public:
  virtual ~KernelBase() = default;
  virtual std::function<double(const diffusion_maps::Vector &,
                               const diffusion_maps::Vector &)>
  translate() const = 0;
};

class GaussianKernel : public KernelBase {
public:
  double gamma;

  GaussianKernel(double gamma) : gamma(gamma) {}
  virtual ~GaussianKernel() override = default;
  virtual std::function<double(const diffusion_maps::Vector &,
                               const diffusion_maps::Vector &)>
  translate() const override {
    return diffusion_maps::kernel::Gaussian(gamma);
  };
};

static py::array_t<double>
_diffusion_maps(const py::array_t<double> data, const std::size_t n_components,
                const KernelBase &kernel, const double diffusion_time,
                const std::optional<std::size_t> rng_seed,
                const double kernel_epsilon, const double eig_solver_tol,
                const unsigned eig_solver_max_iter) {
  const auto info = data.request();
  if (info.ndim != 2) {
    throw std::runtime_error("data must be a 2D array");
  }

  const diffusion_maps::Matrix data_matrix(
      reinterpret_cast<double *>(info.ptr), info.shape[0], info.shape[1],
      info.strides[0] / sizeof(double), info.strides[1] / sizeof(double));

  std::default_random_engine rng(rng_seed ? *rng_seed : std::random_device()());

  diffusion_maps::Matrix *const result =
      new diffusion_maps::Matrix(diffusion_maps::diffusion_maps(
          data_matrix, n_components, kernel.translate(), diffusion_time, rng,
          kernel_epsilon, eig_solver_tol, eig_solver_max_iter));

  return py::array_t<double>({result->n_rows(), result->n_cols()},
                             {result->row_stride() * sizeof(double),
                              result->col_stride() * sizeof(double)},
                             result->data(), py::cast(result));
}

PYBIND11_MODULE(_diffusion_maps, m) {
  m.def("diffusion_maps", &_diffusion_maps);

  py::class_<diffusion_maps::Matrix>(m, "Matrix");

  auto k = m.def_submodule("kernel");
  py::class_<KernelBase>(k, "KernelBase");
  py::class_<GaussianKernel, KernelBase>(k, "Gaussian").def(py::init<double>());
}
