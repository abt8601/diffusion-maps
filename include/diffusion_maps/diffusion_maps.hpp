/// \file
///
/// \brief Diffusion maps algorithm.

#ifndef DIFFUSION_MAPS_DIFFUSION_MAPS_HPP
#define DIFFUSION_MAPS_DIFFUSION_MAPS_HPP

#include <functional>
#include <memory>
#include <random>

#include "diffusion_maps/matrix.hpp"
#include "diffusion_maps/vector.hpp"

/// Namespace for everything related to diffusion maps.
namespace diffusion_maps {

namespace internal {

Matrix diffusion_maps(
    const Matrix &data, std::size_t n_components,
    const std::function<double(const Vector &, const Vector &)> &kernel,
    double diffusion_time, double kernel_epsilon, double eig_solver_tol,
    unsigned eig_solver_max_iter, const std::function<double()> &rng);

}

/// Default kernel epsilon for the diffusion_maps() function.
constexpr double DEFAULT_KERNEL_EPSILON = 1e-6;

/// \brief Default tolerance of the eigendecomposition solver for the
///        diffusion_maps() function.
constexpr double DEFAULT_EIG_SOLVER_TOL = 1e-6;

/// \brief Default maximum number of iterations of the eigendecomposition solver
///        for the diffusion_maps() function.
constexpr unsigned DEFAULT_EIG_SOLVER_MAX_ITER = 100000;

/// \brief Diffusion maps.
///
/// \tparam R The type of the random number generator.
/// \param[in] data The data matrix where each row is a data point.
/// \param[in] n_components The dimension of the projected subspace.
/// \param[in] kernel The kernel function.
/// \param[in] diffusion_time The diffusion time.
/// \param[in,out] rng The random number generator.
/// \param[in] kernel_epsilon The value below which the output of the kernel
///                           would be treated as zero.
/// \param[in] eig_solver_tol The tolerance of the eigendecomposition solver.
/// \param[in] eig_solver_max_iter The maximum number of iterations of the
///                                eigendecomposition solver.
/// \return The lower-dimensional embedding of the data in the diffusion space.
/// \exception std::invalid_argument If \p n_components is greater than the
///                                  number of data points minus 1.
/// \exception std::invalid_argument If \p diffusion_time is negative.
template <typename R>
Matrix diffusion_maps(
    const Matrix &data, std::size_t n_components,
    const std::function<double(const Vector &, const Vector &)> &kernel,
    double diffusion_time, R &rng,
    double kernel_epsilon = DEFAULT_KERNEL_EPSILON,
    double eig_solver_tol = DEFAULT_EIG_SOLVER_TOL,
    unsigned eig_solver_max_iter = DEFAULT_EIG_SOLVER_MAX_ITER) {
  std::normal_distribution dist;
  return internal::diffusion_maps(data, n_components, kernel, diffusion_time,
                                  kernel_epsilon, eig_solver_tol,
                                  eig_solver_max_iter,
                                  [&rng, &dist]() { return dist(rng); });
}

} // namespace diffusion_maps

#endif
