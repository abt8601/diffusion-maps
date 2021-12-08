#ifndef DIFFUSION_MAPS_DIFFUSION_MAPS_HPP
#define DIFFUSION_MAPS_DIFFUSION_MAPS_HPP

#include <functional>
#include <memory>

#include "diffusion_maps/matrix.hpp"
#include "diffusion_maps/vector.hpp"

namespace diffusion_maps {

/**
 * Diffusion maps.
 *
 * @param data The data matrix where each row is a data point.
 * @param n_components The dimension of the projected subspace.
 * @param kernel The kernel function.
 * @param diffusion_time The diffusion time.
 * @param kernel_epsilon The value below which the output of the kernel would be
 * treated as zero.
 * @param eig_solver_tol The tolerance of the eigendecomposition solver.
 * @param eig_solver_max_iter The maximum number of iterations of the
 * eigendecomposition solver.
 * @param eig_solver_max_restarts The maximum number of restarts of the
 * eigendecomposition solver.
 * @return The lower-dimensional embedding of the data in the diffusion space.
 */
std::pair<std::unique_ptr<double[]>, Matrix> diffusion_maps(
    const Matrix &data, std::size_t n_components,
    const std::function<double(const Vector &, const Vector &)> &kernel,
    double diffusion_time, double kernel_epsilon = 1e-6,
    double eig_solver_tol = 1e-6, unsigned eig_solver_max_iter = 1000,
    unsigned eig_solver_max_restarts = 10);

} // namespace diffusion_maps

#endif
