#pragma once
#ifndef DIFFUSION_MAPS_EIG_SOLVER_HPP
#define DIFFUSION_MAPS_EIG_SOLVER_HPP

#include <optional>

#include "diffusion_maps/sparse_matrix.hpp"
#include "diffusion_maps/vector.hpp"

namespace diffusion_maps {

/**
 * Find the dominant eigenvalue and its corresponding eigenvector of a symmetric
 * matrix using the symmetric power method.
 *
 * @param a The matrix.
 * @param x0 The initial guess for the eigenvector.
 * @param tol The tolerance for the eigenvector.
 * @param max_iters The maximum number of iterations.
 * @return The dominant eigenvalue and its corresponding eigenvector. Or nullopt
 * if the maximum number of iterations is exceeded.
 * @throws std::invalid_argument if the dimensions are incorrect.
 */
std::optional<std::pair<double, Vector>>
symmetric_power_method(const SparseMatrix &a, const Vector &x0, double tol,
                       unsigned max_iters);

} // namespace diffusion_maps

#endif
