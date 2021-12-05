#ifndef DIFFUSION_MAPS_EIG_SOLVER_HPP
#define DIFFUSION_MAPS_EIG_SOLVER_HPP

#include <optional>
#include <vector>

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
 * @param epsilon The denominator cut-off for the Aitken's Δ² method. If the
 * absolute value of the denominator is less than this value, the method will
 * stop.
 * @return The dominant eigenvalue and its corresponding eigenvector. Or nullopt
 * if the maximum number of iterations is exceeded.
 * @throws std::invalid_argument if the dimensions are incorrect.
 */
std::optional<std::pair<double, Vector>>
symmetric_power_method(const SparseMatrix &a, const Vector &x0, double tol,
                       unsigned max_iters, double epsilon = 1e-16);

/**
 * Find @p k dominant eigenvalues and their corresponding eigenvectors of a
 * symmetric matrix using the symmetric power method.
 *
 * @param a The matrix.
 * @param k The number of dominant eigenvalues to find.
 * @param tol The tolerance for the eigenvectors.
 * @param max_iters The maximum number of iterations to find each eigenvector.
 * @param max_restarts The maximum number of restarts.
 * @return The dominant eigenvalues and their corresponding eigenvectors. If the
 * method fails to find all @p k eigenvalues and eigenvectors, it will return
 * less than @p k eigenvalues and eigenvectors.
 * @throws std::invalid_argument if @p a is not square.
 * @throws std::invalid_argument if @p k is greater than the number of rows in
 * @p a.
 */
std::vector<std::pair<double, Vector>> eigsh(const SparseMatrix &a, unsigned k,
                                             double tol, unsigned max_iters,
                                             unsigned max_restarts);

} // namespace diffusion_maps

#endif
