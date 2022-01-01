#ifndef DIFFUSION_MAPS_INTERNAL_EIG_SOLVER_HPP
#define DIFFUSION_MAPS_INTERNAL_EIG_SOLVER_HPP

#include <functional>
#include <optional>
#include <vector>

#include "diffusion_maps/sparse_matrix.hpp"
#include "diffusion_maps/vector.hpp"

namespace diffusion_maps {

namespace internal {

/**
 * Find an eigenvalue and its corresponding eigenvector of a symmetric matrix.
 * <p>
 * The numerical method used is the symmetric power method with "reprojection",
 * i.e., during each iteration, the eigenvector is orthogonalised against the
 * previously found eigenvectors. This is done so that we can find the k-th
 * eigenvalue/eigenvector after we have found the first k-1.
 * <p>
 * The reprojection is meant to solve the numerical instability problem of the
 * annihilation technique. Say that A has eigenvalues λ₁, λ₂, ..., λₙ with the
 * corresponding eigenvectors being β₁, β₂, ..., βₙ and we have found the first
 * k-1 of them. The annihilation technique chooses the initial guess for the
 * eigenvector to be x₀ = (A - λₖ₋₁ I) (A - λₖ₋₂ I) ... (A - λ₁ I) x where x is
 * some random vector. Observe that if we write x₀ = ∑ᵢ cᵢ βᵢ, then c₁ = c₂
 * = ... = cₖ₋₁ = 0. If we run the standard symmetric power method on x₀, then
 * the components for β₁, β₂, ..., βₖ₋₁ in the eigenvector should stay zero and
 * the eigenvector should converge to βₖ. However, with the presence of rounding
 * error, the components for β₁, β₂, ..., βₖ₋₁ may start to appear and the
 * standard symmetric power method will converge to a different eigenvector.
 * Reprojection solves this problem by orthogonalising the eigenvector against
 * β₁, β₂, ..., βₖ₋₁ after each time the eigenvector is multiplied by the
 * matrix. Basically, this actively suppresses the components for β₁, β₂, ...,
 * βₖ₋₁ in the eigenvector.
 *
 * @param a The matrix.
 * @param x0 The initial guess for the eigenvector.
 * @param betas The array of previously found eigenvectors, all normalised with
 * respect to the Euclidean norm.
 * @param n_betas The number of previously found eigenvectors.
 * @param tol The tolerance for the Euclidean norm of the eigenvector.
 * @param max_iters The maximum number of iterations.
 * @return An eigenvalue and its corresponding eigenvector. Or nullopt if the
 * maximum number of iterations is exceeded.
 * @exception std::invalid_argument If the dimensions are incorrect.
 */
std::optional<std::pair<double, Vector>>
symmetric_power_method(const SparseMatrix &a, const Vector &x0,
                       const Vector *betas, std::size_t n_betas, double tol,
                       unsigned max_iters);

/**
 * Find @p k dominant eigenvalues and their corresponding eigenvectors of a
 * symmetric matrix using the symmetric power method.
 *
 * @param a The matrix.
 * @param k The number of dominant eigenvalues to find.
 * @param tol The tolerance for the eigenvectors.
 * @param max_iters The maximum number of iterations to find each eigenvector.
 * @param rng A function that generates a random number.
 * @return The dominant eigenvalues and their corresponding eigenvectors. If the
 * method fails to find all @p k eigenvalues and eigenvectors, it will return
 * less than @p k eigenvalues and eigenvectors.
 * @exception std::invalid_argument If @p a is not square.
 * @exception std::invalid_argument If @p k is greater than the number of rows
 * in @p a.
 */
std::pair<std::vector<double>, std::vector<Vector>>
eigsh(const SparseMatrix &a, unsigned k, double tol, unsigned max_iters,
      const std::function<double()> &rng);

} // namespace internal

} // namespace diffusion_maps

#endif
