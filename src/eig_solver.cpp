#include "diffusion_maps/internal/eig_solver.hpp"

#include <random>
#include <stdexcept>

std::optional<std::pair<double, diffusion_maps::Vector>>
diffusion_maps::internal::symmetric_power_method(
    const SparseMatrix &a, const Vector &x0, const Vector *const betas,
    const std::size_t n_betas, const double tol, const unsigned max_iters) {
  if (a.n_rows() != a.n_cols()) { // a is not square.
    throw std::invalid_argument("matrix is not square");
  }
  if (x0.size() != a.n_rows()) { // x0 cannot be multiplied by a.
    throw std::invalid_argument("incompatible dimensions");
  }

  Vector x = x0 / x0.l2_norm();

  for (unsigned k = 0; k < max_iters; ++k) {
    Vector y = a * x;

    // Orthogonalise y against betas.
    for (std::size_t i = 0; i < n_betas; ++i) {
      y -= betas[i] * betas[i].dot(y);
    }

    const double mu = x.dot(y);

    const double l2_norm_y = y.l2_norm();
    if (l2_norm_y == 0) { // a has eigenvalue 0.
      return std::make_pair(0, x);
    }

    y /= l2_norm_y;
    const double err = (x - y).l2_norm();
    x = y;
    if (err < tol) { // Success.
      return std::make_pair(mu, x);
    }
  }

  return std::nullopt; // Failed to converge.
}

std::pair<std::vector<double>, std::vector<diffusion_maps::Vector>>
diffusion_maps::internal::eigsh(const SparseMatrix &a, const unsigned k,
                                const double tol, const unsigned max_iters) {
  if (a.n_rows() != a.n_cols()) { // a is not square.
    throw std::invalid_argument("matrix is not square");
  }
  if (k > a.n_rows()) { // k cannot be larger than the number of rows.
    throw std::invalid_argument("k cannot be larger than the number of rows");
  }

  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0, 1);

  std::vector<double> eigenvalues;
  std::vector<Vector> eigenvectors;
  eigenvalues.reserve(k);
  eigenvectors.reserve(k);

  for (std::size_t i = 0; i < k; ++i) {
    // Generate the initial guess for the eigenvector.

    Vector x0(a.n_rows());
    for (std::size_t i = 0; i < x0.size(); ++i) {
      x0[i] = dist(gen);
    }

    // Use the symmetric power method to find the i-th eigenvalue and
    // eigenvector.

    const auto eig_pair = symmetric_power_method(
        a, x0, eigenvectors.data(), eigenvectors.size(), tol, max_iters);

    // Stop if the eigenvalue is not found.
    if (!eig_pair) {
      break;
    }

    eigenvalues.push_back(eig_pair->first);
    eigenvectors.push_back(eig_pair->second);
  }

  return std::make_pair(eigenvalues, eigenvectors);
}
