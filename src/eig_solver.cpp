#include "diffusion_maps/internal/eig_solver.hpp"

#include <random>
#include <stdexcept>

std::optional<std::pair<double, diffusion_maps::Vector>>
diffusion_maps::internal::symmetric_power_method(const SparseMatrix &a,
                                                 const Vector &x0,
                                                 const double tol,
                                                 const unsigned max_iters,
                                                 const double epsilon) {
  if (a.n_rows() != a.n_cols()) { // a is not square.
    throw std::invalid_argument("matrix is not square");
  }
  if (x0.size() != a.n_rows()) { // x0 cannot be multiplied by a.
    throw std::invalid_argument("incompatible dimensions");
  }

  Vector x = x0 / x0.l2_norm();
  double mu_0 = 0, mu_1 = 0;

  for (unsigned k = 0; k < max_iters; ++k) {
    Vector y = a * x;
    const double mu = x.dot(y);
    const double denom = (mu - mu_1) - (mu_1 - mu_0);
    const double mu_hat = std::abs(denom) < epsilon
                              ? mu_0
                              : mu_0 - ((mu_1 - mu_0) * (mu_1 - mu_0)) / denom;

    const double l2_norm_y = y.l2_norm();
    if (l2_norm_y == 0) { // a has eigenvalue 0.
      return std::make_pair(0, x);
    }

    y /= l2_norm_y;
    const double err = (x - y).l2_norm();
    x = y;
    if (k >= 3 && err < tol) { // Success.
      return std::make_pair(mu_hat, x);
    }

    mu_0 = mu_1;
    mu_1 = mu;
  }

  return std::nullopt; // Failed to converge.
}

std::vector<std::pair<double, diffusion_maps::Vector>>
diffusion_maps::internal::eigsh(const SparseMatrix &a, const unsigned k,
                                const double tol, const unsigned max_iters,
                                const unsigned max_restarts) {
  if (a.n_rows() != a.n_cols()) { // a is not square.
    throw std::invalid_argument("matrix is not square");
  }
  if (k > a.n_rows()) { // k cannot be larger than the number of rows.
    throw std::invalid_argument("k cannot be larger than the number of rows");
  }

  std::vector<std::pair<double, Vector>> eig_pairs;
  eig_pairs.reserve(k);

  for (std::size_t i = 0; i < k; ++i) {
    for (unsigned restarts = 0; restarts < max_restarts; ++restarts) {
      // Construct the initial vector for the symmetric power method
      //   x0 = (A - λᵢ₋₁ I) … (A - λ₁ I) (A - λ₀ I) x
      // where x is a random vector. (Annihilation technique.)

      Vector x0(a.n_rows());

      std::default_random_engine gen;
      std::uniform_real_distribution<double> dist(0, 1);
      for (std::size_t i = 0; i < x0.size(); ++i) {
        x0[i] = dist(gen);
      }

      for (std::size_t j = 0; j < i; j++) {
        const double lambda_j = eig_pairs[j].first;
        x0 = a * x0 - x0 * lambda_j;
      }

      // Use the symmetric power method to find the i-th eigenvalue and
      // eigenvector.

      const auto eig_pair = symmetric_power_method(a, x0, tol, max_iters);

      // Restart if the method does not converge or finds an eigenvalue 0.
      if (!eig_pair.has_value() || eig_pair->first == 0) {
        continue;
      }

      eig_pairs.push_back(eig_pair.value());
      break;
    }
  }

  return eig_pairs;
}
