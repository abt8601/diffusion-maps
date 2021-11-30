#include "diffusion_maps/eig_solver.hpp"

#include <stdexcept>

std::optional<std::pair<double, diffusion_maps::Vector>>
diffusion_maps::symmetric_power_method(const SparseMatrix &a,
                                       const diffusion_maps::Vector &x0,
                                       const double tol,
                                       const unsigned max_iters) {
  if (a.n_rows() != a.n_cols()) { // a is not square.
    throw std::invalid_argument("matrix is not square");
  }
  if (x0.size() != a.n_rows()) { // x0 cannot be multiplied by a.
    throw std::invalid_argument("incompatible dimensions");
  }

  diffusion_maps::Vector x = x0 / x0.l2_norm();
  double mu_0 = 0, mu_1 = 0;

  for (unsigned k = 0; k < max_iters; ++k) {
    diffusion_maps::Vector y = a * x;
    const double mu = x.dot(y);
    const double mu_hat =
        mu_0 - ((mu_1 - mu_0) * (mu_1 - mu_0)) / (mu - 2 * mu_1 + mu_0);

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
