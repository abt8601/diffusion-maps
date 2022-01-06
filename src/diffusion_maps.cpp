#include "diffusion_maps/diffusion_maps.hpp"

#include <vector>

#include "diffusion_maps/internal/eig_solver.hpp"
#include "diffusion_maps/sparse_matrix.hpp"

/**
 * Computes the kernel matrix.
 *
 * @param data The data matrix where each row is a data point.
 * @param kernel The kernel function.
 * @param epsilon The value below which the output of the kernel would be
 * treated as zero.
 * @return The kernel matrix.
 */
static diffusion_maps::SparseMatrix compute_kernel_matrix(
    const diffusion_maps::Matrix &data,
    const std::function<double(const diffusion_maps::Vector &,
                               const diffusion_maps::Vector &)> &kernel,
    const double epsilon) {
  const std::size_t n_samples = data.n_rows();
  std::vector<diffusion_maps::SparseMatrix::Triple> triples;

#ifdef PAR
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < n_samples; ++i) {
    for (std::size_t j = i; j < n_samples; ++j) {
      const double value = kernel(data.row(i), data.row(j));
      if (std::abs(value) > epsilon) {
#ifdef PAR
#pragma omp critical
#endif
        {
          triples.push_back({i, j, value});
          if (i != j) {
            triples.push_back({j, i, value});
          }
        }
      }
    }
  }

  return diffusion_maps::SparseMatrix(n_samples, n_samples, triples);
}

/**
 * Computes the "symmetrised" diffusion matrix from the kernel matrix. The
 * matrix is updated in-place.
 *
 * @param kernel_matrix The kernel matrix.
 * @return The inverse square root of the row sum of the kernel matrix.
 */
static diffusion_maps::Vector compute_symmetrised_diffusion_matrix(
    diffusion_maps::SparseMatrix &kernel_matrix) {
  const diffusion_maps::Vector invsqrt_row_sum =
      (kernel_matrix * diffusion_maps::Vector(kernel_matrix.n_rows(), 1))
          .inv_sqrt();

#ifdef PAR
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < kernel_matrix.n_rows(); ++i) {
    for (std::size_t ir = kernel_matrix.row_ixs()[i];
         ir < kernel_matrix.row_ixs()[i + 1]; ++ir) {
      const std::size_t j = kernel_matrix.col_ixs()[ir];
      double &v = kernel_matrix.data()[ir];

      v *= invsqrt_row_sum[i] * invsqrt_row_sum[j];
    }
  }

  return invsqrt_row_sum;
}

diffusion_maps::Matrix diffusion_maps::internal::diffusion_maps(
    const Matrix &data, const std::size_t n_components,
    const std::function<double(const Vector &, const Vector &)> &kernel,
    const double diffusion_time, const double kernel_epsilon,
    const double eig_solver_tol, const unsigned eig_solver_max_iter,
    const std::function<double()> &rng) {
  const std::size_t n_samples = data.n_rows();
  if (n_components > n_samples - 1) {
    throw std::invalid_argument("too many components");
  }
  if (diffusion_time < 0) {
    throw std::invalid_argument("diffusion time must be non-negative");
  }

  // Step 1: Compute the kernel matrix.

  auto kernel_matrix = compute_kernel_matrix(data, kernel, kernel_epsilon);

  // Step 2: Compute the "symmetrised" diffusion matrix.

  const auto invsqrt_row_sum =
      compute_symmetrised_diffusion_matrix(kernel_matrix);

  // Step 3: Compute the eigenvalues and eigenvectors of the diffusion matrix.

  const auto [eigenvalues, eigenvectors] =
      internal::eigsh(kernel_matrix, n_components + 1, eig_solver_tol,
                      eig_solver_max_iter, rng);

  // Step 4: Compute the diffusion maps.

  const std::size_t n_eigenvalues = eigenvalues.size();
  Matrix diffusion_maps(n_samples, n_eigenvalues == 0 ? 0 : n_eigenvalues - 1);

  if (n_eigenvalues != 0) {
    for (std::size_t i = 0; i < n_samples; ++i) {
      for (std::size_t j = 0; j < n_eigenvalues - 1; ++j) {
        // We drop the first eigenpair because the eigenvector is constant in
        // all dimensions.

        const double lambda = eigenvalues[j + 1];
        const double psi_i = invsqrt_row_sum[i] * eigenvectors[j + 1][i];
        diffusion_maps(i, j) = std::pow(lambda, diffusion_time) * psi_i;
      }
    }
  }

  return diffusion_maps;
}
