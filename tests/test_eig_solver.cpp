#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include <criterion/criterion.h>

#include "diffusion_maps/internal/eig_solver.hpp"
#include "diffusion_maps/sparse_matrix.hpp"
#include "diffusion_maps/vector.hpp"

Test(eig_solver, symmetric_power_method_simple) {
  // Matrix:
  //  4 -1  1
  // -1  3 -2
  //  1 -2  3
  //
  // Dominant eigenvalue: 6
  // Dominant eigenvector: (1, -1, 1)

  std::vector<diffusion_maps::SparseMatrix::Triplet> triplets = {
      {0, 0, 4},  {0, 1, -1}, {0, 2, 1},  {1, 0, -1}, {1, 1, 3},
      {1, 2, -2}, {2, 0, 1},  {2, 1, -2}, {2, 2, 3}};
  diffusion_maps::SparseMatrix matrix(3, 3, triplets);

  const diffusion_maps::Vector x0 = {1, 0, 0};
  const double tol = 1e-10;
  const unsigned max_iters = 100;

  const auto result = diffusion_maps::internal::symmetric_power_method(
      matrix, x0, nullptr, 0, tol, max_iters);

  cr_assert(result.has_value(), "Fail to converge");

  const auto [dominant_eigenvalue, dominant_eigenvector] = *result;

  const double expected_dominant_eigenvalue = 6;
  const diffusion_maps::Vector expected_dominant_eigenvector =
      diffusion_maps::Vector{1, -1, 1} / std::sqrt(3);
  cr_assert_float_eq(
      dominant_eigenvalue, expected_dominant_eigenvalue, tol,
      "Calculated eigenvalue %lf does not match expected eigenvalue %lf",
      dominant_eigenvalue, expected_dominant_eigenvalue);
  cr_assert_lt(
      std::min(
          (dominant_eigenvector - expected_dominant_eigenvector).l2_norm(),
          (dominant_eigenvector - (-expected_dominant_eigenvector)).l2_norm()),
      tol, "Calculated eigenvector is incorrect");
}

Test(eig_solver, eigsh_simple) {
  // Matrix:
  //  4 -1  1
  // -1  3 -2
  //  1 -2  3
  //
  // Eigenvalues:  6          3           1
  // Eigenvectors: (1, -1, 1) (-2, -1, 1) (0, 1, 1)

  std::vector<diffusion_maps::SparseMatrix::Triplet> triplets = {
      {0, 0, 4},  {0, 1, -1}, {0, 2, 1},  {1, 0, -1}, {1, 1, 3},
      {1, 2, -2}, {2, 0, 1},  {2, 1, -2}, {2, 2, 3}};
  diffusion_maps::SparseMatrix matrix(3, 3, triplets);

  const unsigned k = 3;
  const double tol = 1e-9;
  const unsigned max_iters = 100;

  std::default_random_engine rng(std::random_device{}());
  const auto [eigenvalues, eigenvectors] =
      diffusion_maps::internal::eigsh(matrix, k, tol, max_iters, rng);

  cr_assert_eq(eigenvalues.size(), k, "eigsh does not find all eigenvalues");
  cr_assert_eq(eigenvectors.size(), k, "eigsh does not find all eigenvectors");

  const std::vector<std::pair<double, diffusion_maps::Vector>> expected_result =
      {{6, diffusion_maps::Vector{1, -1, 1} / std::sqrt(3)},
       {3, diffusion_maps::Vector{-2, -1, 1} / std::sqrt(6)},
       {1, diffusion_maps::Vector{0, 1, 1} / std::sqrt(2)}};

  for (std::size_t i = 0; i < k; ++i) {
    const double eigenvalue = eigenvalues[i];
    const diffusion_maps::Vector eigenvector = eigenvectors[i];
    const auto [expected_eigenvalue, expected_eigenvector] = expected_result[i];
    cr_assert_float_eq(eigenvalue, expected_eigenvalue, tol,
                       "%zu-th calculated eigenvalue %lf does not match "
                       "expected eigenvalue %lf",
                       i, eigenvalue, expected_eigenvalue);
    cr_assert_lt(std::min((eigenvector - expected_eigenvector).l2_norm(),
                          (eigenvector - (-expected_eigenvector)).l2_norm()),
                 tol, "%zu-th calculated eigenvector is incorrect", i);
  }
}
