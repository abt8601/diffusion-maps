#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include <criterion/criterion.h>

#include "diffusion_maps/eig_solver.hpp"
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

  std::vector<diffusion_maps::SparseMatrix::Triple> triples = {
      {0, 0, 4},  {0, 1, -1}, {0, 2, 1},  {1, 0, -1}, {1, 1, 3},
      {1, 2, -2}, {2, 0, 1},  {2, 1, -2}, {2, 2, 3}};
  diffusion_maps::SparseMatrix matrix(3, 3, triples);

  const diffusion_maps::Vector x0 = {1, 0, 0};
  const double tol = 1e-10;
  const unsigned max_iters = 100;

  const auto result =
      diffusion_maps::symmetric_power_method(matrix, x0, tol, max_iters);

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

  std::vector<diffusion_maps::SparseMatrix::Triple> triples = {
      {0, 0, 4},  {0, 1, -1}, {0, 2, 1},  {1, 0, -1}, {1, 1, 3},
      {1, 2, -2}, {2, 0, 1},  {2, 1, -2}, {2, 2, 3}};
  diffusion_maps::SparseMatrix matrix(3, 3, triples);

  const unsigned k = 3;
  const double tol = 1e-9;
  const unsigned max_iters = 100;
  const unsigned max_restarts = 3;

  const auto result =
      diffusion_maps::eigsh(matrix, k, tol, max_iters, max_restarts);

  cr_assert_eq(result.size(), k, "eigsh does not find all eigenvalues");

  const std::vector<std::pair<double, diffusion_maps::Vector>> expected_result =
      {{6, diffusion_maps::Vector{1, -1, 1} / std::sqrt(3)},
       {3, diffusion_maps::Vector{-2, -1, 1} / std::sqrt(6)},
       {1, diffusion_maps::Vector{0, 1, 1} / std::sqrt(2)}};

  for (std::size_t i = 0; i < k; ++i) {
    const auto [eigenvalue, eigenvector] = result[i];
    const auto [expected_eigenvalue, expected_eigenvector] = expected_result[i];
    cr_assert_float_eq(
        eigenvalue, expected_eigenvalue, tol,
        "Calculated eigenvalue %lf does not match expected eigenvalue %lf",
        eigenvalue, expected_eigenvalue);
    cr_assert_lt(std::min((eigenvector - expected_eigenvector).l2_norm(),
                          (eigenvector - (-expected_eigenvector)).l2_norm()),
                 tol, "Calculated eigenvector is incorrect");
  }
}
