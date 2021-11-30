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
