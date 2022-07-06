#include <algorithm>
#include <random>
#include <vector>

#include <criterion/criterion.h>

#include "diffusion_maps/sparse_matrix.hpp"
#include "diffusion_maps/vector.hpp"

Test(sparse_matrix, sparse_matrix_simple) {
  // Matrix:
  // 0 0 3 0 4
  // 0 0 5 7 0
  // 0 0 0 0 0
  // 0 2 6 0 0

  const std::size_t n_rows = 4, n_cols = 5;
  std::vector<diffusion_maps::SparseMatrix::Triplet> triplets = {
      {0, 2, 3}, {0, 4, 4}, {1, 2, 5}, {1, 3, 7}, {3, 1, 2}, {3, 2, 6}};
  std::shuffle(triplets.begin(), triplets.end(), std::default_random_engine());

  diffusion_maps::SparseMatrix sm(n_rows, n_cols, triplets);

  // Dimension check.
  cr_assert_eq(sm.n_rows(), n_rows);
  cr_assert_eq(sm.n_cols(), n_cols);
  cr_assert_eq(sm.n_nz(), triplets.size());

  // Element check.
  for (std::size_t j = 0; j < n_cols; j++) {
    // Use matrix-vector multiplication to obtain the j-th column.

    diffusion_maps::Vector v(n_cols);
    v[j] = 1;
    const diffusion_maps::Vector col = sm * v;

    // Construct the expected column.

    diffusion_maps::Vector expected_col(n_rows);
    for (const auto &triplet : triplets) {
      if (triplet.col == j) {
        expected_col[triplet.row] = triplet.value;
      }
    }

    // Check the column.
    cr_assert_eq(col, expected_col);
  }
}

Test(sparse_matrix, sparse_matrix_simple_2) {
  // Matrix:
  // -7  2  8  0  0  0  0
  // 10  0  0  0 -1  0  0
  //  0  0 -7  3  8 -8  0
  //  0  0  4  1  7  0  0
  // -1 -9  8  0  0 -3  4

  const std::size_t n_rows = 5, n_cols = 7;
  std::vector<diffusion_maps::SparseMatrix::Triplet> triplets = {
      {0, 0, -7}, {0, 1, 2},  {0, 2, 8},  {1, 0, 10}, {1, 4, -1}, {2, 2, -7},
      {2, 3, 3},  {2, 4, 8},  {2, 5, -8}, {3, 2, 4},  {3, 3, 1},  {3, 4, 7},
      {4, 0, -1}, {4, 1, -9}, {4, 2, 8},  {4, 5, -3}, {4, 6, 4}};
  std::shuffle(triplets.begin(), triplets.end(), std::default_random_engine());

  diffusion_maps::SparseMatrix sm(n_rows, n_cols, triplets);

  // Dimension check.
  cr_assert_eq(sm.n_rows(), n_rows);
  cr_assert_eq(sm.n_cols(), n_cols);
  cr_assert_eq(sm.n_nz(), triplets.size());

  // Element check.
  for (std::size_t j = 0; j < n_cols; j++) {
    // Use matrix-vector multiplication to obtain the j-th column.

    diffusion_maps::Vector v(n_cols);
    v[j] = 1;
    const diffusion_maps::Vector col = sm * v;

    // Construct the expected column.

    diffusion_maps::Vector expected_col(n_rows);
    for (const auto &triplet : triplets) {
      if (triplet.col == j) {
        expected_col[triplet.row] = triplet.value;
      }
    }

    // Check the column.
    cr_assert_eq(col, expected_col);
  }
}

Test(sparse_matrix, sparse_matrix_zero) {
  const std::size_t n_rows = 5, n_cols = 5;
  std::vector<diffusion_maps::SparseMatrix::Triplet> triplets = {};
  diffusion_maps::SparseMatrix sm(n_rows, n_cols, triplets);

  // Dimension check.
  cr_assert_eq(sm.n_rows(), n_rows);
  cr_assert_eq(sm.n_cols(), n_cols);
  cr_assert_eq(sm.n_nz(), 0);

  // Element check.
  for (std::size_t j = 0; j < n_cols; j++) {
    // Use matrix-vector multiplication to obtain the j-th column.

    diffusion_maps::Vector v(n_cols);
    v[j] = 1;
    const diffusion_maps::Vector col = sm * v;

    // Check that the column is a zero vector.
    cr_assert_eq(col, diffusion_maps::Vector(n_rows));
  }
}
