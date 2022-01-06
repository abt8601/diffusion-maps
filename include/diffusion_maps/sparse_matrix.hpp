#ifndef DIFFUSION_MAPS_SPARSE_MATRIX_HPP
#define DIFFUSION_MAPS_SPARSE_MATRIX_HPP

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "diffusion_maps/vector.hpp"

namespace diffusion_maps {

/**
 * Sparse matrix in the CSR format.
 */
class SparseMatrix {
protected:
  /**
   * The number of rows.
   */
  std::size_t _n_rows;
  /**
   * The number of columns.
   */
  std::size_t _n_cols;
  /**
   * The data array of the matrix.
   */
  std::unique_ptr<double[]> _data;
  /**
   * The column indices of each non-zero element.
   */
  std::unique_ptr<std::size_t[]> _col_ixs;
  /**
   * The indices of each row.
   */
  std::unique_ptr<std::size_t[]> _row_ixs;

public:
  class Triple {
  public:
    /**
     * The row index.
     */
    std::size_t row;
    /**
     * The column index.
     */
    std::size_t col;
    /**
     * The value.
     */
    double value;

    /**
     * Less-than comparison operator. Compares the row and column indices.
     *
     * @param other The other triple.
     * @return True if this triple is less than the other triple in the
     * aforementioned order.
     */
    bool operator<(const Triple &other) const {
      return std::tie(row, col) < std::tie(other.row, other.col);
    }
  };

  // Constructors.

  /**
   * Constructs an empty 0×0 matrix.
   */
  SparseMatrix()
      : _n_rows(0), _n_cols(0), _data(nullptr), _col_ixs(nullptr),
        _row_ixs(nullptr) {}

  /**
   * Constructs a sparse matrix from a vector of triples.
   *
   * @param n_rows The number of rows.
   * @param n_cols The number of columns.
   * @param triples The vector of triples. Elements are sorted in-place.
   */
  SparseMatrix(const std::size_t n_rows, const std::size_t n_cols,
               std::vector<Triple> &triples)
      : _n_rows(n_rows), _n_cols(n_cols),
        _data(std::make_unique<double[]>(triples.size())),
        _col_ixs(std::make_unique<std::size_t[]>(triples.size())),
        _row_ixs(std::make_unique<std::size_t[]>(n_rows + 1)) {
    // Sort the triples by row and column indices.
    std::sort(triples.begin(), triples.end());

    // Fill the arrays.
    for (std::size_t ri = 0, ti = 0; ri < n_rows; ++ri) {
      _row_ixs[ri] = ti;
      for (; ti < triples.size() && triples[ti].row == ri; ++ti) {
        _col_ixs[ti] = triples[ti].col;
        _data[ti] = triples[ti].value;
      }
    }
    _row_ixs[n_rows] = triples.size();
  }

  /**
   * Copy constructor.
   *
   * @param other The sparse matrix to copy.
   */
  SparseMatrix(const SparseMatrix &other)
      : _n_rows(other._n_rows), _n_cols(other._n_cols),
        _data(std::make_unique<double[]>(other.n_nz())),
        _col_ixs(std::make_unique<std::size_t[]>(other.n_nz())),
        _row_ixs(std::make_unique<std::size_t[]>(other._n_rows + 1)) {
    std::copy_n(other._data.get(), other.n_nz(), _data.get());
    std::copy_n(other._col_ixs.get(), other.n_nz(), _col_ixs.get());
    std::copy_n(other._row_ixs.get(), other._n_rows + 1, _row_ixs.get());
  }

  /**
   * Move constructor. The moved-from matrix is set to an empty 0×0 matrix.
   *
   * @param other The sparse matrix to move.
   */
  SparseMatrix(SparseMatrix &&other) noexcept
      : _n_rows(other._n_rows), _n_cols(other._n_cols),
        _data(std::move(other._data)), _col_ixs(std::move(other._col_ixs)),
        _row_ixs(std::move(other._row_ixs)) {
    other._n_rows = 0;
    other._n_cols = 0;
  }

  // Destructor.

  /**
   * Destructor.
   */
  virtual ~SparseMatrix() = default;

  // Assignment operators.

  /**
   * Copy assignment operator.
   *
   * @param other The sparse matrix to copy.
   * @return A reference to this sparse matrix.
   */
  SparseMatrix &operator=(const SparseMatrix &other) {
    if (this == &other)
      return *this;

    _n_rows = other._n_rows;
    _n_cols = other._n_cols;
    _data = std::make_unique<double[]>(other.n_nz());
    _col_ixs = std::make_unique<std::size_t[]>(other.n_nz());
    _row_ixs = std::make_unique<std::size_t[]>(other._n_rows + 1);

    std::copy_n(other._data.get(), other.n_nz(), _data.get());
    std::copy_n(other._col_ixs.get(), other.n_nz(), _col_ixs.get());
    std::copy_n(other._row_ixs.get(), other._n_rows + 1, _row_ixs.get());

    return *this;
  }

  /**
   * Move assignment operator. The moved-from matrix is set to an empty 0×0
   * matrix.
   *
   * @param other The sparse matrix to move.
   * @return A reference to this sparse matrix.
   */
  SparseMatrix &operator=(SparseMatrix &&other) {
    if (this == &other)
      return *this;

    _n_rows = other._n_rows;
    _n_cols = other._n_cols;
    _data = std::move(other._data);
    _col_ixs = std::move(other._col_ixs);
    _row_ixs = std::move(other._row_ixs);

    other._n_rows = 0;
    other._n_cols = 0;

    return *this;
  }

  // Accessors.

  /**
   * The number of rows.
   */
  std::size_t n_rows() const { return _n_rows; }

  /**
   * The number of columns.
   */
  std::size_t n_cols() const { return _n_cols; }

  /**
   * The number of non-zero elements.
   */
  std::size_t n_nz() const { return _row_ixs[_n_rows]; }

  /**
   * The data array.
   */
  double *data() { return _data.get(); }

  /**
   * The data array.
   */
  const double *data() const { return _data.get(); }

  /**
   * The column indices array.
   */
  const std::size_t *col_ixs() const { return _col_ixs.get(); }

  /**
   * The row indices array.
   */
  const std::size_t *row_ixs() const { return _row_ixs.get(); }

  // Matrix operations.

  /**
   * Matrix-vector multiplication.
   *
   * @param v The vector to multiply.
   * @return The result of the multiplication.
   * @exception std::invalid_argument If the dimensions are incompatible.
   */
  Vector operator*(const Vector &v) const {
    if (_n_cols != v.size())
      throw std::invalid_argument("incompatible dimensions");

    Vector result(_n_rows);

#ifdef PAR
#pragma omp parallel for
#endif
    for (std::size_t i = 0; i < _n_rows; ++i) {
      double sum = 0;
      for (std::size_t j = _row_ixs[i]; j < _row_ixs[i + 1]; ++j) {
        sum += _data[j] * v[_col_ixs[j]];
      }
      result[i] = sum;
    }

    return result;
  }
};

} // namespace diffusion_maps

#endif
