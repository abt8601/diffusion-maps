#ifndef DIFFUSION_MAPS_MATRIX_HPP
#define DIFFUSION_MAPS_MATRIX_HPP

#include <cstddef>
#include <memory>

#include "diffusion_maps/vector.hpp"

namespace diffusion_maps {

/**
 * Non-owning matrix of doubles.
 */
class Matrix {
public:
  /**
   * A buffer that can be used to store the matrix data.
   */
  class Buffer {
  public:
    /**
     * The data array.
     */
    std::unique_ptr<double[]> data;
    /**
     * The number of rows.
     */
    std::size_t n_rows;
    /**
     * The number of columns.
     */
    std::size_t n_cols;

    /**
     * Constructs a buffer with the given dimensions.
     *
     * @param n_rows The number of rows.
     * @param n_cols The number of columns.
     */
    Buffer(const std::size_t n_rows, const std::size_t n_cols)
        : data(std::make_unique<double[]>(n_rows * n_cols)), n_rows(n_rows),
          n_cols(n_cols) {}

    /**
     * Take the data array of the buffer.
     */
    std::unique_ptr<double[]> take() { return std::move(data); }
  };

protected:
  /**
   * The data array.
   */
  double *_data;
  /**
   * The number of rows.
   */
  std::size_t _n_rows;
  /**
   * The number of columns.
   */
  std::size_t _n_cols;
  /**
   * The stride in the row dimension.
   */
  std::size_t _row_stride;
  /**
   * The stride in the column dimension.
   */
  std::size_t _col_stride;

public:
  /**
   * Constructs a matrix.
   *
   * @param data The data array.
   * @param n_rows The number of rows.
   * @param n_cols The number of columns.
   * @param row_stride The stride in the row dimension.
   * @param col_stride The stride in the column dimension.
   */
  Matrix(double *const data, const std::size_t n_rows, const std::size_t n_cols,
         const std::size_t row_stride, const std::size_t col_stride)
      : _data(data), _n_rows(n_rows), _n_cols(n_cols), _row_stride(row_stride),
        _col_stride(col_stride) {}

  /**
   * Constructs a matrix from a buffer.
   */
  Matrix(const Buffer &buffer)
      : Matrix(buffer.data.get(), buffer.n_rows, buffer.n_cols, buffer.n_cols,
               1) {}

  /**
   * The data array.
   */
  double *data() { return _data; }

  /**
   * The data array.
   */
  const double *data() const { return _data; }

  /**
   * The number of rows.
   */
  std::size_t n_rows() const { return _n_rows; }

  /**
   * The number of columns.
   */
  std::size_t n_cols() const { return _n_cols; }

  /**
   * The stride in the row dimension.
   */
  std::size_t row_stride() const { return _row_stride; }

  /**
   * The stride in the column dimension.
   */
  std::size_t col_stride() const { return _col_stride; }

  /**
   * Returns the ( @p i , @p j )-th element without bounds checking.
   *
   * @param i The row index.
   * @param j The column index.
   * @return The ( @p i , @p j )-th element.
   */
  double &operator()(const std::size_t i, const std::size_t j) {
    return _data[i * _row_stride + j * _col_stride];
  }

  /**
   * Returns the ( @p i , @p j )-th element without bounds checking.
   *
   * @param i The row index.
   * @param j The column index.
   * @return The ( @p i , @p j )-th element.
   */
  double operator()(const std::size_t i, const std::size_t j) const {
    return _data[i * _row_stride + j * _col_stride];
  }

  /**
   * Returns the @p i -th row without bounds checking.
   *
   * @param i The row index.
   * @return The @p i -th row.
   */
  Vector row(const std::size_t i) const {
    Vector result(_n_cols);
    for (std::size_t j = 0; j < _n_cols; ++j) {
      result[j] = (*this)(i, j);
    }
    return result;
  }
};

} // namespace diffusion_maps

#endif
