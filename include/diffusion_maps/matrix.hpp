/// \file
///
/// \brief Matrix.

#ifndef DIFFUSION_MAPS_MATRIX_HPP
#define DIFFUSION_MAPS_MATRIX_HPP

#include <cstddef>
#include <memory>
#include <variant>

#include "diffusion_maps/internal/utils.hpp"
#include "diffusion_maps/vector.hpp"

namespace diffusion_maps {

/// Matrix of doubles that may or may not own its data.
class Matrix {
protected:
  /// The data array.
  std::variant<std::unique_ptr<double[]>, double *> _data;
  /// The number of rows.
  std::size_t _n_rows;
  /// The number of columns.
  std::size_t _n_cols;
  /// The stride in the row dimension.
  std::size_t _row_stride;
  /// The stride in the column dimension.
  std::size_t _col_stride;

public:
  /// \brief Constructs an owning matrix of the given dimensions with
  ///        uninitialized elements.
  ///
  /// \param[in] n_rows The number of rows.
  /// \param[in] n_cols The number of columns.
  Matrix(const std::size_t n_rows, const std::size_t n_cols)
      : _data(std::make_unique<double[]>(n_rows * n_cols)), _n_rows(n_rows),
        _n_cols(n_cols), _row_stride(n_cols), _col_stride(1) {}

  /// \brief Constructs a non-owning matrix.
  ///
  /// \param[in] data The data array.
  /// \param[in] n_rows The number of rows.
  /// \param[in] n_cols The number of columns.
  /// \param[in] row_stride The stride in the row dimension.
  /// \param[in] col_stride The stride in the column dimension.
  Matrix(double *const data, const std::size_t n_rows, const std::size_t n_cols,
         const std::size_t row_stride, const std::size_t col_stride)
      : _data(data), _n_rows(n_rows), _n_cols(n_cols), _row_stride(row_stride),
        _col_stride(col_stride) {}

  /// The data array.
  double *data() {
    return std::visit(
        internal::overloaded{
            [](std::unique_ptr<double[]> &data) { return data.get(); },
            [](double *data) { return data; },
            [](auto &&) -> double * {
              throw std::runtime_error("invalid data array");
            }},
        _data);
  }

  /// The data array.
  const double *data() const {
    return std::visit(
        internal::overloaded{
            [](const std::unique_ptr<double[]> &data) {
              return const_cast<const double *>(data.get());
            },
            [](double *data) { return const_cast<const double *>(data); },
            [](auto &&) -> const double * {
              throw std::runtime_error("invalid data array");
            }},
        _data);
  }

  /// The number of rows.
  std::size_t n_rows() const { return _n_rows; }

  /// The number of columns.
  std::size_t n_cols() const { return _n_cols; }

  /// The stride in the row dimension.
  std::size_t row_stride() const { return _row_stride; }

  /// The stride in the column dimension.
  std::size_t col_stride() const { return _col_stride; }

  /// \brief Returns the ( \p i , \p j )-th element without bounds checking.
  ///
  /// \param[in] i The row index.
  /// \param[in] j The column index.
  /// \return The ( \p i , \p j )-th element.
  double &operator()(const std::size_t i, const std::size_t j) {
    return data()[i * _row_stride + j * _col_stride];
  }

  /// \brief Returns the ( \p i , \p j )-th element without bounds checking.
  ///
  /// \param[in] i The row index.
  /// \param[in] j The column index.
  /// \return The ( \p i , \p j )-th element.
  double operator()(const std::size_t i, const std::size_t j) const {
    return data()[i * _row_stride + j * _col_stride];
  }

  /// \brief Returns the \p i -th row without bounds checking.
  ///
  /// \param[in] i The row index.
  /// \return The \p i -th row.
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
