/// \file
///
/// \brief Vector.

#ifndef DIFFUSION_MAPS_VECTOR_HPP
#define DIFFUSION_MAPS_VECTOR_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>

namespace diffusion_maps {

/// Vector.
class Vector {
protected:
  /// The size.
  std::size_t _size;
  /// The data array.
  std::unique_ptr<double[]> _data;

public:
  // Constructors.

  /// Constructs a vector of size 0.
  Vector() : _size(0), _data(nullptr) {}

  /// \brief Constructs a vector of size \p size.
  ///
  /// \param[in] size The size of the vector.
  Vector(const std::size_t size)
      : _size(size), _data(std::make_unique<double[]>(_size)) {}

  /// \brief Constructs a vector of size \p size with each element set to
  ///        \p value.
  ///
  /// \param[in] size The size of the vector.
  /// \param[in] value The value of each element.
  Vector(const std::size_t size, const double value)
      : _size(size), _data(std::make_unique<double[]>(_size)) {
    std::fill_n(_data.get(), _size, value);
  }

  /// \brief Constructs a vector from an initializer list.
  ///
  /// \param[in] list The initializer list.
  Vector(const std::initializer_list<double> list)
      : _size(list.size()), _data(std::make_unique<double[]>(_size)) {
    std::copy(list.begin(), list.end(), _data.get());
  }

  /// \brief Copy constructor.
  ///
  /// \param[in] other The vector to copy.
  Vector(const Vector &other)
      : _size(other._size), _data(std::make_unique<double[]>(_size)) {
    std::copy_n(other._data.get(), _size, _data.get());
  }

  /// \brief Move constructor. The moved-from vector is set to a 0-sized vector.
  ///
  /// \param[in,out] other The vector to move.
  Vector(Vector &&other) noexcept
      : _size(other._size), _data(std::move(other._data)) {
    other._size = 0;
  }

  // Destructor.

  /// Destructor.
  virtual ~Vector() = default;

  // Assignment operators.

  /// \brief Copy assignment operator.
  ///
  /// \param[in] other The vector to copy.
  /// \return A reference to this vector.
  Vector &operator=(const Vector &other) {
    if (this == &other)
      return *this;

    _size = other._size;
    _data = std::make_unique<double[]>(_size);
    std::copy_n(other._data.get(), _size, _data.get());

    return *this;
  }

  /// \brief Move assignment operator. The moved-from vector is set to a 0-sized
  ///        vector.
  ///
  /// \param[in,out] other The vector to move.
  /// \return A reference to this vector.
  Vector &operator=(Vector &&other) {
    if (this == &other)
      return *this;

    _size = other._size;
    _data = std::move(other._data);

    other._size = 0;

    return *this;
  }

  // Member access.

  /// The size of the vector.
  std::size_t size() const noexcept { return _size; }

  /// The data array of the vector.
  double *data() noexcept { return _data.get(); }

  /// The data array of the vector.
  const double *data() const noexcept { return _data.get(); }

  // Element access.

  /// \brief Gets the element at index \p index without bounds checking.
  ///
  /// \param[in] index The index of the element.
  /// \return The element at index \p index.
  double &operator[](const std::size_t index) { return _data[index]; }

  /// \brief Gets the element at index \p index without bounds checking.
  ///
  /// \param[in] index The index of the element.
  /// \return The element at index \p index.
  double operator[](const std::size_t index) const { return _data[index]; }

  // Equality.

  /// \brief Equality operator.
  ///
  /// \param[in] other The vector to compare with.
  /// \return True if the vectors are equal, false otherwise.
  bool operator==(const Vector &other) const {
    return _size == other._size &&
           std::equal(_data.get(), _data.get() + _size, other._data.get());
  }

  /// \brief Inequality operator.
  ///
  /// \param[in] other The vector to compare with.
  /// \return True if the vectors are not equal, false otherwise.
  bool operator!=(const Vector &other) const { return !(*this == other); }

  // Vector operations.

  /// \brief Negation operator.
  ///
  /// \return The negation of the vector.
  Vector operator-() const {
    Vector result(_size);
    std::transform(_data.get(), _data.get() + _size, result._data.get(),
                   [](double x) { return -x; });
    return result;
  }

  /// \brief Addition operator.
  ///
  /// \param[in] other The vector to add.
  /// \return The sum of the vectors.
  /// \exception std::invalid_argument If the vectors are not of the same size.
  Vector operator+(const Vector &other) const {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    Vector result(_size);
    for (std::size_t i = 0; i < _size; ++i)
      result._data[i] = _data[i] + other._data[i];

    return result;
  }

  /// \brief Addition assignment operator.
  ///
  /// \param[in] other The vector to add.
  /// \return A reference to this vector.
  /// \exception std::invalid_argument If the vectors are not of the same size.
  Vector &operator+=(const Vector &other) {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    for (std::size_t i = 0; i < _size; ++i)
      _data[i] += other._data[i];

    return *this;
  }

  /// \brief Subtraction operator.
  ///
  /// \param[in] other The vector to subtract.
  /// \return The difference of the vectors.
  /// \exception std::invalid_argument If the vectors are not of the same size.
  Vector operator-(const Vector &other) const {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    Vector result(_size);
    for (std::size_t i = 0; i < _size; ++i)
      result._data[i] = _data[i] - other._data[i];

    return result;
  }

  /// \brief Subtraction assignment operator.
  ///
  /// \param[in] other The vector to subtract.
  /// \return A reference to this vector.
  /// \exception std::invalid_argument If the vectors are not of the same size.
  Vector &operator-=(const Vector &other) {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    for (std::size_t i = 0; i < _size; ++i)
      _data[i] -= other._data[i];

    return *this;
  }

  /// \brief Scalar multiplication operator.
  ///
  /// \param[in] scalar The scalar to multiply by.
  /// \return The scaled vector.
  Vector operator*(const double scalar) const {
    Vector result(_size);
    for (std::size_t i = 0; i < _size; ++i)
      result._data[i] = _data[i] * scalar;

    return result;
  }

  /// \brief Scalar multiplication assignment operator.
  ///
  /// \param[in] scalar The scalar to multiply by.
  /// \return A reference to this vector.
  Vector &operator*=(const double scalar) {
    for (std::size_t i = 0; i < _size; ++i)
      _data[i] *= scalar;

    return *this;
  }

  /// \brief Scalar division operator.
  ///
  /// \param[in] scalar The scalar to divide by.
  /// \return The scaled vector.
  Vector operator/(const double scalar) const {
    Vector result(_size);
    for (std::size_t i = 0; i < _size; ++i)
      result._data[i] = _data[i] / scalar;

    return result;
  }

  /// \brief Scalar division assignment operator.
  ///
  /// \param[in] scalar The scalar to divide by.
  /// \return A reference to this vector.
  Vector &operator/=(const double scalar) {
    for (std::size_t i = 0; i < _size; ++i)
      _data[i] /= scalar;

    return *this;
  }

  /// \brief Dot product.
  ///
  /// \param[in] other The vector to dot with.
  /// \return The dot product of the vectors.
  /// \exception std::invalid_argument If the vectors are not of the same size.
  double dot(const Vector &other) const {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    double result = 0;
    for (std::size_t i = 0; i < _size; ++i)
      result += _data[i] * other._data[i];

    return result;
  }

  /// The squared 2-norm (Euclidean norm) of the vector.
  double sq_l2_norm() const { return dot(*this); }

  /// The 2-norm (Euclidean norm) of the vector.
  double l2_norm() const { return std::sqrt(sq_l2_norm()); }

  /// \brief Returns a vector where each element is the inverse square root of
  ///        the corresponding element of this vector.
  ///
  /// \return The inverse square root of the vector.
  Vector inv_sqrt() const {
    Vector result(_size);
    std::transform(_data.get(), _data.get() + _size, result._data.get(),
                   [](double x) { return 1.0 / std::sqrt(x); });
    return result;
  }
};

} // namespace diffusion_maps

#endif
