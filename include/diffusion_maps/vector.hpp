#pragma once
#ifndef DIFFUSION_MAPS_VECTOR_HPP
#define DIFFUSION_MAPS_VECTOR_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>

namespace diffusion_maps {

/**
 * Vector.
 */
class Vector {
protected:
  /**
   * The size of the vector.
   */
  std::size_t _size;
  /**
   * The data array of the vector.
   */
  std::unique_ptr<double[]> _data;

public:
  // Constructors.

  /**
   * Default constructor. Construct a vector of size 0.
   */
  Vector() : _size(0), _data(nullptr) {}

  /**
   * Construct a vector of size @p size.
   * @param size The size of the vector.
   */
  Vector(const std::size_t size)
      : _size(size), _data(std::make_unique<double[]>(_size)) {}

  /**
   * Construct a vector of size @p size with each element set to @p value.
   * @param size The size of the vector.
   * @param value The value of each element.
   */
  Vector(const std::size_t size, const double value)
      : _size(size), _data(std::make_unique<double[]>(_size)) {
    std::fill_n(_data.get(), _size, value);
  }

  /**
   * Construct a vector from an initializer list.
   * @param list The initializer list.
   */
  Vector(const std::initializer_list<double> list)
      : _size(list.size()), _data(std::make_unique<double[]>(_size)) {
    std::copy(list.begin(), list.end(), _data.get());
  }

  /**
   * Copy constructor.
   * @param other The vector to copy.
   */
  Vector(const Vector &other)
      : _size(other._size), _data(std::make_unique<double[]>(_size)) {
    std::copy_n(other._data.get(), _size, _data.get());
  }

  /**
   * Move constructor. The moved-from vector is set to a 0-sized vector.
   * @param other The vector to move.
   */
  Vector(Vector &&other) noexcept
      : _size(other._size), _data(std::move(other._data)) {
    other._size = 0;
  }

  // Destructor.

  /**
   * Destructor.
   */
  virtual ~Vector() = default;

  // Assignment operators.

  /**
   * Copy assignment operator.
   * @param other The vector to copy.
   * @return A reference to this vector.
   */
  Vector &operator=(const Vector &other) {
    if (this == &other)
      return *this;

    _size = other._size;
    _data = std::make_unique<double[]>(_size);
    std::copy_n(other._data.get(), _size, _data.get());

    return *this;
  }

  /**
   * Move assignment operator. The moved-from vector is set to a 0-sized vector.
   * @param other The vector to move.
   * @return A reference to this vector.
   */
  Vector &operator=(Vector &&other) {
    if (this == &other)
      return *this;

    _size = other._size;
    _data = std::move(other._data);

    other._size = 0;

    return *this;
  }

  // Member access.

  /**
   * Get the size of the vector.
   * @return The size of the vector.
   */
  std::size_t size() const noexcept { return _size; }

  /**
   * Get the data array of the vector.
   * @return The data array of the vector.
   */
  double *data() noexcept { return _data.get(); }

  /**
   * Get the data array of the vector.
   * @return The data array of the vector.
   */
  const double *data() const noexcept { return _data.get(); }

  // Element access.

  /**
   * Get the element at index @p index without bounds checking.
   * @param index The index of the element.
   * @return The element at index @p index.
   */
  double &operator[](const std::size_t index) { return _data[index]; }

  /**
   * Get the element at index @p index without bounds checking.
   * @param index The index of the element.
   * @return The element at index @p index.
   */
  double operator[](const std::size_t index) const { return _data[index]; }

  // Equality.

  /**
   * Equality operator.
   * @param other The vector to compare with.
   * @return True if the vectors are equal, false otherwise.
   */
  bool operator==(const Vector &other) const {
    return _size == other._size &&
           std::equal(_data.get(), _data.get() + _size, other._data.get());
  }

  /**
   * Inequality operator.
   * @param other The vector to compare with.
   * @return True if the vectors are not equal, false otherwise.
   */
  bool operator!=(const Vector &other) const { return !(*this == other); }

  // Vector operations.

  /**
   * Negation operator.
   * @return The negation of the vector.
   */
  Vector operator-() const {
    Vector result(_size);
    std::transform(_data.get(), _data.get() + _size, result._data.get(),
                   [](double x) { return -x; });
    return result;
  }

  /**
   * Addition operator.
   * @param other The vector to add.
   * @return The sum of the vectors.
   * @throws std::invalid_argument if the vectors are not of the same size.
   */
  Vector operator+(const Vector &other) const {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    Vector result(_size);
    for (std::size_t i = 0; i < _size; ++i)
      result._data[i] = _data[i] + other._data[i];

    return result;
  }

  /**
   * Addition assignment operator.
   * @param other The vector to add.
   * @return A reference to this vector.
   * @throws std::invalid_argument if the vectors are not of the same size.
   */
  Vector &operator+=(const Vector &other) {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    for (std::size_t i = 0; i < _size; ++i)
      _data[i] += other._data[i];

    return *this;
  }

  /**
   * Subtraction operator.
   * @param other The vector to subtract.
   * @return The difference of the vectors.
   * @throws std::invalid_argument if the vectors are not of the same size.
   */
  Vector operator-(const Vector &other) const {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    Vector result(_size);
    for (std::size_t i = 0; i < _size; ++i)
      result._data[i] = _data[i] - other._data[i];

    return result;
  }

  /**
   * Subtraction assignment operator.
   * @param other The vector to subtract.
   * @return A reference to this vector.
   * @throws std::invalid_argument if the vectors are not of the same size.
   */
  Vector &operator-=(const Vector &other) {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    for (std::size_t i = 0; i < _size; ++i)
      _data[i] -= other._data[i];

    return *this;
  }

  /**
   * Scalar multiplication operator.
   * @param scalar The scalar to multiply by.
   * @return The scaled vector.
   */
  Vector operator*(const double scalar) const {
    Vector result(_size);
    for (std::size_t i = 0; i < _size; ++i)
      result._data[i] = _data[i] * scalar;

    return result;
  }

  /**
   * Scalar multiplication assignment operator.
   * @param scalar The scalar to multiply by.
   * @return A reference to this vector.
   */
  Vector &operator*=(const double scalar) {
    for (std::size_t i = 0; i < _size; ++i)
      _data[i] *= scalar;

    return *this;
  }

  /**
   * Scalar division operator.
   * @param scalar The scalar to divide by.
   * @return The scaled vector.
   */
  Vector operator/(const double scalar) const {
    Vector result(_size);
    for (std::size_t i = 0; i < _size; ++i)
      result._data[i] = _data[i] / scalar;

    return result;
  }

  /**
   * Scalar division assignment operator.
   * @param scalar The scalar to divide by.
   * @return A reference to this vector.
   */
  Vector &operator/=(const double scalar) {
    for (std::size_t i = 0; i < _size; ++i)
      _data[i] /= scalar;

    return *this;
  }

  /**
   * Dot product.
   * @param other The vector to dot with.
   * @return The dot product of the vectors.
   * @throws std::invalid_argument if the vectors are not of the same size.
   */
  double dot(const Vector &other) const {
    if (_size != other._size)
      throw std::invalid_argument("vector sizes are not equal");

    double result = 0;
    for (std::size_t i = 0; i < _size; ++i)
      result += _data[i] * other._data[i];

    return result;
  }

  /**
   * Get the 2-norm (Euclidean norm) of the vector.
   * @return The 2-norm of the vector.
   */
  double l2_norm() const {
    double result = 0;
    for (std::size_t i = 0; i < _size; ++i)
      result += _data[i] * _data[i];

    return std::sqrt(result);
  }
};

} // namespace diffusion_maps

#endif
