/// \file
///
/// \brief Diffusion kernels.

#ifndef DIFFUSION_MAPS_KERNEL_HPP
#define DIFFUSION_MAPS_KERNEL_HPP

#include "diffusion_maps/vector.hpp"

namespace diffusion_maps {

/// Diffusion kernels.
namespace kernel {

/// Gaussian kernel.
class Gaussian {
public:
  /// Kernel parameter γ = 1 / 2σ².
  double gamma;

  /// \brief Constructs a Gaussian kernel with the given parameter γ.
  ///
  /// \param[in] gamma Kernel parameter γ = 1 / 2σ².
  Gaussian(const double gamma) : gamma(gamma) {}

  /// \brief Constructs a Gaussian kernel with the given parameter γ.
  ///
  /// \param[in] gamma Kernel parameter γ = 1 / 2σ².
  static Gaussian with_gamma(const double gamma) { return Gaussian(gamma); }

  /// \brief Constructs a Gaussian kernel with the given σ.
  ///
  /// \param[in] sigma Kernel parameter σ.
  static Gaussian with_sigma(const double sigma) {
    return Gaussian(1.0 / (2.0 * sigma * sigma));
  }

  /// Evaluates the kernel function.
  double operator()(const Vector &x, const Vector &y) const {
    return std::exp(-gamma * (x - y).sq_l2_norm());
  }
};

} // namespace kernel

} // namespace diffusion_maps

#endif
