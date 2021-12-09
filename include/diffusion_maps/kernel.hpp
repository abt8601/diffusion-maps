#ifndef DIFFUSION_MAPS_KERNEL_HPP
#define DIFFUSION_MAPS_KERNEL_HPP

#include "diffusion_maps/vector.hpp"

namespace diffusion_maps {

namespace kernel {

/**
 * Gaussian kernel.
 */
class Gaussian {
public:
  /**
   * Kernel parameter γ = 1 / 2σ².
   */
  double gamma;

  /**
   * Constructor.
   */
  Gaussian(double gamma) : gamma(gamma) {}

  /**
   * Evaluate the kernel.
   */
  double operator()(const Vector &x, const Vector &y) const {
    return std::exp(-gamma * (x - y).sq_l2_norm());
  }
};

} // namespace kernel

} // namespace diffusion_maps

#endif
