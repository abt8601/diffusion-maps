#include <cmath>
#include <functional>

#include <criterion/criterion.h>

#include "diffusion_maps/diffusion_maps.hpp"
#include "diffusion_maps/kernel.hpp"
#include "diffusion_maps/matrix.hpp"

#define PI 3.14159265358979323846

Test(diffusion_maps, diffusion_maps_helix) {
  // Data: helix
  // Dimensions after reduction: 1
  // Expected result: a straight line

  // Generate data.

  diffusion_maps::Matrix helix(1000, 3);

  for (std::size_t i = 0; i < 1000; ++i) {
    const double t = 8 * PI * (i / 999.);
    const double x = std::cos(t);
    const double y = std::sin(t);
    const double z = t / (4 * PI) - 1;
    helix(i, 0) = x;
    helix(i, 1) = y;
    helix(i, 2) = z;
  }

  // Compute diffusion maps.

  const auto result = diffusion_maps::diffusion_maps(
      helix, 1, diffusion_maps::kernel::Gaussian(50), 1);

  // Check the dimensions.

  cr_assert_eq(result.n_rows(), 1000, "Number of data points %zu is incorrect",
               result.n_rows());
  cr_assert_eq(result.n_cols(), 1, "Number of dimensions %zu is incorrect",
               result.n_cols());

  // Check that result is monotonic.

  auto cmp = result(0, 0) < result(1, 0)
                 ? std::function<bool(double, double)>(std::less<double>())
                 : std::function<bool(double, double)>(std::greater<double>());
  for (std::size_t i = 0; i < 999; ++i) {
    cr_assert(cmp(result(i, 0), result(i + 1, 0)), "Result is not monotonic");
  }
}
