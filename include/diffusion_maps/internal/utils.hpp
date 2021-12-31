#ifndef DIFFUSION_MAPS_INTERNAL_UTILS_HPP
#define DIFFUSION_MAPS_INTERNAL_UTILS_HPP

namespace diffusion_maps {

namespace internal {

// Utilities for working with std::variant. Taken from
// https://en.cppreference.com/w/cpp/utility/variant/visit

template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

} // namespace internal
} // namespace diffusion_maps

#endif
