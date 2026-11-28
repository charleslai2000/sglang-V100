/// \file type.cuh
/// \brief Dtype trait system for CUDA scalar/packed types.
///
/// `DTypeTrait<T>` provides per-type metadata: packed type alias,
/// conversion functions (`from`), and unary/binary math operations.
/// Use `device::cast<To>(from_value)` for type conversion on device.

#pragma once
#include <sgl_kernel/utils.cuh>

#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace sglang {

template <typename T>
struct DTypeTrait {};

#define SGL_REGISTER_PACKED(SELF, PACKED) \
  using self_t = SELF;                    \
  using packed_t = PACKED

#define SGL_REGISTER_UNPACK(UNPACK, N) \
  using unpacked_t = UNPACK;           \
  static constexpr size_t kVecSize = N

#define SGL_REGISTER_FROM_DEFAULT()               \
  template <typename S>                           \
  SGL_DEVICE static self_t from(const S& value) { \
    return static_cast<self_t>(value);            \
  }                                               \
  static_assert(true)

#define SGL_REGISTER_FROM_FUNCTION(FROM, FN)     \
  SGL_DEVICE static self_t from(const FROM& x) { \
    return FN(x);                                \
  }                                              \
  static_assert(true)

#define SGL_REGISTER_UNARY_FUNCTION(NAME, FN)      \
  SGL_DEVICE static self_t NAME(const self_t& x) { \
    return FN(x);                                  \
  }                                                \
  static_assert(true)

// Also emits a `kHas_<NAME>` flag so reduction dispatch can detect the op via
// plain member SFINAE (see details::HasMax below) - hipcc mis-evaluates
// requires-expressions that probe device functions, so detection must only
// ever look at data members.
#define SGL_REGISTER_BINARY_FUNCTION(NAME, FN)                      \
  static constexpr bool kHas_##NAME = true;                         \
  SGL_DEVICE static self_t NAME(const self_t& x, const self_t& y) { \
    return FN(x, y);                                                \
  }                                                                 \
  static_assert(true)

template <std::integral T>
struct DTypeTrait<T> {
  SGL_REGISTER_PACKED(T, void);
  SGL_REGISTER_UNPACK(T, 1);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_UNARY_FUNCTION(abs, ::abs);
  SGL_REGISTER_BINARY_FUNCTION(max, ::max);
  SGL_REGISTER_BINARY_FUNCTION(min, ::min);
  static constexpr T kZeroBits = 0;
};

template <>
struct DTypeTrait<fp32_t> {
  SGL_REGISTER_PACKED(fp32_t, fp32x2_t);
  SGL_REGISTER_UNPACK(fp32_t, 1);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp16_t, __half2float);
  SGL_REGISTER_FROM_FUNCTION(bf16_t, __bfloat162float);
  SGL_REGISTER_UNARY_FUNCTION(abs, fabsf);
  SGL_REGISTER_UNARY_FUNCTION(sqrt, sqrtf);
  SGL_REGISTER_UNARY_FUNCTION(rsqrt, rsqrtf);
  SGL_REGISTER_UNARY_FUNCTION(exp, expf);
  SGL_REGISTER_UNARY_FUNCTION(sin, sinf);
  SGL_REGISTER_UNARY_FUNCTION(cos, cosf);
  SGL_REGISTER_BINARY_FUNCTION(max, fmaxf);
  SGL_REGISTER_BINARY_FUNCTION(min, fminf);
  static constexpr float kFloatMax = std::numeric_limits<float>::max();
  static constexpr uint32_t kZeroBits = 0x00000000;
};

// ---------------------------------------------------------------------------
// SM70 (Tesla V100) bf16 intrinsic fallbacks.
//
// CUDA only exposes the bf16 arithmetic intrinsics (__bfloat1622float2,
// __float22bfloat162_rn, __habs/__hmax/__hmin on __nv_bfloat16, ...) when
// targeting sm80+. The `__bfloat162float` / `__float2bfloat16_rn` conversions
// exist for every architecture, so the wrappers below implement the same
// operations on top of them and forward to the intrinsics on sm80+.
// ---------------------------------------------------------------------------
#if !defined(USE_ROCM)

SGL_DEVICE float2 sgl_bf16x2_to_fp32x2(const bf16x2_t& v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
#else
  return __bfloat1622float2(v);
#endif
}

SGL_DEVICE bf16x2_t sgl_fp32x2_to_bf16x2(const float2& v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  bf16x2_t out;
  out.x = __float2bfloat16_rn(v.x);
  out.y = __float2bfloat16_rn(v.y);
  return out;
#else
  return __float22bfloat162_rn(v);
#endif
}

SGL_DEVICE bf16_t sgl_bf16_abs(const bf16_t& v) {
  return __float2bfloat16_rn(fabsf(__bfloat162float(v)));
}

SGL_DEVICE bf16_t sgl_bf16_max(const bf16_t& a, const bf16_t& b) {
  return __float2bfloat16_rn(fmaxf(__bfloat162float(a), __bfloat162float(b)));
}

SGL_DEVICE bf16_t sgl_bf16_min(const bf16_t& a, const bf16_t& b) {
  return __float2bfloat16_rn(fminf(__bfloat162float(a), __bfloat162float(b)));
}

SGL_DEVICE bf16x2_t sgl_bf16x2_abs(const bf16x2_t& v) {
  return sgl_fp32x2_to_bf16x2(make_float2(
      fabsf(__bfloat162float(v.x)), fabsf(__bfloat162float(v.y))));
}

SGL_DEVICE bf16x2_t sgl_bf16x2_add(const bf16x2_t& a, const bf16x2_t& b) {
  return sgl_fp32x2_to_bf16x2(make_float2(
      __bfloat162float(a.x) + __bfloat162float(b.x),
      __bfloat162float(a.y) + __bfloat162float(b.y)));
}

SGL_DEVICE bf16x2_t sgl_bf16x2_max(const bf16x2_t& a, const bf16x2_t& b) {
  return sgl_fp32x2_to_bf16x2(make_float2(
      fmaxf(__bfloat162float(a.x), __bfloat162float(b.x)),
      fmaxf(__bfloat162float(a.y), __bfloat162float(b.y))));
}

SGL_DEVICE bf16x2_t sgl_bf16x2_min(const bf16x2_t& a, const bf16x2_t& b) {
  return sgl_fp32x2_to_bf16x2(make_float2(
      fminf(__bfloat162float(a.x), __bfloat162float(b.x)),
      fminf(__bfloat162float(a.y), __bfloat162float(b.y))));
}

#endif  // !USE_ROCM

template <>
struct DTypeTrait<fp32x2_t> {
  SGL_REGISTER_PACKED(fp32x2_t, fp32x4_t);
  SGL_REGISTER_UNPACK(fp32_t, 2);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp16x2_t, __half22float2);
  SGL_REGISTER_FROM_FUNCTION(bf16x2_t, sgl_bf16x2_to_fp32x2);
};

template <>
struct DTypeTrait<fp32x4_t> {
  SGL_REGISTER_PACKED(fp32x4_t, void);
  SGL_REGISTER_UNPACK(fp32_t, 4);
  SGL_REGISTER_FROM_DEFAULT();
};

template <>
struct DTypeTrait<fp16_t> {
  SGL_REGISTER_PACKED(fp16_t, fp16x2_t);
  SGL_REGISTER_UNPACK(fp16_t, 1);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp32_t, __float2half_rn);
  SGL_REGISTER_UNARY_FUNCTION(abs, __habs);
  SGL_REGISTER_BINARY_FUNCTION(max, __hmax);
  SGL_REGISTER_BINARY_FUNCTION(min, __hmin);
  // CUDA fp16 max clamp value
  static constexpr float kFloatMax = 65504.0f;
  static constexpr uint16_t kZeroBits = 0x0000;
};

template <>
struct DTypeTrait<fp16x2_t> {
  SGL_REGISTER_PACKED(fp16x2_t, void);
  SGL_REGISTER_UNPACK(fp16_t, 2);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22half2_rn);
  SGL_REGISTER_UNARY_FUNCTION(abs, __habs2);
#ifndef USE_ROCM
  SGL_REGISTER_BINARY_FUNCTION(add, __hadd2);
  SGL_REGISTER_BINARY_FUNCTION(max, __hmax2);
  SGL_REGISTER_BINARY_FUNCTION(min, __hmin2);
#else
  // HIP only provides __hmax2/__hmin2 for __hip_bfloat162, not __half2.
  // No `add` registered on HIP (packed SUM falls back to lane-wise scalar).
  static constexpr bool kHas_max = true;
  static constexpr bool kHas_min = true;
  SGL_DEVICE static self_t max(const self_t& x, const self_t& y) {
    return self_t{__hmax(x.x, y.x), __hmax(x.y, y.y)};
  }
  SGL_DEVICE static self_t min(const self_t& x, const self_t& y) {
    return self_t{__hmin(x.x, y.x), __hmin(x.y, y.y)};
  }
#endif
};

template <>
struct DTypeTrait<bf16_t> {
  SGL_REGISTER_PACKED(bf16_t, bf16x2_t);
  SGL_REGISTER_UNPACK(bf16_t, 1);
  SGL_REGISTER_FROM_DEFAULT();
#ifndef USE_ROCM
  SGL_REGISTER_FROM_FUNCTION(fp32_t, __float2bfloat16_rn);
#else
  // HIP has no _rn-suffixed variant; __float2bfloat16 rounds to nearest.
  SGL_REGISTER_FROM_FUNCTION(fp32_t, __float2bfloat16);
#endif
  SGL_REGISTER_UNARY_FUNCTION(abs, sgl_bf16_abs);
  SGL_REGISTER_BINARY_FUNCTION(max, sgl_bf16_max);
  SGL_REGISTER_BINARY_FUNCTION(min, sgl_bf16_min);
  // CUDA bf16 max clamp value
  static constexpr float kFloatMax = 3.38953139e38f;
  static constexpr uint16_t kZeroBits = 0x0000;
};

template <>
struct DTypeTrait<bf16x2_t> {
  SGL_REGISTER_PACKED(bf16x2_t, void);
  SGL_REGISTER_UNPACK(bf16_t, 2);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp32x2_t, sgl_fp32x2_to_bf16x2);
  SGL_REGISTER_UNARY_FUNCTION(abs, sgl_bf16x2_abs);
#ifndef USE_ROCM
  // No `add` on HIP: bf162 __hadd2 is unverified there (packed SUM falls
  // back to lane-wise scalar).
  SGL_REGISTER_BINARY_FUNCTION(add, sgl_bf16x2_add);
#endif
  SGL_REGISTER_BINARY_FUNCTION(max, sgl_bf16x2_max);
  SGL_REGISTER_BINARY_FUNCTION(min, sgl_bf16x2_min);
};

#ifndef USE_ROCM
template <>
struct DTypeTrait<fp8_e4m3_t> {
  SGL_REGISTER_PACKED(fp8_e4m3_t, fp8x2_e4m3_t);
  SGL_REGISTER_UNPACK(fp8_e4m3_t, 1);
  SGL_REGISTER_FROM_DEFAULT();
  // NOTE: CUDA fp8 support explicit cast (i.e. use default from is ok)

  static constexpr float kFloatMax = 448.0f;  // CUDA fp8 max clamp value
  static constexpr uint8_t kZeroBits = 0x00;
};

template <>
struct DTypeTrait<fp8x2_e4m3_t> {
  SGL_REGISTER_PACKED(fp8x2_e4m3_t, fp8x4_e4m3_t);
  SGL_REGISTER_UNPACK(fp8_e4m3_t, 2);
  SGL_REGISTER_FROM_DEFAULT();
  // NOTE: CUDA fp8 support explicit cast (i.e. use default from is ok)
};

template <>
struct DTypeTrait<fp8x4_e4m3_t> {
  SGL_REGISTER_PACKED(fp8x4_e4m3_t, void);
  SGL_REGISTER_UNPACK(fp8_e4m3_t, 4);
  SGL_REGISTER_FROM_DEFAULT();
  // NOTE: CUDA fp8 support explicit cast (i.e. use default from is ok)
};
#endif

#undef SGL_REGISTER_PACKED
#undef SGL_REGISTER_UNPACK
#undef SGL_REGISTER_FROM_DEFAULT
#undef SGL_REGISTER_FROM_FUNCTION
#undef SGL_REGISTER_UNARY_FUNCTION
#undef SGL_REGISTER_BINARY_FUNCTION

/// \brief Alias: the packed (x2) type for `T`.
template <typename T>
using packed_t = typename DTypeTrait<T>::packed_t;

namespace device {

/**
 * \brief Cast a value from type `From` to type `To` on device.
 *
 * Dispatches through `DTypeTrait<To>::from()`, which uses the appropriate
 * CUDA intrinsic (e.g. `__half2float`, `__float22half2_rn`).
 */
template <typename To, typename From>
SGL_DEVICE To cast(const From& value) {
  return DTypeTrait<To>::from(value);
}

/**
 * \brief View a packed value as an array of its `unpacked_t` elements.
 *
 * Returns a reference to `value` reinterpreted as `unpacked_t[kVecSize]`,
 * so element writes propagate back to the original packed value.
 * Constness of `value` is preserved.
 */
template <typename T>
SGL_DEVICE auto& unpack(T& value) {
  using Trait = DTypeTrait<std::remove_const_t<T>>;
  using U = typename Trait::unpacked_t;
  constexpr size_t kVecSize = Trait::kVecSize;
  static_assert(sizeof(T) == sizeof(U) * kVecSize, "packed type must be layout-compatible");
  using A = std::conditional_t<std::is_const_v<T>, const U, U>;
  return reinterpret_cast<A(&)[kVecSize]>(value);
}

enum class ReductionOp : uint8_t { SUM, MAX, MIN };

template <ReductionOp Op, typename T>
struct ReductionTrait {};

namespace details {

// Op detection via the `kHas_*` data members emitted by
// SGL_REGISTER_BINARY_FUNCTION. Deliberately classic void_t member SFINAE:
// hipcc mis-evaluates requires-expressions in device instantiation contexts
// (observed: even `requires { a + b; }` on float came out false), so detection
// must never probe function-call expressions.
template <typename T, typename = void>
struct HasAdd : std::false_type {};
template <typename T>
struct HasAdd<T, std::void_t<decltype(DTypeTrait<T>::kHas_add)>> : std::true_type {};

template <typename T, typename = void>
struct HasMax : std::false_type {};
template <typename T>
struct HasMax<T, std::void_t<decltype(DTypeTrait<T>::kHas_max)>> : std::true_type {};

template <typename T, typename = void>
struct HasMin : std::false_type {};
template <typename T>
struct HasMin<T, std::void_t<decltype(DTypeTrait<T>::kHas_min)>> : std::true_type {};

template <ReductionOp Op, typename T>
SGL_DEVICE T reduce_recursive(const T& x, const T& y) {
  using U = typename DTypeTrait<T>::unpacked_t;
  constexpr size_t kVecSize = DTypeTrait<T>::kVecSize;
  static_assert(kVecSize > 1, "unsupported scalar type for reduction");
  using Trait = ReductionTrait<Op, U>;
  auto& x_unpacked = device::unpack(x);
  auto& y_unpacked = device::unpack(y);
  T result{};
  auto& z_unpacked = device::unpack(result);
#pragma unroll
  for (size_t i = 0; i < kVecSize; ++i) {
    z_unpacked[i] = Trait::reduce(x_unpacked[i], y_unpacked[i]);
  }
  return result;
}

}  // namespace details

// Dispatch rules, chosen so correctness never depends on detection:
// scalars (kVecSize == 1) call the trait member / operator directly - a
// missing op is a clear compile error at the call line; packed types use the
// native op when the trait registered one and fall back to lane-wise
// recursion otherwise (worst case for a mis-detecting compiler is a slightly
// slower but still correct lane-wise path).
template <typename T>
struct ReductionTrait<ReductionOp::SUM, T> {
  SGL_DEVICE static T reduce(const T& x, const T& y) {
    if constexpr (details::HasAdd<T>::value) {
      return DTypeTrait<T>::add(x, y);
    } else if constexpr (DTypeTrait<T>::kVecSize == 1) {
      return static_cast<T>(x + y);
    } else {
      return details::reduce_recursive<ReductionOp::SUM>(x, y);
    }
  }
};

template <typename T>
struct ReductionTrait<ReductionOp::MAX, T> {
  SGL_DEVICE static T reduce(const T& x, const T& y) {
    if constexpr (DTypeTrait<T>::kVecSize == 1) {
      return DTypeTrait<T>::max(x, y);
    } else if constexpr (details::HasMax<T>::value) {
      return DTypeTrait<T>::max(x, y);
    } else {
      return details::reduce_recursive<ReductionOp::MAX>(x, y);
    }
  }
};

template <typename T>
struct ReductionTrait<ReductionOp::MIN, T> {
  SGL_DEVICE static T reduce(const T& x, const T& y) {
    if constexpr (DTypeTrait<T>::kVecSize == 1) {
      return DTypeTrait<T>::min(x, y);
    } else if constexpr (details::HasMin<T>::value) {
      return DTypeTrait<T>::min(x, y);
    } else {
      return details::reduce_recursive<ReductionOp::MIN>(x, y);
    }
  }
};

}  // namespace device

// ---------------------------------------------------------------------------
// FP8 max clamp value - platform-dependent
//   CUDA (e4m3fn):      448.0f
//   AMD FNUZ (e4m3fnuz): 224.0f
//   AMD E4M3 (e4m3fn):  448.0f
// ---------------------------------------------------------------------------
#ifndef USE_ROCM
inline constexpr float kFP8E4M3Max = 448.0f;
#else  // USE_ROCM
#if HIP_FP8_TYPE_FNUZ
inline constexpr float kFP8E4M3Max = 224.0f;
#else   // HIP_FP8_TYPE_E4M3
inline constexpr float kFP8E4M3Max = 448.0f;
#endif  // HIP_FP8_TYPE_FNUZ
#endif  // USE_ROCM

}  // namespace sglang
