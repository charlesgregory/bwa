#ifndef ALTIVEC_SSE_H
#define ALTIVEC_SSE_H

#include <altivec.h>

typedef vector unsigned char __m128i;

static inline __m128i _mm_load_si128(const __m128i *ptr) { return vec_ld(0, ptr); }
static inline __m128i _mm_set1_epi32(int n) { return (__m128i)vec_splats(n); }
static inline void _mm_store_si128(__m128i *ptr, __m128i a) { vec_st(a, 0, ptr); }

static inline __m128i _mm_adds_epu8(__m128i a, __m128i b) { return vec_adds(a, b); }
static inline __m128i _mm_max_epu8(__m128i a, __m128i b) { return vec_max(a, b); }
static inline __m128i _mm_set1_epi8(int8_t n) { return (__m128i)vec_splats(n); }
static inline __m128i _mm_subs_epu8(__m128i a, __m128i b) { return vec_subs(a, b); }

#define M128I(a)  (vector unsigned char)((a))
#define UM128I(a) (vector unsigned char)((a))
#define S16(a)    (vector signed short)((a))
#define U16(a)    (vector unsigned short)((a))

static inline __m128i _mm_adds_epi16(__m128i a, __m128i b) { return M128I(vec_adds(S16(a), S16(b))); }
static inline __m128i _mm_cmpgt_epi16(__m128i a, __m128i b) { return UM128I(vec_cmpgt(S16(a), S16(b))); }
static inline __m128i _mm_max_epi16(__m128i a, __m128i b) { return M128I(vec_max(S16(a), S16(b))); }
static inline __m128i _mm_set1_epi16(int16_t n) { return (__m128i)(vec_splats(n)); }
static inline __m128i _mm_subs_epu16(__m128i a, __m128i b) { return UM128I(vec_subs(U16(a), U16(b))); }

/*===---- emmintrin.h - Implementation of SSE2 intrinsics on PowerPC -------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
typedef __vector double __v2df;
typedef __vector float __v4f;
typedef __vector long long __v2di;
typedef __vector unsigned long long __v2du;
typedef __vector int __v4si;
typedef __vector unsigned int __v4su;
typedef __vector short __v8hi;
typedef __vector unsigned short __v8hu;
typedef __vector signed char __v16qi;
typedef __vector unsigned char __v16qu;

static inline __m128i _mm_slli_si128(__m128i a, const int imm8) {
  __m128i result;
  const __m128i zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
  if (imm8 < 16)
#ifdef __LITTLE_ENDIAN__
    result = vec_sld(a, zeros, imm8);
#else
    result = vec_sld(zeros, a, (16 - imm8));
#endif
  else
    result = zeros;
 
  return (__m128i)result;
}
/* Intrinsic functions that require PowerISA 2.07 minimum.  */
 
/* Return a mask created from the most significant bit of each 8-bit
   element in A.  */
static inline int _mm_movemask_epi8(__m128i a) {
#ifdef _ARCH_PWR10
  return vec_extractm((__m128i)a);
#else
  __vector unsigned long long __result;
  static const __vector unsigned char __perm_mask = {
      0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
      0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00};
 
  __result = ((__vector unsigned long long)vec_vbpermq(
      (__vector unsigned char)a, (__vector unsigned char)__perm_mask));
 
#ifdef __LITTLE_ENDIAN__
  return __result[1];
#else
  return __result[0];
#endif
#endif /* !_ARCH_PWR10 */
}

static inline __m128i _mm_srli_si128(__m128i a, const int n) {
  return _mm_bsrli_si128(a, n);
}

static inline __m128i _mm_bsrli_si128(__m128i a, const int n) {
  __m128i __result;
  const __m128i zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
  if (n < 16)
#ifdef __LITTLE_ENDIAN__
    if (__builtin_constant_p(n))
      /* Would like to use Vector Shift Left Double by Octet
         Immediate here to use the immediate form and avoid
         load of n * 8 value into a separate VR.  */
      __result = vec_sld(zeros, a, (16 - n));
    else
#endif
    {
      __m128i __shift = vec_splats((unsigned char)(n * 8));
#ifdef __LITTLE_ENDIAN__
      __result = vec_sro(a, __shift);
#else
    __result = vec_slo(a, __shift);
#endif
    }
  else
    __result = zeros;
 
  return (__m128i)__result;
}

static inline __m128i _mm_max_epi16(__m128i a, __m128i __B) {
  return (__m128i)vec_max((__v8hi)a, (__v8hi)__B);
}

static inline int _mm_extract_epi16(__m128i const a, int const n) {
  return (unsigned short)((__v8hi)a)[n & 7];
}

#undef M128I
#undef UM128I
#undef S16
#undef U16

#endif
