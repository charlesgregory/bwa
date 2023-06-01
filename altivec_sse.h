#ifndef ALTIVEC_SSE_H
#define ALTIVEC_SSE_H

#include <altivec.h>

typedef vector unsigned char __m128i;

/* Much of the below code was derived from: https://clang.llvm.org/doxygen/ppc__wrappers_2emmintrin_8h_source.html
 * While this code is not part of the LLVM project, it is derived under Apache-2.0 WITH LLVM-exception license
 * Below is the original comment from that file.
 *
 *===---- emmintrin.h - Implementation of SSE2 intrinsics on PowerPC -------===
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

static inline __m128i _mm_set_epi32(int q3, int q2, int q1, int q0) {
    return (__m128i)(__v4si){q0, q1, q2, q3};
}

static inline __m128i _mm_set_epi8(
    int8_t q15, int8_t q14, int8_t q13, int8_t q12, 
    int8_t q11, int8_t q10, int8_t q9, int8_t q8, 
    int8_t q7, int8_t q6, int8_t q5, int8_t q4, 
    int8_t q3, int8_t q2, int8_t q1, int8_t q0
    ) {
    return (__m128i)(__v16qi){q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15};
}

static inline __m128i _mm_set_epi16(int16_t q7, int16_t q6, int16_t q5, int16_t q4, int16_t q3, int16_t q2, int16_t q1, int16_t q0) {
    return (__m128i)(__v8hi){q0, q1, q2, q3, q4, q5, q6, q7};
}

static inline __m128i _mm_set1_epi16(int16_t n) { return _mm_set_epi16(n, n, n, n, n, n, n, n); }

static inline __m128i _mm_load_si128(const __m128i *ptr) { return *ptr; }
static inline __m128i _mm_set1_epi32(int n) { return _mm_set_epi32(n, n, n, n); }
static inline void _mm_store_si128(__m128i *ptr, __m128i a) { vec_st(a, 0, ptr); }

static inline __m128i _mm_adds_epu8(__m128i a, __m128i b) { return vec_adds(a, b); }
static inline __m128i _mm_max_epu8(__m128i a, __m128i b) { return vec_max(a, b); }
static inline __m128i _mm_set1_epi8(int8_t n) { return _mm_set_epi8(n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n); }
static inline __m128i _mm_subs_epu8(__m128i a, __m128i b) { return vec_subs(a, b); }

static inline __m128i _mm_adds_epi16(__m128i a, __m128i b) { return (__m128i)(vec_adds((__v8hi)a, (__v8hi)b)); }
static inline __m128i _mm_cmpgt_epi16(__m128i a, __m128i b) { return (__m128i)(vec_cmpgt((__v8hi)a, (__v8hi)b)); }
static inline __m128i _mm_subs_epu16(__m128i a, __m128i b) { return (__m128i)(vec_subs((__v8hu)(a), (__v8hu)(b))); }



static inline __m128i _mm_slli_si128(__m128i a, const int imm8) {
  __v16qu result;
  const __v16qu zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
  if (imm8 < 16)
#ifdef __LITTLE_ENDIAN__
    result = vec_sld((__v16qu)a, zeros, imm8);
#else
    result = vec_sld(zeros, (__v16qu)a, (16 - imm8));
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
  return vec_extractm((__v16qu)a);
#else
  __vector unsigned long long result;
  static const __vector unsigned char perm_mask = {
      0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
      0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00};
 
  result = ((__vector unsigned long long)vec_vbpermq(
      (__vector unsigned char)a, (__vector unsigned char)perm_mask));
 
#ifdef __LITTLE_ENDIAN__
  return result[1];
#else
  return result[0];
#endif
#endif /* !_ARCH_PWR10 */
}

static inline __m128i _mm_bsrli_si128(__m128i a, const int n) {
  __m128i result;
  const __m128i zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
  if (n < 16)
#ifdef __LITTLE_ENDIAN__
    if (__builtin_constant_p(n))
      /* Would like to use Vector Shift Left Double by Octet
         Immediate here to use the immediate form and avoid
         load of n * 8 value into a separate VR.  */
      result = vec_sld(zeros, a, (16 - n));
    else
#endif
    {
      __m128i __shift = vec_splats((unsigned char)(n * 8));
#ifdef __LITTLE_ENDIAN__
      result = vec_sro(a, __shift);
#else
    result = vec_slo(a, __shift);
#endif
    }
  else
    result = zeros;
 
  return (__m128i)result;
}


static inline __m128i _mm_srli_si128(__m128i a, const int n) {
  return _mm_bsrli_si128(a, n);
}

static inline __m128i _mm_max_epi16(__m128i a, __m128i b) {
  return (__m128i)vec_max((__v8hi)a, (__v8hi)b);
}

static inline int _mm_extract_epi16(__m128i const a, int const n) {
  return (unsigned short)((__v8hi)a)[n & 7];
}

static inline __m128i _mm_cmpeq_epi8(__m128i a, __m128i b) {
  return (__m128i)vec_cmpeq((__v16qi)a, (__v16qi)b);
}

#endif
