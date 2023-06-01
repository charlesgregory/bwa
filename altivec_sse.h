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

#undef M128I
#undef UM128I
#undef S16
#undef U16

#endif
