/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::avx::util::shuffle;
use num_traits::MulAdd;
use std::arch::x86_64::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct AvxStoreD {
    pub(crate) v: __m256d,
}

#[derive(Copy, Clone)]
pub(crate) struct AvxMaskD {
    load_store: __m256i,
    reverse: __m256i,
    len: usize,
}

impl AvxMaskD {
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(len: usize) -> Self {
        debug_assert!(len <= 4, "AVX f64 mask length must be <= 4, got {len}");
        let last = 2 * (len as i32 - 1);
        Self {
            load_store: _mm256_cmpgt_epi64(
                _mm256_set1_epi64x(len as i64),
                _mm256_setr_epi64x(0, 1, 2, 3),
            ),
            reverse: _mm256_setr_epi32(
                last,
                last + 1,
                last - 2,
                last - 1,
                last - 4,
                last - 3,
                last - 6,
                last - 5,
            ),
            len,
        }
    }

    #[inline(always)]
    pub(crate) fn len(self) -> usize {
        self.len
    }
}

pub(crate) trait AvxLanesD: Copy {
    fn len(self) -> usize;
    fn load(self, ptr: &[f64]) -> AvxStoreD;
    fn write(self, value: AvxStoreD, ptr: &mut [f64]);
    fn reverse(self, value: AvxStoreD) -> AvxStoreD;
}

#[derive(Copy, Clone)]
pub(crate) struct AvxFullD;

#[derive(Copy, Clone)]
pub(crate) struct AvxTailD(AvxMaskD);

impl AvxTailD {
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(len: usize) -> Self {
        debug_assert!(len < 4, "full-width blocks must use AvxFullD");
        Self(AvxMaskD::new(len))
    }
}

impl AvxLanesD for AvxFullD {
    #[inline(always)]
    fn len(self) -> usize {
        4
    }

    #[inline(always)]
    fn load(self, ptr: &[f64]) -> AvxStoreD {
        unsafe { AvxStoreD::load(ptr) }
    }

    #[inline(always)]
    fn write(self, value: AvxStoreD, ptr: &mut [f64]) {
        unsafe { value.write(ptr) }
    }

    #[inline(always)]
    fn reverse(self, value: AvxStoreD) -> AvxStoreD {
        value.reverse()
    }
}

impl AvxLanesD for AvxTailD {
    #[inline(always)]
    fn len(self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn load(self, ptr: &[f64]) -> AvxStoreD {
        unsafe { AvxStoreD::load_masked(self.0, ptr) }
    }

    #[inline(always)]
    fn write(self, value: AvxStoreD, ptr: &mut [f64]) {
        unsafe { value.write_masked(self.0, ptr) }
    }

    #[inline(always)]
    fn reverse(self, value: AvxStoreD) -> AvxStoreD {
        unsafe { value.reverse_masked(self.0) }
    }
}

#[repr(C, align(32))]
pub(crate) struct AvxAlignedF64(pub(crate) [f64; 4]);

impl AvxStoreD {
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_values(p0: f64, p1: f64, p2: f64, p3: f64) -> AvxStoreD {
        AvxStoreD::raw(_mm256_setr_pd(p0, p1, p2, p3))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn to_array(self) -> [f64; 4] {
        let mut data = AvxAlignedF64([0.; 4]);
        unsafe {
            _mm256_store_pd(data.0.as_mut_ptr(), self.v);
        }
        data.0
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn dup(v: f64) -> AvxStoreD {
        AvxStoreD::raw(_mm256_set1_pd(v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn xor(&self, p0: AvxStoreD) -> Self {
        AvxStoreD::raw(_mm256_xor_pd(self.v, p0.v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load(ptr: &[f64]) -> Self {
        AvxStoreD::raw(unsafe { _mm256_loadu_pd(ptr.as_ptr()) })
    }

    /// Loads the active lanes without accessing memory beyond the mask length.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load_masked(mask: AvxMaskD, ptr: &[f64]) -> Self {
        debug_assert!(ptr.len() >= mask.len());
        Self::raw(unsafe { _mm256_maskload_pd(ptr.as_ptr(), mask.load_store) })
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load_lanes<L: AvxLanesD>(lanes: L, ptr: &[f64]) -> Self {
        lanes.load(ptr)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load3(ptr: &[f64]) -> Self {
        unsafe {
            let q0 = _mm_loadu_pd(ptr.as_ptr().cast());
            let q1 = _mm_load_sd(ptr.get_unchecked(2..).as_ptr().cast());
            AvxStoreD::raw(_mm256_setr_m128d(q0, q1))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load2(ptr: &[f64]) -> Self {
        unsafe {
            let q0 = _mm_loadu_pd(ptr.as_ptr().cast());
            AvxStoreD::raw(_mm256_castpd128_pd256(q0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load1(ptr: &[f64]) -> Self {
        unsafe {
            let q0 = _mm_load_sd(ptr.as_ptr().cast());
            AvxStoreD::raw(_mm256_castpd128_pd256(q0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write(self, ptr: &mut [f64]) {
        unsafe { _mm256_storeu_pd(ptr.as_mut_ptr(), self.v) }
    }

    /// Stores the active lanes without touching later elements.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_masked(self, mask: AvxMaskD, ptr: &mut [f64]) {
        debug_assert!(ptr.len() >= mask.len());
        unsafe { _mm256_maskstore_pd(ptr.as_mut_ptr(), mask.load_store, self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_lanes<L: AvxLanesD>(self, lanes: L, ptr: &mut [f64]) {
        lanes.write(self, ptr)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write3(self, ptr: &mut [f64]) {
        unsafe {
            _mm_storel_pd(
                ptr.get_unchecked_mut(2..).as_mut_ptr().cast(),
                _mm256_extractf128_pd::<1>(self.v),
            );
            _mm_storeu_pd(ptr.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write2(self, ptr: &mut [f64]) {
        unsafe {
            _mm_storeu_pd(ptr.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write1(self, ptr: &mut [f64]) {
        unsafe {
            _mm_store_sd(ptr.as_mut_ptr().cast(), _mm256_castpd256_pd128(self.v));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn broadcast_last(&self) -> AvxStoreD {
        AvxStoreD::raw(_mm256_permute4x64_pd::<{ shuffle(3, 3, 3, 3) }>(self.v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn last(&self) -> f64 {
        f64::from_bits(_mm256_extract_epi64::<3>(_mm256_castpd_si256(self.v)) as u64)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn prefix_differences(&self, sign: AvxStoreD) -> AvxStoreD {
        let x = self.xor(sign);

        let shift1 = _mm256_permute4x64_pd::<{ shuffle(2, 1, 0, 0) }>(x.v); // [x1, x2, x3, x0] (rotate left)
        let b = _mm256_blend_pd::<0b1110>(_mm256_setzero_pd(), shift1);
        let s1 = _mm256_add_pd(x.v, b); // [x0 - 0, x1 - x0, x2 - x1, x3 - x2]

        // Step 2: shift by 2 and subtract
        let shift2 = _mm256_permute4x64_pd::<{ shuffle(1, 0, 0, 0) }>(s1); // [s1_2, s1_3, s1_0, s1_1]
        let q = _mm256_blend_pd::<0b1100>(_mm256_setzero_pd(), shift2);
        let s2 = _mm256_add_pd(s1, q);

        AvxStoreD::raw(s2).xor(sign)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn raw(v: __m256d) -> Self {
        AvxStoreD { v }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zip(self, other: Self) -> [Self; 2] {
        let r0 = _mm256_shuffle_pd::<0b0000>(self.v, other.v);
        let r1 = _mm256_shuffle_pd::<0b1111>(self.v, other.v);
        let xy0 = _mm256_permute2f128_pd::<32>(r0, r1);
        let xy1 = _mm256_permute2f128_pd::<49>(r0, r1);
        [AvxStoreD::raw(xy0), AvxStoreD::raw(xy1)]
    }

    #[inline(always)]
    pub(crate) fn reverse(self) -> Self {
        unsafe { AvxStoreD::raw(_mm256_permute4x64_pd::<{ shuffle(0, 1, 2, 3) }>(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse_masked(self, mask: AvxMaskD) -> Self {
        // AVX2 has a variable 32-bit permutation but no variable 64-bit one,
        // so the precomputed permutation moves pairs of words together.
        Self::raw(_mm256_castsi256_pd(_mm256_permutevar8x32_epi32(
            _mm256_castpd_si256(self.v),
            mask.reverse,
        )))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse_lanes<L: AvxLanesD>(self, lanes: L) -> Self {
        lanes.reverse(self)
    }

    #[inline(always)]
    pub(crate) fn reverse3(self) -> Self {
        unsafe { AvxStoreD::raw(_mm256_permute4x64_pd::<{ shuffle(0, 0, 1, 2) }>(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse2(self) -> Self {
        let k = _mm256_castpd256_pd128(self.v);
        AvxStoreD::raw(_mm256_castpd128_pd256(_mm_shuffle_pd::<0b01>(k, k)))
    }
}

impl Add<AvxStoreD> for AvxStoreD {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_add_pd(self.v, rhs.v)) }
    }
}

impl Sub<AvxStoreD> for AvxStoreD {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_sub_pd(self.v, rhs.v)) }
    }
}

impl Mul<AvxStoreD> for f64 {
    type Output = AvxStoreD;
    #[inline(always)]
    fn mul(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_mul_pd(_mm256_set1_pd(self), rhs.v)) }
    }
}

impl Mul<f64> for AvxStoreD {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_mul_pd(self.v, _mm256_set1_pd(rhs))) }
    }
}

impl MulAssign<f64> for AvxStoreD {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        *self = unsafe { AvxStoreD::raw(_mm256_mul_pd(self.v, _mm256_set1_pd(rhs))) };
    }
}

impl Mul<AvxStoreD> for AvxStoreD {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: AvxStoreD) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_mul_pd(self.v, rhs.v)) }
    }
}

impl Neg for AvxStoreD {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_xor_pd(self.v, _mm256_set1_pd(-0.0))) }
    }
}

impl MulAdd<AvxStoreD> for AvxStoreD {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: AvxStoreD, b: Self) -> Self::Output {
        unsafe { AvxStoreD::raw(_mm256_fmadd_pd(a.v, self.v, b.v)) }
    }
}

impl AddAssign for AvxStoreD {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            *self = AvxStoreD::raw(_mm256_add_pd(self.v, rhs.v));
        }
    }
}

impl AvxStoreD {
    #[inline(always)]
    pub(crate) fn f64_mul_add(q: f64, a: AvxStoreD, b: Self) -> Self {
        unsafe { AvxStoreD::raw(_mm256_fmadd_pd(a.v, _mm256_set1_pd(q), b.v)) }
    }

    #[inline(always)]
    pub(crate) fn f64_mul_nadd(q: f64, a: AvxStoreD, b: Self) -> Self {
        unsafe { AvxStoreD::raw(_mm256_fnmadd_pd(a.v, _mm256_set1_pd(q), b.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn mul_f64_add(p0: AvxStoreD, p1: f64, p2: AvxStoreD) -> AvxStoreD {
        unsafe { AvxStoreD::raw(_mm256_fmadd_pd(p0.v, _mm256_set1_pd(p1), p2.v)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_load_reverse_store() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let src = [1., 2., 3., 4.];
        for len in 0..=4 {
            let mask = unsafe { AvxMaskD::new(len) };
            let loaded = unsafe { AvxStoreD::load_masked(mask, &src[..len]) };
            let mut loaded_array = [0.; 4];
            unsafe { _mm256_storeu_pd(loaded_array.as_mut_ptr(), loaded.v) };
            assert_eq!(&loaded_array[..len], &src[..len]);
            assert!(loaded_array[len..].iter().all(|&x| x == 0.));

            let mut dst = [99.; 6];
            unsafe {
                loaded.reverse_masked(mask).write_masked(mask, &mut dst);
            }
            assert!(
                dst[..len]
                    .iter()
                    .zip(src[..len].iter().rev())
                    .all(|(&actual, &expected)| actual == expected)
            );
            assert!(dst[len..].iter().all(|&x| x == 99.));
        }
    }
}
