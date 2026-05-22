/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::avx::util::{_mm_unpackhilo_ps64, shuffle};
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::x86_64::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

#[inline]
#[target_feature(enable = "avx2")]
fn _mm256_unzip_ps(a: __m256, b: __m256) -> (__m256, __m256) {
    let t0 = _mm256_shuffle_ps::<{ shuffle(3, 1, 2, 0) }>(a, a);
    let t1 = _mm256_shuffle_ps::<{ shuffle(3, 1, 2, 0) }>(b, b);

    // Now combine even and odd lanes:
    let o0 = _mm256_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(t0, t1);
    let o1 = _mm256_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(t0, t1);
    let u0 = _mm256_castpd_ps(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(
        _mm256_castps_pd(o0),
    ));
    let u1 = _mm256_castpd_ps(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(
        _mm256_castps_pd(o1),
    ));
    (u0, u1)
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct AvxStoreF {
    pub(crate) v: __m256,
}

#[repr(C, align(32))]
pub(crate) struct AvxAlignedF32(pub(crate) [f32; 8]);

impl AvxStoreF {
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zero() -> AvxStoreF {
        AvxStoreF::raw(_mm256_setzero_ps())
    }

    pub(crate) fn to_array(&self) -> [f32; 8] {
        let mut data = AvxAlignedF32([0.; 8]);
        unsafe {
            _mm256_store_ps(data.0.as_mut_ptr(), self.v);
        }
        data.0
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn dup(v: f32) -> Self {
        AvxStoreF::raw(_mm256_set1_ps(v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn swap_complex(&self) -> Self {
        Self::raw(_mm256_castpd_ps(_mm256_permute4x64_pd::<
            { shuffle(0, 1, 2, 3) },
        >(_mm256_castps_pd(self.v))))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn last(&self) -> f32 {
        f32::from_bits(_mm256_extract_epi32::<7>(_mm256_castps_si256(self.v)) as u32)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn broadcast_last(&self) -> AvxStoreF {
        let h = _mm256_extractf128_ps::<1>(self.v);
        let q = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(h, h);
        Self::raw(_mm256_setr_m128(q, q))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn prefix_differences(&self, sign: AvxStoreF) -> Self {
        let mut v = self.xor(sign).v;

        // accumulate running totals within 128-bit sub-lanes
        v = _mm256_add_ps(
            v,
            _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 4)),
        );
        v = _mm256_add_ps(
            v,
            _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 8)),
        );

        // capture the max total in low-lane and broadcast into high-lane
        let lo = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(
            _mm256_castps256_ps128(v),
            _mm256_castps256_ps128(v),
        );
        let t = _mm256_insertf128_ps::<1>(_mm256_setzero_ps(), lo);

        // shift totals, add base and low-lane max
        v = _mm256_add_ps(v, t);
        Self::raw(v).xor(sign)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load_complex(p0: &[Complex<f32>]) -> Self {
        unsafe { Self::raw(_mm256_loadu_ps(p0.as_ptr().cast())) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_values4(p0: f32, p1: f32, p2: f32, p3: f32) -> AvxStoreF {
        AvxStoreF::raw(_mm256_castps128_ps256(_mm_setr_ps(p0, p1, p2, p3)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn set_values8(
        p0: f32,
        p1: f32,
        p2: f32,
        p3: f32,
        p4: f32,
        p5: f32,
        p6: f32,
        p7: f32,
    ) -> AvxStoreF {
        AvxStoreF::raw(_mm256_setr_ps(p0, p1, p2, p3, p4, p5, p6, p7))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load(ptr: &[f32]) -> Self {
        debug_assert!(
            ptr.len() >= 8,
            "Array length must not be less than 8, but got {}",
            ptr.len()
        );
        AvxStoreF::raw(unsafe { _mm256_loadu_ps(ptr.as_ptr()) })
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn mul_by_complex_unpack_real(
        a0: AvxStoreF,
        a1: AvxStoreF,
        b0: AvxStoreF,
        b1: AvxStoreF,
    ) -> Self {
        let a_z0 = _mm256_unzip_ps(a0.v, a1.v);
        let b_z0 = _mm256_unzip_ps(b0.v, b1.v);
        let a_re0 = a_z0.0;
        let a_im0 = a_z0.1;
        let b_re0 = b_z0.0;
        let b_im0 = b_z0.1;

        let real0 = _mm256_fnmadd_ps(a_im0, b_im0, _mm256_mul_ps(a_re0, b_re0));
        Self::raw(real0)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn xor(&self, other: AvxStoreF) -> Self {
        Self::raw(_mm256_xor_ps(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn zip(self, other: Self) -> [Self; 2] {
        let r0 = _mm256_unpacklo_ps(self.v, other.v);
        let r1 = _mm256_unpackhi_ps(self.v, other.v);
        let xy0 = _mm256_permute2f128_ps::<32>(r0, r1);
        let xy1 = _mm256_permute2f128_ps::<49>(r0, r1);
        [AvxStoreF::raw(xy0), AvxStoreF::raw(xy1)]
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load7(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_loadu_ps(ptr.as_ptr().cast());
            let q2 = _mm_castsi128_ps(_mm_loadu_si64(ptr.get_unchecked(4..).as_ptr().cast()));
            let q3 = _mm_load_ss(ptr.get_unchecked(6..).as_ptr().cast());
            let q4 = _mm_insert_ps::<0x20>(q2, q3);
            AvxStoreF::raw(_mm256_setr_m128(q0, q4))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load6(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_loadu_ps(ptr.as_ptr().cast());
            let q1 = _mm_castsi128_ps(_mm_loadu_si64(ptr.get_unchecked(4..).as_ptr().cast()));
            AvxStoreF::raw(_mm256_setr_m128(q0, q1))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load5(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_loadu_ps(ptr.as_ptr().cast());
            let q1 = _mm_load_ss(ptr.get_unchecked(4..).as_ptr().cast());
            AvxStoreF::raw(_mm256_setr_m128(q0, q1))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load4(ptr: &[f32]) -> Self {
        unsafe { AvxStoreF::raw(_mm256_castps128_ps256(_mm_loadu_ps(ptr.as_ptr().cast()))) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load3(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_castsi128_ps(_mm_loadu_si64(ptr.as_ptr().cast()));
            let q1 = _mm_load_ss(ptr.get_unchecked(2..).as_ptr().cast());
            let q2 = _mm_insert_ps::<0x20>(q0, q1);
            AvxStoreF::raw(_mm256_castps128_ps256(q2))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load2(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_castsi128_ps(_mm_loadu_si64(ptr.as_ptr().cast()));
            AvxStoreF::raw(_mm256_castps128_ps256(q0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load1(ptr: &[f32]) -> Self {
        unsafe {
            let q0 = _mm_load_ss(ptr.as_ptr().cast());
            AvxStoreF::raw(_mm256_castps128_ps256(q0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn load_n<const N: usize>(ptr: &[f32]) -> Self {
        const {
            assert!(N <= 8, "N must be <= 8");
        }
        match N {
            8 => Self::load(ptr),
            7 => Self::load7(ptr),
            6 => Self::load6(ptr),
            5 => Self::load5(ptr),
            4 => Self::load4(ptr),
            3 => Self::load3(ptr),
            2 => Self::load2(ptr),
            _ => Self::load1(ptr),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse_n<const N: usize>(self) -> Self {
        const {
            assert!(N <= 8, "N must be <= 8");
        }
        match N {
            8 => self.reverse(),
            7 => self.reverse7(),
            6 => self.reverse6(),
            5 => self.reverse5(),
            4 => self.reverse4(),
            3 => self.reverse3(),
            2 => self.reverse2(),
            _ => self,
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write_n<const N: usize>(self, ptr: &mut [f32]) {
        const {
            assert!(N <= 8, "N must be <= 8");
        }
        match N {
            8 => self.write(ptr),
            7 => self.write7(ptr),
            6 => self.write6(ptr),
            5 => self.write5(ptr),
            4 => self.write4(ptr),
            3 => self.write3(ptr),
            2 => self.write2(ptr),
            _ => self.write1(ptr),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write(self, ptr: &mut [f32]) {
        unsafe { _mm256_storeu_ps(ptr.as_mut_ptr(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write4(self, ptr: &mut [f32]) {
        unsafe { _mm_storeu_ps(ptr.as_mut_ptr(), _mm256_castps256_ps128(self.v)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write6(self, ptr: &mut [f32]) {
        unsafe {
            _mm_storel_pd(
                ptr.get_unchecked_mut(4..).as_mut_ptr().cast(),
                _mm_castps_pd(_mm256_extractf128_ps::<1>(self.v)),
            );
            _mm_storeu_ps(ptr.as_mut_ptr(), _mm256_castps256_ps128(self.v));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write7(self, ptr: &mut [f32]) {
        let hi_part = _mm256_extractf128_ps::<1>(self.v);
        unsafe {
            _mm_storel_pd(
                ptr.get_unchecked_mut(4..).as_mut_ptr().cast(),
                _mm_castps_pd(hi_part),
            );
            let hilo = _mm_unpackhilo_ps64(hi_part, hi_part);
            _mm_store_ss(ptr.get_unchecked_mut(6..).as_mut_ptr(), hilo);
            _mm_storeu_ps(ptr.as_mut_ptr(), _mm256_castps256_ps128(self.v));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write5(self, ptr: &mut [f32]) {
        unsafe {
            _mm_store_ss(
                ptr.get_unchecked_mut(4..).as_mut_ptr(),
                _mm256_extractf128_ps::<1>(self.v),
            );
            _mm_storeu_ps(ptr.as_mut_ptr(), _mm256_castps256_ps128(self.v));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write3(self, ptr: &mut [f32]) {
        unsafe {
            _mm_storel_pd(
                ptr.as_mut_ptr().cast(),
                _mm_castps_pd(_mm256_castps256_ps128(self.v)),
            );
            let hilo = _mm_unpackhilo_ps64(
                _mm256_castps256_ps128(self.v),
                _mm256_castps256_ps128(self.v),
            );
            _mm_store_ss(ptr.get_unchecked_mut(2..).as_mut_ptr(), hilo);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write2(self, ptr: &mut [f32]) {
        unsafe {
            _mm_storel_pd(
                ptr.as_mut_ptr().cast(),
                _mm_castps_pd(_mm256_castps256_ps128(self.v)),
            );
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn write1(self, ptr: &mut [f32]) {
        unsafe {
            _mm_store_ss(ptr.as_mut_ptr(), _mm256_castps256_ps128(self.v));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn raw(v: __m256) -> Self {
        AvxStoreF { v }
    }

    #[inline(always)]
    pub(crate) fn reverse(self) -> Self {
        unsafe {
            AvxStoreF::raw(_mm256_permutevar8x32_ps(
                self.v,
                _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0),
            ))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse7(self) -> Self {
        AvxStoreF::raw(_mm256_permutevar8x32_ps(
            self.v,
            _mm256_setr_epi32(6, 5, 4, 3, 2, 1, 0, 0),
        ))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse6(self) -> Self {
        AvxStoreF::raw(_mm256_permutevar8x32_ps(
            self.v,
            _mm256_setr_epi32(5, 4, 3, 2, 1, 0, 0, 0),
        ))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse5(self) -> Self {
        AvxStoreF::raw(_mm256_permutevar8x32_ps(
            self.v,
            _mm256_setr_epi32(4, 3, 2, 1, 0, 0, 0, 0),
        ))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse4(self) -> Self {
        AvxStoreF::raw(_mm256_castps128_ps256(_mm_shuffle_ps::<
            { shuffle(0, 1, 2, 3) },
        >(
            _mm256_castps256_ps128(self.v),
            _mm256_castps256_ps128(self.v),
        )))
    }

    #[inline(always)]
    pub(crate) fn reverse3(self) -> Self {
        unsafe {
            AvxStoreF::raw(_mm256_castps128_ps256(_mm_shuffle_ps::<
                { shuffle(0, 0, 1, 2) },
            >(
                _mm256_castps256_ps128(self.v),
                _mm256_castps256_ps128(self.v),
            )))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn reverse2(self) -> Self {
        AvxStoreF::raw(_mm256_castps128_ps256(_mm_shuffle_ps::<
            { shuffle(0, 3, 0, 1) },
        >(
            _mm256_castps256_ps128(self.v),
            _mm256_castps256_ps128(self.v),
        )))
    }
}

impl Add<AvxStoreF> for AvxStoreF {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_add_ps(self.v, rhs.v)) }
    }
}

impl Sub<AvxStoreF> for AvxStoreF {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_sub_ps(self.v, rhs.v)) }
    }
}

impl Mul<AvxStoreF> for f32 {
    type Output = AvxStoreF;
    #[inline(always)]
    fn mul(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_mul_ps(_mm256_set1_ps(self), rhs.v)) }
    }
}

impl Mul<f32> for AvxStoreF {
    type Output = AvxStoreF;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_mul_ps(_mm256_set1_ps(rhs), self.v)) }
    }
}

impl MulAssign<f32> for AvxStoreF {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        *self = unsafe { AvxStoreF::raw(_mm256_mul_ps(_mm256_set1_ps(rhs), self.v)) };
    }
}

impl Mul<AvxStoreF> for AvxStoreF {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: AvxStoreF) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_mul_ps(self.v, rhs.v)) }
    }
}

impl Neg for AvxStoreF {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_xor_ps(self.v, _mm256_set1_ps(-0.0))) }
    }
}

impl MulAdd<AvxStoreF> for AvxStoreF {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: AvxStoreF, b: Self) -> Self::Output {
        unsafe { AvxStoreF::raw(_mm256_fmadd_ps(a.v, self.v, b.v)) }
    }
}

impl AddAssign for AvxStoreF {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl AvxStoreF {
    #[inline(always)]
    pub(crate) fn f32_mul_sub(q: f32, a: AvxStoreF, b: Self) -> Self {
        unsafe { AvxStoreF::raw(_mm256_fmsub_ps(a.v, _mm256_set1_ps(q), b.v)) }
    }

    #[inline(always)]
    pub(crate) fn f32_mul_add(q: f32, a: AvxStoreF, b: Self) -> Self {
        unsafe { AvxStoreF::raw(_mm256_fmadd_ps(a.v, _mm256_set1_ps(q), b.v)) }
    }

    #[inline(always)]
    pub(crate) fn mul_f32_add(a: AvxStoreF, b: f32, c: AvxStoreF) -> AvxStoreF {
        unsafe { AvxStoreF::raw(_mm256_fmadd_ps(a.v, _mm256_set1_ps(b), c.v)) }
    }

    #[inline(always)]
    pub(crate) fn f32_mul_nadd(q: f32, a: AvxStoreF, b: Self) -> Self {
        unsafe { AvxStoreF::raw(_mm256_fnmadd_ps(a.v, _mm256_set1_ps(q), b.v)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_array(arr: [f32; 8]) -> AvxStoreF {
        unsafe { AvxStoreF::raw(_mm256_loadu_ps(arr.as_ptr())) }
    }

    fn to_array(v: AvxStoreF) -> [f32; 8] {
        let mut out = [0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v.v) }
        out
    }

    #[test]
    fn test_reverse() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let src = [1., 2., 3., 4., 5., 6., 7., 8.];
        let v = from_array(src);

        let rev = v.reverse();
        let expected = [8., 7., 6., 5., 4., 3., 2., 1.];
        assert_eq!(to_array(rev), expected);
    }

    #[test]
    fn test_reverse7() {
        unsafe {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
            let src = [1., 2., 3., 4., 5., 6., 7., 8.];
            let v = from_array(src);

            let rev = v.reverse7();
            let expected = [7., 6., 5., 4., 3., 2., 1., 1.];
            assert_eq!(to_array(rev), expected);
        }
    }

    #[test]
    fn test_reverse6() {
        unsafe {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
            let src = [1., 2., 3., 4., 5., 6., 7., 8.];
            let v = from_array(src);

            let rev = v.reverse6();
            let expected = [6., 5., 4., 3., 2., 1., 1., 1.];
            assert_eq!(to_array(rev), expected);
        }
    }

    #[test]
    fn test_reverse5() {
        unsafe {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
            let src = [1., 2., 3., 4., 5., 6., 7., 8.];
            let v = from_array(src);

            let rev = v.reverse5();
            let expected = [5., 4., 3., 2., 1., 1., 1., 1.];
            assert_eq!(to_array(rev), expected);
        }
    }

    // For 128-bit shuffles, only lower 4 elements are affected
    #[test]
    fn test_reverse4() {
        unsafe {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
            let src = [1., 2., 3., 4., 5., 6., 7., 8.];
            let v = from_array(src);

            let rev = v.reverse4();
            let expected_lower = [4., 3., 2., 1.];
            let out = to_array(rev);
            assert_eq!(&out[0..4], &expected_lower);
        }
    }

    #[test]
    fn test_reverse3() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let src = [1., 2., 3., 4., 5., 6., 7., 8.];
        let v = from_array(src);

        let rev = v.reverse3();
        let expected_lower = [3., 2., 1.]; // depends on shuffle pattern
        let out = to_array(rev);
        assert_eq!(&out[0..3], &expected_lower);
    }

    #[test]
    fn test_reverse2() {
        unsafe {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
            let src = [1., 2., 3., 4., 5., 6., 7., 8.];
            let v = from_array(src);

            let rev = v.reverse2();
            let expected_lower = [2., 1.]; // depends on shuffle pattern
            let out = to_array(rev);
            assert_eq!(&out[0..2], &expected_lower);
        }
    }
}
