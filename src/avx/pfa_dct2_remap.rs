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
#![cfg(target_pointer_width = "64")]

use crate::avx::util::shuffle;
use crate::type2::Dct2OutputRemapper;
use crate::util::DctSample;
use std::arch::x86_64::*;

#[derive(Default)]
pub(crate) struct AvxPfaDct2Remapper {}

#[inline]
/// Pack 64bytes hi part of integers into 32 bytes using truncation
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn _mm_packhits_epi64(a: __m128i, b: __m128i) -> __m128i {
    let los = _mm_shuffle_epi32::<0b11_01_11_01>(a);
    let his = _mm_shuffle_epi32::<0b11_01_11_01>(b);
    _mm_unpacklo_epi64(los, his)
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
/// Mod function for i64
pub(crate) unsafe fn _mm_abs_epi64x(a: __m128i) -> __m128i {
    let m = _mm_srai_epi32::<31>(_mm_shuffle_epi32::<0xF5>(a));
    _mm_sub_epi64(_mm_xor_si128(a, m), m)
}

impl Dct2OutputRemapper<f32> for AvxPfaDct2Remapper {
    fn remap_output(
        &self,
        src: &[f32],
        dst: &mut [f32],
        indices: &[isize],
        gains: &[isize],
        modulation: &[isize],
        width: usize,
    ) {
        unsafe { self.remap_output_impl(src, dst, indices, gains, modulation, width) }
    }
}

impl AvxPfaDct2Remapper {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn remap_output_impl(
        &self,
        src: &[f32],
        dst: &mut [f32],
        indices: &[isize],
        gains: &[isize],
        modulation: &[isize],
        width: usize,
    ) {
        let f_indices = &indices[..width];
        let f_gains = &gains[..width];

        // first row and first column is always itself and do not need butterflies

        for (&address, &gain) in f_indices.iter().zip(f_gains.iter()) {
            let r_gain = unsafe { *src.get_unchecked(gain.unsigned_abs()) };
            // X = Gain + Modulation
            // hence if address = modulation -> Modulation = X - Gain
            // else if Gain = X - Modulation
            unsafe {
                *dst.get_unchecked_mut(address as usize) = r_gain;
            }
        }

        let q_indices = &indices[width..];
        let q_gains = &gains[width..];
        let q_modulations = &modulation[width..];

        for ((address, gain), modulation) in q_indices
            .chunks_exact(width)
            .zip(q_gains.chunks_exact(width))
            .zip(q_modulations.chunks_exact(width))
        {
            unsafe {
                let r_gain = *src.get_unchecked(gain.get_unchecked(0).unsigned_abs());
                *dst.get_unchecked_mut(*address.get_unchecked(0) as usize) = r_gain;
            }

            let q_indices = &address[1..];
            let q_gains = &gain[1..];
            let q_modulations = &modulation[1..];

            for ((address, gain), modulation) in q_indices
                .chunks_exact(4)
                .zip(q_gains.chunks_exact(4))
                .zip(q_modulations.chunks_exact(4))
            {
                unsafe {
                    let g0g1 = _mm_loadu_si128(gain.as_ptr().cast());
                    let g2g3 = _mm_loadu_si128(gain.get_unchecked(2..).as_ptr().cast());

                    let ug0g1 = _mm_abs_epi64x(g0g1);
                    let ug2g3 = _mm_abs_epi64x(g2g3);

                    let m0m1 = _mm_loadu_si128(modulation.as_ptr().cast());
                    let m2m3 = _mm_loadu_si128(modulation.get_unchecked(2..).as_ptr().cast());

                    let um0m1 = _mm_abs_epi64x(m0m1);
                    let um2m3 = _mm_abs_epi64x(m2m3);

                    let mut r_gain =
                        _mm256_i64gather_ps::<4>(src.as_ptr(), _mm256_set_m128i(ug2g3, ug0g1));
                    let mut r_modulation =
                        _mm256_i64gather_ps::<4>(src.as_ptr(), _mm256_set_m128i(um2m3, um0m1));

                    // X = Gain + Modulation
                    // hence if address = modulation -> Modulation = X - Gain
                    // else if Gain = X - Modulation
                    let qq = _mm_packhits_epi64(g0g1, g2g3);
                    let g0g1g2g3 = _mm_and_si128(qq, _mm_set1_epi32(1i32 << 31));
                    let m0m1m2m3 =
                        _mm_and_si128(_mm_packhits_epi64(m0m1, m2m3), _mm_set1_epi32(1i32 << 31));

                    r_modulation = _mm_xor_ps(r_modulation, _mm_castsi128_ps(m0m1m2m3));
                    r_gain = _mm_xor_ps(r_gain, _mm_castsi128_ps(g0g1g2g3));

                    let product = _mm_add_ps(r_modulation, r_gain);
                    _mm_store_ss(dst.get_unchecked_mut(address[0] as usize), product);
                    _mm_store_ss(
                        dst.get_unchecked_mut(address[1] as usize),
                        _mm_shuffle_ps::<{ shuffle(0, 0, 0, 1) }>(product, product),
                    );
                    _mm_store_ss(
                        dst.get_unchecked_mut(address[2] as usize),
                        _mm_shuffle_ps::<{ shuffle(0, 0, 0, 2) }>(product, product),
                    );
                    _mm_store_ss(
                        dst.get_unchecked_mut(address[3] as usize),
                        _mm_shuffle_ps::<{ shuffle(0, 0, 0, 3) }>(product, product),
                    );
                }
            }

            let q_indices = address.chunks_exact(4).remainder();
            let q_gains = gain.chunks_exact(4).remainder();
            let q_modulations = modulation.chunks_exact(4).remainder();

            for ((&address, &gain), &modulation) in q_indices
                .iter()
                .zip(q_gains.iter())
                .zip(q_modulations.iter())
            {
                let r_gain = unsafe { *src.get_unchecked(gain.unsigned_abs()) };
                let r_modulation = unsafe { *src.get_unchecked(modulation.unsigned_abs()) };

                // X = Gain + Modulation
                // hence if address = modulation -> Modulation = X - Gain
                // else if Gain = X - Modulation
                unsafe {
                    *dst.get_unchecked_mut(address as usize) =
                        r_modulation.mulsigni(modulation) + r_gain.mulsigni(gain);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::has_valid_avx;

    #[test]
    fn test_packing() {
        if !has_valid_avx() {
            return;
        }
        unsafe {
            let mut zeta0 = [0i32, 1, 2, 3];
            let mut zeta1 = [4i32, 5, 6, 7];
            let q0 = _mm_loadu_si128(zeta0.as_mut_ptr().cast());
            let q1 = _mm_loadu_si128(zeta1.as_mut_ptr().cast());
            let packed = _mm_packhits_epi64(q0, q1);
            let zeta3 = [1, 3, 5, 7];
            let mut zetaq = [-1i32, -1i32, -1i32, -1i32];
            _mm_storeu_si128(zetaq.as_mut_ptr().cast(), packed);
            assert_eq!(zetaq, zeta3);
        }
    }
}
