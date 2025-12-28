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
use crate::dct2::Dct2OutputRemapper;
use crate::util::DctSample;
use std::arch::aarch64::*;

#[derive(Default)]
pub(crate) struct NeonPfaDct2Remapper {}

impl Dct2OutputRemapper<f32> for NeonPfaDct2Remapper {
    fn remap_output(
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
                    let g0g1 = vld1q_s64(gain.as_ptr().cast());
                    let g2g3 = vld1q_s64(gain.get_unchecked(2..).as_ptr().cast());

                    let ug0g1 = vabsq_s64(g0g1);
                    let ug2g3 = vabsq_s64(g2g3);

                    let m0m1 = vld1q_s64(modulation.as_ptr().cast());
                    let m2m3 = vld1q_s64(modulation.get_unchecked(2..).as_ptr().cast());

                    let um0m1 = vabsq_s64(m0m1);
                    let um2m3 = vabsq_s64(m2m3);

                    let mut r_gain = vld1q_lane_f32::<0>(
                        src.get_unchecked(vgetq_lane_s64::<0>(ug0g1) as usize),
                        vdupq_n_f32(0.),
                    );
                    let mut r_modulation = vld1q_lane_f32::<0>(
                        src.get_unchecked(vgetq_lane_s64::<0>(um0m1) as usize),
                        vdupq_n_f32(0.),
                    );
                    r_gain = vld1q_lane_f32::<1>(
                        src.get_unchecked(vgetq_lane_s64::<1>(ug0g1) as usize),
                        r_gain,
                    );
                    r_modulation = vld1q_lane_f32::<1>(
                        src.get_unchecked(vgetq_lane_s64::<1>(um0m1) as usize),
                        r_modulation,
                    );
                    r_gain = vld1q_lane_f32::<2>(
                        src.get_unchecked(vgetq_lane_s64::<0>(ug2g3) as usize),
                        r_gain,
                    );
                    r_modulation = vld1q_lane_f32::<2>(
                        src.get_unchecked(vgetq_lane_s64::<0>(um2m3) as usize),
                        r_modulation,
                    );
                    r_gain = vld1q_lane_f32::<3>(
                        src.get_unchecked(vgetq_lane_s64::<1>(ug2g3) as usize),
                        r_gain,
                    );
                    r_modulation = vld1q_lane_f32::<3>(
                        src.get_unchecked(vgetq_lane_s64::<1>(um2m3) as usize),
                        r_modulation,
                    );

                    // X = Gain + Modulation
                    // hence if address = modulation -> Modulation = X - Gain
                    // else if Gain = X - Modulation
                    let qq = vcombine_s32(vqmovn_s64(g0g1), vqmovn_s64(g2g3));
                    let g0g1g2g3 = vandq_s32(qq, vdupq_n_s32(1i32 << 31));
                    let m0m1m2m3 = vandq_s32(
                        vcombine_s32(vqmovn_s64(m0m1), vqmovn_s64(m2m3)),
                        vdupq_n_s32(1i32 << 31),
                    );

                    r_modulation = vreinterpretq_f32_s32(veorq_s32(
                        vreinterpretq_s32_f32(r_modulation),
                        m0m1m2m3,
                    ));
                    r_gain =
                        vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(r_gain), g0g1g2g3));

                    let product = vaddq_f32(r_modulation, r_gain);
                    *dst.get_unchecked_mut(address[0] as usize) = vgetq_lane_f32::<0>(product);
                    *dst.get_unchecked_mut(address[1] as usize) = vgetq_lane_f32::<1>(product);
                    *dst.get_unchecked_mut(address[2] as usize) = vgetq_lane_f32::<2>(product);
                    *dst.get_unchecked_mut(address[3] as usize) = vgetq_lane_f32::<3>(product);
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
