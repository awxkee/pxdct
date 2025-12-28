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
use crate::avx::AvxDct2Butterfly7;
use crate::avx::dct2::mixed_radix7f::{
    dct2_radix7_cos_twiddles_avxf, dct2_radix7_rotation_twiddles_avxf,
};
use crate::avx::storef::AvxStoreF;
use crate::avx::util::fma;
use crate::dct2::MixedRadix7Sample;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::{One, Zero};

pub(crate) struct AvxDct2Butterfly49f {
    rotation_layer: [AvxStoreF; 6],
    cos_twiddles: [AvxStoreF; 6],
    bf7: AvxDct2Butterfly7<f32>,
}

impl Default for AvxDct2Butterfly49f {
    fn default() -> Self {
        AvxDct2Butterfly49f::new()
    }
}

impl AvxDct2Butterfly49f {
    pub(crate) fn new() -> AvxDct2Butterfly49f {
        let rotation_layer = unsafe { dct2_radix7_rotation_twiddles_avxf(7, 49) };

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let cos_twiddles = unsafe { dct2_radix7_cos_twiddles_avxf(7, 49) };
        AvxDct2Butterfly49f {
            rotation_layer: rotation_layer.try_into().unwrap(),
            bf7: AvxDct2Butterfly7::default(),
            cos_twiddles: cos_twiddles.try_into().unwrap(),
        }
    }
}

impl AvxDct2Butterfly49f {
    #[inline(always)]
    pub(crate) fn exec(
        &self,
        data: &mut [f32; 49],
        a_buffer: &mut [f32; 7],
        c_buffer: &mut [f32; 21],
        s_buffer: &mut [f32; 21],
    ) {
        for n in 0..7 {
            a_buffer[n] = data[n * 7 + 3];
        }

        self.bf7.exec(a_buffer);

        let q_modules = 7;

        for m in 0..3 {
            let mut sign = f32::one();
            for n in 0..7 {
                let u0 = data[7 * n + m];
                let u1 = data[7 * n + 7 - m - 1];

                c_buffer[m * 7 + n] = u0 + u1;
                s_buffer[m * 7 + n] = (u0 - u1).mulsign(sign);

                sign = -sign;
            }

            self.bf7
                .exec((&mut c_buffer[m * 7..(m + 1) * 7]).try_into().unwrap());
            self.bf7
                .exec((&mut s_buffer[m * 7..(m + 1) * 7]).try_into().unwrap());
        }

        {
            // Step 3: Recombine transformed buffers with twiddle factors

            // Handle k=0 case (DC and low frequencies)
            let qc = c_buffer[0];
            let mut c0 = qc; // Component C₀ (position 0)
            let mut c1 = qc * f32::R7_COS_EVEN2_M0; // Component C₂ (position 2, uses j=2)
            let mut c2 = qc * f32::R7_COS_EVEN2_M2; // Component C₄ (position 4, uses j=4)
            let mut c3 = qc * f32::R7_COS_EVEN2_M1; // Component C6 (position 4, uses j=6)

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * f32::R7_SIN_ODD0_M0;
            let mut s1 = s0_twiddled * f32::R7_SIN_ODD1_M0;
            let mut s2 = s0_twiddled * f32::R7_SIN_ODD2_M0;

            {
                let ci = c_buffer[q_modules];
                let si = s_buffer[q_modules];

                let ci2 = c_buffer[q_modules * 2];
                let si2 = s_buffer[q_modules * 2];

                c0 = ci + c0 + ci2;

                c1 = fma(ci, f32::R7_COS_EVEN2_M1, c1);
                c1 = fma(ci2, f32::R7_COS_EVEN2_M2, c1);

                c2 = fma(ci, f32::R7_COS_EVEN2_M0, c2);
                c2 = fma(ci2, f32::R7_COS_EVEN2_M1, c2);

                c3 = fma(ci, f32::R7_COS_EVEN2_M2, c3);
                c3 = fma(ci2, f32::R7_COS_EVEN2_M0, c3);

                s0 = fma(si, f32::R7_SIN_ODD0_M1, s0);
                s0 = fma(si2, f32::R7_SIN_ODD0_M2, s0);

                s1 = fma(si, f32::R7_SIN_ODD1_M1, s1);
                s1 = fma(si2, f32::R7_SIN_ODD1_M2, s1);

                s2 = fma(si, f32::R7_SIN_ODD2_M1, s2);
                s2 = fma(si2, f32::R7_SIN_ODD2_M2, s2);
            }

            // Write output: C₀ (pos 0), S₁ (pos q_modules), C₂ (pos 2*q_modules),
            //               S₃ (pos 3*q_modules), C₄ (pos 4*q_modules)
            let a0 = a_buffer[0];
            let dc = c0 + a0;
            data[0] = dc;

            let dc2 = c2 + a0;
            data[q_modules * 4] = dc2;
            data[q_modules * 3] = -s1;
            data[q_modules] = s0;
            let qid2 = -(c1 + a0); // negated 2j
            data[q_modules * 2] = qid2;

            let dc3 = c3 + a0;
            data[q_modules * 6] = -dc3;
            data[q_modules * 5] = s2;

            // Step 4: Handle k≥1 cases with rotation twiddles
            unsafe {
                let k = 1;
                let uk = 0;
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = self.rotation_layer[uk];
                let rotation_twiddle_im = self.rotation_layer[uk + 1];

                let c_forward = AvxStoreF::load6(&c_buffer[k..]);
                let s_forward = AvxStoreF::load6(&s_buffer[q_modules - k - 5..]).reverse6();

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = self.cos_twiddles[uk];
                let twiddle_im = self.cos_twiddles[uk + 1];

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f32::R7_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f32::R7_COS_EVEN2_M2;
                let mut dc6 = twiddled_dc * f32::R7_COS_EVEN2_M1;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f32::R7_SIN_ODD0_M0;
                let mut ds3 = twiddled_ds * f32::R7_SIN_ODD1_M0;
                let mut ds5 = twiddled_ds * f32::R7_SIN_ODD2_M0;

                {
                    let c_forward = AvxStoreF::load6(&c_buffer[q_modules + k..]);
                    let s_forward = AvxStoreF::load6(&s_buffer[q_modules * 2 - k - 5..]).reverse6();

                    let rotation_twiddle_re = self.rotation_layer[uk + 2];
                    let rotation_twiddle_im = self.rotation_layer[uk + 3];

                    let twiddle_re = self.cos_twiddles[uk + 2];
                    let twiddle_im = self.cos_twiddles[uk + 3];

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreF::f32_mul_add(f32::R7_COS_EVEN2_M1, twiddled_dc, dc2);
                    dc4 = AvxStoreF::f32_mul_add(f32::R7_COS_EVEN2_M0, twiddled_dc, dc4);
                    dc6 = AvxStoreF::f32_mul_add(f32::R7_COS_EVEN2_M2, twiddled_dc, dc6);

                    ds1 = AvxStoreF::f32_mul_add(f32::R7_SIN_ODD0_M1, twiddled_ds, ds1);
                    ds3 = AvxStoreF::f32_mul_add(f32::R7_SIN_ODD1_M1, twiddled_ds, ds3);
                    ds5 = AvxStoreF::f32_mul_add(f32::R7_SIN_ODD2_M1, twiddled_ds, ds5);
                }

                {
                    let c_forward = AvxStoreF::load6(&c_buffer[q_modules * 2 + k..]);
                    let s_forward = AvxStoreF::load6(&s_buffer[q_modules * 3 - k - 5..]).reverse6();

                    let rotation_twiddle_re = self.rotation_layer[uk + 4];
                    let rotation_twiddle_im = self.rotation_layer[uk + 5];

                    let twiddle_re = self.cos_twiddles[uk + 4];
                    let twiddle_im = self.cos_twiddles[uk + 5];

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreF::f32_mul_add(f32::R7_COS_EVEN2_M2, twiddled_dc, dc2);
                    dc4 = AvxStoreF::f32_mul_add(f32::R7_COS_EVEN2_M1, twiddled_dc, dc4);
                    dc6 = AvxStoreF::f32_mul_add(f32::R7_COS_EVEN2_M0, twiddled_dc, dc6);

                    ds1 = AvxStoreF::f32_mul_add(f32::R7_SIN_ODD0_M2, twiddled_ds, ds1);
                    ds3 = AvxStoreF::f32_mul_add(f32::R7_SIN_ODD1_M2, twiddled_ds, ds3);
                    ds5 = AvxStoreF::f32_mul_add(f32::R7_SIN_ODD2_M2, twiddled_ds, ds5);
                }

                let a0 = AvxStoreF::load6(&a_buffer[k..]);
                let dc = dc0 + a0;
                dc.write6(&mut data[k..]);

                let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                {
                    let q = dss1.reverse6();
                    q.write6(&mut data[q_modules * 2 - k - 5..]);
                }

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                {
                    dc2.write6(&mut data[q_modules * 2 + k..]);
                }

                let dss3 = AvxStoreF::f32_mul_add(2., -ds3, -dc2);
                {
                    let q = dss3.reverse6();
                    q.write6(&mut data[q_modules * 4 - k - 5..]);
                }

                dc4 += a0;

                let mdc4 = AvxStoreF::f32_mul_add(2., dc4, -dss3);
                {
                    mdc4.write6(&mut data[q_modules * 4 + k..]);
                }

                let dss5 = AvxStoreF::f32_mul_add(2., ds5, -mdc4);
                {
                    let q = dss5.reverse6();
                    q.write6(&mut data[q_modules * 6 - k - 5..]);
                }

                dc6 += a0;
                dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                {
                    dc6.write6(&mut data[q_modules * 6 + k..]);
                }
            }
        }
    }
}

impl AvxDct2Butterfly49f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(49) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [f32::zero(); 7];
        let mut c_buffer = [f32::zero(); 21];
        let mut s_buffer = [f32::zero(); 21];

        for chunk in data.chunks_exact_mut(49) {
            self.exec(
                (&mut chunk[..49]).try_into().unwrap(),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f32> for AvxDct2Butterfly49f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn length(&self) -> usize {
        49
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly_f;
    use crate::tests::naive_dct2_f32;

    gen_test_butterfly_f!(test_bf49_f32, AvxDct2Butterfly49f, 49, 1e-3, naive_dct2_f32);
}
