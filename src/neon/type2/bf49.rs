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
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::mla::fmla;
use crate::neon::type2::mixed_radix3::{
    dct2_radix_n_cos_twiddles_neon, dct2_radix_n_rotation_twiddles_neon,
};
use crate::neon::util::NeonStoreF;
use crate::type2::MixedRadix7Sample;
use crate::type2::prime_butterflies::Dct2Butterfly7;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::{One, Zero};

pub(crate) struct NeonDct2Butterfly49f {
    rotation_layer: [NeonStoreF; 12],
    cos_twiddles: [NeonStoreF; 12],
    bf7: Dct2Butterfly7<f32>,
}

impl Default for NeonDct2Butterfly49f {
    fn default() -> Self {
        NeonDct2Butterfly49f::new()
    }
}

impl NeonDct2Butterfly49f {
    pub(crate) fn new() -> NeonDct2Butterfly49f {
        let rotation_layer = dct2_radix_n_rotation_twiddles_neon(7, 7, 49);

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let cos_twiddles = dct2_radix_n_cos_twiddles_neon(7, 7, 49);
        NeonDct2Butterfly49f {
            rotation_layer: rotation_layer.try_into().unwrap(),
            bf7: Dct2Butterfly7::default(),
            cos_twiddles: cos_twiddles.try_into().unwrap(),
        }
    }
}

impl NeonDct2Butterfly49f {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        a_buffer: &mut [f32; 7],
        c_buffer: &mut [f32; 21],
        s_buffer: &mut [f32; 21],
    ) {
        for n in 0..7 {
            a_buffer[n] = data[n * 7 + 3];
        }

        self.bf7.exec(&mut InPlaceStore::new(a_buffer));

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
                .exec(&mut InPlaceStore::new(&mut c_buffer[m * 7..(m + 1) * 7]));
            self.bf7
                .exec(&mut InPlaceStore::new(&mut s_buffer[m * 7..(m + 1) * 7]));
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

                c1 = fmla(ci, f32::R7_COS_EVEN2_M1, c1);
                c1 = fmla(ci2, f32::R7_COS_EVEN2_M2, c1);

                c2 = fmla(ci, f32::R7_COS_EVEN2_M0, c2);
                c2 = fmla(ci2, f32::R7_COS_EVEN2_M1, c2);

                c3 = fmla(ci, f32::R7_COS_EVEN2_M2, c3);
                c3 = fmla(ci2, f32::R7_COS_EVEN2_M0, c3);

                s0 = fmla(si, f32::R7_SIN_ODD0_M1, s0);
                s0 = fmla(si2, f32::R7_SIN_ODD0_M2, s0);

                s1 = fmla(si, f32::R7_SIN_ODD1_M1, s1);
                s1 = fmla(si2, f32::R7_SIN_ODD1_M2, s1);

                s2 = fmla(si, f32::R7_SIN_ODD2_M1, s2);
                s2 = fmla(si2, f32::R7_SIN_ODD2_M2, s2);
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
            {
                let k = 1;
                let uk = 0;
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = self.rotation_layer[uk];
                let rotation_twiddle_im = self.rotation_layer[uk + 1];

                let c_forward = NeonStoreF::load(&c_buffer[k..]);
                let s_forward = NeonStoreF::load(&s_buffer[q_modules - k - 3..]).reverse();

                let rotated_dc = fmla(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = self.cos_twiddles[uk];
                let twiddle_im = self.cos_twiddles[uk + 1];

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f32::R7_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f32::R7_COS_EVEN2_M2;
                let mut dc6 = twiddled_dc * f32::R7_COS_EVEN2_M1;

                let rotated_ds = fmla(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f32::R7_SIN_ODD0_M0;
                let mut ds3 = twiddled_ds * f32::R7_SIN_ODD1_M0;
                let mut ds5 = twiddled_ds * f32::R7_SIN_ODD2_M0;

                {
                    let c_forward = NeonStoreF::load(&c_buffer[q_modules + k..]);
                    let s_forward = NeonStoreF::load(&s_buffer[q_modules * 2 - k - 3..]).reverse();

                    let rotation_twiddle_re = self.rotation_layer[uk + 2];
                    let rotation_twiddle_im = self.rotation_layer[uk + 3];

                    let twiddle_re = self.cos_twiddles[uk + 2];
                    let twiddle_im = self.cos_twiddles[uk + 3];

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M1, twiddled_dc, dc2);
                    dc4 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M0, twiddled_dc, dc4);
                    dc6 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M2, twiddled_dc, dc6);

                    ds1 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD0_M1, twiddled_ds, ds1);
                    ds3 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD1_M1, twiddled_ds, ds3);
                    ds5 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD2_M1, twiddled_ds, ds5);
                }

                {
                    let c_forward = NeonStoreF::load(&c_buffer[q_modules * 2 + k..]);
                    let s_forward = NeonStoreF::load(&s_buffer[q_modules * 3 - k - 3..]).reverse();

                    let rotation_twiddle_re = self.rotation_layer[uk + 4];
                    let rotation_twiddle_im = self.rotation_layer[uk + 5];

                    let twiddle_re = self.cos_twiddles[uk + 4];
                    let twiddle_im = self.cos_twiddles[uk + 5];

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M2, twiddled_dc, dc2);
                    dc4 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M1, twiddled_dc, dc4);
                    dc6 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M0, twiddled_dc, dc6);

                    ds1 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD0_M2, twiddled_ds, ds1);
                    ds3 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD1_M2, twiddled_ds, ds3);
                    ds5 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD2_M2, twiddled_ds, ds5);
                }

                let a0 = NeonStoreF::load(&a_buffer[k..]);
                let dc = dc0 + a0;
                dc.write(data.slice_from_mut(k..));

                let dss1 = NeonStoreF::f32_mul_add(2., ds1, -dc);
                {
                    let q = dss1.reverse();
                    q.write(data.slice_from_mut(q_modules * 2 - k - 3..));
                }

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = NeonStoreF::f32_mul_add(2., dc2, -dss1);
                {
                    dc2.write(data.slice_from_mut(q_modules * 2 + k..));
                }

                let dss3 = NeonStoreF::f32_mul_add(2., -ds3, -dc2);
                {
                    let q = dss3.reverse();
                    q.write(data.slice_from_mut(q_modules * 4 - k - 3..));
                }

                dc4 += a0;

                let mdc4 = NeonStoreF::f32_mul_add(2., dc4, -dss3);
                {
                    mdc4.write(data.slice_from_mut(q_modules * 4 + k..));
                }

                let dss5 = NeonStoreF::f32_mul_add(2., ds5, -mdc4);
                {
                    let q = dss5.reverse();
                    q.write(data.slice_from_mut(q_modules * 6 - k - 3..));
                }

                dc6 += a0;
                dc6 = NeonStoreF::f32_mul_add(2., -dc6, -dss5);

                {
                    dc6.write(data.slice_from_mut(q_modules * 6 + k..));
                }
            }

            {
                let k = 5;
                let uk = 6;
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = self.rotation_layer[uk];
                let rotation_twiddle_im = self.rotation_layer[uk + 1];

                let c_forward = NeonStoreF::load2(&c_buffer[k..]);
                let s_forward = NeonStoreF::load2(&s_buffer[q_modules - k - 1..]).reverse2();

                let rotated_dc = fmla(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = self.cos_twiddles[uk];
                let twiddle_im = self.cos_twiddles[uk + 1];

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f32::R7_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f32::R7_COS_EVEN2_M2;
                let mut dc6 = twiddled_dc * f32::R7_COS_EVEN2_M1;

                let rotated_ds = fmla(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f32::R7_SIN_ODD0_M0;
                let mut ds3 = twiddled_ds * f32::R7_SIN_ODD1_M0;
                let mut ds5 = twiddled_ds * f32::R7_SIN_ODD2_M0;

                {
                    let c_forward = NeonStoreF::load2(&c_buffer[q_modules + k..]);
                    let s_forward =
                        NeonStoreF::load2(&s_buffer[q_modules * 2 - k - 1..]).reverse2();

                    let rotation_twiddle_re = self.rotation_layer[uk + 2];
                    let rotation_twiddle_im = self.rotation_layer[uk + 3];

                    let twiddle_re = self.cos_twiddles[uk + 2];
                    let twiddle_im = self.cos_twiddles[uk + 3];

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M1, twiddled_dc, dc2);
                    dc4 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M0, twiddled_dc, dc4);
                    dc6 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M2, twiddled_dc, dc6);

                    ds1 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD0_M1, twiddled_ds, ds1);
                    ds3 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD1_M1, twiddled_ds, ds3);
                    ds5 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD2_M1, twiddled_ds, ds5);
                }

                {
                    let c_forward = NeonStoreF::load2(&c_buffer[q_modules * 2 + k..]);
                    let s_forward =
                        NeonStoreF::load2(&s_buffer[q_modules * 3 - k - 1..]).reverse2();

                    let rotation_twiddle_re = self.rotation_layer[uk + 4];
                    let rotation_twiddle_im = self.rotation_layer[uk + 5];

                    let twiddle_re = self.cos_twiddles[uk + 4];
                    let twiddle_im = self.cos_twiddles[uk + 5];

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M2, twiddled_dc, dc2);
                    dc4 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M1, twiddled_dc, dc4);
                    dc6 = NeonStoreF::f32_mul_add(f32::R7_COS_EVEN2_M0, twiddled_dc, dc6);

                    ds1 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD0_M2, twiddled_ds, ds1);
                    ds3 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD1_M2, twiddled_ds, ds3);
                    ds5 = NeonStoreF::f32_mul_add(f32::R7_SIN_ODD2_M2, twiddled_ds, ds5);
                }

                let a0 = NeonStoreF::load2(&a_buffer[k..]);
                let dc = dc0 + a0;
                {
                    dc.write2(data.slice_from_mut(k..));
                }

                let dss1 = NeonStoreF::f32_mul_add(2., ds1, -dc);
                {
                    let q = dss1.reverse2();
                    q.write2(data.slice_from_mut(q_modules * 2 - k - 1..));
                }

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = NeonStoreF::f32_mul_add(2., dc2, -dss1);
                {
                    dc2.write2(data.slice_from_mut(q_modules * 2 + k..));
                }

                let dss3 = NeonStoreF::f32_mul_add(2., -ds3, -dc2);
                {
                    let q = dss3.reverse2();
                    q.write2(data.slice_from_mut(q_modules * 4 - k - 1..));
                }

                dc4 += a0;

                let mdc4 = NeonStoreF::f32_mul_add(2., dc4, -dss3);
                {
                    mdc4.write2(data.slice_from_mut(q_modules * 4 + k..));
                }

                let dss5 = NeonStoreF::f32_mul_add(2., ds5, -mdc4);
                {
                    let q = dss5.reverse2();
                    q.write2(data.slice_from_mut(q_modules * 6 - k - 1..));
                }

                dc6 += a0;
                dc6 = NeonStoreF::f32_mul_add(2., -dc6, -dss5);

                {
                    dc6.write2(data.slice_from_mut(q_modules * 6 + k..));
                }
            }
        }
    }
}

impl PxdctExecutor<f32> for NeonDct2Butterfly49f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(49) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [f32::zero(); 7];
        let mut c_buffer = [f32::zero(); 21];
        let mut s_buffer = [f32::zero(); 21];

        for chunk in data.as_chunks_mut::<49>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        self.execute_into_with_scratch(input, output, &mut [])
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _: &mut [f32],
    ) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 49);

        let mut a_buffer = [f32::zero(); 7];
        let mut c_buffer = [f32::zero(); 21];
        let mut s_buffer = [f32::zero(); 21];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<49>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<49>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
    }

    fn execute_with_scratch(&self, data: &mut [f32], _: &mut [f32]) -> Result<(), PxdctError> {
        self.execute(data)
    }

    fn length(&self) -> usize {
        49
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly_f;
    use crate::tests::naive_dct2_f32;

    gen_test_butterfly_f!(
        test_bf49_f32,
        NeonDct2Butterfly49f,
        49,
        1e-3,
        naive_dct2_f32
    );
}
