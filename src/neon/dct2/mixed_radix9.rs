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
use crate::butterflies::MixedRadix9Sample;
use crate::dct2::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::mla::fmla;
use crate::neon::util::NeonStoreF;
use crate::util::{DctConstants, DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

pub(crate) fn dct2_radix9_rotation_twiddles_neon(q_modules: usize, len: usize) -> Vec<NeonStoreF> {
    let simd_groups = q_modules.div_ceil(4);
    let main_q = 9usize;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 8 * inner_groups);

    let working_modules = q_modules - 1;

    let mut uk = 0usize;
    while uk + 4 <= working_modules {
        let k = uk + 1;

        let layer0 = radixq_rotation_twiddle(main_q, 0, k.as_(), (q_modules - k).as_(), len);
        let layer1 =
            radixq_rotation_twiddle(main_q, 0, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
        let layer2 =
            radixq_rotation_twiddle(main_q, 0, (k + 2).as_(), (q_modules - (k + 2)).as_(), len);
        let layer3 =
            radixq_rotation_twiddle(main_q, 0, (k + 3).as_(), (q_modules - (k + 3)).as_(), len);

        twiddles.push(NeonStoreF::set_values(
            layer0.re, layer1.re, layer2.re, layer3.re,
        ));
        twiddles.push(NeonStoreF::set_values(
            layer0.im, layer1.im, layer2.im, layer3.im,
        ));

        let layer0 = radixq_rotation_twiddle(main_q, 1, k.as_(), (q_modules - k).as_(), len);
        let layer1 =
            radixq_rotation_twiddle(main_q, 1, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
        let layer2 =
            radixq_rotation_twiddle(main_q, 1, (k + 2).as_(), (q_modules - (k + 2)).as_(), len);
        let layer3 =
            radixq_rotation_twiddle(main_q, 1, (k + 3).as_(), (q_modules - (k + 3)).as_(), len);

        twiddles.push(NeonStoreF::set_values(
            layer0.re, layer1.re, layer2.re, layer3.re,
        ));
        twiddles.push(NeonStoreF::set_values(
            layer0.im, layer1.im, layer2.im, layer3.im,
        ));

        let layer0 = radixq_rotation_twiddle(main_q, 2, k.as_(), (q_modules - k).as_(), len);
        let layer1 =
            radixq_rotation_twiddle(main_q, 2, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
        let layer2 =
            radixq_rotation_twiddle(main_q, 2, (k + 2).as_(), (q_modules - (k + 2)).as_(), len);
        let layer3 =
            radixq_rotation_twiddle(main_q, 2, (k + 3).as_(), (q_modules - (k + 3)).as_(), len);

        twiddles.push(NeonStoreF::set_values(
            layer0.re, layer1.re, layer2.re, layer3.re,
        ));
        twiddles.push(NeonStoreF::set_values(
            layer0.im, layer1.im, layer2.im, layer3.im,
        ));

        let layer0 = radixq_rotation_twiddle(main_q, 3, k.as_(), (q_modules - k).as_(), len);
        let layer1 =
            radixq_rotation_twiddle(main_q, 3, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
        let layer2 =
            radixq_rotation_twiddle(main_q, 3, (k + 2).as_(), (q_modules - (k + 2)).as_(), len);
        let layer3 =
            radixq_rotation_twiddle(main_q, 3, (k + 3).as_(), (q_modules - (k + 3)).as_(), len);

        twiddles.push(NeonStoreF::set_values(
            layer0.re, layer1.re, layer2.re, layer3.re,
        ));
        twiddles.push(NeonStoreF::set_values(
            layer0.im, layer1.im, layer2.im, layer3.im,
        ));

        uk += 4;
    }

    let remainder = working_modules - (working_modules / 4) * 4;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for i in 0..remainder {
            let layer =
                radixq_rotation_twiddle(main_q, 0, (k + i).as_(), (q_modules - (k + i)).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..remainder {
            let layer =
                radixq_rotation_twiddle(main_q, 1, (k + i).as_(), (q_modules - (k + i)).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..remainder {
            let layer =
                radixq_rotation_twiddle(main_q, 2, (k + i).as_(), (q_modules - (k + i)).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..remainder {
            let layer =
                radixq_rotation_twiddle(main_q, 3, (k + i).as_(), (q_modules - (k + i)).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));
    }

    twiddles
}

pub(crate) fn dct2_radix9_cos_twiddles_neon(q_modules: usize, len: usize) -> Vec<NeonStoreF> {
    let main_q = 9usize;
    let simd_groups = q_modules.div_ceil(4);
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 8 * inner_groups);

    let working_modules = q_modules - 1;

    let mut uk = 0usize;
    while uk + 4 <= working_modules {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for i in 0..4 {
            array_re[i] = radixq_cos_twiddle(main_q, 0, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 0, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..4 {
            array_re[i] = radixq_cos_twiddle(main_q, 1, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 1, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..4 {
            array_re[i] = radixq_cos_twiddle(main_q, 2, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 2, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..4 {
            array_re[i] = radixq_cos_twiddle(main_q, 3, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 3, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        uk += 4;
    }

    let remainder = working_modules - (working_modules / 4) * 4;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for i in 0..remainder {
            array_re[i] = radixq_cos_twiddle(main_q, 0, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 0, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..remainder {
            array_re[i] = radixq_cos_twiddle(main_q, 1, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 1, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..remainder {
            array_re[i] = radixq_cos_twiddle(main_q, 2, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 2, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        for i in 0..remainder {
            array_re[i] = radixq_cos_twiddle(main_q, 3, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 3, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));
    }

    twiddles
}

pub(crate) struct NeonDct2MixedRadix9f {
    rotation_layer: Vec<NeonStoreF>,
    cos_twiddles: Vec<NeonStoreF>,
    inner_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    execution_length: usize,
}

impl NeonDct2MixedRadix9f {
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<NeonDct2MixedRadix9f, PxdctError> {
        assert!(
            len.is_multiple_of(9),
            "Mixed radix 9 should not be called on sizes no divisible by 9"
        );

        let q_modules = len / 9;

        // always 4 inner groups in Radix-9
        Ok(NeonDct2MixedRadix9f {
            rotation_layer: dct2_radix9_rotation_twiddles_neon(q_modules, len),
            cos_twiddles: dct2_radix9_cos_twiddles_neon(q_modules, len),
            inner_dct,
            execution_length: len,
        })
    }
}

impl PxdctExecutor<f32> for NeonDct2MixedRadix9f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![f32::default(); self.execution_length];

        let q_modules = self.execution_length / 9;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 4);

            // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
            for (n, dst) in a_buffer.iter_mut().enumerate() {
                unsafe {
                    *dst = *chunk.get_unchecked(n * 9 + 4);
                }
            }

            // Extract and combine symmetric pairs with sign alternation for S buffer
            for (m, (c_buffer, s_buffer)) in c_buffer
                .chunks_exact_mut(q_modules)
                .zip(s_buffer.chunks_exact_mut(q_modules))
                .enumerate()
            {
                let mut sign = f32::one();
                for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate()
                {
                    let u0 = unsafe { *chunk.get_unchecked(9 * n + m) };
                    let u1 = unsafe { *chunk.get_unchecked(9 * n + 9 - m - 1) };

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }
            }

            // Step 2: Apply DCT-II to all buffers (A, C₀, C₁, S₀, S₁)
            self.inner_dct.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 4);

            {
                // Step 3: Recombine transformed buffers with twiddle factors

                // Handle k=0 case (DC and low frequencies)
                let qc = c_buffer[0];
                let mut c0 = qc; // Component C₀ (position 0)
                let mut c1 = qc * f32::R9_EVEN_TWIDDLE_0; // Component C₂ (position 2, uses j=2)
                let mut c2 = qc * f32::R9_EVEN_TWIDDLE_2; // Component C₄ (position 4, uses j=4)
                let mut c3 = qc * -f32::HALF; // Component C6 (position 6, uses j=6)
                let mut c4 = qc * f32::R9_EVEN_TWIDDLE_1; // Component C8 (position 8, uses j=8)

                let s0_twiddled = s_buffer[0];

                let mut s0 = s0_twiddled * f32::R9_ODD_TWIDDLE_0;
                let mut s1 = s0_twiddled * -f32::R9_ODD_TWIDDLE_1;
                let mut s2 = s0_twiddled * f32::R9_ODD_TWIDDLE_2;
                let mut s3 = s0_twiddled * -f32::R9_ODD_TWIDDLE_3;

                let ci = unsafe { *c_buffer.get_unchecked(q_modules) };
                let si = unsafe { *s_buffer.get_unchecked(q_modules) };

                let ci2 = unsafe { *c_buffer.get_unchecked(q_modules * 2) };
                let si2 = unsafe { *s_buffer.get_unchecked(q_modules * 2) };

                let ci3 = unsafe { *c_buffer.get_unchecked(q_modules * 3) };
                let si3 = unsafe { *s_buffer.get_unchecked(q_modules * 3) };

                c0 = ci + c0 + ci2 + ci3;

                let a0 = a_buffer[0];

                let dc = c0 + a0;
                unsafe {
                    *chunk.get_unchecked_mut(0) = dc;
                }

                c1 = fmla(ci, -f32::HALF, c1);
                c1 = fmla(ci2, f32::R9_EVEN_TWIDDLE_1, c1);
                c1 = fmla(ci3, f32::R9_EVEN_TWIDDLE_2, c1);

                c2 = fmla(ci, -f32::HALF, c2);
                c2 = fmla(ci2, f32::R9_EVEN_TWIDDLE_0, c2);
                c2 = fmla(ci3, f32::R9_EVEN_TWIDDLE_1, c2);

                let dc2 = c2 + a0;
                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 4) = dc2;
                }

                c3 += ci;
                c3 = fmla(ci2, -f32::HALF, c3);
                c3 = fmla(ci3, -f32::HALF, c3);

                unsafe {
                    let dc3 = c3 + a0;
                    *chunk.get_unchecked_mut(q_modules * 6) = -dc3;
                }

                c4 = fmla(ci, -f32::HALF, c4);
                c4 = fmla(ci2, f32::R9_EVEN_TWIDDLE_2, c4);
                c4 = fmla(ci3, f32::R9_EVEN_TWIDDLE_0, c4);

                unsafe {
                    let dc4 = c4 + a0;
                    *chunk.get_unchecked_mut(q_modules * 8) = dc4;
                }

                s0 = fmla(si, f32::R9_ODD_TWIDDLE_1, s0);
                s0 = fmla(si2, f32::R9_ODD_TWIDDLE_2, s0);
                s0 = fmla(si3, f32::R9_ODD_TWIDDLE_3, s0);

                s1 = fmla(si2, f32::R9_ODD_TWIDDLE_1, s1);
                s1 = fmla(si3, f32::R9_ODD_TWIDDLE_1, s1);

                s2 = fmla(si, -f32::R9_ODD_TWIDDLE_1, s2);
                s2 = fmla(si2, -f32::R9_ODD_TWIDDLE_3, s2);
                s2 = fmla(si3, f32::R9_ODD_TWIDDLE_0, s2);

                s3 = fmla(si, f32::R9_ODD_TWIDDLE_1, s3);
                s3 = fmla(si2, -f32::R9_ODD_TWIDDLE_0, s3);
                s3 = fmla(si3, f32::R9_ODD_TWIDDLE_2, s3);

                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 3) = -s1;
                }

                unsafe {
                    *chunk.get_unchecked_mut(q_modules) = s0;
                }

                unsafe {
                    let idx1 = q_modules * 2;
                    let qid2 = -(c1 + a0); // negated 2j
                    *chunk.get_unchecked_mut(idx1) = qid2;
                }

                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 5) = s2;
                }

                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 7) = -s3;
                }

                let mut k = 1usize;
                let mut uk = 0usize;
                while k + 4 <= q_modules {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = NeonStoreF::load(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        NeonStoreF::load(unsafe { s_buffer.get_unchecked(q_modules - k - 3..) })
                            .reverse();

                    let rotated_dc = fmla(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R9_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R9_EVEN_TWIDDLE_2;
                    let mut dc6 = twiddled_dc * -f32::HALF;
                    let mut dc8 = twiddled_dc * f32::R9_EVEN_TWIDDLE_1;

                    let rotated_ds = fmla(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R9_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R9_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R9_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R9_ODD_TWIDDLE_3;

                    {
                        let c_forward =
                            NeonStoreF::load(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = NeonStoreF::load(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - 3..)
                        })
                        .reverse();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc4);
                        dc6 = twiddled_dc + dc6;
                        dc8 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds5 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds7);
                    }

                    {
                        let c_forward = NeonStoreF::load(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = NeonStoreF::load(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - 3..)
                        })
                        .reverse();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc6);
                        dc8 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_2, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                        ds5 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_0, twiddled_ds, ds7);
                    }

                    {
                        let c_forward = NeonStoreF::load(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = NeonStoreF::load(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - 3..)
                        })
                        .reverse();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc6);
                        dc8 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_0, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                        ds5 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_2, twiddled_ds, ds7);
                    }

                    let a0 = NeonStoreF::load(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;

                    unsafe {
                        dc.write(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = NeonStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse()
                            .write(chunk.get_unchecked_mut(q_modules * 2 - k - 3..));
                    }

                    dc2 = NeonStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }

                    let dss3 = NeonStoreF::f32_mul_add(2., -ds3, -dc2);
                    unsafe {
                        dss3.reverse()
                            .write(chunk.get_unchecked_mut(q_modules * 4 - k - 3..));
                    }

                    let mdc4 = NeonStoreF::f32_mul_add(2., dc4, -dss3);
                    unsafe {
                        mdc4.write(chunk.get_unchecked_mut(q_modules * 4 + k..));
                    }

                    let dss5 = NeonStoreF::f32_mul_add(2., ds5, -mdc4);
                    unsafe {
                        dss5.reverse()
                            .write(chunk.get_unchecked_mut(q_modules * 6 - k - 3..));
                    }

                    dc6 = NeonStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        dc6.write(chunk.get_unchecked_mut(q_modules * 6 + k..));
                    }

                    let dss6 = NeonStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        dss6.reverse()
                            .write(chunk.get_unchecked_mut(q_modules * 8 - k - 3..));
                    }

                    dc8 = NeonStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        dc8.write(chunk.get_unchecked_mut(q_modules * 8 + k..));
                    }
                    k += 4;
                    uk += 8;
                }

                let remainder = q_modules - k;
                if remainder == 3 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = NeonStoreF::load3(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        NeonStoreF::load3(unsafe { s_buffer.get_unchecked(q_modules - k - 2..) })
                            .reverse3();

                    let rotated_dc = fmla(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R9_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R9_EVEN_TWIDDLE_2;
                    let mut dc6 = twiddled_dc * -f32::HALF;
                    let mut dc8 = twiddled_dc * f32::R9_EVEN_TWIDDLE_1;

                    let rotated_ds = fmla(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R9_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R9_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R9_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R9_ODD_TWIDDLE_3;

                    {
                        let c_forward =
                            NeonStoreF::load3(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = NeonStoreF::load3(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - 2..)
                        })
                        .reverse3();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc4);
                        dc6 = twiddled_dc + dc6;
                        dc8 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds5 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds7);
                    }

                    {
                        let c_forward = NeonStoreF::load3(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = NeonStoreF::load3(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - 2..)
                        })
                        .reverse3();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc6);
                        dc8 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_2, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                        ds5 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_0, twiddled_ds, ds7);
                    }

                    {
                        let c_forward = NeonStoreF::load3(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = NeonStoreF::load3(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - 2..)
                        })
                        .reverse3();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc6);
                        dc8 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_0, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                        ds5 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_2, twiddled_ds, ds7);
                    }

                    let a0 = NeonStoreF::load3(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;

                    unsafe {
                        dc.write3(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = NeonStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse3()
                            .write3(chunk.get_unchecked_mut(q_modules * 2 - k - 2..));
                    }

                    dc2 = NeonStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write3(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }

                    let dss3 = NeonStoreF::f32_mul_add(2., -ds3, -dc2);
                    unsafe {
                        dss3.reverse3()
                            .write3(chunk.get_unchecked_mut(q_modules * 4 - k - 2..));
                    }

                    let mdc4 = NeonStoreF::f32_mul_add(2., dc4, -dss3);
                    unsafe {
                        mdc4.write3(chunk.get_unchecked_mut(q_modules * 4 + k..));
                    }

                    let dss5 = NeonStoreF::f32_mul_add(2., ds5, -mdc4);
                    unsafe {
                        dss5.reverse3()
                            .write3(chunk.get_unchecked_mut(q_modules * 6 - k - 2..));
                    }

                    dc6 = NeonStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        dc6.write3(chunk.get_unchecked_mut(q_modules * 6 + k..));
                    }

                    let dss6 = NeonStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        dss6.reverse3()
                            .write3(chunk.get_unchecked_mut(q_modules * 8 - k - 2..));
                    }

                    dc8 = NeonStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        dc8.write3(chunk.get_unchecked_mut(q_modules * 8 + k..));
                    }
                } else if remainder == 2 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = NeonStoreF::load2(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        NeonStoreF::load2(unsafe { s_buffer.get_unchecked(q_modules - k - 1..) })
                            .reverse2();

                    let rotated_dc = fmla(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R9_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R9_EVEN_TWIDDLE_2;
                    let mut dc6 = twiddled_dc * -f32::HALF;
                    let mut dc8 = twiddled_dc * f32::R9_EVEN_TWIDDLE_1;

                    let rotated_ds = fmla(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R9_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R9_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R9_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R9_ODD_TWIDDLE_3;

                    {
                        let c_forward =
                            NeonStoreF::load2(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = NeonStoreF::load2(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - 1..)
                        })
                        .reverse2();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc4);
                        dc6 = twiddled_dc + dc6;
                        dc8 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds5 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds7);
                    }

                    {
                        let c_forward = NeonStoreF::load2(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = NeonStoreF::load2(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - 1..)
                        })
                        .reverse2();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc6);
                        dc8 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_2, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                        ds5 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_0, twiddled_ds, ds7);
                    }

                    {
                        let c_forward = NeonStoreF::load2(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = NeonStoreF::load2(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - 1..)
                        })
                        .reverse2();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc6);
                        dc8 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_0, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                        ds5 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_2, twiddled_ds, ds7);
                    }

                    let a0 = NeonStoreF::load2(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;

                    unsafe {
                        dc.write2(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = NeonStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse2()
                            .write2(chunk.get_unchecked_mut(q_modules * 2 - k - 1..));
                    }

                    dc2 = NeonStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write2(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }

                    let dss3 = NeonStoreF::f32_mul_add(2., -ds3, -dc2);
                    unsafe {
                        dss3.reverse2()
                            .write2(chunk.get_unchecked_mut(q_modules * 4 - k - 1..));
                    }

                    let mdc4 = NeonStoreF::f32_mul_add(2., dc4, -dss3);
                    unsafe {
                        mdc4.write2(chunk.get_unchecked_mut(q_modules * 4 + k..));
                    }

                    let dss5 = NeonStoreF::f32_mul_add(2., ds5, -mdc4);
                    unsafe {
                        dss5.reverse2()
                            .write2(chunk.get_unchecked_mut(q_modules * 6 - k - 1..));
                    }

                    dc6 = NeonStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        dc6.write2(chunk.get_unchecked_mut(q_modules * 6 + k..));
                    }

                    let dss6 = NeonStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        dss6.reverse2()
                            .write2(chunk.get_unchecked_mut(q_modules * 8 - k - 1..));
                    }

                    dc8 = NeonStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        dc8.write2(chunk.get_unchecked_mut(q_modules * 8 + k..));
                    }
                } else if remainder == 1 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = NeonStoreF::load1(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        NeonStoreF::load1(unsafe { s_buffer.get_unchecked(q_modules - k..) });

                    let rotated_dc = fmla(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R9_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R9_EVEN_TWIDDLE_2;
                    let mut dc6 = twiddled_dc * -f32::HALF;
                    let mut dc8 = twiddled_dc * f32::R9_EVEN_TWIDDLE_1;

                    let rotated_ds = fmla(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R9_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R9_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R9_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R9_ODD_TWIDDLE_3;

                    {
                        let c_forward =
                            NeonStoreF::load1(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = NeonStoreF::load1(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k..)
                        });

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc4);
                        dc6 = twiddled_dc + dc6;
                        dc8 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds5 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds7);
                    }

                    {
                        let c_forward = NeonStoreF::load1(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = NeonStoreF::load1(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k..)
                        });

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc6);
                        dc8 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_2, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                        ds5 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(-f32::R9_ODD_TWIDDLE_0, twiddled_ds, ds7);
                    }

                    {
                        let c_forward = NeonStoreF::load1(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = NeonStoreF::load1(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k..)
                        });

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = NeonStoreF::f32_mul_add(-f32::HALF, twiddled_dc, dc6);
                        dc8 = NeonStoreF::f32_mul_add(f32::R9_EVEN_TWIDDLE_0, twiddled_dc, dc8);

                        ds1 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                        ds5 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = NeonStoreF::f32_mul_add(f32::R9_ODD_TWIDDLE_2, twiddled_ds, ds7);
                    }

                    let a0 = NeonStoreF::load1(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;

                    unsafe {
                        dc.write1(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = NeonStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.write1(chunk.get_unchecked_mut(q_modules * 2 - k..));
                    }

                    dc2 = NeonStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write1(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }

                    let dss3 = NeonStoreF::f32_mul_add(2., -ds3, -dc2);
                    unsafe {
                        dss3.write1(chunk.get_unchecked_mut(q_modules * 4 - k..));
                    }

                    let mdc4 = NeonStoreF::f32_mul_add(2., dc4, -dss3);
                    unsafe {
                        mdc4.write1(chunk.get_unchecked_mut(q_modules * 4 + k..));
                    }

                    let dss5 = NeonStoreF::f32_mul_add(2., ds5, -mdc4);
                    unsafe {
                        dss5.write1(chunk.get_unchecked_mut(q_modules * 6 - k..));
                    }

                    dc6 = NeonStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        dc6.write1(chunk.get_unchecked_mut(q_modules * 6 + k..));
                    }

                    let dss6 = NeonStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        dss6.write1(chunk.get_unchecked_mut(q_modules * 8 - k..));
                    }

                    dc8 = NeonStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        dc8.write1(chunk.get_unchecked_mut(q_modules * 8 + k..));
                    }
                }
            }
        }

        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct2_f32;

    #[test]
    fn test_radix9_dct() {
        let mut input = vec![0.; 243];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32 + rand::random::<f32>() * 10.0;
        }
        // let mut input = vec![
        //     7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256,
        //     12.010594, 7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292,
        //     15.516256, 12.010594, 7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984,
        //     9.859292, 15.516256, 12.010594,
        // ];
        let mut reference_input = input.clone();
        // let rr = Pxdcf32::make_dct2_f32(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2_f32(&reference_input);
        let bf =
            NeonDct2MixedRadix9f::new(input.len(), Pxdct::make_dct2_f32(input.len() / 9).unwrap())
                .unwrap();
        bf.execute(&mut input).unwrap();
        println!(
            "{:?}",
            input
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "{:?}",
            reference_input
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                if (src - r0).abs() > 1e-2 {
                    println!(
                        "Difference must be < {}, but it was {}, at position {i}",
                        1e-2,
                        (src - r0).abs()
                    )
                }
            });
    }
}
