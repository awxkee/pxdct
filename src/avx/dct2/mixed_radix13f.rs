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
use crate::avx::storef::AvxStoreF;
use crate::avx::util::fma;
use crate::dct2::prime_butterflies::MixedRadix13Sample;
use crate::dct2::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix13_rotation_twiddles_avxf(q_modules: usize, len: usize) -> Vec<AvxStoreF> {
    let simd_groups = q_modules.div_ceil(8);
    let main_q = 13usize;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 12 * inner_groups);

    let working_modules = q_modules - 1;

    let mut uk = 0usize;
    while uk + 8 <= working_modules {
        let k = uk + 1;

        let mut array_re = [0.; 8];
        let mut array_im = [0.; 8];
        for m in 0..inner_groups {
            for i in 0..8 {
                let layer = radixq_rotation_twiddle(
                    main_q,
                    m,
                    (k + i).as_(),
                    (q_modules - (k + i)).as_(),
                    len,
                );
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }
            twiddles.push(AvxStoreF::load(array_re.as_ref()));
            twiddles.push(AvxStoreF::load(array_im.as_ref()));
        }

        uk += 8;
    }

    let remainder = working_modules - (working_modules / 8) * 8;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0.; 8];
        let mut array_im = [0.; 8];
        for m in 0..inner_groups {
            for i in 0..remainder {
                let layer = radixq_rotation_twiddle(
                    main_q,
                    m,
                    (k + i).as_(),
                    (q_modules - (k + i)).as_(),
                    len,
                );
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }
            twiddles.push(AvxStoreF::load(array_re.as_ref()));
            twiddles.push(AvxStoreF::load(array_im.as_ref()));
        }
    }

    twiddles
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix13_cos_twiddles_avxf(q_modules: usize, len: usize) -> Vec<AvxStoreF> {
    let main_q = 13usize;
    let simd_groups = q_modules.div_ceil(8);
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 12 * inner_groups);

    let working_modules = q_modules - 1;

    let mut uk = 0usize;
    while uk + 8 <= working_modules {
        let k = uk + 1;

        let mut array_re = [0.; 8];
        let mut array_im = [0.; 8];

        for m in 0..inner_groups {
            for i in 0..8 {
                array_re[i] = radixq_cos_twiddle(main_q, m, (k + i).as_(), len);
                array_im[i] = radixq_cos_twiddle(main_q, m, (q_modules - (k + i)).as_(), len);
            }

            twiddles.push(AvxStoreF::load(array_re.as_ref()));
            twiddles.push(AvxStoreF::load(array_im.as_ref()));
        }

        uk += 8;
    }

    let remainder = working_modules - (working_modules / 8) * 8;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0.; 8];
        let mut array_im = [0.; 8];

        for m in 0..inner_groups {
            for i in 0..remainder {
                array_re[i] = radixq_cos_twiddle(main_q, m, (k + i).as_(), len);
                array_im[i] = radixq_cos_twiddle(main_q, m, (q_modules - (k + i)).as_(), len);
            }

            twiddles.push(AvxStoreF::load(array_re.as_ref()));
            twiddles.push(AvxStoreF::load(array_im.as_ref()));
        }
    }

    twiddles
}

pub(crate) struct AvxDct2MixedRadix13f {
    rotation_layer: Vec<AvxStoreF>,
    cos_twiddles: Vec<AvxStoreF>,
    inner_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    execution_length: usize,
}

impl AvxDct2MixedRadix13f {
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<AvxDct2MixedRadix13f, PxdctError> {
        assert!(
            len.is_multiple_of(13),
            "Mixed radix 9 should not be called on sizes no divisible by 9"
        );

        let q_modules = len / 13;

        Ok(AvxDct2MixedRadix13f {
            rotation_layer: unsafe { dct2_radix13_rotation_twiddles_avxf(q_modules, len) },
            cos_twiddles: unsafe { dct2_radix13_cos_twiddles_avxf(q_modules, len) },
            inner_dct,
            execution_length: len,
        })
    }
}

impl PxdctExecutor<f32> for AvxDct2MixedRadix13f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

impl AvxDct2MixedRadix13f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![f32::default(); self.execution_length];

        let q_modules = self.execution_length / 13;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 6);

            // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
            for (n, dst) in a_buffer.iter_mut().enumerate() {
                unsafe {
                    *dst = *chunk.get_unchecked(n * 13 + 6);
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
                    let u0 = unsafe { *chunk.get_unchecked(13 * n + m) };
                    let u1 = unsafe { *chunk.get_unchecked(13 * n + 13 - m - 1) };

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }
            }

            // Step 2: Apply DCT-II to all buffers (A, C₀, C₁, S₀, S₁)
            self.inner_dct.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 6);

            {
                // Step 3: Recombine transformed buffers with twiddle factors

                // Handle k=0 case (DC and low frequencies)
                let qc = c_buffer[0];
                let mut c0 = qc; // Component C₀ (position 0)
                let mut c1 = qc * f32::R13_EVEN_TWIDDLE_0; // Component C₂ (position 2, uses j=2)
                let mut c2 = qc * f32::R13_EVEN_TWIDDLE_5; // Component C₄ (position 4, uses j=4)
                let mut c3 = qc * f32::R13_EVEN_TWIDDLE_1; // Component C6 (position 6, uses j=6)
                let mut c4 = qc * f32::R13_EVEN_TWIDDLE_4; // Component C8 (position 8, uses j=8)
                let mut c5 = qc * f32::R13_EVEN_TWIDDLE_2;
                let mut c6 = qc * f32::R13_EVEN_TWIDDLE_3;

                let s0_twiddled = s_buffer[0];

                let mut s0 = s0_twiddled * f32::R13_ODD_TWIDDLE_0;
                let mut s1 = s0_twiddled * -f32::R13_ODD_TWIDDLE_1;
                let mut s2 = s0_twiddled * f32::R13_ODD_TWIDDLE_2;
                let mut s3 = s0_twiddled * -f32::R13_ODD_TWIDDLE_3;
                let mut s4 = s0_twiddled * f32::R13_ODD_TWIDDLE_4;
                let mut s5 = s0_twiddled * -f32::R13_ODD_TWIDDLE_5;

                let ci = unsafe { *c_buffer.get_unchecked(q_modules) };
                let si = unsafe { *s_buffer.get_unchecked(q_modules) };

                let ci2 = unsafe { *c_buffer.get_unchecked(q_modules * 2) };
                let si2 = unsafe { *s_buffer.get_unchecked(q_modules * 2) };

                let ci3 = unsafe { *c_buffer.get_unchecked(q_modules * 3) };
                let si3 = unsafe { *s_buffer.get_unchecked(q_modules * 3) };

                let ci4 = unsafe { *c_buffer.get_unchecked(q_modules * 4) };
                let si4 = unsafe { *s_buffer.get_unchecked(q_modules * 4) };

                let ci5 = unsafe { *c_buffer.get_unchecked(q_modules * 5) };
                let si5 = unsafe { *s_buffer.get_unchecked(q_modules * 5) };

                c0 = ci + c0 + ci2 + ci3 + ci4 + ci5;

                let a0 = a_buffer[0];

                let dc = c0 + a0;
                unsafe {
                    *chunk.get_unchecked_mut(0) = dc;
                }

                c1 = fma(ci, f32::R13_EVEN_TWIDDLE_1, c1);
                c1 = fma(ci2, f32::R13_EVEN_TWIDDLE_2, c1);
                c1 = fma(ci3, f32::R13_EVEN_TWIDDLE_3, c1);
                c1 = fma(ci4, f32::R13_EVEN_TWIDDLE_4, c1);
                c1 = fma(ci5, f32::R13_EVEN_TWIDDLE_5, c1);

                c2 = fma(ci, f32::R13_EVEN_TWIDDLE_3, c2);
                c2 = fma(ci2, f32::R13_EVEN_TWIDDLE_1, c2);
                c2 = fma(ci3, f32::R13_EVEN_TWIDDLE_0, c2);
                c2 = fma(ci4, f32::R13_EVEN_TWIDDLE_2, c2);
                c2 = fma(ci5, f32::R13_EVEN_TWIDDLE_4, c2);

                let dc2 = c2 + a0;
                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 4) = dc2;
                }

                c3 = fma(ci, f32::R13_EVEN_TWIDDLE_4, c3);
                c3 = fma(ci2, f32::R13_EVEN_TWIDDLE_5, c3);
                c3 = fma(ci3, f32::R13_EVEN_TWIDDLE_2, c3);
                c3 = fma(ci4, f32::R13_EVEN_TWIDDLE_0, c3);
                c3 = fma(ci5, f32::R13_EVEN_TWIDDLE_3, c3);

                unsafe {
                    let dc3 = c3 + a0;
                    *chunk.get_unchecked_mut(q_modules * 6) = -dc3;
                }

                c4 = fma(ci, f32::R13_EVEN_TWIDDLE_0, c4);
                c4 = fma(ci2, f32::R13_EVEN_TWIDDLE_3, c4);
                c4 = fma(ci3, f32::R13_EVEN_TWIDDLE_5, c4);
                c4 = fma(ci4, f32::R13_EVEN_TWIDDLE_1, c4);
                c4 = fma(ci5, f32::R13_EVEN_TWIDDLE_2, c4);

                unsafe {
                    let dc4 = c4 + a0;
                    *chunk.get_unchecked_mut(q_modules * 8) = dc4;
                }

                c5 = fma(ci, f32::R13_EVEN_TWIDDLE_5, c5);
                c5 = fma(ci2, f32::R13_EVEN_TWIDDLE_0, c5);
                c5 = fma(ci3, f32::R13_EVEN_TWIDDLE_4, c5);
                c5 = fma(ci4, f32::R13_EVEN_TWIDDLE_3, c5);
                c5 = fma(ci5, f32::R13_EVEN_TWIDDLE_1, c5);

                unsafe {
                    let dc5 = c5 + a0;
                    *chunk.get_unchecked_mut(q_modules * 10) = -dc5;
                }

                c6 = fma(ci, f32::R13_EVEN_TWIDDLE_2, c6);
                c6 = fma(ci2, f32::R13_EVEN_TWIDDLE_4, c6);
                c6 = fma(ci3, f32::R13_EVEN_TWIDDLE_1, c6);
                c6 = fma(ci4, f32::R13_EVEN_TWIDDLE_5, c6);
                c6 = fma(ci5, f32::R13_EVEN_TWIDDLE_0, c6);

                unsafe {
                    let dc6 = c6 + a0;
                    *chunk.get_unchecked_mut(q_modules * 12) = dc6;
                }

                s0 = fma(si, f32::R13_ODD_TWIDDLE_1, s0);
                s0 = fma(si2, f32::R13_ODD_TWIDDLE_2, s0);
                s0 = fma(si3, f32::R13_ODD_TWIDDLE_3, s0);
                s0 = fma(si4, f32::R13_ODD_TWIDDLE_4, s0);
                s0 = fma(si5, f32::R13_ODD_TWIDDLE_5, s0);

                s1 = fma(si, -f32::R13_ODD_TWIDDLE_4, s1);
                s1 = fma(si2, f32::R13_ODD_TWIDDLE_5, s1);
                s1 = fma(si3, f32::R13_ODD_TWIDDLE_2, s1);
                s1 = fma(si4, f32::R13_ODD_TWIDDLE_0, s1);
                s1 = fma(si5, f32::R13_ODD_TWIDDLE_3, s1);

                s2 = fma(si, -f32::R13_ODD_TWIDDLE_5, s2);
                s2 = fma(si2, -f32::R13_ODD_TWIDDLE_0, s2);
                s2 = fma(si3, -f32::R13_ODD_TWIDDLE_4, s2);
                s2 = fma(si4, f32::R13_ODD_TWIDDLE_3, s2);
                s2 = fma(si5, f32::R13_ODD_TWIDDLE_1, s2);

                s3 = fma(si, f32::R13_ODD_TWIDDLE_2, s3);
                s3 = fma(si2, f32::R13_ODD_TWIDDLE_4, s3);
                s3 = fma(si3, -f32::R13_ODD_TWIDDLE_1, s3);
                s3 = fma(si4, -f32::R13_ODD_TWIDDLE_5, s3);
                s3 = fma(si5, f32::R13_ODD_TWIDDLE_0, s3);

                s4 = fma(si, -f32::R13_ODD_TWIDDLE_0, s4);
                s4 = fma(si2, f32::R13_ODD_TWIDDLE_3, s4);
                s4 = fma(si3, f32::R13_ODD_TWIDDLE_5, s4);
                s4 = fma(si4, -f32::R13_ODD_TWIDDLE_1, s4);
                s4 = fma(si5, f32::R13_ODD_TWIDDLE_2, s4);

                s5 = fma(si, f32::R13_ODD_TWIDDLE_3, s5);
                s5 = fma(si2, -f32::R13_ODD_TWIDDLE_1, s5);
                s5 = fma(si3, f32::R13_ODD_TWIDDLE_0, s5);
                s5 = fma(si4, -f32::R13_ODD_TWIDDLE_2, s5);
                s5 = fma(si5, f32::R13_ODD_TWIDDLE_4, s5);

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

                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 9) = s4;
                }

                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 11) = -s5;
                }

                let mut uk = 0usize;
                let mut k = 1usize;
                while k + 8 <= q_modules {
                    const S: usize = 7;
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                            .reverse();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R13_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R13_EVEN_TWIDDLE_5;
                    let mut dc6 = twiddled_dc * f32::R13_EVEN_TWIDDLE_1;
                    let mut dc8 = twiddled_dc * f32::R13_EVEN_TWIDDLE_4;
                    let mut dc10 = twiddled_dc * f32::R13_EVEN_TWIDDLE_2;
                    let mut dc12 = twiddled_dc * f32::R13_EVEN_TWIDDLE_3;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R13_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R13_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R13_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R13_ODD_TWIDDLE_3;
                    let mut ds9 = twiddled_ds * f32::R13_ODD_TWIDDLE_4;
                    let mut ds11 = twiddled_ds * -f32::R13_ODD_TWIDDLE_5;

                    {
                        let c_forward =
                            AvxStoreF::load(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = AvxStoreF::load(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - S..)
                        })
                        .reverse();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds11);
                    }

                    {
                        let c_forward =
                            AvxStoreF::load(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                        let s_forward = AvxStoreF::load(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - S..)
                        })
                        .reverse();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds11);
                    }

                    {
                        let c_forward =
                            AvxStoreF::load(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                        let s_forward = AvxStoreF::load(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - S..)
                        })
                        .reverse();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds11);
                    }

                    {
                        let c_forward =
                            AvxStoreF::load(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                        let s_forward = AvxStoreF::load(unsafe {
                            s_buffer.get_unchecked(q_modules * 5 - k - S..)
                        })
                        .reverse();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds11);
                    }

                    {
                        let c_forward =
                            AvxStoreF::load(unsafe { c_buffer.get_unchecked(q_modules * 5 + k..) });
                        let s_forward = AvxStoreF::load(unsafe {
                            s_buffer.get_unchecked(q_modules * 6 - k - S..)
                        })
                        .reverse();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 10) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 11) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 10) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 11) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds11);
                    }

                    let a0 = AvxStoreF::load(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;
                    dc10 += a0;
                    dc12 += a0;

                    unsafe {
                        dc.write(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse()
                            .write(chunk.get_unchecked_mut(q_modules * 2 - k - S..));
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = AvxStoreF::f32_mul_add(2f32.as_(), dc2, -dss1);
                    unsafe {
                        dc2.write(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss3 = AvxStoreF::f32_mul_add(2f32.as_(), -ds3, -dc2);
                    unsafe {
                        let idx = q_modules * 4 - k - S;
                        dss3.reverse().write(chunk.get_unchecked_mut(idx..));
                    }

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = AvxStoreF::f32_mul_add(2f32.as_(), dc4, -dss3);
                    unsafe {
                        mdc4.write(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss5 = AvxStoreF::f32_mul_add(2f32.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k - S;
                        dss5.reverse().write(chunk.get_unchecked_mut(idx..));
                    }

                    dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        dc6.write(chunk.get_unchecked_mut(idx..));
                    }

                    let dss6 = AvxStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        let idx = q_modules * 8 - k - S;
                        dss6.reverse().write(chunk.get_unchecked_mut(idx..));
                    }

                    dc8 = AvxStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        let idx = q_modules * 8 + k;
                        dc8.write(chunk.get_unchecked_mut(idx..));
                    }

                    let dss9 = AvxStoreF::f32_mul_add(2., ds9, -dc8);
                    unsafe {
                        let idx = q_modules * 10 - k - S;
                        dss9.reverse().write(chunk.get_unchecked_mut(idx..));
                    }

                    dc10 = AvxStoreF::f32_mul_add(2., -dc10, -dss9);
                    unsafe {
                        let idx = q_modules * 10 + k;
                        dc10.write(chunk.get_unchecked_mut(idx..));
                    }

                    let dss11 = AvxStoreF::f32_mul_add(2., -ds11, -dc10);
                    unsafe {
                        let idx = q_modules * 12 - k - S;
                        dss11.reverse().write(chunk.get_unchecked_mut(idx..));
                    }

                    dc12 = AvxStoreF::f32_mul_add(2., dc12, -dss11);
                    unsafe {
                        let idx = q_modules * 12 + k;
                        dc12.write(chunk.get_unchecked_mut(idx..));
                    }

                    k += 8;
                    uk += 12;
                }

                let remainder = q_modules - k;

                if remainder == 7 {
                    const S: usize = 6;
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load7(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load7(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                            .reverse7();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R13_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R13_EVEN_TWIDDLE_5;
                    let mut dc6 = twiddled_dc * f32::R13_EVEN_TWIDDLE_1;
                    let mut dc8 = twiddled_dc * f32::R13_EVEN_TWIDDLE_4;
                    let mut dc10 = twiddled_dc * f32::R13_EVEN_TWIDDLE_2;
                    let mut dc12 = twiddled_dc * f32::R13_EVEN_TWIDDLE_3;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R13_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R13_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R13_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R13_ODD_TWIDDLE_3;
                    let mut ds9 = twiddled_ds * f32::R13_ODD_TWIDDLE_4;
                    let mut ds11 = twiddled_ds * -f32::R13_ODD_TWIDDLE_5;

                    {
                        let c_forward =
                            AvxStoreF::load7(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = AvxStoreF::load7(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - S..)
                        })
                        .reverse7();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load7(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = AvxStoreF::load7(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - S..)
                        })
                        .reverse7();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load7(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = AvxStoreF::load7(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - S..)
                        })
                        .reverse7();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load7(unsafe {
                            c_buffer.get_unchecked(q_modules * 4 + k..)
                        });
                        let s_forward = AvxStoreF::load7(unsafe {
                            s_buffer.get_unchecked(q_modules * 5 - k - S..)
                        })
                        .reverse7();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load7(unsafe {
                            c_buffer.get_unchecked(q_modules * 5 + k..)
                        });
                        let s_forward = AvxStoreF::load7(unsafe {
                            s_buffer.get_unchecked(q_modules * 6 - k - S..)
                        })
                        .reverse7();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 10) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 11) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 10) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 11) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds11);
                    }

                    let a0 = AvxStoreF::load7(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;
                    dc10 += a0;
                    dc12 += a0;

                    unsafe {
                        dc.write7(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse7()
                            .write7(chunk.get_unchecked_mut(q_modules * 2 - k - S..));
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = AvxStoreF::f32_mul_add(2f32.as_(), dc2, -dss1);
                    unsafe {
                        dc2.write7(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss3 = AvxStoreF::f32_mul_add(2f32.as_(), -ds3, -dc2);
                    unsafe {
                        let idx = q_modules * 4 - k - S;
                        dss3.reverse7().write7(chunk.get_unchecked_mut(idx..));
                    }

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = AvxStoreF::f32_mul_add(2f32.as_(), dc4, -dss3);
                    unsafe {
                        mdc4.write7(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss5 = AvxStoreF::f32_mul_add(2f32.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k - S;
                        dss5.reverse7().write7(chunk.get_unchecked_mut(idx..));
                    }

                    dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        dc6.write7(chunk.get_unchecked_mut(idx..));
                    }

                    let dss6 = AvxStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        let idx = q_modules * 8 - k - S;
                        dss6.reverse7().write7(chunk.get_unchecked_mut(idx..));
                    }

                    dc8 = AvxStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        let idx = q_modules * 8 + k;
                        dc8.write7(chunk.get_unchecked_mut(idx..));
                    }

                    let dss9 = AvxStoreF::f32_mul_add(2., ds9, -dc8);
                    unsafe {
                        let idx = q_modules * 10 - k - S;
                        dss9.reverse7().write7(chunk.get_unchecked_mut(idx..));
                    }

                    dc10 = AvxStoreF::f32_mul_add(2., -dc10, -dss9);
                    unsafe {
                        let idx = q_modules * 10 + k;
                        dc10.write7(chunk.get_unchecked_mut(idx..));
                    }

                    let dss11 = AvxStoreF::f32_mul_add(2., -ds11, -dc10);
                    unsafe {
                        let idx = q_modules * 12 - k - S;
                        dss11.reverse7().write7(chunk.get_unchecked_mut(idx..));
                    }

                    dc12 = AvxStoreF::f32_mul_add(2., dc12, -dss11);
                    unsafe {
                        let idx = q_modules * 12 + k;
                        dc12.write7(chunk.get_unchecked_mut(idx..));
                    }
                } else if remainder == 6 {
                    const S: usize = 5;
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load6(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load6(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                            .reverse6();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R13_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R13_EVEN_TWIDDLE_5;
                    let mut dc6 = twiddled_dc * f32::R13_EVEN_TWIDDLE_1;
                    let mut dc8 = twiddled_dc * f32::R13_EVEN_TWIDDLE_4;
                    let mut dc10 = twiddled_dc * f32::R13_EVEN_TWIDDLE_2;
                    let mut dc12 = twiddled_dc * f32::R13_EVEN_TWIDDLE_3;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R13_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R13_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R13_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R13_ODD_TWIDDLE_3;
                    let mut ds9 = twiddled_ds * f32::R13_ODD_TWIDDLE_4;
                    let mut ds11 = twiddled_ds * -f32::R13_ODD_TWIDDLE_5;

                    {
                        let c_forward =
                            AvxStoreF::load6(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = AvxStoreF::load6(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - S..)
                        })
                        .reverse6();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load6(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = AvxStoreF::load6(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - S..)
                        })
                        .reverse6();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load6(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = AvxStoreF::load6(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - S..)
                        })
                        .reverse6();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load6(unsafe {
                            c_buffer.get_unchecked(q_modules * 4 + k..)
                        });
                        let s_forward = AvxStoreF::load6(unsafe {
                            s_buffer.get_unchecked(q_modules * 5 - k - S..)
                        })
                        .reverse6();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load6(unsafe {
                            c_buffer.get_unchecked(q_modules * 5 + k..)
                        });
                        let s_forward = AvxStoreF::load6(unsafe {
                            s_buffer.get_unchecked(q_modules * 6 - k - S..)
                        })
                        .reverse6();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 10) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 11) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 10) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 11) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds11);
                    }

                    let a0 = AvxStoreF::load6(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;
                    dc10 += a0;
                    dc12 += a0;

                    unsafe {
                        dc.write6(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse6()
                            .write6(chunk.get_unchecked_mut(q_modules * 2 - k - S..));
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = AvxStoreF::f32_mul_add(2f32.as_(), dc2, -dss1);
                    unsafe {
                        dc2.write6(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss3 = AvxStoreF::f32_mul_add(2f32.as_(), -ds3, -dc2);
                    unsafe {
                        let idx = q_modules * 4 - k - S;
                        dss3.reverse6().write6(chunk.get_unchecked_mut(idx..));
                    }

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = AvxStoreF::f32_mul_add(2f32.as_(), dc4, -dss3);
                    unsafe {
                        mdc4.write6(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss5 = AvxStoreF::f32_mul_add(2f32.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k - S;
                        dss5.reverse6().write6(chunk.get_unchecked_mut(idx..));
                    }

                    dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        dc6.write6(chunk.get_unchecked_mut(idx..));
                    }

                    let dss6 = AvxStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        let idx = q_modules * 8 - k - S;
                        dss6.reverse6().write6(chunk.get_unchecked_mut(idx..));
                    }

                    dc8 = AvxStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        let idx = q_modules * 8 + k;
                        dc8.write6(chunk.get_unchecked_mut(idx..));
                    }

                    let dss9 = AvxStoreF::f32_mul_add(2., ds9, -dc8);
                    unsafe {
                        let idx = q_modules * 10 - k - S;
                        dss9.reverse6().write6(chunk.get_unchecked_mut(idx..));
                    }

                    dc10 = AvxStoreF::f32_mul_add(2., -dc10, -dss9);
                    unsafe {
                        let idx = q_modules * 10 + k;
                        dc10.write6(chunk.get_unchecked_mut(idx..));
                    }

                    let dss11 = AvxStoreF::f32_mul_add(2., -ds11, -dc10);
                    unsafe {
                        let idx = q_modules * 12 - k - S;
                        dss11.reverse6().write6(chunk.get_unchecked_mut(idx..));
                    }

                    dc12 = AvxStoreF::f32_mul_add(2., dc12, -dss11);
                    unsafe {
                        let idx = q_modules * 12 + k;
                        dc12.write6(chunk.get_unchecked_mut(idx..));
                    }
                } else if remainder == 5 {
                    const S: usize = 4;
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load5(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load5(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                            .reverse5();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R13_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R13_EVEN_TWIDDLE_5;
                    let mut dc6 = twiddled_dc * f32::R13_EVEN_TWIDDLE_1;
                    let mut dc8 = twiddled_dc * f32::R13_EVEN_TWIDDLE_4;
                    let mut dc10 = twiddled_dc * f32::R13_EVEN_TWIDDLE_2;
                    let mut dc12 = twiddled_dc * f32::R13_EVEN_TWIDDLE_3;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R13_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R13_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R13_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R13_ODD_TWIDDLE_3;
                    let mut ds9 = twiddled_ds * f32::R13_ODD_TWIDDLE_4;
                    let mut ds11 = twiddled_ds * -f32::R13_ODD_TWIDDLE_5;

                    {
                        let c_forward =
                            AvxStoreF::load5(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = AvxStoreF::load5(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - S..)
                        })
                        .reverse5();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load5(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = AvxStoreF::load5(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - S..)
                        })
                        .reverse5();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load5(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = AvxStoreF::load5(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - S..)
                        })
                        .reverse5();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load5(unsafe {
                            c_buffer.get_unchecked(q_modules * 4 + k..)
                        });
                        let s_forward = AvxStoreF::load5(unsafe {
                            s_buffer.get_unchecked(q_modules * 5 - k - S..)
                        })
                        .reverse5();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load5(unsafe {
                            c_buffer.get_unchecked(q_modules * 5 + k..)
                        });
                        let s_forward = AvxStoreF::load5(unsafe {
                            s_buffer.get_unchecked(q_modules * 6 - k - S..)
                        })
                        .reverse5();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 10) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 11) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 10) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 11) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds11);
                    }

                    let a0 = AvxStoreF::load5(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;
                    dc10 += a0;
                    dc12 += a0;

                    unsafe {
                        dc.write5(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse5()
                            .write5(chunk.get_unchecked_mut(q_modules * 2 - k - S..));
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = AvxStoreF::f32_mul_add(2f32.as_(), dc2, -dss1);
                    unsafe {
                        dc2.write5(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss3 = AvxStoreF::f32_mul_add(2f32.as_(), -ds3, -dc2);
                    unsafe {
                        let idx = q_modules * 4 - k - S;
                        dss3.reverse5().write5(chunk.get_unchecked_mut(idx..));
                    }

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = AvxStoreF::f32_mul_add(2f32.as_(), dc4, -dss3);
                    unsafe {
                        mdc4.write5(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss5 = AvxStoreF::f32_mul_add(2f32.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k - S;
                        dss5.reverse5().write5(chunk.get_unchecked_mut(idx..));
                    }

                    dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        dc6.write5(chunk.get_unchecked_mut(idx..));
                    }

                    let dss6 = AvxStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        let idx = q_modules * 8 - k - S;
                        dss6.reverse5().write5(chunk.get_unchecked_mut(idx..));
                    }

                    dc8 = AvxStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        let idx = q_modules * 8 + k;
                        dc8.write5(chunk.get_unchecked_mut(idx..));
                    }

                    let dss9 = AvxStoreF::f32_mul_add(2., ds9, -dc8);
                    unsafe {
                        let idx = q_modules * 10 - k - S;
                        dss9.reverse5().write5(chunk.get_unchecked_mut(idx..));
                    }

                    dc10 = AvxStoreF::f32_mul_add(2., -dc10, -dss9);
                    unsafe {
                        let idx = q_modules * 10 + k;
                        dc10.write5(chunk.get_unchecked_mut(idx..));
                    }

                    let dss11 = AvxStoreF::f32_mul_add(2., -ds11, -dc10);
                    unsafe {
                        let idx = q_modules * 12 - k - S;
                        dss11.reverse5().write5(chunk.get_unchecked_mut(idx..));
                    }

                    dc12 = AvxStoreF::f32_mul_add(2., dc12, -dss11);
                    unsafe {
                        let idx = q_modules * 12 + k;
                        dc12.write5(chunk.get_unchecked_mut(idx..));
                    }
                } else if remainder == 4 {
                    const S: usize = 3;
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load4(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load4(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                            .reverse4();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R13_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R13_EVEN_TWIDDLE_5;
                    let mut dc6 = twiddled_dc * f32::R13_EVEN_TWIDDLE_1;
                    let mut dc8 = twiddled_dc * f32::R13_EVEN_TWIDDLE_4;
                    let mut dc10 = twiddled_dc * f32::R13_EVEN_TWIDDLE_2;
                    let mut dc12 = twiddled_dc * f32::R13_EVEN_TWIDDLE_3;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R13_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R13_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R13_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R13_ODD_TWIDDLE_3;
                    let mut ds9 = twiddled_ds * f32::R13_ODD_TWIDDLE_4;
                    let mut ds11 = twiddled_ds * -f32::R13_ODD_TWIDDLE_5;

                    {
                        let c_forward =
                            AvxStoreF::load4(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = AvxStoreF::load4(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - S..)
                        })
                        .reverse4();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load4(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = AvxStoreF::load4(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - S..)
                        })
                        .reverse4();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load4(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = AvxStoreF::load4(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - S..)
                        })
                        .reverse4();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load4(unsafe {
                            c_buffer.get_unchecked(q_modules * 4 + k..)
                        });
                        let s_forward = AvxStoreF::load4(unsafe {
                            s_buffer.get_unchecked(q_modules * 5 - k - S..)
                        })
                        .reverse4();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load4(unsafe {
                            c_buffer.get_unchecked(q_modules * 5 + k..)
                        });
                        let s_forward = AvxStoreF::load4(unsafe {
                            s_buffer.get_unchecked(q_modules * 6 - k - S..)
                        })
                        .reverse4();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 10) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 11) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 10) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 11) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds11);
                    }

                    let a0 = AvxStoreF::load4(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;
                    dc10 += a0;
                    dc12 += a0;

                    unsafe {
                        dc.write4(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse4()
                            .write4(chunk.get_unchecked_mut(q_modules * 2 - k - S..));
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = AvxStoreF::f32_mul_add(2f32.as_(), dc2, -dss1);
                    unsafe {
                        dc2.write4(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss3 = AvxStoreF::f32_mul_add(2f32.as_(), -ds3, -dc2);
                    unsafe {
                        let idx = q_modules * 4 - k - S;
                        dss3.reverse4().write4(chunk.get_unchecked_mut(idx..));
                    }

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = AvxStoreF::f32_mul_add(2f32.as_(), dc4, -dss3);
                    unsafe {
                        mdc4.write4(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss5 = AvxStoreF::f32_mul_add(2f32.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k - S;
                        dss5.reverse4().write4(chunk.get_unchecked_mut(idx..));
                    }

                    dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        dc6.write4(chunk.get_unchecked_mut(idx..));
                    }

                    let dss6 = AvxStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        let idx = q_modules * 8 - k - S;
                        dss6.reverse4().write4(chunk.get_unchecked_mut(idx..));
                    }

                    dc8 = AvxStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        let idx = q_modules * 8 + k;
                        dc8.write4(chunk.get_unchecked_mut(idx..));
                    }

                    let dss9 = AvxStoreF::f32_mul_add(2., ds9, -dc8);
                    unsafe {
                        let idx = q_modules * 10 - k - S;
                        dss9.reverse4().write4(chunk.get_unchecked_mut(idx..));
                    }

                    dc10 = AvxStoreF::f32_mul_add(2., -dc10, -dss9);
                    unsafe {
                        let idx = q_modules * 10 + k;
                        dc10.write4(chunk.get_unchecked_mut(idx..));
                    }

                    let dss11 = AvxStoreF::f32_mul_add(2., -ds11, -dc10);
                    unsafe {
                        let idx = q_modules * 12 - k - S;
                        dss11.reverse4().write4(chunk.get_unchecked_mut(idx..));
                    }

                    dc12 = AvxStoreF::f32_mul_add(2., dc12, -dss11);
                    unsafe {
                        let idx = q_modules * 12 + k;
                        dc12.write4(chunk.get_unchecked_mut(idx..));
                    }
                } else if remainder == 3 {
                    const S: usize = 2;
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load3(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load3(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                            .reverse3();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R13_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R13_EVEN_TWIDDLE_5;
                    let mut dc6 = twiddled_dc * f32::R13_EVEN_TWIDDLE_1;
                    let mut dc8 = twiddled_dc * f32::R13_EVEN_TWIDDLE_4;
                    let mut dc10 = twiddled_dc * f32::R13_EVEN_TWIDDLE_2;
                    let mut dc12 = twiddled_dc * f32::R13_EVEN_TWIDDLE_3;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R13_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R13_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R13_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R13_ODD_TWIDDLE_3;
                    let mut ds9 = twiddled_ds * f32::R13_ODD_TWIDDLE_4;
                    let mut ds11 = twiddled_ds * -f32::R13_ODD_TWIDDLE_5;

                    {
                        let c_forward =
                            AvxStoreF::load3(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = AvxStoreF::load3(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - S..)
                        })
                        .reverse3();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load3(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = AvxStoreF::load3(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - S..)
                        })
                        .reverse3();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load3(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = AvxStoreF::load3(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - S..)
                        })
                        .reverse3();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load3(unsafe {
                            c_buffer.get_unchecked(q_modules * 4 + k..)
                        });
                        let s_forward = AvxStoreF::load3(unsafe {
                            s_buffer.get_unchecked(q_modules * 5 - k - S..)
                        })
                        .reverse3();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load3(unsafe {
                            c_buffer.get_unchecked(q_modules * 5 + k..)
                        });
                        let s_forward = AvxStoreF::load3(unsafe {
                            s_buffer.get_unchecked(q_modules * 6 - k - S..)
                        })
                        .reverse3();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 10) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 11) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 10) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 11) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds11);
                    }

                    let a0 = AvxStoreF::load3(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;
                    dc10 += a0;
                    dc12 += a0;

                    unsafe {
                        dc.write3(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse3()
                            .write3(chunk.get_unchecked_mut(q_modules * 2 - k - S..));
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = AvxStoreF::f32_mul_add(2f32.as_(), dc2, -dss1);
                    unsafe {
                        dc2.write3(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss3 = AvxStoreF::f32_mul_add(2f32.as_(), -ds3, -dc2);
                    unsafe {
                        let idx = q_modules * 4 - k - S;
                        dss3.reverse3().write3(chunk.get_unchecked_mut(idx..));
                    }

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = AvxStoreF::f32_mul_add(2f32.as_(), dc4, -dss3);
                    unsafe {
                        mdc4.write3(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss5 = AvxStoreF::f32_mul_add(2f32.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k - S;
                        dss5.reverse3().write3(chunk.get_unchecked_mut(idx..));
                    }

                    dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        dc6.write3(chunk.get_unchecked_mut(idx..));
                    }

                    let dss6 = AvxStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        let idx = q_modules * 8 - k - S;
                        dss6.reverse3().write3(chunk.get_unchecked_mut(idx..));
                    }

                    dc8 = AvxStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        let idx = q_modules * 8 + k;
                        dc8.write3(chunk.get_unchecked_mut(idx..));
                    }

                    let dss9 = AvxStoreF::f32_mul_add(2., ds9, -dc8);
                    unsafe {
                        let idx = q_modules * 10 - k - S;
                        dss9.reverse3().write3(chunk.get_unchecked_mut(idx..));
                    }

                    dc10 = AvxStoreF::f32_mul_add(2., -dc10, -dss9);
                    unsafe {
                        let idx = q_modules * 10 + k;
                        dc10.write3(chunk.get_unchecked_mut(idx..));
                    }

                    let dss11 = AvxStoreF::f32_mul_add(2., -ds11, -dc10);
                    unsafe {
                        let idx = q_modules * 12 - k - S;
                        dss11.reverse3().write3(chunk.get_unchecked_mut(idx..));
                    }

                    dc12 = AvxStoreF::f32_mul_add(2., dc12, -dss11);
                    unsafe {
                        let idx = q_modules * 12 + k;
                        dc12.write3(chunk.get_unchecked_mut(idx..));
                    }
                } else if remainder == 2 {
                    const S: usize = 1;
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load2(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load2(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                            .reverse2();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R13_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R13_EVEN_TWIDDLE_5;
                    let mut dc6 = twiddled_dc * f32::R13_EVEN_TWIDDLE_1;
                    let mut dc8 = twiddled_dc * f32::R13_EVEN_TWIDDLE_4;
                    let mut dc10 = twiddled_dc * f32::R13_EVEN_TWIDDLE_2;
                    let mut dc12 = twiddled_dc * f32::R13_EVEN_TWIDDLE_3;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R13_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R13_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R13_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R13_ODD_TWIDDLE_3;
                    let mut ds9 = twiddled_ds * f32::R13_ODD_TWIDDLE_4;
                    let mut ds11 = twiddled_ds * -f32::R13_ODD_TWIDDLE_5;

                    {
                        let c_forward =
                            AvxStoreF::load2(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = AvxStoreF::load2(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - S..)
                        })
                        .reverse2();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load2(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = AvxStoreF::load2(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - S..)
                        })
                        .reverse2();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load2(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = AvxStoreF::load2(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - S..)
                        })
                        .reverse2();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load2(unsafe {
                            c_buffer.get_unchecked(q_modules * 4 + k..)
                        });
                        let s_forward = AvxStoreF::load2(unsafe {
                            s_buffer.get_unchecked(q_modules * 5 - k - S..)
                        })
                        .reverse2();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load2(unsafe {
                            c_buffer.get_unchecked(q_modules * 5 + k..)
                        });
                        let s_forward = AvxStoreF::load2(unsafe {
                            s_buffer.get_unchecked(q_modules * 6 - k - S..)
                        })
                        .reverse2();

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 10) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 11) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 10) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 11) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds11);
                    }

                    let a0 = AvxStoreF::load2(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;
                    dc10 += a0;
                    dc12 += a0;

                    unsafe {
                        dc.write2(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.reverse2()
                            .write2(chunk.get_unchecked_mut(q_modules * 2 - k - S..));
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = AvxStoreF::f32_mul_add(2f32.as_(), dc2, -dss1);
                    unsafe {
                        dc2.write2(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss3 = AvxStoreF::f32_mul_add(2f32.as_(), -ds3, -dc2);
                    unsafe {
                        let idx = q_modules * 4 - k - S;
                        dss3.reverse2().write2(chunk.get_unchecked_mut(idx..));
                    }

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = AvxStoreF::f32_mul_add(2f32.as_(), dc4, -dss3);
                    unsafe {
                        mdc4.write2(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss5 = AvxStoreF::f32_mul_add(2f32.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k - S;
                        dss5.reverse2().write2(chunk.get_unchecked_mut(idx..));
                    }

                    dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        dc6.write2(chunk.get_unchecked_mut(idx..));
                    }

                    let dss6 = AvxStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        let idx = q_modules * 8 - k - S;
                        dss6.reverse2().write2(chunk.get_unchecked_mut(idx..));
                    }

                    dc8 = AvxStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        let idx = q_modules * 8 + k;
                        dc8.write2(chunk.get_unchecked_mut(idx..));
                    }

                    let dss9 = AvxStoreF::f32_mul_add(2., ds9, -dc8);
                    unsafe {
                        let idx = q_modules * 10 - k - S;
                        dss9.reverse2().write2(chunk.get_unchecked_mut(idx..));
                    }

                    dc10 = AvxStoreF::f32_mul_add(2., -dc10, -dss9);
                    unsafe {
                        let idx = q_modules * 10 + k;
                        dc10.write2(chunk.get_unchecked_mut(idx..));
                    }

                    let dss11 = AvxStoreF::f32_mul_add(2., -ds11, -dc10);
                    unsafe {
                        let idx = q_modules * 12 - k - S;
                        dss11.reverse2().write2(chunk.get_unchecked_mut(idx..));
                    }

                    dc12 = AvxStoreF::f32_mul_add(2., dc12, -dss11);
                    unsafe {
                        let idx = q_modules * 12 + k;
                        dc12.write2(chunk.get_unchecked_mut(idx..));
                    }
                } else if remainder == 1 {
                    const S: usize = 0;
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load1(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load1(unsafe { s_buffer.get_unchecked(q_modules - k - S..) });

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * f32::R13_EVEN_TWIDDLE_0;
                    let mut dc4 = twiddled_dc * f32::R13_EVEN_TWIDDLE_5;
                    let mut dc6 = twiddled_dc * f32::R13_EVEN_TWIDDLE_1;
                    let mut dc8 = twiddled_dc * f32::R13_EVEN_TWIDDLE_4;
                    let mut dc10 = twiddled_dc * f32::R13_EVEN_TWIDDLE_2;
                    let mut dc12 = twiddled_dc * f32::R13_EVEN_TWIDDLE_3;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let mut ds1 = twiddled_ds * f32::R13_ODD_TWIDDLE_0;
                    let mut ds3 = twiddled_ds * -f32::R13_ODD_TWIDDLE_1;
                    let mut ds5 = twiddled_ds * f32::R13_ODD_TWIDDLE_2;
                    let mut ds7 = twiddled_ds * -f32::R13_ODD_TWIDDLE_3;
                    let mut ds9 = twiddled_ds * f32::R13_ODD_TWIDDLE_4;
                    let mut ds11 = twiddled_ds * -f32::R13_ODD_TWIDDLE_5;

                    {
                        let c_forward =
                            AvxStoreF::load1(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                        let s_forward = AvxStoreF::load1(unsafe {
                            s_buffer.get_unchecked(q_modules * 2 - k - S..)
                        });

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load1(unsafe {
                            c_buffer.get_unchecked(q_modules * 2 + k..)
                        });
                        let s_forward = AvxStoreF::load1(unsafe {
                            s_buffer.get_unchecked(q_modules * 3 - k - S..)
                        });

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load1(unsafe {
                            c_buffer.get_unchecked(q_modules * 3 + k..)
                        });
                        let s_forward = AvxStoreF::load1(unsafe {
                            s_buffer.get_unchecked(q_modules * 4 - k - S..)
                        });

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load1(unsafe {
                            c_buffer.get_unchecked(q_modules * 4 + k..)
                        });
                        let s_forward = AvxStoreF::load1(unsafe {
                            s_buffer.get_unchecked(q_modules * 5 - k - S..)
                        });

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(-f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds11);
                    }

                    {
                        let c_forward = AvxStoreF::load1(unsafe {
                            c_buffer.get_unchecked(q_modules * 5 + k..)
                        });
                        let s_forward = AvxStoreF::load1(unsafe {
                            s_buffer.get_unchecked(q_modules * 6 - k - S..)
                        });

                        let rotation_twiddle_re =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 10) };
                        let rotation_twiddle_im =
                            unsafe { *self.rotation_layer.get_unchecked(uk + 11) };

                        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 10) };
                        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 11) };

                        let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                        let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                        let twiddled_dc = twiddle_re * rotated_dc1;
                        let twiddled_ds = twiddle_im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_5, twiddled_dc, dc2);
                        dc4 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_4, twiddled_dc, dc4);
                        dc6 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                        dc8 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                        dc10 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_1, twiddled_dc, dc10);
                        dc12 = AvxStoreF::f32_mul_add(f32::R13_EVEN_TWIDDLE_0, twiddled_dc, dc12);

                        ds1 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_5, twiddled_ds, ds1);
                        ds3 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_3, twiddled_ds, ds3);
                        ds5 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_1, twiddled_ds, ds5);
                        ds7 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_0, twiddled_ds, ds7);
                        ds9 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_2, twiddled_ds, ds9);
                        ds11 = AvxStoreF::f32_mul_add(f32::R13_ODD_TWIDDLE_4, twiddled_ds, ds11);
                    }

                    let a0 = AvxStoreF::load1(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;

                    dc2 = -(dc2 + a0);
                    dc4 += a0;
                    dc6 += a0;
                    dc8 += a0;
                    dc10 += a0;
                    dc12 += a0;

                    unsafe {
                        dc.write1(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.write1(chunk.get_unchecked_mut(q_modules * 2 - k - S..));
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = AvxStoreF::f32_mul_add(2f32.as_(), dc2, -dss1);
                    unsafe {
                        dc2.write1(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss3 = AvxStoreF::f32_mul_add(2f32.as_(), -ds3, -dc2);
                    unsafe {
                        let idx = q_modules * 4 - k - S;
                        dss3.write1(chunk.get_unchecked_mut(idx..));
                    }

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = AvxStoreF::f32_mul_add(2f32.as_(), dc4, -dss3);
                    unsafe {
                        mdc4.write1(chunk.get_unchecked_mut(idx1..));
                    }

                    let dss5 = AvxStoreF::f32_mul_add(2f32.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k - S;
                        dss5.write1(chunk.get_unchecked_mut(idx..));
                    }

                    dc6 = AvxStoreF::f32_mul_add(2., -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        dc6.write1(chunk.get_unchecked_mut(idx..));
                    }

                    let dss6 = AvxStoreF::f32_mul_add(2., -ds7, -dc6);
                    unsafe {
                        let idx = q_modules * 8 - k - S;
                        dss6.write1(chunk.get_unchecked_mut(idx..));
                    }

                    dc8 = AvxStoreF::f32_mul_add(2., dc8, -dss6);
                    unsafe {
                        let idx = q_modules * 8 + k;
                        dc8.write1(chunk.get_unchecked_mut(idx..));
                    }

                    let dss9 = AvxStoreF::f32_mul_add(2., ds9, -dc8);
                    unsafe {
                        let idx = q_modules * 10 - k - S;
                        dss9.write1(chunk.get_unchecked_mut(idx..));
                    }

                    dc10 = AvxStoreF::f32_mul_add(2., -dc10, -dss9);
                    unsafe {
                        let idx = q_modules * 10 + k;
                        dc10.write1(chunk.get_unchecked_mut(idx..));
                    }

                    let dss11 = AvxStoreF::f32_mul_add(2., -ds11, -dc10);
                    unsafe {
                        let idx = q_modules * 12 - k - S;
                        dss11.write1(chunk.get_unchecked_mut(idx..));
                    }

                    dc12 = AvxStoreF::f32_mul_add(2., dc12, -dss11);
                    unsafe {
                        let idx = q_modules * 12 + k;
                        dc12.write1(chunk.get_unchecked_mut(idx..));
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct2_f32;

    #[test]
    fn test_radix9_dct() {
        let mut input = vec![0.; 11];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32 + rand::random::<f32>() * 10.0;
        }
        let mut input = vec![
            7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256,
            12.010594, 7.6871257, 1.2637726, 7.6871257, 1.2637726,
            ///
            7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256,
            12.010594, 7.6871257, 1.2637726, 7.6871257, 1.2637726,
        ];
        let mut reference_input = input.clone();
        // let rr = Pxdct::make_dct2_f32(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2_f32(&reference_input);
        let bf =
            AvxDct2MixedRadix13f::new(input.len(), Pxdct::make_dct2_f32(input.len() / 13).unwrap())
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
                if (src - r0).abs() > 1e-1 {
                    println!(
                        "Difference must be < {}, but it was {}, at position {i}",
                        1e-1,
                        (src - r0).abs()
                    )
                }
            });
    }
}
