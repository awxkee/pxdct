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
use crate::avx::stored::AvxStoreD;
use crate::avx::util::{boring_avx_mixed_radix, fma};
use crate::bidirectional::BidirectionalStore;
use crate::dct2::prime_butterflies::MixedRadix11Sample;
use crate::dct2::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix11_rotation_twiddles_avxd(q_modules: usize, len: usize) -> Vec<AvxStoreD> {
    let simd_groups = q_modules.div_ceil(4);
    let main_q = 11usize;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 10 * inner_groups);

    let working_modules = q_modules - 1;

    let mut uk = 0usize;
    while uk + 4 <= working_modules {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for m in 0..inner_groups {
            for i in 0..4 {
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
            twiddles.push(AvxStoreD::load(array_re.as_ref()));
            twiddles.push(AvxStoreD::load(array_im.as_ref()));
        }

        uk += 4;
    }

    let remainder = working_modules - (working_modules / 4) * 4;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
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
            twiddles.push(AvxStoreD::load(array_re.as_ref()));
            twiddles.push(AvxStoreD::load(array_im.as_ref()));
        }
    }

    twiddles
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix11_cos_twiddles_avxd(q_modules: usize, len: usize) -> Vec<AvxStoreD> {
    let main_q = 11usize;
    let simd_groups = q_modules.div_ceil(4);
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 10 * inner_groups);

    let working_modules = q_modules - 1;

    let mut uk = 0usize;
    while uk + 4 <= working_modules {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];

        for m in 0..inner_groups {
            for i in 0..4 {
                array_re[i] = radixq_cos_twiddle(main_q, m, (k + i).as_(), len);
                array_im[i] = radixq_cos_twiddle(main_q, m, (q_modules - (k + i)).as_(), len);
            }

            twiddles.push(AvxStoreD::load(array_re.as_ref()));
            twiddles.push(AvxStoreD::load(array_im.as_ref()));
        }

        uk += 4;
    }

    let remainder = working_modules - (working_modules / 4) * 4;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];

        for m in 0..inner_groups {
            for i in 0..remainder {
                array_re[i] = radixq_cos_twiddle(main_q, m, (k + i).as_(), len);
                array_im[i] = radixq_cos_twiddle(main_q, m, (q_modules - (k + i)).as_(), len);
            }

            twiddles.push(AvxStoreD::load(array_re.as_ref()));
            twiddles.push(AvxStoreD::load(array_im.as_ref()));
        }
    }

    twiddles
}

pub(crate) struct AvxDct2MixedRadix11d {
    rotation_layer: Vec<AvxStoreD>,
    cos_twiddles: Vec<AvxStoreD>,
    inner_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    execution_length: usize,
}

impl AvxDct2MixedRadix11d {
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<AvxDct2MixedRadix11d, PxdctError> {
        assert!(
            len.is_multiple_of(11),
            "Mixed radix 9 should not be called on sizes no divisible by 9"
        );

        let q_modules = len / 11;

        let inner_dct_scratch_size = inner_dct.scratch_size();

        Ok(AvxDct2MixedRadix11d {
            rotation_layer: unsafe { dct2_radix11_rotation_twiddles_avxd(q_modules, len) },
            cos_twiddles: unsafe { dct2_radix11_cos_twiddles_avxd(q_modules, len) },
            execution_length: len,
            inner_dct,
            inner_dct_scratch_size,
        })
    }
}

boring_avx_mixed_radix!(AvxDct2MixedRadix11d, f64);

impl AvxDct2MixedRadix11d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        scratch: &mut [f64],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let q_modules = self.execution_length / 11;
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 5);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 11 + 5];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(q_modules)
            .zip(s_buffer.chunks_exact_mut(q_modules))
            .enumerate()
        {
            let mut sign = f64::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[11 * n + m];
                let u1 = data[11 * n + 11 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-II to all buffers (A, C₀, C₁, S₀, S₁)
        self.inner_dct
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 5);

        {
            // Step 3: Recombine transformed buffers with twiddle factors

            // Handle k=0 case (DC and low frequencies)
            let qc = c_buffer[0];
            let mut c0 = qc; // Component C₀ (position 0)
            let mut c1 = qc * f64::R11_EVEN_TWIDDLE_0; // Component C₂ (position 2, uses j=2)
            let mut c2 = qc * f64::R11_EVEN_TWIDDLE_4; // Component C₄ (position 4, uses j=4)
            let mut c3 = qc * f64::R11_EVEN_TWIDDLE_1; // Component C6 (position 6, uses j=6)
            let mut c4 = qc * f64::R11_EVEN_TWIDDLE_3; // Component C8 (position 8, uses j=8)
            let mut c5 = qc * f64::R11_EVEN_TWIDDLE_2;

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * f64::R11_ODD_TWIDDLE_0;
            let mut s1 = s0_twiddled * -f64::R11_ODD_TWIDDLE_1;
            let mut s2 = s0_twiddled * f64::R11_ODD_TWIDDLE_2;
            let mut s3 = s0_twiddled * -f64::R11_ODD_TWIDDLE_3;
            let mut s4 = s0_twiddled * f64::R11_ODD_TWIDDLE_4;

            let ci = unsafe { *c_buffer.get_unchecked(q_modules) };
            let si = unsafe { *s_buffer.get_unchecked(q_modules) };

            let ci2 = unsafe { *c_buffer.get_unchecked(q_modules * 2) };
            let si2 = unsafe { *s_buffer.get_unchecked(q_modules * 2) };

            let ci3 = unsafe { *c_buffer.get_unchecked(q_modules * 3) };
            let si3 = unsafe { *s_buffer.get_unchecked(q_modules * 3) };

            let ci4 = unsafe { *c_buffer.get_unchecked(q_modules * 4) };
            let si4 = unsafe { *s_buffer.get_unchecked(q_modules * 4) };

            c0 = ci + c0 + ci2 + ci3 + ci4;

            let a0 = a_buffer[0];

            let dc = c0 + a0;
            data[0] = dc;

            c1 = fma(ci, f64::R11_EVEN_TWIDDLE_1, c1);
            c1 = fma(ci2, f64::R11_EVEN_TWIDDLE_2, c1);
            c1 = fma(ci3, f64::R11_EVEN_TWIDDLE_3, c1);
            c1 = fma(ci4, f64::R11_EVEN_TWIDDLE_4, c1);

            c2 = fma(ci, f64::R11_EVEN_TWIDDLE_2, c2);
            c2 = fma(ci2, f64::R11_EVEN_TWIDDLE_0, c2);
            c2 = fma(ci3, f64::R11_EVEN_TWIDDLE_1, c2);
            c2 = fma(ci4, f64::R11_EVEN_TWIDDLE_3, c2);

            let dc2 = c2 + a0;
            data[q_modules * 4] = dc2;

            c3 = fma(ci, f64::R11_EVEN_TWIDDLE_4, c3);
            c3 = fma(ci2, f64::R11_EVEN_TWIDDLE_3, c3);
            c3 = fma(ci3, f64::R11_EVEN_TWIDDLE_0, c3);
            c3 = fma(ci4, f64::R11_EVEN_TWIDDLE_2, c3);
            let dc3 = c3 + a0;
            data[q_modules * 6] = -dc3;

            c4 = fma(ci, f64::R11_EVEN_TWIDDLE_0, c4);
            c4 = fma(ci2, f64::R11_EVEN_TWIDDLE_4, c4);
            c4 = fma(ci3, f64::R11_EVEN_TWIDDLE_2, c4);
            c4 = fma(ci4, f64::R11_EVEN_TWIDDLE_1, c4);
            let dc4 = c4 + a0;
            data[q_modules * 8] = dc4;

            c5 = fma(ci, f64::R11_EVEN_TWIDDLE_3, c5);
            c5 = fma(ci2, f64::R11_EVEN_TWIDDLE_1, c5);
            c5 = fma(ci3, f64::R11_EVEN_TWIDDLE_4, c5);
            c5 = fma(ci4, f64::R11_EVEN_TWIDDLE_0, c5);

            let dc5 = c5 + a0;
            data[q_modules * 10] = -dc5;

            s0 = fma(si, f64::R11_ODD_TWIDDLE_1, s0);
            s0 = fma(si2, f64::R11_ODD_TWIDDLE_2, s0);
            s0 = fma(si3, f64::R11_ODD_TWIDDLE_3, s0);
            s0 = fma(si4, f64::R11_ODD_TWIDDLE_4, s0);

            s1 = fma(si, -f64::R11_ODD_TWIDDLE_4, s1);
            s1 = fma(si2, f64::R11_ODD_TWIDDLE_3, s1);
            s1 = fma(si3, f64::R11_ODD_TWIDDLE_0, s1);
            s1 = fma(si4, f64::R11_ODD_TWIDDLE_2, s1);

            s2 = fma(si, -f64::R11_ODD_TWIDDLE_3, s2);
            s2 = fma(si2, -f64::R11_ODD_TWIDDLE_1, s2);
            s2 = fma(si3, f64::R11_ODD_TWIDDLE_4, s2);
            s2 = fma(si4, f64::R11_ODD_TWIDDLE_0, s2);

            s3 = fma(si, f64::R11_ODD_TWIDDLE_0, s3);
            s3 = fma(si2, -f64::R11_ODD_TWIDDLE_4, s3);
            s3 = fma(si3, -f64::R11_ODD_TWIDDLE_2, s3);
            s3 = fma(si4, f64::R11_ODD_TWIDDLE_1, s3);

            s4 = fma(si, -f64::R11_ODD_TWIDDLE_2, s4);
            s4 = fma(si2, f64::R11_ODD_TWIDDLE_0, s4);
            s4 = fma(si3, -f64::R11_ODD_TWIDDLE_1, s4);
            s4 = fma(si4, f64::R11_ODD_TWIDDLE_3, s4);

            data[q_modules * 3] = -s1;
            data[q_modules] = s0;
            let qid2 = -(c1 + a0); // negated 2j
            data[q_modules * 2] = qid2;
            data[q_modules * 5] = s2;
            data[q_modules * 7] = -s3;
            data[q_modules * 9] = s4;

            let mut uk = 0usize;
            let mut k = 1usize;

            while k + 4 <= q_modules {
                const S: usize = 3;
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                let c_forward = AvxStoreD::load(unsafe { c_buffer.get_unchecked(k..) });
                let s_forward =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                        .reverse();

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R11_EVEN_TWIDDLE_0;
                let mut dc4 = twiddled_dc * f64::R11_EVEN_TWIDDLE_4;
                let mut dc6 = twiddled_dc * f64::R11_EVEN_TWIDDLE_1;
                let mut dc8 = twiddled_dc * f64::R11_EVEN_TWIDDLE_3;
                let mut dc10 = twiddled_dc * f64::R11_EVEN_TWIDDLE_2;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R11_ODD_TWIDDLE_0;
                let mut ds3 = twiddled_ds * -f64::R11_ODD_TWIDDLE_1;
                let mut ds5 = twiddled_ds * f64::R11_ODD_TWIDDLE_2;
                let mut ds7 = twiddled_ds * -f64::R11_ODD_TWIDDLE_3;
                let mut ds9 = twiddled_ds * f64::R11_ODD_TWIDDLE_4;

                {
                    let c_forward =
                        AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                    let s_forward =
                        AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 2 - k - S..) })
                            .reverse();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                    let s_forward =
                        AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 3 - k - S..) })
                            .reverse();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                    let s_forward =
                        AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 4 - k - S..) })
                            .reverse();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                    let s_forward =
                        AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 5 - k - S..) })
                            .reverse();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds9);
                }

                let a0 = AvxStoreD::load(unsafe { a_buffer.get_unchecked(k..) });
                let dc = dc0 + a0;

                dc2 = -(dc2 + a0);
                dc4 += a0;
                dc6 += a0;
                dc8 += a0;
                dc10 += a0;

                dc.write(data.slice_from_mut(k..));

                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                dss1.reverse()
                    .write(data.slice_from_mut(q_modules * 2 - k - S..));

                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write(data.slice_from_mut(q_modules * 2 + k..));

                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                dss3.reverse()
                    .write(data.slice_from_mut(q_modules * 4 - k - S..));

                let idx1 = q_modules * 4 + k;
                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write(data.slice_from_mut(idx1..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                let idx = q_modules * 6 - k;
                dss5.reverse().write(data.slice_from_mut(idx - S..));

                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
                let idx = q_modules * 6 + k;
                dc6.write(data.slice_from_mut(idx..));

                let dss6 = AvxStoreD::f64_mul_add(2., -ds7, -dc6);
                let idx = q_modules * 8 - k;
                dss6.reverse().write(data.slice_from_mut(idx - S..));

                dc8 = AvxStoreD::f64_mul_add(2., dc8, -dss6);
                let idx = q_modules * 8 + k;
                dc8.write(data.slice_from_mut(idx..));

                let dss9 = AvxStoreD::f64_mul_add(2., ds9, -dc8);
                let idx = q_modules * 10 - k;
                dss9.reverse().write(data.slice_from_mut(idx - S..));

                dc10 = AvxStoreD::f64_mul_add(2., -dc10, -dss9);
                let idx = q_modules * 10 + k;
                dc10.write(data.slice_from_mut(idx..));

                k += 4;
                uk += 10;
            }

            let remainder = q_modules - k;
            if remainder == 3 {
                const S: usize = 2;
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                let c_forward = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(k..) });
                let s_forward =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                        .reverse3();

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R11_EVEN_TWIDDLE_0;
                let mut dc4 = twiddled_dc * f64::R11_EVEN_TWIDDLE_4;
                let mut dc6 = twiddled_dc * f64::R11_EVEN_TWIDDLE_1;
                let mut dc8 = twiddled_dc * f64::R11_EVEN_TWIDDLE_3;
                let mut dc10 = twiddled_dc * f64::R11_EVEN_TWIDDLE_2;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R11_ODD_TWIDDLE_0;
                let mut ds3 = twiddled_ds * -f64::R11_ODD_TWIDDLE_1;
                let mut ds5 = twiddled_ds * f64::R11_ODD_TWIDDLE_2;
                let mut ds7 = twiddled_ds * -f64::R11_ODD_TWIDDLE_3;
                let mut ds9 = twiddled_ds * f64::R11_ODD_TWIDDLE_4;

                {
                    let c_forward =
                        AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                    let s_forward = AvxStoreD::load3(unsafe {
                        s_buffer.get_unchecked(q_modules * 2 - k - S..)
                    })
                    .reverse3();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                    let s_forward = AvxStoreD::load3(unsafe {
                        s_buffer.get_unchecked(q_modules * 3 - k - S..)
                    })
                    .reverse3();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                    let s_forward = AvxStoreD::load3(unsafe {
                        s_buffer.get_unchecked(q_modules * 4 - k - S..)
                    })
                    .reverse3();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                    let s_forward = AvxStoreD::load3(unsafe {
                        s_buffer.get_unchecked(q_modules * 5 - k - S..)
                    })
                    .reverse3();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds9);
                }

                let a0 = AvxStoreD::load3(unsafe { a_buffer.get_unchecked(k..) });
                let dc = dc0 + a0;

                dc2 = -(dc2 + a0);
                dc4 += a0;
                dc6 += a0;
                dc8 += a0;
                dc10 += a0;

                dc.write3(data.slice_from_mut(k..));

                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                dss1.reverse3()
                    .write3(data.slice_from_mut(q_modules * 2 - k - S..));

                let idx1 = q_modules * 2 + k;
                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write3(data.slice_from_mut(idx1..));

                let idx = q_modules * 4 - k;
                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                dss3.reverse3().write3(data.slice_from_mut(idx - S..));

                let idx1 = q_modules * 4 + k;
                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write3(data.slice_from_mut(idx1..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                let idx = q_modules * 6 - k;
                dss5.reverse3().write3(data.slice_from_mut(idx - S..));

                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
                let idx = q_modules * 6 + k;
                dc6.write3(data.slice_from_mut(idx..));

                let dss6 = AvxStoreD::f64_mul_add(2., -ds7, -dc6);
                let idx = q_modules * 8 - k;
                dss6.reverse3().write3(data.slice_from_mut(idx - S..));

                dc8 = AvxStoreD::f64_mul_add(2., dc8, -dss6);
                let idx = q_modules * 8 + k;
                dc8.write3(data.slice_from_mut(idx..));

                let dss9 = AvxStoreD::f64_mul_add(2., ds9, -dc8);
                let idx = q_modules * 10 - k;
                dss9.reverse3().write3(data.slice_from_mut(idx - S..));

                dc10 = AvxStoreD::f64_mul_add(2., -dc10, -dss9);
                let idx = q_modules * 10 + k;
                dc10.write3(data.slice_from_mut(idx..));
            } else if remainder == 2 {
                const S: usize = 1;
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                let c_forward = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(k..) });
                let s_forward =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                        .reverse2();

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R11_EVEN_TWIDDLE_0;
                let mut dc4 = twiddled_dc * f64::R11_EVEN_TWIDDLE_4;
                let mut dc6 = twiddled_dc * f64::R11_EVEN_TWIDDLE_1;
                let mut dc8 = twiddled_dc * f64::R11_EVEN_TWIDDLE_3;
                let mut dc10 = twiddled_dc * f64::R11_EVEN_TWIDDLE_2;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R11_ODD_TWIDDLE_0;
                let mut ds3 = twiddled_ds * -f64::R11_ODD_TWIDDLE_1;
                let mut ds5 = twiddled_ds * f64::R11_ODD_TWIDDLE_2;
                let mut ds7 = twiddled_ds * -f64::R11_ODD_TWIDDLE_3;
                let mut ds9 = twiddled_ds * f64::R11_ODD_TWIDDLE_4;

                {
                    let c_forward =
                        AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                    let s_forward = AvxStoreD::load2(unsafe {
                        s_buffer.get_unchecked(q_modules * 2 - k - S..)
                    })
                    .reverse2();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                    let s_forward = AvxStoreD::load2(unsafe {
                        s_buffer.get_unchecked(q_modules * 3 - k - S..)
                    })
                    .reverse2();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                    let s_forward = AvxStoreD::load2(unsafe {
                        s_buffer.get_unchecked(q_modules * 4 - k - S..)
                    })
                    .reverse2();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                    let s_forward = AvxStoreD::load2(unsafe {
                        s_buffer.get_unchecked(q_modules * 5 - k - S..)
                    })
                    .reverse2();

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds9);
                }

                let a0 = AvxStoreD::load2(unsafe { a_buffer.get_unchecked(k..) });
                let dc = dc0 + a0;

                dc2 = -(dc2 + a0);
                dc4 += a0;
                dc6 += a0;
                dc8 += a0;
                dc10 += a0;

                dc.write2(data.slice_from_mut(k..));

                let idx = q_modules * 2 - k;
                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                dss1.reverse2().write2(data.slice_from_mut(idx - S..));

                let idx1 = q_modules * 2 + k;
                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write2(data.slice_from_mut(idx1..));

                let idx = q_modules * 4 - k;
                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                dss3.reverse2().write2(data.slice_from_mut(idx - S..));

                let idx1 = q_modules * 4 + k;
                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write2(data.slice_from_mut(idx1..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                let idx = q_modules * 6 - k;
                dss5.reverse2().write2(data.slice_from_mut(idx - S..));

                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
                let idx = q_modules * 6 + k;
                dc6.write2(data.slice_from_mut(idx..));

                let dss6 = AvxStoreD::f64_mul_add(2., -ds7, -dc6);
                let idx = q_modules * 8 - k;
                dss6.reverse2().write2(data.slice_from_mut(idx - S..));

                dc8 = AvxStoreD::f64_mul_add(2., dc8, -dss6);
                let idx = q_modules * 8 + k;
                dc8.write2(data.slice_from_mut(idx..));

                let dss9 = AvxStoreD::f64_mul_add(2., ds9, -dc8);
                let idx = q_modules * 10 - k;
                dss9.reverse2().write2(data.slice_from_mut(idx - S..));

                dc10 = AvxStoreD::f64_mul_add(2., -dc10, -dss9);
                let idx = q_modules * 10 + k;
                dc10.write2(data.slice_from_mut(idx..));
            } else if remainder == 1 {
                const S: usize = 0;
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                let c_forward = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(k..) });
                let s_forward =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules - k - S..) });

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R11_EVEN_TWIDDLE_0;
                let mut dc4 = twiddled_dc * f64::R11_EVEN_TWIDDLE_4;
                let mut dc6 = twiddled_dc * f64::R11_EVEN_TWIDDLE_1;
                let mut dc8 = twiddled_dc * f64::R11_EVEN_TWIDDLE_3;
                let mut dc10 = twiddled_dc * f64::R11_EVEN_TWIDDLE_2;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R11_ODD_TWIDDLE_0;
                let mut ds3 = twiddled_ds * -f64::R11_ODD_TWIDDLE_1;
                let mut ds5 = twiddled_ds * f64::R11_ODD_TWIDDLE_2;
                let mut ds7 = twiddled_ds * -f64::R11_ODD_TWIDDLE_3;
                let mut ds9 = twiddled_ds * f64::R11_ODD_TWIDDLE_4;

                {
                    let c_forward =
                        AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                    let s_forward = AvxStoreD::load1(unsafe {
                        s_buffer.get_unchecked(q_modules * 2 - k - S..)
                    });

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                    let s_forward = AvxStoreD::load1(unsafe {
                        s_buffer.get_unchecked(q_modules * 3 - k - S..)
                    });

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                    let s_forward = AvxStoreD::load1(unsafe {
                        s_buffer.get_unchecked(q_modules * 4 - k - S..)
                    });

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 6) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 7) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 6) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 7) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(-f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds9);
                }

                {
                    let c_forward =
                        AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                    let s_forward = AvxStoreD::load1(unsafe {
                        s_buffer.get_unchecked(q_modules * 5 - k - S..)
                    });

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 8) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 9) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 8) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 9) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_4, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_3, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_2, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_1, twiddled_dc, dc8);
                    dc10 = AvxStoreD::f64_mul_add(f64::R11_EVEN_TWIDDLE_0, twiddled_dc, dc10);

                    ds1 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_4, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_2, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_0, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_1, twiddled_ds, ds7);
                    ds9 = AvxStoreD::f64_mul_add(f64::R11_ODD_TWIDDLE_3, twiddled_ds, ds9);
                }

                let a0 = AvxStoreD::load1(unsafe { a_buffer.get_unchecked(k..) });
                let dc = dc0 + a0;

                dc2 = -(dc2 + a0);
                dc4 += a0;
                dc6 += a0;
                dc8 += a0;
                dc10 += a0;

                dc.write1(data.slice_from_mut(k..));

                let idx = q_modules * 2 - k;
                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                dss1.write1(data.slice_from_mut(idx - S..));

                let idx1 = q_modules * 2 + k;
                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write1(data.slice_from_mut(idx1..));

                let idx = q_modules * 4 - k;
                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                dss3.write1(data.slice_from_mut(idx - S..));

                let idx1 = q_modules * 4 + k;
                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write1(data.slice_from_mut(idx1..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                let idx = q_modules * 6 - k;
                dss5.write1(data.slice_from_mut(idx - S..));

                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
                let idx = q_modules * 6 + k;
                dc6.write1(data.slice_from_mut(idx..));

                let dss6 = AvxStoreD::f64_mul_add(2., -ds7, -dc6);
                let idx = q_modules * 8 - k;
                dss6.write1(data.slice_from_mut(idx - S..));

                dc8 = AvxStoreD::f64_mul_add(2., dc8, -dss6);
                let idx = q_modules * 8 + k;
                dc8.write1(data.slice_from_mut(idx..));

                let dss9 = AvxStoreD::f64_mul_add(2., ds9, -dc8);
                let idx = q_modules * 10 - k;
                dss9.write1(data.slice_from_mut(idx - S..));

                dc10 = AvxStoreD::f64_mul_add(2., -dc10, -dss9);
                let idx = q_modules * 10 + k;
                dc10.write1(data.slice_from_mut(idx..));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct2;
    use crate::util::has_valid_avx;

    #[test]
    fn test_radix11_dct() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0.; 143];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f64 + rand::random::<f64>() * 10.0;
        }
        // let mut input = vec![
        //     7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256,
        //     12.010594, 7.6871257, 1.2637726, 7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953,
        //     12.343984, 9.859292, 15.516256, 12.010594, 7.6871257, 1.2637726, 7.6871257, 1.2637726,
        //     11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256, 12.010594, 7.6871257,
        //     1.2637726, 7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292,
        //     15.516256, 12.010594, 7.6871257, 1.2637726,
        // ];
        let mut reference_input = input.clone();
        // let rr = Pxdcf64::make_dct2_f64(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2(&reference_input);
        let bf =
            AvxDct2MixedRadix11d::new(input.len(), Pxdct::make_dct2_f64(input.len() / 11).unwrap())
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
