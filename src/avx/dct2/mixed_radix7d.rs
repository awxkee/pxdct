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
use crate::dct2::{MixedRadix7Sample, radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix7_rotation_twiddles_avxd(q_modules: usize, len: usize) -> Vec<AvxStoreD> {
    let simd_groups = q_modules.div_ceil(4);
    let main_q = 7usize;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 6 * inner_groups);

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

        twiddles.push(AvxStoreD::set_values(
            layer0.re, layer1.re, layer2.re, layer3.re,
        ));
        twiddles.push(AvxStoreD::set_values(
            layer0.im, layer1.im, layer2.im, layer3.im,
        ));

        let layer0 = radixq_rotation_twiddle(main_q, 1, k.as_(), (q_modules - k).as_(), len);
        let layer1 =
            radixq_rotation_twiddle(main_q, 1, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
        let layer2 =
            radixq_rotation_twiddle(main_q, 1, (k + 2).as_(), (q_modules - (k + 2)).as_(), len);
        let layer3 =
            radixq_rotation_twiddle(main_q, 1, (k + 3).as_(), (q_modules - (k + 3)).as_(), len);

        twiddles.push(AvxStoreD::set_values(
            layer0.re, layer1.re, layer2.re, layer3.re,
        ));
        twiddles.push(AvxStoreD::set_values(
            layer0.im, layer1.im, layer2.im, layer3.im,
        ));

        let layer0 = radixq_rotation_twiddle(main_q, 2, k.as_(), (q_modules - k).as_(), len);
        let layer1 =
            radixq_rotation_twiddle(main_q, 2, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
        let layer2 =
            radixq_rotation_twiddle(main_q, 2, (k + 2).as_(), (q_modules - (k + 2)).as_(), len);
        let layer3 =
            radixq_rotation_twiddle(main_q, 2, (k + 3).as_(), (q_modules - (k + 3)).as_(), len);

        twiddles.push(AvxStoreD::set_values(
            layer0.re, layer1.re, layer2.re, layer3.re,
        ));
        twiddles.push(AvxStoreD::set_values(
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

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));

        for i in 0..remainder {
            let layer =
                radixq_rotation_twiddle(main_q, 1, (k + i).as_(), (q_modules - (k + i)).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));

        for i in 0..remainder {
            let layer =
                radixq_rotation_twiddle(main_q, 2, (k + i).as_(), (q_modules - (k + i)).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));
    }

    twiddles
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix7_cos_twiddles_avxd(q_modules: usize, len: usize) -> Vec<AvxStoreD> {
    let main_q = 7usize;
    let simd_groups = q_modules.div_ceil(4);
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 6 * inner_groups);

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

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));

        for i in 0..4 {
            array_re[i] = radixq_cos_twiddle(main_q, 1, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 1, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));

        for i in 0..4 {
            array_re[i] = radixq_cos_twiddle(main_q, 2, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 2, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));

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

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));

        for i in 0..remainder {
            array_re[i] = radixq_cos_twiddle(main_q, 1, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 1, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));

        for i in 0..remainder {
            array_re[i] = radixq_cos_twiddle(main_q, 2, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 2, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(AvxStoreD::load(array_re.as_ref()));
        twiddles.push(AvxStoreD::load(array_im.as_ref()));
    }

    twiddles
}

pub(crate) struct AvxDct2MixedRadix7d {
    rotation_layer: Vec<AvxStoreD>,
    cos_twiddles: Vec<AvxStoreD>,
    inner_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    execution_length: usize,
}

impl AvxDct2MixedRadix7d {
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<AvxDct2MixedRadix7d, PxdctError> {
        assert!(
            len.is_multiple_of(7),
            "Mixed radix 5 should not be called on sizes no divisible by 5"
        );

        let q_modules = len / 7;

        // always 3 inner groups in Radix-7

        // Precompute rotation twiddles for k≥1
        // Format: [m0_k1, m1_k1, m0_k2, m1_k2, ...]
        let rotation_layer = unsafe { dct2_radix7_rotation_twiddles_avxd(q_modules, len) };

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let cos_twiddles = unsafe { dct2_radix7_cos_twiddles_avxd(q_modules, len) };

        let inner_dct_scratch_size = inner_dct.scratch_size();

        Ok(AvxDct2MixedRadix7d {
            rotation_layer,
            inner_dct,
            inner_dct_scratch_size,
            cos_twiddles,
            execution_length: len,
        })
    }
}

boring_avx_mixed_radix!(AvxDct2MixedRadix7d, f64);

impl AvxDct2MixedRadix7d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        scratch: &mut [f64],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let q_modules = self.execution_length / 7;
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 3);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 7 + 3];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(q_modules)
            .zip(s_buffer.chunks_exact_mut(q_modules))
            .enumerate()
        {
            let mut sign = f64::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[7 * n + m];
                let u1 = data[7 * n + 7 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-II to all buffers (A, C₀, C₁, S₀, S₁)
        self.inner_dct
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 3);

        {
            // Step 3: Recombine transformed buffers with twiddle factors

            // Handle k=0 case (DC and low frequencies)
            let qc = c_buffer[0];
            let mut c0 = qc; // Component C₀ (position 0)
            let mut c1 = qc * f64::R7_COS_EVEN2_M0; // Component C₂ (position 2, uses j=2)
            let mut c2 = qc * f64::R7_COS_EVEN2_M2; // Component C₄ (position 4, uses j=4)
            let mut c3 = qc * f64::R7_COS_EVEN2_M1; // Component C6 (position 6, uses j=6)

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * f64::R7_SIN_ODD0_M0;
            let mut s1 = s0_twiddled * f64::R7_SIN_ODD1_M0;
            let mut s2 = s0_twiddled * f64::R7_SIN_ODD2_M0;

            {
                let ci = unsafe { *c_buffer.get_unchecked(q_modules) };
                let si = unsafe { *s_buffer.get_unchecked(q_modules) };

                let ci2 = unsafe { *c_buffer.get_unchecked(q_modules * 2) };
                let si2 = unsafe { *s_buffer.get_unchecked(q_modules * 2) };

                c0 = ci + c0 + ci2;

                c1 = fma(ci, f64::R7_COS_EVEN2_M1, c1);
                c1 = fma(ci2, f64::R7_COS_EVEN2_M2, c1);

                c2 = fma(ci, f64::R7_COS_EVEN2_M0, c2);
                c2 = fma(ci2, f64::R7_COS_EVEN2_M1, c2);

                c3 = fma(ci, f64::R7_COS_EVEN2_M2, c3);
                c3 = fma(ci2, f64::R7_COS_EVEN2_M0, c3);

                s0 = fma(si, f64::R7_SIN_ODD0_M1, s0);
                s0 = fma(si2, f64::R7_SIN_ODD0_M2, s0);

                s1 = fma(si, f64::R7_SIN_ODD1_M1, s1);
                s1 = fma(si2, f64::R7_SIN_ODD1_M2, s1);

                s2 = fma(si, f64::R7_SIN_ODD2_M1, s2);
                s2 = fma(si2, f64::R7_SIN_ODD2_M2, s2);
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

            let mut k = 1usize;
            let mut uk = 0usize;

            // Step 4: Handle k≥1 cases with rotation twiddles
            while k + 4 <= q_modules {
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                let c_forward = AvxStoreD::load(unsafe { c_buffer.get_unchecked(k..) });
                let s_forward =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules - k - 3..) })
                        .reverse();

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R7_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f64::R7_COS_EVEN2_M2;
                let mut dc6 = twiddled_dc * f64::R7_COS_EVEN2_M1;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R7_SIN_ODD0_M0;
                let mut ds3 = twiddled_ds * f64::R7_SIN_ODD1_M0;
                let mut ds5 = twiddled_ds * f64::R7_SIN_ODD2_M0;

                {
                    let c_forward =
                        AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                    let s_forward =
                        AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 2 - k - 3..) })
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
                    dc2 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M2, twiddled_dc, dc6);

                    ds1 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD0_M1, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD1_M1, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD2_M1, twiddled_ds, ds5);
                }

                {
                    let c_forward =
                        AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                    let s_forward =
                        AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 3 - k - 3..) })
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
                    dc2 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M0, twiddled_dc, dc6);

                    ds1 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD0_M2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD1_M2, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD2_M2, twiddled_ds, ds5);
                }

                let a0 = AvxStoreD::load(unsafe { a_buffer.get_unchecked(k..) });
                let dc = dc0 + a0;
                dc.write(data.slice_from_mut(k..));

                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                let q = dss1.reverse();
                q.write(data.slice_from_mut(q_modules * 2 - k - 3..));

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write(data.slice_from_mut(q_modules * 2 + k..));

                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                let q = dss3.reverse();
                q.write(data.slice_from_mut(q_modules * 4 - k - 3..));

                dc4 += a0;

                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write(data.slice_from_mut(q_modules * 4 + k..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                let q = dss5.reverse();
                q.write(data.slice_from_mut(q_modules * 6 - k - 3..));

                dc6 += a0;
                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
                dc6.write(data.slice_from_mut(q_modules * 6 + k..));

                k += 4;
                uk += 6;
            }

            let rem = q_modules - k;

            // handle remainder
            if rem == 3 {
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                let c_forward = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(k..) });
                let s_forward =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules - k - 2..) })
                        .reverse3();

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R7_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f64::R7_COS_EVEN2_M2;
                let mut dc6 = twiddled_dc * f64::R7_COS_EVEN2_M1;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R7_SIN_ODD0_M0;
                let mut ds3 = twiddled_ds * f64::R7_SIN_ODD1_M0;
                let mut ds5 = twiddled_ds * f64::R7_SIN_ODD2_M0;

                {
                    let c_forward =
                        AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                    let s_forward = AvxStoreD::load3(unsafe {
                        s_buffer.get_unchecked(q_modules * 2 - k - 2..)
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
                    dc2 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M2, twiddled_dc, dc6);

                    ds1 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD0_M1, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD1_M1, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD2_M1, twiddled_ds, ds5);
                }

                {
                    let c_forward =
                        AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                    let s_forward = AvxStoreD::load3(unsafe {
                        s_buffer.get_unchecked(q_modules * 3 - k - 2..)
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
                    dc2 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M0, twiddled_dc, dc6);

                    ds1 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD0_M2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD1_M2, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD2_M2, twiddled_ds, ds5);
                }

                let a0 = AvxStoreD::load3(unsafe { a_buffer.get_unchecked(k..) });
                let dc = dc0 + a0;
                dc.write3(data.slice_from_mut(k..));

                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                let q = dss1.reverse3();
                q.write3(data.slice_from_mut(q_modules * 2 - k - 2..));

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write3(data.slice_from_mut(q_modules * 2 + k..));

                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                let q = dss3.reverse3();
                q.write3(data.slice_from_mut(q_modules * 4 - k - 2..));

                dc4 += a0;

                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write3(data.slice_from_mut(q_modules * 4 + k..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                let q = dss5.reverse3();
                q.write3(data.slice_from_mut(q_modules * 6 - k - 2..));

                dc6 += a0;
                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
                dc6.write3(data.slice_from_mut(q_modules * 6 + k..));
            } else if rem == 2 {
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                let c_forward = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(k..) });
                let s_forward =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules - k - 1..) })
                        .reverse2();

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R7_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f64::R7_COS_EVEN2_M2;
                let mut dc6 = twiddled_dc * f64::R7_COS_EVEN2_M1;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R7_SIN_ODD0_M0;
                let mut ds3 = twiddled_ds * f64::R7_SIN_ODD1_M0;
                let mut ds5 = twiddled_ds * f64::R7_SIN_ODD2_M0;

                {
                    let c_forward =
                        AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                    let s_forward = AvxStoreD::load2(unsafe {
                        s_buffer.get_unchecked(q_modules * 2 - k - 1..)
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
                    dc2 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M2, twiddled_dc, dc6);

                    ds1 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD0_M1, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD1_M1, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD2_M1, twiddled_ds, ds5);
                }

                {
                    let c_forward =
                        AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                    let s_forward = AvxStoreD::load2(unsafe {
                        s_buffer.get_unchecked(q_modules * 3 - k - 1..)
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
                    dc2 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M0, twiddled_dc, dc6);

                    ds1 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD0_M2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD1_M2, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD2_M2, twiddled_ds, ds5);
                }

                let a0 = AvxStoreD::load2(unsafe { a_buffer.get_unchecked(k..) });
                let dc = dc0 + a0;
                dc.write2(data.slice_from_mut(k..));

                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                let q = dss1.reverse2();
                q.write2(data.slice_from_mut(q_modules * 2 - k - 1..));

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write2(data.slice_from_mut(q_modules * 2 + k..));

                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                let q = dss3.reverse2();
                q.write2(data.slice_from_mut(q_modules * 4 - k - 1..));

                dc4 += a0;

                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write2(data.slice_from_mut(q_modules * 4 + k..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                let q = dss5.reverse2();
                q.write2(data.slice_from_mut(q_modules * 6 - k - 1..));

                dc6 += a0;
                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
                dc6.write2(data.slice_from_mut(q_modules * 6 + k..));
            } else if rem == 1 {
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                let c_forward = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(k..) });
                let s_forward =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules - k..) });

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R7_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f64::R7_COS_EVEN2_M2;
                let mut dc6 = twiddled_dc * f64::R7_COS_EVEN2_M1;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R7_SIN_ODD0_M0;
                let mut ds3 = twiddled_ds * f64::R7_SIN_ODD1_M0;
                let mut ds5 = twiddled_ds * f64::R7_SIN_ODD2_M0;

                {
                    let c_forward =
                        AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                    let s_forward =
                        AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 2 - k..) });

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 2) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 3) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 2) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 3) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M2, twiddled_dc, dc6);

                    ds1 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD0_M1, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD1_M1, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD2_M1, twiddled_ds, ds5);
                }

                {
                    let c_forward =
                        AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                    let s_forward =
                        AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 3 - k..) });

                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk + 4) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 5) };

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk + 4) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 5) };

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(f64::R7_COS_EVEN2_M0, twiddled_dc, dc6);

                    ds1 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD0_M2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD1_M2, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R7_SIN_ODD2_M2, twiddled_ds, ds5);
                }

                let a0 = AvxStoreD::load1(unsafe { a_buffer.get_unchecked(k..) });
                let dc = dc0 + a0;
                dc.write1(data.slice_from_mut(k..));

                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                dss1.write1(data.slice_from_mut(q_modules * 2 - k..));

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write1(data.slice_from_mut(q_modules * 2 + k..));

                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                dss3.write1(data.slice_from_mut(q_modules * 4 - k..));

                dc4 += a0;

                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write1(data.slice_from_mut(q_modules * 4 + k..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                dss5.write1(data.slice_from_mut(q_modules * 6 - k..));

                dc6 += a0;
                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
                dc6.write1(data.slice_from_mut(q_modules * 6 + k..));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::dct2::Dct2MixedRadix7;
    use crate::tests::naive_dct2;

    #[test]
    fn test_radix7_dct() {
        let mut input = vec![0.; 7 * 11];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f64 + rand::random::<f64>() * 10.0;
        }
        let mut reference_input = input.clone();
        let mut reference_input2 = reference_input.clone();
        // let rr = Pxdct::make_dct2_f64(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2(&reference_input);
        let bf = Dct2MixedRadix7::new(input.len(), Pxdct::make_dct2_f64(input.len() / 7).unwrap())
            .unwrap();
        bf.execute(&mut reference_input2).unwrap();
        let bf =
            AvxDct2MixedRadix7d::new(input.len(), Pxdct::make_dct2_f64(input.len() / 7).unwrap())
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
