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
use crate::dct2::{MixedRadix3Sample, radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::util::{DctConstants, DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix3_rotation_twiddles_avx_f(q_modules: usize, len: usize) -> Vec<AvxStoreF> {
    let simd_groups = q_modules.div_ceil(8);
    let main_q = 3usize;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 4 * inner_groups);

    let working_modules = q_modules - 1;

    let mut uk = 0usize;

    let mut array_re = [0f32; 8];
    let mut array_im = [0f32; 8];

    while uk + 8 <= working_modules {
        let k = uk + 1;

        for i in 0..8 {
            let twiddle =
                radixq_rotation_twiddle(main_q, 0, (k + i).as_(), (q_modules - (k + i)).as_(), len);
            array_re[i] = twiddle.re;
            array_im[i] = twiddle.im;
        }

        twiddles.push(AvxStoreF::load(array_re.as_ref()));
        twiddles.push(AvxStoreF::load(array_im.as_ref()));

        uk += 8;
    }

    let remainder = working_modules - (working_modules / 8) * 8;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0f32; 8];
        let mut array_im = [0f32; 8];
        for i in 0..remainder {
            let layer =
                radixq_rotation_twiddle(main_q, 0, (k + i).as_(), (q_modules - (k + i)).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(AvxStoreF::load(array_re.as_ref()));
        twiddles.push(AvxStoreF::load(array_im.as_ref()));
    }

    twiddles
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix3_cos_twiddles_avx_f(q_modules: usize, len: usize) -> Vec<AvxStoreF> {
    let main_q = 3usize;
    let simd_groups = q_modules.div_ceil(8);
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 2 * inner_groups);

    let working_modules = q_modules - 1;

    let mut uk = 0usize;
    while uk + 8 <= working_modules {
        let k = uk + 1;

        let mut array_re = [0f32; 8];
        let mut array_im = [0.; 8];
        for i in 0..8 {
            array_re[i] = radixq_cos_twiddle(main_q, 0, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 0, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(AvxStoreF::load(array_re.as_ref()));
        twiddles.push(AvxStoreF::load(array_im.as_ref()));

        uk += 8;
    }

    let remainder = working_modules - (working_modules / 8) * 8;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0f32; 8];
        let mut array_im = [0.; 8];
        for i in 0..remainder {
            array_re[i] = radixq_cos_twiddle(main_q, 0, (k + i).as_(), len);
            array_im[i] = radixq_cos_twiddle(main_q, 0, (q_modules - (k + i)).as_(), len);
        }

        twiddles.push(AvxStoreF::load(array_re.as_ref()));
        twiddles.push(AvxStoreF::load(array_im.as_ref()));
    }

    twiddles
}

pub(crate) struct AvxDct2MixedRadix3f {
    rotation_layer: Vec<AvxStoreF>,
    cos_twiddles: Vec<AvxStoreF>,
    inner_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    execution_length: usize,
}

impl AvxDct2MixedRadix3f {
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<AvxDct2MixedRadix3f, PxdctError> {
        assert!(
            len.is_multiple_of(3),
            "Mixed radix 5 should not be called on sizes no divisible by 5"
        );

        let q_modules = len / 3;

        // always 3 inner groups in Radix-7

        // Precompute rotation twiddles for k≥1
        // Format: [m0_k1, m1_k1, m0_k2, m1_k2, ...]
        let rotation_layer = unsafe { dct2_radix3_rotation_twiddles_avx_f(q_modules, len) };

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let cos_twiddles = unsafe { dct2_radix3_cos_twiddles_avx_f(q_modules, len) };

        Ok(AvxDct2MixedRadix3f {
            rotation_layer,
            inner_dct,
            cos_twiddles,
            execution_length: len,
        })
    }
}

impl PxdctExecutor<f32> for AvxDct2MixedRadix3f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

impl AvxDct2MixedRadix3f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        assert!(!self.cos_twiddles.is_empty());

        let mut scratch = try_vec![f32::default(); self.execution_length];

        let q_modules = self.execution_length / 3;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules);

            // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
            for (n, dst) in a_buffer.iter_mut().enumerate() {
                unsafe {
                    *dst = *chunk.get_unchecked(n * 3 + 1);
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
                    let u0 = unsafe { *chunk.get_unchecked(3 * n + m) };
                    let u1 = unsafe { *chunk.get_unchecked(3 * n + 3 - m - 1) };

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }
            }

            // Step 2: Apply DCT-II to all buffers (A, C₀, S₀)
            self.inner_dct.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules);

            {
                // Step 3: Recombine transformed buffers with twiddle factors

                // Handle k=0 case (DC and low frequencies)
                let qc = c_buffer[0];
                let c0 = qc; // Component C₀ (position 0)
                let c1 = qc * -f32::HALF; // Component C₂ (position 2, uses j=2)

                let s0_twiddled = s_buffer[0];

                let s0 = s0_twiddled * f32::SIN2PI_OVER_3;

                // Write output: C₀ (pos 0), S₁
                let a0 = a_buffer[0];
                let dc = c0 + a0;
                unsafe {
                    *chunk.get_unchecked_mut(0) = dc;
                }

                unsafe {
                    *chunk.get_unchecked_mut(q_modules) = s0;
                }

                unsafe {
                    let idx1 = q_modules * 2;
                    let qid2 = -(c1 + a0); // negated 2j
                    *chunk.get_unchecked_mut(idx1) = qid2;
                }

                let mut k = 1usize;
                let mut uk = 0usize;

                // Step 4: Handle k≥1 cases with rotation twiddles
                while k + 8 <= q_modules {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load(unsafe { s_buffer.get_unchecked(q_modules - k - 7..) })
                            .reverse();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = -twiddled_dc * f32::HALF;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

                    let a0 = AvxStoreF::load(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;
                    unsafe {
                        dc.write(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        let q = dss1.reverse();
                        q.write(chunk.get_unchecked_mut(q_modules * 2 - k - 7..));
                    }

                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }

                    k += 8;
                    uk += 2;
                }

                let rem = q_modules - k;

                // handle remainder
                if rem == 7 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load7(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load7(unsafe { s_buffer.get_unchecked(q_modules - k - 6..) })
                            .reverse7();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = -twiddled_dc * f32::HALF;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

                    let a0 = AvxStoreF::load7(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;
                    unsafe {
                        dc.write7(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        let q = dss1.reverse7();
                        q.write7(chunk.get_unchecked_mut(q_modules * 2 - k - 6..));
                    }

                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write7(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }
                } else if rem == 6 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load6(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load6(unsafe { s_buffer.get_unchecked(q_modules - k - 5..) })
                            .reverse6();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = -twiddled_dc * f32::HALF;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

                    let a0 = AvxStoreF::load6(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;
                    unsafe {
                        dc.write6(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        let q = dss1.reverse6();
                        q.write6(chunk.get_unchecked_mut(q_modules * 2 - k - 5..));
                    }

                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write6(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }
                } else if rem == 5 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load5(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load5(unsafe { s_buffer.get_unchecked(q_modules - k - 4..) })
                            .reverse5();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = -twiddled_dc * f32::HALF;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

                    let a0 = AvxStoreF::load5(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;
                    unsafe {
                        dc.write5(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        let q = dss1.reverse5();
                        q.write5(chunk.get_unchecked_mut(q_modules * 2 - k - 4..));
                    }

                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write5(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }
                } else if rem == 4 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load4(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load4(unsafe { s_buffer.get_unchecked(q_modules - k - 3..) })
                            .reverse4();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = -twiddled_dc * f32::HALF;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

                    let a0 = AvxStoreF::load4(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;
                    unsafe {
                        dc.write4(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        let q = dss1.reverse4();
                        q.write4(chunk.get_unchecked_mut(q_modules * 2 - k - 3..));
                    }

                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write4(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }
                } else if rem == 3 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load3(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load3(unsafe { s_buffer.get_unchecked(q_modules - k - 2..) })
                            .reverse3();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * -f32::HALF;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

                    let a0 = AvxStoreF::load3(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;
                    unsafe {
                        dc.write3(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        let q = dss1.reverse3();
                        q.write3(chunk.get_unchecked_mut(q_modules * 2 - k - 2..));
                    }

                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write3(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }
                } else if rem == 2 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load2(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load2(unsafe { s_buffer.get_unchecked(q_modules - k - 1..) })
                            .reverse2();

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * -f32::HALF;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

                    let a0 = AvxStoreF::load2(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;
                    unsafe {
                        dc.write2(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        let q = dss1.reverse2();
                        q.write2(chunk.get_unchecked_mut(q_modules * 2 - k - 1..));
                    }

                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write2(chunk.get_unchecked_mut(q_modules * 2 + k..));
                    }
                } else if rem == 1 {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
                    let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

                    let c_forward = AvxStoreF::load1(unsafe { c_buffer.get_unchecked(k..) });
                    let s_forward =
                        AvxStoreF::load1(unsafe { s_buffer.get_unchecked(q_modules - k..) });

                    let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                    let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
                    let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

                    let twiddled_dc = rotated_dc * twiddle_re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * -f32::HALF;

                    let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle_im;

                    let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

                    let a0 = AvxStoreF::load1(unsafe { a_buffer.get_unchecked(k..) });
                    let dc = dc0 + a0;
                    unsafe {
                        dc.write1(chunk.get_unchecked_mut(k..));
                    }

                    let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                    unsafe {
                        dss1.write1(chunk.get_unchecked_mut(q_modules * 2 - k..));
                    }

                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                    unsafe {
                        dc2.write1(chunk.get_unchecked_mut(q_modules * 2 + k..));
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
    fn test_radix7_dct() {
        let mut input = vec![0.; 3 * 13];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32 + rand::random::<f32>() * 10.0;
        }
        let mut reference_input = input.clone();
        reference_input = naive_dct2_f32(&reference_input);
        let bf =
            AvxDct2MixedRadix3f::new(input.len(), Pxdct::make_dct2_f32(input.len() / 3).unwrap())
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
