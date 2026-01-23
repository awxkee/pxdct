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
use crate::avx::util::fma;
use crate::dct4::{Dct4MixedRadix19Sample, radixq_dct4_rotation_twiddle};
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct4_radix19d_rotation_twiddles_avx(q_modules: usize, len: usize) -> Vec<AvxStoreD> {
    let simd_groups = q_modules.div_ceil(4);
    let main_q = 19usize;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    let mut twiddles = Vec::with_capacity(simd_groups * 18 * inner_groups);

    let working_modules = q_modules;

    let mut uk = 0usize;
    while uk + 4 <= working_modules {
        let k = uk;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for m in 0..9 {
            for i in 0..4 {
                let layer = radixq_dct4_rotation_twiddle(main_q, m, (k + i).as_(), len);
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
        let k = uk;

        for m in 0..9 {
            let mut array_re = [0.; 4];
            let mut array_im = [0.; 4];

            for i in 0..remainder {
                let layer = radixq_dct4_rotation_twiddle(main_q, m, (k + i).as_(), len);
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }

            twiddles.push(AvxStoreD::load(array_re.as_ref()));
            twiddles.push(AvxStoreD::load(array_im.as_ref()));
        }
    }

    twiddles
}

pub(crate) struct AvxDct4MixedRadix19d {
    inner_dct4: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    rotation_twiddles: Vec<AvxStoreD>,
    execution_length: usize,
}

impl AvxDct4MixedRadix19d {
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        dct2: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct2.length(),
            len / 19,
            "DCT-IV Mixed-Radix-19 length DCTs must be one nineteenth of DCT-IV"
        );

        Ok(Self {
            inner_dct4: dct2,
            execution_length: len,
            rotation_twiddles: unsafe { dct4_radix19d_rotation_twiddles_avx(len / 19, len) },
        })
    }
}

impl PxdctExecutor<f64> for AvxDct4MixedRadix19d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

impl AvxDct4MixedRadix19d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }
        let mut scratch = try_vec![f64::default(); self.execution_length];

        let q_modules = self.execution_length / 19;

        let s = 2 * self.execution_length / 19;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 9);

            // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
            for (n, dst) in a_buffer.iter_mut().enumerate() {
                unsafe {
                    *dst = *chunk.get_unchecked(n * 19 + 9);
                }
            }

            // Extract and combine symmetric pairs with sign alternation for S buffer
            for (m, (c_buffer, s_buffer)) in c_buffer
                .chunks_exact_mut(q_modules)
                .zip(s_buffer.chunks_exact_mut(q_modules))
                .enumerate()
            {
                let mut sign = f64::one();
                for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate()
                {
                    let u0 = unsafe { *chunk.get_unchecked(19 * n + m) };
                    let u1 = unsafe { *chunk.get_unchecked(19 * n + 19 - m - 1) };

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }
            }

            // Step 2: Apply DCT-IV to all buffers (A, C₀, C₁, S₀, S₁)
            self.inner_dct4.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 9);

            let mut k = 0usize;
            let mut uk = 0usize;

            // Step 4: Handle k≥0 cases with rotation twiddles
            while k + 4 <= q_modules {
                const S: usize = 4;
                let c_v0 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse();
                let a_v0 = AvxStoreD::load(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse();

                let c_v2 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                let s_v2 =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 3 - S - k..) })
                        .reverse();

                let c_v3 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                let s_v3 =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 4 - S - k..) })
                        .reverse();

                let c_v4 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                let s_v4 =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 5 - S - k..) })
                        .reverse();

                let c_v5 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 5 + k..) });
                let s_v5 =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 6 - S - k..) })
                        .reverse();

                let c_v6 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 6 + k..) });
                let s_v6 =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 7 - S - k..) })
                        .reverse();

                let c_v7 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 7 + k..) });
                let s_v7 =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 8 - S - k..) })
                        .reverse();

                let c_v8 = AvxStoreD::load(unsafe { c_buffer.get_unchecked(q_modules * 8 + k..) });
                let s_v8 =
                    AvxStoreD::load(unsafe { s_buffer.get_unchecked(q_modules * 9 - S - k..) })
                        .reverse();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };
                let twiddle2_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 4) };
                let twiddle2_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 5) };
                let twiddle3_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 6) };
                let twiddle3_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 7) };
                let twiddle4_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 8) };
                let twiddle4_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 9) };
                let twiddle5_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 10) };
                let twiddle5_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 11) };
                let twiddle6_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 12) };
                let twiddle6_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 13) };
                let twiddle7_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 14) };
                let twiddle7_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 15) };
                let twiddle8_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 16) };
                let twiddle8_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 17) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f64::D4_R19_ROT_TWIDDLE_2;
                v0 *= f64::D4_R19_ROT_TWIDDLE_3;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_6, u1);
                v0 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_7, v0);

                let iq2 = fma(c_v2, twiddle2_re, s_v2 * twiddle2_im);
                let siq2 = fma(c_v2, twiddle2_im, -s_v2 * twiddle2_re);

                u1 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_1, u1);
                v0 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_0, v0);

                let iq3 = fma(c_v3, twiddle3_re, s_v3 * twiddle3_im);
                let siq3 = fma(c_v3, twiddle3_im, -s_v3 * twiddle3_re);

                u1 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_10, u1);
                v0 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_11, v0);

                let iq4 = fma(c_v4, twiddle4_re, s_v4 * twiddle4_im);
                let siq4 = fma(c_v4, twiddle4_im, -s_v4 * twiddle4_re);

                u1 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_12, u1);
                v0 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_13, v0);

                let iq5 = fma(c_v5, twiddle5_re, s_v5 * twiddle5_im);
                let siq5 = fma(c_v5, twiddle5_im, -s_v5 * twiddle5_re);

                u1 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_16, u1);
                v0 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_17, v0);

                let iq6 = fma(c_v6, twiddle6_re, s_v6 * twiddle6_im);
                let siq6 = fma(c_v6, twiddle6_im, -s_v6 * twiddle6_re);

                u1 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_9, u1);
                v0 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_8, v0);

                let iq7 = fma(c_v7, twiddle7_re, s_v7 * twiddle7_im);
                let siq7 = fma(c_v7, twiddle7_im, -s_v7 * twiddle7_re);

                u1 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_4, u1);
                v0 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_5, v0);

                let iq8 = fma(c_v8, twiddle8_re, s_v8 * twiddle8_im);
                let siq8 = fma(c_v8, twiddle8_im, -s_v8 * twiddle8_re);

                u1 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_14, u1);
                v0 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_15, v0);

                u0 += iq1 + iq2 + iq3 + iq4 + iq5 + iq6 + iq7 + iq8 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                unsafe {
                    u0.write(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse().write(chunk.get_unchecked_mut(s - S - k..));
                }

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f64::D4_R19_ROT_TWIDDLE_14;
                v2 *= f64::D4_R19_ROT_TWIDDLE_15;
                u2 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_9, u2);
                v2 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_8, v2);
                u2 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_12, u2);
                v2 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_13, v2);
                u2 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_1, u2);
                v2 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_0, v2);
                u2 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_2, u2);
                v2 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_3, v2);
                u2 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_6, u2);
                v2 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_7, v2);
                u2 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_10, u2);
                v2 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_11, v2);
                u2 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_16, u2);
                v2 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_17, v2);
                u2 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_4, u2);
                v2 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_5, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    uc2.reverse()
                        .write(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write(chunk.get_unchecked_mut(2 * s + k..));
                }

                let mut u3 = iq0;
                let mut v3 = siq0;
                u3 *= f64::D4_R19_ROT_TWIDDLE_6;
                v3 *= f64::D4_R19_ROT_TWIDDLE_7;
                u3 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_12, u3);
                v3 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_13, v3);
                u3 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_4, u3);
                v3 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_5, v3);
                u3 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_14, u3);
                v3 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_15, v3);
                u3 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_16, u3);
                v3 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_17, v3);
                u3 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_1, u3);
                v3 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_0, v3);
                u3 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_2, u3);
                v3 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_3, v3);
                u3 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_10, u3);
                v3 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_11, v3);
                u3 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_9, u3);
                v3 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_8, v3);
                u3 = u3 - a_v0;
                let uc4 = u3 - v3;
                let uc5 = u3 + v3;

                unsafe {
                    uc4.reverse()
                        .write(chunk.get_unchecked_mut(3 * s - S - k..));
                }

                unsafe {
                    uc5.write(chunk.get_unchecked_mut(3 * s + k..));
                }

                let mut u4 = iq0;
                let mut v4 = siq0;
                u4 *= f64::D4_R19_ROT_TWIDDLE_4;
                v4 *= f64::D4_R19_ROT_TWIDDLE_5;
                u4 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_10, u4);
                v4 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_11, v4);
                u4 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_2, u4);
                v4 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_3, v4);
                u4 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_12, u4);
                v4 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_13, v4);
                u4 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_14, u4);
                v4 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_15, v4);
                u4 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_9, u4);
                v4 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_8, v4);
                u4 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_1, u4);
                v4 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_0, v4);
                u4 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_6, u4);
                v4 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_7, v4);
                u4 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_16, u4);
                v4 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_17, v4);
                u4 += a_v0;
                let uc6 = u4 - v4;
                let uc7 = u4 + v4;

                unsafe {
                    uc6.reverse()
                        .write(chunk.get_unchecked_mut(4 * s - S - k..));
                }

                unsafe {
                    uc7.write(chunk.get_unchecked_mut(4 * s + k..));
                }

                let mut u5 = iq0;
                let mut v5 = siq0;
                u5 *= f64::D4_R19_ROT_TWIDDLE_1;
                v5 *= f64::D4_R19_ROT_TWIDDLE_0;
                u5 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_4, u5);
                v5 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_5, v5);
                u5 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_9, u5);
                v5 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_8, v5);
                u5 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_6, u5);
                v5 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_7, v5);
                u5 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_10, u5);
                v5 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_11, v5);
                u5 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_14, u5);
                v5 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_15, v5);
                u5 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_16, u5);
                v5 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_17, v5);
                u5 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_2, u5);
                v5 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_3, v5);
                u5 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_12, u5);
                v5 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_13, v5);
                u5 = u5 - a_v0;
                let uc8 = u5 - v5;
                let uc9 = u5 + v5;

                unsafe {
                    uc8.reverse()
                        .write(chunk.get_unchecked_mut(5 * s - S - k..));
                }

                unsafe {
                    uc9.write(chunk.get_unchecked_mut(5 * s + k..));
                }

                let mut u6 = iq0;
                let mut v6 = siq0;
                u6 *= f64::D4_R19_ROT_TWIDDLE_9;
                v6 *= f64::D4_R19_ROT_TWIDDLE_8;
                u6 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_2, u6);
                v6 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_3, v6);
                u6 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_16, u6);
                v6 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_17, v6);
                u6 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_4, u6);
                v6 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_5, v6);
                u6 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_6, u6);
                v6 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_7, v6);
                u6 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_12, u6);
                v6 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_13, v6);
                u6 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_14, u6);
                v6 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_15, v6);
                u6 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_1, u6);
                v6 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_0, v6);
                u6 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_10, u6);
                v6 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_11, v6);
                u6 += a_v0;
                let uc10 = u6 - v6;
                let uc11 = u6 + v6;

                unsafe {
                    uc10.reverse()
                        .write(chunk.get_unchecked_mut(6 * s - S - k..));
                }

                unsafe {
                    uc11.write(chunk.get_unchecked_mut(6 * s + k..));
                }

                let mut u7 = iq0;
                let mut v7 = siq0;
                u7 *= f64::D4_R19_ROT_TWIDDLE_10;
                v7 *= f64::D4_R19_ROT_TWIDDLE_11;
                u7 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_14, u7);
                v7 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_15, v7);
                u7 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_6, u7);
                v7 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_7, v7);
                u7 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_16, u7);
                v7 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_17, v7);
                u7 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_9, u7);
                v7 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_8, v7);
                u7 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_2, u7);
                v7 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_3, v7);
                u7 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_4, u7);
                v7 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_5, v7);
                u7 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_12, u7);
                v7 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_13, v7);
                u7 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_1, u7);
                v7 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_0, v7);
                u7 = u7 - a_v0;

                let uc12 = u7 - v7;
                let uc13 = u7 + v7;

                unsafe {
                    uc12.reverse()
                        .write(chunk.get_unchecked_mut(7 * s - S - k..));
                }

                unsafe {
                    uc13.write(chunk.get_unchecked_mut(7 * s + k..));
                }

                let mut u8 = iq0;
                let mut v8 = siq0;
                u8 *= -f64::D4_R19_ROT_TWIDDLE_16;
                v8 *= f64::D4_R19_ROT_TWIDDLE_17;
                u8 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_1, u8);
                v8 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_0, v8);
                u8 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_14, u8);
                v8 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_15, v8);
                u8 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_2, u8);
                v8 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_3, v8);
                u8 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_4, u8);
                v8 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_5, v8);
                u8 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_10, u8);
                v8 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_11, v8);
                u8 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_12, u8);
                v8 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_13, v8);
                u8 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_9, u8);
                v8 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_8, v8);
                u8 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_6, u8);
                v8 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_7, v8);
                u8 += a_v0;
                let uc14 = u8 - v8;
                let uc15 = u8 + v8;

                unsafe {
                    uc14.reverse()
                        .write(chunk.get_unchecked_mut(8 * s - S - k..));
                }

                unsafe {
                    uc15.write(chunk.get_unchecked_mut(8 * s + k..));
                }

                let mut u9 = iq0;
                let mut v9 = siq0;
                u9 *= f64::D4_R19_ROT_TWIDDLE_12;
                v9 *= f64::D4_R19_ROT_TWIDDLE_13;
                u9 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_16, u9);
                v9 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_17, v9);
                u9 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_10, u9);
                v9 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_11, v9);
                u9 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_9, u9);
                v9 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_8, v9);
                u9 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_1, u9);
                v9 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_0, v9);
                u9 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_4, u9);
                v9 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_5, v9);
                u9 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_6, u9);
                v9 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_7, v9);
                u9 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_14, u9);
                v9 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_15, v9);
                u9 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_2, u9);
                v9 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_3, v9);
                u9 = u9 - a_v0;
                let uc16 = u9 - v9;
                let uc17 = u9 + v9;

                unsafe {
                    uc16.reverse()
                        .write(chunk.get_unchecked_mut(9 * s - S - k..));
                }

                unsafe {
                    uc17.write(chunk.get_unchecked_mut(9 * s + k..));
                }

                k += 4;
                uk += 18;
            }

            let rem = q_modules - k;
            if rem == 3 {
                const S: usize = 3;
                let c_v0 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse3();
                let a_v0 = AvxStoreD::load3(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse3();

                let c_v2 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                let s_v2 =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules * 3 - S - k..) })
                        .reverse3();

                let c_v3 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                let s_v3 =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules * 4 - S - k..) })
                        .reverse3();

                let c_v4 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                let s_v4 =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules * 5 - S - k..) })
                        .reverse3();

                let c_v5 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 5 + k..) });
                let s_v5 =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules * 6 - S - k..) })
                        .reverse3();

                let c_v6 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 6 + k..) });
                let s_v6 =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules * 7 - S - k..) })
                        .reverse3();

                let c_v7 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 7 + k..) });
                let s_v7 =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules * 8 - S - k..) })
                        .reverse3();

                let c_v8 = AvxStoreD::load3(unsafe { c_buffer.get_unchecked(q_modules * 8 + k..) });
                let s_v8 =
                    AvxStoreD::load3(unsafe { s_buffer.get_unchecked(q_modules * 9 - S - k..) })
                        .reverse3();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };
                let twiddle2_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 4) };
                let twiddle2_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 5) };
                let twiddle3_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 6) };
                let twiddle3_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 7) };
                let twiddle4_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 8) };
                let twiddle4_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 9) };
                let twiddle5_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 10) };
                let twiddle5_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 11) };
                let twiddle6_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 12) };
                let twiddle6_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 13) };
                let twiddle7_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 14) };
                let twiddle7_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 15) };
                let twiddle8_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 16) };
                let twiddle8_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 17) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f64::D4_R19_ROT_TWIDDLE_2;
                v0 *= f64::D4_R19_ROT_TWIDDLE_3;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_6, u1);
                v0 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_7, v0);

                let iq2 = fma(c_v2, twiddle2_re, s_v2 * twiddle2_im);
                let siq2 = fma(c_v2, twiddle2_im, -s_v2 * twiddle2_re);

                u1 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_1, u1);
                v0 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_0, v0);

                let iq3 = fma(c_v3, twiddle3_re, s_v3 * twiddle3_im);
                let siq3 = fma(c_v3, twiddle3_im, -s_v3 * twiddle3_re);

                u1 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_10, u1);
                v0 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_11, v0);

                let iq4 = fma(c_v4, twiddle4_re, s_v4 * twiddle4_im);
                let siq4 = fma(c_v4, twiddle4_im, -s_v4 * twiddle4_re);

                u1 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_12, u1);
                v0 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_13, v0);

                let iq5 = fma(c_v5, twiddle5_re, s_v5 * twiddle5_im);
                let siq5 = fma(c_v5, twiddle5_im, -s_v5 * twiddle5_re);

                u1 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_16, u1);
                v0 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_17, v0);

                let iq6 = fma(c_v6, twiddle6_re, s_v6 * twiddle6_im);
                let siq6 = fma(c_v6, twiddle6_im, -s_v6 * twiddle6_re);

                u1 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_9, u1);
                v0 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_8, v0);

                let iq7 = fma(c_v7, twiddle7_re, s_v7 * twiddle7_im);
                let siq7 = fma(c_v7, twiddle7_im, -s_v7 * twiddle7_re);

                u1 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_4, u1);
                v0 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_5, v0);

                let iq8 = fma(c_v8, twiddle8_re, s_v8 * twiddle8_im);
                let siq8 = fma(c_v8, twiddle8_im, -s_v8 * twiddle8_re);

                u1 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_14, u1);
                v0 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_15, v0);

                u0 += iq1 + iq2 + iq3 + iq4 + iq5 + iq6 + iq7 + iq8 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                unsafe {
                    u0.write3(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write3(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse3().write3(chunk.get_unchecked_mut(s - S - k..));
                }

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f64::D4_R19_ROT_TWIDDLE_14;
                v2 *= f64::D4_R19_ROT_TWIDDLE_15;
                u2 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_9, u2);
                v2 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_8, v2);
                u2 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_12, u2);
                v2 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_13, v2);
                u2 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_1, u2);
                v2 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_0, v2);
                u2 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_2, u2);
                v2 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_3, v2);
                u2 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_6, u2);
                v2 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_7, v2);
                u2 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_10, u2);
                v2 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_11, v2);
                u2 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_16, u2);
                v2 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_17, v2);
                u2 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_4, u2);
                v2 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_5, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    uc2.reverse3()
                        .write3(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write3(chunk.get_unchecked_mut(2 * s + k..));
                }

                let mut u3 = iq0;
                let mut v3 = siq0;
                u3 *= f64::D4_R19_ROT_TWIDDLE_6;
                v3 *= f64::D4_R19_ROT_TWIDDLE_7;
                u3 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_12, u3);
                v3 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_13, v3);
                u3 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_4, u3);
                v3 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_5, v3);
                u3 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_14, u3);
                v3 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_15, v3);
                u3 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_16, u3);
                v3 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_17, v3);
                u3 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_1, u3);
                v3 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_0, v3);
                u3 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_2, u3);
                v3 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_3, v3);
                u3 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_10, u3);
                v3 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_11, v3);
                u3 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_9, u3);
                v3 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_8, v3);
                u3 = u3 - a_v0;
                let uc4 = u3 - v3;
                let uc5 = u3 + v3;

                unsafe {
                    uc4.reverse3()
                        .write3(chunk.get_unchecked_mut(3 * s - S - k..));
                }

                unsafe {
                    uc5.write3(chunk.get_unchecked_mut(3 * s + k..));
                }

                let mut u4 = iq0;
                let mut v4 = siq0;
                u4 *= f64::D4_R19_ROT_TWIDDLE_4;
                v4 *= f64::D4_R19_ROT_TWIDDLE_5;
                u4 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_10, u4);
                v4 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_11, v4);
                u4 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_2, u4);
                v4 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_3, v4);
                u4 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_12, u4);
                v4 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_13, v4);
                u4 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_14, u4);
                v4 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_15, v4);
                u4 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_9, u4);
                v4 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_8, v4);
                u4 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_1, u4);
                v4 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_0, v4);
                u4 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_6, u4);
                v4 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_7, v4);
                u4 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_16, u4);
                v4 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_17, v4);
                u4 += a_v0;
                let uc6 = u4 - v4;
                let uc7 = u4 + v4;

                unsafe {
                    uc6.reverse3()
                        .write3(chunk.get_unchecked_mut(4 * s - S - k..));
                }

                unsafe {
                    uc7.write3(chunk.get_unchecked_mut(4 * s + k..));
                }

                let mut u5 = iq0;
                let mut v5 = siq0;
                u5 *= f64::D4_R19_ROT_TWIDDLE_1;
                v5 *= f64::D4_R19_ROT_TWIDDLE_0;
                u5 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_4, u5);
                v5 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_5, v5);
                u5 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_9, u5);
                v5 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_8, v5);
                u5 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_6, u5);
                v5 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_7, v5);
                u5 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_10, u5);
                v5 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_11, v5);
                u5 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_14, u5);
                v5 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_15, v5);
                u5 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_16, u5);
                v5 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_17, v5);
                u5 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_2, u5);
                v5 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_3, v5);
                u5 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_12, u5);
                v5 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_13, v5);
                u5 = u5 - a_v0;
                let uc8 = u5 - v5;
                let uc9 = u5 + v5;

                unsafe {
                    uc8.reverse3()
                        .write3(chunk.get_unchecked_mut(5 * s - S - k..));
                }

                unsafe {
                    uc9.write3(chunk.get_unchecked_mut(5 * s + k..));
                }

                let mut u6 = iq0;
                let mut v6 = siq0;
                u6 *= f64::D4_R19_ROT_TWIDDLE_9;
                v6 *= f64::D4_R19_ROT_TWIDDLE_8;
                u6 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_2, u6);
                v6 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_3, v6);
                u6 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_16, u6);
                v6 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_17, v6);
                u6 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_4, u6);
                v6 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_5, v6);
                u6 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_6, u6);
                v6 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_7, v6);
                u6 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_12, u6);
                v6 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_13, v6);
                u6 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_14, u6);
                v6 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_15, v6);
                u6 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_1, u6);
                v6 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_0, v6);
                u6 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_10, u6);
                v6 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_11, v6);
                u6 += a_v0;
                let uc10 = u6 - v6;
                let uc11 = u6 + v6;

                unsafe {
                    uc10.reverse3()
                        .write3(chunk.get_unchecked_mut(6 * s - S - k..));
                }

                unsafe {
                    uc11.write3(chunk.get_unchecked_mut(6 * s + k..));
                }

                let mut u7 = iq0;
                let mut v7 = siq0;
                u7 *= f64::D4_R19_ROT_TWIDDLE_10;
                v7 *= f64::D4_R19_ROT_TWIDDLE_11;
                u7 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_14, u7);
                v7 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_15, v7);
                u7 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_6, u7);
                v7 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_7, v7);
                u7 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_16, u7);
                v7 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_17, v7);
                u7 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_9, u7);
                v7 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_8, v7);
                u7 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_2, u7);
                v7 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_3, v7);
                u7 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_4, u7);
                v7 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_5, v7);
                u7 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_12, u7);
                v7 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_13, v7);
                u7 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_1, u7);
                v7 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_0, v7);
                u7 = u7 - a_v0;

                let uc12 = u7 - v7;
                let uc13 = u7 + v7;

                unsafe {
                    uc12.reverse3()
                        .write3(chunk.get_unchecked_mut(7 * s - S - k..));
                }

                unsafe {
                    uc13.write3(chunk.get_unchecked_mut(7 * s + k..));
                }

                let mut u8 = iq0;
                let mut v8 = siq0;
                u8 *= -f64::D4_R19_ROT_TWIDDLE_16;
                v8 *= f64::D4_R19_ROT_TWIDDLE_17;
                u8 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_1, u8);
                v8 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_0, v8);
                u8 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_14, u8);
                v8 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_15, v8);
                u8 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_2, u8);
                v8 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_3, v8);
                u8 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_4, u8);
                v8 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_5, v8);
                u8 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_10, u8);
                v8 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_11, v8);
                u8 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_12, u8);
                v8 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_13, v8);
                u8 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_9, u8);
                v8 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_8, v8);
                u8 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_6, u8);
                v8 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_7, v8);
                u8 += a_v0;
                let uc14 = u8 - v8;
                let uc15 = u8 + v8;

                unsafe {
                    uc14.reverse3()
                        .write3(chunk.get_unchecked_mut(8 * s - S - k..));
                }

                unsafe {
                    uc15.write3(chunk.get_unchecked_mut(8 * s + k..));
                }

                let mut u9 = iq0;
                let mut v9 = siq0;
                u9 *= f64::D4_R19_ROT_TWIDDLE_12;
                v9 *= f64::D4_R19_ROT_TWIDDLE_13;
                u9 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_16, u9);
                v9 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_17, v9);
                u9 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_10, u9);
                v9 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_11, v9);
                u9 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_9, u9);
                v9 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_8, v9);
                u9 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_1, u9);
                v9 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_0, v9);
                u9 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_4, u9);
                v9 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_5, v9);
                u9 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_6, u9);
                v9 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_7, v9);
                u9 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_14, u9);
                v9 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_15, v9);
                u9 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_2, u9);
                v9 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_3, v9);
                u9 = u9 - a_v0;
                let uc16 = u9 - v9;
                let uc17 = u9 + v9;

                unsafe {
                    uc16.reverse3()
                        .write3(chunk.get_unchecked_mut(9 * s - S - k..));
                }

                unsafe {
                    uc17.write3(chunk.get_unchecked_mut(9 * s + k..));
                }
            } else if rem == 2 {
                const S: usize = 2;
                let c_v0 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse2();
                let a_v0 = AvxStoreD::load2(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse2();

                let c_v2 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                let s_v2 =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules * 3 - S - k..) })
                        .reverse2();

                let c_v3 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                let s_v3 =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules * 4 - S - k..) })
                        .reverse2();

                let c_v4 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                let s_v4 =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules * 5 - S - k..) })
                        .reverse2();

                let c_v5 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 5 + k..) });
                let s_v5 =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules * 6 - S - k..) })
                        .reverse2();

                let c_v6 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 6 + k..) });
                let s_v6 =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules * 7 - S - k..) })
                        .reverse2();

                let c_v7 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 7 + k..) });
                let s_v7 =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules * 8 - S - k..) })
                        .reverse2();

                let c_v8 = AvxStoreD::load2(unsafe { c_buffer.get_unchecked(q_modules * 8 + k..) });
                let s_v8 =
                    AvxStoreD::load2(unsafe { s_buffer.get_unchecked(q_modules * 9 - S - k..) })
                        .reverse2();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };
                let twiddle2_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 4) };
                let twiddle2_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 5) };
                let twiddle3_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 6) };
                let twiddle3_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 7) };
                let twiddle4_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 8) };
                let twiddle4_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 9) };
                let twiddle5_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 10) };
                let twiddle5_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 11) };
                let twiddle6_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 12) };
                let twiddle6_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 13) };
                let twiddle7_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 14) };
                let twiddle7_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 15) };
                let twiddle8_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 16) };
                let twiddle8_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 17) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f64::D4_R19_ROT_TWIDDLE_2;
                v0 *= f64::D4_R19_ROT_TWIDDLE_3;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_6, u1);
                v0 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_7, v0);

                let iq2 = fma(c_v2, twiddle2_re, s_v2 * twiddle2_im);
                let siq2 = fma(c_v2, twiddle2_im, -s_v2 * twiddle2_re);

                u1 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_1, u1);
                v0 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_0, v0);

                let iq3 = fma(c_v3, twiddle3_re, s_v3 * twiddle3_im);
                let siq3 = fma(c_v3, twiddle3_im, -s_v3 * twiddle3_re);

                u1 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_10, u1);
                v0 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_11, v0);

                let iq4 = fma(c_v4, twiddle4_re, s_v4 * twiddle4_im);
                let siq4 = fma(c_v4, twiddle4_im, -s_v4 * twiddle4_re);

                u1 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_12, u1);
                v0 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_13, v0);

                let iq5 = fma(c_v5, twiddle5_re, s_v5 * twiddle5_im);
                let siq5 = fma(c_v5, twiddle5_im, -s_v5 * twiddle5_re);

                u1 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_16, u1);
                v0 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_17, v0);

                let iq6 = fma(c_v6, twiddle6_re, s_v6 * twiddle6_im);
                let siq6 = fma(c_v6, twiddle6_im, -s_v6 * twiddle6_re);

                u1 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_9, u1);
                v0 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_8, v0);

                let iq7 = fma(c_v7, twiddle7_re, s_v7 * twiddle7_im);
                let siq7 = fma(c_v7, twiddle7_im, -s_v7 * twiddle7_re);

                u1 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_4, u1);
                v0 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_5, v0);

                let iq8 = fma(c_v8, twiddle8_re, s_v8 * twiddle8_im);
                let siq8 = fma(c_v8, twiddle8_im, -s_v8 * twiddle8_re);

                u1 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_14, u1);
                v0 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_15, v0);

                u0 += iq1 + iq2 + iq3 + iq4 + iq5 + iq6 + iq7 + iq8 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                unsafe {
                    u0.write2(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write2(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse2().write2(chunk.get_unchecked_mut(s - S - k..));
                }

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f64::D4_R19_ROT_TWIDDLE_14;
                v2 *= f64::D4_R19_ROT_TWIDDLE_15;
                u2 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_9, u2);
                v2 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_8, v2);
                u2 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_12, u2);
                v2 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_13, v2);
                u2 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_1, u2);
                v2 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_0, v2);
                u2 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_2, u2);
                v2 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_3, v2);
                u2 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_6, u2);
                v2 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_7, v2);
                u2 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_10, u2);
                v2 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_11, v2);
                u2 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_16, u2);
                v2 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_17, v2);
                u2 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_4, u2);
                v2 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_5, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    uc2.reverse2()
                        .write2(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write2(chunk.get_unchecked_mut(2 * s + k..));
                }

                let mut u3 = iq0;
                let mut v3 = siq0;
                u3 *= f64::D4_R19_ROT_TWIDDLE_6;
                v3 *= f64::D4_R19_ROT_TWIDDLE_7;
                u3 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_12, u3);
                v3 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_13, v3);
                u3 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_4, u3);
                v3 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_5, v3);
                u3 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_14, u3);
                v3 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_15, v3);
                u3 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_16, u3);
                v3 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_17, v3);
                u3 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_1, u3);
                v3 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_0, v3);
                u3 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_2, u3);
                v3 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_3, v3);
                u3 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_10, u3);
                v3 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_11, v3);
                u3 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_9, u3);
                v3 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_8, v3);
                u3 = u3 - a_v0;
                let uc4 = u3 - v3;
                let uc5 = u3 + v3;

                unsafe {
                    uc4.reverse2()
                        .write2(chunk.get_unchecked_mut(3 * s - S - k..));
                }

                unsafe {
                    uc5.write2(chunk.get_unchecked_mut(3 * s + k..));
                }

                let mut u4 = iq0;
                let mut v4 = siq0;
                u4 *= f64::D4_R19_ROT_TWIDDLE_4;
                v4 *= f64::D4_R19_ROT_TWIDDLE_5;
                u4 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_10, u4);
                v4 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_11, v4);
                u4 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_2, u4);
                v4 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_3, v4);
                u4 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_12, u4);
                v4 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_13, v4);
                u4 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_14, u4);
                v4 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_15, v4);
                u4 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_9, u4);
                v4 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_8, v4);
                u4 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_1, u4);
                v4 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_0, v4);
                u4 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_6, u4);
                v4 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_7, v4);
                u4 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_16, u4);
                v4 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_17, v4);
                u4 += a_v0;
                let uc6 = u4 - v4;
                let uc7 = u4 + v4;

                unsafe {
                    uc6.reverse2()
                        .write2(chunk.get_unchecked_mut(4 * s - S - k..));
                }

                unsafe {
                    uc7.write2(chunk.get_unchecked_mut(4 * s + k..));
                }

                let mut u5 = iq0;
                let mut v5 = siq0;
                u5 *= f64::D4_R19_ROT_TWIDDLE_1;
                v5 *= f64::D4_R19_ROT_TWIDDLE_0;
                u5 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_4, u5);
                v5 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_5, v5);
                u5 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_9, u5);
                v5 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_8, v5);
                u5 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_6, u5);
                v5 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_7, v5);
                u5 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_10, u5);
                v5 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_11, v5);
                u5 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_14, u5);
                v5 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_15, v5);
                u5 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_16, u5);
                v5 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_17, v5);
                u5 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_2, u5);
                v5 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_3, v5);
                u5 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_12, u5);
                v5 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_13, v5);
                u5 = u5 - a_v0;
                let uc8 = u5 - v5;
                let uc9 = u5 + v5;

                unsafe {
                    uc8.reverse2()
                        .write2(chunk.get_unchecked_mut(5 * s - S - k..));
                }

                unsafe {
                    uc9.write2(chunk.get_unchecked_mut(5 * s + k..));
                }

                let mut u6 = iq0;
                let mut v6 = siq0;
                u6 *= f64::D4_R19_ROT_TWIDDLE_9;
                v6 *= f64::D4_R19_ROT_TWIDDLE_8;
                u6 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_2, u6);
                v6 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_3, v6);
                u6 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_16, u6);
                v6 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_17, v6);
                u6 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_4, u6);
                v6 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_5, v6);
                u6 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_6, u6);
                v6 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_7, v6);
                u6 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_12, u6);
                v6 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_13, v6);
                u6 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_14, u6);
                v6 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_15, v6);
                u6 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_1, u6);
                v6 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_0, v6);
                u6 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_10, u6);
                v6 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_11, v6);
                u6 += a_v0;
                let uc10 = u6 - v6;
                let uc11 = u6 + v6;

                unsafe {
                    uc10.reverse2()
                        .write2(chunk.get_unchecked_mut(6 * s - S - k..));
                }

                unsafe {
                    uc11.write2(chunk.get_unchecked_mut(6 * s + k..));
                }

                let mut u7 = iq0;
                let mut v7 = siq0;
                u7 *= f64::D4_R19_ROT_TWIDDLE_10;
                v7 *= f64::D4_R19_ROT_TWIDDLE_11;
                u7 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_14, u7);
                v7 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_15, v7);
                u7 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_6, u7);
                v7 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_7, v7);
                u7 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_16, u7);
                v7 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_17, v7);
                u7 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_9, u7);
                v7 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_8, v7);
                u7 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_2, u7);
                v7 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_3, v7);
                u7 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_4, u7);
                v7 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_5, v7);
                u7 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_12, u7);
                v7 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_13, v7);
                u7 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_1, u7);
                v7 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_0, v7);
                u7 = u7 - a_v0;

                let uc12 = u7 - v7;
                let uc13 = u7 + v7;

                unsafe {
                    uc12.reverse2()
                        .write2(chunk.get_unchecked_mut(7 * s - S - k..));
                }

                unsafe {
                    uc13.write2(chunk.get_unchecked_mut(7 * s + k..));
                }

                let mut u8 = iq0;
                let mut v8 = siq0;
                u8 *= -f64::D4_R19_ROT_TWIDDLE_16;
                v8 *= f64::D4_R19_ROT_TWIDDLE_17;
                u8 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_1, u8);
                v8 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_0, v8);
                u8 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_14, u8);
                v8 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_15, v8);
                u8 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_2, u8);
                v8 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_3, v8);
                u8 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_4, u8);
                v8 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_5, v8);
                u8 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_10, u8);
                v8 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_11, v8);
                u8 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_12, u8);
                v8 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_13, v8);
                u8 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_9, u8);
                v8 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_8, v8);
                u8 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_6, u8);
                v8 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_7, v8);
                u8 += a_v0;
                let uc14 = u8 - v8;
                let uc15 = u8 + v8;

                unsafe {
                    uc14.reverse2()
                        .write2(chunk.get_unchecked_mut(8 * s - S - k..));
                }

                unsafe {
                    uc15.write2(chunk.get_unchecked_mut(8 * s + k..));
                }

                let mut u9 = iq0;
                let mut v9 = siq0;
                u9 *= f64::D4_R19_ROT_TWIDDLE_12;
                v9 *= f64::D4_R19_ROT_TWIDDLE_13;
                u9 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_16, u9);
                v9 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_17, v9);
                u9 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_10, u9);
                v9 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_11, v9);
                u9 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_9, u9);
                v9 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_8, v9);
                u9 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_1, u9);
                v9 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_0, v9);
                u9 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_4, u9);
                v9 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_5, v9);
                u9 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_6, u9);
                v9 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_7, v9);
                u9 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_14, u9);
                v9 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_15, v9);
                u9 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_2, u9);
                v9 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_3, v9);
                u9 = u9 - a_v0;
                let uc16 = u9 - v9;
                let uc17 = u9 + v9;

                unsafe {
                    uc16.reverse2()
                        .write2(chunk.get_unchecked_mut(9 * s - S - k..));
                }

                unsafe {
                    uc17.write2(chunk.get_unchecked_mut(9 * s + k..));
                }
            } else if rem == 1 {
                const S: usize = 1;
                let c_v0 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules - S - k..) });
                let a_v0 = AvxStoreD::load1(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) });

                let c_v2 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 2 + k..) });
                let s_v2 =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 3 - S - k..) });

                let c_v3 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 3 + k..) });
                let s_v3 =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 4 - S - k..) });

                let c_v4 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 4 + k..) });
                let s_v4 =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 5 - S - k..) });

                let c_v5 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 5 + k..) });
                let s_v5 =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 6 - S - k..) });

                let c_v6 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 6 + k..) });
                let s_v6 =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 7 - S - k..) });

                let c_v7 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 7 + k..) });
                let s_v7 =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 8 - S - k..) });

                let c_v8 = AvxStoreD::load1(unsafe { c_buffer.get_unchecked(q_modules * 8 + k..) });
                let s_v8 =
                    AvxStoreD::load1(unsafe { s_buffer.get_unchecked(q_modules * 9 - S - k..) });

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };
                let twiddle2_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 4) };
                let twiddle2_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 5) };
                let twiddle3_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 6) };
                let twiddle3_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 7) };
                let twiddle4_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 8) };
                let twiddle4_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 9) };
                let twiddle5_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 10) };
                let twiddle5_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 11) };
                let twiddle6_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 12) };
                let twiddle6_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 13) };
                let twiddle7_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 14) };
                let twiddle7_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 15) };
                let twiddle8_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 16) };
                let twiddle8_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 17) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f64::D4_R19_ROT_TWIDDLE_2;
                v0 *= f64::D4_R19_ROT_TWIDDLE_3;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_6, u1);
                v0 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_7, v0);

                let iq2 = fma(c_v2, twiddle2_re, s_v2 * twiddle2_im);
                let siq2 = fma(c_v2, twiddle2_im, -s_v2 * twiddle2_re);

                u1 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_1, u1);
                v0 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_0, v0);

                let iq3 = fma(c_v3, twiddle3_re, s_v3 * twiddle3_im);
                let siq3 = fma(c_v3, twiddle3_im, -s_v3 * twiddle3_re);

                u1 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_10, u1);
                v0 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_11, v0);

                let iq4 = fma(c_v4, twiddle4_re, s_v4 * twiddle4_im);
                let siq4 = fma(c_v4, twiddle4_im, -s_v4 * twiddle4_re);

                u1 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_12, u1);
                v0 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_13, v0);

                let iq5 = fma(c_v5, twiddle5_re, s_v5 * twiddle5_im);
                let siq5 = fma(c_v5, twiddle5_im, -s_v5 * twiddle5_re);

                u1 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_16, u1);
                v0 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_17, v0);

                let iq6 = fma(c_v6, twiddle6_re, s_v6 * twiddle6_im);
                let siq6 = fma(c_v6, twiddle6_im, -s_v6 * twiddle6_re);

                u1 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_9, u1);
                v0 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_8, v0);

                let iq7 = fma(c_v7, twiddle7_re, s_v7 * twiddle7_im);
                let siq7 = fma(c_v7, twiddle7_im, -s_v7 * twiddle7_re);

                u1 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_4, u1);
                v0 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_5, v0);

                let iq8 = fma(c_v8, twiddle8_re, s_v8 * twiddle8_im);
                let siq8 = fma(c_v8, twiddle8_im, -s_v8 * twiddle8_re);

                u1 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_14, u1);
                v0 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_15, v0);

                u0 += iq1 + iq2 + iq3 + iq4 + iq5 + iq6 + iq7 + iq8 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                unsafe {
                    u0.write1(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write1(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.write1(chunk.get_unchecked_mut(s - S - k..));
                }

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f64::D4_R19_ROT_TWIDDLE_14;
                v2 *= f64::D4_R19_ROT_TWIDDLE_15;
                u2 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_9, u2);
                v2 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_8, v2);
                u2 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_12, u2);
                v2 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_13, v2);
                u2 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_1, u2);
                v2 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_0, v2);
                u2 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_2, u2);
                v2 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_3, v2);
                u2 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_6, u2);
                v2 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_7, v2);
                u2 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_10, u2);
                v2 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_11, v2);
                u2 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_16, u2);
                v2 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_17, v2);
                u2 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_4, u2);
                v2 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_5, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    uc2.write1(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write1(chunk.get_unchecked_mut(2 * s + k..));
                }

                let mut u3 = iq0;
                let mut v3 = siq0;
                u3 *= f64::D4_R19_ROT_TWIDDLE_6;
                v3 *= f64::D4_R19_ROT_TWIDDLE_7;
                u3 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_12, u3);
                v3 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_13, v3);
                u3 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_4, u3);
                v3 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_5, v3);
                u3 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_14, u3);
                v3 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_15, v3);
                u3 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_16, u3);
                v3 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_17, v3);
                u3 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_1, u3);
                v3 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_0, v3);
                u3 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_2, u3);
                v3 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_3, v3);
                u3 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_10, u3);
                v3 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_11, v3);
                u3 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_9, u3);
                v3 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_8, v3);
                u3 = u3 - a_v0;
                let uc4 = u3 - v3;
                let uc5 = u3 + v3;

                unsafe {
                    uc4.write1(chunk.get_unchecked_mut(3 * s - S - k..));
                }

                unsafe {
                    uc5.write1(chunk.get_unchecked_mut(3 * s + k..));
                }

                let mut u4 = iq0;
                let mut v4 = siq0;
                u4 *= f64::D4_R19_ROT_TWIDDLE_4;
                v4 *= f64::D4_R19_ROT_TWIDDLE_5;
                u4 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_10, u4);
                v4 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_11, v4);
                u4 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_2, u4);
                v4 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_3, v4);
                u4 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_12, u4);
                v4 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_13, v4);
                u4 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_14, u4);
                v4 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_15, v4);
                u4 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_9, u4);
                v4 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_8, v4);
                u4 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_1, u4);
                v4 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_0, v4);
                u4 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_6, u4);
                v4 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_7, v4);
                u4 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_16, u4);
                v4 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_17, v4);
                u4 += a_v0;
                let uc6 = u4 - v4;
                let uc7 = u4 + v4;

                unsafe {
                    uc6.write1(chunk.get_unchecked_mut(4 * s - S - k..));
                }

                unsafe {
                    uc7.write1(chunk.get_unchecked_mut(4 * s + k..));
                }

                let mut u5 = iq0;
                let mut v5 = siq0;
                u5 *= f64::D4_R19_ROT_TWIDDLE_1;
                v5 *= f64::D4_R19_ROT_TWIDDLE_0;
                u5 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_4, u5);
                v5 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_5, v5);
                u5 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_9, u5);
                v5 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_8, v5);
                u5 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_6, u5);
                v5 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_7, v5);
                u5 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_10, u5);
                v5 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_11, v5);
                u5 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_14, u5);
                v5 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_15, v5);
                u5 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_16, u5);
                v5 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_17, v5);
                u5 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_2, u5);
                v5 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_3, v5);
                u5 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_12, u5);
                v5 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_13, v5);
                u5 = u5 - a_v0;
                let uc8 = u5 - v5;
                let uc9 = u5 + v5;

                unsafe {
                    uc8.write1(chunk.get_unchecked_mut(5 * s - S - k..));
                }

                unsafe {
                    uc9.write1(chunk.get_unchecked_mut(5 * s + k..));
                }

                let mut u6 = iq0;
                let mut v6 = siq0;
                u6 *= f64::D4_R19_ROT_TWIDDLE_9;
                v6 *= f64::D4_R19_ROT_TWIDDLE_8;
                u6 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_2, u6);
                v6 = AvxStoreD::mul_f64_add(siq1, f64::D4_R19_ROT_TWIDDLE_3, v6);
                u6 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R19_ROT_TWIDDLE_16, u6);
                v6 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_17, v6);
                u6 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_4, u6);
                v6 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_5, v6);
                u6 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_6, u6);
                v6 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_7, v6);
                u6 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_12, u6);
                v6 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_13, v6);
                u6 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_14, u6);
                v6 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_15, v6);
                u6 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_1, u6);
                v6 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_0, v6);
                u6 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_10, u6);
                v6 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_11, v6);
                u6 += a_v0;
                let uc10 = u6 - v6;
                let uc11 = u6 + v6;

                unsafe {
                    uc10.write1(chunk.get_unchecked_mut(6 * s - S - k..));
                }

                unsafe {
                    uc11.write1(chunk.get_unchecked_mut(6 * s + k..));
                }

                let mut u7 = iq0;
                let mut v7 = siq0;
                u7 *= f64::D4_R19_ROT_TWIDDLE_10;
                v7 *= f64::D4_R19_ROT_TWIDDLE_11;
                u7 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_14, u7);
                v7 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_15, v7);
                u7 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_6, u7);
                v7 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R19_ROT_TWIDDLE_7, v7);
                u7 = AvxStoreD::mul_f64_add(iq3, f64::D4_R19_ROT_TWIDDLE_16, u7);
                v7 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_17, v7);
                u7 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R19_ROT_TWIDDLE_9, u7);
                v7 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_8, v7);
                u7 = AvxStoreD::mul_f64_add(iq5, f64::D4_R19_ROT_TWIDDLE_2, u7);
                v7 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_3, v7);
                u7 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_4, u7);
                v7 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_5, v7);
                u7 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_12, u7);
                v7 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_13, v7);
                u7 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_1, u7);
                v7 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_0, v7);
                u7 = u7 - a_v0;

                let uc12 = u7 - v7;
                let uc13 = u7 + v7;

                unsafe {
                    uc12.write1(chunk.get_unchecked_mut(7 * s - S - k..));
                }

                unsafe {
                    uc13.write1(chunk.get_unchecked_mut(7 * s + k..));
                }

                let mut u8 = iq0;
                let mut v8 = siq0;
                u8 *= -f64::D4_R19_ROT_TWIDDLE_16;
                v8 *= f64::D4_R19_ROT_TWIDDLE_17;
                u8 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R19_ROT_TWIDDLE_1, u8);
                v8 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_0, v8);
                u8 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_14, u8);
                v8 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_15, v8);
                u8 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_2, u8);
                v8 = AvxStoreD::mul_f64_add(siq3, f64::D4_R19_ROT_TWIDDLE_3, v8);
                u8 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_4, u8);
                v8 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R19_ROT_TWIDDLE_5, v8);
                u8 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_10, u8);
                v8 = AvxStoreD::mul_f64_add(siq5, f64::D4_R19_ROT_TWIDDLE_11, v8);
                u8 = AvxStoreD::mul_f64_add(iq6, -f64::D4_R19_ROT_TWIDDLE_12, u8);
                v8 = AvxStoreD::mul_f64_add(siq6, -f64::D4_R19_ROT_TWIDDLE_13, v8);
                u8 = AvxStoreD::mul_f64_add(iq7, f64::D4_R19_ROT_TWIDDLE_9, u8);
                v8 = AvxStoreD::mul_f64_add(siq7, f64::D4_R19_ROT_TWIDDLE_8, v8);
                u8 = AvxStoreD::mul_f64_add(iq8, -f64::D4_R19_ROT_TWIDDLE_6, u8);
                v8 = AvxStoreD::mul_f64_add(siq8, -f64::D4_R19_ROT_TWIDDLE_7, v8);
                u8 += a_v0;
                let uc14 = u8 - v8;
                let uc15 = u8 + v8;

                unsafe {
                    uc14.write1(chunk.get_unchecked_mut(8 * s - S - k..));
                }

                unsafe {
                    uc15.write1(chunk.get_unchecked_mut(8 * s + k..));
                }

                let mut u9 = iq0;
                let mut v9 = siq0;
                u9 *= f64::D4_R19_ROT_TWIDDLE_12;
                v9 *= f64::D4_R19_ROT_TWIDDLE_13;
                u9 = AvxStoreD::mul_f64_add(iq1, f64::D4_R19_ROT_TWIDDLE_16, u9);
                v9 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R19_ROT_TWIDDLE_17, v9);
                u9 = AvxStoreD::mul_f64_add(iq2, f64::D4_R19_ROT_TWIDDLE_10, u9);
                v9 = AvxStoreD::mul_f64_add(siq2, f64::D4_R19_ROT_TWIDDLE_11, v9);
                u9 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R19_ROT_TWIDDLE_9, u9);
                v9 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R19_ROT_TWIDDLE_8, v9);
                u9 = AvxStoreD::mul_f64_add(iq4, f64::D4_R19_ROT_TWIDDLE_1, u9);
                v9 = AvxStoreD::mul_f64_add(siq4, f64::D4_R19_ROT_TWIDDLE_0, v9);
                u9 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R19_ROT_TWIDDLE_4, u9);
                v9 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R19_ROT_TWIDDLE_5, v9);
                u9 = AvxStoreD::mul_f64_add(iq6, f64::D4_R19_ROT_TWIDDLE_6, u9);
                v9 = AvxStoreD::mul_f64_add(siq6, f64::D4_R19_ROT_TWIDDLE_7, v9);
                u9 = AvxStoreD::mul_f64_add(iq7, -f64::D4_R19_ROT_TWIDDLE_14, u9);
                v9 = AvxStoreD::mul_f64_add(siq7, -f64::D4_R19_ROT_TWIDDLE_15, v9);
                u9 = AvxStoreD::mul_f64_add(iq8, f64::D4_R19_ROT_TWIDDLE_2, u9);
                v9 = AvxStoreD::mul_f64_add(siq8, f64::D4_R19_ROT_TWIDDLE_3, v9);
                u9 = u9 - a_v0;
                let uc16 = u9 - v9;
                let uc17 = u9 + v9;

                unsafe {
                    uc16.write1(chunk.get_unchecked_mut(9 * s - S - k..));
                }

                unsafe {
                    uc17.write1(chunk.get_unchecked_mut(9 * s + k..));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct4::Dct4Identity;
    use crate::tests::naive_dct4;
    use rand::Rng;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 19];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = AvxDct4MixedRadix19d::new(input.len(), Arc::new(Dct4Identity::default())).unwrap();
        bf.execute(&mut input).unwrap();
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
