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
use crate::dct4::{Dct4MixedRadix5Sample, radixq_dct4_rotation_twiddle};
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct4_radix5_rotation_twiddles_neon(q_modules: usize, len: usize) -> Vec<AvxStoreF> {
    let simd_groups = q_modules.div_ceil(8);
    let main_q = 5usize;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    let mut twiddles = Vec::with_capacity(simd_groups * 4 * inner_groups);

    let working_modules = q_modules;

    let mut uk = 0usize;
    while uk + 8 <= working_modules {
        let k = uk;

        let mut array_re = [0.; 8];
        let mut array_im = [0.; 8];
        for i in 0..8 {
            let layer = radixq_dct4_rotation_twiddle(main_q, 0, (k + i).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(AvxStoreF::load(array_re.as_ref()));
        twiddles.push(AvxStoreF::load(array_im.as_ref()));

        for i in 0..8 {
            let layer = radixq_dct4_rotation_twiddle(main_q, 1, (k + i).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(AvxStoreF::load(array_re.as_ref()));
        twiddles.push(AvxStoreF::load(array_im.as_ref()));

        uk += 8;
    }

    let remainder = working_modules - (working_modules / 8) * 8;
    if remainder > 0 {
        let k = uk;

        let mut array_re = [0.; 8];
        let mut array_im = [0.; 8];
        for i in 0..remainder {
            let layer = radixq_dct4_rotation_twiddle(main_q, 0, (k + i).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(AvxStoreF::load(array_re.as_ref()));
        twiddles.push(AvxStoreF::load(array_im.as_ref()));

        for i in 0..remainder {
            let layer = radixq_dct4_rotation_twiddle(main_q, 1, (k + i).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(AvxStoreF::load(array_re.as_ref()));
        twiddles.push(AvxStoreF::load(array_im.as_ref()));
    }

    twiddles
}

pub(crate) struct AvxDct4MixedRadix5f {
    inner_dct4: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    rotation_twiddles: Vec<AvxStoreF>,
    execution_length: usize,
}

impl AvxDct4MixedRadix5f {
    pub(crate) fn new(
        len: usize,
        dct2: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct2.length(),
            len / 5,
            "DCT-IV Mixed-Radix-5 length DCTs must be one fifth of DCT-IV"
        );

        Ok(Self {
            inner_dct4: dct2,
            execution_length: len,
            rotation_twiddles: unsafe { dct4_radix5_rotation_twiddles_neon(len / 5, len) },
        })
    }
}

impl AvxDct4MixedRadix5f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }
        let mut scratch = try_vec![f32::default(); self.execution_length];

        let q_modules = self.execution_length / 5;

        let s = 2 * self.execution_length / 5;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 2);

            // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
            for (n, dst) in a_buffer.iter_mut().enumerate() {
                unsafe {
                    *dst = *chunk.get_unchecked(n * 5 + 2);
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
                    let u0 = unsafe { *chunk.get_unchecked(5 * n + m) };
                    let u1 = unsafe { *chunk.get_unchecked(5 * n + 5 - m - 1) };

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }
            }

            // Step 2: Apply DCT-IV to all buffers (A, C₀, C₁, S₀, S₁)
            self.inner_dct4.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 2);

            let mut uk = 0usize;

            // Step 4: Handle k≥0 cases with rotation twiddles
            let mut k = 0usize;
            while k + 8 <= q_modules {
                const S: usize = 8;
                let c_v0 = AvxStoreF::load(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreF::load(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse();
                let a_v0 = AvxStoreF::load(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreF::load(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreF::load(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f32::D4_R5_ROT_TWIDDLE_0;
                v0 *= f32::D4_R5_ROT_TWIDDLE_1;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreF::f32_mul_add(-f32::D4_R5_ROT_TWIDDLE_3, iq1, u1);
                v0 = AvxStoreF::f32_mul_add(f32::D4_R5_ROT_TWIDDLE_2, siq1, v0);

                u0 += iq1 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f32::D4_R5_ROT_TWIDDLE_3;
                v2 *= f32::D4_R5_ROT_TWIDDLE_2;
                u2 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R5_ROT_TWIDDLE_0, u2);
                v2 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R5_ROT_TWIDDLE_1, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    u0.write(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse().write(chunk.get_unchecked_mut(s - S - k..));
                }

                unsafe {
                    uc2.reverse()
                        .write(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write(chunk.get_unchecked_mut(2 * s + k..));
                }
                uk += 4;
                k += 8;
            }

            let rem = q_modules - k;
            if rem == 7 {
                const S: usize = 7;
                let c_v0 = AvxStoreF::load7(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreF::load7(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse7();
                let a_v0 = AvxStoreF::load7(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreF::load7(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreF::load7(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse7();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f32::D4_R5_ROT_TWIDDLE_0;
                v0 *= f32::D4_R5_ROT_TWIDDLE_1;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreF::f32_mul_add(-f32::D4_R5_ROT_TWIDDLE_3, iq1, u1);
                v0 = AvxStoreF::f32_mul_add(f32::D4_R5_ROT_TWIDDLE_2, siq1, v0);

                u0 += iq1 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f32::D4_R5_ROT_TWIDDLE_3;
                v2 *= f32::D4_R5_ROT_TWIDDLE_2;
                u2 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R5_ROT_TWIDDLE_0, u2);
                v2 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R5_ROT_TWIDDLE_1, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    u0.write7(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write7(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse7().write7(chunk.get_unchecked_mut(s - S - k..));
                }

                unsafe {
                    uc2.reverse7()
                        .write7(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write7(chunk.get_unchecked_mut(2 * s + k..));
                }
            } else if rem == 6 {
                const S: usize = 6;
                let c_v0 = AvxStoreF::load6(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreF::load6(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse6();
                let a_v0 = AvxStoreF::load6(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreF::load6(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreF::load6(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse6();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f32::D4_R5_ROT_TWIDDLE_0;
                v0 *= f32::D4_R5_ROT_TWIDDLE_1;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreF::f32_mul_add(-f32::D4_R5_ROT_TWIDDLE_3, iq1, u1);
                v0 = AvxStoreF::f32_mul_add(f32::D4_R5_ROT_TWIDDLE_2, siq1, v0);

                u0 += iq1 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f32::D4_R5_ROT_TWIDDLE_3;
                v2 *= f32::D4_R5_ROT_TWIDDLE_2;
                u2 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R5_ROT_TWIDDLE_0, u2);
                v2 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R5_ROT_TWIDDLE_1, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    u0.write6(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write6(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse6().write6(chunk.get_unchecked_mut(s - S - k..));
                }

                unsafe {
                    uc2.reverse6()
                        .write6(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write6(chunk.get_unchecked_mut(2 * s + k..));
                }
            } else if rem == 5 {
                const S: usize = 5;
                let c_v0 = AvxStoreF::load5(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreF::load5(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse5();
                let a_v0 = AvxStoreF::load5(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreF::load5(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreF::load5(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse5();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f32::D4_R5_ROT_TWIDDLE_0;
                v0 *= f32::D4_R5_ROT_TWIDDLE_1;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreF::f32_mul_add(-f32::D4_R5_ROT_TWIDDLE_3, iq1, u1);
                v0 = AvxStoreF::f32_mul_add(f32::D4_R5_ROT_TWIDDLE_2, siq1, v0);

                u0 += iq1 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f32::D4_R5_ROT_TWIDDLE_3;
                v2 *= f32::D4_R5_ROT_TWIDDLE_2;
                u2 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R5_ROT_TWIDDLE_0, u2);
                v2 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R5_ROT_TWIDDLE_1, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    u0.write5(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write5(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse5().write5(chunk.get_unchecked_mut(s - S - k..));
                }

                unsafe {
                    uc2.reverse5()
                        .write5(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write5(chunk.get_unchecked_mut(2 * s + k..));
                }
            } else if rem == 4 {
                const S: usize = 4;
                let c_v0 = AvxStoreF::load4(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreF::load4(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse4();
                let a_v0 = AvxStoreF::load4(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreF::load4(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreF::load4(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse4();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f32::D4_R5_ROT_TWIDDLE_0;
                v0 *= f32::D4_R5_ROT_TWIDDLE_1;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreF::f32_mul_add(-f32::D4_R5_ROT_TWIDDLE_3, iq1, u1);
                v0 = AvxStoreF::f32_mul_add(f32::D4_R5_ROT_TWIDDLE_2, siq1, v0);

                u0 += iq1 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f32::D4_R5_ROT_TWIDDLE_3;
                v2 *= f32::D4_R5_ROT_TWIDDLE_2;
                u2 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R5_ROT_TWIDDLE_0, u2);
                v2 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R5_ROT_TWIDDLE_1, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    u0.write4(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write4(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse4().write4(chunk.get_unchecked_mut(s - S - k..));
                }

                unsafe {
                    uc2.reverse4()
                        .write4(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write4(chunk.get_unchecked_mut(2 * s + k..));
                }
            } else if rem == 3 {
                const S: usize = 3;
                let c_v0 = AvxStoreF::load3(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreF::load3(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse3();
                let a_v0 = AvxStoreF::load3(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreF::load3(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreF::load3(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse3();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f32::D4_R5_ROT_TWIDDLE_0;
                v0 *= f32::D4_R5_ROT_TWIDDLE_1;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreF::f32_mul_add(-f32::D4_R5_ROT_TWIDDLE_3, iq1, u1);
                v0 = AvxStoreF::f32_mul_add(f32::D4_R5_ROT_TWIDDLE_2, siq1, v0);

                u0 += iq1 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f32::D4_R5_ROT_TWIDDLE_3;
                v2 *= f32::D4_R5_ROT_TWIDDLE_2;
                u2 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R5_ROT_TWIDDLE_0, u2);
                v2 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R5_ROT_TWIDDLE_1, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    u0.write3(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write3(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse3().write3(chunk.get_unchecked_mut(s - S - k..));
                }

                unsafe {
                    uc2.reverse3()
                        .write3(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write3(chunk.get_unchecked_mut(2 * s + k..));
                }
            } else if rem == 2 {
                const S: usize = 2;
                let c_v0 = AvxStoreF::load2(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreF::load2(unsafe { s_buffer.get_unchecked(q_modules - S - k..) })
                    .reverse2();
                let a_v0 = AvxStoreF::load2(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreF::load2(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreF::load2(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) })
                        .reverse2();

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f32::D4_R5_ROT_TWIDDLE_0;
                v0 *= f32::D4_R5_ROT_TWIDDLE_1;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreF::f32_mul_add(-f32::D4_R5_ROT_TWIDDLE_3, iq1, u1);
                v0 = AvxStoreF::f32_mul_add(f32::D4_R5_ROT_TWIDDLE_2, siq1, v0);

                u0 += iq1 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f32::D4_R5_ROT_TWIDDLE_3;
                v2 *= f32::D4_R5_ROT_TWIDDLE_2;
                u2 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R5_ROT_TWIDDLE_0, u2);
                v2 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R5_ROT_TWIDDLE_1, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    u0.write2(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write2(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.reverse2().write2(chunk.get_unchecked_mut(s - S - k..));
                }

                unsafe {
                    uc2.reverse2()
                        .write2(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write2(chunk.get_unchecked_mut(2 * s + k..));
                }
            } else if rem == 1 {
                const S: usize = 1;
                let c_v0 = AvxStoreF::load1(unsafe { c_buffer.get_unchecked(k..) });
                let s_v0 = AvxStoreF::load1(unsafe { s_buffer.get_unchecked(q_modules - S - k..) });
                let a_v0 = AvxStoreF::load1(unsafe { a_buffer.get_unchecked(k..) });

                let c_v1 = AvxStoreF::load1(unsafe { c_buffer.get_unchecked(q_modules + k..) });
                let s_v1 =
                    AvxStoreF::load1(unsafe { s_buffer.get_unchecked(q_modules * 2 - S - k..) });

                let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
                let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
                let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

                let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
                let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= f32::D4_R5_ROT_TWIDDLE_0;
                v0 *= f32::D4_R5_ROT_TWIDDLE_1;

                let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
                let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

                u1 = AvxStoreF::f32_mul_add(-f32::D4_R5_ROT_TWIDDLE_3, iq1, u1);
                v0 = AvxStoreF::f32_mul_add(f32::D4_R5_ROT_TWIDDLE_2, siq1, v0);

                u0 += iq1 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= f32::D4_R5_ROT_TWIDDLE_3;
                v2 *= f32::D4_R5_ROT_TWIDDLE_2;
                u2 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R5_ROT_TWIDDLE_0, u2);
                v2 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R5_ROT_TWIDDLE_1, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    u0.write1(chunk.get_unchecked_mut(k..));
                }

                unsafe {
                    uc1.write1(chunk.get_unchecked_mut(s + k..));
                }

                unsafe {
                    uc0.write1(chunk.get_unchecked_mut(s - S - k..));
                }

                unsafe {
                    uc2.write1(chunk.get_unchecked_mut(2 * s - S - k..));
                }

                unsafe {
                    uc3.write1(chunk.get_unchecked_mut(2 * s + k..));
                }
            }
        }

        Ok(())
    }
}

impl PxdctExecutor<f32> for AvxDct4MixedRadix5f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct4::Dct4Butterfly8;
    use crate::tests::naive_dct4_f32;
    use crate::util::has_valid_avx;
    use rand::Rng;

    #[test]
    fn test_split_dct4() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0.; 40];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4_f32(&reference_input);
        let bf = AvxDct4MixedRadix5f::new(40, Arc::new(Dct4Butterfly8::default())).unwrap();
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
