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
use crate::dct4::radixq_dct4_rotation_twiddle;
use crate::mla::fmla;
use crate::neon::util::NeonStoreF;
use crate::util::{DctConstants, DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

pub(crate) fn dct4_radix3_rotation_twiddles_neon(q_modules: usize, len: usize) -> Vec<NeonStoreF> {
    let simd_groups = q_modules.div_ceil(4);
    let main_q = 3usize;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    let mut twiddles = Vec::with_capacity(simd_groups * 2 * inner_groups);

    let working_modules = q_modules;

    let mut uk = 0usize;
    while uk + 4 <= working_modules {
        let k = uk;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for i in 0..4 {
            let layer = radixq_dct4_rotation_twiddle(main_q, 0, (k + i).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        uk += 4;
    }

    let remainder = working_modules - (working_modules / 4) * 4;
    if remainder > 0 {
        let k = uk;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for i in 0..remainder {
            let layer = radixq_dct4_rotation_twiddle(main_q, 0, (k + i).as_(), len);
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));
    }

    twiddles
}

pub(crate) struct NeonDct4MixedRadix3f {
    inner_dct4: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    rotation_twiddles: Vec<NeonStoreF>,
    execution_length: usize,
}

impl NeonDct4MixedRadix3f {
    pub(crate) fn new(
        len: usize,
        dct2: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct2.length(),
            len / 3,
            "DCT-IV Mixed-Radix-3 length DCTs must be third of DCT-IV"
        );

        Ok(Self {
            inner_dct4: dct2,
            execution_length: len,
            rotation_twiddles: dct4_radix3_rotation_twiddles_neon(len / 3, len),
        })
    }
}

impl PxdctExecutor<f32> for NeonDct4MixedRadix3f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }
        let mut scratch = try_vec![f32::default(); self.execution_length];

        let q_modules = self.execution_length / 3;

        let s = 2 * self.execution_length / 3;

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

            // Step 2: Apply DCT-IV to all buffers (A, C₀, C₁, S₀, S₁)
            self.inner_dct4.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules);

            // Step 4: Handle rotation twiddles
            let mut k = 0usize;
            let mut uk = 0usize;

            while k + 4 <= q_modules {
                const S: usize = 4;
                let c_v = NeonStoreF::load(unsafe { c_buffer.get_unchecked(k..) });
                let s_v = NeonStoreF::load(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                    .reverse();
                let a_v = NeonStoreF::load(unsafe { a_buffer.get_unchecked(k..) });

                let twiddle_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };

                let mut u0 = fmla(c_v, twiddle_re, s_v * twiddle_im);
                let mut u1 = u0;
                let mut v0 = fmla(c_v, twiddle_im, -s_v * twiddle_re);

                u0 += a_v;
                u1 *= f32::HALF;
                v0 *= f32::SQRT_3_OVER_2;
                u1 = u1 - a_v;

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
                k += 4;
                uk += 2;
            }

            let remainder = q_modules - k;
            if remainder == 3 {
                const S: usize = 3;
                let c_v = NeonStoreF::load3(unsafe { c_buffer.get_unchecked(k..) });
                let s_v = NeonStoreF::load3(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                    .reverse3();
                let a_v = NeonStoreF::load3(unsafe { a_buffer.get_unchecked(k..) });

                let twiddle_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };

                let mut u0 = fmla(c_v, twiddle_re, s_v * twiddle_im);
                let mut u1 = u0;
                let mut v0 = fmla(c_v, twiddle_im, -s_v * twiddle_re);

                u0 += a_v;
                u1 *= f32::HALF;
                v0 *= f32::SQRT_3_OVER_2;
                u1 = u1 - a_v;

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
            } else if remainder == 2 {
                const S: usize = 2;
                let c_v = NeonStoreF::load2(unsafe { c_buffer.get_unchecked(k..) });
                let s_v = NeonStoreF::load2(unsafe { s_buffer.get_unchecked(q_modules - k - S..) })
                    .reverse2();
                let a_v = NeonStoreF::load2(unsafe { a_buffer.get_unchecked(k..) });

                let twiddle_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };

                let mut u0 = fmla(c_v, twiddle_re, s_v * twiddle_im);
                let mut u1 = u0;
                let mut v0 = fmla(c_v, twiddle_im, -s_v * twiddle_re);

                u0 += a_v;
                u1 *= f32::HALF;
                v0 *= f32::SQRT_3_OVER_2;
                u1 = u1 - a_v;

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
            } else if remainder == 1 {
                const S: usize = 1;
                let c_v = NeonStoreF::load1(unsafe { c_buffer.get_unchecked(k..) });
                let s_v = NeonStoreF::load1(unsafe { s_buffer.get_unchecked(q_modules - k - S..) });
                let a_v = NeonStoreF::load1(unsafe { a_buffer.get_unchecked(k..) });

                let twiddle_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
                let twiddle_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };

                let mut u0 = fmla(c_v, twiddle_re, s_v * twiddle_im);
                let mut u1 = u0;
                let mut v0 = fmla(c_v, twiddle_im, -s_v * twiddle_re);

                u0 += a_v;
                u1 *= f32::HALF;
                v0 *= f32::SQRT_3_OVER_2;
                u1 = u1 - a_v;

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
    use crate::dct4::Dct4Butterfly3;
    use crate::tests::naive_dct4_f32;
    use rand::Rng;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 9];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let mut input = vec![
            1.8842070837939089f32,
            1.744160875935288,
            1.0859464680821782,
            1.8842070837939089,
            1.744160875935288,
            1.0859464680821782,
            1.8842070837939089,
            1.744160875935288,
            1.0859464680821782,
        ];
        let reference_input0 = input.clone();
        let reference_input = naive_dct4_f32(&reference_input0);
        let bf = NeonDct4MixedRadix3f::new(9, Arc::new(Dct4Butterfly3::default())).unwrap();
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
