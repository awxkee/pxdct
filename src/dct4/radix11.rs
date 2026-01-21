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
use crate::dct4::prime_butterflies::Dct4MixedRadix11Sample;
use crate::dct4::utils::radixq_dct4_rotation_twiddle;
use crate::mla::fmla;
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct Dct4MixedRadix11<T> {
    inner_dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    rotation_twiddles: Vec<Complex<T>>,
    execution_length: usize,
}

impl<T: DctSample> Dct4MixedRadix11<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        dct2: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct2.length(),
            len / 11,
            "DCT-IV Mixed-Radix-11 length DCTs must be one eleventh of DCT-IV"
        );

        let mut twiddles = try_vec![Complex::<T>::default(); len / 11 * 5];
        for (k, dst) in twiddles.chunks_exact_mut(5).enumerate() {
            dst[0] = radixq_dct4_rotation_twiddle(11, 0, k, len);
            dst[1] = radixq_dct4_rotation_twiddle(11, 1, k, len);
            dst[2] = radixq_dct4_rotation_twiddle(11, 2, k, len);
            dst[3] = radixq_dct4_rotation_twiddle(11, 3, k, len);
            dst[4] = radixq_dct4_rotation_twiddle(11, 4, k, len);
        }

        Ok(Self {
            inner_dct4: dct2,
            execution_length: len,
            rotation_twiddles: twiddles,
        })
    }
}

impl<T: DctSample + Dct4MixedRadix11Sample> PxdctExecutor<T> for Dct4MixedRadix11<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }
        let mut scratch = try_vec![T::default(); self.execution_length];

        let q_modules = self.execution_length / 11;

        let s = 2 * self.execution_length / 11;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 5);

            // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
            for (n, dst) in a_buffer.iter_mut().enumerate() {
                unsafe {
                    *dst = *chunk.get_unchecked(n * 11 + 5);
                }
            }

            // Extract and combine symmetric pairs with sign alternation for S buffer
            for (m, (c_buffer, s_buffer)) in c_buffer
                .chunks_exact_mut(q_modules)
                .zip(s_buffer.chunks_exact_mut(q_modules))
                .enumerate()
            {
                let mut sign = T::one();
                for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate()
                {
                    let u0 = unsafe { *chunk.get_unchecked(11 * n + m) };
                    let u1 = unsafe { *chunk.get_unchecked(11 * n + 11 - m - 1) };

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }
            }

            // Step 2: Apply DCT-IV to all buffers (A, C₀, C₁, S₀, S₁)
            self.inner_dct4.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 5);

            // Step 4: Handle k≥0 cases with rotation twiddles
            for k in 0..q_modules {
                let c_v0 = unsafe { *c_buffer.get_unchecked(k) };
                let s_v0 = unsafe { *s_buffer.get_unchecked(q_modules - 1 - k) };
                let a_v0 = unsafe { *a_buffer.get_unchecked(k) };

                let c_v1 = unsafe { *c_buffer.get_unchecked(q_modules + k) };
                let s_v1 = unsafe { *s_buffer.get_unchecked(q_modules * 2 - 1 - k) };

                let c_v2 = unsafe { *c_buffer.get_unchecked(q_modules * 2 + k) };
                let s_v2 = unsafe { *s_buffer.get_unchecked(q_modules * 3 - 1 - k) };

                let c_v3 = unsafe { *c_buffer.get_unchecked(q_modules * 3 + k) };
                let s_v3 = unsafe { *s_buffer.get_unchecked(q_modules * 4 - 1 - k) };

                let c_v4 = unsafe { *c_buffer.get_unchecked(q_modules * 4 + k) };
                let s_v4 = unsafe { *s_buffer.get_unchecked(q_modules * 5 - 1 - k) };

                let twiddle0 = unsafe { self.rotation_twiddles.get_unchecked(k * 5) };
                let twiddle1 = unsafe { self.rotation_twiddles.get_unchecked(k * 5 + 1) };
                let twiddle2 = unsafe { self.rotation_twiddles.get_unchecked(k * 5 + 2) };
                let twiddle3 = unsafe { self.rotation_twiddles.get_unchecked(k * 5 + 3) };
                let twiddle4 = unsafe { self.rotation_twiddles.get_unchecked(k * 5 + 4) };

                let iq0 = fmla(c_v0, twiddle0.re, s_v0 * twiddle0.im);
                let siq0 = fmla(c_v0, twiddle0.im, -s_v0 * twiddle0.re);
                let mut u0 = iq0;
                let mut u1 = u0;
                let mut v0 = siq0;

                u1 *= T::D4_R11_ROT_TWIDDLE_2;
                v0 *= T::D4_R11_ROT_TWIDDLE_3;

                let iq1 = fmla(c_v1, twiddle1.re, s_v1 * twiddle1.im);
                let siq1 = fmla(c_v1, twiddle1.im, -s_v1 * twiddle1.re);

                u1 = fmla(iq1, T::D4_R11_ROT_TWIDDLE_1, u1);
                v0 = fmla(siq1, T::D4_R11_ROT_TWIDDLE_0, v0);

                let iq2 = fmla(c_v2, twiddle2.re, s_v2 * twiddle2.im);
                let siq2 = fmla(c_v2, twiddle2.im, -s_v2 * twiddle2.re);

                u1 = fmla(iq2, T::D4_R11_ROT_TWIDDLE_8, u1);
                v0 = fmla(siq2, T::D4_R11_ROT_TWIDDLE_9, v0);

                let iq3 = fmla(c_v3, twiddle3.re, s_v3 * twiddle3.im);
                let siq3 = fmla(c_v3, twiddle3.im, -s_v3 * twiddle3.re);

                u1 = fmla(iq3, -T::D4_R11_ROT_TWIDDLE_7, u1);
                v0 = fmla(siq3, T::D4_R11_ROT_TWIDDLE_6, v0);

                let iq4 = fmla(c_v4, twiddle4.re, s_v4 * twiddle4.im);
                let siq4 = fmla(c_v4, twiddle4.im, -s_v4 * twiddle4.re);

                u1 = fmla(iq4, -T::D4_R11_ROT_TWIDDLE_4, u1);
                v0 = fmla(siq4, T::D4_R11_ROT_TWIDDLE_5, v0);

                u0 += iq1 + iq2 + iq3 + iq4 + a_v0;
                u1 = u1 - a_v0;

                let uc0 = u1 - v0;
                let uc1 = u1 + v0;

                unsafe {
                    *chunk.get_unchecked_mut(k) = u0;
                }

                unsafe {
                    *chunk.get_unchecked_mut(s + k) = uc1;
                }

                unsafe {
                    *chunk.get_unchecked_mut(s - 1 - k) = uc0;
                }

                let mut u2 = iq0;
                let mut v2 = siq0;
                u2 *= T::D4_R11_ROT_TWIDDLE_4;
                v2 *= T::D4_R11_ROT_TWIDDLE_5;
                u2 = fmla(iq1, -T::D4_R11_ROT_TWIDDLE_8, u2);
                v2 = fmla(siq1, T::D4_R11_ROT_TWIDDLE_9, v2);
                u2 = fmla(iq2, -T::D4_R11_ROT_TWIDDLE_2, u2);
                v2 = fmla(siq2, T::D4_R11_ROT_TWIDDLE_3, v2);
                u2 = fmla(iq3, -T::D4_R11_ROT_TWIDDLE_1, u2);
                v2 = fmla(siq3, -T::D4_R11_ROT_TWIDDLE_0, v2);
                u2 = fmla(iq4, T::D4_R11_ROT_TWIDDLE_7, u2);
                v2 = fmla(siq4, -T::D4_R11_ROT_TWIDDLE_6, v2);
                u2 += a_v0;
                let uc2 = u2 - v2;
                let uc3 = u2 + v2;

                unsafe {
                    *chunk.get_unchecked_mut(2 * s - 1 - k) = uc2;
                }

                unsafe {
                    *chunk.get_unchecked_mut(2 * s + k) = uc3;
                }

                let mut u3 = iq0;
                let mut v3 = siq0;
                u3 *= T::D4_R11_ROT_TWIDDLE_1;
                v3 *= T::D4_R11_ROT_TWIDDLE_0;
                u3 = fmla(iq1, -T::D4_R11_ROT_TWIDDLE_4, u3);
                v3 = fmla(siq1, T::D4_R11_ROT_TWIDDLE_5, v3);
                u3 = fmla(iq2, -T::D4_R11_ROT_TWIDDLE_7, u3);
                v3 = fmla(siq2, -T::D4_R11_ROT_TWIDDLE_6, v3);
                u3 = fmla(iq3, T::D4_R11_ROT_TWIDDLE_2, u3);
                v3 = fmla(siq3, -T::D4_R11_ROT_TWIDDLE_3, v3);
                u3 = fmla(iq4, T::D4_R11_ROT_TWIDDLE_8, u3);
                v3 = fmla(siq4, T::D4_R11_ROT_TWIDDLE_9, v3);
                u3 = u3 - a_v0;
                let uc4 = u3 - v3;
                let uc5 = u3 + v3;

                unsafe {
                    *chunk.get_unchecked_mut(3 * s - 1 - k) = uc4;
                }

                unsafe {
                    *chunk.get_unchecked_mut(3 * s + k) = uc5;
                }

                let mut u4 = iq0;
                let mut v4 = siq0;
                u4 *= T::D4_R11_ROT_TWIDDLE_7;
                v4 *= T::D4_R11_ROT_TWIDDLE_6;
                u4 = fmla(iq1, -T::D4_R11_ROT_TWIDDLE_2, u4);
                v4 = fmla(siq1, -T::D4_R11_ROT_TWIDDLE_3, v4);
                u4 = fmla(iq2, T::D4_R11_ROT_TWIDDLE_4, u4);
                v4 = fmla(siq2, -T::D4_R11_ROT_TWIDDLE_5, v4);
                u4 = fmla(iq3, -T::D4_R11_ROT_TWIDDLE_8, u4);
                v4 = fmla(siq3, T::D4_R11_ROT_TWIDDLE_9, v4);
                u4 = fmla(iq4, -T::D4_R11_ROT_TWIDDLE_1, u4);
                v4 = fmla(siq4, -T::D4_R11_ROT_TWIDDLE_0, v4);
                u4 += a_v0;
                let uc6 = u4 - v4;
                let uc7 = u4 + v4;

                unsafe {
                    *chunk.get_unchecked_mut(4 * s - 1 - k) = uc6;
                }

                unsafe {
                    *chunk.get_unchecked_mut(4 * s + k) = uc7;
                }

                let mut u5 = iq0;
                let mut v5 = siq0;
                u5 *= T::D4_R11_ROT_TWIDDLE_8;
                v5 *= T::D4_R11_ROT_TWIDDLE_9;
                u5 = fmla(iq1, -T::D4_R11_ROT_TWIDDLE_7, u5);
                v5 = fmla(siq1, -T::D4_R11_ROT_TWIDDLE_6, v5);
                u5 = fmla(iq2, T::D4_R11_ROT_TWIDDLE_1, u5);
                v5 = fmla(siq2, T::D4_R11_ROT_TWIDDLE_0, v5);
                u5 = fmla(iq3, -T::D4_R11_ROT_TWIDDLE_4, u5);
                v5 = fmla(siq3, -T::D4_R11_ROT_TWIDDLE_5, v5);
                u5 = fmla(iq4, T::D4_R11_ROT_TWIDDLE_2, u5);
                v5 = fmla(siq4, T::D4_R11_ROT_TWIDDLE_3, v5);
                u5 = u5 - a_v0;
                let uc8 = u5 - v5;
                let uc9 = u5 + v5;

                unsafe {
                    *chunk.get_unchecked_mut(5 * s - 1 - k) = uc8;
                }

                unsafe {
                    *chunk.get_unchecked_mut(5 * s + k) = uc9;
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
    use crate::tests::naive_dct4;
    use rand::Rng;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 33];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4MixedRadix11::new(input.len(), Arc::new(Dct4Butterfly3::default())).unwrap();
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
