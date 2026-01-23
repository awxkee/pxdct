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
use crate::dct4::utils::radixq_dct4_rotation_twiddle;
use crate::mla::fmla;
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct Dct4MixedRadix3<T> {
    inner_dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    rotation_twiddles: Vec<Complex<T>>,
    execution_length: usize,
}

impl<T: DctSample> Dct4MixedRadix3<T>
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
            len / 3,
            "DCT-IV Mixed-Radix-3 length DCTs must be third of DCT-IV"
        );

        let mut twiddles = try_vec![Complex::<T>::default(); len / 3];
        for (k, dst) in twiddles.iter_mut().enumerate() {
            *dst = radixq_dct4_rotation_twiddle(3, 0, k, len);
        }

        Ok(Self {
            inner_dct4: dct2,
            execution_length: len,
            rotation_twiddles: twiddles,
        })
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4MixedRadix3<T>
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
                let mut sign = T::one();
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

            // Step 4: Handle k≥1 cases with rotation twiddles
            for k in 0..q_modules {
                let c_v = unsafe { *c_buffer.get_unchecked(k) };
                let s_v = unsafe { *s_buffer.get_unchecked(q_modules - 1 - k) };
                let a_v = unsafe { *a_buffer.get_unchecked(k) };

                let twiddle = unsafe { self.rotation_twiddles.get_unchecked(k) };

                let mut u0 = fmla(c_v, twiddle.re, s_v * twiddle.im);
                let mut u1 = u0;
                let mut v0 = fmla(c_v, twiddle.im, -s_v * twiddle.re);

                u0 += a_v;
                u1 *= T::HALF;
                v0 *= T::SQRT_3_OVER_2;
                u1 = u1 - a_v;

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
        let mut input = vec![0.; 9];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let mut input = vec![
            1.8842070837939089,
            1.744160875935288,
            1.0859464680821782,
            1.8842070837939089,
            1.744160875935288,
            1.0859464680821782,
            1.8842070837939089,
            1.744160875935288,
            1.0859464680821782,
        ];
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4MixedRadix3::new(9, Arc::new(Dct4Butterfly3::default())).unwrap();
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
