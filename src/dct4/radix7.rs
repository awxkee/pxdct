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
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::dct4::prime_butterflies::Dct4MixedRadix7Sample;
use crate::dct4::utils::radixq_dct4_rotation_twiddle;
use crate::mla::fmla;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct Dct4MixedRadix7<T> {
    inner_dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    rotation_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    q_modules: usize,
    s: usize,
    inner_dct4_scratch_size: usize,
}

impl<T: DctSample> Dct4MixedRadix7<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct4.length(),
            len / 7,
            "DCT-IV Mixed-Radix-7 length DCTs must be one seventh of DCT-IV"
        );

        let mut twiddles = try_vec![Complex::<T>::default(); len / 7 * 3];
        for (k, dst) in twiddles.chunks_exact_mut(3).enumerate() {
            dst[0] = radixq_dct4_rotation_twiddle(7, 0, k, len);
            dst[1] = radixq_dct4_rotation_twiddle(7, 1, k, len);
            dst[2] = radixq_dct4_rotation_twiddle(7, 2, k, len);
        }

        let q_modules = len / 7;
        let s = 2 * len / 7;

        let inner_dct4_scratch_size = dct4.scratch_size();

        Ok(Self {
            inner_dct4: dct4,
            execution_length: len,
            rotation_twiddles: twiddles,
            q_modules,
            s,
            inner_dct4_scratch_size,
        })
    }
}

impl<T: DctSample + Dct4MixedRadix7Sample> Dct4MixedRadix7<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(self.q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(self.q_modules * 3);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 7 + 3];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(self.q_modules)
            .zip(s_buffer.chunks_exact_mut(self.q_modules))
            .enumerate()
        {
            let mut sign = T::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[7 * n + m];
                let u1 = data[7 * n + 7 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-IV to all buffers (A, C₀, C₁, S₀, S₁)
        self.inner_dct4
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(self.q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(self.q_modules * 3);

        // Step 4: Handle k≥0 cases with rotation twiddles
        for k in 0..self.q_modules {
            let c_v0 = unsafe { *c_buffer.get_unchecked(k) };
            let s_v0 = unsafe { *s_buffer.get_unchecked(self.q_modules - 1 - k) };
            let a_v0 = unsafe { *a_buffer.get_unchecked(k) };

            let c_v1 = unsafe { *c_buffer.get_unchecked(self.q_modules + k) };
            let s_v1 = unsafe { *s_buffer.get_unchecked(self.q_modules * 2 - 1 - k) };

            let c_v2 = unsafe { *c_buffer.get_unchecked(self.q_modules * 2 + k) };
            let s_v2 = unsafe { *s_buffer.get_unchecked(self.q_modules * 3 - 1 - k) };

            let twiddle0 = unsafe { self.rotation_twiddles.get_unchecked(k * 3) };
            let twiddle1 = unsafe { self.rotation_twiddles.get_unchecked(k * 3 + 1) };
            let twiddle2 = unsafe { self.rotation_twiddles.get_unchecked(k * 3 + 2) };

            let iq0 = fmla(c_v0, twiddle0.re, s_v0 * twiddle0.im);
            let siq0 = fmla(c_v0, twiddle0.im, -s_v0 * twiddle0.re);
            let mut u0 = iq0;
            let mut u1 = u0;
            let mut v0 = siq0;

            u1 *= T::D4_R7_ROT_TWIDDLE_2;
            v0 *= T::D4_R7_ROT_TWIDDLE_3;

            let iq1 = fmla(c_v1, twiddle1.re, s_v1 * twiddle1.im);
            let siq1 = fmla(c_v1, twiddle1.im, -s_v1 * twiddle1.re);

            u1 = fmla(iq1, T::D4_R7_ROT_TWIDDLE_4, u1);
            v0 = fmla(siq1, T::D4_R7_ROT_TWIDDLE_5, v0);

            let iq2 = fmla(c_v2, twiddle2.re, s_v2 * twiddle2.im);
            let siq2 = fmla(c_v2, twiddle2.im, -s_v2 * twiddle2.re);

            u1 = fmla(iq2, -T::D4_R7_ROT_TWIDDLE_1, u1);
            v0 = fmla(siq2, T::D4_R7_ROT_TWIDDLE_0, v0);

            u0 += iq1 + iq2 + a_v0;
            u1 = u1 - a_v0;

            let uc0 = u1 - v0;
            let uc1 = u1 + v0;

            let mut u2 = iq0;
            let mut v2 = siq0;
            u2 *= T::D4_R7_ROT_TWIDDLE_1;
            v2 *= T::D4_R7_ROT_TWIDDLE_0;
            u2 = fmla(iq1, -T::D4_R7_ROT_TWIDDLE_2, u2);
            v2 = fmla(siq1, T::D4_R7_ROT_TWIDDLE_3, v2);
            u2 = fmla(iq2, -T::D4_R7_ROT_TWIDDLE_4, u2);
            v2 = fmla(siq2, -T::D4_R7_ROT_TWIDDLE_5, v2);
            u2 += a_v0;
            let uc2 = u2 - v2;
            let uc3 = u2 + v2;
            let mut u3 = iq0;
            let mut v3 = siq0;
            u3 *= T::D4_R7_ROT_TWIDDLE_4;
            v3 *= T::D4_R7_ROT_TWIDDLE_5;
            u3 = fmla(iq1, -T::D4_R7_ROT_TWIDDLE_1, u3);
            v3 = fmla(siq1, -T::D4_R7_ROT_TWIDDLE_0, v3);
            u3 = fmla(iq2, T::D4_R7_ROT_TWIDDLE_2, u3);
            v3 = fmla(siq2, T::D4_R7_ROT_TWIDDLE_3, v3);
            u3 = u3 - a_v0;
            let uc4 = u3 - v3;
            let uc5 = u3 + v3;

            data[k] = u0;
            data[self.s + k] = uc1;
            data[self.s - 1 - k] = uc0;
            data[2 * self.s - 1 - k] = uc2;
            data[2 * self.s + k] = uc3;
            data[3 * self.s - 1 - k] = uc4;
            data[3 * self.s + k] = uc5;
        }
        Ok(())
    }
}

impl<T: DctSample + Dct4MixedRadix7Sample> PxdctExecutor<T> for Dct4MixedRadix7<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        for chunk in data.chunks_exact_mut(self.execution_length) {
            self.execute_with_store(&mut InPlaceStore::new(chunk), full_scratch)?;
        }

        Ok(())
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_into_with_scratch(input, output, &mut scratch)
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, self.execution_length);

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.execute_with_store(&mut BiStore::new(src, dst), full_scratch)?;
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        self.execution_length + self.inner_dct4_scratch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct4::Dct4Identity;
    use crate::tests::naive_dct4;
    use rand::RngExt;

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
            1.8842070837939089,
            1.744160875935288,
        ];
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4MixedRadix7::new(7, Arc::new(Dct4Identity::default())).unwrap();
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
