/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::spectrum_mul::DctSpectrumMul;
use crate::util::{
    DctSample, create_dct2_3_real, force_cast_real_scratch_to_complex, try_vec, validate_scratch,
};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::R2CFftExecutor;

pub(crate) struct Dct2Fft<T> {
    twiddles: Vec<Complex<T>>,
    fft_executor: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    execution_length: usize,
    spectrum_mul: Arc<dyn DctSpectrumMul<T> + Send + Sync>,
    fft_scratch_size: usize,
}

create_dct2_3_real!(Dct2Fft);

impl<T: DctSample> PxdctExecutor<T> for Dct2Fft<T>
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

        let complex_len = self.execution_length / 2 + 1;
        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(self.execution_length);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        let even_end = self.execution_length.div_ceil(2);

        for chunk in data.chunks_exact_mut(self.execution_length) {
            for (dst, &src) in scratch_real
                .iter_mut()
                .zip(chunk.iter().step_by(2))
                .take(even_end)
            {
                *dst = src;
            }

            // the second half is the odd elements, in reverse order
            if self.execution_length > 1 {
                let odd_end = self.execution_length - self.execution_length % 2;
                let buffer = &mut scratch_real[even_end..even_end + self.execution_length / 2];
                let data_cutoff = &chunk[..odd_end];
                for (dst, &src) in buffer
                    .iter_mut()
                    .zip(data_cutoff.iter().rev().step_by(2))
                    .take(self.execution_length / 2)
                {
                    *dst = src;
                }
            }

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            self.spectrum_mul
                .mul_spectrum_to_real(scratch_complex, &self.twiddles, chunk);
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

        let complex_len = self.execution_length / 2 + 1;
        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(self.execution_length);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        let even_end = self.execution_length.div_ceil(2);

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            for (dst, &src) in scratch_real
                .iter_mut()
                .zip(src.iter().step_by(2))
                .take(even_end)
            {
                *dst = src;
            }

            // the second half is the odd elements, in reverse order
            if self.execution_length > 1 {
                let odd_end = self.execution_length - self.execution_length % 2;
                let buffer = &mut scratch_real[even_end..even_end + self.execution_length / 2];
                let data_cutoff = &src[..odd_end];
                for (dst, &src) in buffer
                    .iter_mut()
                    .zip(data_cutoff.iter().rev().step_by(2))
                    .take(self.execution_length / 2)
                {
                    *dst = src;
                }
            }

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            self.spectrum_mul
                .mul_spectrum_to_real(scratch_complex, &self.twiddles, dst);
        }
        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        let complex_len = self.execution_length / 2 + 1;
        self.execution_length + complex_len * 2 + self.fft_scratch_size * 2
    }
}

#[cfg(test)]
mod tests {
    use crate::PxdctExecutor;
    use crate::dct2::dct2fft::Dct2Fft;

    #[test]
    fn test_14() {
        let mut array = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let dct0 = Dct2Fft::<f32>::new(array.len()).unwrap();

        static CONTROL: [f32; 14] = [
            91.0,
            -39.6342,
            -1.6270865e-6,
            -4.326397,
            -2.255481e-6,
            -1.4956715,
            -2.5164427e-6,
            -0.70710677,
            -2.006796e-6,
            -0.3710423,
            -1.0861824e-6,
            -0.18536007,
            -3.7137184e-7,
            -0.05669479,
        ];

        dct0.execute(&mut array).unwrap();
        array.iter().zip(CONTROL.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-4, "Difference to control values exceeded 1e-4 when it shouldn't, value {x}, control {c} at {i}");
            });
    }

    #[test]
    fn test_15() {
        let mut array = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
        ];
        let dct0 = Dct2Fft::<f32>::new(array.len()).unwrap();

        static CONTROL: [f32; 15] = [
            105.0,
            -45.51088,
            -3.4831464e-7,
            -4.9797983,
            4.172325e-7,
            -1.7320545,
            -9.536743e-7,
            -0.82989204,
            5.364418e-7,
            -0.4490286,
            -2.0861626e-6,
            -0.24368116,
            -3.8148508e-7,
            -0.10865294,
            8.247489e-7,
        ];

        dct0.execute(&mut array).unwrap();
        array.iter().zip(CONTROL.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-4, "Difference to control values exceeded 1e-4 when it shouldn't, value {x}, control {c} at {i}");
        });
    }
}
