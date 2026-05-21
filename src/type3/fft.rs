/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
    DctSample, create_dct2_3, force_cast_real_scratch_to_complex, try_vec, validate_scratch,
};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor};

pub(crate) struct Dct3Fft<T> {
    twiddles: Vec<Complex<T>>,
    fft_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    execution_length: usize,
    spectrum_mul: Arc<dyn DctSpectrumMul<T> + Send + Sync>,
    fft_scratch_size: usize,
}

create_dct2_3!(Dct3Fft);

impl<T: DctSample> PxdctExecutor<T> for Dct3Fft<T> {
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
        let (unprepared_scratch, c1) = full_scratch.split_at_mut(self.execution_length * 2);
        let scratch = force_cast_real_scratch_to_complex(unprepared_scratch, self.execution_length);
        let fft_scratch = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for chunk in data.chunks_exact_mut(self.execution_length) {
            // compute the FFT buffer based on the twiddle factors
            self.spectrum_mul
                .mul_spectrum_and_half(chunk, &self.twiddles, scratch);

            // run the fft
            self.fft_executor
                .execute_with_scratch(scratch, fft_scratch)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            // copy the first half of the fft output into the even elements of the buffer
            let even_end = chunk.len().div_ceil(2);
            for (dst, src) in chunk
                .iter_mut()
                .step_by(2)
                .zip(scratch.iter())
                .take(even_end)
            {
                *dst = src.re;
            }

            // copy the second half of the fft buffer into the odd elements, reversed
            if self.execution_length > 1 {
                let odd_end = self.execution_length - self.execution_length % 2;
                let buffer = &mut chunk[..odd_end];
                let data_cutoff = &scratch[even_end..even_end + self.execution_length / 2];
                for (dst, src) in buffer
                    .iter_mut()
                    .rev()
                    .step_by(2)
                    .zip(data_cutoff.iter())
                    .take(self.execution_length / 2)
                {
                    *dst = src.re;
                }
            }
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
        let (unprepared_scratch, c1) = full_scratch.split_at_mut(self.execution_length * 2);
        let scratch = force_cast_real_scratch_to_complex(unprepared_scratch, self.execution_length);
        let fft_scratch = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            // compute the FFT buffer based on the twiddle factors
            self.spectrum_mul
                .mul_spectrum_and_half(src, &self.twiddles, scratch);

            // run the fft
            self.fft_executor
                .execute_with_scratch(scratch, fft_scratch)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            // copy the first half of the fft output into the even elements of the buffer
            let even_end = dst.len().div_ceil(2);
            for (dst, src) in dst.iter_mut().step_by(2).zip(scratch.iter()).take(even_end) {
                *dst = src.re;
            }

            // copy the second half of the fft buffer into the odd elements, reversed
            if self.execution_length > 1 {
                let odd_end = self.execution_length - self.execution_length % 2;
                let buffer = &mut dst[..odd_end];
                let data_cutoff = &scratch[even_end..even_end + self.execution_length / 2];
                for (dst, src) in buffer
                    .iter_mut()
                    .rev()
                    .step_by(2)
                    .zip(data_cutoff.iter())
                    .take(self.execution_length / 2)
                {
                    *dst = src.re;
                }
            }
        }
        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        self.execution_length * 2 + self.fft_scratch_size * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;

    #[test]
    fn test_dst_14() {
        let mut array = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let dct0 = Dct3Fft::<f32>::new(array.len()).unwrap();

        static CONTROL: [f32; 14] = [
            45.127357,
            -50.10906,
            21.035442,
            -18.6066,
            11.597095,
            -10.612154,
            7.2699604,
            -6.7052383,
            4.5909004,
            -4.1951275,
            2.6066022,
            -2.2874153,
            0.9321268,
            -0.64390063,
        ];

        dct0.execute(&mut array).unwrap();
        array.iter().zip(CONTROL.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-4, "Difference to control values exceeded 1e-4 when it shouldn't, value {x}, control {c} at {i}");
        })
    }

    #[test]
    fn test_dst_15() {
        let mut array = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
        ];
        let dct0 = Dct3Fft::<f32>::new(array.len()).unwrap();

        static CONTROL: [f32; 15] = [
            51.836082, -57.569, 24.258331, -21.48479, 13.506619, -12.391785, 8.630486, -7.999997,
            5.659441, -5.2259927, 3.5065365, -3.1658192, 1.7416692, -1.4441564, 0.14237356,
        ];

        dct0.execute(&mut array).unwrap();
        array.iter().zip(CONTROL.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-4, "Difference to control values exceeded 1e-4 when it shouldn't, value {x}, control {c} at {i}");
        })
    }
}
