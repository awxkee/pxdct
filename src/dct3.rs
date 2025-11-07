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
use crate::dct2::create_dct2_3;
use crate::spectrum_mul::DctSpectrumMul;
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor};

pub(crate) struct Dct3Fft<T> {
    twiddles: Vec<Complex<T>>,
    fft_executor: Box<dyn FftExecutor<T> + Send + Sync>,
    length: usize,
    spectrum_mul: Arc<dyn DctSpectrumMul<T> + Send + Sync>,
}

create_dct2_3!(Dct3Fft);

impl<T: DctSample> PxdctExecutor<T> for Dct3Fft<T> {
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.length) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length));
        }

        let mut scratch = try_vec![Complex::<T>::default(); data.len()];

        for chunk in data.chunks_exact_mut(self.length) {
            // compute the FFT buffer based on the twiddle factors
            self.spectrum_mul
                .mul_spectrum_and_half(chunk, &self.twiddles, &mut scratch);

            // run the fft
            self.fft_executor
                .execute(&mut scratch)
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
            if self.length > 1 {
                let odd_end = self.length - self.length % 2;
                let buffer = &mut chunk[..odd_end];
                let data_cutoff = &scratch[even_end..even_end + self.length / 2];
                for (dst, src) in buffer
                    .iter_mut()
                    .rev()
                    .step_by(2)
                    .zip(data_cutoff.iter())
                    .take(self.length / 2)
                {
                    *dst = src.re;
                }
            }
        }

        Ok(())
    }

    fn length(&self) -> usize {
        self.length
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
