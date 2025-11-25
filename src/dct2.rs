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
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor};

pub(crate) struct Dct2Fft<T> {
    twiddles: Vec<Complex<T>>,
    fft_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    length: usize,
    spectrum_mul: Arc<dyn DctSpectrumMul<T> + Send + Sync>,
}

macro_rules! create_dct2_3 {
    ($clazz: ident) => {
        impl<T: DctSample> $clazz<T>
            where
                f64: AsPrimitive<T>,
            {
                pub(crate) fn new(len: usize) -> Result<$clazz<T>, PxdctError> {
                    let fft = T::make_fft(len, FftDirection::Forward)?;
                    use crate::twiddles::compute_twiddle;
                    let mut twiddles = try_vec![Complex::<T>::default(); len];
                    for (i, twiddle) in twiddles.iter_mut().enumerate() {
                        *twiddle = compute_twiddle::<T>(i, len * 4);
                    }

                    Ok($clazz {
                        twiddles,
                        fft_executor: fft,
                        length: len,
                        spectrum_mul: T::create_mul_spectrum_to_real(),
                    })
                }
            }
    };
}

pub(crate) use create_dct2_3;

create_dct2_3!(Dct2Fft);

impl<T: DctSample> PxdctExecutor<T> for Dct2Fft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.length) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length));
        }

        let mut scratch = try_vec![Complex::<T>::default(); data.len()];

        let even_end = data.len().div_ceil(2);

        for chunk in data.chunks_exact_mut(self.length) {
            for (dst, src) in scratch
                .iter_mut()
                .zip(chunk.iter().step_by(2))
                .take(even_end)
            {
                *dst = Complex::from(src);
            }

            // the second half is the odd elements, in reverse order
            if self.length > 1 {
                let odd_end = self.length() - self.length() % 2;
                let buffer = &mut scratch[even_end..even_end + self.length / 2];
                let data_cutoff = &chunk[..odd_end];
                for (dst, src) in buffer
                    .iter_mut()
                    .zip(data_cutoff.iter().rev().step_by(2))
                    .take(self.length / 2)
                {
                    *dst = Complex::from(src);
                }
            }

            self.fft_executor
                .execute(&mut scratch)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            self.spectrum_mul
                .mul_spectrum_to_real(&scratch, &self.twiddles, chunk);
        }

        Ok(())
    }

    fn length(&self) -> usize {
        self.length
    }
}

#[cfg(test)]
mod tests {
    use crate::PxdctExecutor;
    use crate::dct2::Dct2Fft;

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
