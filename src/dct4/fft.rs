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
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor};

pub struct Dct4Fft<T> {
    fft_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    execution_length: usize,
}

impl<T: DctSample> Dct4Fft<T> {
    /// Creates a new DCT4 context that will process signals of length `inner_fft.len()`. `inner_fft.len()` must be odd.
    pub fn new(fft_executor: Arc<dyn FftExecutor<T> + Send + Sync>) -> Self {
        assert_eq!(
            fft_executor.direction(),
            FftDirection::Forward,
            "Dct4Odd requires a forward FFT, but an inverse FFT was provided"
        );

        let len = fft_executor.length();

        assert!(
            !len.is_multiple_of(2),
            "Dct4Odd size must be odd. Got {}",
            len
        );

        Self {
            execution_length: len,
            fft_executor,
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4Fft<T>
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
        let mut scratch = try_vec![Complex::<T>::default(); self.execution_length];

        let len = self.length();
        let half_len = len / 2;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let mut input_index = half_len;
            let mut fft_index = 0;
            while input_index < len {
                scratch[fft_index] = Complex {
                    re: chunk[input_index],
                    im: T::zero(),
                };

                input_index += 4;
                fft_index += 1;
            }

            //subtract len to simulate modular arithmetic
            input_index -= len;
            while input_index < len {
                scratch[fft_index] = Complex {
                    re: -chunk[len - input_index - 1],
                    im: T::zero(),
                };

                input_index += 4;
                fft_index += 1;
            }

            input_index -= len;
            while input_index < len {
                scratch[fft_index] = Complex {
                    re: -chunk[input_index],
                    im: T::zero(),
                };

                input_index += 4;
                fft_index += 1;
            }

            input_index -= len;
            while input_index < len {
                scratch[fft_index] = Complex {
                    re: chunk[len - input_index - 1],
                    im: T::zero(),
                };

                input_index += 4;
                fft_index += 1;
            }

            input_index -= len;
            while fft_index < len {
                scratch[fft_index] = Complex {
                    re: chunk[input_index],
                    im: T::zero(),
                };

                input_index += 4;
                fft_index += 1;
            }

            // run the fft
            self.fft_executor
                .execute(&mut scratch)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            let result_scale = T::SQRT_2 * T::HALF;
            let second_half_sign = if len % 4 == 1 { T::one() } else { -T::one() };

            //post-process the results 4 at a time
            let mut output_sign = T::one();

            let (left, right) = chunk.split_at_mut(half_len);

            for ((l_dst, r_dst), scratch) in left
                .chunks_exact_mut(2)
                .zip(right.rchunks_exact_mut(2))
                .zip(scratch.chunks_exact(4))
            {
                let fft_result = scratch[1] * (output_sign * result_scale);
                let next_result = scratch[3] * (output_sign * result_scale);

                l_dst[0] = fft_result.re + fft_result.im;
                l_dst[1] = -next_result.re + next_result.im;

                r_dst[0] = (next_result.re + next_result.im) * second_half_sign;
                r_dst[1] = (fft_result.re - fft_result.im) * second_half_sign;

                output_sign = output_sign.neg();
            }

            //we either have 1 or 3 elements left over that we couldn't get in the above loop, handle them here
            if len % 4 == 1 {
                chunk[half_len] = scratch[0].re * output_sign * result_scale;
            } else {
                let fft_result = scratch[len - 2] * (output_sign * result_scale);

                chunk[half_len - 1] = fft_result.re + fft_result.im;
                chunk[half_len + 1] = -fft_result.re + fft_result.im;
                chunk[half_len] = -scratch[0].re * output_sign * result_scale;
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
    use crate::tests::naive_dct4;
    use rand::Rng;
    use zaft::Zaft;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 15];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4Fft::new(Zaft::make_forward_fft_f64(15).unwrap());
        bf.execute(&mut input).unwrap();
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                assert!(
                    (src - r0).abs() < 1e-7,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-7,
                    (src - r0).abs()
                )
            });
    }

    #[test]
    fn test_split_dct4_35() {
        let mut input = vec![0.; 35];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4Fft::new(Zaft::make_forward_fft_f64(35).unwrap());
        bf.execute(&mut input).unwrap();
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                assert!(
                    (src - r0).abs() < 1e-7,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-7,
                    (src - r0).abs()
                )
            });
    }

    #[test]
    fn test_split_dct4_17() {
        let mut input = vec![0.; 17];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4Fft::new(Zaft::make_forward_fft_f64(17).unwrap());
        bf.execute(&mut input).unwrap();
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                assert!(
                    (src - r0).abs() < 1e-7,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-7,
                    (src - r0).abs()
                )
            });
    }
}
