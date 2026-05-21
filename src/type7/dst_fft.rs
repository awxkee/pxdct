/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

use crate::util::{DctSample, force_cast_real_scratch_to_complex, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::R2CFftExecutor;

pub(crate) struct Dst7Fft<T> {
    fft_executor: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    execution_length: usize,
    fft_scratch_size: usize,
}

impl<T: DctSample> Dst7Fft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(len: usize) -> Result<Dst7Fft<T>, PxdctError> {
        let fft_size = 2 * len + 1;
        let inner_fft = T::make_fft_r2c(fft_size)?;
        let fft_scratch_size = inner_fft.complex_scratch_length();
        Ok(Dst7Fft {
            fft_executor: inner_fft,
            fft_scratch_size,
            execution_length: len,
        })
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dst7Fft<T>
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

        let n = self.execution_length;
        let fft_size = 2 * n + 1;
        let complex_len = fft_size / 2 + 1;

        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(fft_size);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for chunk in data.chunks_exact_mut(n) {
            scratch_real[0] = T::zero();

            // Map odd-indexed sequence elements
            let odd_src = chunk.iter().skip(1).step_by(2);

            let (front, back) = scratch_real[..fft_size].split_at_mut(fft_size / 2);
            let front_iter = front.iter_mut().skip(1); // indices 1, 2, 3, ...
            let back_iter = back.iter_mut().rev(); // indices fft_size-1, fft_size-2, ...

            odd_src
                .zip(front_iter)
                .zip(back_iter)
                .for_each(|((&s, lo), hi)| {
                    *lo = s.neg();
                    *hi = s;
                });

            // Map even-indexed sequence elements
            let even_src = chunk.iter().step_by(2);

            let (front, back) = scratch_real.split_at_mut(fft_size - n);

            even_src
                .zip(front[..=n].iter_mut().rev())
                .zip(back[..n].iter_mut())
                .for_each(|((&s, lo), hi)| {
                    *lo = s.neg();
                    *hi = s;
                });

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            let half = n.div_ceil(2); // number of iterations where idx <= n

            for (k, dst) in chunk[..half].iter_mut().enumerate() {
                let idx = 2 * k + 1;
                unsafe {
                    *dst = scratch_complex.get_unchecked(idx).im * T::HALF;
                }
            }

            for (k, dst) in chunk[half..].iter_mut().enumerate() {
                let idx = 2 * (k + half) + 1;
                unsafe {
                    *dst = scratch_complex.get_unchecked(fft_size - idx).im * -T::HALF;
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

        let n = self.execution_length;
        let fft_size = 2 * n + 1;
        let complex_len = fft_size / 2 + 1;

        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(fft_size);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for (src, dst) in input.chunks_exact(n).zip(output.chunks_exact_mut(n)) {
            scratch_real[0] = T::zero();

            // Map odd-indexed sequence elements
            let odd_src = src.iter().skip(1).step_by(2);

            let (front, back) = scratch_real[..fft_size].split_at_mut(fft_size / 2);
            let front_iter = front.iter_mut().skip(1); // indices 1, 2, 3, ...
            let back_iter = back.iter_mut().rev(); // indices fft_size-1, fft_size-2, ...

            odd_src
                .zip(front_iter)
                .zip(back_iter)
                .for_each(|((&s, lo), hi)| {
                    *lo = s.neg();
                    *hi = s;
                });

            // Map even-indexed sequence elements
            let even_src = src.iter().step_by(2);

            let (front, back) = scratch_real.split_at_mut(fft_size - n);

            even_src
                .zip(front[..=n].iter_mut().rev())
                .zip(back[..n].iter_mut())
                .for_each(|((&s, lo), hi)| {
                    *lo = s.neg();
                    *hi = s;
                });

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            let half = n.div_ceil(2); // number of iterations where idx <= n

            for (k, dst) in dst[..half].iter_mut().enumerate() {
                let idx = 2 * k + 1;
                unsafe {
                    *dst = scratch_complex.get_unchecked(idx).im * T::HALF;
                }
            }

            for (k, dst) in dst[half..].iter_mut().enumerate() {
                let idx = 2 * (k + half) + 1;
                unsafe {
                    *dst = scratch_complex.get_unchecked(fft_size - idx).im * -T::HALF;
                }
            }
        }

        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        let fft_size = 2 * self.execution_length + 1;
        let complex_len = fft_size / 2 + 1;
        fft_size + self.fft_scratch_size * 2 + complex_len * 2
    }
}

#[cfg(test)]
mod tests {
    use super::Dst7Fft;
    use crate::PxdctExecutor;
    use crate::tests::naive_dst7;

    #[test]
    fn test_dst7_size2() {
        let mut array = vec![1.0f64, 3.0];
        let dst = Dst7Fft::<f64>::new(array.len()).unwrap();
        let control = naive_dst7(&array);
        dst.execute(&mut array).unwrap();
        array
            .iter()
            .zip(control.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }

    #[test]
    fn test_dst7_size7() {
        let mut array = vec![1., 2., 3., 4., 5., 6., 7.];
        let dst = Dst7Fft::<f64>::new(array.len()).unwrap();
        let control = naive_dst7(&array);
        dst.execute(&mut array).unwrap();
        array
            .iter()
            .zip(control.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }

    #[test]
    fn test_dst7_size14() {
        let mut array: Vec<f64> = (1..=14).map(|x| x as f64).collect();
        let dst = Dst7Fft::<f64>::new(array.len()).unwrap();
        let control = naive_dst7(&array);
        dst.execute(&mut array).unwrap();
        array
            .iter()
            .zip(control.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }

    #[test]
    fn test_dst7_into_size7() {
        let input = vec![1., 2., 3., 4., 5., 6., 7.];
        let mut output = vec![0.0f64; 7];
        let dst = Dst7Fft::<f64>::new(input.len()).unwrap();
        let control = naive_dst7(&input);
        dst.execute_into(&input, &mut output).unwrap();
        output
            .iter()
            .zip(control.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }
}
