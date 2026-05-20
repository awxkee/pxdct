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

pub(crate) struct Dst1Fft<T> {
    fft_executor: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    execution_length: usize,
    fft_scratch_size: usize,
}

impl<T: DctSample> Dst1Fft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(len: usize) -> Result<Dst1Fft<T>, PxdctError> {
        // DST-I of size M needs FFT of size 2*(M+1)
        let inner_fft = T::make_fft_r2c(2 * (len + 1))?;
        let inner_fft_scratch = inner_fft.complex_scratch_length();
        Ok(Dst1Fft {
            fft_executor: inner_fft,
            fft_scratch_size: inner_fft_scratch,
            execution_length: len,
        })
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dst1Fft<T>
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

        let m = self.execution_length;
        let fft_size = 2 * (m + 1);
        let complex_len = fft_size / 2 + 1;

        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(fft_size);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for chunk in data.chunks_exact_mut(m) {
            // Antisymmetric extension:
            // [0, x_0, ..., x_{M-1}, 0, -x_{M-1}, ..., -x_0]
            scratch_real[0] = T::zero();
            scratch_real[1..=m].copy_from_slice(chunk);
            scratch_real[m + 1] = T::zero();
            scratch_real[m + 2..]
                .iter_mut()
                .zip(chunk.iter().rev())
                .for_each(|(dst, s)| *dst = s.neg());

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            chunk
                .iter_mut()
                .zip(scratch_complex[1..].iter())
                .for_each(|(d, s)| *d = s.im.neg() * T::HALF);
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

        let m = self.execution_length;
        let fft_size = 2 * (m + 1);
        let complex_len = fft_size / 2 + 1;

        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(fft_size);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for (src, dst) in input.chunks_exact(m).zip(output.chunks_exact_mut(m)) {
            scratch_real[0] = T::zero();
            scratch_real[1..=m].copy_from_slice(src);
            scratch_real[m + 1] = T::zero();
            scratch_real[m + 2..]
                .iter_mut()
                .zip(src.iter().rev())
                .for_each(|(dst, s)| *dst = s.neg());

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            dst.iter_mut()
                .zip(scratch_complex[1..].iter())
                .for_each(|(d, s)| *d = s.im.neg() * T::HALF);
        }

        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        let fft_size = 2 * (self.execution_length + 1);
        let complex_len = fft_size / 2 + 1;
        // scratch_real + fft_scratch (as real) + complex output (as real)
        fft_size + self.fft_scratch_size * 2 + complex_len * 2
    }
}

#[cfg(test)]
mod tests {
    use super::Dst1Fft;
    use crate::PxdctExecutor;
    use crate::tests::naive_dst1;

    #[test]
    fn test_dst1_size2() {
        let mut array = vec![1.0f64, 3.0];
        let dst = Dst1Fft::<f64>::new(array.len()).unwrap();
        let control = naive_dst1(&array);
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
    fn test_dst1_size7() {
        let mut array = vec![1., 2., 3., 4., 5., 6., 7.];
        let dst = Dst1Fft::<f64>::new(array.len()).unwrap();
        let control = naive_dst1(&array);
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
    fn test_dst1_size14() {
        let mut array: Vec<f64> = (1..=14).map(|x| x as f64).collect();
        let dst = Dst1Fft::<f64>::new(array.len()).unwrap();
        let control = naive_dst1(&array);
        dst.execute(&mut array).unwrap();
        array
            .iter()
            .zip(control.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }
}
