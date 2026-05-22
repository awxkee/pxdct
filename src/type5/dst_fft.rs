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

// DST-V via real FFT.

use crate::util::{DctSample, force_cast_real_scratch_to_complex, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::R2CFftExecutor;

pub(crate) struct Dst5Fft<T> {
    fft_executor: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    execution_length: usize,
    fft_scratch_size: usize,
}

impl<T: DctSample> Dst5Fft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(len: usize) -> Result<Dst5Fft<T>, PxdctError> {
        if len == 0 {
            return Err(PxdctError::InvalidSizeMultiplier(0, 0));
        }
        // DST-V of length N maps to a real DFT of length M = 2N + 1.
        // N == 1 gives M = 3 — no degenerate case to special-case.
        let fft_size = 2 * len + 1;
        let inner_fft = T::make_fft_r2c(fft_size)?;
        let fft_scratch_size = inner_fft.complex_scratch_length();
        Ok(Dst5Fft {
            fft_executor: inner_fft,
            fft_scratch_size,
            execution_length: len,
        })
    }

    /// Build the length-M = 2N+1 real, antisymmetric input for the DFT.
    fn build_scratch(scratch_real: &mut [T], chunk: &[T]) {
        let n = chunk.len();
        let m = 2 * n + 1;

        let (first, rest) = scratch_real[..m].split_first_mut().unwrap();
        *first = T::zero();

        let (left, right) = rest.split_at_mut(n);
        // left  covers indices 1..=n
        // right covers indices n+1..=2n, length n

        chunk
            .iter()
            .zip(left.iter_mut())
            .zip(right.iter_mut().rev())
            .for_each(|((&v, l), r)| {
                *l = v;
                *r = v.neg();
            });
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dst5Fft<T>
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
        let complex_len = fft_size / 2 + 1; // = N + 1

        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(fft_size);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for chunk in data.chunks_exact_mut(n) {
            Self::build_scratch(scratch_real, chunk);

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            // X_k = -Im(Y_{k+1}) / 2 for k = 0..N-1.
            // Bins 1..N are all in the R2C output (complex_len = N+1), no folding.
            chunk
                .iter_mut()
                .zip(scratch_complex.iter().skip(1))
                .for_each(|(d, src)| {
                    *d = src.im.neg() * T::HALF;
                });
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
            Self::build_scratch(scratch_real, src);

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            dst.iter_mut()
                .zip(scratch_complex.iter().skip(1))
                .for_each(|(d, src)| {
                    *d = src.im.neg() * T::HALF;
                });
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
    use super::Dst5Fft;
    use crate::PxdctExecutor;
    use crate::tests::naive_dst5;

    fn run_case(n: usize, tol: f64) {
        let array: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        let mut working = array.clone();
        let dst = Dst5Fft::<f64>::new(n).unwrap();
        let control = naive_dst5(&array);
        dst.execute(&mut working).unwrap();
        for (i, (&x, &c)) in working.iter().zip(control.iter()).enumerate() {
            assert!(
                (x - c).abs() < tol,
                "size {n}, index {i}: got {x}, expected {c}"
            );
        }
    }

    #[test]
    fn test_dst5_size1() {
        run_case(1, 1e-12);
    }

    #[test]
    fn test_dst5_size2() {
        run_case(2, 1e-9);
    }

    #[test]
    fn test_dst5_size7() {
        run_case(7, 1e-9);
    }

    #[test]
    fn test_dst5_size14() {
        run_case(14, 1e-9);
    }

    #[test]
    fn test_dst5_into_size7() {
        let input: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let mut output = vec![0.0f64; 7];
        let dst = Dst5Fft::<f64>::new(input.len()).unwrap();
        let control = naive_dst5(&input);
        dst.execute_into(&input, &mut output).unwrap();
        for (i, (&x, &c)) in output.iter().zip(control.iter()).enumerate() {
            assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
        }
    }

    #[test]
    fn test_dst5_all_sizes() {
        for n in 1..150 {
            run_case(n, 1e-7);
        }
    }
}
