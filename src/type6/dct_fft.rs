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

pub(crate) struct Dct6Fft<T> {
    fft_executor: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    execution_length: usize,
    fft_scratch_size: usize,
}

impl<T: DctSample> Dct6Fft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(len: usize) -> Result<Dct6Fft<T>, PxdctError> {
        if len == 0 {
            return Err(PxdctError::InvalidSizeMultiplier(0, 0));
        }
        // N == 1 is handled trivially; still allocate a small FFT so the
        // executor field stays well-defined.
        let fft_size = if len == 1 { 2 } else { 4 * len - 2 };
        let inner_fft = T::make_fft_r2c(fft_size)?;
        let fft_scratch_size = inner_fft.complex_scratch_length();
        Ok(Dct6Fft {
            fft_executor: inner_fft,
            fft_scratch_size,
            execution_length: len,
        })
    }

    /// Build the length-L = 4N-2 real, even-symmetric input for the DFT.
    ///
    /// Layout:
    ///     y_{2n+1}      = x_n          for n = 0 .. N-1   (odd indices 1, 3, ..., 2N-1)
    ///     y_{L - (2n+1)} = x_n         for n = 0 .. N-2   (mirror; n = N-1 is self-mirrored)
    ///     all other indices            = 0
    ///
    fn build_scratch(scratch_real: &mut [T], chunk: &[T]) {
        let n = chunk.len();
        debug_assert!(n >= 2);
        let l = 4 * n - 2; // L = 2M = 4N - 2
        scratch_real[..l].fill(T::zero());

        // x_n at index 2n+1, mirrored to L - (2n+1) = 4N - 3 - 2n.
        // For n = N-1: 2n+1 = 2N-1 = M, and L - M = M, so the slot is self-mirrored;
        // writing only the forward slot is correct (matches the once-only Nyquist term).
        for (i, &v) in chunk.iter().enumerate() {
            let pos = 2 * i + 1;
            unsafe {
                *scratch_real.get_unchecked_mut(pos) = v;
            }
            if i + 1 != n {
                // Equivalent to `pos != l - pos`, i.e. not the self-mirrored index.
                unsafe {
                    *scratch_real.get_unchecked_mut(l - pos) = v;
                }
            }
        }
    }

    fn trivial_inplace(data: &mut [T]) {
        for v in data.iter_mut() {
            *v *= T::HALF;
        }
    }

    fn trivial_into(input: &[T], output: &mut [T]) {
        for (s, d) in input.iter().zip(output.iter_mut()) {
            *d = *s * T::HALF;
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct6Fft<T>
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

        // N == 1 closed form: X_0 = 0.5 * x_0.
        if n == 1 {
            Self::trivial_inplace(data);
            return Ok(());
        }

        let fft_size = 4 * n - 2;
        let complex_len = fft_size / 2 + 1; // = 2N - 1 + 1 = 2N

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

            // X_k = Re(Y_k) / 2 for k = 0..N-1. The R2C output has complex_len = 2N
            // bins, so we just read the first N. No folding required.
            for (d, c) in chunk.iter_mut().zip(scratch_complex.iter()) {
                *d = c.re * T::HALF;
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

        if n == 1 {
            Self::trivial_into(input, output);
            return Ok(());
        }

        let fft_size = 4 * n - 2;
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

            for (d, c) in dst.iter_mut().zip(scratch_complex.iter()) {
                *d = c.re * T::HALF;
            }
        }

        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        if self.execution_length == 1 {
            return 0;
        }
        let fft_size = 4 * self.execution_length - 2;
        let complex_len = fft_size / 2 + 1;
        fft_size + self.fft_scratch_size * 2 + complex_len * 2
    }
}

#[cfg(test)]
mod tests {
    use super::Dct6Fft;
    use crate::PxdctExecutor;
    use crate::tests::naive_dct6;

    fn run_case(n: usize, tol: f64) {
        let array: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        let mut working = array.clone();
        let dct = Dct6Fft::<f64>::new(n).unwrap();
        let control = naive_dct6(&array);
        dct.execute(&mut working).unwrap();
        for (i, (&x, &c)) in working.iter().zip(control.iter()).enumerate() {
            assert!(
                (x - c).abs() < tol,
                "size {n}, index {i}: got {x}, expected {c}"
            );
        }
    }

    #[test]
    fn test_dct6_size1() {
        run_case(1, 1e-12);
    }

    #[test]
    fn test_dct6_size2() {
        run_case(2, 1e-9);
    }

    #[test]
    fn test_dct6_size7() {
        run_case(7, 1e-9);
    }

    #[test]
    fn test_dct6_size14() {
        run_case(14, 1e-9);
    }

    #[test]
    fn test_dct6_into_size7() {
        let input: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let mut output = vec![0.0f64; 7];
        let dct = Dct6Fft::<f64>::new(input.len()).unwrap();
        let control = naive_dct6(&input);
        dct.execute_into(&input, &mut output).unwrap();
        for (i, (&x, &c)) in output.iter().zip(control.iter()).enumerate() {
            assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
        }
    }

    #[test]
    fn test_dct6_all_sizes() {
        for n in 1..150 {
            run_case(n, 1e-7);
        }
    }
}
