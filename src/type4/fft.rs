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
use crate::util::{DctSample, force_cast_real_scratch_to_complex, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::R2CFftExecutor;

pub(crate) struct Dct4Fft<T> {
    fft_executor: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    fft_scratch_size: usize,
    execution_length: usize,
}

impl<T: DctSample> Dct4Fft<T> {
    /// Creates a new DCT4 context that will process signals of length `inner_fft.len()`.
    /// `inner_fft.len()` must be odd.
    pub fn new(fft_executor: Arc<dyn R2CFftExecutor<T> + Send + Sync>) -> Self {
        let len = fft_executor.real_length();
        let fft_scratch_size = fft_executor.complex_scratch_length();

        assert!(
            !len.is_multiple_of(2),
            "Dct4Odd size must be odd. Got {}",
            len
        );

        Self {
            execution_length: len,
            fft_executor,
            fft_scratch_size,
        }
    }

    /// Fills `real_buf` (length N) with the permuted/signed real sequence,
    /// runs the R2C FFT into `spectrum` (length N/2+1), then reconstructs
    /// the full N-point complex spectrum in `scratch` via Hermitian symmetry.
    fn run_r2c(
        &self,
        src: &[T],
        real_buf: &mut [T],          // length N
        spectrum: &mut [Complex<T>], // length N/2 + 1  (R2C output)
        full: &mut [Complex<T>],     // length N        (reconstructed)
        fft_scratch: &mut [Complex<T>],
    ) -> Result<(), PxdctError> {
        let len = self.execution_length;
        let half_len = len / 2;

        // ── same five-loop permutation as before, but into real_buf ──────────
        let mut input_index = half_len;
        let mut fft_index = 0usize;

        while input_index < len {
            unsafe { *real_buf.get_unchecked_mut(fft_index) = *src.get_unchecked(input_index) };
            input_index += 4;
            fft_index += 1;
        }
        input_index -= len;
        while input_index < len {
            unsafe {
                *real_buf.get_unchecked_mut(fft_index) = -*src.get_unchecked(len - input_index - 1)
            };
            input_index += 4;
            fft_index += 1;
        }
        input_index -= len;
        while input_index < len {
            unsafe { *real_buf.get_unchecked_mut(fft_index) = -*src.get_unchecked(input_index) };
            input_index += 4;
            fft_index += 1;
        }
        input_index -= len;
        while input_index < len {
            unsafe {
                *real_buf.get_unchecked_mut(fft_index) = *src.get_unchecked(len - input_index - 1)
            };
            input_index += 4;
            fft_index += 1;
        }
        input_index -= len;
        while fft_index < len {
            unsafe { *real_buf.get_unchecked_mut(fft_index) = *src.get_unchecked(input_index) };
            input_index += 4;
            fft_index += 1;
        }

        // ── R2C FFT: real_buf → spectrum[0..=N/2] ───────────────────────────
        self.fft_executor
            .execute_with_scratch(real_buf, spectrum, fft_scratch)
            .map_err(|x| PxdctError::FftError(x.to_string()))?;

        // ── Reconstruct full N-point spectrum via Hermitian symmetry ─────────
        // spectrum[0] and (if N even) spectrum[N/2] are purely real.
        // For k = 1 .. N/2:   full[k]   = spectrum[k]
        //                      full[N-k] = conj(spectrum[k])
        full[0] = spectrum[0];
        for k in 1..=len / 2 {
            let s = unsafe { *spectrum.get_unchecked(k) };
            unsafe {
                *full.get_unchecked_mut(k) = s;
                *full.get_unchecked_mut(len - k) = s.conj();
            }
        }

        Ok(())
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4Fft<T>
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
        let len = self.execution_length;
        let half_len = len / 2;
        let r2c_out_len = len / 2 + 1; // number of unique bins from R2C

        // Layout (all T):
        //   [0 .. len)          → real_buf  (R2C input)
        //   [len .. len + 2*r2c_out_len) → spectrum (Complex<T>, r2c_out_len elems)
        //   [len + 2*r2c_out_len .. len + 2*r2c_out_len + 2*len) → full spectrum (Complex<T>, len elems)
        //   remainder           → fft_scratch
        let (real_buf_slice, rest) = full_scratch.split_at_mut(len);
        let (spectrum_slice, rest) = rest.split_at_mut(r2c_out_len * 2);
        let (full_slice, fft_scratch_slice) = rest.split_at_mut(len * 2);

        let spectrum = force_cast_real_scratch_to_complex(spectrum_slice, r2c_out_len);
        let full = force_cast_real_scratch_to_complex(full_slice, len);
        let fft_scratch =
            force_cast_real_scratch_to_complex(fft_scratch_slice, self.fft_scratch_size);

        for chunk in data.chunks_exact_mut(self.execution_length) {
            self.run_r2c(chunk, real_buf_slice, spectrum, full, fft_scratch)?;

            // ── post-process: identical to original, but reading from `full` ─
            let result_scale = T::SQRT_2 * T::HALF;
            let second_half_sign = if len % 4 == 1 { T::one() } else { -T::one() };
            let mut output_sign = T::one();

            let (left, right) = chunk.split_at_mut(half_len);

            for ((l_dst, r_dst), s) in left
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(right.as_rchunks_mut::<2>().1.iter_mut().rev())
                .zip(full.as_chunks::<4>().0.iter())
            {
                let fft_result = s[1] * (output_sign * result_scale);
                let next_result = s[3] * (output_sign * result_scale);

                l_dst[0] = fft_result.re + fft_result.im;
                l_dst[1] = -next_result.re + next_result.im;
                r_dst[0] = (next_result.re + next_result.im) * second_half_sign;
                r_dst[1] = (fft_result.re - fft_result.im) * second_half_sign;

                output_sign = output_sign.neg();
            }

            if len % 4 == 1 {
                chunk[half_len] = full[0].re * output_sign * result_scale;
            } else {
                let fft_result = full[len - 2] * (output_sign * result_scale);
                chunk[half_len - 1] = fft_result.re + fft_result.im;
                chunk[half_len + 1] = -fft_result.re + fft_result.im;
                chunk[half_len] = -full[0].re * output_sign * result_scale;
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
        let len = self.execution_length;
        let half_len = len / 2;
        let r2c_out_len = len / 2 + 1;

        let (real_buf_slice, rest) = full_scratch.split_at_mut(len);
        let (spectrum_slice, rest) = rest.split_at_mut(r2c_out_len * 2);
        let (full_slice, fft_scratch_slice) = rest.split_at_mut(len * 2);

        let spectrum = force_cast_real_scratch_to_complex(spectrum_slice, r2c_out_len);
        let full = force_cast_real_scratch_to_complex(full_slice, len);
        let fft_scratch =
            force_cast_real_scratch_to_complex(fft_scratch_slice, self.fft_scratch_size);

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.run_r2c(src, real_buf_slice, spectrum, full, fft_scratch)?;

            let result_scale = T::SQRT_2 * T::HALF;
            let second_half_sign = if len % 4 == 1 { T::one() } else { -T::one() };
            let mut output_sign = T::one();

            let (left, right) = dst.split_at_mut(half_len);

            for ((l_dst, r_dst), s) in left
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(right.as_rchunks_mut::<2>().1.iter_mut().rev())
                .zip(full.as_chunks::<4>().0.iter())
            {
                let fft_result = s[1] * (output_sign * result_scale);
                let next_result = s[3] * (output_sign * result_scale);

                l_dst[0] = fft_result.re + fft_result.im;
                l_dst[1] = -next_result.re + next_result.im;
                r_dst[0] = (next_result.re + next_result.im) * second_half_sign;
                r_dst[1] = (fft_result.re - fft_result.im) * second_half_sign;

                output_sign = output_sign.neg();
            }

            if len % 4 == 1 {
                dst[half_len] = full[0].re * output_sign * result_scale;
            } else {
                let fft_result = full[len - 2] * (output_sign * result_scale);
                dst[half_len - 1] = fft_result.re + fft_result.im;
                dst[half_len + 1] = -fft_result.re + fft_result.im;
                dst[half_len] = -full[0].re * output_sign * result_scale;
            }
        }

        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        let len = self.execution_length;
        let r2c_out_len = len / 2 + 1;
        // real_buf(N) + spectrum(r2c_out_len * 2 reals) + full(N * 2 reals) + fft_scratch
        len + r2c_out_len * 2 + len * 2 + self.fft_scratch_size * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::naive_dct4;
    use rand::RngExt;
    use zaft::Zaft;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 15];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4Fft::new(Zaft::make_r2c_fft_f64(15).unwrap());
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
        let bf = Dct4Fft::new(Zaft::make_r2c_fft_f64(35).unwrap());
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
        let bf = Dct4Fft::new(Zaft::make_r2c_fft_f64(17).unwrap());
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
