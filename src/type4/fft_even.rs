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

use crate::mla::c_mul_fast;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, force_cast_real_scratch_to_complex, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor};

pub struct Dct4FftEven<T> {
    fft_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    fft_scratch_size: usize,
    execution_length: usize,
    /// Unified twiddle layout: [twiddles_a (N), twiddles_b (N), twiddles_p (N)]
    twiddles: Vec<Complex<T>>,
}

impl<T: DctSample> Dct4FftEven<T>
where
    f64: AsPrimitive<T>,
{
    /// Creates a new DCT4 context that will process signals of length `inner_fft.len()`. `inner_fft.len()` must be even.
    pub fn new(fft_executor: Arc<dyn FftExecutor<T> + Send + Sync>) -> Self {
        assert_eq!(
            fft_executor.direction(),
            FftDirection::Forward,
            "Dct4FftEven requires a forward FFT, but an inverse FFT was provided"
        );

        let len = fft_executor.length();
        let fft_scratch_size = fft_executor.scratch_length();

        assert!(
            len.is_multiple_of(2),
            "Dct4FftEven size must be even. Got {}",
            len
        );

        let mut twiddles = Vec::with_capacity(len * 3);

        // twiddles_a[m] = e^{-iπm/(2N)}
        for m in 0..len {
            twiddles.push(compute_twiddle(m, 4 * len));
        }

        // twiddles_b[m] = e^{-3iπm/(2N)} = e^{-2πi·3m/(4N)}
        for m in 0..len {
            twiddles.push(compute_twiddle(3 * m, 4 * len));
        }

        // twiddles_p[p] = e^{-iπ(p+0.5)/(2N)} = e^{-2πi·(2p+1)/(8N)}
        for p in 0..len {
            twiddles.push(compute_twiddle(2 * p + 1, 8 * len));
        }

        Self {
            execution_length: len,
            fft_executor,
            fft_scratch_size,
            twiddles,
        }
    }

    #[inline]
    fn twiddles_a(&self) -> &[Complex<T>] {
        &self.twiddles[..self.execution_length]
    }

    #[inline]
    fn twiddles_b(&self) -> &[Complex<T>] {
        &self.twiddles[self.execution_length..self.execution_length * 2]
    }

    #[inline]
    fn twiddles_p(&self) -> &[Complex<T>] {
        &self.twiddles[self.execution_length * 2..self.execution_length * 3]
    }

    fn prepare_inputs(&self, src: &[T], a: &mut [Complex<T>], b: &mut [Complex<T>]) {
        a.iter_mut()
            .zip(b.iter_mut())
            .zip(src.iter())
            .zip(src.iter().rev())
            .zip(self.twiddles_a().iter())
            .zip(self.twiddles_b().iter())
            .for_each(|(((((a_out, b_out), &xm), &x_rev), &tw_a), &tw_b)| {
                *a_out = c_mul_fast(Complex { re: xm, im: x_rev }, tw_a);
                *b_out = c_mul_fast(Complex { re: xm, im: -x_rev }, tw_b);
            });
    }

    fn extract_output(&self, a: &[Complex<T>], b: &[Complex<T>], dst: &mut [T]) {
        dst.iter_mut()
            .zip(self.twiddles_p().iter())
            .enumerate()
            .for_each(|(p, (out, &tw_p))| {
                let zp = if p % 2 == 0 {
                    unsafe { *a.get_unchecked(p / 2) }
                } else {
                    unsafe { *b.get_unchecked(p / 2) }
                };
                *out = (zp * tw_p).re * T::HALF;
            });
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4FftEven<T>
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
        let (ab_slice, c1) = full_scratch.split_at_mut(self.execution_length * 4);

        let ab = force_cast_real_scratch_to_complex(ab_slice, self.execution_length * 2);
        let fft_scratch = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for chunk in data.chunks_exact_mut(self.execution_length) {
            // Prepare inputs for two N-point complex FFTs corresponding to even/odd output sequences
            let (a, b) = ab.split_at_mut(self.execution_length);
            self.prepare_inputs(chunk, a, b);

            // Execute the two underlying N-point FFTs
            self.fft_executor
                .execute_with_scratch(ab, fft_scratch)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            let (a, b) = ab.split_at_mut(self.execution_length);
            self.extract_output(a, b, chunk);
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
        let (ab_slice, c1) = full_scratch.split_at_mut(self.execution_length * 4);

        let ab = force_cast_real_scratch_to_complex(ab_slice, self.execution_length * 2);
        let fft_scratch = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            let (a, b) = ab.split_at_mut(self.execution_length);
            self.prepare_inputs(src, a, b);

            // Execute the two underlying N-point FFTs
            self.fft_executor
                .execute_with_scratch(ab, fft_scratch)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            let (a, b) = ab.split_at_mut(self.execution_length);
            self.extract_output(a, b, dst);
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        // Needs A, B length arrays dynamically resolved against real allocations + scratch mappings
        self.execution_length * 4 + self.fft_scratch_size * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::naive_dct4;
    use rand::RngExt;
    use zaft::Zaft;

    #[test]
    fn test_even_dct4_size4() {
        let mut input = vec![0f64; 4];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct4(&input);
        // use execute_into properly
        let mut out = input.clone();
        let bf = Dct4FftEven::new(Zaft::make_forward_fft_f64(4).unwrap());
        bf.execute(&mut out).unwrap();
        out.iter()
            .zip(reference.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }

    #[test]
    fn test_even_dct4_size8() {
        let mut input = vec![0f64; 8];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct4(&input);
        let mut out = vec![0f64; 8];
        let bf = Dct4FftEven::new(Zaft::make_forward_fft_f64(8).unwrap());
        bf.execute_into(&input, &mut out).unwrap();
        out.iter()
            .zip(reference.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }

    #[test]
    fn test_even_dct4_size14() {
        let input: Vec<f64> = (1..=14).map(|x| x as f64).collect();
        let reference = naive_dct4(&input);
        let mut out = vec![0f64; 14];
        let bf = Dct4FftEven::new(Zaft::make_forward_fft_f64(14).unwrap());
        bf.execute_into(&input, &mut out).unwrap();
        out.iter()
            .zip(reference.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }

    #[test]
    fn test_even_dct4_all_even_sizes() {
        for n in (2..300usize).step_by(2) {
            let input: Vec<f64> = (1..=n).map(|x| x as f64).collect();
            let reference = naive_dct4(&input);
            let mut out = vec![0f64; n];
            let bf = Dct4FftEven::new(Zaft::make_forward_fft_f64(n).unwrap());
            bf.execute_into(&input, &mut out).unwrap();
            out.iter()
                .zip(reference.iter())
                .enumerate()
                .for_each(|(i, (&x, &c))| {
                    assert!(
                        (x - c).abs() < 1e-6,
                        "size {n}, index {i}: got {x}, expected {c}"
                    );
                });
        }
    }
}
