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

/// Optimized MDCT via an N/2-point complex FFT.
use crate::mla::fmla;
use crate::util::{DctSample, force_cast_real_scratch_to_complex, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor};

/// Forward MDCT planner.
///
/// `n` is the number of output bins (the MDCT spectrum size). The input block
/// for each transform is `2 * n` real samples.
pub(crate) struct MdctFft<T> {
    /// Length-N/2 complex forward FFT.
    fft_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    /// MDCT bin count (output length); input length is 2 * n.
    n: usize,
    twiddles: Vec<Complex<T>>,
    fft_scratch_size: usize,
}

impl<T: DctSample> MdctFft<T>
where
    f64: AsPrimitive<T>,
{
    /// Create an MDCT plan for `n` output bins (input block size `2 * n`).
    pub(crate) fn new(n: usize) -> Result<MdctFft<T>, PxdctError> {
        if n < 4 {
            return Err(PxdctError::InvalidSizeMultiplier(n, 4));
        }
        if !n.is_multiple_of(2) {
            return Err(PxdctError::InvalidSizeMultiplier(n, 2));
        }

        let m = n / 2; // complex FFT length
        let inner_fft = T::make_fft(m, FftDirection::Forward)?;
        let fft_scratch_size = inner_fft.scratch_length();

        // Precompute the N/2 twiddles.
        let mut twiddles = Vec::with_capacity(m);
        for k in 0..m {
            let angle: T = (-((8 * k + 1) as f64 / (8 * n) as f64)).as_();
            let (v_sin, v_cos) = angle.sincos_pi();
            twiddles.push(Complex {
                re: v_cos,
                im: v_sin,
            });
        }

        Ok(MdctFft {
            fft_executor: inner_fft,
            n,
            twiddles,
            fft_scratch_size,
        })
    }

    /// Pre-twiddle / fold: read 2N real input samples and produce N/2 complex
    /// pre-rotated samples into `fft_buf`.
    fn pre_twiddle(&self, input: &[T], fft_buf: &mut [Complex<T>]) {
        let n = self.n;
        let n2 = n / 2;
        let n32 = 3 * n / 2;
        let n52 = 5 * n / 2;

        fft_buf
            .iter_mut()
            .zip(self.twiddles.iter())
            .enumerate()
            .for_each(|(k, (buf, twiddle))| {
                let nn = 2 * k;
                let (r0, i0) = unsafe {
                    if nn < n2 {
                        (
                            *input.get_unchecked(n32 - 1 - nn) + *input.get_unchecked(n32 + nn),
                            *input.get_unchecked(n2 + nn) - *input.get_unchecked(n2 - 1 - nn),
                        )
                    } else {
                        (
                            *input.get_unchecked(n32 - 1 - nn) - *input.get_unchecked(nn - n2),
                            *input.get_unchecked(n2 + nn) + *input.get_unchecked(n52 - 1 - nn),
                        )
                    }
                };

                let c = twiddle.re;
                let neg_s = twiddle.im;
                *buf = Complex {
                    re: fmla(r0, c, -i0 * neg_s),
                    im: fmla(r0, neg_s, i0 * c),
                };
            });
    }

    /// Post-twiddle / unpack: read N/2 complex FFT output and produce N real
    /// MDCT coefficients into `output`.
    fn post_twiddle(&self, fft_out: &[Complex<T>], output: &mut [T]) {
        let n = self.n;

        fft_out
            .iter()
            .zip(self.twiddles.iter())
            .enumerate()
            .for_each(|(k, (z, twiddle))| {
                let c = twiddle.re;
                let neg_s = twiddle.im;
                let w_re = fmla(z.re, c, -z.im * neg_s);
                let w_im = fmla(z.re, neg_s, z.im * c);

                let nn = 2 * k;
                unsafe {
                    *output.get_unchecked_mut(nn) = w_re.neg();
                    *output.get_unchecked_mut(n - 1 - nn) = w_im;
                }
            });
    }
}

impl<T: DctSample> PxdctExecutor<T> for MdctFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, _data: &mut [T]) -> Result<(), PxdctError> {
        // MDCT cannot be in-place: input and output have different lengths
        // (2N -> N). Reject in-place calls.
        Err(PxdctError::InvalidSizeMultiplier(self.n, 2 * self.n))
    }

    fn execute_with_scratch(&self, _data: &mut [T], _scratch: &mut [T]) -> Result<(), PxdctError> {
        Err(PxdctError::InvalidSizeMultiplier(self.n, 2 * self.n))
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
        // Each MDCT block: 2N real input -> N real output.
        let in_block = 2 * self.n;
        let out_block = self.n;
        if !input.len().is_multiple_of(in_block) {
            return Err(PxdctError::InvalidSizeMultiplier(input.len(), in_block));
        }
        if !output.len().is_multiple_of(out_block) {
            return Err(PxdctError::InvalidSizeMultiplier(output.len(), out_block));
        }
        if input.len() / in_block != output.len() / out_block {
            return Err(PxdctError::InvalidSizeMultiplier(
                output.len(),
                input.len() / 2,
            ));
        }

        let m = self.n / 2;

        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (fft_buf_real, rest) = full_scratch.split_at_mut(m * 2);
        let fft_buf = force_cast_real_scratch_to_complex(fft_buf_real, m);
        let fft_scratch = force_cast_real_scratch_to_complex(rest, self.fft_scratch_size);

        for (src, dst) in input
            .chunks_exact(in_block)
            .zip(output.chunks_exact_mut(out_block))
        {
            self.pre_twiddle(src, fft_buf);

            self.fft_executor
                .execute_with_scratch(fft_buf, fft_scratch)
                .map_err(|e| PxdctError::FftError(e.to_string()))?;

            self.post_twiddle(fft_buf, dst);
        }

        Ok(())
    }

    fn length(&self) -> usize {
        self.n
    }

    fn scratch_size(&self) -> usize {
        // FFT buffer: m complex = 2m reals. FFT internal scratch: complex scratch
        // counted as 2 reals per element.
        self.n + self.fft_scratch_size * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;

    /// Naive reference MDCT for validation.
    fn naive_mdct(input: &[f64]) -> Vec<f64> {
        let len = input.len();
        assert!(len.is_multiple_of(2));
        let n = len / 2;
        let mut out = vec![0.0; n];
        for k in 0..n {
            let mut s = 0.0;
            for nn in 0..len {
                let arg = std::f64::consts::PI / n as f64
                    * (nn as f64 + 0.5 + n as f64 / 2.0)
                    * (k as f64 + 0.5);
                s += input[nn] * arg.cos();
            }
            out[k] = s;
        }
        out
    }

    fn run_case(n: usize, tol: f64) {
        let input: Vec<f64> = (1..=2 * n).map(|x| x as f64).collect();
        let mut output = vec![0.0f64; n];
        let mdct = MdctFft::<f64>::new(n).unwrap();
        mdct.execute_into(&input, &mut output).unwrap();
        let control = naive_mdct(&input);
        for (i, (&x, &c)) in output.iter().zip(control.iter()).enumerate() {
            let abs_tol = tol * (1.0 + c.abs());
            assert!(
                (x - c).abs() < abs_tol,
                "N={n}, k={i}: got {x}, expected {c}, diff {}",
                (x - c).abs()
            );
        }
    }

    #[test]
    fn test_mdct_size4() {
        run_case(4, 1e-10);
    }

    #[test]
    fn test_mdct_size8() {
        run_case(8, 1e-10);
    }

    #[test]
    fn test_mdct_size16() {
        run_case(16, 1e-10);
    }

    #[test]
    fn test_mdct_size_odd_fft_len() {
        // N = 6 → FFT length 3 (odd); N = 10 → FFT length 5; etc.
        for n in [6, 10, 14, 22, 30, 50] {
            run_case(n, 1e-10);
        }
    }

    #[test]
    fn test_mdct_power_of_two_sizes() {
        // Typical audio codec sizes.
        for shift in 2..=10 {
            // N from 4 to 1024
            run_case(1 << shift, 1e-9);
        }
    }

    #[test]
    fn test_mdct_multiple_blocks() {
        // 3 consecutive blocks of size N=8.
        let n = 8;
        let input: Vec<f64> = (1..=6 * n).map(|x| x as f64).collect();
        let mut output = vec![0.0f64; 3 * n];
        let mdct = MdctFft::<f64>::new(n).unwrap();
        mdct.execute_into(&input, &mut output).unwrap();

        // Each block independently should match naive.
        for block in 0..3 {
            let block_in = &input[block * 2 * n..(block + 1) * 2 * n];
            let block_out = &output[block * n..(block + 1) * n];
            let control = naive_mdct(block_in);
            for (i, (&x, &c)) in block_out.iter().zip(control.iter()).enumerate() {
                assert!(
                    (x - c).abs() < 1e-9,
                    "block {block}, k={i}: got {x}, expected {c}"
                );
            }
        }
    }

    #[test]
    fn test_mdct_rejects_odd_n() {
        assert!(MdctFft::<f64>::new(5).is_err());
        assert!(MdctFft::<f64>::new(7).is_err());
    }

    #[test]
    fn test_mdct_rejects_too_small() {
        assert!(MdctFft::<f64>::new(2).is_err());
        // N=3 is rejected as odd before the size check, but N=2 is even and too small.
    }
}
