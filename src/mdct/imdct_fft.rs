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

//! Optimized IMDCT via an N/2-point complex FFT.

use crate::mla::fmla;
use crate::twiddles::FftTrigonometry;
use crate::util::{DctSample, force_cast_real_scratch_to_complex, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::{FftDirection, FftExecutor};

/// Inverse MDCT planner.
///
/// `n` is the number of input bins (the MDCT spectrum size that was produced
/// by `MdctFft`). The output block for each transform is `2 * n` real samples.
pub(crate) struct ImdctFft<T> {
    /// Length-N/2 complex forward FFT (same direction as MDCT — DCT-IV is
    /// self-inverse up to scale).
    fft_executor: Arc<dyn FftExecutor<T> + Send + Sync>,
    /// MDCT bin count (input length); output length is 2 * n.
    n: usize,
    twiddles: Vec<Complex<T>>,
    fft_scratch_size: usize,
}

impl<T: DctSample> ImdctFft<T>
where
    f64: AsPrimitive<T>,
{
    /// Create an IMDCT plan for `n` input bins (output block size `2 * n`).
    pub(crate) fn new(n: usize) -> Result<ImdctFft<T>, PxdctError> {
        if n < 4 {
            return Err(PxdctError::InvalidSizeMultiplier(n, 4));
        }
        if !n.is_multiple_of(2) {
            return Err(PxdctError::InvalidSizeMultiplier(n, 2));
        }

        let m = n / 2;
        let inner_fft = T::make_fft(m, FftDirection::Forward)?;
        let fft_scratch_size = inner_fft.scratch_length();

        let alpha = 1.0 / (8.0 * n as f64);
        let omega = 1.0 / n as f64;
        let mut twiddles = Vec::with_capacity(m);
        for k in 0..m {
            let theta = omega * k as f64 + alpha;
            twiddles.push(Complex {
                re: theta.cospi().as_(),
                im: theta.sinpi().as_(),
            });
        }

        Ok(ImdctFft {
            fft_executor: inner_fft,
            n,
            twiddles,
            fft_scratch_size,
        })
    }

    /// Pre-twiddle: read N MDCT coefficients, write M = N/2 complex samples.
    fn pre_twiddle(&self, input: &[T], fft_buf: &mut [Complex<T>]) {
        let n = self.n;
        let m = n / 2;

        for (k, (output, twiddle)) in fft_buf[..m]
            .iter_mut()
            .zip(self.twiddles.iter())
            .enumerate()
        {
            let nn = 2 * k;
            let r0 = input[nn];
            let i0 = input[n - 1 - nn];
            let c = twiddle.re;
            let s = twiddle.im;
            *output = Complex {
                re: i0.neg() * s - r0 * c,
                im: i0.neg() * c + r0 * s,
            };
        }
    }

    /// Post-twiddle and fan-out: read M complex FFT bins, write 2N real samples.
    fn post_twiddle(&self, fft_out: &[Complex<T>], output: &mut [T]) {
        let n = self.n;
        let m = n / 2;
        let n2 = n / 2;
        let n32 = 3 * n / 2;
        let n52 = 5 * n / 2;

        for (k, (fft_val, twiddle)) in fft_out[..m].iter().zip(self.twiddles.iter()).enumerate() {
            let nn = 2 * k;
            let r0 = fft_val.re;
            let i0 = fft_val.im;
            let c = twiddle.re;
            let s = twiddle.im;

            // r1, i1 from reference:
            //   r1 = r0*c + i0*s
            //   i1 = r0*s - i0*c
            let r1 = fmla(r0, c, i0 * s);
            let i1 = fmla(r0, s, -i0 * c);

            if nn < n2 {
                // First-half fan-out (4 distinct slots in lower 3N/2 of output)
                unsafe {
                    *output.get_unchecked_mut(n32 - 1 - nn) = r1;
                    *output.get_unchecked_mut(n32 + nn) = r1;
                    *output.get_unchecked_mut(n2 + nn) = i1;
                    *output.get_unchecked_mut(n2 - 1 - nn) = i1.neg();
                }
            } else {
                // Second-half fan-out (wraps into both ends of the buffer)
                unsafe {
                    *output.get_unchecked_mut(n32 - 1 - nn) = r1;
                    *output.get_unchecked_mut(nn - n2) = r1.neg();
                    *output.get_unchecked_mut(n2 + nn) = i1;
                    *output.get_unchecked_mut(n52 - 1 - nn) = i1;
                }
            }
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for ImdctFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, _data: &mut [T]) -> Result<(), PxdctError> {
        // IMDCT cannot be in-place: input length is N, output is 2N.
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
        let in_block = self.n;
        let out_block = 2 * self.n;
        if !input.len().is_multiple_of(in_block) {
            return Err(PxdctError::InvalidSizeMultiplier(input.len(), in_block));
        }
        if !output.len().is_multiple_of(out_block) {
            return Err(PxdctError::InvalidSizeMultiplier(output.len(), out_block));
        }
        if input.len() / in_block != output.len() / out_block {
            return Err(PxdctError::InvalidSizeMultiplier(
                output.len(),
                input.len() * 2,
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
        self.n + self.fft_scratch_size * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;

    /// Naive reference IMDCT for validation.
    fn naive_imdct(coeffs: &[f64]) -> Vec<f64> {
        let n = coeffs.len();
        let mut out = vec![0.0; 2 * n];
        for nn in 0..2 * n {
            let mut s = 0.0;
            for k in 0..n {
                let arg = std::f64::consts::PI / n as f64
                    * (nn as f64 + 0.5 + n as f64 / 2.0)
                    * (k as f64 + 0.5);
                s += coeffs[k] * arg.cos();
            }
            out[nn] = s;
        }
        out
    }

    fn run_case(n: usize, tol: f64) {
        let input: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        let mut output = vec![0.0f64; 2 * n];
        let imdct = ImdctFft::<f64>::new(n).unwrap();
        imdct.execute_into(&input, &mut output).unwrap();
        let control = naive_imdct(&input);
        for (i, (&x, &c)) in output.iter().zip(control.iter()).enumerate() {
            let abs_tol = tol * (1.0 + c.abs());
            assert!(
                (x - c).abs() < abs_tol,
                "N={n}, n={i}: got {x}, expected {c}, diff {}",
                (x - c).abs()
            );
        }
    }

    #[test]
    fn test_imdct_size4() {
        run_case(4, 1e-10);
    }

    #[test]
    fn test_imdct_size8() {
        run_case(8, 1e-10);
    }

    #[test]
    fn test_imdct_size16() {
        run_case(16, 1e-10);
    }

    #[test]
    fn test_imdct_odd_fft_lengths() {
        // N = 6 → M = 3 (odd FFT), N = 10 → M = 5, etc.
        for n in [6, 10, 14, 22, 30, 50] {
            run_case(n, 1e-10);
        }
    }

    #[test]
    fn test_imdct_power_of_two_sizes() {
        for shift in 2..=10 {
            run_case(1 << shift, 1e-9);
        }
    }

    #[test]
    fn test_imdct_multiple_blocks() {
        let n = 8;
        let input: Vec<f64> = (1..=3 * n).map(|x| x as f64).collect();
        let mut output = vec![0.0f64; 6 * n];
        let imdct = ImdctFft::<f64>::new(n).unwrap();
        imdct.execute_into(&input, &mut output).unwrap();

        for block in 0..3 {
            let block_in = &input[block * n..(block + 1) * n];
            let block_out = &output[block * 2 * n..(block + 1) * 2 * n];
            let control = naive_imdct(block_in);
            for (i, (&x, &c)) in block_out.iter().zip(control.iter()).enumerate() {
                assert!(
                    (x - c).abs() < 1e-9,
                    "block {block}, n={i}: got {x}, expected {c}"
                );
            }
        }
    }

    /// IMDCT(MDCT(x)) should equal N * x_aliased, where x_aliased contains the
    /// signed-folded TDAC pattern. This is the formal MDCT-IMDCT identity (not
    /// "MDCT/IMDCT is an inverse pair" — it isn't on a single block).
    ///
    /// Specifically (one common form): for a 2N input x,
    ///   IMDCT(MDCT(x))[n] = N * (x[n] - x[N - 1 - n])              for 0 <= n < N/2
    ///   IMDCT(MDCT(x))[n] = N * (x[n] + x[3N - 1 - n])             for N/2 <= n < N    (with reflections)
    /// We just check that the roundtrip output's structure is consistent with
    /// the naive_imdct(naive_mdct(x)) output.
    fn naive_mdct(x: &[f64]) -> Vec<f64> {
        let len = x.len();
        let n = len / 2;
        let mut out = vec![0.0; n];
        for k in 0..n {
            let mut s = 0.0;
            for nn in 0..len {
                let arg = std::f64::consts::PI / n as f64
                    * (nn as f64 + 0.5 + n as f64 / 2.0)
                    * (k as f64 + 0.5);
                s += x[nn] * arg.cos();
            }
            out[k] = s;
        }
        out
    }

    #[test]
    fn test_imdct_matches_naive_roundtrip() {
        // Forward MDCT via naive, then inverse via fast IMDCT — output should
        // match naive IMDCT applied to the same coefficients.
        let n = 32;
        let x: Vec<f64> = (1..=2 * n).map(|v| v as f64 * 0.5).collect();
        let coeffs = naive_mdct(&x);

        let mut fast_out = vec![0.0f64; 2 * n];
        ImdctFft::<f64>::new(n)
            .unwrap()
            .execute_into(&coeffs, &mut fast_out)
            .unwrap();
        let naive_out = naive_imdct(&coeffs);

        for (i, (&fa, &na)) in fast_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (fa - na).abs() < 1e-9 * (1.0 + na.abs()),
                "roundtrip n={i}: fast={fa}, naive={na}"
            );
        }
    }

    #[test]
    fn test_imdct_rejects_odd_n() {
        assert!(ImdctFft::<f64>::new(5).is_err());
        assert!(ImdctFft::<f64>::new(7).is_err());
    }

    #[test]
    fn test_imdct_rejects_too_small() {
        assert!(ImdctFft::<f64>::new(2).is_err());
    }
}
