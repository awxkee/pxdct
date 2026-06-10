/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

//! MDCT computed as a time-domain-aliasing fold followed by a DCT-IV.
//!
//! The classic fast MDCT (`mdct_fft`) packs the `2N` real inputs into an
//! `N/2`-point complex FFT with bespoke pre/post rotation. That path uses a
//! *generic* complex FFT and never benefits from the heavily specialized
//! DCT-IV kernels in this crate (split-radix / mixed-radix / butterflies up to
//! 32, AVX2/NEON). This executor instead reuses them.
//!
//! For the MDCT convention used here,
//! `X_k = Σ_{n=0}^{2N-1} x_n cos( (π/N)(n + 1/2 + N/2)(k + 1/2) )`,
//! the transform is *exactly* the crate's (unnormalized) DCT-IV of a folded
//! length-`N` sequence. With `h = N/2`:
//!
//! ```text
//! z[n]     = -x[3N/2 - 1 - n] - x[3N/2 + n]      n = 0 .. h-1
//! z[h + n] =  x[n]            - x[N - 1 - n]      n = 0 .. h-1
//! X        =  DCT4(z)
//! ```
//!
//! No pre/post twiddle and no scaling correction are required: the fold maps
//! the MDCT basis onto the DCT-IV basis one-to-one (verified against the naive
//! reference and against `MdctFft`).

use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

/// Forward MDCT via DCT-IV.
///
/// `n` is the number of output bins (the MDCT spectrum size). The input block
/// for each transform is `2 * n` real samples.
pub(crate) struct MdctDct4<T: DctSample> {
    /// Length-`n` DCT-IV executor (carries the optimal sub-strategy).
    dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    /// MDCT bin count (output length); input length is `2 * n`.
    n: usize,
}

impl<T: DctSample> MdctDct4<T>
where
    f64: AsPrimitive<T>,
{
    /// Build an MDCT plan for `n` output bins from a length-`n` DCT-IV executor.
    pub(crate) fn new(
        dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<MdctDct4<T>, PxdctError> {
        let n = dct4.length();
        if n < 4 {
            return Err(PxdctError::InvalidSizeMultiplier(n, 4));
        }
        if !n.is_multiple_of(2) {
            return Err(PxdctError::InvalidSizeMultiplier(n, 2));
        }
        Ok(MdctDct4 { dct4, n })
    }

    /// Time-domain-aliasing fold: `2N` real input -> `N` real `fold_buf`.
    #[inline]
    fn fold(&self, input: &[T], fold_buf: &mut [T]) {
        let n = self.n;
        let h = n / 2;
        let n32 = 3 * n / 2;
        // First half: combine the third/fourth quarters (negated, mirrored).
        for k in 0..h {
            unsafe {
                *fold_buf.get_unchecked_mut(k) =
                    (*input.get_unchecked(n32 - 1 - k) + *input.get_unchecked(n32 + k)).neg();
            }
        }
        // Second half: first quarter minus mirrored second quarter.
        for k in 0..h {
            unsafe {
                *fold_buf.get_unchecked_mut(h + k) =
                    *input.get_unchecked(k) - *input.get_unchecked(n - 1 - k);
            }
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for MdctDct4<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, _data: &mut [T]) -> Result<(), PxdctError> {
        // MDCT cannot be in-place: input (2N) and output (N) differ in length.
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

        let full_scratch = validate_scratch!(scratch, self.scratch_size());
        let (fold_buf, dct4_scratch) = full_scratch.split_at_mut(self.n);

        for (src, dst) in input
            .chunks_exact(in_block)
            .zip(output.chunks_exact_mut(out_block))
        {
            self.fold(src, fold_buf);
            self.dct4
                .execute_into_with_scratch(fold_buf, dst, dct4_scratch)?;
        }

        Ok(())
    }

    fn length(&self) -> usize {
        self.n
    }

    fn scratch_size(&self) -> usize {
        // Fold buffer (N reals) + whatever the inner DCT-IV needs.
        self.n + self.dct4.scratch_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;

    /// Naive reference MDCT for validation (same convention as `mdct_fft`).
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
        let dct4 = Pxdct::strategy_dct4(n).unwrap();
        let mdct = MdctDct4::<f64>::new(dct4).unwrap();
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
    fn test_mdct_dct4_small() {
        for n in [4, 6, 8, 10, 12, 14, 16] {
            run_case(n, 1e-9);
        }
    }

    #[test]
    fn test_mdct_dct4_power_of_two() {
        for shift in 2..=11 {
            run_case(1 << shift, 1e-8);
        }
    }

    #[test]
    fn test_mdct_dct4_smooth_and_odd_half() {
        for n in [18, 20, 24, 30, 36, 48, 50, 60, 96, 120, 240, 360, 480, 960] {
            run_case(n, 1e-8);
        }
    }

    #[test]
    fn test_mdct_dct4_matches_fft_path() {
        use crate::mdct::MdctFft;
        for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let input: Vec<f64> = (0..2 * n).map(|x| (x as f64 * 0.31).sin()).collect();
            let mut a = vec![0.0f64; n];
            let mut b = vec![0.0f64; n];
            let fft = MdctFft::<f64>::new(n).unwrap();
            let dct4 = MdctDct4::<f64>::new(Pxdct::strategy_dct4(n).unwrap()).unwrap();
            fft.execute_into(&input, &mut a).unwrap();
            dct4.execute_into(&input, &mut b).unwrap();
            for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-7 * (1.0 + x.abs()),
                    "N={n}, k={i}: fft={x}, dct4={y}"
                );
            }
        }
    }

    #[test]
    fn test_mdct_dct4_multiple_blocks() {
        let n = 8;
        let input: Vec<f64> = (1..=6 * n).map(|x| x as f64).collect();
        let mut output = vec![0.0f64; 3 * n];
        let dct4 = Pxdct::strategy_dct4(n).unwrap();
        let mdct = MdctDct4::<f64>::new(dct4).unwrap();
        mdct.execute_into(&input, &mut output).unwrap();
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
}
