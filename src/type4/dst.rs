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

use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

/// DST-IV via DCT-IV:
/// DST-IV(x)[k] = DCT-IV(x')[k]
/// where x'[n] = (-1)^n * x[N-1-n]  (reverse + alternate signs)
/// Result is already the DST-IV output.
pub(crate) struct Dst4<T: DctSample> {
    dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
}

impl<T: DctSample> Dst4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Dst4<T>, PxdctError> {
        let len = dct4.length();
        Ok(Dst4 {
            dct4,
            execution_length: len,
        })
    }

    #[inline]
    fn preprocess(chunk: &mut [T]) {
        // Reverse in-place then apply (-1)^n
        chunk.reverse();
        let mut sign = T::one();
        let neg_one = -T::one();
        for x in chunk.iter_mut() {
            *x = *x * sign;
            sign *= neg_one;
        }
    }

    #[inline]
    fn preprocess_into(src: &[T], dst: &mut [T]) {
        let len = src.len();
        let mut sign = T::one();
        let neg_one = -T::one();
        for i in 0..len {
            dst[i] = src[len - 1 - i] * sign;
            sign *= neg_one;
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dst4<T>
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

        for chunk in data.chunks_exact_mut(self.execution_length) {
            Self::preprocess(chunk);
            self.dct4.execute_with_scratch(chunk, full_scratch)?;
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

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            Self::preprocess_into(src, dst);
            self.dct4.execute_with_scratch(dst, full_scratch)?;
        }

        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        self.dct4.scratch_size()
    }
}

#[cfg(test)]
mod tests {
    use super::Dst4;
    use crate::{Pxdct, PxdctExecutor};

    fn naive_dst4(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let angle = std::f64::consts::PI
                            * (2 * i + 1) as f64
                            * (2 * k + 1) as f64
                            / (4 * n) as f64;
                        x * angle.sin()
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn test_dst4_size7() {
        let input = vec![1., 2., 3., 4., 5., 6., 7.];
        let mut working = input.clone();
        let control = naive_dst4(&input);
        let dct4 = Pxdct::strategy_dct4(input.len()).unwrap();
        let dst4 = Dst4::new(dct4).unwrap();
        dst4.execute(&mut working).unwrap();
        working.iter().zip(control.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
        });
    }

    #[test]
    fn test_dst4_size8() {
        let input = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut working = input.clone();
        let control = naive_dst4(&input);
        let dct4 = Pxdct::strategy_dct4(input.len()).unwrap();
        let dst4 = Dst4::new(dct4).unwrap();
        dst4.execute(&mut working).unwrap();
        working.iter().zip(control.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
        });
    }

    #[test]
    fn test_dst4_size14() {
        let input: Vec<f64> = (1..=14).map(|x| x as f64).collect();
        let mut working = input.clone();
        let control = naive_dst4(&input);
        let dct4 = Pxdct::strategy_dct4(input.len()).unwrap();
        let dst4 = Dst4::new(dct4).unwrap();
        dst4.execute(&mut working).unwrap();
        working.iter().zip(control.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
        });
    }

    #[test]
    fn test_dst4_into_size7() {
        let input = vec![1., 2., 3., 4., 5., 6., 7.];
        let mut output = vec![0.0f64; 7];
        let control = naive_dst4(&input);
        let dct4 = Pxdct::strategy_dct4(input.len()).unwrap();
        let dst4 = Dst4::new(dct4).unwrap();
        dst4.execute_into(&input, &mut output).unwrap();
        output.iter().zip(control.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
        });
    }

    #[test]
    fn test_dst4_all_sizes() {
        for n in 1..150 {
            let input: Vec<f64> = (1..=n).map(|x| x as f64).collect();
            let mut working = input.clone();
            let control = naive_dst4(&input);
            let dct4 = Pxdct::strategy_dct4(n).unwrap();
            let dst4 = Dst4::new(dct4).unwrap();
            dst4.execute(&mut working).unwrap();
            working.iter().zip(control.iter()).enumerate().for_each(|(i, (&x, &c))| {
                assert!(
                    (x - c).abs() < 1e-7,
                    "size {n}, index {i}: got {x}, expected {c}"
                );
            });
        }
    }
}