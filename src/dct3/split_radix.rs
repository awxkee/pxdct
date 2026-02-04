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
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::mla::fmla;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct SplitRadixDct3<T: DctSample> {
    twiddles: Vec<Complex<T>>,
    half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    quarter_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
    inner_scratch_size: usize,
    half_dct_scratch_size: usize,
    quarter_dct_scratch_size: usize,
}

impl<T: DctSample> SplitRadixDct3<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<SplitRadixDct3<T>, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half for Split-Radix DCT-III"
        );
        use crate::twiddles::compute_twiddle;
        let mut twiddles = try_vec![Complex::<T>::default(); len / 4];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            *twiddle = compute_twiddle::<T>(2 * i + 1, len * 4).conj();
        }

        let half_dct_scratch_size = half_dct.scratch_size();
        let quarter_dct_scratch_size = quarter_dct.scratch_size();

        Ok(SplitRadixDct3 {
            twiddles,
            half_dct,
            quarter_dct,
            execution_length: len,
            half_dct_scratch_size,
            quarter_dct_scratch_size,
            inner_scratch_size: len + half_dct_scratch_size.max(quarter_dct_scratch_size),
        })
    }
}

impl<T: DctSample> SplitRadixDct3<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, r) = scratch.split_at_mut(self.execution_length);
        let len = self.length();
        let half_len = len / 2;
        let quarter_len = len / 4;
        // divide the output into 3 sub-lists to use for our inner DCTs, one of size N/2 and two of size N/4
        let (recursive_input_evens, recursive_input_odds) = scratch.split_at_mut(half_len);
        let (recursive_input_n1, recursive_input_n3) =
            recursive_input_odds.split_at_mut(quarter_len);

        unsafe {
            *recursive_input_evens.get_unchecked_mut(0) = data[0];
            *recursive_input_evens.get_unchecked_mut(1) = data[2];
            *recursive_input_n1.get_unchecked_mut(0) = data[1] * T::TWO;
            *recursive_input_n3.get_unchecked_mut(0) = data[len - 1] * T::TWO;
        }

        // populate the recursive input arrays
        for i in 1..quarter_len {
            let k = 4 * i;

            unsafe {
                // the evens are the easy ones - just copy straight over
                *recursive_input_evens.get_unchecked_mut(i * 2) = data[k];
                *recursive_input_evens.get_unchecked_mut(i * 2 + 1) = data[k + 2];

                let b = data[k - 1];
                let f = data[k + 1];

                *recursive_input_n1.get_unchecked_mut(i) = b + f;
                *recursive_input_n3.get_unchecked_mut(quarter_len - i) = b - f;
            }
        }

        //perform our recursive DCTs, using the original buffer as scratch space

        let (half_dct_scratch, _) = r.split_at_mut(self.half_dct_scratch_size);

        self.half_dct
            .execute_with_scratch(recursive_input_evens, half_dct_scratch)?;

        let (quarter_dct_scratch, _) = r.split_at_mut(self.quarter_dct_scratch_size);

        self.quarter_dct
            .execute_with_scratch(recursive_input_n1, quarter_dct_scratch)?;
        self.quarter_dct
            .execute_with_scratch(recursive_input_n3, quarter_dct_scratch)?;

        let mut phase_sign = T::one();

        for (i, ((twiddle, &cosine_value), &sine_value)) in self
            .twiddles
            .iter()
            .zip(recursive_input_n1.iter())
            .zip(recursive_input_n3.iter())
            .enumerate()
        {
            // flip the sign of every other sine value to compute DST3 using DCT3
            let sine_value = sine_value.mulsign(phase_sign);

            let lower_dct4 = fmla(cosine_value, twiddle.re, sine_value * twiddle.im);
            let upper_dct4 = fmla(cosine_value, twiddle.im, -sine_value * twiddle.re);

            unsafe {
                let lower_dct3 = *recursive_input_evens.get_unchecked(i);
                let upper_dct3 = *recursive_input_evens.get_unchecked(half_len - i - 1);

                data[i] = lower_dct3 + lower_dct4;
                data[len - i - 1] = lower_dct3 - lower_dct4;

                data[half_len - i - 1] = upper_dct3 + upper_dct4;
                data[half_len + i] = upper_dct3 - upper_dct4;
            }
            phase_sign = -phase_sign;
        }
        Ok(())
    }
}

impl<T: DctSample> PxdctExecutor<T> for SplitRadixDct3<T>
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
            self.execute_with_store(&mut InPlaceStore::new(chunk), full_scratch)?;
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

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.execute_with_store(&mut BiStore::new(src, dst), full_scratch)?;
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        self.inner_scratch_size
    }
}

pub(crate) struct SplitRadixDst3<T: DctSample> {
    split_radix_dct3: SplitRadixDct3<T>,
    execution_length: usize,
}

impl<T: DctSample> SplitRadixDst3<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<SplitRadixDst3<T>, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half for Split-Radix DST-III"
        );
        Ok(SplitRadixDst3 {
            split_radix_dct3: SplitRadixDct3::new(len, half_dct, quarter_dct)?,
            execution_length: len,
        })
    }
}

impl<T: DctSample> PxdctExecutor<T> for SplitRadixDst3<T>
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

        let scratch = validate_scratch!(scratch, self.scratch_size());

        for chunk in data.chunks_exact_mut(self.execution_length) {
            chunk.reverse();

            self.split_radix_dct3.execute_with_scratch(chunk, scratch)?;

            for dst in chunk.chunks_exact_mut(2) {
                dst[1] = dst[1].neg();
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

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            for (dst, src) in dst.iter_mut().zip(src.iter().rev()) {
                *dst = *src;
            }

            self.split_radix_dct3.execute_with_scratch(dst, scratch)?;

            for dst in dst.chunks_exact_mut(2) {
                dst[1] = dst[1].neg();
            }
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        self.split_radix_dct3.scratch_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct3::Dct3Butterfly8;
    use crate::dct3::bf_f2::Dct3Butterfly16;
    use crate::tests::{naive_dct3, naive_dst3};
    use rand::Rng;

    #[test]
    fn test_split_dct3() {
        let mut input = vec![0.; 32];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct3(&reference_input);
        let bf = SplitRadixDct3::new(
            32,
            Arc::new(Dct3Butterfly16::default()),
            Arc::new(Dct3Butterfly8::default()),
        )
        .unwrap();
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
    fn test_split_dst3() {
        let mut input = vec![0.; 32];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dst3(&reference_input);
        let bf = SplitRadixDst3::new(
            32,
            Arc::new(Dct3Butterfly16::default()),
            Arc::new(Dct3Butterfly8::default()),
        )
        .unwrap();
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
