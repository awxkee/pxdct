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

pub(crate) struct SplitRadixDct2Impl<T: DctSample, const SCALED: bool> {
    twiddles: Vec<Complex<T>>,
    half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    quarter_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
    inner_scratch_size: usize,
    half_dct_scratch_size: usize,
    quarter_dct_scratch_size: usize,
}

pub(crate) type SplitRadixDct2<T> = SplitRadixDct2Impl<T, false>;
pub(crate) type ScaledSplitRadixDct2<T> = SplitRadixDct2Impl<T, true>;

impl<T: DctSample, const SCALED: bool> SplitRadixDct2Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<SplitRadixDct2Impl<T, SCALED>, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half"
        );

        let scale = (2f64 / len as f64).sqrt().as_();

        use crate::twiddles::compute_twiddle;
        let mut twiddles = try_vec![Complex::<T>::default(); len / 4];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            if SCALED {
                *twiddle = compute_twiddle::<T>(2 * i + 1, len * 4).conj() * scale;
            } else {
                *twiddle = compute_twiddle::<T>(2 * i + 1, len * 4).conj();
            }
        }

        let half_dct_scratch_size = half_dct.scratch_size();
        let quarter_dct_scratch_size = quarter_dct.scratch_size();

        Ok(SplitRadixDct2Impl {
            twiddles,
            half_dct,
            quarter_dct,
            execution_length: len,
            inner_scratch_size: len + half_dct_scratch_size.max(quarter_dct_scratch_size),
            half_dct_scratch_size,
            quarter_dct_scratch_size,
        })
    }
}

impl<T: DctSample, const SCALED: bool> SplitRadixDct2Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    fn execute_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
        scale: T,
    ) -> Result<(), PxdctError> {
        let (scratch, r) = scratch.split_at_mut(self.execution_length);
        let len = self.length();
        let half_len = len / 2;
        let quarter_len = len / 4;
        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
        let (input_dct2, input_dct4) = scratch.split_at_mut(half_len);
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(quarter_len);

        for (i, twiddle) in self.twiddles.iter().enumerate() {
            let input_bottom = data[i];
            let input_top = data[len - i - 1];

            let input_half_bottom = data[half_len - i - 1];
            let input_half_top = data[half_len + i];

            //prepare the inner DCT2
            if SCALED {
                unsafe { *input_dct2.get_unchecked_mut(i) = (input_top + input_bottom) * scale };
                unsafe {
                    *input_dct2.get_unchecked_mut(half_len - i - 1) =
                        (input_half_bottom + input_half_top) * scale
                };
            } else {
                unsafe { *input_dct2.get_unchecked_mut(i) = input_top + input_bottom };
                unsafe {
                    *input_dct2.get_unchecked_mut(half_len - i - 1) =
                        input_half_bottom + input_half_top
                };
            }

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle.re, upper_dct4 * twiddle.im);
            let sin_input = fmla(upper_dct4, twiddle.re, -lower_dct4 * twiddle.im);

            unsafe { *input_dct4_even.get_unchecked_mut(i) = cos_input };
            unsafe {
                *input_dct4_odd.get_unchecked_mut(quarter_len - i - 1) =
                    if i % 2 == 0 { sin_input } else { -sin_input }
            };
        }

        let (half_dct_scratch, _) = r.split_at_mut(self.half_dct_scratch_size);

        self.half_dct
            .execute_with_scratch(input_dct2, half_dct_scratch)?;

        let (quarter_dct_scratch, _) = r.split_at_mut(self.quarter_dct_scratch_size);

        self.quarter_dct
            .execute_with_scratch(input_dct4, quarter_dct_scratch)?;

        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(quarter_len);

        unsafe {
            //post process the 3 DCT2 outputs. the first few and the last will be done outside the loop
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            for i in 1..quarter_len {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + quarter_len) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(quarter_len - i)
                } else {
                    *input_dct4_odd.get_unchecked(quarter_len - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[len - 1] = -*input_dct4_odd.get_unchecked(0);
        }
        Ok(())
    }
}

impl<T: DctSample, const SCALED: bool> PxdctExecutor<T> for SplitRadixDct2Impl<T, SCALED>
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

        let scale = (2f64 / self.execution_length as f64).sqrt().as_();

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        for chunk in data.chunks_exact_mut(self.execution_length) {
            self.execute_store(&mut InPlaceStore::new(chunk), full_scratch, scale)?;
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

        let scale = (2f64 / self.execution_length as f64).sqrt().as_();

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.execute_store(&mut BiStore::new(src, dst), full_scratch, scale)?;
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        self.inner_scratch_size
    }
}

#[allow(unused)]
pub(crate) struct SplitRadixDst2<T: DctSample> {
    split_radix_dct2: SplitRadixDct2<T>,
    execution_length: usize,
}

impl<T: DctSample> SplitRadixDst2<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<SplitRadixDst2<T>, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half in DST-II"
        );

        Ok(SplitRadixDst2 {
            split_radix_dct2: SplitRadixDct2::new(len, half_dct, quarter_dct)?,
            execution_length: len,
        })
    }
}

impl<T: DctSample> PxdctExecutor<T> for SplitRadixDst2<T>
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
            for dst in chunk.as_chunks_mut::<2>().0.iter_mut() {
                dst[1] = dst[1].neg();
            }

            self.split_radix_dct2.execute_with_scratch(chunk, scratch)?;

            chunk.reverse();
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

        let scratch = validate_scratch!(scratch, self.scratch_size());

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            for (src, dst) in src
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<2>().0.iter_mut())
            {
                dst[0] = src[0];
                dst[1] = src[1].neg();
            }

            self.split_radix_dct2.execute_with_scratch(dst, scratch)?;

            dst.reverse();
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        self.split_radix_dct2.scratch_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{naive_dct2, naive_dst2};
    use crate::type2::power2_butterflies::{Dct2Butterfly8, Dct2Butterfly16};
    use rand::RngExt;

    #[test]
    fn test_split_dct2() {
        let mut input = vec![0.; 32];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2(&reference_input);
        let bf = SplitRadixDct2::new(
            32,
            Arc::new(Dct2Butterfly16::default()),
            Arc::new(Dct2Butterfly8::default()),
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
    fn test_split_dst2() {
        let mut input = vec![0.; 32];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dst2(&reference_input);
        let bf = SplitRadixDst2::new(
            32,
            Arc::new(Dct2Butterfly16::default()),
            Arc::new(Dct2Butterfly8::default()),
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
