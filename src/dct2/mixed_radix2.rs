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
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct Dct2MixedRadix2<T> {
    twiddles: Vec<T>,
    half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
    half_dct_length: usize,
}

impl<T: DctSample> Dct2MixedRadix2<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Dct2MixedRadix2<T>, PxdctError> {
        assert_eq!(
            len,
            half_dct.length() * 2,
            "Invalid DCT was received, half size is not multiple of full size"
        );
        assert!(
            len.is_multiple_of(2),
            "DCT-II Mixed Radix-2 can do only multiples of 2"
        );
        let half_size = half_dct.length();
        let mut twiddles = vec![T::default(); half_size];
        let length_scale = (1f64 / (2f64 * len as f64)).as_();
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            *twiddle = 2f64.as_() * (((i as f64 * 2.).as_() + 1f64.as_()) * length_scale).cospi();
        }

        Ok(Dct2MixedRadix2 {
            half_dct,
            twiddles,
            execution_length: len,
            half_dct_length: half_size,
        })
    }
}

impl<T: DctSample> Dct2MixedRadix2<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn execute_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, b_buffer) = scratch.split_at_mut(self.half_dct_length);

        for (i, &twiddle) in self.twiddles.iter().enumerate() {
            let left = data[i];
            let right = data[self.execution_length - i - 1];

            unsafe {
                *a_buffer.get_unchecked_mut(i) = left + right;
                *b_buffer.get_unchecked_mut(i) = (left - right) * twiddle;
            }
        }

        if a_buffer.len() > 1 {
            self.half_dct.execute_with_scratch(scratch, inner_scratch)?;
        }

        let (a_buffer, b_buffer) = scratch.split_at_mut(self.half_dct_length);
        b_buffer[0] *= T::HALF;

        let mut last_odd = T::zero();

        for (i, (&even, &odd)) in a_buffer.iter().zip(b_buffer.iter()).enumerate() {
            data[i * 2] = even;
            last_odd = odd - last_odd;
            data[i * 2 + 1] = last_odd;
        }
        Ok(())
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2MixedRadix2<T>
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
            self.execute_store(&mut InPlaceStore::new(chunk), full_scratch)?;
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
            self.execute_store(&mut BiStore::new(src, dst), full_scratch)?;
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        self.execution_length + self.half_dct.scratch_size()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Dct2Butterfly6;
    use crate::tests::naive_dct2;
    use rand::RngExt;

    #[test]
    fn test_radix2_dct() {
        let mut input = vec![0.; 12];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2(&reference_input);
        let bf = Dct2MixedRadix2::new(12, Arc::new(Dct2Butterfly6::default())).unwrap();
        bf.execute(&mut input).unwrap();
        println!("{:?}", input);
        println!("{:?}", reference_input);
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
