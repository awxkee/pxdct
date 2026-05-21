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
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct SplitRadixDct1Impl<T: DctSample, const SCALED: bool> {
    half_dct1_p1: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    half_dct3: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
    inner_scratch_size: usize,
    half_dct1_scratch: usize,
    half_dct3_scratch: usize,
}

pub(crate) type SplitRadixDct1<T> = SplitRadixDct1Impl<T, false>;

impl<T: DctSample, const SCALED: bool> SplitRadixDct1Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        len: usize,
        half_dct1_p1: Arc<dyn PxdctExecutor<T> + Send + Sync>,
        half_dct3: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<SplitRadixDct1Impl<T, SCALED>, PxdctError> {
        let n1 = len / 2;
        assert_eq!(
            half_dct1_p1.length(),
            n1 + 1,
            "Invalid DCT was received, half size + 1 DCT-I is not match to {n1}"
        );
        assert_eq!(
            half_dct3.length(),
            n1,
            "Invalid DCT was received, half size DCT-III is not match to {n1}"
        );

        let half_dct1_scratch = half_dct1_p1.scratch_size();
        let half_dct3_scratch = half_dct3.scratch_size();

        Ok(SplitRadixDct1Impl {
            half_dct1_p1,
            half_dct3,
            execution_length: len,
            inner_scratch_size: len + half_dct1_scratch.max(half_dct3_scratch),
            half_dct1_scratch,
            half_dct3_scratch,
        })
    }
}

impl<T: DctSample, const SCALED: bool> SplitRadixDct1Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    fn execute_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let len = self.length();

        let (hadamard, inner_scratch) = scratch.split_at_mut(self.execution_length);

        let n1 = self.execution_length / 2;
        let src_slice = data.slice_from(0..);
        let (left, right) = src_slice.split_at(n1);
        let mid_and_right = &right[1..];

        let (left_dst, right_dst_q) = hadamard.split_at_mut(n1);
        let (_, right_dst_p1) = right_dst_q.split_at_mut(1);

        left.iter()
            .zip(mid_and_right.iter().rev())
            .zip(left_dst.iter_mut())
            .zip(right_dst_p1.iter_mut())
            .for_each(|(((&a, &b), sum), dif)| {
                *sum = a + b;
                *dif = a - b;
            });

        hadamard[n1] = T::TWO * data[n1];

        let (half_dct1_scratch, _) = inner_scratch.split_at_mut(self.half_dct1_scratch);
        self.half_dct1_p1
            .execute_with_scratch(&mut hadamard[0..=n1], half_dct1_scratch)?;
        let (half_dct3_scratch, _) = inner_scratch.split_at_mut(self.half_dct3_scratch);
        self.half_dct3
            .execute_with_scratch(&mut hadamard[n1 + 1..len], half_dct3_scratch)?;

        // for i in 0..=n1 {
        //     data[2 * i] = hadamard[i];
        // }
        // for i in 0..n1 {
        //     data[2 * i + 1] = hadamard[n1 + 1 + i] * T::TWO;
        // }
        unsafe {
            for i in 0..n1 {
                data[2 * i] = *hadamard.get_unchecked(i);
                data[2 * i + 1] = *hadamard.get_unchecked(n1 + 1 + i) * T::TWO;
            }
            data[2 * n1] = *hadamard.get_unchecked(n1);
        }
        Ok(())
    }
}

impl<T: DctSample, const SCALED: bool> PxdctExecutor<T> for SplitRadixDct1Impl<T, SCALED>
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

    fn scratch_size(&self) -> usize {
        self.inner_scratch_size
    }
}

#[cfg(test)]
mod tests {
    use crate::PxdctExecutor;
    use crate::tests::naive_dct1;
    use crate::type1::butterflies::Dct1Butterfly5;
    use crate::type1::split_radix::SplitRadixDct1;
    use crate::type3::Dct3Butterfly4;
    use rand::RngExt;
    use std::sync::Arc;

    #[test]
    fn test_split_dct1() {
        let mut input = vec![0.; 9];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct1(&reference_input);
        let bf = SplitRadixDct1::new(
            9,
            Arc::new(Dct1Butterfly5::default()),
            Arc::new(Dct3Butterfly4::default()),
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
