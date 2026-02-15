/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
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

pub(crate) struct Dct4Radix2<T> {
    dct2: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    dct2_scratch_size: usize,
}

impl<T: DctSample> Dct4Radix2<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        len: usize,
        dct2: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct2.length(),
            len / 2,
            "DCT-II even length DCTs must be half of DCT-IV"
        );
        let inner_len = dct2.length();

        use crate::twiddles::compute_twiddle;
        let mut twiddles = try_vec![Complex::<T>::default(); inner_len];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            *twiddle = compute_twiddle::<T>(2 * i + 1, len * 8).conj();
        }

        let dct2_scratch_size = dct2.scratch_size();

        Ok(Self {
            dct2,
            twiddles,
            execution_length: len,
            dct2_scratch_size,
        })
    }
}

impl<T: DctSample> Dct4Radix2<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);

        let len = self.length();
        let half_len = len / 2;
        let quarter_len = len / 4;

        // This kernel implements a radix-2 DCT-IV using one inner DCT-II.
        // It follows the classical even/odd pre-rotation → DCT-II → post-butterfly scheme.
        //
        // For each length-N block:
        //
        //   DCT4(N)  =  PostRotate · ( DCT2(N/2) ⊕ DCT2(N/2) ) · PreRotate
        //
        // where the pre/post rotations are fused with alternating sign flips to minimize
        // twiddle storage and multiplications.
        let (left, right) = scratch.split_at_mut(half_len);

        let mut sign_re = -T::one();
        let mut sign_im = T::one();

        // -------- Pre-rotation / even-odd folding --------
        // Fold symmetric samples (x[i], x[N-1-i]) into two half-length sequences.
        for (i, twiddle) in self.twiddles.iter().enumerate() {
            let front = data[i];
            let back = data[len - i - 1];
            unsafe {
                *left.get_unchecked_mut(i) = fmla(twiddle.re, front, twiddle.im * back);
            }
            unsafe {
                *right.get_unchecked_mut(half_len - i - 1) = fmla(
                    twiddle.re.mulsign(sign_re),
                    back,
                    twiddle.im.mulsign(sign_im) * front,
                );
            }
            sign_im = -sign_im;
            sign_re = -sign_re;
        }

        self.dct2.execute_with_scratch(scratch, inner_scratch)?;

        let (left, right) = scratch.split_at_mut(half_len);

        data[0] = left[0];
        data[len - 1] = right[0];

        let mut sign_left = -T::one();
        let mut sign_right = T::one();

        // -------- Post-butterfly recombination --------
        // Interleave even and odd spectra back into full DCT-IV ordering.
        for i in 1..quarter_len {
            let il = unsafe { *left.get_unchecked(i) };
            let rr = unsafe { *right.get_unchecked(half_len - i) };
            let rl = unsafe { *left.get_unchecked(half_len - i) };
            let ir = unsafe { *right.get_unchecked(i) };

            let q = i - 1;
            data[q * 2 + 1] = fmla(sign_left, rr, il);
            data[q * 2 + 2] = fmla(sign_right, rr, il);

            data[len - q * 2 - 3] = fmla(sign_left, ir, rl);
            data[len - q * 2 - 2] = fmla(sign_right, ir, rl);

            sign_left = -sign_left;
            sign_right = -sign_right;
        }

        unsafe {
            let ir = *right.get_unchecked(quarter_len);
            let il = *left.get_unchecked(quarter_len);
            data[half_len - 1] = fmla(sign_left, ir, il);
            data[half_len] = fmla(sign_right, ir, il);
        }
        Ok(())
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4Radix2<T>
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
        self.execution_length + self.dct2_scratch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct2::power2_butterflies::Dct2Butterfly32;
    use crate::tests::naive_dct4;
    use rand::Rng;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 64];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4Radix2::new(64, Arc::new(Dct2Butterfly32::default())).unwrap();
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
