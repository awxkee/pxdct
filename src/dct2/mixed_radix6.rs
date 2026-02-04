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
use crate::util::{DctSample, mixed_radix_inner_twiddle, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Dct2MixedRadix6<T> {
    inner_layer: Vec<Complex<T>>,
    sixth_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
    sixth_dct_scratch_size: usize,
    sixth_length: usize,
}

impl<T: DctSample> Dct2MixedRadix6<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        sixth_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Dct2MixedRadix6<T>, PxdctError> {
        assert_eq!(
            len,
            sixth_dct.length() * 6,
            "Invalid DCT was received, third size is not multiple of full size"
        );
        let inner_layer_groups = sixth_dct.length();
        let mut inner_layer = vec![Complex::<T>::zero(); inner_layer_groups * 4];
        for (i, layer) in inner_layer.chunks_exact_mut(4).enumerate() {
            let angle = (2. * i as f64 + 1.).as_();
            layer[0] = mixed_radix_inner_twiddle(angle, len);
            layer[0].im *= T::SQRT_3;
            layer[1] = mixed_radix_inner_twiddle(2f64.as_() * angle, len);
            layer[1].im *= T::SQRT_3;
            layer[2] = mixed_radix_inner_twiddle(3f64.as_() * angle, len);
            layer[2].im *= T::SQRT_3;
            layer[3] = mixed_radix_inner_twiddle(5f64.as_() * angle, len);
            layer[3].im = -layer[3].im * T::SQRT_3;
        }

        let sixth_dct_scratch_size = sixth_dct.scratch_size();
        let sixth_length = sixth_dct.length();

        Ok(Dct2MixedRadix6 {
            inner_layer,
            sixth_dct,
            execution_length: len,
            sixth_dct_scratch_size,
            sixth_length,
        })
    }
}

impl<T: DctSample> Dct2MixedRadix6<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let len = self.length();
        let s_n = len / 3;
        let s_2n = 2 * len / 3;
        let (a_buffer, rem) = scratch.split_at_mut(self.sixth_length);
        let (b_buffer, rem) = rem.split_at_mut(self.sixth_length);
        let (c_buffer, rem) = rem.split_at_mut(self.sixth_length);
        let (d_buffer, rem) = rem.split_at_mut(self.sixth_length);
        let (e_buffer, rem) = rem.split_at_mut(self.sixth_length);
        let (f_buffer, _) = rem.split_at_mut(self.sixth_length);

        for (i, inner_layer) in self.inner_layer.chunks_exact(4).enumerate() {
            let ai = data[i];
            let bi = data[s_n - i - 1];
            let ci = data[s_n + i];
            let di = data[s_2n - i - 1];
            let ei = data[s_2n + i];
            let fi = data[len - i - 1];

            let cos_sin_ai = inner_layer[0];
            let cos_sin_2ai = inner_layer[1];
            let cos_sin_3ai = inner_layer[2];
            let cos_sin_5ai = inner_layer[3];

            let s2 = bi + ei;
            let dcd = ci - di;
            let dbe = bi - ei;

            let ai2 = T::TWO * ai;
            let fi2 = T::TWO * fi;
            let scd = ci + di;

            let sdbedcd = dbe + dcd;
            let ai2dbedcd = ai2 + sdbedcd - fi2;

            let s2scd = s2 + scd;

            let a_comp = ai + s2scd + fi;
            let c_comp = ai2 - s2scd + fi2;
            let d_comp = T::TWO * (ai - sdbedcd - fi);

            let dbedcd = dbe - dcd;

            let c_img = s2 - ci - di;
            let b_zet = dbedcd * cos_sin_ai.im;
            let c_zet = c_img * cos_sin_2ai.im;
            let f_zet = dbedcd * cos_sin_5ai.im;

            let e_comp = fmla(
                T::TWO * cos_sin_2ai.re,
                fmla(c_comp, cos_sin_2ai.re, -c_zet),
                -c_comp,
            );

            unsafe {
                *a_buffer.get_unchecked_mut(i) = a_comp;
                *b_buffer.get_unchecked_mut(i) = fmla(ai2dbedcd, cos_sin_ai.re, b_zet);
                *c_buffer.get_unchecked_mut(i) = fmla(c_comp, cos_sin_2ai.re, c_zet);
                *d_buffer.get_unchecked_mut(i) = d_comp * cos_sin_3ai.re;
                *e_buffer.get_unchecked_mut(i) = e_comp;
                *f_buffer.get_unchecked_mut(i) = fmla(ai2dbedcd, cos_sin_5ai.re, f_zet);
            }
        }

        if a_buffer.len() > 1 {
            self.sixth_dct
                .execute_with_scratch(scratch, inner_scratch)?;
        }

        let (a_buffer, rem) = scratch.split_at_mut(self.sixth_length);
        let (b_buffer, rem) = rem.split_at_mut(self.sixth_length);
        let (c_buffer, rem) = rem.split_at_mut(self.sixth_length);
        let (d_buffer, rem) = rem.split_at_mut(self.sixth_length);
        let (e_buffer, rem) = rem.split_at_mut(self.sixth_length);
        let (f_buffer, _) = rem.split_at_mut(self.sixth_length);

        data[0] = a_buffer[0];
        let b0 = b_buffer[0] * T::HALF;
        data[1] = b0;
        let c0 = c_buffer[0] * T::HALF;
        data[2] = c0;
        let d0 = d_buffer[0] * T::HALF;
        data[3] = d0;
        let e0 = e_buffer[0] * T::HALF;
        data[4] = e0;
        let f0 = f_buffer[0] * T::HALF;
        data[5] = f0;

        let mut b_diff = f0;
        let mut c_diff = e0;
        let mut e_diff = d0;
        let mut d_diff = c0;
        let mut f_diff = b0;

        for k in 1..self.sixth_length {
            let deferred_d_diff;
            let deferred_f_diff;
            unsafe {
                data[6 * k] = *a_buffer.get_unchecked(k);
            }
            unsafe {
                deferred_f_diff = *b_buffer.get_unchecked(k) - b_diff;
                data[6 * k + 1] = deferred_f_diff;
            }
            unsafe {
                deferred_d_diff = *c_buffer.get_unchecked(k) - c_diff;
                data[6 * k + 2] = deferred_d_diff;
            }
            unsafe {
                e_diff = *d_buffer.get_unchecked(k) - e_diff;
                data[6 * k + 3] = e_diff;
            }
            unsafe {
                let new_d = *e_buffer.get_unchecked(k) - d_diff;
                data[6 * k + 4] = new_d;
                c_diff = new_d;
                d_diff = deferred_d_diff;
            }
            unsafe {
                let new_f = *f_buffer.get_unchecked(k) - f_diff;
                b_diff = new_f;
                f_diff = deferred_f_diff;
                data[6 * k + 5] = new_f;
            }
        }
        Ok(())
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2MixedRadix6<T>
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
        self.execution_length + self.sixth_dct_scratch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Dct2Butterfly36;
    use crate::tests::naive_dct2;
    use rand::Rng;

    #[test]
    fn test_radix6_dct2() {
        let mut input = vec![0.; 216];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2(&reference_input);
        let bf = Dct2MixedRadix6::new(216, Arc::new(Dct2Butterfly36::default())).unwrap();
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
