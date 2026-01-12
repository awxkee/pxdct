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
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct Dct2MixedRadix2<T> {
    twiddles: Vec<T>,
    half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
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
        })
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2MixedRadix2<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![T::default(); self.execution_length];

        let half_len = self.half_dct.length();

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, b_buffer) = scratch.split_at_mut(half_len);

            let (left, right) = chunk.split_at(half_len);

            for ((((dst_l, dst_r), &l), &r), &twiddle) in a_buffer
                .iter_mut()
                .zip(b_buffer.iter_mut())
                .zip(left.iter())
                .zip(right.iter().rev())
                .zip(self.twiddles.iter())
            {
                *dst_l = l + r;
                *dst_r = (l - r) * twiddle;
            }

            if a_buffer.len() > 1 {
                self.half_dct.execute(&mut scratch)?;
            }

            let (a_buffer, b_buffer) = scratch.split_at_mut(half_len);
            b_buffer[0] *= T::HALF;

            let mut last_odd = T::zero();

            for ((dst, &even), &odd) in chunk
                .chunks_exact_mut(2)
                .zip(a_buffer.iter())
                .zip(b_buffer.iter())
            {
                dst[0] = even;
                last_odd = odd - last_odd;
                dst[1] = last_odd;
            }
        }

        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Dct2Butterfly6;
    use crate::tests::naive_dct2;
    use rand::Rng;

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
