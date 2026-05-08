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
use crate::neon::store_d::NeonStoreD;
use crate::neon::util::NeonStoreF;
use crate::util::{DctConstants, DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) trait NeonDct2Modulation: Sized {
    fn modulate_input(
        a_buffer: &mut [Self],
        b_buffer: &mut [Self],
        left: &[Self],
        right: &[Self],
        twiddles: &[Self],
    );
}

impl NeonDct2Modulation for f32 {
    fn modulate_input(
        a_buffer: &mut [Self],
        b_buffer: &mut [Self],
        left: &[Self],
        right: &[Self],
        twiddles: &[Self],
    ) {
        let mut j = 0usize;
        let total_length = a_buffer.len();

        while j + 16 <= total_length {
            unsafe {
                let a0 = NeonStoreF::load(left.get_unchecked(j..));
                let a1 = NeonStoreF::load(left.get_unchecked(j + 4..));
                let a2 = NeonStoreF::load(left.get_unchecked(j + 8..));
                let a3 = NeonStoreF::load(left.get_unchecked(j + 12..));

                let mut b0 = NeonStoreF::load(right.get_unchecked(total_length - j - 4..));
                let mut b1 = NeonStoreF::load(right.get_unchecked(total_length - j - 8..));
                let mut b2 = NeonStoreF::load(right.get_unchecked(total_length - j - 12..));
                let mut b3 = NeonStoreF::load(right.get_unchecked(total_length - j - 16..));

                let tw0 = NeonStoreF::load(twiddles.get_unchecked(j..));
                let tw1 = NeonStoreF::load(twiddles.get_unchecked(j + 4..));
                let tw2 = NeonStoreF::load(twiddles.get_unchecked(j + 8..));
                let tw3 = NeonStoreF::load(twiddles.get_unchecked(j + 12..));

                b0 = b0.reverse();
                b1 = b1.reverse();
                b2 = b2.reverse();
                b3 = b3.reverse();

                let s0 = a0 + b0;
                let s1 = a1 + b1;
                let s2 = a2 + b2;
                let s3 = a3 + b3;

                let d0 = (a0 - b0) * tw0;
                let d1 = (a1 - b1) * tw1;
                let d2 = (a2 - b2) * tw2;
                let d3 = (a3 - b3) * tw3;

                s0.write(a_buffer.get_unchecked_mut(j..));
                s1.write(a_buffer.get_unchecked_mut(j + 4..));
                s2.write(a_buffer.get_unchecked_mut(j + 8..));
                s3.write(a_buffer.get_unchecked_mut(j + 12..));

                d0.write(b_buffer.get_unchecked_mut(j..));
                d1.write(b_buffer.get_unchecked_mut(j + 4..));
                d2.write(b_buffer.get_unchecked_mut(j + 8..));
                d3.write(b_buffer.get_unchecked_mut(j + 12..));
            }
            j += 16;
        }

        while j + 4 <= total_length {
            unsafe {
                let a0 = NeonStoreF::load(left.get_unchecked(j..));
                let mut b0 = NeonStoreF::load(right.get_unchecked(total_length - j - 4..));

                let tw0 = NeonStoreF::load(twiddles.get_unchecked(j..));

                b0 = b0.reverse();

                let s0 = a0 + b0;
                let d0 = (a0 - b0) * tw0;

                s0.write(a_buffer.get_unchecked_mut(j..));
                d0.write(b_buffer.get_unchecked_mut(j..));
            }
            j += 4;
        }

        let rem = total_length - j;
        assert!(rem < 4);

        match rem {
            3 => unsafe {
                let a0 = NeonStoreF::load3(left.get_unchecked(j..));
                let mut b0 = NeonStoreF::load3(right.get_unchecked(total_length - j - 3..));
                b0 = b0.reverse3();

                let tw0 = NeonStoreF::load3(twiddles.get_unchecked(j..));

                let s0 = a0 + b0;
                let d0 = (a0 - b0) * tw0;

                s0.write3(a_buffer.get_unchecked_mut(j..));
                d0.write3(b_buffer.get_unchecked_mut(j..));
            },
            2 => unsafe {
                let a0 = NeonStoreF::load2(left.get_unchecked(j..));
                let mut b0 = NeonStoreF::load2(right.get_unchecked(total_length - j - 2..));
                let tw0 = NeonStoreF::load2(twiddles.get_unchecked(j..));

                b0 = b0.reverse2();

                let s0 = a0 + b0;
                let d0 = (a0 - b0) * tw0;

                s0.write2(a_buffer.get_unchecked_mut(j..));
                d0.write2(b_buffer.get_unchecked_mut(j..));
            },
            1 => unsafe {
                let a0 = NeonStoreF::load1(left.get_unchecked(j..));
                let b0 = NeonStoreF::load1(right.get_unchecked(total_length - j - 1..));
                let tw0 = NeonStoreF::load1(twiddles.get_unchecked(j..));

                let s0 = a0 + b0;
                let d0 = (a0 - b0) * tw0;

                s0.write1(a_buffer.get_unchecked_mut(j..));
                d0.write1(b_buffer.get_unchecked_mut(j..));
            },
            _ => {}
        }
    }
}

impl NeonDct2Modulation for f64 {
    fn modulate_input(
        a_buffer: &mut [Self],
        b_buffer: &mut [Self],
        left: &[Self],
        right: &[Self],
        twiddles: &[Self],
    ) {
        let mut j = 0usize;
        let total_length = a_buffer.len();

        while j + 8 <= total_length {
            unsafe {
                let a0 = NeonStoreD::load(left.get_unchecked(j..));
                let a1 = NeonStoreD::load(left.get_unchecked(j + 2..));
                let a2 = NeonStoreD::load(left.get_unchecked(j + 4..));
                let a3 = NeonStoreD::load(left.get_unchecked(j + 6..));

                let mut b0 = NeonStoreD::load(right.get_unchecked(total_length - j - 2..));
                let mut b1 = NeonStoreD::load(right.get_unchecked(total_length - j - 4..));
                let mut b2 = NeonStoreD::load(right.get_unchecked(total_length - j - 6..));
                let mut b3 = NeonStoreD::load(right.get_unchecked(total_length - j - 8..));

                let tw0 = NeonStoreD::load(twiddles.get_unchecked(j..));
                let tw1 = NeonStoreD::load(twiddles.get_unchecked(j + 2..));
                let tw2 = NeonStoreD::load(twiddles.get_unchecked(j + 4..));
                let tw3 = NeonStoreD::load(twiddles.get_unchecked(j + 6..));

                b0 = b0.reverse();
                b1 = b1.reverse();
                b2 = b2.reverse();
                b3 = b3.reverse();

                let s0 = a0 + b0;
                let s1 = a1 + b1;
                let s2 = a2 + b2;
                let s3 = a3 + b3;

                let d0 = (a0 - b0) * tw0;
                let d1 = (a1 - b1) * tw1;
                let d2 = (a2 - b2) * tw2;
                let d3 = (a3 - b3) * tw3;

                s0.write(a_buffer.get_unchecked_mut(j..));
                s1.write(a_buffer.get_unchecked_mut(j + 2..));
                s2.write(a_buffer.get_unchecked_mut(j + 4..));
                s3.write(a_buffer.get_unchecked_mut(j + 6..));

                d0.write(b_buffer.get_unchecked_mut(j..));
                d1.write(b_buffer.get_unchecked_mut(j + 2..));
                d2.write(b_buffer.get_unchecked_mut(j + 4..));
                d3.write(b_buffer.get_unchecked_mut(j + 6..));
            }
            j += 8;
        }

        while j + 2 <= total_length {
            unsafe {
                let a0 = NeonStoreD::load(left.get_unchecked(j..));
                let mut b0 = NeonStoreD::load(right.get_unchecked(total_length - j - 2..));
                let tw0 = NeonStoreD::load(twiddles.get_unchecked(j..));

                b0 = b0.reverse();

                let s0 = a0 + b0;
                let d0 = (a0 - b0) * tw0;

                s0.write(a_buffer.get_unchecked_mut(j..));
                d0.write(b_buffer.get_unchecked_mut(j..));
            }
            j += 2;
        }

        let rem = total_length - j;
        assert!(rem < 2);

        if rem == 1 {
            unsafe {
                let a0 = NeonStoreD::load1(left.get_unchecked(j..));
                let b0 = NeonStoreD::load1(right.get_unchecked(total_length - j - 1..));
                let tw0 = NeonStoreD::load1(twiddles.get_unchecked(j..));

                let s0 = a0 + b0;
                let d0 = (a0 - b0) * tw0;

                s0.write1(a_buffer.get_unchecked_mut(j..));
                d0.write1(b_buffer.get_unchecked_mut(j..));
            }
        }
    }
}

trait MixedRadix2Differences<T> {
    fn accumulate(a_buffer: &[T], b_buffer: &mut [T], output: &mut [T]);
}

impl MixedRadix2Differences<f32> for f32 {
    fn accumulate(a_buffer: &[f32], b_buffer: &mut [f32], chunk: &mut [f32]) {
        b_buffer[0] *= f32::HALF;

        let mut last_odd = 0.;

        let mut j = 0usize;
        let mut dst_idx = 0usize;

        if j + 4 < a_buffer.len() {
            let mut differences = NeonStoreF::set_values(0., 0., 0., 0.);

            // [1 -1 1 -1]
            let sign = NeonStoreF::set_values(0.0, -0.0, 0.0, -0.0);

            while j + 8 < a_buffer.len() {
                differences = differences.xor(sign);

                let evens0 = NeonStoreF::load(unsafe { a_buffer.get_unchecked(j..) });
                let odds0 = NeonStoreF::load(unsafe { b_buffer.get_unchecked(j..) });

                let evens1 = NeonStoreF::load(unsafe { a_buffer.get_unchecked(j + 4..) });
                let odds1 = NeonStoreF::load(unsafe { b_buffer.get_unchecked(j + 4..) });

                let differences0 = odds0.prefix_differences(sign) - differences;

                differences = differences0;
                differences = differences.broadcast_last();
                differences = differences.xor(sign);

                let differences1 = odds1.prefix_differences(sign) - differences;

                let zipped0 = NeonStoreF::zip(evens0, differences0);
                let zipped1 = NeonStoreF::zip(evens1, differences1);

                differences = differences1.broadcast_last();

                unsafe {
                    zipped0[0].write(chunk.get_unchecked_mut(dst_idx..));
                    zipped0[1].write(chunk.get_unchecked_mut(dst_idx + 4..));

                    zipped1[0].write(chunk.get_unchecked_mut(dst_idx + 8..));
                    zipped1[1].write(chunk.get_unchecked_mut(dst_idx + 12..));
                }

                dst_idx += 16;
                j += 8;
            }

            while j + 4 < a_buffer.len() {
                differences = differences.xor(sign);
                let evens = NeonStoreF::load(unsafe { a_buffer.get_unchecked(j..) });
                let odds = NeonStoreF::load(unsafe { b_buffer.get_unchecked(j..) });

                differences = odds.prefix_differences(sign) - differences;

                // differences = odds - differences;
                let zipped = NeonStoreF::zip(evens, differences);
                differences = differences.broadcast_last();
                unsafe {
                    zipped[0].write(chunk.get_unchecked_mut(dst_idx..));
                    zipped[1].write(chunk.get_unchecked_mut(dst_idx + 4..));
                }
                dst_idx += 8;
                j += 4;
            }

            last_odd = differences.last();
        }

        let chunk = &mut chunk[dst_idx..];
        let a_buffer = &a_buffer[j..];
        let b_buffer = &b_buffer[j..];

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
}

impl MixedRadix2Differences<f64> for f64 {
    fn accumulate(a_buffer: &[f64], b_buffer: &mut [f64], chunk: &mut [f64]) {
        b_buffer[0] *= f64::HALF;

        let mut last_odd = 0.;

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
}

pub(crate) struct NeonDct2MixedRadix2<T> {
    twiddles: Vec<T>,
    half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    half_dct_scratch_size: usize,
    execution_length: usize,
}

impl<T: DctSample> NeonDct2MixedRadix2<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<NeonDct2MixedRadix2<T>, PxdctError> {
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

        let half_dct_scratch_size = half_dct.scratch_size();

        Ok(NeonDct2MixedRadix2 {
            half_dct,
            half_dct_scratch_size,
            twiddles,
            execution_length: len,
        })
    }
}

impl<T: DctSample + NeonDct2Modulation + MixedRadix2Differences<T>> PxdctExecutor<T>
    for NeonDct2MixedRadix2<T>
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
        let (scratch, inner_scratch) = full_scratch.split_at_mut(self.execution_length);

        let half_len = self.half_dct.length();

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, b_buffer) = scratch.split_at_mut(half_len);

            let (left, right) = chunk.split_at(half_len);

            T::modulate_input(a_buffer, b_buffer, left, right, &self.twiddles);

            if a_buffer.len() > 1 {
                self.half_dct.execute_with_scratch(scratch, inner_scratch)?;
            }

            let (a_buffer, b_buffer) = scratch.split_at_mut(half_len);

            T::accumulate(a_buffer, b_buffer, chunk);
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
        let (scratch, inner_scratch) = full_scratch.split_at_mut(self.execution_length);

        let half_len = self.half_dct.length();

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            let (a_buffer, b_buffer) = scratch.split_at_mut(half_len);

            let (left, right) = src.split_at(half_len);

            T::modulate_input(a_buffer, b_buffer, left, right, &self.twiddles);

            if a_buffer.len() > 1 {
                self.half_dct.execute_with_scratch(scratch, inner_scratch)?;
            }

            let (a_buffer, b_buffer) = scratch.split_at_mut(half_len);

            T::accumulate(a_buffer, b_buffer, dst);
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        self.execution_length + self.half_dct_scratch_size
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Dct2Butterfly6;
    use crate::tests::{naive_dct2, naive_dct2_f32};
    use rand::RngExt;

    #[test]
    fn test_radix2_dct() {
        let mut input = vec![0.; 12];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2(&reference_input);
        let bf = NeonDct2MixedRadix2::new(12, Arc::new(Dct2Butterfly6::default())).unwrap();
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

    #[test]
    fn test_radix2_dct_f32() {
        let mut input = vec![0f32; 12];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2_f32(&reference_input);
        let bf = NeonDct2MixedRadix2::new(12, Arc::new(Dct2Butterfly6::default())).unwrap();
        bf.execute(&mut input).unwrap();
        println!("{:?}", input);
        println!("{:?}", reference_input);
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                assert!(
                    (src - r0).abs() < 1e-3,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-3,
                    (src - r0).abs()
                )
            });
    }
}
