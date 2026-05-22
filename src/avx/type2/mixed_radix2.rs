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
use crate::avx::stored::AvxStoreD;
use crate::avx::storef::AvxStoreF;
use crate::util::{DctConstants, DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) trait AvxDct2Modulation: Sized {
    fn modulate_input(
        a_buffer: &mut [Self],
        b_buffer: &mut [Self],
        left: &[Self],
        right: &[Self],
        twiddles: &[Self],
    );
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn avx_modulate_input_f32(
    a_buffer: &mut [f32],
    b_buffer: &mut [f32],
    left: &[f32],
    right: &[f32],
    twiddles: &[f32],
) {
    let mut j = 0usize;
    let total_length = a_buffer.len();

    while j + 32 <= total_length {
        unsafe {
            let a0 = AvxStoreF::load(left.get_unchecked(j..));
            let a1 = AvxStoreF::load(left.get_unchecked(j + 8..));
            let a2 = AvxStoreF::load(left.get_unchecked(j + 16..));
            let a3 = AvxStoreF::load(left.get_unchecked(j + 24..));

            let mut b0 = AvxStoreF::load(right.get_unchecked(total_length - j - 8..));
            let mut b1 = AvxStoreF::load(right.get_unchecked(total_length - j - 16..));
            let mut b2 = AvxStoreF::load(right.get_unchecked(total_length - j - 24..));
            let mut b3 = AvxStoreF::load(right.get_unchecked(total_length - j - 32..));

            let tw0 = AvxStoreF::load(twiddles.get_unchecked(j..));
            let tw1 = AvxStoreF::load(twiddles.get_unchecked(j + 8..));
            let tw2 = AvxStoreF::load(twiddles.get_unchecked(j + 16..));
            let tw3 = AvxStoreF::load(twiddles.get_unchecked(j + 24..));

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
            s1.write(a_buffer.get_unchecked_mut(j + 8..));
            s2.write(a_buffer.get_unchecked_mut(j + 16..));
            s3.write(a_buffer.get_unchecked_mut(j + 24..));

            d0.write(b_buffer.get_unchecked_mut(j..));
            d1.write(b_buffer.get_unchecked_mut(j + 8..));
            d2.write(b_buffer.get_unchecked_mut(j + 16..));
            d3.write(b_buffer.get_unchecked_mut(j + 24..));
        }
        j += 32;
    }

    while j + 8 <= total_length {
        unsafe {
            let a0 = AvxStoreF::load(left.get_unchecked(j..));
            let mut b0 = AvxStoreF::load(right.get_unchecked(total_length - j - 8..));
            let tw0 = AvxStoreF::load(twiddles.get_unchecked(j..));
            b0 = b0.reverse();

            let s0 = a0 + b0;
            let d0 = (a0 - b0) * tw0;

            s0.write(a_buffer.get_unchecked_mut(j..));
            d0.write(b_buffer.get_unchecked_mut(j..));
        }
        j += 8;
    }

    let rem = total_length - j;
    assert!(rem < 8);

    match rem {
        7 => unsafe {
            let a0 = AvxStoreF::load7(left.get_unchecked(j..));
            let mut b0 = AvxStoreF::load7(right.get_unchecked(total_length - j - 7..));
            let tw0 = AvxStoreF::load7(twiddles.get_unchecked(j..));
            b0 = b0.reverse7();

            let s0 = a0 + b0;
            let d0 = (a0 - b0) * tw0;

            s0.write7(a_buffer.get_unchecked_mut(j..));
            d0.write7(b_buffer.get_unchecked_mut(j..));
        },
        6 => unsafe {
            let a0 = AvxStoreF::load6(left.get_unchecked(j..));
            let mut b0 = AvxStoreF::load6(right.get_unchecked(total_length - j - 6..));
            let tw0 = AvxStoreF::load6(twiddles.get_unchecked(j..));
            b0 = b0.reverse6();

            let s0 = a0 + b0;
            let d0 = (a0 - b0) * tw0;

            s0.write6(a_buffer.get_unchecked_mut(j..));
            d0.write6(b_buffer.get_unchecked_mut(j..));
        },
        5 => unsafe {
            let a0 = AvxStoreF::load5(left.get_unchecked(j..));
            let mut b0 = AvxStoreF::load5(right.get_unchecked(total_length - j - 5..));
            let tw0 = AvxStoreF::load5(twiddles.get_unchecked(j..));
            b0 = b0.reverse5();

            let s0 = a0 + b0;
            let d0 = (a0 - b0) * tw0;

            s0.write5(a_buffer.get_unchecked_mut(j..));
            d0.write5(b_buffer.get_unchecked_mut(j..));
        },
        4 => unsafe {
            let a0 = AvxStoreF::load4(left.get_unchecked(j..));
            let mut b0 = AvxStoreF::load4(right.get_unchecked(total_length - j - 4..));
            let tw0 = AvxStoreF::load4(twiddles.get_unchecked(j..));
            b0 = b0.reverse4();

            let s0 = a0 + b0;
            let d0 = (a0 - b0) * tw0;

            s0.write4(a_buffer.get_unchecked_mut(j..));
            d0.write4(b_buffer.get_unchecked_mut(j..));
        },
        3 => unsafe {
            let a0 = AvxStoreF::load3(left.get_unchecked(j..));
            let mut b0 = AvxStoreF::load3(right.get_unchecked(total_length - j - 3..));
            let tw0 = AvxStoreF::load3(twiddles.get_unchecked(j..));

            b0 = b0.reverse3();

            let s0 = a0 + b0;
            let d0 = (a0 - b0) * tw0;

            s0.write3(a_buffer.get_unchecked_mut(j..));
            d0.write3(b_buffer.get_unchecked_mut(j..));
        },
        2 => unsafe {
            let a0 = AvxStoreF::load2(left.get_unchecked(j..));
            let mut b0 = AvxStoreF::load2(right.get_unchecked(total_length - j - 2..));
            let tw0 = AvxStoreF::load2(twiddles.get_unchecked(j..));

            b0 = b0.reverse2();

            let s0 = a0 + b0;
            let d0 = (a0 - b0) * tw0;

            s0.write2(a_buffer.get_unchecked_mut(j..));
            d0.write2(b_buffer.get_unchecked_mut(j..));
        },
        1 => unsafe {
            let a0 = AvxStoreF::load1(left.get_unchecked(j..));
            let b0 = AvxStoreF::load1(right.get_unchecked(total_length - j - 1..));
            let tw0 = AvxStoreF::load1(twiddles.get_unchecked(j..));

            let s0 = a0 + b0;
            let d0 = (a0 - b0) * tw0;

            s0.write1(a_buffer.get_unchecked_mut(j..));
            d0.write1(b_buffer.get_unchecked_mut(j..));
        },
        _ => {}
    }
}

impl AvxDct2Modulation for f32 {
    #[inline(always)]
    fn modulate_input(
        a_buffer: &mut [Self],
        b_buffer: &mut [Self],
        left: &[Self],
        right: &[Self],
        twiddles: &[Self],
    ) {
        unsafe {
            avx_modulate_input_f32(a_buffer, b_buffer, left, right, twiddles);
        }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
fn modulate_input_f64(
    a_buffer: &mut [f64],
    b_buffer: &mut [f64],
    left: &[f64],
    right: &[f64],
    twiddles: &[f64],
) {
    for ((((dst_l, dst_r), &l), &r), &twiddle) in a_buffer
        .iter_mut()
        .zip(b_buffer.iter_mut())
        .zip(left.iter())
        .zip(right.iter().rev())
        .zip(twiddles.iter())
    {
        *dst_l = l + r;
        *dst_r = (l - r) * twiddle;
    }
}

impl AvxDct2Modulation for f64 {
    #[inline(always)]
    fn modulate_input(
        a_buffer: &mut [Self],
        b_buffer: &mut [Self],
        left: &[Self],
        right: &[Self],
        twiddles: &[Self],
    ) {
        unsafe {
            modulate_input_f64(a_buffer, b_buffer, left, right, twiddles);
        }
    }
}

pub(crate) struct AvxDct2MixedRadix2<T> {
    twiddles: Vec<T>,
    half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    inner_dct_scratch_size: usize,
    execution_length: usize,
}

impl<T: DctSample> AvxDct2MixedRadix2<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<AvxDct2MixedRadix2<T>, PxdctError> {
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

        let inner_dct_scratch_size = half_dct.scratch_size();

        Ok(AvxDct2MixedRadix2 {
            half_dct,
            twiddles,
            execution_length: len,
            inner_dct_scratch_size,
        })
    }
}

pub(crate) trait MixedRadix2Differences<T> {
    fn accumulate(a_buffer: &[T], b_buffer: &mut [T], output: &mut [T]);
}

impl<T: DctSample + AvxDct2Modulation + MixedRadix2Differences<T>> PxdctExecutor<T>
    for AvxDct2MixedRadix2<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        unsafe { self.execute_impl(data, &mut scratch) }
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data, scratch) }
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        unsafe { self.execute_into_impl(input, output, &mut scratch) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output, scratch) }
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        self.execution_length + self.inner_dct_scratch_size
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
fn accumulate_f32(a_buffer: &[f32], b_buffer: &mut [f32], output: &mut [f32]) {
    b_buffer[0] *= f32::HALF;

    let mut j = 0usize;
    let mut dst_idx = 0usize;

    let mut last_odd = 0.;

    if j + 8 < a_buffer.len() {
        let mut differences = AvxStoreF::set_values8(0., 0., 0., 0., 0., 0., 0., 0.);

        // [1 -1 1 -1]
        let sign = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

        while j + 8 < a_buffer.len() {
            differences = differences.xor(sign);
            let evens = AvxStoreF::load(unsafe { a_buffer.get_unchecked(j..) });
            let odds = AvxStoreF::load(unsafe { b_buffer.get_unchecked(j..) });

            let diffs = odds.prefix_differences(sign);
            differences = diffs - differences;

            // differences = odds - differences;
            let zipped = AvxStoreF::zip(evens, differences);
            differences = differences.broadcast_last();
            unsafe {
                zipped[0].write(output.get_unchecked_mut(dst_idx..));
                zipped[1].write(output.get_unchecked_mut(dst_idx + 8..));
            }
            dst_idx += 16;
            j += 8;
        }

        last_odd = differences.last();
    }

    let chunk = &mut output[dst_idx..];
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

impl MixedRadix2Differences<f32> for f32 {
    fn accumulate(a_buffer: &[f32], b_buffer: &mut [f32], output: &mut [f32]) {
        unsafe {
            accumulate_f32(a_buffer, b_buffer, output);
        }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
fn accumulate_f64(a_buffer: &[f64], b_buffer: &mut [f64], output: &mut [f64]) {
    b_buffer[0] *= f64::HALF;

    let mut j = 0usize;
    let mut dst_idx = 0usize;

    let mut last_odd = 0.;

    if j + 4 < a_buffer.len() {
        let mut differences = AvxStoreD::set_values(0., 0., 0., 0.);

        // [1 -1 1 -1]
        let sign = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

        while j + 4 < a_buffer.len() {
            differences = differences.xor(sign);
            let evens = AvxStoreD::load(unsafe { a_buffer.get_unchecked(j..) });
            let odds = AvxStoreD::load(unsafe { b_buffer.get_unchecked(j..) });

            let diffs = odds.prefix_differences(sign);
            differences = diffs - differences;

            // differences = odds - differences;
            let zipped = AvxStoreD::zip(evens, differences);
            differences = differences.broadcast_last();
            unsafe {
                zipped[0].write(output.get_unchecked_mut(dst_idx..));
                zipped[1].write(output.get_unchecked_mut(dst_idx + 4..));
            }
            dst_idx += 8;
            j += 4;
        }

        last_odd = differences.last();
    }

    let chunk = &mut output[dst_idx..];
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

impl MixedRadix2Differences<f64> for f64 {
    fn accumulate(a_buffer: &[f64], b_buffer: &mut [f64], output: &mut [f64]) {
        unsafe {
            accumulate_f64(a_buffer, b_buffer, output);
        }
    }
}

impl<T: DctSample + AvxDct2Modulation + MixedRadix2Differences<T>> AvxDct2MixedRadix2<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
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

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Dct2Butterfly6;
    use crate::tests::{naive_dct2, naive_dct2_f32};
    use crate::type2::prime_butterflies::Dct2Butterfly17;
    use crate::util::has_valid_avx;

    #[test]
    fn test_radix2_dct() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0.; 34];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32;
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2_f32(&reference_input);
        let bf = AvxDct2MixedRadix2::new(34, Arc::new(Dct2Butterfly17::default())).unwrap();
        bf.execute(&mut input).unwrap();
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

    #[test]
    fn test_radix2_dct_f64() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0.; 12];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f64;
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2(&reference_input);
        let bf = AvxDct2MixedRadix2::new(12, Arc::new(Dct2Butterfly6::default())).unwrap();
        bf.execute(&mut input).unwrap();
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
