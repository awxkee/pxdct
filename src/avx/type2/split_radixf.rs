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
use crate::avx::storef::AvxStoreF;
use crate::avx::type2::bf_split_radix2f::dct2_split_radix_rotation_twiddles_avxf;
use crate::avx::util::{boring_avx_split_radix, fma};
use crate::bidirectional::BidirectionalStore;
use crate::util::{try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use std::ops::Neg;
use std::sync::Arc;

pub(crate) struct AvxSplitRadixDct2f {
    twiddles: Vec<AvxStoreF>,
    half_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    quarter_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    execution_length: usize,
    inner_scratch_size: usize,
    half_dct_scratch_size: usize,
    quarter_dct_scratch_size: usize,
}

impl AvxSplitRadixDct2f {
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<AvxSplitRadixDct2f, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half"
        );

        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avxf(len) };

        let half_dct_scratch_size = half_dct.scratch_size();
        let quarter_dct_scratch_size = quarter_dct.scratch_size();

        Ok(AvxSplitRadixDct2f {
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

boring_avx_split_radix!(AvxSplitRadixDct2f, f32);

impl AvxSplitRadixDct2f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        let (scratch, r) = scratch.split_at_mut(self.execution_length);

        let len = self.length();
        let half_len = len / 2;
        let quarter_len = len / 4;
        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
        let (input_dct2, input_dct4) = scratch.split_at_mut(half_len);
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(quarter_len);

        let conj_odd = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

        for (i, twiddle_pack) in self.twiddles.as_chunks::<2>().0.iter().enumerate() {
            let twiddle_re = twiddle_pack[0];
            let twiddle_im = twiddle_pack[1];
            let input_bottom = AvxStoreF::load(data.slice_from(i * 8..));
            let input_top = AvxStoreF::load(data.slice_from(len - i * 8 - 8..)).reverse();

            let input_half_bottom =
                AvxStoreF::load(data.slice_from(half_len - i * 8 - 8..)).reverse();
            let input_half_top = AvxStoreF::load(data.slice_from(half_len + i * 8..));

            //prepare the inner DCT2
            let q = input_top + input_bottom;
            unsafe {
                q.write(input_dct2.get_unchecked_mut(i * 8..));
            }

            let dq = input_half_bottom + input_half_top;
            unsafe {
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(half_len - i * 8 - 8..));
            }

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
            let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

            unsafe {
                cos_input.write(input_dct4_even.get_unchecked_mut(i * 8..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(quarter_len - i * 8 - 8..));
            }
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

            let mut i = 1usize;
            while i + 8 <= quarter_len {
                let dct4_cos_output_v = AvxStoreF::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreF::load(input_dct4_odd.get_unchecked(quarter_len - i - 7..))
                        .reverse()
                        .xor(conj_odd);

                let [dct4_cos_output0, dct4_cos_output1] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, mut dct4_sin_output1] =
                    dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                let input0 = AvxStoreF::load(input_dct2.get_unchecked(i * 2..));
                let input1 = AvxStoreF::load(input_dct2.get_unchecked(i * 2 + 8..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;
                let sums1 = dct4_cos_output1 + dct4_sin_output1;

                let zipped0 = sums0.zip(input0);
                let zipped1 = sums1.zip(input1);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 7..));
                zipped1[0].write(data.slice_from_mut(i * 4 + 15..));
                zipped1[1].write(data.slice_from_mut(i * 4 + 23..));
                i += 8;
            }

            while i + 4 <= quarter_len {
                let dct4_cos_output_v = AvxStoreF::load4(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreF::load4(input_dct4_odd.get_unchecked(quarter_len - i - 3..))
                        .reverse4()
                        .xor(conj_odd);

                let [dct4_cos_output0, _] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, _] = dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);

                let input0 = AvxStoreF::load(input_dct2.get_unchecked(i * 2..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;

                let zipped0 = sums0.zip(input0);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 7..));
                i += 4;
            }

            for i in i..quarter_len {
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

pub(crate) struct AvxSplitRadixDst2f {
    split_radix_dct2: AvxSplitRadixDct2f,
    execution_length: usize,
}

impl AvxSplitRadixDst2f {
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<AvxSplitRadixDst2f, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half in DST-II"
        );

        Ok(AvxSplitRadixDst2f {
            split_radix_dct2: AvxSplitRadixDct2f::new(len, half_dct, quarter_dct)?,
            execution_length: len,
        })
    }
}

impl AvxSplitRadixDst2f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32], scratch: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let scratch = validate_scratch!(scratch, self.scratch_size());

        for chunk in data.chunks_exact_mut(self.execution_length) {
            for dst in chunk.chunks_exact_mut(2) {
                dst[1] = dst[1].neg();
            }

            self.split_radix_dct2.execute_with_scratch(chunk, scratch)?;

            chunk.reverse();
        }

        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
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
}

impl PxdctExecutor<f32> for AvxSplitRadixDst2f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![f32::default(); self.scratch_size()];
        unsafe { self.execute_impl(data, &mut scratch) }
    }

    fn execute_with_scratch(
        &self,
        data: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data, scratch) }
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![f32::default(); self.scratch_size()];
        self.execute_into_with_scratch(input, output, &mut scratch)
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output, scratch) }
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
    use crate::tests::naive_dct2_f32;
    use crate::type2::power2_butterflies::{Dct2Butterfly8, Dct2Butterfly16};
    use crate::util::has_valid_avx;
    use rand::RngExt;

    #[test]
    fn test_split_dct2() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0f32; 32];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2_f32(&reference_input);
        let bf = AvxSplitRadixDct2f::new(
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
                    (src - r0).abs() < 1e-3,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-3,
                    (src - r0).abs()
                )
            });
    }
}
