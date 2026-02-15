/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use crate::avx::dct2::dct2_split_radix_rotation_twiddles_avxf;
use crate::avx::storef::AvxStoreF;
use crate::avx::util::{boring_avx_split_radix, fma};
use crate::bidirectional::BidirectionalStore;
use crate::util::{DctConstants, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use std::sync::Arc;

pub(crate) struct AvxSplitRadixDct3f {
    twiddles: Vec<AvxStoreF>,
    half_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    quarter_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    execution_length: usize,
    inner_scratch_size: usize,
    half_dct_scratch_size: usize,
    quarter_dct_scratch_size: usize,
}

impl AvxSplitRadixDct3f {
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<AvxSplitRadixDct3f, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half for Split-Radix DCT-III"
        );

        let half_dct_scratch_size = half_dct.scratch_size();
        let quarter_dct_scratch_size = quarter_dct.scratch_size();

        Ok(AvxSplitRadixDct3f {
            twiddles: unsafe { dct2_split_radix_rotation_twiddles_avxf(len) },
            half_dct,
            quarter_dct,
            execution_length: len,
            inner_scratch_size: len + half_dct_scratch_size.max(quarter_dct_scratch_size),
            half_dct_scratch_size,
            quarter_dct_scratch_size,
        })
    }
}

boring_avx_split_radix!(AvxSplitRadixDct3f, f32);

impl AvxSplitRadixDct3f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        let len = self.length();
        let half_len = len / 2;
        let quarter_len = len / 4;

        let (scratch, r) = scratch.split_at_mut(self.execution_length);
        // divide the output into 3 sub-lists to use for our inner DCTs, one of size N/2 and two of size N/4
        let (recursive_input_evens, recursive_input_odds) = scratch.split_at_mut(half_len);
        let (recursive_input_n1, recursive_input_n3) =
            recursive_input_odds.split_at_mut(quarter_len);

        unsafe {
            *recursive_input_evens.get_unchecked_mut(0) = data[0];
            *recursive_input_evens.get_unchecked_mut(1) = data[2];
            *recursive_input_n1.get_unchecked_mut(0) = data[1] * f32::TWO;
            *recursive_input_n3.get_unchecked_mut(0) = data[len - 1] * f32::TWO;
        }

        let conj_odd = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

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

        let (half_dct_scratch, _) = r.split_at_mut(self.half_dct_scratch_size);

        self.half_dct
            .execute_with_scratch(recursive_input_evens, half_dct_scratch)?;

        let (quarter_dct_scratch, _) = r.split_at_mut(self.quarter_dct_scratch_size);

        self.quarter_dct
            .execute_with_scratch(recursive_input_n1, quarter_dct_scratch)?;
        self.quarter_dct
            .execute_with_scratch(recursive_input_n3, quarter_dct_scratch)?;

        let mut i = 0usize;
        let mut tw_idx = 0usize;

        while i + 8 <= quarter_len {
            const S: usize = 8;
            let sine_value =
                AvxStoreF::load(unsafe { recursive_input_n3.get_unchecked(i..) }).xor(conj_odd);
            let cos_value = AvxStoreF::load(unsafe { recursive_input_n1.get_unchecked(i..) });
            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tw_idx) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tw_idx + 1) };

            let lower_dct4 = fma(cos_value, twiddle_re, sine_value * twiddle_im);
            let upper_dct4 = fma(cos_value, twiddle_im, -sine_value * twiddle_re);

            unsafe {
                let lower_dct3 = AvxStoreF::load(recursive_input_evens.get_unchecked(i..));
                let upper_dct3 =
                    AvxStoreF::load(recursive_input_evens.get_unchecked(half_len - i - S..))
                        .reverse();

                let v0 = lower_dct3 + lower_dct4;
                let v1 = lower_dct3 - lower_dct4;
                let v2 = upper_dct3 + upper_dct4;
                let v3 = upper_dct3 - upper_dct4;
                v0.write(data.slice_from_mut(i..));
                v1.reverse().write(data.slice_from_mut(len - i - S..));
                v2.reverse().write(data.slice_from_mut(half_len - i - S..));
                v3.write(data.slice_from_mut(half_len + i..));
            }

            tw_idx += 2;
            i += 8;
        }

        let rem = quarter_len - i;
        if rem == 4 {
            const S: usize = 4;
            let sine_value =
                AvxStoreF::load4(unsafe { recursive_input_n3.get_unchecked(i..) }).xor(conj_odd);
            let cos_value = AvxStoreF::load4(unsafe { recursive_input_n1.get_unchecked(i..) });
            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tw_idx) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tw_idx + 1) };

            let lower_dct4 = fma(cos_value, twiddle_re, sine_value * twiddle_im);
            let upper_dct4 = fma(cos_value, twiddle_im, -sine_value * twiddle_re);

            unsafe {
                let lower_dct3 = AvxStoreF::load4(recursive_input_evens.get_unchecked(i..));
                let upper_dct3 =
                    AvxStoreF::load4(recursive_input_evens.get_unchecked(half_len - i - S..))
                        .reverse4();

                let v0 = lower_dct3 + lower_dct4;
                let v1 = lower_dct3 - lower_dct4;
                let v2 = upper_dct3 + upper_dct4;
                let v3 = upper_dct3 - upper_dct4;
                v0.write4(data.slice_from_mut(i..));
                v1.reverse4().write4(data.slice_from_mut(len - i - S..));
                v2.reverse4()
                    .write4(data.slice_from_mut(half_len - i - S..));
                v3.write4(data.slice_from_mut(half_len + i..));
            }
        } else if rem == 2 {
            const S: usize = 2;
            let sine_value =
                AvxStoreF::load2(unsafe { recursive_input_n3.get_unchecked(i..) }).xor(conj_odd);
            let cos_value = AvxStoreF::load2(unsafe { recursive_input_n1.get_unchecked(i..) });
            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tw_idx) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tw_idx + 1) };

            let lower_dct4 = fma(cos_value, twiddle_re, sine_value * twiddle_im);
            let upper_dct4 = fma(cos_value, twiddle_im, -sine_value * twiddle_re);

            unsafe {
                let lower_dct3 = AvxStoreF::load2(recursive_input_evens.get_unchecked(i..));
                let upper_dct3 =
                    AvxStoreF::load2(recursive_input_evens.get_unchecked(half_len - i - S..))
                        .reverse2();

                let v0 = lower_dct3 + lower_dct4;
                let v1 = lower_dct3 - lower_dct4;
                let v2 = upper_dct3 + upper_dct4;
                let v3 = upper_dct3 - upper_dct4;
                v0.write2(data.slice_from_mut(i..));
                v1.reverse().write2(data.slice_from_mut(len - i - S..));
                v2.reverse().write2(data.slice_from_mut(half_len - i - S..));
                v3.write2(data.slice_from_mut(half_len + i..));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct3::{Dct3Butterfly8, Dct3Butterfly16};
    use crate::tests::naive_dct3_f32;
    use crate::util::has_valid_avx;
    use rand::Rng;

    #[test]
    fn test_split_dct3() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0.; 32];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct3_f32(&reference_input);
        let bf = AvxSplitRadixDct3f::new(
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
                    (src - r0).abs() < 1e-3,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-3,
                    (src - r0).abs()
                )
            });
    }
}
