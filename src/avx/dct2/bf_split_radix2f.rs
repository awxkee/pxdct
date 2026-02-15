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
use crate::avx::util::fma;
use crate::avx::{AvxDct2Butterfly8, AvxDct2Butterfly16};
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::factory_dct2::Dct2Factory;
use crate::twiddles::compute_twiddle;
use crate::{PxdctError, PxdctExecutor};
use num_traits::Zero;
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_split_radix_rotation_twiddles_avxf(len: usize) -> Vec<AvxStoreF> {
    let twiddles_len = len / 4;
    let simd_groups = len.div_ceil(8);

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 2);

    let mut uk = 0usize;
    while uk + 8 <= twiddles_len {
        let mut array_re = [0.; 8];
        let mut array_im = [0.; 8];
        for ki in 0..8 {
            let i = uk + ki;
            let twiddle = compute_twiddle::<f32>(2 * i + 1, len * 4).conj();
            array_re[ki] = twiddle.re;
            array_im[ki] = twiddle.im;
        }

        twiddles.push(AvxStoreF::load(&array_re));
        twiddles.push(AvxStoreF::load(&array_im));

        uk += 8;
    }

    let remainder = twiddles_len - (twiddles_len / 8) * 8;
    if remainder > 0 {
        let mut array_re = [0.; 8];
        let mut array_im = [0.; 8];
        for i in 0..remainder {
            let i = uk + i;
            let twiddle = compute_twiddle::<f32>(2 * i + 1, len * 4).conj();
            array_re[i] = twiddle.re;
            array_im[i] = twiddle.im;
        }

        twiddles.push(AvxStoreF::load(&array_re));
        twiddles.push(AvxStoreF::load(&array_im));
    }

    twiddles
}

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly32f {
    bf8: AvxDct2Butterfly8<f32>,
    bf16: AvxDct2Butterfly16<f32>,
    twiddles: [AvxStoreF; 2],
}

impl Default for AvxDct2Butterfly32f {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avxf(32) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf8: AvxDct2Butterfly8::default(),
            bf16: AvxDct2Butterfly16::default(),
        }
    }
}

impl AvxDct2Butterfly32f {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f32; 16],
        input_dct4_even: &mut [f32; 8],
        input_dct4_odd: &mut [f32; 8],
    ) {
        unsafe {
            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

            let conj_odd = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

            {
                let i = 0;
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreF::load(data.slice_from(i * 4..));
                let input_top = AvxStoreF::load(data.slice_from(32 - i * 4 - 8..)).reverse();

                let input_half_bottom =
                    AvxStoreF::load(data.slice_from(16 - i * 4 - 8..)).reverse();
                let input_half_top = AvxStoreF::load(data.slice_from(16 + i * 4..));

                //prepare the inner DCT2
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 4..));
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(16 - i * 4 - 8..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 4..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(8 - i * 4 - 8..));
            }

            self.bf16.exec(&mut InPlaceStore::new(input_dct2));
            self.bf8.exec(&mut InPlaceStore::new(input_dct4_even));
            self.bf8.exec(&mut InPlaceStore::new(input_dct4_odd));

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            for i in 1..8 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 8) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(8 - i)
                } else {
                    *input_dct4_odd.get_unchecked(8 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[32 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl AvxDct2Butterfly32f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(32) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f32::default(); 16];
        let mut input_dct4_even = [f32::default(); 8];
        let mut input_dct4_odd = [f32::default(); 8];

        for chunk in data.chunks_exact_mut(32) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4_even,
                &mut input_dct4_odd,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 32);

        let mut input_dct2 = [f32::default(); 16];
        let mut input_dct4_even = [f32::default(); 8];
        let mut input_dct4_odd = [f32::default(); 8];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(32).zip(output.chunks_exact_mut(32)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4_even,
                &mut input_dct4_odd,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f32> for AvxDct2Butterfly32f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f32], _: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _: &mut [f32],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        32
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly64f {
    bf32: AvxDct2Butterfly32f,
    bf16: AvxDct2Butterfly16<f32>,
    twiddles: [AvxStoreF; 4],
}

impl Default for AvxDct2Butterfly64f {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avxf(64) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf16: AvxDct2Butterfly16::default(),
            bf32: AvxDct2Butterfly32f::default(),
        }
    }
}

impl AvxDct2Butterfly64f {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f32; 32],
        input_dct4_even: &mut [f32; 16],
        input_dct4_odd: &mut [f32; 16],
        input_dct21: &mut [f32; 16],
        input_dct4_even1: &mut [f32; 8],
        input_dct4_odd1: &mut [f32; 8],
    ) {
        unsafe {
            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
            let conj_odd = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

            for i in 0..2 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreF::load(data.slice_from(i * 8..));
                let input_top = AvxStoreF::load(data.slice_from(64 - i * 8 - 8..)).reverse();

                let input_half_bottom =
                    AvxStoreF::load(data.slice_from(32 - i * 8 - 8..)).reverse();
                let input_half_top = AvxStoreF::load(data.slice_from(32 + i * 8..));

                //prepare the inner DCT2
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 8..));

                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(32 - i * 8 - 8..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 8..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(16 - i * 8 - 8..));
            }

            self.bf32.exec(
                &mut InPlaceStore::new(input_dct2),
                input_dct21,
                input_dct4_even1,
                input_dct4_odd1,
            );
            self.bf16.exec(&mut InPlaceStore::new(input_dct4_even));
            self.bf16.exec(&mut InPlaceStore::new(input_dct4_odd));

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let mut i = 1usize;

            while i + 8 <= 16 {
                let dct4_cos_output_v = AvxStoreF::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v = AvxStoreF::load(input_dct4_odd.get_unchecked(16 - i - 7..))
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

            {
                let dct4_cos_output_v = AvxStoreF::load7(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreF::load7(input_dct4_odd.get_unchecked(16 - i - 6..))
                        .reverse7()
                        .xor(conj_odd);

                let [dct4_cos_output0, dct4_cos_output1] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, mut dct4_sin_output1] =
                    dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                let input0 = AvxStoreF::load(input_dct2.get_unchecked(i * 2..));
                let input1 = AvxStoreF::load6(input_dct2.get_unchecked(i * 2 + 8..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;
                let sums1 = dct4_cos_output1 + dct4_sin_output1;

                let zipped0 = sums0.zip(input0);
                let zipped1 = sums1.zip(input1);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 7..));
                zipped1[0].write(data.slice_from_mut(i * 4 + 15..));
                zipped1[1].write4(data.slice_from_mut(i * 4 + 23..));
            }

            data[64 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl AvxDct2Butterfly64f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(64) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f32::default(); 32];
        let mut input_dct4_even = [f32::default(); 16];
        let mut input_dct4_odd = [f32::default(); 16];

        let mut input_dct21 = [f32::default(); 16];
        let mut input_dct4_even1 = [f32::default(); 8];
        let mut input_dct4_odd1 = [f32::default(); 8];

        for chunk in data.chunks_exact_mut(64) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4_even,
                &mut input_dct4_odd,
                &mut input_dct21,
                &mut input_dct4_even1,
                &mut input_dct4_odd1,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 64);

        let mut input_dct2 = [f32::default(); 32];
        let mut input_dct4_even = [f32::default(); 16];
        let mut input_dct4_odd = [f32::default(); 16];

        let mut input_dct21 = [f32::default(); 16];
        let mut input_dct4_even1 = [f32::default(); 8];
        let mut input_dct4_odd1 = [f32::default(); 8];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(64).zip(output.chunks_exact_mut(64)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4_even,
                &mut input_dct4_odd,
                &mut input_dct21,
                &mut input_dct4_even1,
                &mut input_dct4_odd1,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f32> for AvxDct2Butterfly64f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f32], _: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _: &mut [f32],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        64
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub(crate) struct AvxDct2Butterfly128f {
    bf64: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    bf32: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    twiddles: [AvxStoreF; 8],
}

impl Default for AvxDct2Butterfly128f {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avxf(128) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf64: f32::dct2_butterfly64(),
            bf32: f32::dct2_butterfly32(),
        }
    }
}

impl AvxDct2Butterfly128f {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f32; 64],
        input_dct4: &mut [f32; 64],
    ) {
        unsafe {
            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(32);

            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
            let conj_odd = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

            for i in 0..4 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreF::load(data.slice_from(i * 8..));
                let input_top = AvxStoreF::load(data.slice_from(128 - i * 8 - 8..)).reverse();

                let input_half_bottom =
                    AvxStoreF::load(data.slice_from(64 - i * 8 - 8..)).reverse();
                let input_half_top = AvxStoreF::load(data.slice_from(64 + i * 8..));

                //prepare the inner DCT2
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 8..));
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(64 - i * 8 - 8..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 8..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(32 - i * 8 - 8..));
            }

            _ = self.bf64.execute(input_dct2);
            _ = self.bf32.execute(input_dct4);

            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(32);

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let conj_odd = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

            let mut i = 1usize;
            while i + 8 <= 32 {
                let dct4_cos_output_v = AvxStoreF::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v = AvxStoreF::load(input_dct4_odd.get_unchecked(32 - i - 7..))
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

            while i + 4 <= 32 {
                let dct4_cos_output_v = AvxStoreF::load4(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreF::load4(input_dct4_odd.get_unchecked(32 - i - 3..))
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

            for i in i..32 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 32) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(32 - i)
                } else {
                    *input_dct4_odd.get_unchecked(32 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[128 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl AvxDct2Butterfly128f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(128) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f32::zero(); 64];
        let mut input_dct4 = [f32::zero(); 64];

        for chunk in data.chunks_exact_mut(128) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 128);

        let mut input_dct2 = [f32::zero(); 64];
        let mut input_dct4 = [f32::zero(); 64];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(128).zip(output.chunks_exact_mut(128)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f32> for AvxDct2Butterfly128f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f32], _: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _: &mut [f32],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        128
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub(crate) struct AvxDct2Butterfly256f {
    bf128: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    bf64: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    twiddles: [AvxStoreF; 16],
}

impl Default for AvxDct2Butterfly256f {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avxf(256) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf64: f32::dct2_butterfly64(),
            bf128: f32::dct2_butterfly128(),
        }
    }
}

impl AvxDct2Butterfly256f {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f32; 128],
        input_dct4: &mut [f32; 128],
    ) {
        unsafe {
            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(64);

            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

            let conj_odd = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

            for i in 0..8 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreF::load(data.slice_from(i * 8..));
                let input_top = AvxStoreF::load(data.slice_from(256 - i * 8 - 8..)).reverse();

                let input_half_bottom =
                    AvxStoreF::load(data.slice_from(128 - i * 8 - 8..)).reverse();
                let input_half_top = AvxStoreF::load(data.slice_from(128 + i * 8..));

                //prepare the inner DCT2
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 8..));
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(128 - i * 8 - 8..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 8..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(64 - i * 8 - 8..));
            }

            _ = self.bf128.execute(input_dct2);
            _ = self.bf64.execute(input_dct4);

            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(64);

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let mut i = 1usize;
            while i + 8 <= 64 {
                let dct4_cos_output_v = AvxStoreF::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v = AvxStoreF::load(input_dct4_odd.get_unchecked(64 - i - 7..))
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

            while i + 4 <= 64 {
                let dct4_cos_output_v = AvxStoreF::load4(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreF::load4(input_dct4_odd.get_unchecked(64 - i - 3..))
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

            for i in i..64 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 64) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(64 - i)
                } else {
                    *input_dct4_odd.get_unchecked(64 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[256 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl AvxDct2Butterfly256f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(256) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f32::zero(); 128];
        let mut input_dct4 = [f32::zero(); 128];

        for chunk in data.chunks_exact_mut(256) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 256);

        let mut input_dct2 = [f32::zero(); 128];
        let mut input_dct4 = [f32::zero(); 128];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(256).zip(output.chunks_exact_mut(256)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f32> for AvxDct2Butterfly256f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f32], _: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _: &mut [f32],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        256
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub(crate) struct AvxDct2Butterfly512f {
    bf128: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    bf256: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    twiddles: [AvxStoreF; 32],
}

impl Default for AvxDct2Butterfly512f {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avxf(512) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf256: f32::dct2_butterfly256(),
            bf128: f32::dct2_butterfly128(),
        }
    }
}

impl AvxDct2Butterfly512f {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f32; 256],
        input_dct4: &mut [f32; 256],
    ) {
        unsafe {
            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(128);

            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

            let conj_odd = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

            for i in 0..16 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreF::load(data.slice_from(i * 8..));
                let input_top = AvxStoreF::load(data.slice_from(512 - i * 8 - 8..)).reverse();

                let input_half_bottom =
                    AvxStoreF::load(data.slice_from(256 - i * 8 - 8..)).reverse();
                let input_half_top = AvxStoreF::load(data.slice_from(256 + i * 8..));

                //prepare the inner DCT2
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 8..));
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(256 - i * 8 - 8..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 8..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(128 - i * 8 - 8..));
            }

            _ = self.bf256.execute(input_dct2);
            _ = self.bf128.execute(input_dct4);

            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(128);

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let mut i = 1usize;
            while i + 8 <= 128 {
                let dct4_cos_output_v = AvxStoreF::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreF::load(input_dct4_odd.get_unchecked(128 - i - 7..))
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

            while i + 4 <= 128 {
                let dct4_cos_output_v = AvxStoreF::load4(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreF::load4(input_dct4_odd.get_unchecked(128 - i - 3..))
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

            for i in i..128 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 128) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(128 - i)
                } else {
                    *input_dct4_odd.get_unchecked(128 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[512 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl AvxDct2Butterfly512f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(512) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f32::zero(); 256];
        let mut input_dct4 = [f32::zero(); 256];

        for chunk in data.chunks_exact_mut(512) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 512);

        let mut input_dct2 = [f32::zero(); 256];
        let mut input_dct4 = [f32::zero(); 256];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(512).zip(output.chunks_exact_mut(512)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f32> for AvxDct2Butterfly512f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f32], _: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _: &mut [f32],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        512
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::avx::dct2_bf_power2::gen_test_avx_butterfly;
    use crate::tests::naive_dct2_f32;

    gen_test_avx_butterfly!(
        test_avx_bf32f,
        AvxDct2Butterfly32f,
        32,
        1e-3,
        naive_dct2_f32
    );
    gen_test_avx_butterfly!(
        test_avx_bf64f,
        AvxDct2Butterfly64f,
        64,
        1e-3,
        naive_dct2_f32
    );
    gen_test_avx_butterfly!(
        test_avx_bf128f,
        AvxDct2Butterfly128f,
        128,
        1e-3,
        naive_dct2_f32
    );
    gen_test_avx_butterfly!(
        test_avx_bf256f,
        AvxDct2Butterfly256f,
        256,
        1e-2,
        naive_dct2_f32
    );
    gen_test_avx_butterfly!(
        test_avx_bf512f,
        AvxDct2Butterfly512f,
        512,
        1e-2,
        naive_dct2_f32
    );
}
