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
use crate::avx::stored::AvxStoreD;
use crate::avx::util::fma;
use crate::avx::{AvxDct2Butterfly8, AvxDct2Butterfly16};
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::factory_dct2::Dct2Factory;
use crate::twiddles::compute_twiddle;
use crate::{PxdctError, PxdctExecutor};
use num_traits::Zero;
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_split_radix_rotation_twiddles_avx(len: usize) -> Vec<AvxStoreD> {
    let twiddles_len = len / 4;
    let simd_groups = len.div_ceil(4);

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 2);

    let mut uk = 0usize;
    while uk + 4 <= twiddles_len {
        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for ki in 0..4 {
            let i = uk + ki;
            let twiddle = compute_twiddle::<f64>(2 * i + 1, len * 4).conj();
            array_re[ki] = twiddle.re;
            array_im[ki] = twiddle.im;
        }

        twiddles.push(AvxStoreD::load(&array_re));
        twiddles.push(AvxStoreD::load(&array_im));

        uk += 4;
    }

    let remainder = twiddles_len - (twiddles_len / 4) * 4;
    if remainder > 0 {
        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for i in 0..remainder {
            let i = uk + i;
            let twiddle = compute_twiddle::<f64>(2 * i + 1, len * 4).conj();
            array_re[i] = twiddle.re;
            array_im[i] = twiddle.im;
        }

        twiddles.push(AvxStoreD::load(&array_re));
        twiddles.push(AvxStoreD::load(&array_im));
    }

    twiddles
}

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly32d {
    bf8: AvxDct2Butterfly8<f64>,
    bf16: AvxDct2Butterfly16<f64>,
    twiddles: [AvxStoreD; 4],
}

impl Default for AvxDct2Butterfly32d {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avx(32) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf8: AvxDct2Butterfly8::default(),
            bf16: AvxDct2Butterfly16::default(),
        }
    }
}

impl AvxDct2Butterfly32d {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f64; 16],
        input_dct4_even: &mut [f64; 8],
        input_dct4_odd: &mut [f64; 8],
    ) {
        unsafe {
            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

            let conj_odd = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

            for i in 0..2 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreD::load(data.slice_from(i * 4..));
                let input_top = AvxStoreD::load(data.slice_from(32 - i * 4 - 4..)).reverse();

                let input_half_bottom =
                    AvxStoreD::load(data.slice_from(16 - i * 4 - 4..)).reverse();
                let input_half_top = AvxStoreD::load(data.slice_from(16 + i * 4..));

                //prepare the inner DCT2
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 4..));
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(16 - i * 4 - 4..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 4..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(8 - i * 4 - 4..));
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

impl AvxDct2Butterfly32d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(32) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::default(); 16];
        let mut input_dct4_even = [f64::default(); 8];
        let mut input_dct4_odd = [f64::default(); 8];

        for chunk in data.as_chunks_mut::<32>().0.iter_mut() {
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
    fn execute_into_impl(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 32);

        let mut input_dct2 = [f64::default(); 16];
        let mut input_dct4_even = [f64::default(); 8];
        let mut input_dct4_odd = [f64::default(); 8];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<32>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<32>().0.iter_mut())
        {
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

impl PxdctExecutor<f64> for AvxDct2Butterfly32d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f64], _: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
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
pub(crate) struct AvxDct2Butterfly64d {
    bf32: AvxDct2Butterfly32d,
    bf16: AvxDct2Butterfly16<f64>,
    twiddles: [AvxStoreD; 8],
}

impl Default for AvxDct2Butterfly64d {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avx(64) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf16: AvxDct2Butterfly16::default(),
            bf32: AvxDct2Butterfly32d::default(),
        }
    }
}

impl AvxDct2Butterfly64d {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f64; 32],
        input_dct4_even: &mut [f64; 16],
        input_dct4_odd: &mut [f64; 16],
        input_dct2_32: &mut [f64; 16],
        input_dct4_even_32: &mut [f64; 8],
        input_dct4_odd_32: &mut [f64; 8],
    ) {
        unsafe {
            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
            let conj_odd = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

            for i in 0..4 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreD::load(data.slice_from(i * 4..));
                let input_top = AvxStoreD::load(data.slice_from(64 - i * 4 - 4..)).reverse();

                let input_half_bottom =
                    AvxStoreD::load(data.slice_from(32 - i * 4 - 4..)).reverse();
                let input_half_top = AvxStoreD::load(data.slice_from(32 + i * 4..));

                //prepare the inner DCT2

                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 4..));

                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(32 - i * 4 - 4..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 4..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(16 - i * 4 - 4..));
            }

            self.bf32.exec(
                &mut InPlaceStore::new(input_dct2),
                input_dct2_32,
                input_dct4_even_32,
                input_dct4_odd_32,
            );
            self.bf16.exec(&mut InPlaceStore::new(input_dct4_even));
            self.bf16.exec(&mut InPlaceStore::new(input_dct4_odd));

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            for i in 1..16 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 16) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(16 - i)
                } else {
                    *input_dct4_odd.get_unchecked(16 - i)
                };

                let q0 = dct4_cos_output + dct4_sin_output;

                data[i * 4 - 1] = q0;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[64 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl AvxDct2Butterfly64d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(64) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2_16 = [f64::default(); 16];
        let mut input_dct4_even_8 = [f64::default(); 8];
        let mut input_dct4_odd_8 = [f64::default(); 8];

        let mut input_dct2_32 = [f64::default(); 32];
        let mut input_dct4_even_16 = [f64::default(); 16];
        let mut input_dct4_odd_16 = [f64::default(); 16];

        for chunk in data.as_chunks_mut::<64>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2_32,
                &mut input_dct4_even_16,
                &mut input_dct4_odd_16,
                &mut input_dct2_16,
                &mut input_dct4_even_8,
                &mut input_dct4_odd_8,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 64);

        let mut input_dct2_16 = [f64::default(); 16];
        let mut input_dct4_even_8 = [f64::default(); 8];
        let mut input_dct4_odd_8 = [f64::default(); 8];

        let mut input_dct2_32 = [f64::default(); 32];
        let mut input_dct4_even_16 = [f64::default(); 16];
        let mut input_dct4_odd_16 = [f64::default(); 16];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<64>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<64>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2_32,
                &mut input_dct4_even_16,
                &mut input_dct4_odd_16,
                &mut input_dct2_16,
                &mut input_dct4_even_8,
                &mut input_dct4_odd_8,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxDct2Butterfly64d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f64], _: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
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
pub(crate) struct AvxDct2Butterfly128d {
    bf64: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    bf32: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    twiddles: [AvxStoreD; 16],
}

impl Default for AvxDct2Butterfly128d {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avx(128) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf64: f64::dct2_butterfly64(),
            bf32: f64::dct2_butterfly32(),
        }
    }
}

impl AvxDct2Butterfly128d {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f64; 64],
        input_dct4: &mut [f64; 64],
    ) {
        unsafe {
            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(32);

            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
            let conj_odd = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

            for i in 0..8 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreD::load(data.slice_from(i * 4..));
                let input_top = AvxStoreD::load(data.slice_from(128 - i * 4 - 4..)).reverse();

                let input_half_bottom =
                    AvxStoreD::load(data.slice_from(64 - i * 4 - 4..)).reverse();
                let input_half_top = AvxStoreD::load(data.slice_from(64 + i * 4..));

                //prepare the inner DCT2
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 4..));
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(64 - i * 4 - 4..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 4..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(32 - i * 4 - 4..));
            }

            _ = self.bf64.execute(input_dct2);
            _ = self.bf32.execute(input_dct4);

            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(32);

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let conj_odd = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

            let mut i = 1usize;
            while i + 4 <= 32 {
                let dct4_cos_output_v = AvxStoreD::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v = AvxStoreD::load(input_dct4_odd.get_unchecked(32 - i - 3..))
                    .reverse()
                    .xor(conj_odd);

                let [dct4_cos_output0, dct4_cos_output1] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, mut dct4_sin_output1] =
                    dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                let input0 = AvxStoreD::load(input_dct2.get_unchecked(i * 2..));
                let input1 = AvxStoreD::load(input_dct2.get_unchecked(i * 2 + 4..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;
                let sums1 = dct4_cos_output1 + dct4_sin_output1;

                let zipped0 = sums0.zip(input0);
                let zipped1 = sums1.zip(input1);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 3..));
                zipped1[0].write(data.slice_from_mut(i * 4 + 7..));
                zipped1[1].write(data.slice_from_mut(i * 4 + 11..));
                i += 4;
            }

            while i + 2 <= 32 {
                let dct4_cos_output_v = AvxStoreD::load2(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreD::load2(input_dct4_odd.get_unchecked(32 - i - 1..))
                        .reverse2()
                        .xor(conj_odd);

                let [dct4_cos_output0, _] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, _] = dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);

                let input0 = AvxStoreD::load(input_dct2.get_unchecked(i * 2..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;

                let zipped0 = sums0.zip(input0);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 3..));
                i += 2;
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

impl AvxDct2Butterfly128d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(128) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::default(); 64];
        let mut input_dct4 = [f64::default(); 64];

        for chunk in data.as_chunks_mut::<128>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 128);

        let mut input_dct2 = [f64::default(); 64];
        let mut input_dct4 = [f64::default(); 64];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<128>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<128>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxDct2Butterfly128d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f64], _: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
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
pub(crate) struct AvxDct2Butterfly256d {
    bf128: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    bf64: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    twiddles: [AvxStoreD; 32],
}

impl Default for AvxDct2Butterfly256d {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avx(256) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf64: f64::dct2_butterfly64(),
            bf128: f64::dct2_butterfly128(),
        }
    }
}

impl AvxDct2Butterfly256d {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f64; 128],
        input_dct4: &mut [f64; 128],
    ) {
        unsafe {
            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(64);

            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

            let conj_odd = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

            for i in 0..16 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreD::load(data.slice_from(i * 4..));
                let input_top = AvxStoreD::load(data.slice_from(256 - i * 4 - 4..)).reverse();

                let input_half_bottom =
                    AvxStoreD::load(data.slice_from(128 - i * 4 - 4..)).reverse();
                let input_half_top = AvxStoreD::load(data.slice_from(128 + i * 4..));

                //prepare the inner DCT2

                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 4..));
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(128 - i * 4 - 4..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 4..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(64 - i * 4 - 4..));
            }

            _ = self.bf128.execute(input_dct2);
            _ = self.bf64.execute(input_dct4);

            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(64);

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let mut i = 1usize;
            while i + 4 <= 64 {
                let dct4_cos_output_v = AvxStoreD::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v = AvxStoreD::load(input_dct4_odd.get_unchecked(64 - i - 3..))
                    .reverse()
                    .xor(conj_odd);

                let [dct4_cos_output0, dct4_cos_output1] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, mut dct4_sin_output1] =
                    dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                let input0 = AvxStoreD::load(input_dct2.get_unchecked(i * 2..));
                let input1 = AvxStoreD::load(input_dct2.get_unchecked(i * 2 + 4..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;
                let sums1 = dct4_cos_output1 + dct4_sin_output1;

                let zipped0 = sums0.zip(input0);
                let zipped1 = sums1.zip(input1);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 3..));
                zipped1[0].write(data.slice_from_mut(i * 4 + 7..));
                zipped1[1].write(data.slice_from_mut(i * 4 + 11..));
                i += 4;
            }

            while i + 2 <= 64 {
                let dct4_cos_output_v = AvxStoreD::load2(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreD::load2(input_dct4_odd.get_unchecked(64 - i - 1..))
                        .reverse2()
                        .xor(conj_odd);

                let [dct4_cos_output0, _] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, _] = dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);

                let input0 = AvxStoreD::load(input_dct2.get_unchecked(i * 2..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;

                let zipped0 = sums0.zip(input0);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 3..));
                i += 2;
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

impl AvxDct2Butterfly256d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(256) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::zero(); 128];
        let mut input_dct4 = [f64::zero(); 128];

        for chunk in data.as_chunks_mut::<256>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 256);

        let mut input_dct2 = [f64::zero(); 128];
        let mut input_dct4 = [f64::zero(); 128];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<256>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<256>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxDct2Butterfly256d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f64], _: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
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
pub(crate) struct AvxDct2Butterfly512d {
    bf128: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    bf256: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    twiddles: [AvxStoreD; 64],
}

impl Default for AvxDct2Butterfly512d {
    fn default() -> Self {
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avx(512) };
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf256: f64::dct2_butterfly256(),
            bf128: f64::dct2_butterfly128(),
        }
    }
}

impl AvxDct2Butterfly512d {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        input_dct2: &mut [f64; 256],
        input_dct4: &mut [f64; 256],
    ) {
        unsafe {
            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(128);

            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

            let conj_odd = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

            for i in 0..32 {
                let twiddle_re = self.twiddles[i * 2];
                let twiddle_im = self.twiddles[i * 2 + 1];
                let input_bottom = AvxStoreD::load(data.slice_from(i * 4..));
                let input_top = AvxStoreD::load(data.slice_from(512 - i * 4 - 4..)).reverse();

                let input_half_bottom =
                    AvxStoreD::load(data.slice_from(256 - i * 4 - 4..)).reverse();
                let input_half_top = AvxStoreD::load(data.slice_from(256 + i * 4..));

                //prepare the inner DCT2
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 4..));
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(256 - i * 4 - 4..));

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                cos_input.write(input_dct4_even.get_unchecked_mut(i * 4..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(128 - i * 4 - 4..));
            }

            _ = self.bf256.execute(input_dct2);
            _ = self.bf128.execute(input_dct4);

            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(128);

            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let mut i = 1usize;
            while i + 4 <= 128 {
                let dct4_cos_output_v = AvxStoreD::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreD::load(input_dct4_odd.get_unchecked(128 - i - 3..))
                        .reverse()
                        .xor(conj_odd);

                let [dct4_cos_output0, dct4_cos_output1] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, mut dct4_sin_output1] =
                    dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                let input0 = AvxStoreD::load(input_dct2.get_unchecked(i * 2..));
                let input1 = AvxStoreD::load(input_dct2.get_unchecked(i * 2 + 4..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;
                let sums1 = dct4_cos_output1 + dct4_sin_output1;

                let zipped0 = sums0.zip(input0);
                let zipped1 = sums1.zip(input1);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 3..));
                zipped1[0].write(data.slice_from_mut(i * 4 + 7..));
                zipped1[1].write(data.slice_from_mut(i * 4 + 11..));
                i += 4;
            }

            while i + 2 <= 128 {
                let dct4_cos_output_v = AvxStoreD::load2(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    AvxStoreD::load2(input_dct4_odd.get_unchecked(128 - i - 1..))
                        .reverse2()
                        .xor(conj_odd);

                let [dct4_cos_output0, _] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, _] = dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);

                let input0 = AvxStoreD::load(input_dct2.get_unchecked(i * 2..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;

                let zipped0 = sums0.zip(input0);

                zipped0[0].write(data.slice_from_mut(i * 4 - 1..));
                zipped0[1].write(data.slice_from_mut(i * 4 + 3..));
                i += 2;
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

impl AvxDct2Butterfly512d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(512) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::default(); 256];
        let mut input_dct4 = [f64::default(); 256];

        for chunk in data.as_chunks_mut::<512>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 512);

        let mut input_dct2 = [f64::default(); 256];
        let mut input_dct4 = [f64::default(); 256];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<512>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<512>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxDct2Butterfly512d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f64], _: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
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
    use crate::tests::naive_dct2;

    gen_test_avx_butterfly!(test_avx_bf32, AvxDct2Butterfly32d, 32, 1e-3, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf64, AvxDct2Butterfly64d, 64, 1e-3, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf128, AvxDct2Butterfly128d, 128, 1e-3, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf256, AvxDct2Butterfly256d, 256, 1e-2, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf512, AvxDct2Butterfly512d, 512, 1e-2, naive_dct2);
}
