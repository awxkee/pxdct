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
use crate::dct2::power2_butterflies::{Dct2Butterfly8, Dct2Butterfly16};
use crate::factory_dct2::Dct2Factory;
use crate::mla::fmla;
use crate::neon::store_d::NeonStoreD;
use crate::twiddles::compute_twiddle;
use crate::{PxdctError, PxdctExecutor};
use num_traits::Zero;
use std::sync::Arc;

pub(crate) fn dct2_split_radix_rotation_twiddles_neond(len: usize) -> Vec<NeonStoreD> {
    let twiddles_len = len / 4;
    let simd_groups = len.div_ceil(2);

    // We need 2 complex values per k (rotation_re and rotation_im)
    // Each complex has re and im, so 4 values per k
    // Times inner_groups for each m
    let mut twiddles = Vec::with_capacity(simd_groups * 2);

    let mut uk = 0usize;
    while uk + 2 <= twiddles_len {
        let mut array_re = [0.; 2];
        let mut array_im = [0.; 2];
        for ki in 0..2 {
            let i = uk + ki;
            let twiddle = compute_twiddle::<f64>(2 * i + 1, len * 4).conj();
            array_re[ki] = twiddle.re;
            array_im[ki] = twiddle.im;
        }

        twiddles.push(NeonStoreD::load(&array_re));
        twiddles.push(NeonStoreD::load(&array_im));

        uk += 2;
    }

    let remainder = twiddles_len - (twiddles_len / 2) * 2;
    if remainder > 0 {
        let mut array_re = [0.; 2];
        let mut array_im = [0.; 2];
        for i in 0..remainder {
            let i = uk + i;
            let twiddle = compute_twiddle::<f64>(2 * i + 1, len * 4).conj();
            array_re[i] = twiddle.re;
            array_im[i] = twiddle.im;
        }

        twiddles.push(NeonStoreD::load(&array_re));
        twiddles.push(NeonStoreD::load(&array_im));
    }

    twiddles
}

#[derive(Debug, Clone)]
pub(crate) struct NeonDct2Butterfly32d {
    bf8: Dct2Butterfly8<f64>,
    bf16: Dct2Butterfly16<f64>,
    twiddles: [NeonStoreD; 8],
}

impl Default for NeonDct2Butterfly32d {
    fn default() -> Self {
        let twiddles = dct2_split_radix_rotation_twiddles_neond(32);
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf8: Dct2Butterfly8::default(),
            bf16: Dct2Butterfly16::default(),
        }
    }
}

impl NeonDct2Butterfly32d {
    #[inline(always)]
    fn exec(
        &self,
        data: &mut [f64; 32],
        input_dct2: &mut [f64; 16],
        input_dct4_even: &mut [f64; 8],
        input_dct4_odd: &mut [f64; 8],
    ) {
        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

        let conj_odd = NeonStoreD::set_values(0.0, -0.0);

        for i in 0..4 {
            let twiddle_re = self.twiddles[i * 2];
            let twiddle_im = self.twiddles[i * 2 + 1];
            let input_bottom = NeonStoreD::load(&data[i * 2..]);
            let input_top = NeonStoreD::load(&data[32 - i * 2 - 2..]).reverse();

            let input_half_bottom = NeonStoreD::load(&data[16 - i * 2 - 2..]).reverse();
            let input_half_top = NeonStoreD::load(&data[16 + i * 2..]);

            //prepare the inner DCT2
            unsafe {
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 2..));
            };
            unsafe {
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(16 - i * 2 - 2..));
            };

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
            let sin_input = fmla(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

            unsafe {
                cos_input.write(input_dct4_even.get_unchecked_mut(i * 2..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(8 - i * 2 - 2..));
            };
        }

        self.bf16.exec(input_dct2);
        self.bf8.exec(input_dct4_even);
        self.bf8.exec(input_dct4_odd);

        unsafe {
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

                *data.get_unchecked_mut(i * 4 - 1) = dct4_cos_output + dct4_sin_output;
                *data.get_unchecked_mut(i * 4) = *input_dct2.get_unchecked(i * 2);

                *data.get_unchecked_mut(i * 4 + 1) = dct4_cos_output - dct4_sin_output;
                *data.get_unchecked_mut(i * 4 + 2) = *input_dct2.get_unchecked(i * 2 + 1);
            }

            *data.get_unchecked_mut(32 - 1) = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl PxdctExecutor<f64> for NeonDct2Butterfly32d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(32) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::zero(); 16];
        let mut input_dct4_even = [f64::zero(); 8];
        let mut input_dct4_odd = [f64::zero(); 8];

        for chunk in data.chunks_exact_mut(32) {
            self.exec(
                (&mut chunk[..32]).try_into().unwrap(),
                &mut input_dct2,
                &mut input_dct4_even,
                &mut input_dct4_odd,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        32
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NeonDct2Butterfly64d {
    bf32: NeonDct2Butterfly32d,
    bf16: Dct2Butterfly16<f64>,
    twiddles: [NeonStoreD; 16],
}

impl Default for NeonDct2Butterfly64d {
    fn default() -> Self {
        let twiddles = dct2_split_radix_rotation_twiddles_neond(64);
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf16: Dct2Butterfly16::default(),
            bf32: NeonDct2Butterfly32d::default(),
        }
    }
}

impl NeonDct2Butterfly64d {
    #[inline(always)]
    fn exec(
        &self,
        data: &mut [f64; 64],
        input_dct2: &mut [f64; 32],
        input_dct4_even: &mut [f64; 16],
        input_dct4_odd: &mut [f64; 16],
        input_dct21: &mut [f64; 16],
        input_dct4_even1: &mut [f64; 8],
        input_dct4_odd1: &mut [f64; 8],
    ) {
        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
        let conj_odd = NeonStoreD::set_values(0.0, -0.0);

        for i in 0..8 {
            let twiddle_re = self.twiddles[i * 2];
            let twiddle_im = self.twiddles[i * 2 + 1];
            let input_bottom = NeonStoreD::load(&data[i * 2..]);
            let input_top = NeonStoreD::load(&data[64 - i * 2 - 2..]).reverse();

            let input_half_bottom = NeonStoreD::load(&data[32 - i * 2 - 2..]).reverse();
            let input_half_top = NeonStoreD::load(&data[32 + i * 2..]);

            //prepare the inner DCT2
            unsafe {
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 2..));
            };
            unsafe {
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(32 - i * 2 - 2..));
            };

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
            let sin_input = fmla(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

            unsafe {
                cos_input.write(input_dct4_even.get_unchecked_mut(i * 2..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(16 - i * 2 - 2..));
            };
        }

        self.bf32
            .exec(input_dct2, input_dct21, input_dct4_even1, input_dct4_odd1);
        self.bf16.exec(input_dct4_even);
        self.bf16.exec(input_dct4_odd);

        unsafe {
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

                *data.get_unchecked_mut(i * 4 - 1) = q0;
                *data.get_unchecked_mut(i * 4) = *input_dct2.get_unchecked(i * 2);

                *data.get_unchecked_mut(i * 4 + 1) = dct4_cos_output - dct4_sin_output;
                *data.get_unchecked_mut(i * 4 + 2) = *input_dct2.get_unchecked(i * 2 + 1);
            }

            *data.get_unchecked_mut(64 - 1) = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl PxdctExecutor<f64> for NeonDct2Butterfly64d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(64) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::zero(); 32];
        let mut input_dct4_even = [f64::zero(); 16];
        let mut input_dct4_odd = [f64::zero(); 16];
        let mut input_dct21 = [f64::zero(); 16];
        let mut input_dct4_even1 = [f64::zero(); 8];
        let mut input_dct4_odd1 = [f64::zero(); 8];

        for chunk in data.chunks_exact_mut(64) {
            self.exec(
                (&mut chunk[..64]).try_into().unwrap(),
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

    fn length(&self) -> usize {
        64
    }
}

#[derive(Clone)]
pub(crate) struct NeonDct2Butterfly128d {
    bf64: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    bf32: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    twiddles: [NeonStoreD; 32],
}

impl Default for NeonDct2Butterfly128d {
    fn default() -> Self {
        let twiddles = dct2_split_radix_rotation_twiddles_neond(128);
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf64: f64::dct2_butterfly64(),
            bf32: f64::dct2_butterfly32(),
        }
    }
}

impl NeonDct2Butterfly128d {
    #[inline(always)]
    fn exec(&self, data: &mut [f64; 128], input_dct2: &mut [f64; 64], input_dct4: &mut [f64; 64]) {
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(32);

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
        let conj_odd = NeonStoreD::set_values(0.0, -0.0);

        for i in 0..16 {
            let twiddle_re = self.twiddles[i * 2];
            let twiddle_im = self.twiddles[i * 2 + 1];
            let input_bottom = NeonStoreD::load(&data[i * 2..]);
            let input_top = NeonStoreD::load(&data[128 - i * 2 - 2..]).reverse();

            let input_half_bottom = NeonStoreD::load(&data[64 - i * 2 - 2..]).reverse();
            let input_half_top = NeonStoreD::load(&data[64 + i * 2..]);

            //prepare the inner DCT2
            unsafe {
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 2..));
            };
            unsafe {
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(64 - i * 2 - 2..));
            };

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
            let sin_input = fmla(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

            unsafe {
                cos_input.write(input_dct4_even.get_unchecked_mut(i * 2..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(32 - i * 2 - 2..));
            };
        }

        _ = self.bf64.execute(input_dct2);
        _ = self.bf32.execute(input_dct4);

        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(32);

        unsafe {
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let conj_odd = NeonStoreD::set_values(0.0, -0.0);

            let mut i = 1usize;

            while i + 2 <= 32 {
                let dct4_cos_output_v = NeonStoreD::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    NeonStoreD::load(input_dct4_odd.get_unchecked(32 - i - 1..))
                        .reverse()
                        .xor(conj_odd);

                let [dct4_cos_output0, dct4_cos_output1] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, mut dct4_sin_output1] =
                    dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                let input0 = NeonStoreD::load(input_dct2.get_unchecked(i * 2..));
                let input1 = NeonStoreD::load(input_dct2.get_unchecked(i * 2 + 2..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;
                let sums1 = dct4_cos_output1 + dct4_sin_output1;

                let zipped0 = sums0.zip(input0);
                let zipped1 = sums1.zip(input1);

                zipped0[0].write(data.get_unchecked_mut(i * 4 - 1..));
                zipped0[1].write(data.get_unchecked_mut(i * 4 + 1..));
                zipped1[0].write(data.get_unchecked_mut(i * 4 + 3..));
                zipped1[1].write(data.get_unchecked_mut(i * 4 + 5..));
                i += 2;
            }

            for i in i..32 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 32) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(32 - i)
                } else {
                    *input_dct4_odd.get_unchecked(32 - i)
                };

                *data.get_unchecked_mut(i * 4 - 1) = dct4_cos_output + dct4_sin_output;
                *data.get_unchecked_mut(i * 4) = *input_dct2.get_unchecked(i * 2);

                *data.get_unchecked_mut(i * 4 + 1) = dct4_cos_output - dct4_sin_output;
                *data.get_unchecked_mut(i * 4 + 2) = *input_dct2.get_unchecked(i * 2 + 1);
            }

            *data.get_unchecked_mut(128 - 1) = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl PxdctExecutor<f64> for NeonDct2Butterfly128d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(128) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::zero(); 64];
        let mut input_dct4 = [f64::zero(); 64];

        for chunk in data.chunks_exact_mut(128) {
            self.exec(
                (&mut chunk[..128]).try_into().unwrap(),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        128
    }
}

#[derive(Clone)]
pub(crate) struct NeonDct2Butterfly256d {
    bf128: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    bf64: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    twiddles: [NeonStoreD; 64],
}

impl Default for NeonDct2Butterfly256d {
    fn default() -> Self {
        let twiddles = dct2_split_radix_rotation_twiddles_neond(256);
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf64: f64::dct2_butterfly64(),
            bf128: f64::dct2_butterfly128(),
        }
    }
}

impl NeonDct2Butterfly256d {
    #[inline(always)]
    fn exec(
        &self,
        data: &mut [f64; 256],
        input_dct2: &mut [f64; 128],
        input_dct4: &mut [f64; 128],
    ) {
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(64);

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

        let conj_odd = NeonStoreD::set_values(0.0, -0.0);

        for i in 0..32 {
            let twiddle_re = self.twiddles[i * 2];
            let twiddle_im = self.twiddles[i * 2 + 1];
            let input_bottom = NeonStoreD::load(&data[i * 2..]);
            let input_top = NeonStoreD::load(&data[256 - i * 2 - 2..]).reverse();

            let input_half_bottom = NeonStoreD::load(&data[128 - i * 2 - 2..]).reverse();
            let input_half_top = NeonStoreD::load(&data[128 + i * 2..]);

            //prepare the inner DCT2
            unsafe {
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 2..));
            };
            unsafe {
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(128 - i * 2 - 2..));
            };

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
            let sin_input = fmla(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

            unsafe {
                cos_input.write(input_dct4_even.get_unchecked_mut(i * 2..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(64 - i * 2 - 2..));
            };
        }

        _ = self.bf128.execute(input_dct2);
        _ = self.bf64.execute(input_dct4);

        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(64);

        unsafe {
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let mut i = 1usize;

            while i + 2 <= 64 {
                let dct4_cos_output_v = NeonStoreD::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    NeonStoreD::load(input_dct4_odd.get_unchecked(64 - i - 1..))
                        .reverse()
                        .xor(conj_odd);

                let [dct4_cos_output0, dct4_cos_output1] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, mut dct4_sin_output1] =
                    dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                let input0 = NeonStoreD::load(input_dct2.get_unchecked(i * 2..));
                let input1 = NeonStoreD::load(input_dct2.get_unchecked(i * 2 + 2..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;
                let sums1 = dct4_cos_output1 + dct4_sin_output1;

                let zipped0 = sums0.zip(input0);
                let zipped1 = sums1.zip(input1);

                zipped0[0].write(data.get_unchecked_mut(i * 4 - 1..));
                zipped0[1].write(data.get_unchecked_mut(i * 4 + 1..));
                zipped1[0].write(data.get_unchecked_mut(i * 4 + 3..));
                zipped1[1].write(data.get_unchecked_mut(i * 4 + 5..));
                i += 2;
            }

            for i in i..64 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 64) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(64 - i)
                } else {
                    *input_dct4_odd.get_unchecked(64 - i)
                };

                *data.get_unchecked_mut(i * 4 - 1) = dct4_cos_output + dct4_sin_output;
                *data.get_unchecked_mut(i * 4) = *input_dct2.get_unchecked(i * 2);

                *data.get_unchecked_mut(i * 4 + 1) = dct4_cos_output - dct4_sin_output;
                *data.get_unchecked_mut(i * 4 + 2) = *input_dct2.get_unchecked(i * 2 + 1);
            }

            *data.get_unchecked_mut(256 - 1) = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl PxdctExecutor<f64> for NeonDct2Butterfly256d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(256) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::zero(); 128];
        let mut input_dct4 = [f64::zero(); 128];

        for chunk in data.chunks_exact_mut(256) {
            self.exec(
                (&mut chunk[..256]).try_into().unwrap(),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        256
    }
}

#[derive(Clone)]
pub(crate) struct NeonDct2Butterfly512d {
    bf128: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    bf256: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    twiddles: [NeonStoreD; 128],
}

impl Default for NeonDct2Butterfly512d {
    fn default() -> Self {
        let twiddles = dct2_split_radix_rotation_twiddles_neond(512);
        Self {
            twiddles: twiddles.try_into().unwrap(),
            bf256: f64::dct2_butterfly256(),
            bf128: f64::dct2_butterfly128(),
        }
    }
}

impl NeonDct2Butterfly512d {
    #[inline(always)]
    fn exec(
        &self,
        data: &mut [f64; 512],
        input_dct2: &mut [f64; 256],
        input_dct4: &mut [f64; 256],
    ) {
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(128);

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

        let conj_odd = NeonStoreD::set_values(0.0, -0.0);

        for i in 0..64 {
            let twiddle_re = self.twiddles[i * 2];
            let twiddle_im = self.twiddles[i * 2 + 1];
            let input_bottom = NeonStoreD::load(&data[i * 2..]);
            let input_top = NeonStoreD::load(&data[512 - i * 2 - 2..]).reverse();

            let input_half_bottom = NeonStoreD::load(&data[256 - i * 2 - 2..]).reverse();
            let input_half_top = NeonStoreD::load(&data[256 + i * 2..]);

            //prepare the inner DCT2
            unsafe {
                let q = input_top + input_bottom;
                q.write(input_dct2.get_unchecked_mut(i * 2..));
            };
            unsafe {
                let dq = input_half_bottom + input_half_top;
                dq.reverse()
                    .write(input_dct2.get_unchecked_mut(256 - i * 2 - 2..));
            };

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
            let sin_input = fmla(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

            unsafe {
                cos_input.write(input_dct4_even.get_unchecked_mut(i * 2..));
                let conj_odd = sin_input.xor(conj_odd).reverse();
                conj_odd.write(input_dct4_odd.get_unchecked_mut(128 - i * 2 - 2..));
            };
        }

        _ = self.bf256.execute(input_dct2);
        _ = self.bf128.execute(input_dct4);

        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(128);

        unsafe {
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            let mut i = 1usize;
            while i + 2 <= 128 {
                let dct4_cos_output_v = NeonStoreD::load(input_dct4_even.get_unchecked(i..));
                let dct4_sin_output_v =
                    NeonStoreD::load(input_dct4_odd.get_unchecked(128 - i - 1..))
                        .reverse()
                        .xor(conj_odd);

                let [dct4_cos_output0, dct4_cos_output1] = dct4_cos_output_v.zip(dct4_cos_output_v);
                let [mut dct4_sin_output0, mut dct4_sin_output1] =
                    dct4_sin_output_v.zip(dct4_sin_output_v);

                dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                let input0 = NeonStoreD::load(input_dct2.get_unchecked(i * 2..));
                let input1 = NeonStoreD::load(input_dct2.get_unchecked(i * 2 + 2..));

                let sums0 = dct4_cos_output0 + dct4_sin_output0;
                let sums1 = dct4_cos_output1 + dct4_sin_output1;

                let zipped0 = sums0.zip(input0);
                let zipped1 = sums1.zip(input1);

                zipped0[0].write(data.get_unchecked_mut(i * 4 - 1..));
                zipped0[1].write(data.get_unchecked_mut(i * 4 + 1..));
                zipped1[0].write(data.get_unchecked_mut(i * 4 + 3..));
                zipped1[1].write(data.get_unchecked_mut(i * 4 + 5..));
                i += 2;
            }

            for i in i..128 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 128) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(128 - i)
                } else {
                    *input_dct4_odd.get_unchecked(128 - i)
                };

                *data.get_unchecked_mut(i * 4 - 1) = dct4_cos_output + dct4_sin_output;
                *data.get_unchecked_mut(i * 4) = *input_dct2.get_unchecked(i * 2);

                *data.get_unchecked_mut(i * 4 + 1) = dct4_cos_output - dct4_sin_output;
                *data.get_unchecked_mut(i * 4 + 2) = *input_dct2.get_unchecked(i * 2 + 1);
            }

            *data.get_unchecked_mut(512 - 1) = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl PxdctExecutor<f64> for NeonDct2Butterfly512d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(512) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [f64::zero(); 256];
        let mut input_dct4 = [f64::zero(); 256];

        for chunk in data.chunks_exact_mut(512) {
            self.exec(
                (&mut chunk[..512]).try_into().unwrap(),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        512
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly_f;
    use crate::tests::naive_dct2;

    gen_test_butterfly_f!(test_neon_bf32d, NeonDct2Butterfly32d, 32, 1e-7, naive_dct2);
    gen_test_butterfly_f!(test_neon_bf64d, NeonDct2Butterfly64d, 64, 1e-7, naive_dct2);
    gen_test_butterfly_f!(
        test_neon_bf128d,
        NeonDct2Butterfly128d,
        128,
        1e-7,
        naive_dct2
    );
    gen_test_butterfly_f!(
        test_neon_bf256,
        NeonDct2Butterfly256d,
        256,
        1e-7,
        naive_dct2
    );
    gen_test_butterfly_f!(
        test_neon_bf512d,
        NeonDct2Butterfly512d,
        512,
        1e-7,
        naive_dct2
    );
}
