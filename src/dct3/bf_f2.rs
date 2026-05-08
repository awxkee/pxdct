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
use crate::dst3_butterfly::{Dst3Butterfly2, Dst3Butterfly4};
use crate::factory_dct3::Dct3Factory;
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct3Butterfly2<T> {
    phantom: PhantomData<T>,
}

impl<T: DctSample> Dct3Butterfly2<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        #[cfg(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        ))]
        {
            let half_0 = data[0] * T::HALF;
            data[0] = fmla(data[1], T::FRAC_1_SQRT_2, half_0);
            data[1] = fmla(data[1], -T::FRAC_1_SQRT_2, half_0);
        }
        #[cfg(not(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        )))]
        {
            let half_0 = data[0] * T::HALF;
            let frac_1 = data[1] * T::FRAC_1_SQRT_2;
            data[0] = half_0 + frac_1;
            data[1] = half_0 - frac_1;
        }
    }
}

define_in_place_butterfly!(Dct3Butterfly2, 2);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly4<T> {
    twiddle: Complex<T>,
    bf2: Dct3Butterfly2<T>,
}

impl<T: DctSample> Default for Dct3Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 16).conj(),
            bf2: Dct3Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly4<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // DCT-3 split radix with n = 4
        let mut evens = [data[0], data[2]];
        self.bf2.exec(&mut InPlaceStore::new(&mut evens));
        let lower_dct4 = fmla(data[1], self.twiddle.re, data[3] * self.twiddle.im);
        let upper_dct4 = fmla(data[1], self.twiddle.im, -data[3] * self.twiddle.re);

        data[1] = evens[1] + upper_dct4;
        data[3] = evens[0] - lower_dct4;
        data[0] = evens[0] + lower_dct4;
        data[2] = evens[1] - upper_dct4;
    }
}

define_in_place_butterfly!(Dct3Butterfly4, 4);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly8<T> {
    twiddles: [Complex<T>; 2],
    bf2: Dct3Butterfly2<T>,
    bf2_dst: Dst3Butterfly2<T>,
    bf4: Dct3Butterfly4<T>,
}

impl<T: DctSample> Default for Dct3Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf2: Dct3Butterfly2::default(),
            bf4: Dct3Butterfly4::default(),
            bf2_dst: Dst3Butterfly2::default(),
            twiddles: [compute_twiddle(1, 32).conj(), compute_twiddle(3, 32).conj()],
        }
    }
}

impl<T: DctSample> Dct3Butterfly8<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        //process the evens
        let mut dct3_buffer = [data[0], data[2], data[4], data[6]];
        self.bf4.exec(&mut InPlaceStore::new(&mut dct3_buffer));

        //process the odds
        let mut odds_n1 = [data[1] * T::TWO, data[3] + data[5]];
        let mut odds_n3 = [data[3] - data[5], data[7] * T::TWO];

        self.bf2.exec(&mut InPlaceStore::new(&mut odds_n1));
        self.bf2_dst.exec(&mut InPlaceStore::new(&mut odds_n3));

        let twiddle0 = self.twiddles[0];

        let lower_dct4_0 = fmla(odds_n1[0], twiddle0.re, odds_n3[0] * twiddle0.im);
        let upper_dct4_0 = fmla(odds_n1[0], twiddle0.im, -odds_n3[0] * twiddle0.re);

        let lower_dct3_0 = dct3_buffer[0];
        let upper_dct3_0 = dct3_buffer[3];

        data[0] = lower_dct3_0 + lower_dct4_0;
        data[7] = lower_dct3_0 - lower_dct4_0;

        data[3] = upper_dct3_0 + upper_dct4_0;
        data[4] = upper_dct3_0 - upper_dct4_0;

        let twiddle1 = self.twiddles[1];

        let lower_dct4_1 = fmla(odds_n1[1], twiddle1.re, odds_n3[1] * twiddle1.im);
        let upper_dct4_1 = fmla(odds_n1[1], twiddle1.im, -odds_n3[1] * twiddle1.re);

        let lower_dct3 = dct3_buffer[1];
        let upper_dct3 = dct3_buffer[2];

        data[1] = lower_dct3 + lower_dct4_1;
        data[6] = lower_dct3 - lower_dct4_1;

        data[2] = upper_dct3 + upper_dct4_1;
        data[5] = upper_dct3 - upper_dct4_1;
    }
}

define_in_place_butterfly!(Dct3Butterfly8, 8);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly16<T> {
    twiddles: [Complex<T>; 4],
    bf4: Dct3Butterfly4<T>,
    bf4_dst: Dst3Butterfly4<T>,
    bf8: Dct3Butterfly8<T>,
}

impl<T: DctSample> Default for Dct3Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: Dct3Butterfly4::default(),
            bf4_dst: Dst3Butterfly4::default(),
            bf8: Dct3Butterfly8::default(),
            twiddles: [
                compute_twiddle(1, 64).conj(),
                compute_twiddle(3, 64).conj(),
                compute_twiddle(5, 64).conj(),
                compute_twiddle(7, 64).conj(),
            ],
        }
    }
}

impl<T: DctSample> Dct3Butterfly16<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        //process the evens
        let mut dct3_buffer = [
            data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
        ];
        self.bf8.exec(&mut InPlaceStore::new(&mut dct3_buffer));

        //process the odds
        let mut odds_n1 = [
            data[1] * T::TWO,
            data[3] + data[5],
            data[7] + data[9],
            data[11] + data[13],
        ];
        let mut odds_n3 = [
            data[3] - data[5],
            data[7] - data[9],
            data[11] - data[13],
            data[15] * T::TWO,
        ];
        self.bf4.exec(&mut InPlaceStore::new(&mut odds_n1));
        self.bf4_dst.exec(&mut InPlaceStore::new(&mut odds_n3));

        for i in 0..4 {
            let lower_dct4 = fmla(
                odds_n1[i],
                self.twiddles[i].re,
                odds_n3[i] * self.twiddles[i].im,
            );
            let upper_dct4 = fmla(
                odds_n1[i],
                self.twiddles[i].im,
                -odds_n3[i] * self.twiddles[i].re,
            );

            let lower_dct3 = dct3_buffer[i];
            let upper_dct3 = dct3_buffer[7 - i];

            data[i] = lower_dct3 + lower_dct4;
            data[15 - i] = lower_dct3 - lower_dct4;

            data[7 - i] = upper_dct3 + upper_dct4;
            data[8 + i] = upper_dct3 - upper_dct4;
        }
    }
}

define_in_place_butterfly!(Dct3Butterfly16, 16);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly32<T> {
    twiddles: [Complex<T>; 8],
    bf8: Dct3Butterfly8<T>,
    bf16: Dct3Butterfly16<T>,
}

impl<T: DctSample> Default for Dct3Butterfly32<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf8: Dct3Butterfly8::default(),
            bf16: Dct3Butterfly16::default(),
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 32 * 4).conj()),
        }
    }
}

impl<T: DctSample> Dct3Butterfly32<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut evens = [
            data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16],
            data[18], data[20], data[22], data[24], data[26], data[28], data[30],
        ];
        let mut recursive_input_n1 = [
            data[1] * T::TWO,
            data[3] + data[5],
            data[7] + data[9],
            data[11] + data[13],
            data[15] + data[17],
            data[19] + data[21],
            data[23] + data[25],
            data[27] + data[29],
        ];
        let mut recursive_input_n3 = [
            data[31] * T::TWO,
            data[27] - data[29],
            data[23] - data[25],
            data[19] - data[21],
            data[15] - data[17],
            data[11] - data[13],
            data[7] - data[9],
            data[3] - data[5],
        ];

        self.bf16.exec(&mut InPlaceStore::new(&mut evens));
        self.bf8
            .exec(&mut InPlaceStore::new(&mut recursive_input_n1));
        self.bf8
            .exec(&mut InPlaceStore::new(&mut recursive_input_n3));

        let tw0 = self.twiddles[0];
        let cosine_value0 = recursive_input_n1[0];
        let sine_value0 = recursive_input_n3[0];
        let lower_dct40 = fmla(cosine_value0, tw0.re, sine_value0 * tw0.im);
        let upper_dct40 = fmla(cosine_value0, tw0.im, -sine_value0 * tw0.re);
        let lower_dct30 = evens[0];
        let upper_dct30 = evens[15];
        data[0] = lower_dct30 + lower_dct40;
        data[31] = lower_dct30 - lower_dct40;
        data[15] = upper_dct30 + upper_dct40;
        data[16] = upper_dct30 - upper_dct40;
        let tw1 = self.twiddles[1];
        let cosine_value1 = recursive_input_n1[1];
        let sine_value1 = -recursive_input_n3[1];
        let lower_dct41 = fmla(cosine_value1, tw1.re, sine_value1 * tw1.im);
        let upper_dct41 = fmla(cosine_value1, tw1.im, -sine_value1 * tw1.re);
        let lower_dct31 = evens[1];
        let upper_dct31 = evens[14];
        data[1] = lower_dct31 + lower_dct41;
        data[30] = lower_dct31 - lower_dct41;
        data[14] = upper_dct31 + upper_dct41;
        data[17] = upper_dct31 - upper_dct41;
        let tw2 = self.twiddles[2];
        let cosine_value2 = recursive_input_n1[2];
        let sine_value2 = recursive_input_n3[2];
        let lower_dct42 = fmla(cosine_value2, tw2.re, sine_value2 * tw2.im);
        let upper_dct42 = fmla(cosine_value2, tw2.im, -sine_value2 * tw2.re);
        let lower_dct32 = evens[2];
        let upper_dct32 = evens[13];
        data[2] = lower_dct32 + lower_dct42;
        data[29] = lower_dct32 - lower_dct42;
        data[13] = upper_dct32 + upper_dct42;
        data[18] = upper_dct32 - upper_dct42;
        let tw3 = self.twiddles[3];
        let cosine_value3 = recursive_input_n1[3];
        let sine_value3 = -recursive_input_n3[3];
        let lower_dct43 = fmla(cosine_value3, tw3.re, sine_value3 * tw3.im);
        let upper_dct43 = fmla(cosine_value3, tw3.im, -sine_value3 * tw3.re);
        let lower_dct33 = evens[3];
        let upper_dct33 = evens[12];
        data[3] = lower_dct33 + lower_dct43;
        data[28] = lower_dct33 - lower_dct43;
        data[12] = upper_dct33 + upper_dct43;
        data[19] = upper_dct33 - upper_dct43;
        let tw4 = self.twiddles[4];
        let cosine_value4 = recursive_input_n1[4];
        let sine_value4 = recursive_input_n3[4];
        let lower_dct44 = fmla(cosine_value4, tw4.re, sine_value4 * tw4.im);
        let upper_dct44 = fmla(cosine_value4, tw4.im, -sine_value4 * tw4.re);
        let lower_dct34 = evens[4];
        let upper_dct34 = evens[11];
        data[4] = lower_dct34 + lower_dct44;
        data[27] = lower_dct34 - lower_dct44;
        data[11] = upper_dct34 + upper_dct44;
        data[20] = upper_dct34 - upper_dct44;
        let tw5 = self.twiddles[5];
        let cosine_value5 = recursive_input_n1[5];
        let sine_value5 = -recursive_input_n3[5];
        let lower_dct45 = fmla(cosine_value5, tw5.re, sine_value5 * tw5.im);
        let upper_dct45 = fmla(cosine_value5, tw5.im, -sine_value5 * tw5.re);
        let lower_dct35 = evens[5];
        let upper_dct35 = evens[10];
        data[5] = lower_dct35 + lower_dct45;
        data[26] = lower_dct35 - lower_dct45;
        data[10] = upper_dct35 + upper_dct45;
        data[21] = upper_dct35 - upper_dct45;
        let tw6 = self.twiddles[6];
        let cosine_value6 = recursive_input_n1[6];
        let sine_value6 = recursive_input_n3[6];
        let lower_dct46 = fmla(cosine_value6, tw6.re, sine_value6 * tw6.im);
        let upper_dct46 = fmla(cosine_value6, tw6.im, -sine_value6 * tw6.re);
        let lower_dct36 = evens[6];
        let upper_dct36 = evens[9];
        data[6] = lower_dct36 + lower_dct46;
        data[25] = lower_dct36 - lower_dct46;
        data[9] = upper_dct36 + upper_dct46;
        data[22] = upper_dct36 - upper_dct46;
        let tw7 = self.twiddles[7];
        let cosine_value7 = recursive_input_n1[7];
        let sine_value7 = -recursive_input_n3[7];
        let lower_dct47 = fmla(cosine_value7, tw7.re, sine_value7 * tw7.im);
        let upper_dct47 = fmla(cosine_value7, tw7.im, -sine_value7 * tw7.re);
        let lower_dct37 = evens[7];
        let upper_dct37 = evens[8];
        data[7] = lower_dct37 + lower_dct47;
        data[24] = lower_dct37 - lower_dct47;
        data[8] = upper_dct37 + upper_dct47;
        data[23] = upper_dct37 - upper_dct47;
    }
}

define_in_place_butterfly!(Dct3Butterfly32, 32);

#[derive(Clone)]
pub(crate) struct Dct3Butterfly64<T> {
    twiddles: [Complex<T>; 16],
    bf16: Dct3Butterfly16<T>,
    bf32: Arc<dyn PxdctExecutor<T> + Send + Sync>,
}

impl<T: DctSample + Dct3Factory> Default for Dct3Butterfly64<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf16: Dct3Butterfly16::default(),
            bf32: T::dct3_butterfly32(),
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 64 * 4).conj()),
        }
    }
}

impl<T: DctSample> Dct3Butterfly64<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut evens = [
            data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16],
            data[18], data[20], data[22], data[24], data[26], data[28], data[30], data[32],
            data[34], data[36], data[38], data[40], data[42], data[44], data[46], data[48],
            data[50], data[52], data[54], data[56], data[58], data[60], data[62],
        ];
        let mut recursive_input_n1 = [
            data[1] * T::TWO,
            data[3] + data[5],
            data[7] + data[9],
            data[11] + data[13],
            data[15] + data[17],
            data[19] + data[21],
            data[23] + data[25],
            data[27] + data[29],
            data[31] + data[33],
            data[35] + data[37],
            data[39] + data[41],
            data[43] + data[45],
            data[47] + data[49],
            data[51] + data[53],
            data[55] + data[57],
            data[59] + data[61],
        ];
        let mut recursive_input_n3 = [
            data[63] * T::TWO,
            data[59] - data[61],
            data[55] - data[57],
            data[51] - data[53],
            data[47] - data[49],
            data[43] - data[45],
            data[39] - data[41],
            data[35] - data[37],
            data[31] - data[33],
            data[27] - data[29],
            data[23] - data[25],
            data[19] - data[21],
            data[15] - data[17],
            data[11] - data[13],
            data[7] - data[9],
            data[3] - data[5],
        ];

        _ = self.bf32.execute(&mut evens);
        self.bf16
            .exec(&mut InPlaceStore::new(&mut recursive_input_n1));
        self.bf16
            .exec(&mut InPlaceStore::new(&mut recursive_input_n3));

        let tw0 = self.twiddles[0];
        let cosine_value0 = recursive_input_n1[0];
        let sine_value0 = recursive_input_n3[0];
        let lower_dct40 = fmla(cosine_value0, tw0.re, sine_value0 * tw0.im);
        let upper_dct40 = fmla(cosine_value0, tw0.im, -sine_value0 * tw0.re);
        let lower_dct30 = evens[0];
        let upper_dct30 = evens[31];
        data[0] = lower_dct30 + lower_dct40;
        data[63] = lower_dct30 - lower_dct40;
        data[31] = upper_dct30 + upper_dct40;
        data[32] = upper_dct30 - upper_dct40;
        let tw1 = self.twiddles[1];
        let cosine_value1 = recursive_input_n1[1];
        let sine_value1 = -recursive_input_n3[1];
        let lower_dct41 = fmla(cosine_value1, tw1.re, sine_value1 * tw1.im);
        let upper_dct41 = fmla(cosine_value1, tw1.im, -sine_value1 * tw1.re);
        let lower_dct31 = evens[1];
        let upper_dct31 = evens[30];
        data[1] = lower_dct31 + lower_dct41;
        data[62] = lower_dct31 - lower_dct41;
        data[30] = upper_dct31 + upper_dct41;
        data[33] = upper_dct31 - upper_dct41;
        let tw2 = self.twiddles[2];
        let cosine_value2 = recursive_input_n1[2];
        let sine_value2 = recursive_input_n3[2];
        let lower_dct42 = fmla(cosine_value2, tw2.re, sine_value2 * tw2.im);
        let upper_dct42 = fmla(cosine_value2, tw2.im, -sine_value2 * tw2.re);
        let lower_dct32 = evens[2];
        let upper_dct32 = evens[29];
        data[2] = lower_dct32 + lower_dct42;
        data[61] = lower_dct32 - lower_dct42;
        data[29] = upper_dct32 + upper_dct42;
        data[34] = upper_dct32 - upper_dct42;
        let tw3 = self.twiddles[3];
        let cosine_value3 = recursive_input_n1[3];
        let sine_value3 = -recursive_input_n3[3];
        let lower_dct43 = fmla(cosine_value3, tw3.re, sine_value3 * tw3.im);
        let upper_dct43 = fmla(cosine_value3, tw3.im, -sine_value3 * tw3.re);
        let lower_dct33 = evens[3];
        let upper_dct33 = evens[28];
        data[3] = lower_dct33 + lower_dct43;
        data[60] = lower_dct33 - lower_dct43;
        data[28] = upper_dct33 + upper_dct43;
        data[35] = upper_dct33 - upper_dct43;
        let tw4 = self.twiddles[4];
        let cosine_value4 = recursive_input_n1[4];
        let sine_value4 = recursive_input_n3[4];
        let lower_dct44 = fmla(cosine_value4, tw4.re, sine_value4 * tw4.im);
        let upper_dct44 = fmla(cosine_value4, tw4.im, -sine_value4 * tw4.re);
        let lower_dct34 = evens[4];
        let upper_dct34 = evens[27];
        data[4] = lower_dct34 + lower_dct44;
        data[59] = lower_dct34 - lower_dct44;
        data[27] = upper_dct34 + upper_dct44;
        data[36] = upper_dct34 - upper_dct44;
        let tw5 = self.twiddles[5];
        let cosine_value5 = recursive_input_n1[5];
        let sine_value5 = -recursive_input_n3[5];
        let lower_dct45 = fmla(cosine_value5, tw5.re, sine_value5 * tw5.im);
        let upper_dct45 = fmla(cosine_value5, tw5.im, -sine_value5 * tw5.re);
        let lower_dct35 = evens[5];
        let upper_dct35 = evens[26];
        data[5] = lower_dct35 + lower_dct45;
        data[58] = lower_dct35 - lower_dct45;
        data[26] = upper_dct35 + upper_dct45;
        data[37] = upper_dct35 - upper_dct45;
        let tw6 = self.twiddles[6];
        let cosine_value6 = recursive_input_n1[6];
        let sine_value6 = recursive_input_n3[6];
        let lower_dct46 = fmla(cosine_value6, tw6.re, sine_value6 * tw6.im);
        let upper_dct46 = fmla(cosine_value6, tw6.im, -sine_value6 * tw6.re);
        let lower_dct36 = evens[6];
        let upper_dct36 = evens[25];
        data[6] = lower_dct36 + lower_dct46;
        data[57] = lower_dct36 - lower_dct46;
        data[25] = upper_dct36 + upper_dct46;
        data[38] = upper_dct36 - upper_dct46;
        let tw7 = self.twiddles[7];
        let cosine_value7 = recursive_input_n1[7];
        let sine_value7 = -recursive_input_n3[7];
        let lower_dct47 = fmla(cosine_value7, tw7.re, sine_value7 * tw7.im);
        let upper_dct47 = fmla(cosine_value7, tw7.im, -sine_value7 * tw7.re);
        let lower_dct37 = evens[7];
        let upper_dct37 = evens[24];
        data[7] = lower_dct37 + lower_dct47;
        data[56] = lower_dct37 - lower_dct47;
        data[24] = upper_dct37 + upper_dct47;
        data[39] = upper_dct37 - upper_dct47;
        let tw8 = self.twiddles[8];
        let cosine_value8 = recursive_input_n1[8];
        let sine_value8 = recursive_input_n3[8];
        let lower_dct48 = fmla(cosine_value8, tw8.re, sine_value8 * tw8.im);
        let upper_dct48 = fmla(cosine_value8, tw8.im, -sine_value8 * tw8.re);
        let lower_dct38 = evens[8];
        let upper_dct38 = evens[23];
        data[8] = lower_dct38 + lower_dct48;
        data[55] = lower_dct38 - lower_dct48;
        data[23] = upper_dct38 + upper_dct48;
        data[40] = upper_dct38 - upper_dct48;
        let tw9 = self.twiddles[9];
        let cosine_value9 = recursive_input_n1[9];
        let sine_value9 = -recursive_input_n3[9];
        let lower_dct49 = fmla(cosine_value9, tw9.re, sine_value9 * tw9.im);
        let upper_dct49 = fmla(cosine_value9, tw9.im, -sine_value9 * tw9.re);
        let lower_dct39 = evens[9];
        let upper_dct39 = evens[22];
        data[9] = lower_dct39 + lower_dct49;
        data[54] = lower_dct39 - lower_dct49;
        data[22] = upper_dct39 + upper_dct49;
        data[41] = upper_dct39 - upper_dct49;
        let tw10 = self.twiddles[10];
        let cosine_value10 = recursive_input_n1[10];
        let sine_value10 = recursive_input_n3[10];
        let lower_dct410 = fmla(cosine_value10, tw10.re, sine_value10 * tw10.im);
        let upper_dct410 = fmla(cosine_value10, tw10.im, -sine_value10 * tw10.re);
        let lower_dct310 = evens[10];
        let upper_dct310 = evens[21];
        data[10] = lower_dct310 + lower_dct410;
        data[53] = lower_dct310 - lower_dct410;
        data[21] = upper_dct310 + upper_dct410;
        data[42] = upper_dct310 - upper_dct410;
        let tw11 = self.twiddles[11];
        let cosine_value11 = recursive_input_n1[11];
        let sine_value11 = -recursive_input_n3[11];
        let lower_dct411 = fmla(cosine_value11, tw11.re, sine_value11 * tw11.im);
        let upper_dct411 = fmla(cosine_value11, tw11.im, -sine_value11 * tw11.re);
        let lower_dct311 = evens[11];
        let upper_dct311 = evens[20];
        data[11] = lower_dct311 + lower_dct411;
        data[52] = lower_dct311 - lower_dct411;
        data[20] = upper_dct311 + upper_dct411;
        data[43] = upper_dct311 - upper_dct411;
        let tw12 = self.twiddles[12];
        let cosine_value12 = recursive_input_n1[12];
        let sine_value12 = recursive_input_n3[12];
        let lower_dct412 = fmla(cosine_value12, tw12.re, sine_value12 * tw12.im);
        let upper_dct412 = fmla(cosine_value12, tw12.im, -sine_value12 * tw12.re);
        let lower_dct312 = evens[12];
        let upper_dct312 = evens[19];
        data[12] = lower_dct312 + lower_dct412;
        data[51] = lower_dct312 - lower_dct412;
        data[19] = upper_dct312 + upper_dct412;
        data[44] = upper_dct312 - upper_dct412;
        let tw13 = self.twiddles[13];
        let cosine_value13 = recursive_input_n1[13];
        let sine_value13 = -recursive_input_n3[13];
        let lower_dct413 = fmla(cosine_value13, tw13.re, sine_value13 * tw13.im);
        let upper_dct413 = fmla(cosine_value13, tw13.im, -sine_value13 * tw13.re);
        let lower_dct313 = evens[13];
        let upper_dct313 = evens[18];
        data[13] = lower_dct313 + lower_dct413;
        data[50] = lower_dct313 - lower_dct413;
        data[18] = upper_dct313 + upper_dct413;
        data[45] = upper_dct313 - upper_dct413;
        let tw14 = self.twiddles[14];
        let cosine_value14 = recursive_input_n1[14];
        let sine_value14 = recursive_input_n3[14];
        let lower_dct414 = fmla(cosine_value14, tw14.re, sine_value14 * tw14.im);
        let upper_dct414 = fmla(cosine_value14, tw14.im, -sine_value14 * tw14.re);
        let lower_dct314 = evens[14];
        let upper_dct314 = evens[17];
        data[14] = lower_dct314 + lower_dct414;
        data[49] = lower_dct314 - lower_dct414;
        data[17] = upper_dct314 + upper_dct414;
        data[46] = upper_dct314 - upper_dct414;
        let tw15 = self.twiddles[15];
        let cosine_value15 = recursive_input_n1[15];
        let sine_value15 = -recursive_input_n3[15];
        let lower_dct415 = fmla(cosine_value15, tw15.re, sine_value15 * tw15.im);
        let upper_dct415 = fmla(cosine_value15, tw15.im, -sine_value15 * tw15.re);
        let lower_dct315 = evens[15];
        let upper_dct315 = evens[16];
        data[15] = lower_dct315 + lower_dct415;
        data[48] = lower_dct315 - lower_dct415;
        data[16] = upper_dct315 + upper_dct415;
        data[47] = upper_dct315 - upper_dct415;
    }
}

define_in_place_butterfly!(Dct3Butterfly64, 64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_dct3;
    use rand::RngExt;

    gen_test_butterfly!(test_bf_dct3_2, f64, Dct3Butterfly2, 2, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_4, f64, Dct3Butterfly4, 4, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_8, f64, Dct3Butterfly8, 8, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_16, f64, Dct3Butterfly16, 16, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_32, f64, Dct3Butterfly32, 32, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_64, f64, Dct3Butterfly64, 64, 1e-7, naive_dct3);
}
