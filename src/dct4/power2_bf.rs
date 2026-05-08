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
use crate::dct2::power2_butterflies::{
    Dct2Butterfly2, Dct2Butterfly4, Dct2Butterfly8, Dct2Butterfly16,
};
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly2<T> {
    twiddle_sin: T,
    twiddle_cos: T,
}

impl<T: DctSample> Default for Dct4Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        let vals = (1f64.as_() / 8f64.as_()).sincos_pi();
        Self {
            twiddle_cos: vals.1,
            twiddle_sin: vals.0,
        }
    }
}

impl<T: DctSample> Dct4Butterfly2<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let y0 = fmla(data[0], self.twiddle_cos, data[1] * self.twiddle_sin);
        let y1 = fmla(data[0], self.twiddle_sin, -data[1] * self.twiddle_cos);
        data[0] = y0;
        data[1] = y1;
    }
}

define_in_place_butterfly!(Dct4Butterfly2, 2);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly4<T> {
    twiddles: [Complex<T>; 2],
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample> Default for Dct4Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle(2 * x + 1, 4 * 8).conj()),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let z0 = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[3]);
        let z1 = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[2]);
        let z2 = fmla(-self.twiddles[1].im, data[1], self.twiddles[1].re * data[2]);
        let z3 = fmla(self.twiddles[0].im, data[0], -self.twiddles[0].re * data[3]);

        let mut row0: [T; 2] = [z0, z1];
        let mut row1: [T; 2] = [z2, z3];

        self.bf2.exec(&mut InPlaceStore::new(&mut row0));
        self.bf2.exec(&mut InPlaceStore::new(&mut row1));

        data[0] = row0[0];
        data[1] = row0[1] - row1[1];
        data[2] = row1[1] + row0[1];
        data[3] = row1[0];
    }
}

define_in_place_butterfly!(Dct4Butterfly4, 4);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly8<T: DctSample> {
    twiddles: [Complex<T>; 4],
    bf4: Dct2Butterfly4<T>,
}

impl<T: DctSample> Default for Dct4Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle(2 * x + 1, 8 * 8).conj()),
            bf4: Dct2Butterfly4::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let z0 = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[7]);
        let z1 = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[6]);
        let z2 = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[5]);
        let z3 = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[4]);
        let z4 = fmla(-self.twiddles[3].im, data[3], self.twiddles[3].re * data[4]);
        let z5 = fmla(self.twiddles[2].im, data[2], -self.twiddles[2].re * data[5]);
        let z6 = fmla(-self.twiddles[1].im, data[1], self.twiddles[1].re * data[6]);
        let z7 = fmla(self.twiddles[0].im, data[0], -self.twiddles[0].re * data[7]);

        let mut row0: [T; 4] = [z0, z1, z2, z3];
        let mut row1: [T; 4] = [z4, z5, z6, z7];

        self.bf4.exec(&mut InPlaceStore::new(&mut row0));
        self.bf4.exec(&mut InPlaceStore::new(&mut row1));

        let out0 = row0[0];
        let out1 = row0[1] - row1[3];
        let out2 = row0[1] + row1[3];
        let out3 = row0[2] + row1[2];
        let out4 = row0[2] - row1[2];
        let out5 = row0[3] - row1[1];
        let out6 = row1[1] + row0[3];
        let out7 = row1[0];

        data[0] = out0;
        data[1] = out1;
        data[2] = out2;
        data[3] = out3;
        data[5] = out5;
        data[4] = out4;
        data[6] = out6;
        data[7] = out7;
    }
}

define_in_place_butterfly!(Dct4Butterfly8, 8);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly16<T: DctSample> {
    twiddles: [Complex<T>; 8],
    bf8: Dct2Butterfly8<T>,
}

impl<T: DctSample> Default for Dct4Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 16 * 8).conj()),
            bf8: Dct2Butterfly8::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 16 points
        let mut left = [T::zero(); 8];
        let mut right = [T::zero(); 8];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[15]);
        right[7] = fmla(
            -self.twiddles[0].re,
            data[15],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[14]);
        right[6] = fmla(
            self.twiddles[1].re,
            data[14],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[13]);
        right[5] = fmla(
            -self.twiddles[2].re,
            data[13],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[12]);
        right[4] = fmla(
            self.twiddles[3].re,
            data[12],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[11]);
        right[3] = fmla(
            -self.twiddles[4].re,
            data[11],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[10]);
        right[2] = fmla(
            self.twiddles[5].re,
            data[10],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[9]);
        right[1] = fmla(-self.twiddles[6].re, data[9], self.twiddles[6].im * data[6]);
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[8]);
        right[0] = fmla(self.twiddles[7].re, data[8], -self.twiddles[7].im * data[7]);
        self.bf8.exec(&mut InPlaceStore::new(&mut left));
        self.bf8.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[16 - 1] = right[0];
        data[1] = left[1] - right[7];
        data[2] = left[1] + right[7];
        data[13] = left[7] - right[1];
        data[14] = left[7] + right[1];
        data[3] = left[2] + right[6];
        data[4] = left[2] - right[6];
        data[11] = left[6] + right[2];
        data[12] = left[6] - right[2];
        data[5] = left[3] - right[5];
        data[6] = left[3] + right[5];
        data[9] = left[5] - right[3];
        data[10] = left[5] + right[3];
        data[7] = left[4] + right[4];
        data[8] = left[4] - right[4];
    }
}

define_in_place_butterfly!(Dct4Butterfly16, 16);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly32<T: DctSample> {
    twiddles: [Complex<T>; 16],
    bf16: Dct2Butterfly16<T>,
}
impl<T: DctSample> Default for Dct4Butterfly32<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 32 * 8).conj()),
            bf16: Dct2Butterfly16::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly32<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 32 points
        let mut left = [T::zero(); 16];
        let mut right = [T::zero(); 16];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[31]);
        right[15] = fmla(
            -self.twiddles[0].re,
            data[31],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[30]);
        right[14] = fmla(
            self.twiddles[1].re,
            data[30],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[29]);
        right[13] = fmla(
            -self.twiddles[2].re,
            data[29],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[28]);
        right[12] = fmla(
            self.twiddles[3].re,
            data[28],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[27]);
        right[11] = fmla(
            -self.twiddles[4].re,
            data[27],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[26]);
        right[10] = fmla(
            self.twiddles[5].re,
            data[26],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[25]);
        right[9] = fmla(
            -self.twiddles[6].re,
            data[25],
            self.twiddles[6].im * data[6],
        );
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[24]);
        right[8] = fmla(
            self.twiddles[7].re,
            data[24],
            -self.twiddles[7].im * data[7],
        );
        left[8] = fmla(self.twiddles[8].re, data[8], self.twiddles[8].im * data[23]);
        right[7] = fmla(
            -self.twiddles[8].re,
            data[23],
            self.twiddles[8].im * data[8],
        );
        left[9] = fmla(self.twiddles[9].re, data[9], self.twiddles[9].im * data[22]);
        right[6] = fmla(
            self.twiddles[9].re,
            data[22],
            -self.twiddles[9].im * data[9],
        );
        left[10] = fmla(
            self.twiddles[10].re,
            data[10],
            self.twiddles[10].im * data[21],
        );
        right[5] = fmla(
            -self.twiddles[10].re,
            data[21],
            self.twiddles[10].im * data[10],
        );
        left[11] = fmla(
            self.twiddles[11].re,
            data[11],
            self.twiddles[11].im * data[20],
        );
        right[4] = fmla(
            self.twiddles[11].re,
            data[20],
            -self.twiddles[11].im * data[11],
        );
        left[12] = fmla(
            self.twiddles[12].re,
            data[12],
            self.twiddles[12].im * data[19],
        );
        right[3] = fmla(
            -self.twiddles[12].re,
            data[19],
            self.twiddles[12].im * data[12],
        );
        left[13] = fmla(
            self.twiddles[13].re,
            data[13],
            self.twiddles[13].im * data[18],
        );
        right[2] = fmla(
            self.twiddles[13].re,
            data[18],
            -self.twiddles[13].im * data[13],
        );
        left[14] = fmla(
            self.twiddles[14].re,
            data[14],
            self.twiddles[14].im * data[17],
        );
        right[1] = fmla(
            -self.twiddles[14].re,
            data[17],
            self.twiddles[14].im * data[14],
        );
        left[15] = fmla(
            self.twiddles[15].re,
            data[15],
            self.twiddles[15].im * data[16],
        );
        right[0] = fmla(
            self.twiddles[15].re,
            data[16],
            -self.twiddles[15].im * data[15],
        );
        self.bf16.exec(&mut InPlaceStore::new(&mut left));
        self.bf16.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[32 - 1] = right[0];
        data[1] = left[1] - right[15];
        data[2] = left[1] + right[15];
        data[29] = left[15] - right[1];
        data[30] = left[15] + right[1];
        data[3] = left[2] + right[14];
        data[4] = left[2] - right[14];
        data[27] = left[14] + right[2];
        data[28] = left[14] - right[2];
        data[5] = left[3] - right[13];
        data[6] = left[3] + right[13];
        data[25] = left[13] - right[3];
        data[26] = left[13] + right[3];
        data[7] = left[4] + right[12];
        data[8] = left[4] - right[12];
        data[23] = left[12] + right[4];
        data[24] = left[12] - right[4];
        data[9] = left[5] - right[11];
        data[10] = left[5] + right[11];
        data[21] = left[11] - right[5];
        data[22] = left[11] + right[5];
        data[11] = left[6] + right[10];
        data[12] = left[6] - right[10];
        data[19] = left[10] + right[6];
        data[20] = left[10] - right[6];
        data[13] = left[7] - right[9];
        data[14] = left[7] + right[9];
        data[17] = left[9] - right[7];
        data[18] = left[9] + right[7];
        data[15] = left[8] + right[8];
        data[16] = left[8] - right[8];
    }
}

define_in_place_butterfly!(Dct4Butterfly32, 32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_dct4;
    use rand::RngExt;

    gen_test_butterfly!(test_bf_dct4_2, f64, Dct4Butterfly2, 2, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct4_4, f64, Dct4Butterfly4, 4, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct4_8, f64, Dct4Butterfly8, 8, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct4_16, f64, Dct4Butterfly16, 16, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct4_32, f64, Dct4Butterfly32, 32, 1e-7, naive_dct4);
}
