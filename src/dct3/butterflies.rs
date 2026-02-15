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
use crate::dct3::prime_butterflies::Dct3Butterfly3;
use crate::dct3::{
    Dct3Butterfly2, Dct3Butterfly4, Dct3Butterfly5, Dct3Butterfly7, Dct3Butterfly8, Dct3Butterfly13,
};
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly6<T> {
    bf3: Dct3Butterfly3<T>,
    bf2: Dct3Butterfly2<T>,
}

impl<T: DctSample> Default for Dct3Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf3: Dct3Butterfly3::default(),
            bf2: Dct3Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 2x3
        let mut col0 = [data[0], data[3]];
        let mut col1 = [data[2] * T::TWO, data[1] + data[5]];
        let mut col2 = [data[4] * T::TWO, data[1] - data[5]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col0));
        self.bf2.exec(&mut InPlaceStore::new(&mut col1));
        self.bf2.exec(&mut InPlaceStore::new(&mut col2));

        let mut row0 = [col0[0] * T::TWO, col1[0], col2[0]];
        let mut row1 = [col0[1] * T::TWO, col1[1], col2[1]];

        self.bf3.exec(&mut InPlaceStore::new(&mut row0));
        self.bf3.exec(&mut InPlaceStore::new(&mut row1));

        data[0] = row0[0];
        data[4] = row0[1];
        data[3] = row0[2];
        data[5] = row1[0];
        data[1] = row1[1];
        data[2] = row1[2];
    }
}

define_in_place_butterfly!(Dct3Butterfly6, 6);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly9<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: T,
    twiddle3: Complex<T>,
}

impl<T: DctSample> Default for Dct3Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 36).conj(),
            twiddle1: compute_twiddle(2, 36).conj(),
            twiddle2: compute_twiddle(3, 36).re,
            twiddle3: compute_twiddle(4, 36).conj(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0] * T::HALF;
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];
        let x7 = data[7];
        let x8 = data[8];

        let t0 = fmla(self.twiddle0.im, x8, x0);
        let t1 = fmla(-T::HALF, x8, x0);
        let t2 = fmla(self.twiddle3.re, x8, x0);
        let t3 = fmla(-self.twiddle1.re, x8, x0);

        let t4 = self.twiddle1.im * x7;
        let t5 = self.twiddle2 * x7;
        let t6 = self.twiddle0.re * x7;
        let t7 = self.twiddle3.im * x7;
        let t8 = T::HALF * x6;
        let t9 = self.twiddle1.im * x5;

        let q0 = t8 + t0;
        let q1 = t6 - t9;
        let q2 = t1 - x6;
        let q3 = t8 + t3;

        let t10 = self.twiddle3.im * x5;
        let t11 = self.twiddle2 * x5;
        let t12 = self.twiddle0.re * x5;

        let q4 = t4 + t10;
        let q5 = t5 + t11;
        let q6 = t2 + t8;
        let q7 = t12 - t7;

        let t13 = self.twiddle3.re * x4;
        let t14 = T::HALF * x4;
        let t15 = self.twiddle1.re * x4;
        let t16 = self.twiddle0.im * x4;

        let q8 = q0 + t13;
        let q9 = q2 - t14;
        let q10 = q6 - t15;
        let q11 = q3 + t16;

        let t17 = self.twiddle2 * x3;

        let q12 = q4 + t17;
        let q13 = q1 - t17;
        let q14 = q7 - t17;

        let t18 = self.twiddle1.re * x2;
        let t19 = T::HALF * x2;
        let t20 = self.twiddle0.im * x2;
        let t21 = self.twiddle3.re * x2;

        let q15 = q8 + t18;
        let q16 = q9 + t19;
        let q17 = q10 - t20;
        let q18 = q11 - t21;

        let y0 = fmla(self.twiddle0.re, x1, q15 + q12);
        let y1 = fmla(self.twiddle2, x1, q16 - q5);
        let y2 = fmla(self.twiddle3.im, x1, q13 + q17);
        let y3 = fmla(self.twiddle1.im, x1, q18 + q14);
        let y4 = x0 - x2 + x4 - x6 + x8;
        let y5 = fmla(-self.twiddle1.im, x1, q18 - q14);
        let y6 = fmla(-self.twiddle3.im, x1, q17 - q13);
        let y7 = fmla(-self.twiddle2, x1, q16 + q5);
        let y8 = fmla(-self.twiddle0.re, x1, q15 - q12);

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
        data[5] = y5;
        data[6] = y6;
        data[7] = y7;
        data[8] = y8;
    }
}

define_in_place_butterfly!(Dct3Butterfly9, 9);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly10<T> {
    bf2: Dct3Butterfly2<T>,
    bf5: Dct3Butterfly5<T>,
}

impl<T: DctSample> Default for Dct3Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf2: Dct3Butterfly2::default(),
            bf5: Dct3Butterfly5::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 2x5
        let mut col0 = [data[0], data[5]];
        let mut col1 = [data[2] * T::TWO, data[3] + data[7]];
        let mut col2 = [data[4] * T::TWO, data[1] + data[9]];
        let mut col3 = [data[6] * T::TWO, data[1] - data[9]];
        let mut col4 = [data[8] * T::TWO, data[3] - data[7]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col0));
        self.bf2.exec(&mut InPlaceStore::new(&mut col1));
        self.bf2.exec(&mut InPlaceStore::new(&mut col2));
        self.bf2.exec(&mut InPlaceStore::new(&mut col3));
        self.bf2.exec(&mut InPlaceStore::new(&mut col4));

        let mut row0 = [col0[0] * T::TWO, col1[0], col2[0], col3[0], col4[0]];
        let mut row1 = [col0[1] * T::TWO, col1[1], col2[1], col3[1], col4[1]];

        self.bf5.exec(&mut InPlaceStore::new(&mut row0));
        self.bf5.exec(&mut InPlaceStore::new(&mut row1));

        data[0] = row0[0];
        data[8] = row0[1];
        data[7] = row0[2];
        data[3] = row0[3];
        data[4] = row0[4];
        data[9] = row1[0];
        data[1] = row1[1];
        data[2] = row1[2];
        data[6] = row1[3];
        data[5] = row1[4];
    }
}

define_in_place_butterfly!(Dct3Butterfly10, 10);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly12<T> {
    bf3: Dct3Butterfly3<T>,
    bf4: Dct3Butterfly4<T>,
}

impl<T: DctSample> Default for Dct3Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf3: Dct3Butterfly3::default(),
            bf4: Dct3Butterfly4::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated Prime-Factor algorithm for size 4x3
        let mut col0 = [data[0], data[3], data[6], data[9]];
        let mut col1 = [
            data[4] * T::TWO,
            data[1] + data[7],
            data[2] + data[10],
            data[5] - data[11],
        ];
        let mut col2 = [
            data[8] * T::TWO,
            data[5] + data[11],
            data[2] - data[10],
            data[1] - data[7],
        ];

        self.bf4.exec(&mut InPlaceStore::new(&mut col0));
        self.bf4.exec(&mut InPlaceStore::new(&mut col1));
        self.bf4.exec(&mut InPlaceStore::new(&mut col2));

        let mut row0 = [col0[0] * T::TWO, col1[0], col2[0]];
        let mut row1 = [col0[1] * T::TWO, col1[1], col2[1]];
        let mut row2 = [col0[2] * T::TWO, col1[2], col2[2]];
        let mut row3 = [col0[3] * T::TWO, col1[3], col2[3]];

        self.bf3.exec(&mut InPlaceStore::new(&mut row0));
        self.bf3.exec(&mut InPlaceStore::new(&mut row1));
        self.bf3.exec(&mut InPlaceStore::new(&mut row2));
        self.bf3.exec(&mut InPlaceStore::new(&mut row3));

        data[0] = row0[0];
        data[7] = row0[1];
        data[8] = row0[2];
        data[6] = row1[0];
        data[1] = row1[1];
        data[9] = row1[2];
        data[5] = row2[0];
        data[10] = row2[1];
        data[2] = row2[2];
        data[11] = row3[0];
        data[4] = row3[1];
        data[3] = row3[2];
    }
}

define_in_place_butterfly!(Dct3Butterfly12, 12);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly14<T> {
    bf2: Dct3Butterfly2<T>,
    bf7: Dct3Butterfly7<T>,
}

impl<T: DctSample> Default for Dct3Butterfly14<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf2: Dct3Butterfly2::default(),
            bf7: Dct3Butterfly7::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly14<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 2x7
        let mut col0 = [data[0], data[7]];
        let mut col1 = [data[2] * T::TWO, data[5] + data[9]];
        let mut col2 = [data[4] * T::TWO, data[3] + data[11]];
        self.bf2.exec(&mut InPlaceStore::new(&mut col0));
        self.bf2.exec(&mut InPlaceStore::new(&mut col1));
        self.bf2.exec(&mut InPlaceStore::new(&mut col2));
        let mut col3 = [data[6] * T::TWO, data[1] + data[13]];
        let mut col4 = [data[8] * T::TWO, data[1] - data[13]];
        let mut col5 = [data[10] * T::TWO, data[3] - data[11]];
        let mut col6 = [data[12] * T::TWO, data[5] - data[9]];
        self.bf2.exec(&mut InPlaceStore::new(&mut col3));
        self.bf2.exec(&mut InPlaceStore::new(&mut col4));
        self.bf2.exec(&mut InPlaceStore::new(&mut col5));
        self.bf2.exec(&mut InPlaceStore::new(&mut col6));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
            col6[0],
        ];
        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
            col6[1],
        ];

        self.bf7.exec(&mut InPlaceStore::new(&mut row0));
        self.bf7.exec(&mut InPlaceStore::new(&mut row1));

        data[0] = row0[0];
        data[12] = row0[1];
        data[11] = row0[2];
        data[3] = row0[3];
        data[4] = row0[4];
        data[8] = row0[5];
        data[7] = row0[6];
        data[13] = row1[0];
        data[1] = row1[1];
        data[2] = row1[2];
        data[10] = row1[3];
        data[9] = row1[4];
        data[5] = row1[5];
        data[6] = row1[6];
    }
}

define_in_place_butterfly!(Dct3Butterfly14, 14);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly15<T> {
    bf3: Dct3Butterfly3<T>,
    bf5: Dct3Butterfly5<T>,
}

impl<T: DctSample> Default for Dct3Butterfly15<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf3: Dct3Butterfly3::default(),
            bf5: Dct3Butterfly5::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly15<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 3x5
        let mut col0 = [data[0], data[5], data[10]];
        let mut col1 = [data[3] * T::TWO, data[2] + data[8], data[7] + data[13]];
        let mut col2 = [data[6] * T::TWO, data[1] + data[11], data[4] - data[14]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col0));
        self.bf3.exec(&mut InPlaceStore::new(&mut col1));
        self.bf3.exec(&mut InPlaceStore::new(&mut col2));
        let mut col3 = [data[9] * T::TWO, data[4] + data[14], data[1] - data[11]];
        let mut col4 = [data[12] * T::TWO, data[7] - data[13], data[2] - data[8]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col3));
        self.bf3.exec(&mut InPlaceStore::new(&mut col4));

        let mut row0 = [col0[0] * T::TWO, col1[0], col2[0], col3[0], col4[0]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row0));
        let mut row1 = [col0[1] * T::TWO, col1[1], col2[1], col3[1], col4[1]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row1));
        let mut row2 = [col0[2] * T::TWO, col1[2], col2[2], col3[2], col4[2]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row2));

        data[0] = row0[0];
        data[11] = row0[1];
        data[12] = row0[2];
        data[6] = row0[3];
        data[5] = row0[4];
        data[10] = row1[0];
        data[1] = row1[1];
        data[7] = row1[2];
        data[13] = row1[3];
        data[4] = row1[4];
        data[9] = row2[0];
        data[8] = row2[1];
        data[2] = row2[2];
        data[3] = row2[3];
        data[14] = row2[4];
    }
}

define_in_place_butterfly!(Dct3Butterfly15, 15);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly18<T> {
    bf2: Dct3Butterfly2<T>,
    bf9: Dct3Butterfly9<T>,
}

impl<T: DctSample> Default for Dct3Butterfly18<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf2: Dct3Butterfly2::default(),
            bf9: Dct3Butterfly9::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly18<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 2x9
        let mut col0 = [data[0], data[9]];
        let mut col1 = [data[2] * T::TWO, data[7] + data[11]];
        let mut col2 = [data[4] * T::TWO, data[5] + data[13]];
        let mut col3 = [data[6] * T::TWO, data[3] + data[15]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col0));
        self.bf2.exec(&mut InPlaceStore::new(&mut col1));
        self.bf2.exec(&mut InPlaceStore::new(&mut col2));
        self.bf2.exec(&mut InPlaceStore::new(&mut col3));

        let mut col4 = [data[8] * T::TWO, data[1] + data[17]];
        let mut col5 = [data[10] * T::TWO, data[1] - data[17]];
        let mut col6 = [data[12] * T::TWO, data[3] - data[15]];
        let mut col7 = [data[14] * T::TWO, data[5] - data[13]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col4));
        self.bf2.exec(&mut InPlaceStore::new(&mut col5));
        self.bf2.exec(&mut InPlaceStore::new(&mut col6));
        self.bf2.exec(&mut InPlaceStore::new(&mut col7));

        let mut col8 = [data[16] * T::TWO, data[7] - data[11]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col8));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
            col6[0],
            col7[0],
            col8[0],
        ];

        self.bf9.exec(&mut InPlaceStore::new(&mut row0));

        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
            col6[1],
            col7[1],
            col8[1],
        ];

        self.bf9.exec(&mut InPlaceStore::new(&mut row1));

        data[0] = row0[0];
        data[16] = row0[1];
        data[15] = row0[2];
        data[3] = row0[3];
        data[4] = row0[4];
        data[12] = row0[5];
        data[11] = row0[6];
        data[7] = row0[7];
        data[8] = row0[8];
        data[17] = row1[0];
        data[1] = row1[1];
        data[2] = row1[2];
        data[14] = row1[3];
        data[13] = row1[4];
        data[5] = row1[5];
        data[6] = row1[6];
        data[10] = row1[7];
        data[9] = row1[8];
    }
}

define_in_place_butterfly!(Dct3Butterfly18, 18);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly20<T> {
    bf4: Dct3Butterfly4<T>,
    bf5: Dct3Butterfly5<T>,
}

impl<T: DctSample> Default for Dct3Butterfly20<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: Dct3Butterfly4::default(),
            bf5: Dct3Butterfly5::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly20<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 4x5
        let mut col0 = [data[0], data[5], data[10], data[15]];
        let mut col1 = [
            data[4] * T::TWO,
            data[1] + data[9],
            data[6] + data[14],
            data[11] + data[19],
        ];

        self.bf4.exec(&mut InPlaceStore::new(&mut col0));
        self.bf4.exec(&mut InPlaceStore::new(&mut col1));

        let mut col2 = [
            data[8] * T::TWO,
            data[3] + data[13],
            data[2] + data[18],
            data[7] - data[17],
        ];
        let mut col3 = [
            data[12] * T::TWO,
            data[7] + data[17],
            data[2] - data[18],
            data[3] - data[13],
        ];

        self.bf4.exec(&mut InPlaceStore::new(&mut col2));
        self.bf4.exec(&mut InPlaceStore::new(&mut col3));

        let mut col4 = [
            data[16] * T::TWO,
            data[11] - data[19],
            data[6] - data[14],
            data[1] - data[9],
        ];

        self.bf4.exec(&mut InPlaceStore::new(&mut col4));

        let mut row0 = [col0[0] * T::TWO, col1[0], col2[0], col3[0], col4[0]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row0));
        let mut row1 = [col0[1] * T::TWO, col1[1], col2[1], col3[1], col4[1]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row1));
        let mut row2 = [col0[2] * T::TWO, col1[2], col2[2], col3[2], col4[2]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row2));
        let mut row3 = [col0[3] * T::TWO, col1[3], col2[3], col3[3], col4[3]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row3));

        data[0] = row0[0];
        data[8] = row0[1];
        data[7] = row0[2];
        data[16] = row0[3];
        data[15] = row0[4];
        data[9] = row1[0];
        data[1] = row1[1];
        data[17] = row1[2];
        data[6] = row1[3];
        data[14] = row1[4];
        data[10] = row2[0];
        data[18] = row2[1];
        data[2] = row2[2];
        data[13] = row2[3];
        data[5] = row2[4];
        data[19] = row3[0];
        data[11] = row3[1];
        data[12] = row3[2];
        data[3] = row3[3];
        data[4] = row3[4];
    }
}

define_in_place_butterfly!(Dct3Butterfly20, 20);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly21<T> {
    bf3: Dct3Butterfly3<T>,
    bf7: Dct3Butterfly7<T>,
}

impl<T: DctSample> Default for Dct3Butterfly21<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf3: Dct3Butterfly3::default(),
            bf7: Dct3Butterfly7::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly21<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 3x7
        let mut col0 = [data[0], data[7], data[14]];
        let mut col1 = [data[3] * T::TWO, data[4] + data[10], data[11] + data[17]];
        let mut col2 = [data[6] * T::TWO, data[1] + data[13], data[8] + data[20]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col0));
        self.bf3.exec(&mut InPlaceStore::new(&mut col1));
        self.bf3.exec(&mut InPlaceStore::new(&mut col2));
        let mut col3 = [data[9] * T::TWO, data[2] + data[16], data[5] - data[19]];
        let mut col4 = [data[12] * T::TWO, data[5] + data[19], data[2] - data[16]];
        let mut col5 = [data[15] * T::TWO, data[8] - data[20], data[1] - data[13]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col3));
        self.bf3.exec(&mut InPlaceStore::new(&mut col4));
        self.bf3.exec(&mut InPlaceStore::new(&mut col5));

        let mut col6 = [data[18] * T::TWO, data[11] - data[17], data[4] - data[10]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col6));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
            col6[0],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row0));
        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
            col6[1],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row1));
        let mut row2 = [
            col0[2] * T::TWO,
            col1[2],
            col2[2],
            col3[2],
            col4[2],
            col5[2],
            col6[2],
        ];

        self.bf7.exec(&mut InPlaceStore::new(&mut row2));

        data[0] = row0[0];
        data[12] = row0[1];
        data[11] = row0[2];
        data[17] = row0[3];
        data[18] = row0[4];
        data[5] = row0[5];
        data[6] = row0[6];
        data[13] = row1[0];
        data[1] = row1[1];
        data[16] = row1[2];
        data[10] = row1[3];
        data[4] = row1[4];
        data[19] = row1[5];
        data[7] = row1[6];
        data[14] = row2[0];
        data[15] = row2[1];
        data[2] = row2[2];
        data[3] = row2[3];
        data[9] = row2[4];
        data[8] = row2[5];
        data[20] = row2[6];
    }
}

define_in_place_butterfly!(Dct3Butterfly21, 21);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly24<T> {
    bf3: Dct3Butterfly3<T>,
    bf8: Dct3Butterfly8<T>,
}

impl<T: DctSample> Default for Dct3Butterfly24<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf3: Dct3Butterfly3::default(),
            bf8: Dct3Butterfly8::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly24<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 3x8
        let mut col0 = [data[0], data[8], data[16]];
        let mut col1 = [data[3] * T::TWO, data[5] + data[11], data[13] + data[19]];
        let mut col2 = [data[6] * T::TWO, data[2] + data[14], data[10] + data[22]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col0));
        self.bf3.exec(&mut InPlaceStore::new(&mut col1));
        self.bf3.exec(&mut InPlaceStore::new(&mut col2));

        let mut col3 = [data[9] * T::TWO, data[1] + data[17], data[7] - data[23]];
        let mut col4 = [data[12] * T::TWO, data[4] + data[20], data[4] - data[20]];
        let mut col5 = [data[15] * T::TWO, data[7] + data[23], data[1] - data[17]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col3));
        self.bf3.exec(&mut InPlaceStore::new(&mut col4));
        self.bf3.exec(&mut InPlaceStore::new(&mut col5));

        let mut col6 = [data[18] * T::TWO, data[10] - data[22], data[2] - data[14]];
        let mut col7 = [data[21] * T::TWO, data[13] - data[19], data[5] - data[11]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col6));
        self.bf3.exec(&mut InPlaceStore::new(&mut col7));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
            col6[0],
            col7[0],
        ];
        self.bf8.exec(&mut InPlaceStore::new(&mut row0));
        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
            col6[1],
            col7[1],
        ];
        self.bf8.exec(&mut InPlaceStore::new(&mut row1));
        let mut row2 = [
            col0[2] * T::TWO,
            col1[2],
            col2[2],
            col3[2],
            col4[2],
            col5[2],
            col6[2],
            col7[2],
        ];

        self.bf8.exec(&mut InPlaceStore::new(&mut row2));

        data[0] = row0[0];
        data[17] = row0[1];
        data[18] = row0[2];
        data[12] = row0[3];
        data[11] = row0[4];
        data[5] = row0[5];
        data[6] = row0[6];
        data[23] = row0[7];
        data[16] = row1[0];
        data[1] = row1[1];
        data[13] = row1[2];
        data[19] = row1[3];
        data[4] = row1[4];
        data[10] = row1[5];
        data[22] = row1[6];
        data[7] = row1[7];
        data[15] = row2[0];
        data[14] = row2[1];
        data[2] = row2[2];
        data[3] = row2[3];
        data[20] = row2[4];
        data[21] = row2[5];
        data[9] = row2[6];
        data[8] = row2[7];
    }
}

define_in_place_butterfly!(Dct3Butterfly24, 24);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly26<T> {
    bf2: Dct3Butterfly2<T>,
    bf13: Dct3Butterfly13<T>,
}

impl<T: DctSample> Default for Dct3Butterfly26<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf2: Dct3Butterfly2::default(),
            bf13: Dct3Butterfly13::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly26<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 2x13
        let mut col0 = [data[0], data[13]];
        let mut col1 = [data[2] * T::TWO, data[11] + data[15]];
        let mut col2 = [data[4] * T::TWO, data[9] + data[17]];
        let mut col3 = [data[6] * T::TWO, data[7] + data[19]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col0));
        self.bf2.exec(&mut InPlaceStore::new(&mut col1));
        self.bf2.exec(&mut InPlaceStore::new(&mut col2));
        self.bf2.exec(&mut InPlaceStore::new(&mut col3));

        let mut col4 = [data[8] * T::TWO, data[5] + data[21]];
        let mut col5 = [data[10] * T::TWO, data[3] + data[23]];
        let mut col6 = [data[12] * T::TWO, data[1] + data[25]];
        let mut col7 = [data[14] * T::TWO, data[1] - data[25]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col4));
        self.bf2.exec(&mut InPlaceStore::new(&mut col5));
        self.bf2.exec(&mut InPlaceStore::new(&mut col6));
        self.bf2.exec(&mut InPlaceStore::new(&mut col7));

        let mut col8 = [data[16] * T::TWO, data[3] - data[23]];
        let mut col9 = [data[18] * T::TWO, data[5] - data[21]];
        let mut col10 = [data[20] * T::TWO, data[7] - data[19]];
        let mut col11 = [data[22] * T::TWO, data[9] - data[17]];
        let mut col12 = [data[24] * T::TWO, data[11] - data[15]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col8));
        self.bf2.exec(&mut InPlaceStore::new(&mut col9));
        self.bf2.exec(&mut InPlaceStore::new(&mut col10));
        self.bf2.exec(&mut InPlaceStore::new(&mut col11));
        self.bf2.exec(&mut InPlaceStore::new(&mut col12));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
            col6[0],
            col7[0],
            col8[0],
            col9[0],
            col10[0],
            col11[0],
            col12[0],
        ];
        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
            col6[1],
            col7[1],
            col8[1],
            col9[1],
            col10[1],
            col11[1],
            col12[1],
        ];

        self.bf13.exec(&mut InPlaceStore::new(&mut row0));
        self.bf13.exec(&mut InPlaceStore::new(&mut row1));

        data[0] = row0[0];
        data[24] = row0[1];
        data[23] = row0[2];
        data[3] = row0[3];
        data[4] = row0[4];
        data[20] = row0[5];
        data[19] = row0[6];
        data[7] = row0[7];
        data[8] = row0[8];
        data[16] = row0[9];
        data[15] = row0[10];
        data[11] = row0[11];
        data[12] = row0[12];
        data[25] = row1[0];
        data[1] = row1[1];
        data[2] = row1[2];
        data[22] = row1[3];
        data[21] = row1[4];
        data[5] = row1[5];
        data[6] = row1[6];
        data[18] = row1[7];
        data[17] = row1[8];
        data[9] = row1[9];
        data[10] = row1[10];
        data[14] = row1[11];
        data[13] = row1[12];
    }
}

define_in_place_butterfly!(Dct3Butterfly26, 26);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly28<T> {
    bf4: Dct3Butterfly4<T>,
    bf7: Dct3Butterfly7<T>,
}

impl<T: DctSample> Default for Dct3Butterfly28<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: Dct3Butterfly4::default(),
            bf7: Dct3Butterfly7::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly28<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 4x7
        let mut col0 = [data[0], data[7], data[14], data[21]];
        let mut col1 = [
            data[4] * T::TWO,
            data[3] + data[11],
            data[10] + data[18],
            data[17] + data[25],
        ];
        let mut col2 = [
            data[8] * T::TWO,
            data[1] + data[15],
            data[6] + data[22],
            data[13] - data[27],
        ];

        self.bf4.exec(&mut InPlaceStore::new(&mut col0));
        self.bf4.exec(&mut InPlaceStore::new(&mut col1));
        self.bf4.exec(&mut InPlaceStore::new(&mut col2));

        let mut col3 = [
            data[12] * T::TWO,
            data[5] + data[19],
            data[2] + data[26],
            data[9] - data[23],
        ];
        let mut col4 = [
            data[16] * T::TWO,
            data[9] + data[23],
            data[2] - data[26],
            data[5] - data[19],
        ];
        let mut col5 = [
            data[20] * T::TWO,
            data[13] + data[27],
            data[6] - data[22],
            data[1] - data[15],
        ];

        self.bf4.exec(&mut InPlaceStore::new(&mut col3));
        self.bf4.exec(&mut InPlaceStore::new(&mut col4));
        self.bf4.exec(&mut InPlaceStore::new(&mut col5));

        let mut col6 = [
            data[24] * T::TWO,
            data[17] - data[25],
            data[10] - data[18],
            data[3] - data[11],
        ];

        self.bf4.exec(&mut InPlaceStore::new(&mut col6));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
            col6[0],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row0));
        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
            col6[1],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row1));
        let mut row2 = [
            col0[2] * T::TWO,
            col1[2],
            col2[2],
            col3[2],
            col4[2],
            col5[2],
            col6[2],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row2));
        let mut row3 = [
            col0[3] * T::TWO,
            col1[3],
            col2[3],
            col3[3],
            col4[3],
            col5[3],
            col6[3],
        ];

        self.bf7.exec(&mut InPlaceStore::new(&mut row3));

        data[0] = row0[0];
        data[15] = row0[1];
        data[16] = row0[2];
        data[24] = row0[3];
        data[23] = row0[4];
        data[8] = row0[5];
        data[7] = row0[6];
        data[14] = row1[0];
        data[1] = row1[1];
        data[25] = row1[2];
        data[17] = row1[3];
        data[9] = row1[4];
        data[22] = row1[5];
        data[6] = row1[6];
        data[13] = row2[0];
        data[26] = row2[1];
        data[2] = row2[2];
        data[10] = row2[3];
        data[18] = row2[4];
        data[5] = row2[5];
        data[21] = row2[6];
        data[27] = row3[0];
        data[12] = row3[1];
        data[11] = row3[2];
        data[3] = row3[3];
        data[4] = row3[4];
        data[19] = row3[5];
        data[20] = row3[6];
    }
}

define_in_place_butterfly!(Dct3Butterfly28, 28);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly30<T> {
    bf5: Dct3Butterfly5<T>,
    bf6: Dct3Butterfly6<T>,
}

impl<T: DctSample> Default for Dct3Butterfly30<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf5: Dct3Butterfly5::default(),
            bf6: Dct3Butterfly6::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly30<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 5x6
        let mut col0 = [data[0], data[6], data[12], data[18], data[24]];
        let mut col1 = [
            data[5] * T::TWO,
            data[1] + data[11],
            data[7] + data[17],
            data[13] + data[23],
            data[19] + data[29],
        ];

        self.bf5.exec(&mut InPlaceStore::new(&mut col0));
        self.bf5.exec(&mut InPlaceStore::new(&mut col1));

        let mut col2 = [
            data[10] * T::TWO,
            data[4] + data[16],
            data[2] + data[22],
            data[8] + data[28],
            data[14] - data[26],
        ];
        let mut col3 = [
            data[15] * T::TWO,
            data[9] + data[21],
            data[3] + data[27],
            data[3] - data[27],
            data[9] - data[21],
        ];

        self.bf5.exec(&mut InPlaceStore::new(&mut col2));
        self.bf5.exec(&mut InPlaceStore::new(&mut col3));

        let mut col4 = [
            data[20] * T::TWO,
            data[14] + data[26],
            data[8] - data[28],
            data[2] - data[22],
            data[4] - data[16],
        ];
        let mut col5 = [
            data[25] * T::TWO,
            data[19] - data[29],
            data[13] - data[23],
            data[7] - data[17],
            data[1] - data[11],
        ];

        self.bf5.exec(&mut InPlaceStore::new(&mut col4));
        self.bf5.exec(&mut InPlaceStore::new(&mut col5));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
        ];
        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
        ];

        self.bf6.exec(&mut InPlaceStore::new(&mut row0));
        self.bf6.exec(&mut InPlaceStore::new(&mut row1));

        let mut row2 = [
            col0[2] * T::TWO,
            col1[2],
            col2[2],
            col3[2],
            col4[2],
            col5[2],
        ];
        let mut row3 = [
            col0[3] * T::TWO,
            col1[3],
            col2[3],
            col3[3],
            col4[3],
            col5[3],
        ];

        self.bf6.exec(&mut InPlaceStore::new(&mut row2));
        self.bf6.exec(&mut InPlaceStore::new(&mut row3));

        let mut row4 = [
            col0[4] * T::TWO,
            col1[4],
            col2[4],
            col3[4],
            col4[4],
            col5[4],
        ];

        self.bf6.exec(&mut InPlaceStore::new(&mut row4));

        data[0] = row0[0];
        data[10] = row0[1];
        data[9] = row0[2];
        data[20] = row0[3];
        data[19] = row0[4];
        data[29] = row0[5];
        data[11] = row1[0];
        data[1] = row1[1];
        data[21] = row1[2];
        data[8] = row1[3];
        data[28] = row1[4];
        data[18] = row1[5];
        data[12] = row2[0];
        data[22] = row2[1];
        data[2] = row2[2];
        data[27] = row2[3];
        data[7] = row2[4];
        data[17] = row2[5];
        data[23] = row3[0];
        data[13] = row3[1];
        data[26] = row3[2];
        data[3] = row3[3];
        data[16] = row3[4];
        data[6] = row3[5];
        data[24] = row4[0];
        data[25] = row4[1];
        data[14] = row4[2];
        data[15] = row4[3];
        data[4] = row4[4];
        data[5] = row4[5];
    }
}

define_in_place_butterfly!(Dct3Butterfly30, 30);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly36<T> {
    bf4: Dct3Butterfly4<T>,
    bf9: Dct3Butterfly9<T>,
}

impl<T: DctSample> Default for Dct3Butterfly36<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: Dct3Butterfly4::default(),
            bf9: Dct3Butterfly9::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly36<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 4x9
        let mut col0 = [data[0], data[9], data[18], data[27]];
        let mut col1 = [
            data[4] * T::TWO,
            data[5] + data[13],
            data[14] + data[22],
            data[23] + data[31],
        ];
        self.bf4.exec(&mut InPlaceStore::new(&mut col0));
        self.bf4.exec(&mut InPlaceStore::new(&mut col1));
        let mut col2 = [
            data[8] * T::TWO,
            data[1] + data[17],
            data[10] + data[26],
            data[19] + data[35],
        ];
        let mut col3 = [
            data[12] * T::TWO,
            data[3] + data[21],
            data[6] + data[30],
            data[15] - data[33],
        ];
        self.bf4.exec(&mut InPlaceStore::new(&mut col2));
        self.bf4.exec(&mut InPlaceStore::new(&mut col3));
        let mut col4 = [
            data[16] * T::TWO,
            data[7] + data[25],
            data[2] + data[34],
            data[11] - data[29],
        ];
        let mut col5 = [
            data[20] * T::TWO,
            data[11] + data[29],
            data[2] - data[34],
            data[7] - data[25],
        ];
        self.bf4.exec(&mut InPlaceStore::new(&mut col4));
        self.bf4.exec(&mut InPlaceStore::new(&mut col5));
        let mut col6 = [
            data[24] * T::TWO,
            data[15] + data[33],
            data[6] - data[30],
            data[3] - data[21],
        ];
        let mut col7 = [
            data[28] * T::TWO,
            data[19] - data[35],
            data[10] - data[26],
            data[1] - data[17],
        ];
        self.bf4.exec(&mut InPlaceStore::new(&mut col6));
        self.bf4.exec(&mut InPlaceStore::new(&mut col7));
        let mut col8 = [
            data[32] * T::TWO,
            data[23] - data[31],
            data[14] - data[22],
            data[5] - data[13],
        ];
        self.bf4.exec(&mut InPlaceStore::new(&mut col8));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
            col6[0],
            col7[0],
            col8[0],
        ];
        self.bf9.exec(&mut InPlaceStore::new(&mut row0));
        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
            col6[1],
            col7[1],
            col8[1],
        ];
        self.bf9.exec(&mut InPlaceStore::new(&mut row1));
        let mut row2 = [
            col0[2] * T::TWO,
            col1[2],
            col2[2],
            col3[2],
            col4[2],
            col5[2],
            col6[2],
            col7[2],
            col8[2],
        ];
        self.bf9.exec(&mut InPlaceStore::new(&mut row2));
        let mut row3 = [
            col0[3] * T::TWO,
            col1[3],
            col2[3],
            col3[3],
            col4[3],
            col5[3],
            col6[3],
            col7[3],
            col8[3],
        ];
        self.bf9.exec(&mut InPlaceStore::new(&mut row3));

        data[0] = row0[0];
        data[16] = row0[1];
        data[15] = row0[2];
        data[32] = row0[3];
        data[31] = row0[4];
        data[23] = row0[5];
        data[24] = row0[6];
        data[7] = row0[7];
        data[8] = row0[8];
        data[17] = row1[0];
        data[1] = row1[1];
        data[33] = row1[2];
        data[14] = row1[3];
        data[22] = row1[4];
        data[30] = row1[5];
        data[6] = row1[6];
        data[25] = row1[7];
        data[9] = row1[8];
        data[18] = row2[0];
        data[34] = row2[1];
        data[2] = row2[2];
        data[21] = row2[3];
        data[13] = row2[4];
        data[5] = row2[5];
        data[29] = row2[6];
        data[10] = row2[7];
        data[26] = row2[8];
        data[35] = row3[0];
        data[19] = row3[1];
        data[20] = row3[2];
        data[3] = row3[3];
        data[4] = row3[4];
        data[12] = row3[5];
        data[11] = row3[6];
        data[28] = row3[7];
        data[27] = row3[8];
    }
}

define_in_place_butterfly!(Dct3Butterfly36, 36);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly35<T> {
    bf5: Dct3Butterfly5<T>,
    bf7: Dct3Butterfly7<T>,
}

impl<T: DctSample> Default for Dct3Butterfly35<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf5: Dct3Butterfly5::default(),
            bf7: Dct3Butterfly7::default(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly35<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated DCT-III Prime-Factor algorithm for size 5x7
        let mut col0 = [data[0], data[7], data[14], data[21], data[28]];
        let mut col1 = [
            data[5] * T::TWO,
            data[2] + data[12],
            data[9] + data[19],
            data[16] + data[26],
            data[23] + data[33],
        ];
        self.bf5.exec(&mut InPlaceStore::new(&mut col0));
        self.bf5.exec(&mut InPlaceStore::new(&mut col1));
        let mut col2 = [
            data[10] * T::TWO,
            data[3] + data[17],
            data[4] + data[24],
            data[11] + data[31],
            data[18] - data[32],
        ];
        let mut col3 = [
            data[15] * T::TWO,
            data[8] + data[22],
            data[1] + data[29],
            data[6] - data[34],
            data[13] - data[27],
        ];
        self.bf5.exec(&mut InPlaceStore::new(&mut col2));
        self.bf5.exec(&mut InPlaceStore::new(&mut col3));
        let mut col4 = [
            data[20] * T::TWO,
            data[13] + data[27],
            data[6] + data[34],
            data[1] - data[29],
            data[8] - data[22],
        ];
        let mut col5 = [
            data[25] * T::TWO,
            data[18] + data[32],
            data[11] - data[31],
            data[4] - data[24],
            data[3] - data[17],
        ];
        self.bf5.exec(&mut InPlaceStore::new(&mut col4));
        self.bf5.exec(&mut InPlaceStore::new(&mut col5));
        let mut col6 = [
            data[30] * T::TWO,
            data[23] - data[33],
            data[16] - data[26],
            data[9] - data[19],
            data[2] - data[12],
        ];
        self.bf5.exec(&mut InPlaceStore::new(&mut col6));

        let mut row0 = [
            col0[0] * T::TWO,
            col1[0],
            col2[0],
            col3[0],
            col4[0],
            col5[0],
            col6[0],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row0));
        let mut row1 = [
            col0[1] * T::TWO,
            col1[1],
            col2[1],
            col3[1],
            col4[1],
            col5[1],
            col6[1],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row1));
        let mut row2 = [
            col0[2] * T::TWO,
            col1[2],
            col2[2],
            col3[2],
            col4[2],
            col5[2],
            col6[2],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row2));
        let mut row3 = [
            col0[3] * T::TWO,
            col1[3],
            col2[3],
            col3[3],
            col4[3],
            col5[3],
            col6[3],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row3));
        let mut row4 = [
            col0[4] * T::TWO,
            col1[4],
            col2[4],
            col3[4],
            col4[4],
            col5[4],
            col6[4],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row4));

        data[0] = row0[0];
        data[29] = row0[1];
        data[30] = row0[2];
        data[10] = row0[3];
        data[9] = row0[4];
        data[19] = row0[5];
        data[20] = row0[6];
        data[28] = row1[0];
        data[1] = row1[1];
        data[11] = row1[2];
        data[31] = row1[3];
        data[18] = row1[4];
        data[8] = row1[5];
        data[21] = row1[6];
        data[27] = row2[0];
        data[12] = row2[1];
        data[2] = row2[2];
        data[17] = row2[3];
        data[32] = row2[4];
        data[22] = row2[5];
        data[7] = row2[6];
        data[13] = row3[0];
        data[26] = row3[1];
        data[16] = row3[2];
        data[3] = row3[3];
        data[23] = row3[4];
        data[33] = row3[5];
        data[6] = row3[6];
        data[14] = row4[0];
        data[15] = row4[1];
        data[25] = row4[2];
        data[24] = row4[3];
        data[4] = row4[4];
        data[5] = row4[5];
        data[34] = row4[6];
    }
}

define_in_place_butterfly!(Dct3Butterfly35, 35);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_dct3;
    use rand::Rng;

    gen_test_butterfly!(test_bf_dct3_4, f64, Dct3Butterfly4, 4, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_5, f64, Dct3Butterfly5, 5, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_6, f64, Dct3Butterfly6, 6, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_7, f64, Dct3Butterfly7, 7, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_9, f64, Dct3Butterfly9, 9, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_10, f64, Dct3Butterfly10, 10, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_12, f64, Dct3Butterfly12, 12, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_14, f64, Dct3Butterfly14, 14, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_15, f64, Dct3Butterfly15, 15, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_18, f64, Dct3Butterfly18, 18, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_20, f64, Dct3Butterfly20, 20, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_21, f64, Dct3Butterfly21, 21, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_24, f64, Dct3Butterfly24, 24, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_26, f64, Dct3Butterfly26, 26, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_28, f64, Dct3Butterfly28, 28, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_30, f64, Dct3Butterfly30, 30, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_35, f64, Dct3Butterfly35, 35, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_36, f64, Dct3Butterfly36, 36, 1e-7, naive_dct3);
}
