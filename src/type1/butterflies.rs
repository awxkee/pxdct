/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::bidirectional::BidirectionalStore;
use crate::mla::fmla;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct1Butterfly2<T> {
    phantom_data: PhantomData<T>,
}

impl<T: DctSample> Dct1Butterfly2<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let v0 = u0 + u1;
        let v1 = u0 - u1;
        data[0] = v0;
        data[1] = v1;
    }
}

define_in_place_butterfly!(Dct1Butterfly2, 2);

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct1Butterfly4<T> {
    phantom_data: PhantomData<T>,
}

impl<T: DctSample> Dct1Butterfly4<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let sum_0_3 = x0 + x3;
        let diff_0_3 = x0 - x3;
        let sum_1_2 = x1 + x2;
        let diff_1_2 = x1 - x2;
        data[0] = fmla(T::TWO, sum_1_2, sum_0_3);
        data[1] = diff_0_3 + diff_1_2;
        data[2] = sum_0_3 - sum_1_2;
        data[3] = fmla(-T::TWO, diff_1_2, diff_0_3);
    }
}

define_in_place_butterfly!(Dct1Butterfly4, 4);

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct1Butterfly5<T> {
    phantom_data: PhantomData<T>,
}

impl<T: DctSample> Dct1Butterfly5<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];

        let sum_0_4 = x0 + x4;
        let diff_0_4 = x0 - x4;
        let sum_1_3 = x1 + x3;
        let diff_1_3 = x1 - x3;

        let intermediate_0 = fmla(T::TWO, x2, sum_0_4);
        data[0] = fmla(T::TWO, sum_1_3, intermediate_0);

        data[1] = fmla(T::SQRT_2, diff_1_3, diff_0_4);

        data[2] = fmla(-T::TWO, x2, sum_0_4);

        data[3] = fmla(-T::SQRT_2, diff_1_3, diff_0_4);

        let intermediate_4 = fmla(T::TWO, x2, sum_0_4);
        data[4] = fmla(-T::TWO, sum_1_3, intermediate_4);
    }
}

define_in_place_butterfly!(Dct1Butterfly5, 5);

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct1Butterfly3<T> {
    phantom_data: PhantomData<T>,
}

impl<T: DctSample> Dct1Butterfly3<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];

        let sum_0_2 = x0 + x2;
        let diff_0_2 = x0 - x2;

        data[0] = fmla(T::TWO, x1, sum_0_2);
        data[1] = diff_0_2;
        data[2] = fmla(-T::TWO, x1, sum_0_2);
    }
}

define_in_place_butterfly!(Dct1Butterfly3, 3);

#[derive(Debug, Clone)]
pub(crate) struct Dct1Butterfly6<T> {
    c1: T,
    c2: T,
}

impl<T: DctSample> Default for Dct1Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        let sqrt5 = 5.0f64.sqrt();
        Self {
            c1: (0.5 * (1.0 + sqrt5)).as_(),
            c2: (0.5 * (sqrt5 - 1.0)).as_(),
        }
    }
}

impl<T: DctSample> Dct1Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];

        // 1. Compute Symmetries and Antisymmetries
        let s0_5 = x0 + x5;
        let d0_5 = x0 - x5;

        let s1_4 = x1 + x4;
        let d1_4 = x1 - x4;

        let s2_3 = x2 + x3;
        let d2_3 = x2 - x3;

        let c1 = self.c1;
        let c2 = self.c2;

        // X[0]
        data[0] = fmla(T::TWO, s1_4 + s2_3, s0_5);

        // X[1]
        let intermediate = fmla(c1, d1_4, d0_5);
        data[1] = fmla(c2, d2_3, intermediate);

        // X[2]
        let intermediate = fmla(c2, s1_4, s0_5);
        data[2] = fmla(-c1, s2_3, intermediate);

        // X[3]
        let intermediate = fmla(-c2, d1_4, d0_5);
        data[3] = fmla(-c1, d2_3, intermediate);

        // X[4]
        let intermediate = fmla(-c1, s1_4, s0_5);
        data[4] = fmla(c2, s2_3, intermediate);

        // X[5]
        data[5] = fmla(T::TWO, d2_3 - d1_4, d0_5);
    }
}

define_in_place_butterfly!(Dct1Butterfly6, 6);

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct1Butterfly7<T> {
    _phantom_data: PhantomData<T>,
}

impl<T: DctSample> Dct1Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];

        let s0_6 = x0 + x6;
        let d0_6 = x0 - x6;

        let s1_5 = x1 + x5;
        let d1_5 = x1 - x5;

        let s2_4 = x2 + x4;
        let d2_4 = x2 - x4;

        let c1 = T::SQRT_3;

        let sum_inner = s1_5 + s2_4 + x3;
        data[0] = fmla(T::TWO, sum_inner, s0_6);

        let intermediate = fmla(c1, d1_5, d0_6);
        data[1] = intermediate + d2_4;

        let intermediate = fmla(-T::TWO, x3, s0_6);
        data[2] = (intermediate + s1_5) - s2_4;

        data[3] = fmla(-T::TWO, d2_4, d0_6);

        let intermediate = fmla(T::TWO, x3, s0_6);
        data[4] = intermediate - s1_5 - s2_4;

        let intermediate = fmla(-c1, d1_5, d0_6);
        data[5] = intermediate + d2_4;

        let alt_sum = s2_4 - x3 - s1_5;
        data[6] = fmla(T::TWO, alt_sum, s0_6);
    }
}

define_in_place_butterfly!(Dct1Butterfly7, 7);

#[derive(Debug, Clone)]
pub(crate) struct Dct1Butterfly8<T> {
    c1: T,
    c2: T,
    c3: T,
    c4: T,
}

impl<T: DctSample> Default for Dct1Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            c1: T::TWO * (1.0 / 7.0f64).as_().cospi(),
            c2: T::TWO * (2.0 / 7.0f64).as_().cospi(),
            c3: T::TWO * (3.0 / 7.0f64).as_().cospi(),
            c4: T::TWO * (4.0 / 7.0f64).as_().cospi(),
        }
    }
}

impl<T: DctSample> Dct1Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];
        let x7 = data[7];

        let s0_7 = x0 + x7;
        let d0_7 = x0 - x7;
        let s1_6 = x1 + x6;
        let d1_6 = x1 - x6;
        let s2_5 = x2 + x5;
        let d2_5 = x2 - x5;
        let s3_4 = x3 + x4;
        let d3_4 = x3 - x4;

        let c1 = self.c1;
        let c2 = self.c2;
        let c3 = self.c3;
        let c4 = self.c4;

        let total_sum = s1_6 + s2_5 + s3_4;
        data[0] = fmla(T::TWO, total_sum, s0_7);

        let intermediate_1 = fmla(c1, d1_6, d0_7);
        let intermediate_1 = fmla(c2, d2_5, intermediate_1);
        data[1] = fmla(c3, d3_4, intermediate_1);

        let intermediate_2 = fmla(c2, s1_6, s0_7);
        let intermediate_2 = fmla(c4, s2_5, intermediate_2);
        data[2] = fmla(-c1, s3_4, intermediate_2);

        let intermediate_3 = fmla(c3, d1_6, d0_7);
        let intermediate_3 = fmla(-c1, d2_5, intermediate_3);
        data[3] = fmla(-c2, d3_4, intermediate_3);

        let intermediate_4 = fmla(c4, s1_6, s0_7);
        let intermediate_4 = fmla(-c1, s2_5, intermediate_4);
        data[4] = fmla(c2, s3_4, intermediate_4);

        let intermediate_5 = fmla(-c2, d1_6, d0_7);
        let intermediate_5 = fmla(c4, d2_5, intermediate_5);
        data[5] = fmla(c1, d3_4, intermediate_5);

        let intermediate_6 = fmla(-c1, s1_6, s0_7);
        let intermediate_6 = fmla(c2, s2_5, intermediate_6);
        data[6] = fmla(-c3, s3_4, intermediate_6);

        let total_diff = (d1_6 - d2_5) + d3_4;
        data[7] = fmla(-T::TWO, total_diff, d0_7);
    }
}

define_in_place_butterfly!(Dct1Butterfly8, 8);

#[derive(Debug, Clone)]
pub(crate) struct Dct1Butterfly9<T> {
    c1: T,
    c3: T,
}

impl<T: DctSample> Default for Dct1Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        let c1_f64 = T::TWO * (1.0 / 8.0f64).as_().cospi();
        let c3_f64 = T::TWO * (3.0 / 8.0f64).as_().cospi();
        Self {
            c1: c1_f64,
            c3: c3_f64,
        }
    }
}

impl<T: DctSample> Dct1Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];
        let x7 = data[7];
        let x8 = data[8];

        let s0_8 = x0 + x8;
        let d0_8 = x0 - x8;
        let s1_7 = x1 + x7;
        let d1_7 = x1 - x7;
        let s2_6 = x2 + x6;
        let d2_6 = x2 - x6;
        let s3_5 = x3 + x5;
        let d3_5 = x3 - x5;

        let c1 = self.c1;
        let c3 = self.c3;

        // X[0]
        let sum_inner = s1_7 + s2_6 + s3_5 + x4;
        data[0] = fmla(T::TWO, sum_inner, s0_8);

        // X[1]
        let intermediate_1 = fmla(c1, d1_7, d0_8);
        let intermediate_1 = fmla(T::SQRT_2, d2_6, intermediate_1);
        data[1] = fmla(c3, d3_5, intermediate_1);

        // X[2]
        let intermediate_2 = fmla(-T::TWO, x4, s0_8);
        let intermediate_2 = fmla(T::SQRT_2, s1_7, intermediate_2);
        data[2] = fmla(-T::SQRT_2, s3_5, intermediate_2);

        // X[3]
        let intermediate_3 = fmla(c3, d1_7, d0_8);
        let intermediate_3 = fmla(-T::SQRT_2, d2_6, intermediate_3);
        data[3] = fmla(-c1, d3_5, intermediate_3);

        // X[4]
        let intermediate_4 = fmla(T::TWO, x4, s0_8);
        data[4] = fmla(-T::TWO, s2_6, intermediate_4);

        // X[5]
        let intermediate_5 = fmla(-c3, d1_7, d0_8);
        let intermediate_5 = fmla(-T::SQRT_2, d2_6, intermediate_5);
        data[5] = fmla(c1, d3_5, intermediate_5);

        // X[6]
        let intermediate_6 = fmla(-T::TWO, x4, s0_8);
        let intermediate_6 = fmla(-T::SQRT_2, s1_7, intermediate_6);
        data[6] = fmla(T::SQRT_2, s3_5, intermediate_6);

        // X[7]
        let intermediate_7 = fmla(-c1, d1_7, d0_8);
        let intermediate_7 = fmla(T::SQRT_2, d2_6, intermediate_7);
        data[7] = fmla(-c3, d3_5, intermediate_7);

        // X[8]
        let alt_sum = s2_6 + x4 - s1_7 - s3_5;
        data[8] = fmla(T::TWO, alt_sum, s0_8);
    }
}

define_in_place_butterfly!(Dct1Butterfly9, 9);
#[derive(Debug, Clone)]
pub(crate) struct Dct1Butterfly17<T> {
    c1: T,
    c2: T,
    c3: T,
    c5: T,
    c6: T,
    c7: T,
}

impl<T: DctSample> Default for Dct1Butterfly17<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            c1: T::TWO * (1.0 / 16.0f64).as_().cospi(),
            c2: T::TWO * (2.0 / 16.0f64).as_().cospi(),
            c3: T::TWO * (3.0 / 16.0f64).as_().cospi(),
            c5: T::TWO * (5.0 / 16.0f64).as_().cospi(),
            c6: T::TWO * (6.0 / 16.0f64).as_().cospi(),
            c7: T::TWO * (7.0 / 16.0f64).as_().cospi(),
        }
    }
}

impl<T: DctSample> Dct1Butterfly17<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];
        let x7 = data[7];
        let x8 = data[8];
        let x9 = data[9];
        let x10 = data[10];
        let x11 = data[11];
        let x12 = data[12];
        let x13 = data[13];
        let x14 = data[14];
        let x15 = data[15];
        let x16 = data[16];

        let s0_16 = x0 + x16;
        let d0_16 = x0 - x16;
        let s1_15 = x1 + x15;
        let d1_15 = x1 - x15;
        let s2_14 = x2 + x14;
        let d2_14 = x2 - x14;
        let s3_13 = x3 + x13;
        let d3_13 = x3 - x13;
        let s4_12 = x4 + x12;
        let d4_12 = x4 - x12;
        let s5_11 = x5 + x11;
        let d5_11 = x5 - x11;
        let s6_10 = x6 + x10;
        let d6_10 = x6 - x10;
        let s7_9 = x7 + x9;
        let d7_9 = x7 - x9;

        let c1 = self.c1;
        let c2 = self.c2;
        let c3 = self.c3;
        let c4 = T::SQRT_2;
        let c5 = self.c5;
        let c6 = self.c6;
        let c7 = self.c7;

        // X[0]
        let sum_inner = s1_15 + s2_14 + s3_13 + s4_12 + s5_11 + s6_10 + s7_9 + x8;
        data[0] = fmla(T::TWO, sum_inner, s0_16);

        // X[1]
        let intermediate = fmla(c1, d1_15, d0_16);
        let intermediate = fmla(c2, d2_14, intermediate);
        let intermediate = fmla(c3, d3_13, intermediate);
        let intermediate = fmla(c4, d4_12, intermediate);
        let intermediate = fmla(c5, d5_11, intermediate);
        let intermediate = fmla(c6, d6_10, intermediate);
        data[1] = fmla(c7, d7_9, intermediate);

        // X[2]
        let intermediate = fmla(-T::TWO, x8, s0_16);
        let intermediate = fmla(c2, s1_15, intermediate);
        let intermediate = fmla(c4, s2_14, intermediate);
        let intermediate = fmla(c6, s3_13, intermediate);
        let intermediate = fmla(-c6, s5_11, intermediate);
        let intermediate = fmla(-c4, s6_10, intermediate);
        data[2] = fmla(-c2, s7_9, intermediate);

        // X[3]
        let intermediate = fmla(c3, d1_15, d0_16);
        let intermediate = fmla(c6, d2_14, intermediate);
        let intermediate = fmla(-c7, d3_13, intermediate);
        let intermediate = fmla(-c4, d4_12, intermediate);
        let intermediate = fmla(-c1, d5_11, intermediate);
        let intermediate = fmla(-c2, d6_10, intermediate);
        data[3] = fmla(-c5, d7_9, intermediate);

        // X[4]
        let intermediate = fmla(T::TWO, x8, s0_16);
        let intermediate = fmla(c4, s1_15, intermediate);
        let intermediate = fmla(-c4, s3_13, intermediate);
        let intermediate = fmla(-T::TWO, s4_12, intermediate);
        let intermediate = fmla(-c4, s5_11, intermediate);
        data[4] = fmla(c4, s7_9, intermediate);

        // X[5]
        let intermediate = fmla(c5, d1_15, d0_16);
        let intermediate = fmla(-c6, d2_14, intermediate);
        let intermediate = fmla(-c1, d3_13, intermediate);
        let intermediate = fmla(-c4, d4_12, intermediate);
        let intermediate = fmla(c7, d5_11, intermediate);
        let intermediate = fmla(c2, d6_10, intermediate);
        data[5] = fmla(c3, d7_9, intermediate);

        // X[6]
        let intermediate = fmla(-T::TWO, x8, s0_16);
        let intermediate = fmla(c6, s1_15, intermediate);
        let intermediate = fmla(-c4, s2_14, intermediate);
        let intermediate = fmla(-c2, s3_13, intermediate);
        let intermediate = fmla(c2, s5_11, intermediate);
        let intermediate = fmla(c4, s6_10, intermediate);
        data[6] = fmla(-c6, s7_9, intermediate);

        // X[7]
        let intermediate = fmla(c7, d1_15, d0_16);
        let intermediate = fmla(-c2, d2_14, intermediate);
        let intermediate = fmla(-c5, d3_13, intermediate);
        let intermediate = fmla(c4, d4_12, intermediate);
        let intermediate = fmla(c3, d5_11, intermediate);
        let intermediate = fmla(-c6, d6_10, intermediate);
        data[7] = fmla(-c1, d7_9, intermediate);

        // X[8]
        let intermediate = fmla(T::TWO, x8, s0_16);
        let intermediate = fmla(-T::TWO, s2_14, intermediate);
        let intermediate = fmla(T::TWO, s4_12, intermediate);
        data[8] = fmla(-T::TWO, s6_10, intermediate);

        // X[9]
        let intermediate = fmla(-c7, d1_15, d0_16);
        let intermediate = fmla(-c2, d2_14, intermediate);
        let intermediate = fmla(c5, d3_13, intermediate);
        let intermediate = fmla(c4, d4_12, intermediate);
        let intermediate = fmla(-c3, d5_11, intermediate);
        let intermediate = fmla(-c6, d6_10, intermediate);
        data[9] = fmla(c1, d7_9, intermediate);

        // X[10]
        let intermediate = fmla(-T::TWO, x8, s0_16);
        let intermediate = fmla(-c6, s1_15, intermediate);
        let intermediate = fmla(-c4, s2_14, intermediate);
        let intermediate = fmla(c2, s3_13, intermediate);
        let intermediate = fmla(-c2, s5_11, intermediate);
        let intermediate = fmla(c4, s6_10, intermediate);
        data[10] = fmla(c6, s7_9, intermediate);

        // X[11]
        let intermediate = fmla(-c5, d1_15, d0_16);
        let intermediate = fmla(-c6, d2_14, intermediate);
        let intermediate = fmla(c1, d3_13, intermediate);
        let intermediate = fmla(-c4, d4_12, intermediate);
        let intermediate = fmla(-c7, d5_11, intermediate);
        let intermediate = fmla(c2, d6_10, intermediate);
        data[11] = fmla(-c3, d7_9, intermediate);

        // X[12]
        let intermediate = fmla(T::TWO, x8, s0_16);
        let intermediate = fmla(-c4, s1_15, intermediate);
        let intermediate = fmla(c4, s3_13, intermediate);
        let intermediate = fmla(-T::TWO, s4_12, intermediate);
        let intermediate = fmla(c4, s5_11, intermediate);
        data[12] = fmla(-c4, s7_9, intermediate);

        // X[13]
        let intermediate = fmla(-c3, d1_15, d0_16);
        let intermediate = fmla(c6, d2_14, intermediate);
        let intermediate = fmla(c7, d3_13, intermediate);
        let intermediate = fmla(-c4, d4_12, intermediate);
        let intermediate = fmla(c1, d5_11, intermediate);
        let intermediate = fmla(-c2, d6_10, intermediate);
        data[13] = fmla(c5, d7_9, intermediate);

        // X[14]
        let intermediate = fmla(-T::TWO, x8, s0_16);
        let intermediate = fmla(-c2, s1_15, intermediate);
        let intermediate = fmla(c4, s2_14, intermediate);
        let intermediate = fmla(-c6, s3_13, intermediate);
        let intermediate = fmla(c6, s5_11, intermediate);
        let intermediate = fmla(-c4, s6_10, intermediate);
        data[14] = fmla(c2, s7_9, intermediate);

        // X[15]
        let intermediate = fmla(-c1, d1_15, d0_16);
        let intermediate = fmla(c2, d2_14, intermediate);
        let intermediate = fmla(-c3, d3_13, intermediate);
        let intermediate = fmla(c4, d4_12, intermediate);
        let intermediate = fmla(-c5, d5_11, intermediate);
        let intermediate = fmla(c6, d6_10, intermediate);
        data[15] = fmla(-c7, d7_9, intermediate);

        // X[16]
        let alt_sum = s2_14 + s4_12 + s6_10 + x8 - s1_15 - s3_13 - s5_11 - s7_9;
        data[16] = fmla(T::TWO, alt_sum, s0_16);
    }
}

define_in_place_butterfly!(Dct1Butterfly17, 17);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::gen_test_butterfly;

    gen_test_butterfly!(test_bf2_dct1, f64, Dct1Butterfly2, 2, 1e-7, naive_dct1);
    gen_test_butterfly!(test_bf4_dct1, f64, Dct1Butterfly4, 4, 1e-7, naive_dct1);
    gen_test_butterfly!(test_bf3_dct1, f64, Dct1Butterfly3, 3, 1e-7, naive_dct1);
    gen_test_butterfly!(test_bf5_dct1, f64, Dct1Butterfly5, 5, 1e-7, naive_dct1);
    gen_test_butterfly!(test_bf6_dct1, f64, Dct1Butterfly6, 6, 1e-7, naive_dct1);
    gen_test_butterfly!(test_bf7_dct1, f64, Dct1Butterfly7, 7, 1e-7, naive_dct1);
    gen_test_butterfly!(test_bf8_dct1, f64, Dct1Butterfly8, 8, 1e-7, naive_dct1);
    gen_test_butterfly!(test_bf9_dct1, f64, Dct1Butterfly9, 9, 1e-7, naive_dct1);
    gen_test_butterfly!(test_bf17_dct1, f64, Dct1Butterfly17, 17, 1e-7, naive_dct1);
}
