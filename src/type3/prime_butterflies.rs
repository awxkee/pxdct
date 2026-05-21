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
use crate::bidirectional::BidirectionalStore;
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly3<T> {
    twiddle: T,
}

impl<T: DctSample> Default for Dct3Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 12).re,
        }
    }
}

impl<T: DctSample> Dct3Butterfly3<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let buffer0_half = data[0] * T::HALF;
        let buffer1 = data[1];
        let buffer2 = data[2];
        let buffer2_half = buffer2 * T::HALF;

        let half02 = buffer0_half + buffer2_half;

        data[0] = fmla(buffer1, self.twiddle, half02);
        data[1] = buffer0_half - buffer2;
        data[2] = fmla(buffer1, -self.twiddle, half02);
    }
}

define_in_place_butterfly!(Dct3Butterfly3, 3);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly5<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
}

impl<T: DctSample> Default for Dct3Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 20),
            twiddle1: compute_twiddle(2, 20),
        }
    }
}

impl<T: DctSample> Dct3Butterfly5<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0] * T::HALF;
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];

        let t0 = fmla(-self.twiddle0.im, x4, x0);
        let t1 = fmla(-self.twiddle1.re, x4, x0);
        let t2 = x3 * self.twiddle1.im;
        let t3 = x3 * self.twiddle0.re;
        let t4 = x2 * self.twiddle0.im;
        let t5 = x2 * self.twiddle1.re;
        let t6 = x1 * self.twiddle1.im;
        let t7 = x1 * self.twiddle0.re;

        let q0 = t4 + t1;
        let q1 = t5 + t0;
        let q2 = t3 + t6;

        let y0 = t7 + q1 - t2;
        let y1 = q0 - q2;

        let y2 = x0 - x2 + x4;

        let y3 = q0 + q2;
        let y4 = q1 + t2 - t7;

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
    }
}

define_in_place_butterfly!(Dct3Butterfly5, 5);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly7<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}

impl<T: DctSample> Default for Dct3Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 28).conj(),
            twiddle1: compute_twiddle(2, 28).conj(),
            twiddle2: compute_twiddle(3, 28).conj(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly7<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0] * T::HALF;
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];

        let t0 = fmla(self.twiddle0.im, x6, x0);
        let t1 = fmla(-self.twiddle2.im, x6, x0);
        let t2 = fmla(self.twiddle1.re, x6, x0);
        let t3 = x5 * self.twiddle1.im;
        let t4 = x5 * self.twiddle0.re;
        let t5 = x4 * self.twiddle2.im;
        let t6 = x4 * self.twiddle1.re;
        let t7 = x5 * self.twiddle2.re;
        let t8 = x4 * self.twiddle0.im;

        let q0 = t0 + t5;
        let q1 = t2 - t8;

        let t9 = x3 * self.twiddle2.re;
        let t10 = x3 * self.twiddle1.im;

        let q2 = t1 - t6;
        let q3 = t4 + t10;

        let t11 = x3 * self.twiddle0.re;
        let t12 = x2 * self.twiddle1.re;
        let t13 = x2 * self.twiddle0.im;
        let t14 = x2 * self.twiddle2.im;

        let q4 = t3 + t9;
        let q5 = t7 - t11;
        let q6 = q0 + t12;
        let q7 = q2 + t13;
        let q8 = q1 - t14;

        let y0 = fmla(self.twiddle0.re, x1, q6 + q4);
        let y1 = fmla(self.twiddle2.re, x1, q7 - q3);
        let y2 = fmla(self.twiddle1.im, x1, q5 + q8);
        let y3 = x0 - x2 + x4 - x6;
        let y4 = fmla(-self.twiddle1.im, x1, q8 - q5);
        let y5 = fmla(-self.twiddle2.re, x1, q7 + q3);
        let y6 = fmla(-self.twiddle0.re, x1, q6 - q4);

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
        data[5] = y5;
        data[6] = y6;
    }
}

define_in_place_butterfly!(Dct3Butterfly7, 7);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly11<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
}

impl<T: DctSample> Default for Dct3Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 44).conj(),
            twiddle1: compute_twiddle(2, 44).conj(),
            twiddle2: compute_twiddle(3, 44).conj(),
            twiddle3: compute_twiddle(4, 44).conj(),
            twiddle4: compute_twiddle(5, 44).conj(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly11<T>
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
        let x9 = data[9];
        let x10 = data[10];

        let t0 = fmla(self.twiddle0.im, x10, x0);
        let t1 = fmla(-self.twiddle2.im, x10, x0);
        let t2 = fmla(self.twiddle4.im, x10, x0);
        let t3 = fmla(-self.twiddle3.re, x10, x0);
        let t4 = fmla(self.twiddle1.re, x10, x0);

        let t5 = self.twiddle1.im * x9;
        let t6 = self.twiddle4.re * x9;
        let t7 = self.twiddle0.re * x9;
        let t8 = self.twiddle2.re * x9;
        let t9 = self.twiddle3.im * x9;

        let t10 = self.twiddle2.im * x8;
        let t11 = self.twiddle1.re * x8;
        let t12 = self.twiddle3.re * x8;
        let t13 = self.twiddle0.im * x8;
        let t14 = self.twiddle4.im * x8;

        let q0 = t0 + t10;
        let q1 = t1 - t11;
        let q2 = t2 + t12;
        let q3 = t3 - t13;
        let q4 = t4 - t14;

        let t15 = self.twiddle3.im * x7;
        let t16 = self.twiddle0.re * x7;
        let t17 = self.twiddle1.im * x7;
        let t18 = self.twiddle4.re * x7;
        let t19 = self.twiddle2.re * x7;

        let q6 = t5 + t15;
        let q7 = t16 + t6;
        let q8 = t17 + t7;
        let q9 = t18 - t8;
        let q10 = t9 - t19;

        let t20 = self.twiddle4.im * x6;
        let t21 = self.twiddle3.re * x6;
        let t22 = self.twiddle2.im * x6;
        let t23 = self.twiddle1.re * x6;
        let t24 = self.twiddle0.im * x6;

        let q11 = t20 + q0;
        let q12 = q1 - t21;
        let q13 = q2 - t22;
        let q14 = t23 + q3;
        let q15 = t24 + q4;

        let t25 = self.twiddle4.re * x5;
        let t26 = self.twiddle3.im * x5;
        let t27 = self.twiddle2.re * x5;
        let t28 = self.twiddle1.im * x5;
        let t29 = self.twiddle0.re * x5;

        let q16 = q6 + t25;
        let q17 = t26 + q7;
        let q18 = q8 - t27;
        let q19 = t28 + q9;
        let q20 = t29 + q10;

        let t30 = self.twiddle3.re * x4;
        let t31 = self.twiddle0.im * x4;
        let t32 = self.twiddle1.re * x4;
        let t33 = self.twiddle4.im * x4;
        let t34 = self.twiddle2.im * x4;

        let q21 = t30 + q11;
        let q22 = q12 - t31;
        let q23 = q13 - t32;
        let q24 = q14 - t33;
        let q25 = q15 + t34;

        let t35 = self.twiddle2.re * x3;
        let t36 = self.twiddle1.im * x3;
        let t37 = self.twiddle3.im * x3;
        let t38 = self.twiddle0.re * x3;
        let t39 = self.twiddle4.re * x3;

        let q26 = q16 + t35;
        let q27 = t36 - q17;
        let q28 = q18 - t37;
        let q29 = q19 - t38;
        let q30 = q20 - t39;

        let t40 = self.twiddle1.re * x2;
        let t41 = self.twiddle4.im * x2;
        let t42 = self.twiddle0.im * x2;
        let t43 = self.twiddle2.im * x2;
        let t44 = self.twiddle3.re * x2;

        let q31 = t40 + q21;
        let q32 = t41 + q22;
        let q33 = t42 + q23;
        let q34 = q24 - t43;
        let q35 = q25 - t44;

        let y0 = fmla(self.twiddle0.re, x1, q31 + q26);
        let y1 = fmla(self.twiddle2.re, x1, q32 + q27);
        let y2 = fmla(self.twiddle4.re, x1, q33 + q28);
        let y3 = fmla(self.twiddle3.im, x1, q34 + q29);
        let y4 = fmla(self.twiddle1.im, x1, q35 + q30);
        let y5 = x0 - x2 + x4 - x6 + x8 - x10;
        let y6 = fmla(-self.twiddle1.im, x1, q35 - q30);
        let y7 = fmla(-self.twiddle3.im, x1, q34 - q29);
        let y8 = fmla(-self.twiddle4.re, x1, q33 - q28);
        let y9 = fmla(-self.twiddle2.re, x1, q32 - q27);
        let y10 = fmla(-self.twiddle0.re, x1, q31 - q26);

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
        data[5] = y5;
        data[6] = y6;
        data[7] = y7;
        data[8] = y8;
        data[9] = y9;
        data[10] = y10;
    }
}

define_in_place_butterfly!(Dct3Butterfly11, 11);

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly13<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
}

impl<T: DctSample> Default for Dct3Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 52).conj(),
            twiddle1: compute_twiddle(2, 52).conj(),
            twiddle2: compute_twiddle(3, 52).conj(),
            twiddle3: compute_twiddle(4, 52).conj(),
            twiddle4: compute_twiddle(5, 52).conj(),
            twiddle5: compute_twiddle(6, 52).conj(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly13<T>
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
        let x9 = data[9];
        let x10 = data[10];
        let x11 = data[11];
        let x12 = data[12];

        let t0 = fmla(self.twiddle0.im, x12, x0);
        let t1 = fmla(-self.twiddle2.im, x12, x0);
        let t2 = fmla(self.twiddle4.im, x12, x0);
        let t3 = fmla(-self.twiddle5.re, x12, x0);
        let t4 = fmla(self.twiddle3.re, x12, x0);
        let t5 = fmla(-self.twiddle1.re, x12, x0);

        let t6 = self.twiddle1.im * x11;
        let t7 = self.twiddle5.im * x11;
        let t8 = self.twiddle2.re * x11;
        let t9 = self.twiddle0.re * x11;
        let t10 = self.twiddle4.re * x11;
        let t11 = self.twiddle3.im * x11;

        let t13 = self.twiddle2.im * x10;
        let t14 = self.twiddle3.re * x10;
        let t15 = self.twiddle1.re * x10;
        let t16 = self.twiddle4.im * x10;
        let t17 = self.twiddle0.im * x10;
        let t18 = self.twiddle5.re * x10;

        let q0 = t13 + t0;
        let q1 = t1 - t14;
        let q2 = t2 + t15;
        let q3 = t3 - t16;
        let q4 = t4 - t17;
        let q5 = t5 + t18;

        let t19 = self.twiddle3.im * x9;
        let t20 = self.twiddle0.re * x9;
        let t21 = self.twiddle5.im * x9;
        let t22 = self.twiddle1.im * x9;
        let t23 = self.twiddle2.re * x9;
        let t24 = self.twiddle4.re * x9;

        let q6 = t19 + t6;
        let q7 = t7 + t20;
        let q8 = t21 + t8;
        let q9 = t22 - t9;
        let q10 = t10 - t23;
        let q11 = t24 - t11;

        let t25 = self.twiddle4.im * x8;
        let t26 = self.twiddle1.re * x8;
        let t27 = self.twiddle0.im * x8;
        let t28 = self.twiddle3.re * x8;
        let t29 = self.twiddle5.re * x8;
        let t30 = self.twiddle2.im * x8;

        let q12 = t25 + q0;
        let q13 = q1 - t26;
        let q14 = q2 + t27;
        let q15 = q3 + t28;
        let q16 = q4 - t29;
        let q17 = q5 - t30;

        let t31 = self.twiddle5.im * x7;
        let t32 = self.twiddle4.re * x7;
        let t33 = self.twiddle3.im * x7;
        let t34 = self.twiddle2.re * x7;
        let t35 = self.twiddle1.im * x7;
        let t36 = self.twiddle0.re * x7;

        let q18 = t31 + q6;
        let q19 = q7 + t32;
        let q20 = q8 - t33;
        let q21 = t34 + q9;
        let q22 = t35 + q10;
        let q23 = q11 - t36;

        let t37 = self.twiddle5.re * x6;
        let t38 = self.twiddle4.im * x6;
        let t39 = self.twiddle3.re * x6;
        let t40 = self.twiddle2.im * x6;
        let t41 = self.twiddle1.re * x6;
        let t42 = self.twiddle0.im * x6;

        let q24 = q12 + t37;
        let q25 = q13 - t38;
        let q26 = q14 - t39;
        let q27 = t40 + q15;
        let q28 = t41 + q16;
        let q29 = q17 - t42;

        let t43 = self.twiddle4.re * x5;
        let t44 = self.twiddle1.im * x5;
        let t45 = self.twiddle0.re * x5;
        let t46 = self.twiddle3.im * x5;
        let t47 = self.twiddle5.im * x5;
        let t48 = self.twiddle2.re * x5;

        let q30 = t43 + q18;
        let q31 = t44 + q19;
        let q32 = q20 - t45;
        let q33 = q21 - t46;
        let q34 = t47 + q22;
        let q35 = t48 + q23;

        let t49 = self.twiddle3.re * x4;
        let t50 = self.twiddle0.im * x4;
        let t51 = self.twiddle5.re * x4;
        let t52 = self.twiddle1.re * x4;
        let t53 = self.twiddle2.im * x4;
        let t54 = self.twiddle4.im * x4;

        let q36 = t49 + q24;
        let q37 = t50 + q25;
        let q38 = q26 - t51;
        let q39 = q27 - t52;
        let q40 = q28 - t53;
        let q41 = t54 + q29;

        let t55 = self.twiddle2.re * x3;
        let t56 = self.twiddle3.im * x3;
        let t57 = self.twiddle1.im * x3;
        let t58 = self.twiddle4.re * x3;
        let t59 = self.twiddle0.re * x3;
        let t60 = self.twiddle5.im * x3;

        let q42 = t55 + q30;
        let q43 = t56 - q31;
        let q44 = q32 - t57;
        let q45 = q33 - t58;
        let q46 = q34 - t59;
        let q47 = q35 - t60;

        let t61 = self.twiddle1.re * x2;
        let t62 = self.twiddle5.re * x2;
        let t63 = self.twiddle2.im * x2;
        let t64 = self.twiddle0.im * x2;
        let t65 = self.twiddle4.im * x2;
        let t66 = self.twiddle3.re * x2;

        let q48 = t61 + q36;
        let q49 = t62 + q37;
        let q50 = t63 + q38;
        let q51 = q39 - t64;
        let q52 = q40 - t65;
        let q53 = q41 - t66;

        let y0 = fmla(self.twiddle0.re, x1, q48 + q42);
        let y1 = fmla(self.twiddle2.re, x1, q49 + q43);
        let y2 = fmla(self.twiddle4.re, x1, q50 + q44);
        let y3 = fmla(self.twiddle5.im, x1, q51 + q45);
        let y4 = fmla(self.twiddle3.im, x1, q52 + q46);
        let y5 = fmla(self.twiddle1.im, x1, q53 + q47);
        let y6 = x0 - x2 + x4 - x6 + x8 - x10 + x12;
        let y7 = fmla(-self.twiddle1.im, x1, q53 - q47);
        let y8 = fmla(-self.twiddle3.im, x1, q52 - q46);
        let y9 = fmla(-self.twiddle5.im, x1, q51 - q45);
        let y10 = fmla(-self.twiddle4.re, x1, q50 - q44);
        let y11 = fmla(-self.twiddle2.re, x1, q49 - q43);
        let y12 = fmla(-self.twiddle0.re, x1, q48 - q42);

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
        data[5] = y5;
        data[6] = y6;
        data[7] = y7;
        data[8] = y8;
        data[9] = y9;
        data[10] = y10;
        data[11] = y11;
        data[12] = y12;
    }
}

define_in_place_butterfly!(Dct3Butterfly13, 13);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::gen_test_butterfly;

    gen_test_butterfly!(test_bf_dct3_3, f64, Dct3Butterfly3, 3, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_5, f64, Dct3Butterfly5, 5, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_7, f64, Dct3Butterfly7, 7, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_11, f64, Dct3Butterfly11, 11, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_13, f64, Dct3Butterfly13, 13, 1e-7, naive_dct3);
}
