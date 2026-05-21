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
use crate::twiddles::FftTrigonometry;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct Dct7Butterfly2<T: DctSample> {
    c: [T; 1],
}

impl<T: DctSample> Default for Dct7Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            c: [
                (1.0 / 3.0).cospi().as_(), // C1 = cos(1*pi/3)
            ],
        }
    }
}

impl<T: DctSample> Dct7Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];

        // Analytically reduced from a 6-point RDFT
        let acc0 = x0 * T::HALF;
        let y0 = fmla(x1, self.c[0], acc0);
        data[0] = y0;
        let y1 = acc0 - x1;
        data[1] = y1;
    }
}

define_in_place_butterfly!(Dct7Butterfly2, 2);

#[derive(Debug, Clone)]
pub(crate) struct Dct7Butterfly3<T: DctSample> {
    c: [T; 2],
}

impl<T: DctSample> Default for Dct7Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            c: [
                (1.0 / 5.0).cospi().as_(), // C1 = cos(1*pi/5)
                (2.0 / 5.0).cospi().as_(), // C2 = cos(2*pi/5)
            ],
        }
    }
}

impl<T: DctSample> Dct7Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];

        // Analytically reduced from a 10-point RDFT

        let acc0 = x0 * T::HALF;
        let acc0 = fmla(x1, self.c[0], acc0);
        let y0 = fmla(x2, self.c[1], acc0);
        data[0] = y0;

        let acc1 = x0 * T::HALF;
        let acc1 = fmla(x2, -self.c[0], acc1);
        let y1 = fmla(x1, -self.c[1], acc1);
        data[1] = y1;

        let y2 = x0 * T::HALF + x2 - x1;
        data[2] = y2;
    }
}

define_in_place_butterfly!(Dct7Butterfly3, 3);

#[derive(Debug, Clone)]
pub(crate) struct Dct7Butterfly4<T: DctSample> {
    c: [T; 3],
}

impl<T: DctSample> Default for Dct7Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            c: [
                (1.0 / 7.0).cospi().as_(), // C1 = cos(1*pi/7)
                (2.0 / 7.0).cospi().as_(), // C2 = cos(2*pi/7)
                (3.0 / 7.0).cospi().as_(), // C3 = cos(3*pi/7)
            ],
        }
    }
}

impl<T: DctSample> Dct7Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];

        // Analytically reduced from a 14-point RDFT
        // H = 1/2, Ck = cos(k*pi/7), c[k-1] = Ck
        let h0 = x0 * T::HALF;

        let acc0 = h0;
        let acc0 = fmla(x1, self.c[0], acc0);
        let acc0 = fmla(x2, self.c[1], acc0);
        let y0 = fmla(x3, self.c[2], acc0);
        data[0] = y0;

        let acc1 = h0;
        let acc1 = fmla(x2, -self.c[0], acc1);
        let acc1 = fmla(x3, -self.c[1], acc1);
        let y1 = fmla(x1, self.c[2], acc1);
        data[1] = y1;

        let acc2 = h0;
        let acc2 = fmla(x3, self.c[0], acc2);
        let acc2 = fmla(x1, -self.c[1], acc2);
        let y2 = fmla(x2, -self.c[2], acc2);
        data[2] = y2;

        let y3 = h0 + x2 - x1 - x3;
        data[3] = y3;
    }
}

define_in_place_butterfly!(Dct7Butterfly4, 4);

#[derive(Debug, Clone)]
pub(crate) struct Dct7Butterfly8<T: DctSample> {
    c: [T; 7],
}

impl<T: DctSample> Default for Dct7Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            c: [
                (1.0 / 15.0).cospi().as_(), // C1 = cos(1*pi/15)
                (2.0 / 15.0).cospi().as_(), // C2 = cos(2*pi/15)
                (3.0 / 15.0).cospi().as_(), // C3 = cos(3*pi/15)
                (4.0 / 15.0).cospi().as_(), // C4 = cos(4*pi/15)
                (5.0 / 15.0).cospi().as_(), // C5 = cos(5*pi/15)
                (6.0 / 15.0).cospi().as_(), // C6 = cos(6*pi/15)
                (7.0 / 15.0).cospi().as_(), // C7 = cos(7*pi/15)
            ],
        }
    }
}

impl<T: DctSample> Dct7Butterfly8<T>
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

        // Analytically reduced from a 30-point RDFT
        // H = 1/2, Ck = cos(k*pi/15), c[k-1] = Ck
        // dp3_0: row 0 Gp pattern
        let dp3_0 = x3 * self.c[2];
        let dp3_0 = fmla(x6, self.c[5], dp3_0);
        // dp3_1: row 1 Gp pattern
        let dp3_1 = -(x3 * self.c[5]);
        let dp3_1 = fmla(x6, -self.c[2], dp3_1);
        // dp3_2: row 2 Gp pattern
        let dp3_2 = -x3;
        let dp3_2 = dp3_2 + x6;

        // G5 ({x5})  ->  2 canonical dot-product(s)
        // dp5_0: row 0 Gp pattern
        let dp5_0 = x5 * self.c[4];
        // dp5_1: row 1 Gp pattern
        let dp5_1 = -x5;

        // Hoisted pre-sums reused across multiple rows.
        let r0 = x1 - x4;
        let r1 = x2 - x7;
        let r2 = x1 + x7 - x2 - x4;

        let h0 = x0 * T::HALF;

        let acc0 = h0;
        let acc0 = fmla(x1, self.c[0], acc0);
        let acc0 = fmla(x2, self.c[1], acc0);
        let acc0 = fmla(x4, self.c[3], acc0);
        let acc0 = fmla(x7, self.c[6], acc0);
        let y0 = acc0 + dp3_0 + dp5_0;
        data[0] = y0;

        let acc1 = h0;
        let acc1 = fmla(r0, self.c[2], acc1);
        let acc1 = fmla(r1, self.c[5], acc1);
        let y1 = acc1 + dp3_1 + dp5_1;
        data[1] = y1;

        let acc2 = h0;
        let acc2 = fmla(r2, self.c[4], acc2);
        let y2 = acc2 + dp3_2 + dp5_0;
        data[2] = y2;

        let acc3 = h0;
        let acc3 = fmla(x2, -self.c[0], acc3);
        let acc3 = fmla(x4, self.c[1], acc3);
        let acc3 = fmla(x7, -self.c[3], acc3);
        let acc3 = fmla(x1, self.c[6], acc3);
        let y3 = acc3 + dp3_1 + dp5_0;
        data[3] = y3;

        let acc4 = h0;
        let acc4 = fmla(r1, -self.c[2], acc4);
        let acc4 = fmla(r0, -self.c[5], acc4);
        let y4 = acc4 + dp3_0 + dp5_1;
        data[4] = y4;

        let acc5 = h0;
        let acc5 = fmla(x4, -self.c[0], acc5);
        let acc5 = fmla(x7, -self.c[1], acc5);
        let acc5 = fmla(x1, -self.c[3], acc5);
        let acc5 = fmla(x2, -self.c[6], acc5);
        let y5 = acc5 + dp3_0 + dp5_0;
        data[5] = y5;

        let acc6 = h0;
        let acc6 = fmla(x7, self.c[0], acc6);
        let acc6 = fmla(x1, -self.c[1], acc6);
        let acc6 = fmla(x2, self.c[3], acc6);
        let acc6 = fmla(x4, -self.c[6], acc6);
        let y6 = acc6 + dp3_1 + dp5_0;
        data[6] = y6;

        let acc7 = h0;
        let acc7 = acc7 + x2 + x4 - x1 - x7;
        let y7 = acc7 + dp3_2 + dp5_1;
        data[7] = y7;
    }
}

define_in_place_butterfly!(Dct7Butterfly8, 8);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::gen_test_butterfly;

    gen_test_butterfly!(test_bf_dst7_2, f64, Dct7Butterfly2, 2, 1e-7, naive_dct7);
    gen_test_butterfly!(test_bf_dst7_3, f64, Dct7Butterfly3, 3, 1e-7, naive_dct7);
    gen_test_butterfly!(test_bf_dst7_4, f64, Dct7Butterfly4, 4, 1e-7, naive_dct7);
    gen_test_butterfly!(test_bf_dst7_8, f64, Dct7Butterfly8, 8, 1e-7, naive_dct7);
}
