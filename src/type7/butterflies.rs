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
pub(crate) struct Dst7Butterfly2<T: DctSample> {
    s: [T; 2],
}

impl<T: DctSample> Default for Dst7Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            s: [(1.0 / 5.0).sinpi().as_(), (2.0 / 5.0).sinpi().as_()],
        }
    }
}

impl<T: DctSample> Dst7Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];

        // Analytically reduced from a 5-point RDFT
        let y0 = fmla(x1, self.s[1], x0 * self.s[0]);
        let y1 = fmla(x1, -self.s[0], x0 * self.s[1]);

        data[0] = y0;
        data[1] = y1;
    }
}

define_in_place_butterfly!(Dst7Butterfly2, 2);

#[derive(Debug, Clone)]
pub(crate) struct Dst7Butterfly3<T: DctSample> {
    s: [T; 3],
}

impl<T: DctSample> Default for Dst7Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            s: [
                (1.0 / 7.0).sinpi().as_(), // S1 = sin(1*pi/7)
                (2.0 / 7.0).sinpi().as_(), // S2 = sin(2*pi/7)
                (3.0 / 7.0).sinpi().as_(), // S3 = sin(3*pi/7)
            ],
        }
    }
}

impl<T: DctSample> Dst7Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];

        // Analytically reduced from a 7-point RDFT

        let acc0 = x0 * self.s[0];
        let acc0 = fmla(x1, self.s[1], acc0);
        let y0 = fmla(x2, self.s[2], acc0);
        data[0] = y0;

        let acc1 = x1 * self.s[0];
        let acc1 = fmla(x2, -self.s[1], acc1);
        let y1 = fmla(x0, self.s[2], acc1);
        data[1] = y1;

        let acc2 = x2 * self.s[0];
        let acc2 = fmla(x0, self.s[1], acc2);
        let y2 = fmla(x1, -self.s[2], acc2);
        data[2] = y2;
    }
}

define_in_place_butterfly!(Dst7Butterfly3, 3);

#[derive(Debug, Clone)]
pub(crate) struct Dst7Butterfly4<T: DctSample> {
    s: [T; 4],
}

impl<T: DctSample> Default for Dst7Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            s: [
                (1.0 / 9.0).sinpi().as_(), // S1 = sin(1*pi/9)
                (2.0 / 9.0).sinpi().as_(), // S2 = sin(2*pi/9)
                (3.0 / 9.0).sinpi().as_(), // S3 = sin(3*pi/9)
                (4.0 / 9.0).sinpi().as_(), // S4 = sin(4*pi/9)
            ],
        }
    }
}

impl<T: DctSample> Dst7Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];

        // Analytically reduced from a 9-point RDFT
        let dp3_0 = x2 * self.s[2];

        let acc0 = x0 * self.s[0];
        let acc0 = fmla(x1, self.s[1], acc0);
        let acc0 = fmla(x3, self.s[3], acc0);
        let y0 = acc0 + dp3_0;
        data[0] = y0;

        let y1 = (x0 + x1 - x3) * self.s[2];
        data[1] = y1;

        let acc2 = -(x1 * self.s[0]);
        let acc2 = fmla(x3, self.s[1], acc2);
        let acc2 = fmla(x0, self.s[3], acc2);
        let y2 = acc2 - dp3_0;
        data[2] = y2;

        let acc3 = -(x3 * self.s[0]);
        let acc3 = fmla(x0, self.s[1], acc3);
        let acc3 = fmla(x1, -self.s[3], acc3);
        let y3 = acc3 + dp3_0;
        data[3] = y3;
    }
}

define_in_place_butterfly!(Dst7Butterfly4, 4);

#[derive(Debug, Clone)]
pub(crate) struct Dst7Butterfly5<T: DctSample> {
    s: [T; 5],
}

impl<T: DctSample> Default for Dst7Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            s: [
                (1.0 / 11.0).sinpi().as_(), // S1 = sin(1*pi/11)
                (2.0 / 11.0).sinpi().as_(), // S2 = sin(2*pi/11)
                (3.0 / 11.0).sinpi().as_(), // S3 = sin(3*pi/11)
                (4.0 / 11.0).sinpi().as_(), // S4 = sin(4*pi/11)
                (5.0 / 11.0).sinpi().as_(), // S5 = sin(5*pi/11)
            ],
        }
    }
}

impl<T: DctSample> Dst7Butterfly5<T>
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

        // Analytically reduced from a 11-point RDFT
        // Sk = sin(k*pi/11), s[k-1] = Sk

        let acc0 = x0 * self.s[0];
        let acc0 = fmla(x1, self.s[1], acc0);
        let acc0 = fmla(x2, self.s[2], acc0);
        let acc0 = fmla(x3, self.s[3], acc0);
        let y0 = fmla(x4, self.s[4], acc0);
        data[0] = y0;

        let acc1 = -(x3 * self.s[0]);
        let acc1 = fmla(x2, self.s[1], acc1);
        let acc1 = fmla(x0, self.s[2], acc1);
        let acc1 = fmla(x4, -self.s[3], acc1);
        let y1 = fmla(x1, self.s[4], acc1);
        data[1] = y1;

        let acc2 = x1 * self.s[0];
        let acc2 = fmla(x3, -self.s[1], acc2);
        let acc2 = fmla(x4, self.s[2], acc2);
        let acc2 = fmla(x2, -self.s[3], acc2);
        let y2 = fmla(x0, self.s[4], acc2);
        data[2] = y2;

        let acc3 = -(x2 * self.s[0]);
        let acc3 = fmla(x4, -self.s[1], acc3);
        let acc3 = fmla(x1, -self.s[2], acc3);
        let acc3 = fmla(x0, self.s[3], acc3);
        let y3 = fmla(x3, self.s[4], acc3);
        data[3] = y3;

        let acc4 = x4 * self.s[0];
        let acc4 = fmla(x0, self.s[1], acc4);
        let acc4 = fmla(x3, -self.s[2], acc4);
        let acc4 = fmla(x1, -self.s[3], acc4);
        let y4 = fmla(x2, self.s[4], acc4);
        data[4] = y4;
    }
}

define_in_place_butterfly!(Dst7Butterfly5, 5);

#[derive(Debug, Clone)]
pub(crate) struct Dst7Butterfly6<T: DctSample> {
    s: [T; 6],
}

impl<T: DctSample> Default for Dst7Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            s: [
                (1.0 / 13.0).sinpi().as_(), // S1 = sin(1*pi/13)
                (2.0 / 13.0).sinpi().as_(), // S2 = sin(2*pi/13)
                (3.0 / 13.0).sinpi().as_(), // S3 = sin(3*pi/13)
                (4.0 / 13.0).sinpi().as_(), // S4 = sin(4*pi/13)
                (5.0 / 13.0).sinpi().as_(), // S5 = sin(5*pi/13)
                (6.0 / 13.0).sinpi().as_(), // S6 = sin(6*pi/13)
            ],
        }
    }
}

impl<T: DctSample> Dst7Butterfly6<T>
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

        // Analytically reduced from a 13-point RDFT
        let acc0 = x0 * self.s[0];
        let acc0 = fmla(x1, self.s[1], acc0);
        let acc0 = fmla(x2, self.s[2], acc0);
        let acc0 = fmla(x3, self.s[3], acc0);
        let acc0 = fmla(x4, self.s[4], acc0);
        let y0 = fmla(x5, self.s[5], acc0);
        data[0] = y0;

        let acc1 = x3 * self.s[0];
        let acc1 = fmla(x4, -self.s[1], acc1);
        let acc1 = fmla(x0, self.s[2], acc1);
        let acc1 = fmla(x2, self.s[3], acc1);
        let acc1 = fmla(x5, -self.s[4], acc1);
        let y1 = fmla(x1, self.s[5], acc1);
        data[1] = y1;

        let acc2 = -(x4 * self.s[0]);
        let acc2 = fmla(x2, -self.s[1], acc2);
        let acc2 = fmla(x1, self.s[2], acc2);
        let acc2 = fmla(x5, self.s[3], acc2);
        let acc2 = fmla(x0, self.s[4], acc2);
        let y2 = fmla(x3, -self.s[5], acc2);
        data[2] = y2;

        let acc3 = -(x1 * self.s[0]);
        let acc3 = fmla(x3, self.s[1], acc3);
        let acc3 = fmla(x5, -self.s[2], acc3);
        let acc3 = fmla(x4, self.s[3], acc3);
        let acc3 = fmla(x2, -self.s[4], acc3);
        let y3 = fmla(x0, self.s[5], acc3);
        data[3] = y3;

        let acc4 = x2 * self.s[0];
        let acc4 = fmla(x5, self.s[1], acc4);
        let acc4 = fmla(x3, self.s[2], acc4);
        let acc4 = fmla(x0, self.s[3], acc4);
        let acc4 = fmla(x1, -self.s[4], acc4);
        let y4 = fmla(x4, -self.s[5], acc4);
        data[4] = y4;

        let acc5 = -(x5 * self.s[0]);
        let acc5 = fmla(x0, self.s[1], acc5);
        let acc5 = fmla(x4, self.s[2], acc5);
        let acc5 = fmla(x1, -self.s[3], acc5);
        let acc5 = fmla(x3, -self.s[4], acc5);
        let y5 = fmla(x2, self.s[5], acc5);
        data[5] = y5;
    }
}

define_in_place_butterfly!(Dst7Butterfly6, 6);

#[derive(Debug, Clone)]
pub(crate) struct Dst7Butterfly7<T: DctSample> {
    s: [T; 7],
}

impl<T: DctSample> Default for Dst7Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            s: [
                (1.0 / 15.0).sinpi().as_(), // S1 = sin(1*pi/15)
                (2.0 / 15.0).sinpi().as_(), // S2 = sin(2*pi/15)
                (3.0 / 15.0).sinpi().as_(), // S3 = sin(3*pi/15)
                (4.0 / 15.0).sinpi().as_(), // S4 = sin(4*pi/15)
                (5.0 / 15.0).sinpi().as_(), // S5 = sin(5*pi/15)
                (6.0 / 15.0).sinpi().as_(), // S6 = sin(6*pi/15)
                (7.0 / 15.0).sinpi().as_(), // S7 = sin(7*pi/15)
            ],
        }
    }
}

impl<T: DctSample> Dst7Butterfly7<T>
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

        // Analytically reduced from a 15-point RDFT

        // dp3_0: row 0 Gp pattern
        let dp3_0 = x2 * self.s[2];
        let dp3_0 = fmla(x5, self.s[5], dp3_0);
        // dp3_1: row 1 Gp pattern
        let dp3_1 = x2 * self.s[5];
        let dp3_1 = fmla(x5, -self.s[2], dp3_1);

        let dp5_0 = x4 * self.s[4];

        let r0 = x0 + x3;
        let r1 = x1 - x6;

        let acc0 = x0 * self.s[0];
        let acc0 = fmla(x1, self.s[1], acc0);
        let acc0 = fmla(x3, self.s[3], acc0);
        let acc0 = fmla(x6, self.s[6], acc0);
        let y0 = acc0 + dp3_0 + dp5_0;
        data[0] = y0;

        let acc1 = r0 * self.s[2];
        let acc1 = fmla(r1, self.s[5], acc1);
        let y1 = acc1 + dp3_1;
        data[1] = y1;

        let acc2 = (x0 + x1 + x6 - x3) * self.s[4];
        let y2 = acc2 - dp5_0;
        data[2] = y2;

        let acc3 = x1 * self.s[0];
        let acc3 = fmla(x3, -self.s[1], acc3);
        let acc3 = fmla(x6, -self.s[3], acc3);
        let acc3 = fmla(x0, self.s[6], acc3);
        let y3 = acc3 - dp3_1 + dp5_0;
        data[3] = y3;

        let acc4 = -(r1 * self.s[2]);
        let acc4 = fmla(r0, self.s[5], acc4);
        let y4 = acc4 - dp3_0;
        data[4] = y4;

        let acc5 = x3 * self.s[0];
        let acc5 = fmla(x6, -self.s[1], acc5);
        let acc5 = fmla(x0, self.s[3], acc5);
        let acc5 = fmla(x1, -self.s[6], acc5);
        let y5 = acc5 + dp3_0 - dp5_0;
        data[5] = y5;

        let acc6 = x6 * self.s[0];
        let acc6 = fmla(x0, self.s[1], acc6);
        let acc6 = fmla(x1, -self.s[3], acc6);
        let acc6 = fmla(x3, -self.s[6], acc6);
        let y6 = acc6 + dp3_1 + dp5_0;
        data[6] = y6;
    }
}

define_in_place_butterfly!(Dst7Butterfly7, 7);

#[derive(Debug, Clone)]
pub(crate) struct Dst7Butterfly8<T: DctSample> {
    s: [T; 8],
}

impl<T: DctSample> Default for Dst7Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            s: [
                (1.0 / 17.0).sinpi().as_(), // S1 = sin(1*pi/17)
                (2.0 / 17.0).sinpi().as_(), // S2 = sin(2*pi/17)
                (3.0 / 17.0).sinpi().as_(), // S3 = sin(3*pi/17)
                (4.0 / 17.0).sinpi().as_(), // S4 = sin(4*pi/17)
                (5.0 / 17.0).sinpi().as_(), // S5 = sin(5*pi/17)
                (6.0 / 17.0).sinpi().as_(), // S6 = sin(6*pi/17)
                (7.0 / 17.0).sinpi().as_(), // S7 = sin(7*pi/17)
                (8.0 / 17.0).sinpi().as_(), // S8 = sin(8*pi/17)
            ],
        }
    }
}

impl<T: DctSample> Dst7Butterfly8<T>
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

        // Analytically reduced from a 17-point RDFT

        let acc0 = x0 * self.s[0];
        let acc0 = fmla(x1, self.s[1], acc0);
        let acc0 = fmla(x2, self.s[2], acc0);
        let acc0 = fmla(x3, self.s[3], acc0);
        let acc0 = fmla(x4, self.s[4], acc0);
        let acc0 = fmla(x5, self.s[5], acc0);
        let acc0 = fmla(x6, self.s[6], acc0);
        let y0 = fmla(x7, self.s[7], acc0);
        data[0] = y0;

        let acc1 = -(x5 * self.s[0]);
        let acc1 = fmla(x4, self.s[1], acc1);
        let acc1 = fmla(x0, self.s[2], acc1);
        let acc1 = fmla(x6, -self.s[3], acc1);
        let acc1 = fmla(x3, self.s[4], acc1);
        let acc1 = fmla(x1, self.s[5], acc1);
        let acc1 = fmla(x7, -self.s[6], acc1);
        let y1 = fmla(x2, self.s[7], acc1);
        data[1] = y1;

        let acc2 = x6 * self.s[0];
        let acc2 = fmla(x2, self.s[1], acc2);
        let acc2 = fmla(x3, -self.s[2], acc2);
        let acc2 = fmla(x5, -self.s[3], acc2);
        let acc2 = fmla(x0, self.s[4], acc2);
        let acc2 = fmla(x7, self.s[5], acc2);
        let acc2 = fmla(x1, self.s[6], acc2);
        let y2 = fmla(x4, -self.s[7], acc2);
        data[2] = y2;

        let acc3 = x4 * self.s[0];
        let acc3 = fmla(x6, self.s[1], acc3);
        let acc3 = fmla(x1, self.s[2], acc3);
        let acc3 = fmla(x2, -self.s[3], acc3);
        let acc3 = fmla(x7, -self.s[4], acc3);
        let acc3 = fmla(x3, -self.s[5], acc3);
        let acc3 = fmla(x0, self.s[6], acc3);
        let y3 = fmla(x5, self.s[7], acc3);
        data[3] = y3;

        let acc4 = -(x1 * self.s[0]);
        let acc4 = fmla(x3, self.s[1], acc4);
        let acc4 = fmla(x5, -self.s[2], acc4);
        let acc4 = fmla(x7, self.s[3], acc4);
        let acc4 = fmla(x6, -self.s[4], acc4);
        let acc4 = fmla(x4, self.s[5], acc4);
        let acc4 = fmla(x2, -self.s[6], acc4);
        let y4 = fmla(x0, self.s[7], acc4);
        data[4] = y4;

        let acc5 = -(x2 * self.s[0]);
        let acc5 = fmla(x5, -self.s[1], acc5);
        let acc5 = fmla(x7, -self.s[2], acc5);
        let acc5 = fmla(x4, -self.s[3], acc5);
        let acc5 = fmla(x1, -self.s[4], acc5);
        let acc5 = fmla(x0, self.s[5], acc5);
        let acc5 = fmla(x3, self.s[6], acc5);
        let y5 = fmla(x6, self.s[7], acc5);
        data[5] = y5;

        let acc6 = -(x3 * self.s[0]);
        let acc6 = fmla(x7, self.s[1], acc6);
        let acc6 = fmla(x4, -self.s[2], acc6);
        let acc6 = fmla(x0, self.s[3], acc6);
        let acc6 = fmla(x2, self.s[4], acc6);
        let acc6 = fmla(x6, -self.s[5], acc6);
        let acc6 = fmla(x5, self.s[6], acc6);
        let y6 = fmla(x1, -self.s[7], acc6);
        data[6] = y6;

        let acc7 = -(x7 * self.s[0]);
        let acc7 = fmla(x0, self.s[1], acc7);
        let acc7 = fmla(x6, self.s[2], acc7);
        let acc7 = fmla(x1, -self.s[3], acc7);
        let acc7 = fmla(x5, -self.s[4], acc7);
        let acc7 = fmla(x2, self.s[5], acc7);
        let acc7 = fmla(x4, self.s[6], acc7);
        let y7 = fmla(x3, -self.s[7], acc7);
        data[7] = y7;
    }
}

define_in_place_butterfly!(Dst7Butterfly8, 8);

#[derive(Debug, Clone)]
pub(crate) struct Dst7Butterfly16<T: DctSample> {
    s: [T; 16],
}

impl<T: DctSample> Default for Dst7Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            s: [
                (1.0 / 33.0).sinpi().as_(),  // S1 = sin(1*pi/33)
                (2.0 / 33.0).sinpi().as_(),  // S2 = sin(2*pi/33)
                (3.0 / 33.0).sinpi().as_(),  // S3 = sin(3*pi/33)
                (4.0 / 33.0).sinpi().as_(),  // S4 = sin(4*pi/33)
                (5.0 / 33.0).sinpi().as_(),  // S5 = sin(5*pi/33)
                (6.0 / 33.0).sinpi().as_(),  // S6 = sin(6*pi/33)
                (7.0 / 33.0).sinpi().as_(),  // S7 = sin(7*pi/33)
                (8.0 / 33.0).sinpi().as_(),  // S8 = sin(8*pi/33)
                (9.0 / 33.0).sinpi().as_(),  // S9 = sin(9*pi/33)
                (10.0 / 33.0).sinpi().as_(), // S10 = sin(10*pi/33)
                (11.0 / 33.0).sinpi().as_(), // S11 = sin(11*pi/33)
                (12.0 / 33.0).sinpi().as_(), // S12 = sin(12*pi/33)
                (13.0 / 33.0).sinpi().as_(), // S13 = sin(13*pi/33)
                (14.0 / 33.0).sinpi().as_(), // S14 = sin(14*pi/33)
                (15.0 / 33.0).sinpi().as_(), // S15 = sin(15*pi/33)
                (16.0 / 33.0).sinpi().as_(), // S16 = sin(16*pi/33)
            ],
        }
    }
}

impl<T: DctSample> Dst7Butterfly16<T>
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

        // Analytically reduced from a 33-point RDFT
        // dp3_0: row 0 Gp pattern
        let dp3_0 = x2 * self.s[2];
        let dp3_0 = fmla(x5, self.s[5], dp3_0);
        let dp3_0 = fmla(x8, self.s[8], dp3_0);
        let dp3_0 = fmla(x11, self.s[11], dp3_0);
        let dp3_0 = fmla(x14, self.s[14], dp3_0);
        // dp3_1: row 1 Gp pattern
        let dp3_1 = x2 * self.s[8];
        let dp3_1 = fmla(x5, self.s[14], dp3_1);
        let dp3_1 = fmla(x8, self.s[5], dp3_1);
        let dp3_1 = fmla(x11, -self.s[2], dp3_1);
        let dp3_1 = fmla(x14, -self.s[11], dp3_1);
        // dp3_2: row 2 Gp pattern
        let dp3_2 = x2 * self.s[14];
        let dp3_2 = fmla(x5, self.s[2], dp3_2);
        let dp3_2 = fmla(x8, -self.s[11], dp3_2);
        let dp3_2 = fmla(x11, -self.s[5], dp3_2);
        let dp3_2 = fmla(x14, self.s[8], dp3_2);
        // dp3_3: row 3 Gp pattern
        let dp3_3 = x2 * self.s[11];
        let dp3_3 = fmla(x5, -self.s[8], dp3_3);
        let dp3_3 = fmla(x8, -self.s[2], dp3_3);
        let dp3_3 = fmla(x11, self.s[14], dp3_3);
        let dp3_3 = fmla(x14, -self.s[5], dp3_3);
        // dp3_4: row 4 Gp pattern
        let dp3_4 = x2 * self.s[5];
        let dp3_4 = fmla(x5, -self.s[11], dp3_4);
        let dp3_4 = fmla(x8, self.s[14], dp3_4);
        let dp3_4 = fmla(x11, -self.s[8], dp3_4);
        let dp3_4 = fmla(x14, self.s[2], dp3_4);

        // G11 ({x10})  ->  1 canonical dot-product(s)
        // dp11_0: row 0 Gp pattern
        let dp11_0 = x10 * self.s[10];

        // Hoisted pre-sums reused across multiple rows.
        let r0 = x0 + x9;
        let r1 = x1 - x12;
        let r2 = x3 + x6;
        let r3 = x4 - x15;
        let r4 = x7 - x13;

        let acc0 = x0 * self.s[0];
        let acc0 = fmla(x1, self.s[1], acc0);
        let acc0 = fmla(x3, self.s[3], acc0);
        let acc0 = fmla(x4, self.s[4], acc0);
        let acc0 = fmla(x6, self.s[6], acc0);
        let acc0 = fmla(x7, self.s[7], acc0);
        let acc0 = fmla(x9, self.s[9], acc0);
        let acc0 = fmla(x12, self.s[12], acc0);
        let acc0 = fmla(x13, self.s[13], acc0);
        let acc0 = fmla(x15, self.s[15], acc0);
        let y0 = acc0 + dp3_0 + dp11_0;
        data[0] = y0;

        let acc1 = r0 * self.s[2];
        let acc1 = fmla(r1, self.s[5], acc1);
        let acc1 = fmla(r4, self.s[8], acc1);
        let acc1 = fmla(r2, self.s[11], acc1);
        let acc1 = fmla(r3, self.s[14], acc1);
        let y1 = acc1 + dp3_1;
        data[1] = y1;

        let acc2 = -(x12 * self.s[0]);
        let acc2 = fmla(x6, -self.s[1], acc2);
        let acc2 = fmla(x13, self.s[3], acc2);
        let acc2 = fmla(x0, self.s[4], acc2);
        let acc2 = fmla(x7, -self.s[6], acc2);
        let acc2 = fmla(x4, self.s[7], acc2);
        let acc2 = fmla(x1, self.s[9], acc2);
        let acc2 = fmla(x3, self.s[12], acc2);
        let acc2 = fmla(x15, self.s[13], acc2);
        let acc2 = fmla(x9, -self.s[15], acc2);
        let y2 = acc2 + dp3_2 - dp11_0;
        data[2] = y2;

        let acc3 = x13 * self.s[0];
        let acc3 = fmla(x4, -self.s[1], acc3);
        let acc3 = fmla(x9, self.s[3], acc3);
        let acc3 = fmla(x3, self.s[4], acc3);
        let acc3 = fmla(x0, self.s[6], acc3);
        let acc3 = fmla(x12, self.s[7], acc3);
        let acc3 = fmla(x7, -self.s[9], acc3);
        let acc3 = fmla(x15, -self.s[12], acc3);
        let acc3 = fmla(x1, self.s[13], acc3);
        let acc3 = fmla(x6, -self.s[15], acc3);
        let y3 = acc3 + dp3_3 + dp11_0;
        data[3] = y3;

        let acc4 = -(r2 * self.s[2]);
        let acc4 = fmla(r4, self.s[5], acc4);
        let acc4 = fmla(r0, self.s[8], acc4);
        let acc4 = fmla(r3, -self.s[11], acc4);
        let acc4 = fmla(r1, self.s[14], acc4);
        let y4 = acc4 + dp3_4;
        data[4] = y4;

        let acc5 = (x0 + x1 + x6 + x7 + x12 + x13 - x3 - x4 - x9 - x15) * self.s[10];
        let y5 = acc5 - dp11_0;
        data[5] = y5;

        let acc6 = -(x4 * self.s[0]);
        let acc6 = fmla(x9, -self.s[1], acc6);
        let acc6 = fmla(x12, -self.s[3], acc6);
        let acc6 = fmla(x7, -self.s[4], acc6);
        let acc6 = fmla(x1, self.s[6], acc6);
        let acc6 = fmla(x6, self.s[7], acc6);
        let acc6 = fmla(x15, self.s[9], acc6);
        let acc6 = fmla(x0, self.s[12], acc6);
        let acc6 = fmla(x3, -self.s[13], acc6);
        let acc6 = fmla(x13, -self.s[15], acc6);
        let y6 = acc6 - dp3_4 + dp11_0;
        data[6] = y6;

        let acc7 = r1 * self.s[2];
        let acc7 = fmla(r2, -self.s[5], acc7);
        let acc7 = fmla(r3, self.s[8], acc7);
        let acc7 = fmla(r4, -self.s[11], acc7);
        let acc7 = fmla(r0, self.s[14], acc7);
        let y7 = acc7 - dp3_3;
        data[7] = y7;

        let acc8 = -(x1 * self.s[0]);
        let acc8 = fmla(x3, self.s[1], acc8);
        let acc8 = fmla(x7, self.s[3], acc8);
        let acc8 = fmla(x9, -self.s[4], acc8);
        let acc8 = fmla(x13, -self.s[6], acc8);
        let acc8 = fmla(x15, self.s[7], acc8);
        let acc8 = fmla(x12, self.s[9], acc8);
        let acc8 = fmla(x6, -self.s[12], acc8);
        let acc8 = fmla(x4, self.s[13], acc8);
        let acc8 = fmla(x0, self.s[15], acc8);
        let y8 = acc8 - dp3_2 - dp11_0;
        data[8] = y8;

        let acc9 = x6 * self.s[0];
        let acc9 = fmla(x13, self.s[1], acc9);
        let acc9 = fmla(x4, self.s[3], acc9);
        let acc9 = fmla(x1, -self.s[4], acc9);
        let acc9 = fmla(x15, -self.s[6], acc9);
        let acc9 = fmla(x9, -self.s[7], acc9);
        let acc9 = fmla(x3, self.s[9], acc9);
        let acc9 = fmla(x7, self.s[12], acc9);
        let acc9 = fmla(x0, self.s[13], acc9);
        let acc9 = fmla(x12, -self.s[15], acc9);
        let y9 = acc9 - dp3_1 + dp11_0;
        data[9] = y9;

        let acc10 = -(r4 * self.s[2]);
        let acc10 = fmla(r3, -self.s[5], acc10);
        let acc10 = fmla(r1, -self.s[8], acc10);
        let acc10 = fmla(r0, self.s[11], acc10);
        let acc10 = fmla(r2, self.s[14], acc10);
        let y10 = acc10 - dp3_0;
        data[10] = y10;

        let acc11 = x9 * self.s[0];
        let acc11 = fmla(x12, -self.s[1], acc11);
        let acc11 = fmla(x6, self.s[3], acc11);
        let acc11 = fmla(x15, -self.s[4], acc11);
        let acc11 = fmla(x3, self.s[6], acc11);
        let acc11 = fmla(x13, -self.s[7], acc11);
        let acc11 = fmla(x0, self.s[9], acc11);
        let acc11 = fmla(x1, -self.s[12], acc11);
        let acc11 = fmla(x7, -self.s[13], acc11);
        let acc11 = fmla(x4, -self.s[15], acc11);
        let y11 = acc11 + dp3_0 - dp11_0;
        data[11] = y11;

        let acc12 = -(x3 * self.s[0]);
        let acc12 = fmla(x7, self.s[1], acc12);
        let acc12 = fmla(x15, self.s[3], acc12);
        let acc12 = fmla(x12, -self.s[4], acc12);
        let acc12 = fmla(x4, -self.s[6], acc12);
        let acc12 = fmla(x0, self.s[7], acc12);
        let acc12 = fmla(x6, -self.s[9], acc12);
        let acc12 = fmla(x13, self.s[12], acc12);
        let acc12 = fmla(x9, -self.s[13], acc12);
        let acc12 = fmla(x1, -self.s[15], acc12);
        let y12 = acc12 + dp3_1 + dp11_0;
        data[12] = y12;

        let acc13 = r3 * self.s[2];
        let acc13 = fmla(r0, self.s[5], acc13);
        let acc13 = fmla(r2, -self.s[8], acc13);
        let acc13 = fmla(r1, -self.s[11], acc13);
        let acc13 = fmla(r4, self.s[14], acc13);
        let y13 = acc13 + dp3_2;
        data[13] = y13;

        let acc14 = -(x7 * self.s[0]);
        let acc14 = fmla(x15, self.s[1], acc14);
        let acc14 = fmla(x0, self.s[3], acc14);
        let acc14 = fmla(x6, self.s[4], acc14);
        let acc14 = fmla(x9, self.s[6], acc14);
        let acc14 = fmla(x1, -self.s[7], acc14);
        let acc14 = fmla(x13, self.s[9], acc14);
        let acc14 = fmla(x4, self.s[12], acc14);
        let acc14 = fmla(x12, -self.s[13], acc14);
        let acc14 = fmla(x3, -self.s[15], acc14);
        let y14 = acc14 + dp3_3 - dp11_0;
        data[14] = y14;

        let acc15 = -(x15 * self.s[0]);
        let acc15 = fmla(x0, self.s[1], acc15);
        let acc15 = fmla(x1, -self.s[3], acc15);
        let acc15 = fmla(x13, -self.s[4], acc15);
        let acc15 = fmla(x12, self.s[6], acc15);
        let acc15 = fmla(x3, -self.s[7], acc15);
        let acc15 = fmla(x4, self.s[9], acc15);
        let acc15 = fmla(x9, -self.s[12], acc15);
        let acc15 = fmla(x6, self.s[13], acc15);
        let acc15 = fmla(x7, -self.s[15], acc15);
        let y15 = acc15 + dp3_4 + dp11_0;
        data[15] = y15;
    }
}

define_in_place_butterfly!(Dst7Butterfly16, 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::gen_test_butterfly;

    gen_test_butterfly!(test_bf_dst7_2, f64, Dst7Butterfly2, 2, 1e-7, naive_dst7);
    gen_test_butterfly!(test_bf_dst7_3, f64, Dst7Butterfly3, 3, 1e-7, naive_dst7);
    gen_test_butterfly!(test_bf_dst7_4, f64, Dst7Butterfly4, 4, 1e-7, naive_dst7);
    gen_test_butterfly!(test_bf_dst7_5, f64, Dst7Butterfly5, 5, 1e-7, naive_dst7);
    gen_test_butterfly!(test_bf_dst7_6, f64, Dst7Butterfly6, 6, 1e-7, naive_dst7);
    gen_test_butterfly!(test_bf_dst7_7, f64, Dst7Butterfly7, 7, 1e-7, naive_dst7);
    gen_test_butterfly!(test_bf_dst7_8, f64, Dst7Butterfly8, 8, 1e-7, naive_dst7);
    gen_test_butterfly!(test_bf_dst7_16, f64, Dst7Butterfly16, 16, 1e-7, naive_dst7);
}
