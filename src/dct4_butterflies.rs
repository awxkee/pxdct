/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::mla::fmla;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly5<T> {
    twiddles: [T; 5],
}

impl<T: DctSample> Default for Dct4Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| {
                ((x as f64 + 0.5f64).as_() * 0.5f64.as_() / 5f64.as_()).cospi()
            }),
        }
    }
}

impl<T: DctSample> Dct4Butterfly5<T> {
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 5]) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];

        let y0 = fmla(
            self.twiddles[0],
            x0,
            fmla(
                self.twiddles[1],
                x1,
                fmla(
                    self.twiddles[2],
                    x2,
                    fmla(self.twiddles[3], x3, self.twiddles[4] * x4),
                ),
            ),
        );
        let y1 = fmla(
            self.twiddles[1],
            x0,
            fmla(
                self.twiddles[4],
                x1,
                fmla(
                    -self.twiddles[2],
                    x2,
                    fmla(-self.twiddles[0], x3, -self.twiddles[3] * x4),
                ),
            ),
        );
        let y2 = fmla(
            self.twiddles[2],
            x0,
            fmla(
                -self.twiddles[2],
                x1,
                fmla(
                    -self.twiddles[2],
                    x2,
                    fmla(self.twiddles[2], x3, self.twiddles[2] * x4),
                ),
            ),
        );
        let y3 = fmla(
            self.twiddles[3],
            x0,
            fmla(
                -self.twiddles[0],
                x1,
                fmla(
                    self.twiddles[2],
                    x2,
                    fmla(self.twiddles[4], x3, -self.twiddles[1] * x4),
                ),
            ),
        );
        let y4 = fmla(
            self.twiddles[4],
            x0,
            fmla(
                -self.twiddles[3],
                x1,
                fmla(
                    self.twiddles[2],
                    x2,
                    fmla(-self.twiddles[1], x3, self.twiddles[0] * x4),
                ),
            ),
        );

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(5) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(5) {
            self.exec((&mut chunk[..5]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        5
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly7<T> {
    twiddles: [T; 7],
}

impl<T: DctSample> Default for Dct4Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| {
                ((x as f64 + 0.5f64).as_() * 0.5f64.as_() / 7f64.as_()).cospi()
            }),
        }
    }
}

impl<T: DctSample> Dct4Butterfly7<T> {
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 7]) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];

        let y0 = fmla(
            self.twiddles[0],
            x0,
            self.twiddles[1] * x1
                + self.twiddles[2] * x2
                + self.twiddles[3] * x3
                + self.twiddles[4] * x4
                + self.twiddles[5] * x5
                + self.twiddles[6] * x6,
        );
        let y1 = fmla(
            self.twiddles[1],
            x0,
            self.twiddles[4] * x1
                - self.twiddles[6] * x2
                - self.twiddles[3] * x3
                - self.twiddles[0] * x4
                - self.twiddles[2] * x5
                - self.twiddles[5] * x6,
        );
        let y2 = fmla(
            self.twiddles[2],
            x0,
            -self.twiddles[6] * x1 - self.twiddles[1] * x2 - self.twiddles[3] * x3
                + self.twiddles[5] * x4
                + self.twiddles[0] * x5
                + self.twiddles[4] * x6,
        );
        let y3 = fmla(
            self.twiddles[3],
            x0,
            -self.twiddles[3] * x1 - self.twiddles[3] * x2
                + self.twiddles[3] * x3
                + self.twiddles[3] * x4
                - self.twiddles[3] * x5
                - self.twiddles[3] * x6,
        );
        let y4 = fmla(
            self.twiddles[4],
            x0,
            -self.twiddles[0] * x1 + self.twiddles[5] * x2 + self.twiddles[3] * x3
                - self.twiddles[1] * x4
                + self.twiddles[6] * x5
                + self.twiddles[2] * x6,
        );
        let y5 = fmla(
            self.twiddles[5],
            x0,
            -self.twiddles[2] * x1 + self.twiddles[0] * x2 - self.twiddles[3] * x3
                + self.twiddles[6] * x4
                + self.twiddles[4] * x5
                - self.twiddles[1] * x6,
        );
        let y6 = fmla(
            self.twiddles[6],
            x0,
            -self.twiddles[5] * x1 + self.twiddles[4] * x2 - self.twiddles[3] * x3
                + self.twiddles[2] * x4
                - self.twiddles[1] * x5
                + self.twiddles[0] * x6,
        );

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
        data[5] = y5;
        data[6] = y6;
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(7) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(7) {
            self.exec((&mut chunk[..7]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        7
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly9<T> {
    twiddles: [T; 9],
}

impl<T: DctSample> Default for Dct4Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| {
                ((x as f64 + 0.5f64).as_() * 0.5f64.as_() / 9f64.as_()).cospi()
            }),
        }
    }
}

impl<T: DctSample> Dct4Butterfly9<T> {
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 9]) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];
        let x7 = data[7];
        let x8 = data[8];

        let tmp5 = self.twiddles[7] * x7;
        let tmp6 = self.twiddles[1] * x1;
        let tmp1 = self.twiddles[4] * x1;
        let tmp2 = self.twiddles[1] * x7;
        let tmp3 = self.twiddles[7] * x1;
        let tmp0 = self.twiddles[4] * x4;
        let tmp4 = self.twiddles[4] * x7;

        let y0 = fmla(
            self.twiddles[0],
            x0,
            tmp6 + self.twiddles[2] * x2
                + self.twiddles[3] * x3
                + tmp0
                + self.twiddles[5] * x5
                + self.twiddles[6] * x6
                + tmp5
                + self.twiddles[8] * x8,
        );
        let y1 = fmla(
            self.twiddles[1],
            x0,
            tmp1 + self.twiddles[7] * x2
                - self.twiddles[7] * x3
                - tmp0
                - self.twiddles[1] * x5
                - self.twiddles[1] * x6
                - tmp4
                - self.twiddles[7] * x8,
        );
        let y2 = fmla(
            self.twiddles[2],
            x0,
            tmp3 - self.twiddles[5] * x2 - self.twiddles[0] * x3 - tmp0
                + self.twiddles[8] * x5
                + self.twiddles[3] * x6
                + tmp2
                + self.twiddles[6] * x8,
        );
        let y3 = fmla(
            self.twiddles[3],
            x0,
            -tmp3 - self.twiddles[0] * x2 - self.twiddles[6] * x3 + tmp0 + self.twiddles[2] * x5
                - self.twiddles[8] * x6
                - tmp2
                - self.twiddles[5] * x8,
        );
        let y4 = fmla(
            self.twiddles[4],
            x0,
            -tmp1 - self.twiddles[4] * x2 + self.twiddles[4] * x3 + tmp0
                - self.twiddles[4] * x5
                - self.twiddles[4] * x6
                + tmp4
                + self.twiddles[4] * x8,
        );
        let y5 = fmla(
            self.twiddles[5],
            x0,
            -tmp6 + self.twiddles[8] * x2 + self.twiddles[2] * x3 - tmp0 - self.twiddles[6] * x5
                + self.twiddles[0] * x6
                - tmp5
                - self.twiddles[3] * x8,
        );
        let y6 = fmla(
            self.twiddles[6],
            x0,
            -tmp6 + self.twiddles[3] * x2 - self.twiddles[8] * x3 - tmp0 + self.twiddles[0] * x5
                - self.twiddles[5] * x6
                - tmp5
                + self.twiddles[2] * x8,
        );
        let y7 = fmla(
            self.twiddles[7],
            x0,
            -tmp1 + self.twiddles[1] * x2 - self.twiddles[1] * x3 + tmp0
                - self.twiddles[7] * x5
                - self.twiddles[7] * x6
                + tmp4
                - self.twiddles[1] * x8,
        );
        let y8 = fmla(
            self.twiddles[8],
            x0,
            -tmp3 + self.twiddles[6] * x2 - self.twiddles[5] * x3 + tmp0 - self.twiddles[3] * x5
                + self.twiddles[2] * x6
                - tmp2
                + self.twiddles[0] * x8,
        );

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

impl<T: DctSample> PxdctExecutor<T> for Dct4Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(9) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(9) {
            self.exec((&mut chunk[..9]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        9
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly11<T> {
    twiddles: [T; 11],
}

impl<T: DctSample> Default for Dct4Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| {
                ((x as f64 + 0.5f64).as_() * 0.5f64.as_() / 11f64.as_()).cospi()
            }),
        }
    }
}

impl<T: DctSample> Dct4Butterfly11<T> {
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 11]) {
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

        let y0 = fmla(
            self.twiddles[0],
            x0,
            self.twiddles[1] * x1
                + self.twiddles[2] * x2
                + self.twiddles[3] * x3
                + self.twiddles[4] * x4
                + self.twiddles[5] * x5
                + self.twiddles[6] * x6
                + self.twiddles[7] * x7
                + self.twiddles[8] * x8
                + self.twiddles[9] * x9
                + self.twiddles[10] * x10,
        );
        let y1 = fmla(
            self.twiddles[1],
            x0,
            self.twiddles[4] * x1 + self.twiddles[7] * x2 + self.twiddles[10] * x3
                - self.twiddles[8] * x4
                - self.twiddles[5] * x5
                - self.twiddles[2] * x6
                - self.twiddles[0] * x7
                - self.twiddles[3] * x8
                - self.twiddles[6] * x9
                - self.twiddles[9] * x10,
        );
        let y2 = fmla(
            self.twiddles[2],
            x0,
            self.twiddles[7] * x1
                - self.twiddles[9] * x2
                - self.twiddles[4] * x3
                - self.twiddles[0] * x4
                - self.twiddles[5] * x5
                - self.twiddles[10] * x6
                + self.twiddles[6] * x7
                + self.twiddles[1] * x8
                + self.twiddles[3] * x9
                + self.twiddles[8] * x10,
        );
        let y3 = fmla(
            self.twiddles[3],
            x0,
            self.twiddles[10] * x1
                - self.twiddles[4] * x2
                - self.twiddles[2] * x3
                - self.twiddles[9] * x4
                + self.twiddles[5] * x5
                + self.twiddles[1] * x6
                + self.twiddles[8] * x7
                - self.twiddles[6] * x8
                - self.twiddles[0] * x9
                - self.twiddles[7] * x10,
        );
        let y4 = fmla(
            self.twiddles[4],
            x0,
            -self.twiddles[8] * x1 - self.twiddles[0] * x2 - self.twiddles[9] * x3
                + self.twiddles[3] * x4
                + self.twiddles[5] * x5
                - self.twiddles[7] * x6
                - self.twiddles[1] * x7
                - self.twiddles[10] * x8
                + self.twiddles[2] * x9
                + self.twiddles[6] * x10,
        );
        let y5 = fmla(
            self.twiddles[5],
            x0,
            -self.twiddles[5] * x1 - self.twiddles[5] * x2
                + self.twiddles[5] * x3
                + self.twiddles[5] * x4
                - self.twiddles[5] * x5
                - self.twiddles[5] * x6
                + self.twiddles[5] * x7
                + self.twiddles[5] * x8
                - self.twiddles[5] * x9
                - self.twiddles[5] * x10,
        );
        let y6 = fmla(
            self.twiddles[6],
            x0,
            -self.twiddles[2] * x1 - self.twiddles[10] * x2 + self.twiddles[1] * x3
                - self.twiddles[7] * x4
                - self.twiddles[5] * x5
                + self.twiddles[3] * x6
                + self.twiddles[9] * x7
                - self.twiddles[0] * x8
                + self.twiddles[8] * x9
                + self.twiddles[4] * x10,
        );
        let y7 = fmla(
            self.twiddles[7],
            x0,
            -self.twiddles[0] * x1 + self.twiddles[6] * x2 + self.twiddles[8] * x3
                - self.twiddles[1] * x4
                + self.twiddles[5] * x5
                + self.twiddles[9] * x6
                - self.twiddles[2] * x7
                + self.twiddles[4] * x8
                + self.twiddles[10] * x9
                - self.twiddles[3] * x10,
        );
        let y8 = fmla(
            self.twiddles[8],
            x0,
            -self.twiddles[3] * x1 + self.twiddles[1] * x2
                - self.twiddles[6] * x3
                - self.twiddles[10] * x4
                + self.twiddles[5] * x5
                - self.twiddles[0] * x6
                + self.twiddles[4] * x7
                - self.twiddles[9] * x8
                - self.twiddles[7] * x9
                + self.twiddles[2] * x10,
        );
        let y9 = fmla(
            self.twiddles[9],
            x0,
            -self.twiddles[6] * x1 + self.twiddles[3] * x2 - self.twiddles[0] * x3
                + self.twiddles[2] * x4
                - self.twiddles[5] * x5
                + self.twiddles[8] * x6
                + self.twiddles[10] * x7
                - self.twiddles[7] * x8
                + self.twiddles[4] * x9
                - self.twiddles[1] * x10,
        );
        let y10 = fmla(
            self.twiddles[10],
            x0,
            -self.twiddles[9] * x1 + self.twiddles[8] * x2 - self.twiddles[7] * x3
                + self.twiddles[6] * x4
                - self.twiddles[5] * x5
                + self.twiddles[4] * x6
                - self.twiddles[3] * x7
                + self.twiddles[2] * x8
                - self.twiddles[1] * x9
                + self.twiddles[0] * x10,
        );

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

impl<T: DctSample> PxdctExecutor<T> for Dct4Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(11) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(11) {
            self.exec((&mut chunk[..11]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        11
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly13<T> {
    twiddles: [T; 13],
}

impl<T: DctSample> Default for Dct4Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| {
                ((x as f64 + 0.5f64).as_() * 0.5f64.as_() / 13f64.as_()).cospi()
            }),
        }
    }
}

impl<T: DctSample> Dct4Butterfly13<T> {
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 13]) {
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

        let y0 = fmla(
            self.twiddles[0],
            x0,
            self.twiddles[1] * x1
                + self.twiddles[2] * x2
                + self.twiddles[3] * x3
                + self.twiddles[4] * x4
                + self.twiddles[5] * x5
                + self.twiddles[6] * x6
                + self.twiddles[7] * x7
                + self.twiddles[8] * x8
                + self.twiddles[9] * x9
                + self.twiddles[10] * x10
                + self.twiddles[11] * x11
                + self.twiddles[12] * x12,
        );
        let y1 = fmla(
            self.twiddles[1],
            x0,
            self.twiddles[4] * x1 + self.twiddles[7] * x2 + self.twiddles[10] * x3
                - self.twiddles[12] * x4
                - self.twiddles[9] * x5
                - self.twiddles[6] * x6
                - self.twiddles[3] * x7
                - self.twiddles[0] * x8
                - self.twiddles[2] * x9
                - self.twiddles[5] * x10
                - self.twiddles[8] * x11
                - self.twiddles[11] * x12,
        );
        let y2 = fmla(
            self.twiddles[2],
            x0,
            self.twiddles[7] * x1 + self.twiddles[12] * x2
                - self.twiddles[8] * x3
                - self.twiddles[3] * x4
                - self.twiddles[1] * x5
                - self.twiddles[6] * x6
                - self.twiddles[11] * x7
                + self.twiddles[9] * x8
                + self.twiddles[4] * x9
                + self.twiddles[0] * x10
                + self.twiddles[5] * x11
                + self.twiddles[10] * x12,
        );
        let y3 = fmla(
            self.twiddles[3],
            x0,
            self.twiddles[10] * x1
                - self.twiddles[8] * x2
                - self.twiddles[1] * x3
                - self.twiddles[5] * x4
                - self.twiddles[12] * x5
                + self.twiddles[6] * x6
                + self.twiddles[0] * x7
                + self.twiddles[7] * x8
                - self.twiddles[11] * x9
                - self.twiddles[4] * x10
                - self.twiddles[2] * x11
                - self.twiddles[9] * x12,
        );
        let y4 = fmla(
            self.twiddles[4],
            x0,
            -self.twiddles[12] * x1 - self.twiddles[3] * x2 - self.twiddles[5] * x3
                + self.twiddles[11] * x4
                + self.twiddles[2] * x5
                + self.twiddles[6] * x6
                - self.twiddles[10] * x7
                - self.twiddles[1] * x8
                - self.twiddles[7] * x9
                + self.twiddles[9] * x10
                + self.twiddles[0] * x11
                + self.twiddles[8] * x12,
        );
        let y5 = fmla(
            self.twiddles[5],
            x0,
            -self.twiddles[9] * x1 - self.twiddles[1] * x2 - self.twiddles[12] * x3
                + self.twiddles[2] * x4
                + self.twiddles[8] * x5
                - self.twiddles[6] * x6
                - self.twiddles[4] * x7
                + self.twiddles[10] * x8
                + self.twiddles[0] * x9
                + self.twiddles[11] * x10
                - self.twiddles[3] * x11
                - self.twiddles[7] * x12,
        );
        let y6 = fmla(
            self.twiddles[6],
            x0,
            -self.twiddles[6] * x1 - self.twiddles[6] * x2
                + self.twiddles[6] * x3
                + self.twiddles[6] * x4
                - self.twiddles[6] * x5
                - self.twiddles[6] * x6
                + self.twiddles[6] * x7
                + self.twiddles[6] * x8
                - self.twiddles[6] * x9
                - self.twiddles[6] * x10
                + self.twiddles[6] * x11
                + self.twiddles[6] * x12,
        );
        let y7 = fmla(
            self.twiddles[7],
            x0,
            -self.twiddles[3] * x1 - self.twiddles[11] * x2 + self.twiddles[0] * x3
                - self.twiddles[10] * x4
                - self.twiddles[4] * x5
                + self.twiddles[6] * x6
                + self.twiddles[8] * x7
                - self.twiddles[2] * x8
                - self.twiddles[12] * x9
                + self.twiddles[1] * x10
                - self.twiddles[9] * x11
                - self.twiddles[5] * x12,
        );
        let y8 = fmla(
            self.twiddles[8],
            x0,
            -self.twiddles[0] * x1 + self.twiddles[9] * x2 + self.twiddles[7] * x3
                - self.twiddles[1] * x4
                + self.twiddles[10] * x5
                + self.twiddles[6] * x6
                - self.twiddles[2] * x7
                + self.twiddles[11] * x8
                + self.twiddles[5] * x9
                - self.twiddles[3] * x10
                + self.twiddles[12] * x11
                + self.twiddles[4] * x12,
        );
        let y9 = fmla(
            self.twiddles[9],
            x0,
            -self.twiddles[2] * x1 + self.twiddles[4] * x2
                - self.twiddles[11] * x3
                - self.twiddles[7] * x4
                + self.twiddles[0] * x5
                - self.twiddles[6] * x6
                - self.twiddles[12] * x7
                + self.twiddles[5] * x8
                - self.twiddles[1] * x9
                + self.twiddles[8] * x10
                + self.twiddles[10] * x11
                - self.twiddles[3] * x12,
        );
        let y10 = fmla(
            self.twiddles[10],
            x0,
            -self.twiddles[5] * x1 + self.twiddles[0] * x2 - self.twiddles[4] * x3
                + self.twiddles[9] * x4
                + self.twiddles[11] * x5
                - self.twiddles[6] * x6
                + self.twiddles[1] * x7
                - self.twiddles[3] * x8
                + self.twiddles[8] * x9
                + self.twiddles[12] * x10
                - self.twiddles[7] * x11
                + self.twiddles[2] * x12,
        );
        let y11 = fmla(
            self.twiddles[11],
            x0,
            -self.twiddles[8] * x1 + self.twiddles[5] * x2 - self.twiddles[2] * x3
                + self.twiddles[0] * x4
                - self.twiddles[3] * x5
                + self.twiddles[6] * x6
                - self.twiddles[9] * x7
                + self.twiddles[12] * x8
                + self.twiddles[10] * x9
                - self.twiddles[7] * x10
                + self.twiddles[4] * x11
                - self.twiddles[1] * x12,
        );
        let y12 = fmla(
            self.twiddles[12],
            x0,
            -self.twiddles[11] * x1 + self.twiddles[10] * x2 - self.twiddles[9] * x3
                + self.twiddles[8] * x4
                - self.twiddles[7] * x5
                + self.twiddles[6] * x6
                - self.twiddles[5] * x7
                + self.twiddles[4] * x8
                - self.twiddles[3] * x9
                + self.twiddles[2] * x10
                - self.twiddles[1] * x11
                + self.twiddles[0] * x12,
        );

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

impl<T: DctSample> PxdctExecutor<T> for Dct4Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(13) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(13) {
            self.exec((&mut chunk[..13]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        13
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_dct4;
    use rand::Rng;

    gen_test_butterfly!(test_bf_dct4_5, f64, Dct4Butterfly5, 5, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct4_7, f64, Dct4Butterfly7, 7, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct4_9, f64, Dct4Butterfly9, 9, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct4_11, f64, Dct4Butterfly11, 11, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct4_13, f64, Dct4Butterfly13, 13, 1e-7, naive_dct4);
}
