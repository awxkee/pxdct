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
use crate::dst3_butterfly::{Dst3Butterfly2, Dst3Butterfly4};
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct3Butterfly2<T> {
    phantom: PhantomData<T>,
}

impl<T: DctSample> Dct3Butterfly2<T> {
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 2]) {
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

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly2<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(2) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(2) {
            self.exec((&mut chunk[..2]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        2
    }
}

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
    pub(crate) fn exec(&self, data: &mut [T; 3]) {
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

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(3) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(3) {
            self.exec((&mut chunk[..3]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        3
    }
}

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
    pub(crate) fn exec(&self, data: &mut [T; 5]) {
        let x0 = data[0] * T::HALF;
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];

        let y0 = fmla(
            self.twiddle0.re,
            x1,
            fmla(
                self.twiddle1.re,
                x2,
                fmla(-self.twiddle1.im, x3, fmla(-self.twiddle0.im, x4, x0)),
            ),
        );

        let y1 = fmla(
            -self.twiddle1.im,
            x1,
            fmla(
                self.twiddle0.im,
                x2,
                fmla(-self.twiddle0.re, x3, fmla(-self.twiddle1.re, x4, x0)),
            ),
        );

        let y2 = x0 - x2 + x4;

        let y3 = fmla(
            self.twiddle1.im,
            x1,
            fmla(
                self.twiddle0.im,
                x2,
                fmla(self.twiddle0.re, x3, fmla(-self.twiddle1.re, x4, x0)),
            ),
        );

        let y4 = fmla(
            -self.twiddle0.re,
            x1,
            fmla(
                self.twiddle1.re,
                x2,
                fmla(self.twiddle1.im, x3, fmla(-self.twiddle0.im, x4, x0)),
            ),
        );

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly5<T>
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
    pub(crate) fn exec(&self, data: &mut [T; 4]) {
        // DCT-3 split radix with n = 4
        let mut evens = [data[0], data[2]];
        self.bf2.exec(&mut evens);
        let lower_dct4 = fmla(data[1], self.twiddle.re, data[3] * self.twiddle.im);
        let upper_dct4 = fmla(data[1], self.twiddle.im, -data[3] * self.twiddle.re);

        data[1] = evens[1] + upper_dct4;
        data[3] = evens[0] - lower_dct4;
        data[0] = evens[0] + lower_dct4;
        data[2] = evens[1] - upper_dct4;
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(4) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(4) {
            self.exec((&mut chunk[..4]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        4
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly6<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}

impl<T: DctSample> Default for Dct3Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 24).conj(),
            twiddle1: compute_twiddle(2, 24).conj(),
            twiddle2: compute_twiddle(3, 24).conj(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 6]) {
        let x0 = data[0] * T::HALF;
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];

        let tmp1 = self.twiddle2.re * x3;
        let tmp5 = self.twiddle1.im * x4;
        let tmp0 = self.twiddle1.re * x2;
        let tmp3 = self.twiddle2.re * x1;
        let tmp2 = self.twiddle0.im * x5;
        let tmp8 = self.twiddle0.re * x5;
        let tmp4 = self.twiddle0.im * x1;
        let tmp7 = self.twiddle2.re * x5;
        let tmp6 = self.twiddle0.re * x1;
        let y0 = tmp6 + tmp0 + tmp1 + tmp5 + tmp2 + x0;
        let y1 = tmp3 - tmp1 - x4 - tmp7 + x0;
        let y2 = tmp4 - tmp0 - tmp1 + tmp5 + tmp8 + x0;
        let y3 = -tmp4 - tmp0 + tmp1 + tmp5 - tmp8 + x0;
        let y4 = -tmp3 + tmp1 - x4 + tmp7 + x0;
        let y5 = -tmp6 + tmp0 - tmp1 + tmp5 - tmp2 + x0;

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
        data[5] = y5;
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(6) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(6) {
            self.exec((&mut chunk[..6]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        6
    }
}

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
    pub(crate) fn exec(&self, data: &mut [T; 7]) {
        let x0 = data[0] * T::HALF;
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];

        let y0 = fmla(
            self.twiddle0.re,
            x1,
            fmla(
                self.twiddle1.re,
                x2,
                fmla(
                    self.twiddle2.re,
                    x3,
                    fmla(
                        self.twiddle2.im,
                        x4,
                        fmla(self.twiddle1.im, x5, fmla(self.twiddle0.im, x6, x0)),
                    ),
                ),
            ),
        );
        let y1 = fmla(
            self.twiddle2.re,
            x1,
            fmla(
                self.twiddle0.im,
                x2,
                fmla(
                    -self.twiddle1.im,
                    x3,
                    fmla(
                        -self.twiddle1.re,
                        x4,
                        fmla(-self.twiddle0.re, x5, fmla(-self.twiddle2.im, x6, x0)),
                    ),
                ),
            ),
        );
        let y2 = fmla(
            self.twiddle1.im,
            x1,
            fmla(
                -self.twiddle2.im,
                x2,
                fmla(
                    -self.twiddle0.re,
                    x3,
                    fmla(
                        -self.twiddle0.im,
                        x4,
                        fmla(self.twiddle2.re, x5, fmla(self.twiddle1.re, x6, x0)),
                    ),
                ),
            ),
        );
        let y3 = x0 - x2 + x4 - x6;
        let y4 = fmla(
            -self.twiddle1.im,
            x1,
            fmla(
                -self.twiddle2.im,
                x2,
                fmla(
                    self.twiddle0.re,
                    x3,
                    fmla(
                        -self.twiddle0.im,
                        x4,
                        fmla(-self.twiddle2.re, x5, fmla(self.twiddle1.re, x6, x0)),
                    ),
                ),
            ),
        );
        let y5 = fmla(
            -self.twiddle2.re,
            x1,
            fmla(
                self.twiddle0.im,
                x2,
                fmla(
                    self.twiddle1.im,
                    x3,
                    fmla(
                        -self.twiddle1.re,
                        x4,
                        fmla(self.twiddle0.re, x5, fmla(-self.twiddle2.im, x6, x0)),
                    ),
                ),
            ),
        );
        let y6 = fmla(
            -self.twiddle0.re,
            x1,
            fmla(
                self.twiddle1.re,
                x2,
                fmla(
                    -self.twiddle2.re,
                    x3,
                    fmla(
                        self.twiddle2.im,
                        x4,
                        fmla(-self.twiddle1.im, x5, fmla(self.twiddle0.im, x6, x0)),
                    ),
                ),
            ),
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

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly7<T>
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
    pub(crate) fn exec(&self, data: &mut [T; 8]) {
        //process the evens
        let mut dct3_buffer = [data[0], data[2], data[4], data[6]];
        self.bf4.exec(&mut dct3_buffer);

        //process the odds
        let mut odds_n1 = [data[1] * T::TWO, data[3] + data[5]];
        let mut odds_n3 = [data[3] - data[5], data[7] * T::TWO];

        self.bf2.exec(&mut odds_n1);
        self.bf2_dst.exec(&mut odds_n3);

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

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(8) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(8) {
            self.exec((&mut chunk[..8]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        8
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct3Butterfly9<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
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
            twiddle2: compute_twiddle(3, 36).conj(),
            twiddle3: compute_twiddle(4, 36).conj(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 9]) {
        let x0 = data[0] * T::HALF;
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4];
        let x5 = data[5];
        let x6 = data[6];
        let x7 = data[7];
        let x8 = data[8];

        let y0 = fmla(
            self.twiddle0.re,
            x1,
            self.twiddle1.re * x2
                + self.twiddle2.re * x3
                + self.twiddle3.re * x4
                + self.twiddle3.im * x5
                + self.twiddle2.im * x6
                + self.twiddle1.im * x7
                + self.twiddle0.im * x8
                + x0,
        );
        let y1 = fmla(
            self.twiddle2.re,
            x1,
            self.twiddle2.im * x2
                - self.twiddle2.im * x4
                - self.twiddle2.re * x5
                - x6
                - self.twiddle2.re * x7
                - self.twiddle2.im * x8
                + x0,
        );
        let y2 = fmla(
            self.twiddle3.im,
            x1,
            -self.twiddle0.im * x2
                - self.twiddle2.re * x3
                - self.twiddle1.re * x4
                - self.twiddle1.im * x5
                + self.twiddle2.im * x6
                + self.twiddle0.re * x7
                + self.twiddle3.re * x8
                + x0,
        );
        let y3 = fmla(
            self.twiddle1.im,
            x1,
            -self.twiddle3.re * x2 - self.twiddle2.re * x3
                + self.twiddle0.im * x4
                + self.twiddle0.re * x5
                + self.twiddle2.im * x6
                - self.twiddle3.im * x7
                - self.twiddle1.re * x8
                + x0,
        );
        let y4 = x0 - x2 + x4 - x6 + x8;
        let y5 = fmla(
            -self.twiddle1.im,
            x1,
            -self.twiddle3.re * x2 + self.twiddle2.re * x3 + self.twiddle0.im * x4
                - self.twiddle0.re * x5
                + self.twiddle2.im * x6
                + self.twiddle3.im * x7
                - self.twiddle1.re * x8
                + x0,
        );
        let y6 = fmla(
            -self.twiddle3.im,
            x1,
            -self.twiddle0.im * x2 + self.twiddle2.re * x3 - self.twiddle1.re * x4
                + self.twiddle1.im * x5
                + self.twiddle2.im * x6
                - self.twiddle0.re * x7
                + self.twiddle3.re * x8
                + x0,
        );
        let y7 = fmla(
            -self.twiddle2.re,
            x1,
            self.twiddle2.im * x2 - self.twiddle2.im * x4 + self.twiddle2.re * x5 - x6
                + self.twiddle2.re * x7
                - self.twiddle2.im * x8
                + x0,
        );
        let y8 = fmla(
            -self.twiddle0.re,
            x1,
            self.twiddle1.re * x2 - self.twiddle2.re * x3 + self.twiddle3.re * x4
                - self.twiddle3.im * x5
                + self.twiddle2.im * x6
                - self.twiddle1.im * x7
                + self.twiddle0.im * x8
                + x0,
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

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly9<T>
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
    pub(crate) fn exec(&self, data: &mut [T; 11]) {
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

        let y0 = fmla(
            self.twiddle0.re,
            x1,
            self.twiddle1.re * x2
                + self.twiddle2.re * x3
                + self.twiddle3.re * x4
                + self.twiddle4.re * x5
                + self.twiddle4.im * x6
                + self.twiddle3.im * x7
                + self.twiddle2.im * x8
                + self.twiddle1.im * x9
                + self.twiddle0.im * x10
                + x0,
        );
        let y1 = fmla(
            self.twiddle2.re,
            x1,
            self.twiddle4.im * x2 + self.twiddle1.im * x3
                - self.twiddle0.im * x4
                - self.twiddle3.im * x5
                - self.twiddle3.re * x6
                - self.twiddle0.re * x7
                - self.twiddle1.re * x8
                - self.twiddle4.re * x9
                - self.twiddle2.im * x10
                + x0,
        );
        let y2 = fmla(
            self.twiddle4.re,
            x1,
            self.twiddle0.im * x2
                - self.twiddle3.im * x3
                - self.twiddle1.re * x4
                - self.twiddle2.re * x5
                - self.twiddle2.im * x6
                + self.twiddle1.im * x7
                + self.twiddle3.re * x8
                + self.twiddle0.re * x9
                + self.twiddle4.im * x10
                + x0,
        );
        let y3 = fmla(
            self.twiddle3.im,
            x1,
            -self.twiddle2.im * x2 - self.twiddle0.re * x3 - self.twiddle4.im * x4
                + self.twiddle1.im * x5
                + self.twiddle1.re * x6
                + self.twiddle4.re * x7
                - self.twiddle0.im * x8
                - self.twiddle2.re * x9
                - self.twiddle3.re * x10
                + x0,
        );
        let y4 = fmla(
            self.twiddle1.im,
            x1,
            -self.twiddle3.re * x2 - self.twiddle4.re * x3
                + self.twiddle2.im * x4
                + self.twiddle0.re * x5
                + self.twiddle0.im * x6
                - self.twiddle2.re * x7
                - self.twiddle4.im * x8
                + self.twiddle3.im * x9
                + self.twiddle1.re * x10
                + x0,
        );
        let y5 = x0 - x2 + x4 - x6 + x8 - x10;
        let y6 = fmla(
            -self.twiddle1.im,
            x1,
            -self.twiddle3.re * x2 + self.twiddle4.re * x3 + self.twiddle2.im * x4
                - self.twiddle0.re * x5
                + self.twiddle0.im * x6
                + self.twiddle2.re * x7
                - self.twiddle4.im * x8
                - self.twiddle3.im * x9
                + self.twiddle1.re * x10
                + x0,
        );
        let y7 = fmla(
            -self.twiddle3.im,
            x1,
            -self.twiddle2.im * x2 + self.twiddle0.re * x3
                - self.twiddle4.im * x4
                - self.twiddle1.im * x5
                + self.twiddle1.re * x6
                - self.twiddle4.re * x7
                - self.twiddle0.im * x8
                + self.twiddle2.re * x9
                - self.twiddle3.re * x10
                + x0,
        );
        let y8 = fmla(
            -self.twiddle4.re,
            x1,
            self.twiddle0.im * x2 + self.twiddle3.im * x3 - self.twiddle1.re * x4
                + self.twiddle2.re * x5
                - self.twiddle2.im * x6
                - self.twiddle1.im * x7
                + self.twiddle3.re * x8
                - self.twiddle0.re * x9
                + self.twiddle4.im * x10
                + x0,
        );
        let y9 = fmla(
            -self.twiddle2.re,
            x1,
            self.twiddle4.im * x2 - self.twiddle1.im * x3 - self.twiddle0.im * x4
                + self.twiddle3.im * x5
                - self.twiddle3.re * x6
                + self.twiddle0.re * x7
                - self.twiddle1.re * x8
                + self.twiddle4.re * x9
                - self.twiddle2.im * x10
                + x0,
        );
        let y10 = fmla(
            -self.twiddle0.re,
            x1,
            self.twiddle1.re * x2 - self.twiddle2.re * x3 + self.twiddle3.re * x4
                - self.twiddle4.re * x5
                + self.twiddle4.im * x6
                - self.twiddle3.im * x7
                + self.twiddle2.im * x8
                - self.twiddle1.im * x9
                + self.twiddle0.im * x10
                + x0,
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

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly11<T>
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
pub(crate) struct Dct3Butterfly12<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
}

impl<T: DctSample> Default for Dct3Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 48).conj(),
            twiddle1: compute_twiddle(2, 48).conj(),
            twiddle2: compute_twiddle(3, 48).conj(),
            twiddle3: compute_twiddle(4, 48).conj(),
            twiddle4: compute_twiddle(5, 48).conj(),
            twiddle5: compute_twiddle(6, 48).conj(),
        }
    }
}

impl<T: DctSample> Dct3Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 12]) {
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

        let tmp2 = self.twiddle2.re * x3;
        let tmp3 = self.twiddle1.re * x10;
        let tmp12 = self.twiddle2.im * x5;
        let tmp28 = self.twiddle5.re * x6;
        let tmp19 = self.twiddle2.re * x5;
        let tmp8 = self.twiddle2.im * x7;
        let tmp16 = self.twiddle4.re * x1;
        let tmp31 = self.twiddle4.im * x7;
        let tmp36 = self.twiddle2.im * x3;
        let tmp24 = self.twiddle2.im * x9;
        let tmp1 = self.twiddle0.im * x5;
        let tmp6 = self.twiddle1.im * x2;
        let tmp11 = self.twiddle3.im * x8;
        let tmp25 = self.twiddle4.re * x5;
        let tmp4 = self.twiddle0.re * x7;
        let tmp20 = self.twiddle2.im * x11;
        let tmp14 = self.twiddle0.im * x7;
        let tmp0 = self.twiddle5.re * x10;
        let tmp18 = self.twiddle5.re * x2;
        let tmp13 = self.twiddle2.re * x7;
        let tmp30 = self.twiddle1.im * x10;
        let tmp35 = self.twiddle1.re * x2;
        let tmp34 = self.twiddle3.re * x4;
        let tmp9 = self.twiddle4.re * x11;
        let tmp7 = self.twiddle4.im * x11;
        let tmp26 = self.twiddle0.re * x11;
        let tmp5 = self.twiddle4.im * x5;
        let tmp21 = self.twiddle0.re * x1;
        let tmp15 = self.twiddle0.im * x11;
        let tmp23 = self.twiddle0.im * x1;
        let tmp17 = self.twiddle2.re * x9;
        let tmp27 = self.twiddle4.re * x7;
        let tmp33 = self.twiddle2.im * x1;
        let tmp32 = self.twiddle4.im * x1;
        let tmp29 = self.twiddle2.re * x11;
        let tmp22 = self.twiddle2.re * x1;
        let tmp10 = self.twiddle0.re * x5;
        let y0 = tmp21
            + tmp35
            + tmp2
            + tmp34
            + tmp25
            + tmp28
            + tmp31
            + tmp11
            + tmp24
            + tmp30
            + tmp15
            + x0;
        let y1 = tmp22 + tmp18 + tmp36 - tmp12 - tmp28 - tmp13 - x8 - tmp17 - tmp0 - tmp20 + x0;
        let y2 =
            tmp16 + tmp6 - tmp36 - tmp34 - tmp10 - tmp28 - tmp14 + tmp11 + tmp17 + tmp3 + tmp7 + x0;
        let y3 =
            tmp32 - tmp6 - tmp2 - tmp34 - tmp1 + tmp28 + tmp4 + tmp11 - tmp24 - tmp3 - tmp9 + x0;
        let y4 = tmp33 - tmp18 - tmp2 + tmp19 + tmp28 - tmp8 - x8 - tmp24 + tmp0 + tmp29 + x0;
        let y5 =
            tmp23 - tmp35 - tmp36 + tmp34 + tmp5 - tmp28 - tmp27 + tmp11 + tmp17 - tmp30 - tmp26
                + x0;
        let y6 = -tmp23 - tmp35 + tmp36 + tmp34 - tmp5 - tmp28 + tmp27 + tmp11 - tmp17 - tmp30
            + tmp26
            + x0;
        let y7 = -tmp33 - tmp18 + tmp2 - tmp19 + tmp28 + tmp8 - x8 + tmp24 + tmp0 - tmp29 + x0;
        let y8 =
            -tmp32 - tmp6 + tmp2 - tmp34 + tmp1 + tmp28 - tmp4 + tmp11 + tmp24 - tmp3 + tmp9 + x0;
        let y9 = -tmp16 + tmp6 + tmp36 - tmp34 + tmp10 - tmp28 + tmp14 + tmp11 - tmp17 + tmp3
            - tmp7
            + x0;
        let y10 = -tmp22 + tmp18 - tmp36 + tmp12 - tmp28 + tmp13 - x8 + tmp17 - tmp0 + tmp20 + x0;
        let y11 = -tmp21 + tmp35 - tmp2 + tmp34 - tmp25 + tmp28 - tmp31 + tmp11 - tmp24 + tmp30
            - tmp15
            + x0;

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
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(12) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(12) {
            self.exec((&mut chunk[..12]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        12
    }
}

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
    pub(crate) fn exec(&self, data: &mut [T; 16]) {
        //process the evens
        let mut dct3_buffer = [
            data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
        ];
        self.bf8.exec(&mut dct3_buffer);

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
        self.bf4.exec(&mut odds_n1);
        self.bf4_dst.exec(&mut odds_n3);

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

impl<T: DctSample> PxdctExecutor<T> for Dct3Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(16) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(16) {
            self.exec((&mut chunk[..16]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_dct3;
    use rand::Rng;

    gen_test_butterfly!(test_bf_dct3_2, f64, Dct3Butterfly2, 2, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_3, f64, Dct3Butterfly3, 3, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_4, f64, Dct3Butterfly4, 4, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_5, f64, Dct3Butterfly5, 5, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_6, f64, Dct3Butterfly6, 6, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_7, f64, Dct3Butterfly7, 7, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_8, f64, Dct3Butterfly8, 8, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_9, f64, Dct3Butterfly9, 9, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_11, f64, Dct3Butterfly11, 11, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_12, f64, Dct3Butterfly12, 12, 1e-7, naive_dct3);
    gen_test_butterfly!(test_bf_dct3_16, f64, Dct3Butterfly16, 16, 1e-7, naive_dct3);
}
