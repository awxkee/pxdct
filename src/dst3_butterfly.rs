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
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::dct3::{Dct3Butterfly2, Dct3Butterfly4, Dct3Butterfly8};
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[derive(Debug, Clone, Default)]
pub(crate) struct Dst3Butterfly2<T> {
    phantom: PhantomData<T>,
}

impl<T: DctSample> Dst3Butterfly2<T> {
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
            let half_1 = data[1] * T::HALF;
            let q0 = data[0];
            data[0] = fmla(q0, T::FRAC_1_SQRT_2, half_1);
            data[1] = fmla(q0, T::FRAC_1_SQRT_2, -half_1);
        }
        #[cfg(not(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            target_arch = "aarch64"
        )))]
        {
            let frac_0 = data[0] * T::FRAC_1_SQRT_2;
            let half_1 = data[1] * T::HALF;

            data[0] = frac_0 + half_1;
            data[1] = frac_0 - half_1;
        }
    }
}

define_in_place_butterfly!(Dst3Butterfly2, 2);

#[derive(Debug, Clone)]
pub(crate) struct Dst3Butterfly3<T> {
    twiddle: T,
}

impl<T: DctSample> Default for Dst3Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 12).re,
        }
    }
}

impl<T: DctSample> Dst3Butterfly3<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let buffer0_half = data[2] * T::HALF;
        let buffer1 = data[1];
        let buffer2 = data[0];
        let buffer2_half = buffer2 * T::HALF;

        let half2_3 = buffer2_half + buffer0_half;

        data[0] = fmla(buffer1, self.twiddle, half2_3);
        data[1] = buffer2 - buffer0_half;
        data[2] = fmla(buffer1, -self.twiddle, half2_3);
    }
}

define_in_place_butterfly!(Dst3Butterfly3, 3);

#[derive(Debug, Clone)]
pub(crate) struct Dst3Butterfly4<T> {
    twiddle: Complex<T>,
    bf2: Dct3Butterfly2<T>,
}

impl<T: DctSample> Default for Dst3Butterfly4<T>
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

impl<T: DctSample> Dst3Butterfly4<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // DST-3 split radix with n = 4

        let mut odds = [data[3], data[1]];
        self.bf2.exec(&mut InPlaceStore::new(&mut odds));

        let lower_dct4 = fmla(data[2], self.twiddle.re, data[0] * self.twiddle.im);
        let upper_dct4 = fmla(data[2], self.twiddle.im, -data[0] * self.twiddle.re);

        // Merge our results
        data[0] = odds[0] + lower_dct4;
        data[2] = odds[1] - upper_dct4;
        data[1] = -(odds[1] + upper_dct4);
        data[3] = lower_dct4 - odds[0];
    }
}

define_in_place_butterfly!(Dst3Butterfly4, 4);

#[derive(Debug, Clone)]
pub(crate) struct Dst3Butterfly5<T> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
}

impl<T: DctSample> Default for Dst3Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 20).conj(),
            twiddle1: compute_twiddle(2, 20).conj(),
        }
    }
}

impl<T: DctSample> Dst3Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let x0 = data[0];
        let x1 = data[1];
        let x2 = data[2];
        let x3 = data[3];
        let x4 = data[4] * T::HALF;

        let y0 = fmla(
            self.twiddle0.im,
            x0,
            fmla(
                self.twiddle1.im,
                x1,
                fmla(self.twiddle1.re, x2, fmla(self.twiddle0.re, x3, x4)),
            ),
        );

        let y1 = fmla(
            self.twiddle1.re,
            x0,
            fmla(
                self.twiddle0.re,
                x1,
                fmla(self.twiddle0.im, x2, fmla(-self.twiddle1.im, x3, -x4)),
            ),
        );

        let y2 = x0 - x2 + x4;

        let y3 = fmla(
            self.twiddle1.re,
            x0,
            fmla(
                -self.twiddle0.re,
                x1,
                fmla(self.twiddle0.im, x2, fmla(self.twiddle1.im, x3, -x4)),
            ),
        );

        let y4 = fmla(
            self.twiddle0.im,
            x0,
            fmla(
                -self.twiddle1.im,
                x1,
                fmla(self.twiddle1.re, x2, fmla(-self.twiddle0.re, x3, x4)),
            ),
        );

        data[0] = y0;
        data[1] = y1;
        data[2] = y2;
        data[3] = y3;
        data[4] = y4;
    }
}

define_in_place_butterfly!(Dst3Butterfly5, 5);

#[derive(Debug, Clone)]
pub(crate) struct Dst3Butterfly8<T> {
    twiddles: [Complex<T>; 2],
    bf2: Dct3Butterfly2<T>,
    bf2_dst: Dst3Butterfly2<T>,
    bf4: Dct3Butterfly4<T>,
}

impl<T: DctSample> Default for Dst3Butterfly8<T>
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

impl<T: DctSample> Dst3Butterfly8<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // Derived from process_inplace_dct3, reversing the inputs and negating the odd outputs

        //process the evens
        let mut dct3_buffer = [data[7], data[5], data[3], data[1]];
        self.bf4.exec(&mut InPlaceStore::new(&mut dct3_buffer));

        //process the odds
        let mut odds_n1 = [data[6] * T::TWO, data[4] + data[2]];
        let mut odds_n3 = [data[4] - data[2], data[0] * T::TWO];
        self.bf2.exec(&mut InPlaceStore::new(&mut odds_n1));
        self.bf2_dst.exec(&mut InPlaceStore::new(&mut odds_n3));

        let merged_odds = [
            fmla(
                odds_n1[0],
                self.twiddles[0].re,
                odds_n3[0] * self.twiddles[0].im,
            ),
            fmla(
                odds_n1[0],
                self.twiddles[0].im,
                -odds_n3[0] * self.twiddles[0].re,
            ),
            fmla(
                odds_n1[1],
                self.twiddles[1].re,
                odds_n3[1] * self.twiddles[1].im,
            ),
            fmla(
                odds_n1[1],
                self.twiddles[1].im,
                -odds_n3[1] * self.twiddles[1].re,
            ),
        ];

        data[0] = dct3_buffer[0] + merged_odds[0];
        data[7] = merged_odds[0] - dct3_buffer[0];

        data[3] = -(dct3_buffer[3] + merged_odds[1]);
        data[4] = dct3_buffer[3] - merged_odds[1];

        data[1] = -(dct3_buffer[1] + merged_odds[2]);
        data[6] = dct3_buffer[1] - merged_odds[2];

        data[2] = dct3_buffer[2] + merged_odds[3];
        data[5] = merged_odds[3] - dct3_buffer[2];
    }
}

define_in_place_butterfly!(Dst3Butterfly8, 8);

#[derive(Debug, Clone)]
pub(crate) struct Dst3Butterfly16<T> {
    twiddles: [Complex<T>; 4],
    bf4: Dct3Butterfly4<T>,
    bf4_dst: Dst3Butterfly4<T>,
    bf8: Dct3Butterfly8<T>,
}

impl<T: DctSample> Default for Dst3Butterfly16<T>
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

impl<T: DctSample> Dst3Butterfly16<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut dct3_buffer = [
            data[15], data[13], data[11], data[9], data[7], data[5], data[3], data[1],
        ];
        self.bf8.exec(&mut InPlaceStore::new(&mut dct3_buffer));

        //process the odds
        let mut evens_n1 = [
            data[14] * T::TWO,
            data[12] + data[10],
            data[8] + data[6],
            data[4] + data[2],
        ];
        let mut evens_n3 = [
            data[12] - data[10],
            data[8] - data[6],
            data[4] - data[2],
            data[0] * T::TWO,
        ];
        self.bf4.exec(&mut InPlaceStore::new(&mut evens_n1));
        self.bf4_dst.exec(&mut InPlaceStore::new(&mut evens_n3));

        let merged_odds = [
            fmla(
                evens_n1[0],
                self.twiddles[0].re,
                evens_n3[0] * self.twiddles[0].im,
            ),
            fmla(
                evens_n1[0],
                self.twiddles[0].im,
                -evens_n3[0] * self.twiddles[0].re,
            ),
            fmla(
                evens_n1[1],
                self.twiddles[1].re,
                evens_n3[1] * self.twiddles[1].im,
            ),
            fmla(
                evens_n1[1],
                self.twiddles[1].im,
                -evens_n3[1] * self.twiddles[1].re,
            ),
            fmla(
                evens_n1[2],
                self.twiddles[2].re,
                evens_n3[2] * self.twiddles[2].im,
            ),
            fmla(
                evens_n1[2],
                self.twiddles[2].im,
                -evens_n3[2] * self.twiddles[2].re,
            ),
            fmla(
                evens_n1[3],
                self.twiddles[3].re,
                evens_n3[3] * self.twiddles[3].im,
            ),
            fmla(
                evens_n1[3],
                self.twiddles[3].im,
                -evens_n3[3] * self.twiddles[3].re,
            ),
        ];

        // merge the temp buffers into the final output
        data[0] = dct3_buffer[0] + merged_odds[0];
        data[15] = merged_odds[0] - dct3_buffer[0];

        data[7] = -(dct3_buffer[7] + merged_odds[1]);
        data[8] = dct3_buffer[7] - merged_odds[1];

        data[1] = -(dct3_buffer[1] + merged_odds[2]);
        data[14] = dct3_buffer[1] - merged_odds[2];

        data[6] = dct3_buffer[6] + merged_odds[3];
        data[9] = merged_odds[3] - dct3_buffer[6];

        data[2] = dct3_buffer[2] + merged_odds[4];
        data[13] = merged_odds[4] - dct3_buffer[2];

        data[5] = -(dct3_buffer[5] + merged_odds[5]);
        data[10] = dct3_buffer[5] - merged_odds[5];

        data[3] = -(dct3_buffer[3] + merged_odds[6]);
        data[12] = dct3_buffer[3] - merged_odds[6];

        data[4] = dct3_buffer[4] + merged_odds[7];
        data[11] = merged_odds[7] - dct3_buffer[4];
    }
}

define_in_place_butterfly!(Dst3Butterfly16, 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::gen_test_butterfly;

    gen_test_butterfly!(test_bf_dst3_2, f64, Dst3Butterfly2, 2, 1e-7, naive_dst3);
    gen_test_butterfly!(test_bf_dst3_3, f64, Dst3Butterfly3, 3, 1e-7, naive_dst3);
    gen_test_butterfly!(test_bf_dst3_4, f64, Dst3Butterfly4, 4, 1e-7, naive_dst3);
    gen_test_butterfly!(test_bf_dst3_5, f64, Dst3Butterfly5, 5, 1e-7, naive_dst3);
    gen_test_butterfly!(test_bf_dst3_8, f64, Dst3Butterfly8, 8, 1e-7, naive_dst3);
    gen_test_butterfly!(test_bf_dst3_16, f64, Dst3Butterfly16, 16, 1e-7, naive_dst3);
}
