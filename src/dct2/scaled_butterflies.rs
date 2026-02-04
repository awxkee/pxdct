/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
    Dct2Butterfly2, Dct2Butterfly4, Dct2Butterfly8, Dst2Butterfly2, Dst2Butterfly4,
};
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[derive(Debug, Clone, Default)]
pub(crate) struct UpscalingDct2<T> {
    phantom_data: PhantomData<T>,
}

impl<T: DctSample> UpscalingDct2<T> {
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 2]) {
        let sum = data[0] + data[1];
        let v1 = (data[0] - data[1]) * T::HALF;
        let v0 = sum * T::FRAC_1_SQRT_2;
        data[0] = v0;
        data[1] = v1;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ScaledDct2Butterfly4<T: DctSample> {
    twiddle: Complex<T>,
    bf2: UpscalingDct2<T>,
}

impl<T: DctSample> Default for ScaledDct2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 16).conj() * T::FRAC_1_SQRT_2,
            bf2: UpscalingDct2::default(),
        }
    }
}

impl<T: DctSample> ScaledDct2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];
        let u3 = data[3];

        let lower_dct4 = u0 - u3;
        let upper_dct4 = u2 - u1;

        let mut dct_evens = [u0 + u3, u2 + u1];

        self.bf2.exec(&mut dct_evens);

        let v1 = fmla(lower_dct4, self.twiddle.re, -upper_dct4 * self.twiddle.im);
        let v3 = fmla(upper_dct4, self.twiddle.re, lower_dct4 * self.twiddle.im);
        data[0] = dct_evens[0];
        data[1] = v1;
        data[2] = dct_evens[1];
        data[3] = v3;
    }
}

define_in_place_butterfly!(ScaledDct2Butterfly4, 4);

#[derive(Debug, Clone)]
pub(crate) struct ScaledDct2Butterfly8<T: DctSample> {
    bf4: Dct2Butterfly4<T>,
    bf2_dct: Dct2Butterfly2<T>,
    bf2_dst: Dst2Butterfly2<T>,
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
}

impl<T: DctSample> Default for ScaledDct2Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: Dct2Butterfly4::default(),
            twiddle0: compute_twiddle(1, 32).conj() * T::HALF,
            twiddle1: compute_twiddle(3, 32).conj() * T::HALF,
            bf2_dct: Dct2Butterfly2::default(),
            bf2_dst: Dst2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> ScaledDct2Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];
        let u3 = data[3];
        let u4 = data[4];
        let u5 = data[5];
        let u6 = data[6];
        let u7 = data[7];

        let mut dct2_buffer = [
            (u0 + u7) * T::HALF,
            (u1 + u6) * T::HALF,
            (u2 + u5) * T::HALF,
            (u3 + u4) * T::HALF,
        ];
        self.bf4.exec(&mut InPlaceStore::new(&mut dct2_buffer));

        // odds
        let differences = [u0 - u7, u3 - u4, u1 - u6, u2 - u5];

        let mut dct4_even_buffer = [
            fmla(
                differences[0],
                self.twiddle0.re,
                differences[1] * self.twiddle0.im,
            ),
            fmla(
                differences[2],
                self.twiddle1.re,
                differences[3] * self.twiddle1.im,
            ),
        ];

        self.bf2_dct
            .exec(&mut InPlaceStore::new(&mut dct4_even_buffer));
        let mut dct4_odd_buffer = [
            fmla(
                differences[3],
                self.twiddle1.re,
                -differences[2] * self.twiddle1.im,
            ),
            fmla(
                differences[1],
                self.twiddle0.re,
                -differences[0] * self.twiddle0.im,
            ),
        ];
        self.bf2_dst
            .exec(&mut InPlaceStore::new(&mut dct4_odd_buffer));

        // combine the results
        data[0] = dct2_buffer[0];
        data[1] = dct4_even_buffer[0];
        data[2] = dct2_buffer[1];
        data[3] = dct4_even_buffer[1] - dct4_odd_buffer[0];
        data[4] = dct2_buffer[2];
        data[5] = dct4_even_buffer[1] + dct4_odd_buffer[0];
        data[6] = dct2_buffer[3];
        data[7] = dct4_odd_buffer[1];
    }
}

define_in_place_butterfly!(ScaledDct2Butterfly8, 8);

#[derive(Debug, Clone)]
pub(crate) struct ScaledDct2Butterfly16<T: DctSample> {
    bf8: Dct2Butterfly8<T>,
    bf4_dst: Dst2Butterfly4<T>,
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

impl<T: DctSample> Default for ScaledDct2Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf8: Dct2Butterfly8::default(),
            bf4_dst: Dst2Butterfly4::default(),
            twiddle0: compute_twiddle(1, 64).conj() * T::SQRT_2_OVER_16,
            twiddle1: compute_twiddle(3, 64).conj() * T::SQRT_2_OVER_16,
            twiddle2: compute_twiddle(5, 64).conj() * T::SQRT_2_OVER_16,
            twiddle3: compute_twiddle(7, 64).conj() * T::SQRT_2_OVER_16,
        }
    }
}

impl<T: DctSample> ScaledDct2Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, chunk: &mut S) {
        let u0 = chunk[0];
        let u1 = chunk[1];
        let u2 = chunk[2];
        let u3 = chunk[3];
        let u4 = chunk[4];
        let u5 = chunk[5];
        let u6 = chunk[6];
        let u7 = chunk[7];
        let u8 = chunk[8];
        let u9 = chunk[9];
        let u10 = chunk[10];
        let u11 = chunk[11];
        let u12 = chunk[12];
        let u13 = chunk[13];
        let u14 = chunk[14];
        let u15 = chunk[15];

        //process the evens
        let mut dct2_buffer = [
            (u0 + u15) * T::SQRT_2_OVER_16,
            (u1 + u14) * T::SQRT_2_OVER_16,
            (u2 + u13) * T::SQRT_2_OVER_16,
            (u3 + u12) * T::SQRT_2_OVER_16,
            (u4 + u11) * T::SQRT_2_OVER_16,
            (u5 + u10) * T::SQRT_2_OVER_16,
            (u6 + u9) * T::SQRT_2_OVER_16,
            (u7 + u8) * T::SQRT_2_OVER_16,
        ];
        self.bf8.exec(&mut InPlaceStore::new(&mut dct2_buffer));

        //process the odds
        let differences = [
            u0 - u15,
            u7 - u8,
            u1 - u14,
            u6 - u9,
            u2 - u13,
            u5 - u10,
            u3 - u12,
            u4 - u11,
        ];

        let mut dct4_even_buffer = [
            fmla(
                differences[0],
                self.twiddle0.re,
                differences[1] * self.twiddle0.im,
            ),
            fmla(
                differences[2],
                self.twiddle1.re,
                differences[3] * self.twiddle1.im,
            ),
            fmla(
                differences[4],
                self.twiddle2.re,
                differences[5] * self.twiddle2.im,
            ),
            fmla(
                differences[6],
                self.twiddle3.re,
                differences[7] * self.twiddle3.im,
            ),
        ];
        let mut dct4_odd_buffer = [
            fmla(
                differences[7],
                self.twiddle3.re,
                -differences[6] * self.twiddle3.im,
            ),
            fmla(
                differences[5],
                self.twiddle2.re,
                -differences[4] * self.twiddle2.im,
            ),
            fmla(
                differences[3],
                self.twiddle1.re,
                -differences[2] * self.twiddle1.im,
            ),
            fmla(
                differences[1],
                self.twiddle0.re,
                -differences[0] * self.twiddle0.im,
            ),
        ];

        self.bf8
            .bf4
            .exec(&mut InPlaceStore::new(&mut dct4_even_buffer));
        self.bf4_dst
            .exec(&mut InPlaceStore::new(&mut dct4_odd_buffer));

        // combine the results
        chunk[0] = dct2_buffer[0];
        chunk[1] = dct4_even_buffer[0];
        chunk[2] = dct2_buffer[1];
        chunk[3] = dct4_even_buffer[1] - dct4_odd_buffer[0];
        chunk[4] = dct2_buffer[2];
        chunk[5] = dct4_even_buffer[1] + dct4_odd_buffer[0];
        chunk[6] = dct2_buffer[3];
        chunk[7] = dct4_even_buffer[2] + dct4_odd_buffer[1];
        chunk[8] = dct2_buffer[4];
        chunk[9] = dct4_even_buffer[2] - dct4_odd_buffer[1];
        chunk[10] = dct2_buffer[5];
        chunk[11] = dct4_even_buffer[3] - dct4_odd_buffer[2];
        chunk[12] = dct2_buffer[6];
        chunk[13] = dct4_even_buffer[3] + dct4_odd_buffer[2];
        chunk[14] = dct2_buffer[7];
        chunk[15] = dct4_odd_buffer[3];
    }
}

define_in_place_butterfly!(ScaledDct2Butterfly16, 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_scaled_dct2;
    use rand::Rng;

    gen_test_butterfly!(
        test_bf_scaled_dct4,
        f64,
        ScaledDct2Butterfly4,
        4,
        1e-7,
        naive_scaled_dct2
    );
    gen_test_butterfly!(
        test_bf_scaled_dct8,
        f64,
        ScaledDct2Butterfly8,
        8,
        1e-7,
        naive_scaled_dct2
    );
    gen_test_butterfly!(
        test_bf_scaled_dct16,
        f64,
        ScaledDct2Butterfly16,
        16,
        1e-7,
        naive_scaled_dct2
    );
}
