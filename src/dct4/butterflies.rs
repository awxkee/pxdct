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
use crate::butterflies::{
    Dct2Butterfly6, Dct2Butterfly9, Dct2Butterfly10, Dct2Butterfly12, Dct2Butterfly14,
    Dct2Butterfly15,
};
use crate::dct2::prime_butterflies::{
    Dct2Butterfly3, Dct2Butterfly5, Dct2Butterfly7, Dct2Butterfly11, Dct2Butterfly13,
    MixedRadix11Sample,
};
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly6<T: DctSample> {
    twiddles: [Complex<T>; 3],
    bf3: Dct2Butterfly3<T>,
}
impl<T: DctSample> Default for Dct4Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 6 * 8).conj()),
            bf3: Dct2Butterfly3::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 6 points
        let mut left = [T::zero(); 3];
        let mut right = [T::zero(); 3];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[5]);
        right[2] = fmla(-self.twiddles[0].re, data[5], self.twiddles[0].im * data[0]);
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[4]);
        right[1] = fmla(self.twiddles[1].re, data[4], -self.twiddles[1].im * data[1]);
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[3]);
        right[0] = fmla(-self.twiddles[2].re, data[3], self.twiddles[2].im * data[2]);
        self.bf3.exec(&mut InPlaceStore::new(&mut left));
        self.bf3.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[1] = left[1] + right[2];
        data[2] = left[1] - right[2];
        data[3] = left[2] - right[1];
        data[4] = left[2] + right[1];
        data[5] = right[0];
    }
}

define_in_place_butterfly!(Dct4Butterfly6, 6);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly10<T: DctSample> {
    twiddles: [Complex<T>; 5],
    bf5: Dct2Butterfly5<T>,
}
impl<T: DctSample> Default for Dct4Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 10 * 8).conj()),
            bf5: Dct2Butterfly5::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 10 points
        let mut left = [T::zero(); 5];
        let mut right = [T::zero(); 5];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[9]);
        right[4] = fmla(-self.twiddles[0].re, data[9], self.twiddles[0].im * data[0]);
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[8]);
        right[3] = fmla(self.twiddles[1].re, data[8], -self.twiddles[1].im * data[1]);
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[7]);
        right[2] = fmla(-self.twiddles[2].re, data[7], self.twiddles[2].im * data[2]);
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[6]);
        right[1] = fmla(self.twiddles[3].re, data[6], -self.twiddles[3].im * data[3]);
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[5]);
        right[0] = fmla(-self.twiddles[4].re, data[5], self.twiddles[4].im * data[4]);
        self.bf5.exec(&mut InPlaceStore::new(&mut left));
        self.bf5.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[9] = right[0];
        data[1] = left[1] + right[4];
        data[2] = left[1] - right[4];
        data[3] = left[2] - right[3];
        data[4] = left[2] + right[3];
        data[5] = left[3] + right[2];
        data[6] = left[3] - right[2];
        data[7] = left[4] - right[1];
        data[8] = left[4] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly10, 10);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly12<T: DctSample> {
    twiddles: [Complex<T>; 6],
    bf6: Dct2Butterfly6<T>,
}
impl<T: DctSample> Default for Dct4Butterfly12<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 12 * 8).conj()),
            bf6: Dct2Butterfly6::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 12 points
        let mut left = [T::zero(); 6];
        let mut right = [T::zero(); 6];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[11]);
        right[5] = fmla(
            -self.twiddles[0].re,
            data[11],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[10]);
        right[4] = fmla(
            self.twiddles[1].re,
            data[10],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[9]);
        right[3] = fmla(-self.twiddles[2].re, data[9], self.twiddles[2].im * data[2]);
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[8]);
        right[2] = fmla(self.twiddles[3].re, data[8], -self.twiddles[3].im * data[3]);
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[7]);
        right[1] = fmla(-self.twiddles[4].re, data[7], self.twiddles[4].im * data[4]);
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[6]);
        right[0] = fmla(self.twiddles[5].re, data[6], -self.twiddles[5].im * data[5]);
        self.bf6.exec(&mut InPlaceStore::new(&mut left));
        self.bf6.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[11] = right[0];
        data[1] = left[1] - right[5];
        data[2] = left[1] + right[5];
        data[3] = left[2] + right[4];
        data[4] = left[2] - right[4];
        data[5] = left[3] - right[3];
        data[6] = left[3] + right[3];
        data[7] = left[4] + right[2];
        data[8] = left[4] - right[2];
        data[9] = left[5] - right[1];
        data[10] = left[5] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly12, 12);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly14<T: DctSample> {
    twiddles: [Complex<T>; 7],
    bf7: Dct2Butterfly7<T>,
}
impl<T: DctSample> Default for Dct4Butterfly14<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 14 * 8).conj()),
            bf7: Dct2Butterfly7::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly14<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 14 points
        let mut left = [T::zero(); 7];
        let mut right = [T::zero(); 7];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[13]);
        right[6] = fmla(
            -self.twiddles[0].re,
            data[13],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[12]);
        right[5] = fmla(
            self.twiddles[1].re,
            data[12],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[11]);
        right[4] = fmla(
            -self.twiddles[2].re,
            data[11],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[10]);
        right[3] = fmla(
            self.twiddles[3].re,
            data[10],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[9]);
        right[2] = fmla(-self.twiddles[4].re, data[9], self.twiddles[4].im * data[4]);
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[8]);
        right[1] = fmla(self.twiddles[5].re, data[8], -self.twiddles[5].im * data[5]);
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[7]);
        right[0] = fmla(-self.twiddles[6].re, data[7], self.twiddles[6].im * data[6]);
        self.bf7.exec(&mut InPlaceStore::new(&mut left));
        self.bf7.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[13] = right[0];
        data[1] = left[1] + right[6];
        data[2] = left[1] - right[6];
        data[3] = left[2] - right[5];
        data[4] = left[2] + right[5];
        data[5] = left[3] + right[4];
        data[6] = left[3] - right[4];
        data[7] = left[4] - right[3];
        data[8] = left[4] + right[3];
        data[9] = left[5] + right[2];
        data[10] = left[5] - right[2];
        data[11] = left[6] - right[1];
        data[12] = left[6] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly14, 14);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly18<T: DctSample> {
    twiddles: [Complex<T>; 9],
    bf9: Dct2Butterfly9<T>,
}
impl<T: DctSample> Default for Dct4Butterfly18<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 18 * 8).conj()),
            bf9: Dct2Butterfly9::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly18<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 18 points
        let mut left = [T::zero(); 9];
        let mut right = [T::zero(); 9];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[17]);
        right[8] = fmla(
            -self.twiddles[0].re,
            data[17],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[16]);
        right[7] = fmla(
            self.twiddles[1].re,
            data[16],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[15]);
        right[6] = fmla(
            -self.twiddles[2].re,
            data[15],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[14]);
        right[5] = fmla(
            self.twiddles[3].re,
            data[14],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[13]);
        right[4] = fmla(
            -self.twiddles[4].re,
            data[13],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[12]);
        right[3] = fmla(
            self.twiddles[5].re,
            data[12],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[11]);
        right[2] = fmla(
            -self.twiddles[6].re,
            data[11],
            self.twiddles[6].im * data[6],
        );
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[10]);
        right[1] = fmla(
            self.twiddles[7].re,
            data[10],
            -self.twiddles[7].im * data[7],
        );
        left[8] = fmla(self.twiddles[8].re, data[8], self.twiddles[8].im * data[9]);
        right[0] = fmla(-self.twiddles[8].re, data[9], self.twiddles[8].im * data[8]);
        self.bf9.exec(&mut InPlaceStore::new(&mut left));
        self.bf9.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[17] = right[0];
        data[1] = left[1] + right[8];
        data[2] = left[1] - right[8];
        data[3] = left[2] - right[7];
        data[4] = left[2] + right[7];
        data[5] = left[3] + right[6];
        data[6] = left[3] - right[6];
        data[7] = left[4] - right[5];
        data[8] = left[4] + right[5];
        data[9] = left[5] + right[4];
        data[10] = left[5] - right[4];
        data[11] = left[6] - right[3];
        data[12] = left[6] + right[3];
        data[13] = left[7] + right[2];
        data[14] = left[7] - right[2];
        data[15] = left[8] - right[1];
        data[16] = left[8] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly18, 18);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly20<T: DctSample> {
    twiddles: [Complex<T>; 10],
    bf10: Dct2Butterfly10<T>,
}
impl<T: DctSample> Default for Dct4Butterfly20<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 20 * 8).conj()),
            bf10: Dct2Butterfly10::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly20<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 20 points
        let mut left = [T::zero(); 10];
        let mut right = [T::zero(); 10];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[19]);
        right[9] = fmla(
            -self.twiddles[0].re,
            data[19],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[18]);
        right[8] = fmla(
            self.twiddles[1].re,
            data[18],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[17]);
        right[7] = fmla(
            -self.twiddles[2].re,
            data[17],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[16]);
        right[6] = fmla(
            self.twiddles[3].re,
            data[16],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[15]);
        right[5] = fmla(
            -self.twiddles[4].re,
            data[15],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[14]);
        right[4] = fmla(
            self.twiddles[5].re,
            data[14],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[13]);
        right[3] = fmla(
            -self.twiddles[6].re,
            data[13],
            self.twiddles[6].im * data[6],
        );
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[12]);
        right[2] = fmla(
            self.twiddles[7].re,
            data[12],
            -self.twiddles[7].im * data[7],
        );
        left[8] = fmla(self.twiddles[8].re, data[8], self.twiddles[8].im * data[11]);
        right[1] = fmla(
            -self.twiddles[8].re,
            data[11],
            self.twiddles[8].im * data[8],
        );
        left[9] = fmla(self.twiddles[9].re, data[9], self.twiddles[9].im * data[10]);
        right[0] = fmla(
            self.twiddles[9].re,
            data[10],
            -self.twiddles[9].im * data[9],
        );
        self.bf10.exec(&mut InPlaceStore::new(&mut left));
        self.bf10.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[19] = right[0];
        data[1] = left[1] - right[9];
        data[2] = left[1] + right[9];
        data[3] = left[2] + right[8];
        data[4] = left[2] - right[8];
        data[5] = left[3] - right[7];
        data[6] = left[3] + right[7];
        data[7] = left[4] + right[6];
        data[8] = left[4] - right[6];
        data[9] = left[5] - right[5];
        data[10] = left[5] + right[5];
        data[11] = left[6] + right[4];
        data[12] = left[6] - right[4];
        data[13] = left[7] - right[3];
        data[14] = left[7] + right[3];
        data[15] = left[8] + right[2];
        data[16] = left[8] - right[2];
        data[17] = left[9] - right[1];
        data[18] = left[9] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly20, 20);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly22<T: DctSample> {
    twiddles: [Complex<T>; 11],
    bf11: Dct2Butterfly11<T>,
}
impl<T: DctSample> Default for Dct4Butterfly22<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 22 * 8).conj()),
            bf11: Dct2Butterfly11::default(),
        }
    }
}

impl<T: DctSample + MixedRadix11Sample> Dct4Butterfly22<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 22 points
        let mut left = [T::zero(); 11];
        let mut right = [T::zero(); 11];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[21]);
        right[10] = fmla(
            -self.twiddles[0].re,
            data[21],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[20]);
        right[9] = fmla(
            self.twiddles[1].re,
            data[20],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[19]);
        right[8] = fmla(
            -self.twiddles[2].re,
            data[19],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[18]);
        right[7] = fmla(
            self.twiddles[3].re,
            data[18],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[17]);
        right[6] = fmla(
            -self.twiddles[4].re,
            data[17],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[16]);
        right[5] = fmla(
            self.twiddles[5].re,
            data[16],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[15]);
        right[4] = fmla(
            -self.twiddles[6].re,
            data[15],
            self.twiddles[6].im * data[6],
        );
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[14]);
        right[3] = fmla(
            self.twiddles[7].re,
            data[14],
            -self.twiddles[7].im * data[7],
        );
        left[8] = fmla(self.twiddles[8].re, data[8], self.twiddles[8].im * data[13]);
        right[2] = fmla(
            -self.twiddles[8].re,
            data[13],
            self.twiddles[8].im * data[8],
        );
        left[9] = fmla(self.twiddles[9].re, data[9], self.twiddles[9].im * data[12]);
        right[1] = fmla(
            self.twiddles[9].re,
            data[12],
            -self.twiddles[9].im * data[9],
        );
        left[10] = fmla(
            self.twiddles[10].re,
            data[10],
            self.twiddles[10].im * data[11],
        );
        right[0] = fmla(
            -self.twiddles[10].re,
            data[11],
            self.twiddles[10].im * data[10],
        );
        self.bf11.exec(&mut InPlaceStore::new(&mut left));
        self.bf11.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[21] = right[0];
        data[1] = left[1] + right[10];
        data[2] = left[1] - right[10];
        data[3] = left[2] - right[9];
        data[4] = left[2] + right[9];
        data[5] = left[3] + right[8];
        data[6] = left[3] - right[8];
        data[7] = left[4] - right[7];
        data[8] = left[4] + right[7];
        data[9] = left[5] + right[6];
        data[10] = left[5] - right[6];
        data[11] = left[6] - right[5];
        data[12] = left[6] + right[5];
        data[13] = left[7] + right[4];
        data[14] = left[7] - right[4];
        data[15] = left[8] - right[3];
        data[16] = left[8] + right[3];
        data[17] = left[9] + right[2];
        data[18] = left[9] - right[2];
        data[19] = left[10] - right[1];
        data[20] = left[10] + right[1];
    }
}

impl<T: DctSample + MixedRadix11Sample> PxdctExecutor<T> for Dct4Butterfly22<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(22) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(22) {
            self.exec(&mut InPlaceStore::new(chunk));
        }
        Ok(())
    }

    fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
        self.execute(data)
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        self.execute_into_with_scratch(input, output, &mut [])
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        _: &mut [T],
    ) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 22);
        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(22).zip(output.chunks_exact_mut(22)) {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }

    fn length(&self) -> usize {
        22
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly24<T: DctSample> {
    twiddles: [Complex<T>; 12],
    bf12: Dct2Butterfly12<T>,
}
impl<T: DctSample> Default for Dct4Butterfly24<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 24 * 8).conj()),
            bf12: Dct2Butterfly12::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly24<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 24 points
        let mut left = [T::zero(); 12];
        let mut right = [T::zero(); 12];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[23]);
        right[11] = fmla(
            -self.twiddles[0].re,
            data[23],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[22]);
        right[10] = fmla(
            self.twiddles[1].re,
            data[22],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[21]);
        right[9] = fmla(
            -self.twiddles[2].re,
            data[21],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[20]);
        right[8] = fmla(
            self.twiddles[3].re,
            data[20],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[19]);
        right[7] = fmla(
            -self.twiddles[4].re,
            data[19],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[18]);
        right[6] = fmla(
            self.twiddles[5].re,
            data[18],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[17]);
        right[5] = fmla(
            -self.twiddles[6].re,
            data[17],
            self.twiddles[6].im * data[6],
        );
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[16]);
        right[4] = fmla(
            self.twiddles[7].re,
            data[16],
            -self.twiddles[7].im * data[7],
        );
        left[8] = fmla(self.twiddles[8].re, data[8], self.twiddles[8].im * data[15]);
        right[3] = fmla(
            -self.twiddles[8].re,
            data[15],
            self.twiddles[8].im * data[8],
        );
        left[9] = fmla(self.twiddles[9].re, data[9], self.twiddles[9].im * data[14]);
        right[2] = fmla(
            self.twiddles[9].re,
            data[14],
            -self.twiddles[9].im * data[9],
        );
        left[10] = fmla(
            self.twiddles[10].re,
            data[10],
            self.twiddles[10].im * data[13],
        );
        right[1] = fmla(
            -self.twiddles[10].re,
            data[13],
            self.twiddles[10].im * data[10],
        );
        left[11] = fmla(
            self.twiddles[11].re,
            data[11],
            self.twiddles[11].im * data[12],
        );
        right[0] = fmla(
            self.twiddles[11].re,
            data[12],
            -self.twiddles[11].im * data[11],
        );
        self.bf12.exec(&mut InPlaceStore::new(&mut left));
        self.bf12.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[23] = right[0];
        data[1] = left[1] - right[11];
        data[2] = left[1] + right[11];
        data[3] = left[2] + right[10];
        data[4] = left[2] - right[10];
        data[5] = left[3] - right[9];
        data[6] = left[3] + right[9];
        data[7] = left[4] + right[8];
        data[8] = left[4] - right[8];
        data[9] = left[5] - right[7];
        data[10] = left[5] + right[7];
        data[11] = left[6] + right[6];
        data[12] = left[6] - right[6];
        data[13] = left[7] - right[5];
        data[14] = left[7] + right[5];
        data[15] = left[8] + right[4];
        data[16] = left[8] - right[4];
        data[17] = left[9] - right[3];
        data[18] = left[9] + right[3];
        data[19] = left[10] + right[2];
        data[20] = left[10] - right[2];
        data[21] = left[11] - right[1];
        data[22] = left[11] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly24, 24);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly26<T: DctSample> {
    twiddles: [Complex<T>; 13],
    bf13: Dct2Butterfly13<T>,
}
impl<T: DctSample> Default for Dct4Butterfly26<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 26 * 8).conj()),
            bf13: Dct2Butterfly13::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly26<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 26 points
        let mut left = [T::zero(); 13];
        let mut right = [T::zero(); 13];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[25]);
        right[12] = fmla(
            -self.twiddles[0].re,
            data[25],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[24]);
        right[11] = fmla(
            self.twiddles[1].re,
            data[24],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[23]);
        right[10] = fmla(
            -self.twiddles[2].re,
            data[23],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[22]);
        right[9] = fmla(
            self.twiddles[3].re,
            data[22],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[21]);
        right[8] = fmla(
            -self.twiddles[4].re,
            data[21],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[20]);
        right[7] = fmla(
            self.twiddles[5].re,
            data[20],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[19]);
        right[6] = fmla(
            -self.twiddles[6].re,
            data[19],
            self.twiddles[6].im * data[6],
        );
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[18]);
        right[5] = fmla(
            self.twiddles[7].re,
            data[18],
            -self.twiddles[7].im * data[7],
        );
        left[8] = fmla(self.twiddles[8].re, data[8], self.twiddles[8].im * data[17]);
        right[4] = fmla(
            -self.twiddles[8].re,
            data[17],
            self.twiddles[8].im * data[8],
        );
        left[9] = fmla(self.twiddles[9].re, data[9], self.twiddles[9].im * data[16]);
        right[3] = fmla(
            self.twiddles[9].re,
            data[16],
            -self.twiddles[9].im * data[9],
        );
        left[10] = fmla(
            self.twiddles[10].re,
            data[10],
            self.twiddles[10].im * data[15],
        );
        right[2] = fmla(
            -self.twiddles[10].re,
            data[15],
            self.twiddles[10].im * data[10],
        );
        left[11] = fmla(
            self.twiddles[11].re,
            data[11],
            self.twiddles[11].im * data[14],
        );
        right[1] = fmla(
            self.twiddles[11].re,
            data[14],
            -self.twiddles[11].im * data[11],
        );
        left[12] = fmla(
            self.twiddles[12].re,
            data[12],
            self.twiddles[12].im * data[13],
        );
        right[0] = fmla(
            -self.twiddles[12].re,
            data[13],
            self.twiddles[12].im * data[12],
        );
        self.bf13.exec(&mut InPlaceStore::new(&mut left));
        self.bf13.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[25] = right[0];
        data[1] = left[1] + right[12];
        data[2] = left[1] - right[12];
        data[3] = left[2] - right[11];
        data[4] = left[2] + right[11];
        data[5] = left[3] + right[10];
        data[6] = left[3] - right[10];
        data[7] = left[4] - right[9];
        data[8] = left[4] + right[9];
        data[9] = left[5] + right[8];
        data[10] = left[5] - right[8];
        data[11] = left[6] - right[7];
        data[12] = left[6] + right[7];
        data[13] = left[7] + right[6];
        data[14] = left[7] - right[6];
        data[15] = left[8] - right[5];
        data[16] = left[8] + right[5];
        data[17] = left[9] + right[4];
        data[18] = left[9] - right[4];
        data[19] = left[10] - right[3];
        data[20] = left[10] + right[3];
        data[21] = left[11] + right[2];
        data[22] = left[11] - right[2];
        data[23] = left[12] - right[1];
        data[24] = left[12] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly26, 26);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly28<T: DctSample> {
    twiddles: [Complex<T>; 14],
    bf14: Dct2Butterfly14<T>,
}
impl<T: DctSample> Default for Dct4Butterfly28<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 28 * 8).conj()),
            bf14: Dct2Butterfly14::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly28<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 28 points
        let mut left = [T::zero(); 14];
        let mut right = [T::zero(); 14];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[27]);
        right[13] = fmla(
            -self.twiddles[0].re,
            data[27],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[26]);
        right[12] = fmla(
            self.twiddles[1].re,
            data[26],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[25]);
        right[11] = fmla(
            -self.twiddles[2].re,
            data[25],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[24]);
        right[10] = fmla(
            self.twiddles[3].re,
            data[24],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[23]);
        right[9] = fmla(
            -self.twiddles[4].re,
            data[23],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[22]);
        right[8] = fmla(
            self.twiddles[5].re,
            data[22],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[21]);
        right[7] = fmla(
            -self.twiddles[6].re,
            data[21],
            self.twiddles[6].im * data[6],
        );
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[20]);
        right[6] = fmla(
            self.twiddles[7].re,
            data[20],
            -self.twiddles[7].im * data[7],
        );
        left[8] = fmla(self.twiddles[8].re, data[8], self.twiddles[8].im * data[19]);
        right[5] = fmla(
            -self.twiddles[8].re,
            data[19],
            self.twiddles[8].im * data[8],
        );
        left[9] = fmla(self.twiddles[9].re, data[9], self.twiddles[9].im * data[18]);
        right[4] = fmla(
            self.twiddles[9].re,
            data[18],
            -self.twiddles[9].im * data[9],
        );
        left[10] = fmla(
            self.twiddles[10].re,
            data[10],
            self.twiddles[10].im * data[17],
        );
        right[3] = fmla(
            -self.twiddles[10].re,
            data[17],
            self.twiddles[10].im * data[10],
        );
        left[11] = fmla(
            self.twiddles[11].re,
            data[11],
            self.twiddles[11].im * data[16],
        );
        right[2] = fmla(
            self.twiddles[11].re,
            data[16],
            -self.twiddles[11].im * data[11],
        );
        left[12] = fmla(
            self.twiddles[12].re,
            data[12],
            self.twiddles[12].im * data[15],
        );
        right[1] = fmla(
            -self.twiddles[12].re,
            data[15],
            self.twiddles[12].im * data[12],
        );
        left[13] = fmla(
            self.twiddles[13].re,
            data[13],
            self.twiddles[13].im * data[14],
        );
        right[0] = fmla(
            self.twiddles[13].re,
            data[14],
            -self.twiddles[13].im * data[13],
        );
        self.bf14.exec(&mut InPlaceStore::new(&mut left));
        self.bf14.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[27] = right[0];
        data[1] = left[1] - right[13];
        data[2] = left[1] + right[13];
        data[3] = left[2] + right[12];
        data[4] = left[2] - right[12];
        data[5] = left[3] - right[11];
        data[6] = left[3] + right[11];
        data[7] = left[4] + right[10];
        data[8] = left[4] - right[10];
        data[9] = left[5] - right[9];
        data[10] = left[5] + right[9];
        data[11] = left[6] + right[8];
        data[12] = left[6] - right[8];
        data[13] = left[7] - right[7];
        data[14] = left[7] + right[7];
        data[15] = left[8] + right[6];
        data[16] = left[8] - right[6];
        data[17] = left[9] - right[5];
        data[18] = left[9] + right[5];
        data[19] = left[10] + right[4];
        data[20] = left[10] - right[4];
        data[21] = left[11] - right[3];
        data[22] = left[11] + right[3];
        data[23] = left[12] + right[2];
        data[24] = left[12] - right[2];
        data[25] = left[13] - right[1];
        data[26] = left[13] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly28, 28);

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly30<T: DctSample> {
    twiddles: [Complex<T>; 15],
    bf15: Dct2Butterfly15<T>,
}
impl<T: DctSample> Default for Dct4Butterfly30<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, 30 * 8).conj()),
            bf15: Dct2Butterfly15::default(),
        }
    }
}

impl<T: DctSample> Dct4Butterfly30<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is auto-generated factorization of DCT-IV butterfly for 30 points
        let mut left = [T::zero(); 15];
        let mut right = [T::zero(); 15];

        left[0] = fmla(self.twiddles[0].re, data[0], self.twiddles[0].im * data[29]);
        right[14] = fmla(
            -self.twiddles[0].re,
            data[29],
            self.twiddles[0].im * data[0],
        );
        left[1] = fmla(self.twiddles[1].re, data[1], self.twiddles[1].im * data[28]);
        right[13] = fmla(
            self.twiddles[1].re,
            data[28],
            -self.twiddles[1].im * data[1],
        );
        left[2] = fmla(self.twiddles[2].re, data[2], self.twiddles[2].im * data[27]);
        right[12] = fmla(
            -self.twiddles[2].re,
            data[27],
            self.twiddles[2].im * data[2],
        );
        left[3] = fmla(self.twiddles[3].re, data[3], self.twiddles[3].im * data[26]);
        right[11] = fmla(
            self.twiddles[3].re,
            data[26],
            -self.twiddles[3].im * data[3],
        );
        left[4] = fmla(self.twiddles[4].re, data[4], self.twiddles[4].im * data[25]);
        right[10] = fmla(
            -self.twiddles[4].re,
            data[25],
            self.twiddles[4].im * data[4],
        );
        left[5] = fmla(self.twiddles[5].re, data[5], self.twiddles[5].im * data[24]);
        right[9] = fmla(
            self.twiddles[5].re,
            data[24],
            -self.twiddles[5].im * data[5],
        );
        left[6] = fmla(self.twiddles[6].re, data[6], self.twiddles[6].im * data[23]);
        right[8] = fmla(
            -self.twiddles[6].re,
            data[23],
            self.twiddles[6].im * data[6],
        );
        left[7] = fmla(self.twiddles[7].re, data[7], self.twiddles[7].im * data[22]);
        right[7] = fmla(
            self.twiddles[7].re,
            data[22],
            -self.twiddles[7].im * data[7],
        );
        left[8] = fmla(self.twiddles[8].re, data[8], self.twiddles[8].im * data[21]);
        right[6] = fmla(
            -self.twiddles[8].re,
            data[21],
            self.twiddles[8].im * data[8],
        );
        left[9] = fmla(self.twiddles[9].re, data[9], self.twiddles[9].im * data[20]);
        right[5] = fmla(
            self.twiddles[9].re,
            data[20],
            -self.twiddles[9].im * data[9],
        );
        left[10] = fmla(
            self.twiddles[10].re,
            data[10],
            self.twiddles[10].im * data[19],
        );
        right[4] = fmla(
            -self.twiddles[10].re,
            data[19],
            self.twiddles[10].im * data[10],
        );
        left[11] = fmla(
            self.twiddles[11].re,
            data[11],
            self.twiddles[11].im * data[18],
        );
        right[3] = fmla(
            self.twiddles[11].re,
            data[18],
            -self.twiddles[11].im * data[11],
        );
        left[12] = fmla(
            self.twiddles[12].re,
            data[12],
            self.twiddles[12].im * data[17],
        );
        right[2] = fmla(
            -self.twiddles[12].re,
            data[17],
            self.twiddles[12].im * data[12],
        );
        left[13] = fmla(
            self.twiddles[13].re,
            data[13],
            self.twiddles[13].im * data[16],
        );
        right[1] = fmla(
            self.twiddles[13].re,
            data[16],
            -self.twiddles[13].im * data[13],
        );
        left[14] = fmla(
            self.twiddles[14].re,
            data[14],
            self.twiddles[14].im * data[15],
        );
        right[0] = fmla(
            -self.twiddles[14].re,
            data[15],
            self.twiddles[14].im * data[14],
        );
        self.bf15.exec(&mut InPlaceStore::new(&mut left));
        self.bf15.exec(&mut InPlaceStore::new(&mut right));
        data[0] = left[0];
        data[29] = right[0];
        data[1] = left[1] + right[14];
        data[2] = left[1] - right[14];
        data[3] = left[2] - right[13];
        data[4] = left[2] + right[13];
        data[5] = left[3] + right[12];
        data[6] = left[3] - right[12];
        data[7] = left[4] - right[11];
        data[8] = left[4] + right[11];
        data[9] = left[5] + right[10];
        data[10] = left[5] - right[10];
        data[11] = left[6] - right[9];
        data[12] = left[6] + right[9];
        data[13] = left[7] + right[8];
        data[14] = left[7] - right[8];
        data[15] = left[8] - right[7];
        data[16] = left[8] + right[7];
        data[17] = left[9] + right[6];
        data[18] = left[9] - right[6];
        data[19] = left[10] - right[5];
        data[20] = left[10] + right[5];
        data[21] = left[11] + right[4];
        data[22] = left[11] - right[4];
        data[23] = left[12] - right[3];
        data[24] = left[12] + right[3];
        data[25] = left[13] + right[2];
        data[26] = left[13] - right[2];
        data[27] = left[14] - right[1];
        data[28] = left[14] + right[1];
    }
}

define_in_place_butterfly!(Dct4Butterfly30, 30);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_dct4;
    use rand::Rng;

    gen_test_butterfly!(test_bf_dct4_4, f64, Dct4Butterfly6, 6, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct10_4, f64, Dct4Butterfly10, 10, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct12_4, f64, Dct4Butterfly12, 12, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct14_4, f64, Dct4Butterfly14, 14, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct18_4, f64, Dct4Butterfly18, 18, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct20_4, f64, Dct4Butterfly20, 20, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct22_4, f64, Dct4Butterfly22, 22, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct24_4, f64, Dct4Butterfly24, 24, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct26_4, f64, Dct4Butterfly26, 26, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct28_4, f64, Dct4Butterfly28, 28, 1e-7, naive_dct4);
    gen_test_butterfly!(test_bf_dct30_4, f64, Dct4Butterfly30, 30, 1e-7, naive_dct4);
}
